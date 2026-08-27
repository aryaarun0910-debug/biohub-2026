"""4-D rotary position encoding (RoPE) for the vendored ``SimpleNodeTransformer``.

Cell association is a RELATIVE-geometry problem: which detection at t+1 continues a detection
at t depends on the displacement (dt, dz, dy, dx), not on where in the field the pair sits.
The deployed edge model only sees ABSOLUTE positions (sinusoidal features over t, z, y, x fed
in with the UNet features) and its attention logits carry no relative term at all -- the
only relative quantity in the vendored model is the ``(c_t - c_t1) / 100`` input of the
pair MLP, downstream of attention. Rotating queries and keys by a position-dependent
rotation R(p) makes ``q_i^T R(p_i)^T R(p_j) k_j = q_i^T R(p_j - p_i) k_j`` -- the rotations
are a commutative group over the four coordinates, so every attention logit depends on the
displacement only and the whole attention stack becomes translation-equivariant.

Contract
--------
* No parameters. Frequencies are non-persistent buffers, so ``state_dict()`` of a model with
  RoPE installed is key-for-key identical to the public architecture: the public checkpoint
  loads strictly in both modes, and the exported ``edge_predictor_best.pth`` loads into the
  untouched vendored class at inference, which then re-installs the rotation from
  ``config.json`` (``h1r_rope4d_inference_patch.py``).
* The absolute sinusoidal features are KEPT. Dropping them would change ``proj`` input
  width (the public weights would no longer load) and they carry legitimate information
  (distance to the crop boundary, where tracks leave the field). RoPE ADDS relative structure
  inside attention; it does not replace the absolute inputs.
* Coordinates: the transformer receives FULL-RES voxel (z, y, x) -- training multiplies the
  downsampled Zh001r coordinates by ``DEPLOY_DOWNSAMPLE`` and the predict script multiplies by
  ``downsample`` -- so they are scaled by ``VOXEL_UM`` = (1.625, 0.40625, 0.40625) um/voxel
  (the FACT-0040 convention) before any angle is formed. Time is the window-relative frame
  offset (0 for the source frame, 1 for the target frame), exactly what both the trainer and
  the predict script feed the sinusoidal features. The cross-attention is strictly between
  adjacent frames, so dt is always +-1: the t-axis rotation is a constant phase per band.
* Frequency bands: K per axis, geometric from the longest wavelength (200 um, 20 frames by
  default) down to the shortest (10 um, 2 frames). Each band owns one (2i, 2i+1) dim pair of
  every head, so 8*K <= head_dim (the deployed head_dim is 32: K=4 rotates every dim).
* No thread-local or global state: coordinates travel as forward arguments and buffers are
  replicated by DataParallel like any other module state.
* Deliberately NO ``from __future__ import annotations``: the factory splices this file into
  the MIDDLE of the trainer cell (before ``from h1r_edge_data import ...``), where a
  future-import is a SyntaxError.
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt


POS_ENCODINGS = ("sinusoidal", "rope4d")
VOXEL_UM = (1.625, 0.40625, 0.40625)          # full-res (z, y, x) voxel pitch, um
DEFAULT_BANDS = 4
DEFAULT_SPACE_WAVELENGTHS_UM = (10.0, 200.0)   # shortest, longest
DEFAULT_TIME_WAVELENGTHS_FRAMES = (2.0, 20.0)  # shortest, longest
DEFAULT_FRAME_OFFSETS = (0.0, 1.0)             # source frame, target frame
ENCODING_KEY = "h1r_pos_encoding"
CONFIG_KEY = "h1r_rope4d"


def resolve_pos_encoding(value=None):
    """Validate the ``H1R_EDGE_POS_ENCODING`` knob (read from the env when ``value`` is None)."""
    if value is None:
        value = os.environ.get("H1R_EDGE_POS_ENCODING", "sinusoidal")
    value = str(value).strip().lower()
    if value not in POS_ENCODINGS:
        raise ValueError(f"H1R_EDGE_POS_ENCODING must be one of {POS_ENCODINGS}, got {value!r}")
    return value


def _wavelength_pair(text, default, name):
    if text is None or not str(text).strip():
        return (float(default[0]), float(default[1]))
    parts = [float(v) for v in str(text).split(",")]
    if len(parts) != 2 or parts[0] <= 0.0 or parts[1] < parts[0]:
        raise ValueError(f"{name} must be shortest,longest with 0 < shortest <= longest, got {text!r}")
    return (parts[0], parts[1])


def rope4d_config_from_env(env=None):
    """The rope4d constructor arguments, read from the ``H1R_ROPE_*`` knobs."""
    env = os.environ if env is None else env
    return {
        "bands": int(env.get("H1R_ROPE_BANDS", DEFAULT_BANDS)),
        "space_wavelengths_um": list(_wavelength_pair(
            env.get("H1R_ROPE_SPACE_WAVELENGTHS_UM"), DEFAULT_SPACE_WAVELENGTHS_UM,
            "H1R_ROPE_SPACE_WAVELENGTHS_UM")),
        "time_wavelengths_frames": list(_wavelength_pair(
            env.get("H1R_ROPE_TIME_WAVELENGTHS"), DEFAULT_TIME_WAVELENGTHS_FRAMES,
            "H1R_ROPE_TIME_WAVELENGTHS")),
        "voxel_um": list(VOXEL_UM),
        "frame_offsets": list(DEFAULT_FRAME_OFFSETS),
    }


def geometric_wavelengths(shortest, longest, bands):
    """``bands`` wavelengths from ``longest`` down to ``shortest`` in geometric steps."""
    if bands == 1:
        return [float(longest)]
    ratio = float(shortest) / float(longest)
    return [float(longest) * ratio ** (k / (bands - 1)) for k in range(bands)]


class Rope4D(nn.Module):
    """Per-axis rotation angles ``theta[axis, band] * coord[axis]`` over (t, z, y, x)."""

    def __init__(self, head_dim, bands=DEFAULT_BANDS,
                 space_wavelengths_um=DEFAULT_SPACE_WAVELENGTHS_UM,
                 time_wavelengths_frames=DEFAULT_TIME_WAVELENGTHS_FRAMES,
                 voxel_um=VOXEL_UM, frame_offsets=DEFAULT_FRAME_OFFSETS):
        super().__init__()
        self.head_dim = int(head_dim)
        self.bands = int(bands)
        if self.bands < 1:
            raise ValueError(f"rope4d needs at least one band, got {bands!r}")
        self.pairs = 4 * self.bands
        if 2 * self.pairs > self.head_dim:
            raise ValueError(f"rope4d needs 8 * bands <= head_dim: bands={self.bands}, head_dim={self.head_dim}")
        self.space_wavelengths_um = tuple(float(v) for v in space_wavelengths_um)
        self.time_wavelengths_frames = tuple(float(v) for v in time_wavelengths_frames)
        self.voxel_um = tuple(float(v) for v in voxel_um)
        self.frame_offsets = tuple(float(v) for v in frame_offsets)
        if len(self.space_wavelengths_um) != 2 or len(self.time_wavelengths_frames) != 2:
            raise ValueError("wavelength settings must be (shortest, longest) pairs")
        if len(self.voxel_um) != 3 or len(self.frame_offsets) != 2:
            raise ValueError("voxel_um must have 3 entries and frame_offsets 2")
        wl_t = geometric_wavelengths(*self.time_wavelengths_frames, self.bands)
        wl_s = geometric_wavelengths(*self.space_wavelengths_um, self.bands)
        wavelengths = torch.tensor([wl_t, wl_s, wl_s, wl_s], dtype=torch.float32)  # (4, K)
        # Buffers are NOT persistent: the state_dict must stay identical to the public model.
        self.register_buffer("theta", 2.0 * math.pi / wavelengths, persistent=False)
        self.register_buffer("scale", torch.tensor((1.0,) + self.voxel_um, dtype=torch.float32),
                             persistent=False)

    def extra_repr(self):
        return (f"head_dim={self.head_dim}, bands={self.bands}, rotated_dims={2 * self.pairs}, "
                f"space_um={self.space_wavelengths_um}, time_frames={self.time_wavelengths_frames}, "
                f"voxel_um={self.voxel_um}, frame_offsets={self.frame_offsets}")

    def coords4(self, coords, frame):
        """(..., 3) full-res voxel (z, y, x) -> (..., 4) [t, z, y, x] with a constant frame index."""
        t = torch.full((*coords.shape[:-1], 1), float(frame), dtype=torch.float32, device=coords.device)
        return torch.cat([t, coords.float()], dim=-1)

    def angles(self, coords4):
        """(..., 4) [t, z_vox, y_vox, x_vox] -> (..., 4 * bands) angles, fp32, physical units."""
        phys = coords4.float() * self.scale                      # frames, um, um, um
        ang = phys.unsqueeze(-1) * self.theta                    # (..., 4, K)
        return ang.reshape(*coords4.shape[:-1], self.pairs)

    def cos_sin(self, coords4):
        a = self.angles(coords4)
        return a.cos(), a.sin()

    @staticmethod
    def rotate(x, cos, sin):
        """Rotate dim pairs (2i, 2i+1), i < P, of ``x`` (B, H, N, Dh) by ``cos``/``sin`` (B, N, P).

        Computed in fp32 whatever the autocast dtype; the remaining ``Dh - 2P`` dims pass through.
        """
        pairs = cos.shape[-1]
        xf = x.float()
        head, tail = xf[..., : 2 * pairs], xf[..., 2 * pairs:]
        x1, x2 = head[..., 0::2], head[..., 1::2]
        c, s = cos.unsqueeze(1), sin.unsqueeze(1)                 # broadcast over heads
        rot = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).reshape(head.shape)
        return torch.cat([rot, tail], dim=-1).to(x.dtype)

    def export_config(self):
        """The keys the S5 trainer writes into ``config.json`` for the inference patch."""
        return {ENCODING_KEY: "rope4d", CONFIG_KEY: {
            "bands": self.bands,
            "space_wavelengths_um": list(self.space_wavelengths_um),
            "time_wavelengths_frames": list(self.time_wavelengths_frames),
            "voxel_um": list(self.voxel_um),
            "frame_offsets": list(self.frame_offsets),
        }}


def _check_mha(mha):
    if not isinstance(mha, nn.MultiheadAttention):
        raise TypeError(f"expected nn.MultiheadAttention, got {type(mha).__name__}")
    if not (mha.batch_first and mha._qkv_same_embed_dim) or mha.bias_k is not None or mha.add_zero_attn:
        raise ValueError("rope4d supports only the vendored MultiheadAttention configuration "
                         "(batch_first, packed qkv, no bias_k/bias_v, no add_zero_attn)")


def rope_cross_attention(mha, xq, xkv, key_padding_mask, cos_q, sin_q, cos_kv, sin_kv, training):
    """Functional twin of ``mha(xq, xkv, xkv, key_padding_mask=...)[0]`` with rotated q and k.

    Values are never rotated. With ``cos_q is None`` nothing is rotated and the result equals
    ``nn.MultiheadAttention`` to float tolerance -- the parity check pinned by the tests.
    ``key_padding_mask`` follows the nn.MultiheadAttention convention (True = ignore).
    """
    embed, heads = mha.embed_dim, mha.num_heads
    head_dim = embed // heads
    w, b = mha.in_proj_weight, mha.in_proj_bias
    q = F.linear(xq, w[:embed], None if b is None else b[:embed])
    k = F.linear(xkv, w[embed:2 * embed], None if b is None else b[embed:2 * embed])
    v = F.linear(xkv, w[2 * embed:], None if b is None else b[2 * embed:])
    batch, n_q, _ = q.shape
    n_kv = k.shape[1]
    q = q.view(batch, n_q, heads, head_dim).transpose(1, 2)
    k = k.view(batch, n_kv, heads, head_dim).transpose(1, 2)
    v = v.view(batch, n_kv, heads, head_dim).transpose(1, 2)
    if cos_q is not None:
        q = Rope4D.rotate(q, cos_q, sin_q)
        k = Rope4D.rotate(k, cos_kv, sin_kv)
    attn_mask = None
    if key_padding_mask is not None:
        attn_mask = (~key_padding_mask)[:, None, None, :]        # SDPA: True = may attend
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=float(mha.dropout) if training else 0.0)
    out = out.transpose(1, 2).reshape(batch, n_q, embed)
    return mha.out_proj(out)


def _rope_block(block, q, kv, kv_mask, cos_q, sin_q, cos_kv, sin_kv):
    """``CrossAttentionBlock.forward`` with the rotation applied inside its attention."""
    key_padding_mask = ~kv_mask if kv_mask is not None else None
    attn_out = rope_cross_attention(block.cross_attn, block.norm1(q), block.norm1(kv),
                                    key_padding_mask, cos_q, sin_q, cos_kv, sin_kv, block.training)
    q = q + attn_out
    return q + block.mlp(block.norm2(q))


class Rope4DNodeTransformer(nn.Module):
    """``SimpleNodeTransformer.forward`` with rotary q/k, sharing the vendored submodules.

    The submodules are held under the SAME attribute names as the vendored class, so the
    state_dict keys (``proj.*``, ``norm_in.*``, ``blocks.*``, ``norm_out.*``, ``pair_mlp.*``)
    are unchanged and every pretrained parameter is the very same tensor.
    """

    def __init__(self, inner, rope):
        super().__init__()
        self.pair_chunk_size = inner.pair_chunk_size
        self.proj = inner.proj
        self.norm_in = inner.norm_in
        self.blocks = inner.blocks
        self.norm_out = inner.norm_out
        self.pair_mlp = inner.pair_mlp
        self.rope = rope

    def forward(self, feat_t, feat_t1, coords_t, coords_t1, mask_t=None, mask_t1=None):
        unbatched = feat_t.ndim == 2
        if unbatched:
            feat_t = feat_t.unsqueeze(0)
            feat_t1 = feat_t1.unsqueeze(0)
            coords_t = coords_t.unsqueeze(0)
            coords_t1 = coords_t1.unsqueeze(0)

        q = self.norm_in(self.proj(feat_t))   # (B, N_t, hidden)
        k = self.norm_in(self.proj(feat_t1))  # (B, N_t1, hidden)
        rope = self.rope
        cos_t, sin_t = rope.cos_sin(rope.coords4(coords_t, rope.frame_offsets[0]))
        cos_t1, sin_t1 = rope.cos_sin(rope.coords4(coords_t1, rope.frame_offsets[1]))

        # Bi-directional cross-attention, as vendored: t attends to t+1 and vice versa.
        for block in self.blocks:
            def _fn(x, kv, mask, cos_x, sin_x, cos_kv, sin_kv, _b=block):
                return _rope_block(_b, x, kv, mask, cos_x, sin_x, cos_kv, sin_kv)

            if torch.is_grad_enabled():
                q = grad_ckpt(_fn, q, k, mask_t1, cos_t, sin_t, cos_t1, sin_t1, use_reentrant=False)
                k = grad_ckpt(_fn, k, q, mask_t, cos_t1, sin_t1, cos_t, sin_t, use_reentrant=False)
            else:
                q = _fn(q, k, mask_t1, cos_t, sin_t, cos_t1, sin_t1)
                k = _fn(k, q, mask_t, cos_t1, sin_t1, cos_t, sin_t)

        q = self.norm_out(q)
        k = self.norm_out(k)

        # Pairwise logits in chunks over N_t, verbatim from the vendored forward.
        n_t = q.shape[1]
        chunk = self.pair_chunk_size or n_t
        chunks = []
        pair_mlp = self.pair_mlp

        for i in range(0, n_t, chunk):
            q_c = q[:, i: i + chunk, :]
            coords_c = coords_t[:, i: i + chunk, :]

            def _chunk_fn(qc, kk, cc, cc1, _pm=pair_mlp):
                nc_i = qc.shape[1]
                n1 = kk.shape[1]
                qe = qc.unsqueeze(2).expand(-1, -1, n1, -1)
                ke = kk.unsqueeze(1).expand(-1, nc_i, -1, -1)
                rel = (cc.unsqueeze(2) - cc1.unsqueeze(1)) / 100.0
                return _pm(torch.cat([qe, ke, rel], dim=-1)).squeeze(-1)

            if torch.is_grad_enabled():
                out = grad_ckpt(_chunk_fn, q_c, k, coords_c, coords_t1, use_reentrant=False)
            else:
                out = _chunk_fn(q_c, k, coords_c, coords_t1)
            chunks.append(out)

        logits = torch.cat(chunks, dim=1)  # (B, N_t, N_t1)
        if unbatched:
            logits = logits.squeeze(0)
        return logits


def install_rope4d(model, config):
    """Swap ``model.transformer`` for its rotary twin. Returns the ``Rope4D`` module.

    ``model`` is a ``UNetNodeTransformer`` whose weights are ALREADY loaded; nothing else on
    it (UNet, detect_head) is touched. ``config`` holds the ``Rope4D`` constructor arguments
    minus ``head_dim`` (``rope4d_config_from_env()`` / ``config.json["h1r_rope4d"]``).
    """
    inner = model.transformer
    if isinstance(inner, Rope4DNodeTransformer):
        raise RuntimeError("rope4d is already installed on this model")
    if not hasattr(inner, "blocks") or len(inner.blocks) == 0:
        raise TypeError(f"{type(inner).__name__} has no cross-attention blocks to rotate")
    mha = inner.blocks[0].cross_attn
    _check_mha(mha)
    rope = Rope4D(head_dim=mha.embed_dim // mha.num_heads, **dict(config))
    model.transformer = Rope4DNodeTransformer(inner, rope)
    return rope


def install_rope4d_from_config(model, config):
    """Inference-side install from a checkpoint ``config.json`` written by the S5 trainer."""
    encoding = config.get(ENCODING_KEY)
    if encoding != "rope4d":
        raise ValueError(f"config does not declare a rope4d checkpoint ({ENCODING_KEY}={encoding!r})")
    if not isinstance(config.get(CONFIG_KEY), dict):
        raise ValueError(f"rope4d checkpoint config lacks the {CONFIG_KEY!r} block")
    return install_rope4d(model, config[CONFIG_KEY])


if os.environ.get("H1R_KERNEL") == "1":
    # Kaggle factory splices this module into the trainer cell. Register that cell's
    # namespace so h1r_edge_train's normal import resolves (same idiom as h1r_edge_data).
    sys.modules.setdefault("h1r_rope4d", sys.modules[__name__])
