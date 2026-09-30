"""The four components between my 0.845 baseline and the deployed 0.926.

  detection TTA   average logits over flips; Z is EXCLUDED because the data is
                  ~4x coarser in Z and a Z-flip is out of distribution
  dual seed       standardise the secondary's logits to the primary's scale,
                  then blend (w=0.80 on the secondary), with a frame-level
                  retention guard that falls back to primary if the blend loses
                  more than 10% of candidates
  bidirectional   weighted HARMONIC mean of forward and reverse link
                  probabilities, so a link survives only if BOTH time
                  directions support it
  edge-feature TTA  reuse the augmented encoder passes to average the feature
                  map the linker reads, not just the detection logits
"""
from __future__ import annotations

import numpy as np
import torch

FLIPS = [(), (-1,), (-2,), (-2, -1)]      # identity + x, y, xy. never Z.
SEC_DET_W, SEC_EDGE_W, LOW_MARGIN_MAX = 0.80, 0.15, 0.35
BIDIR_W, MIN_RETENTION = 0.15, 0.90


def _align(x, ref, dim=None):
    """Standardise x onto ref's scale, with the ratio clipped to [0.5, 2]."""
    if dim is None:
        mx, sx, mr, sr = x.mean(), x.std().clamp_min(1e-4), ref.mean(), ref.std().clamp_min(1e-4)
    else:
        mx, sx = x.mean(dim, keepdim=True), x.std(dim, keepdim=True).clamp_min(1e-4)
        mr, sr = ref.mean(dim, keepdim=True), ref.std(dim, keepdim=True).clamp_min(1e-4)
    return (x - mx) * (sr / sx).clamp(0.5, 2.0) + mr


@torch.no_grad()
def encode_tta(model, imgs, flips=FLIPS):
    """Return (unet_out_mean, det_logits_mean) averaged over the flip group."""
    acc_u = acc_d = None
    for dims in flips:
        v = imgs.flip(dims) if dims else imgs
        u, d = model.encode(v)
        if dims:
            u = u.flip(dims)
            d = [x.flip(dims) for x in d]
        ds = torch.stack(d, 1)
        acc_u = u if acc_u is None else acc_u + u
        acc_d = ds if acc_d is None else acc_d + ds
    n = len(flips)
    return acc_u / n, acc_d / n


def blend_detection(det_p, det_s, w=SEC_DET_W):
    return (1 - w) * det_p + w * _align(det_s, det_p)


def retention_ok(n_blend, n_primary, min_ret=MIN_RETENTION):
    """False means fall back to the primary map for this frame."""
    if n_primary <= 0:
        return True
    return (n_blend / n_primary) >= min_ret


def harmonic_bidirectional(fwd, rev_t, w=BIDIR_W):
    """fwd, rev_t: (n_src, n_tgt) logits; rev_t already transposed to fwd's shape.

    Softmax over SOURCES for each target, weighted harmonic mean, renormalise,
    back to log space, restandardised so downstream thresholds keep meaning.
    The harmonic mean is dominated by the smaller probability, so a link needs
    mutual support to survive.
    """
    f = torch.as_tensor(fwd, dtype=torch.float32)
    r = _align(torch.as_tensor(rev_t, dtype=torch.float32), f, dim=0)
    pf = torch.softmax(f, 0).clamp_min(1e-8)
    pr = torch.softmax(r, 0).clamp_min(1e-8)
    ph = 1.0 / ((1 - w) / pf + w / pr)
    ph = ph / ph.sum(0, keepdim=True).clamp_min(1e-12)
    return _align(torch.log(ph), f, dim=0).numpy()


def low_margin_consensus(lp, ls, w_s=SEC_EDGE_W, M=LOW_MARGIN_MAX):
    """Blend the secondary in only where the models AGREE and the primary is unsure."""
    Lp = torch.as_tensor(lp, dtype=torch.float32)
    Ls = _align(torch.as_tensor(ls, dtype=torch.float32), Lp, dim=0)
    if Lp.shape[0] < 2:
        return lp
    Pp, Ps = torch.softmax(Lp, 0), torch.softmax(Ls, 0)
    top2 = Pp.topk(2, dim=0).values
    m = top2[0] - top2[1]
    u = ((M - m) / M).clamp(0, 1)
    agree = Pp.argmax(0) == Ps.argmax(0)
    w = torch.where(agree, w_s * u, torch.zeros_like(u))
    return ((1 - w) * Lp + w * Ls).numpy()
