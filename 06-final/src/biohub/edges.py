"""Edge scoring + the baseline one-to-one linker.

Two things recovered from the support pack that matter for faithfulness:

1. `_index_features` uses INTEGER indexing (round, clamp), not grid_sample.
   The probe deliberately uses grid_sample instead -- sub-voxel is better when
   you are asking what the representation contains -- but anything reproducing
   the deployed edge scores must index the way the model was trained.

2. Coordinates handed to predict_edges are in ORIGINAL resolution
   (downsampled-grid coords x (1,4,4)), while positional features are built
   from DOWNSAMPLED coords with WINDOW-RELATIVE time (0, 1) normalised by W.
   Getting either wrong produces plausible-looking garbage.

The baseline linker is a one-to-one Hungarian on `distance - beta*p`, two
passes (tight then relaxed). That is what actually produces the deployed
submission: the public pipeline's motion-relink stage discards the ILP's edges
entirely and rebuilds from a 1:1 assignment, which by construction cannot
express a division. Every division in every public submission comes from the
bolted-on safe-div rule afterwards.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

POS_EMBED_DIM = 8          # per axis; total = 4 axes x 8 = 32
DOWNSAMPLE = np.array([1, 4, 4], dtype=np.float32)
GRID_UM = 1.625            # downsampled grid is isotropic
TIGHT_UM, RELAXED_UM = 6.0, 10.0
VELOCITY_WEIGHT, LEARNED_BONUS = 0.5, 1.0


def pos_features(coords_tzyx: np.ndarray, shape: tuple) -> np.ndarray:
    """Sinusoidal embedding of [t, z, y, x], each normalised by its axis size."""
    freqs = 2 ** np.arange(POS_EMBED_DIM // 2)
    out = []
    for c, s in zip(coords_tzyx.T, shape):
        ang = (c / max(s, 1))[:, None] * freqs * np.pi
        out.append(np.concatenate([np.sin(ang), np.cos(ang)], axis=1))
    return np.concatenate(out, axis=1).astype(np.float32)


def index_features(fmap: torch.Tensor, coords: np.ndarray) -> torch.Tensor:
    """Integer indexing, matching the training-time contract exactly."""
    C, Z, Y, X = fmap.shape[-4:]
    c = torch.as_tensor(coords, device=fmap.device)
    z = c[:, 0].long().clamp(0, Z - 1)
    y = c[:, 1].long().clamp(0, Y - 1)
    x = c[:, 2].long().clamp(0, X - 1)
    return fmap.reshape(C, Z, Y, X)[:, z, y, x].T          # (N, C)


@torch.no_grad()
def edge_logits(model, fsrc, ftgt, csrc, ctgt, device, spatial=(64, 64, 64)) -> np.ndarray:
    """(n_src, n_tgt) logits for one consecutive frame pair.

    csrc/ctgt are (N,3) coords in the DOWNSAMPLED grid.
    """
    W = 2
    shape = (W,) + spatial
    ps = pos_features(np.c_[np.zeros(len(csrc)), csrc], shape)
    pt = pos_features(np.c_[np.ones(len(ctgt)), ctgt], shape)
    dev = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
    ds = torch.as_tensor(DOWNSAMPLE, device=device)
    out = model.transformer(
        torch.cat([fsrc.unsqueeze(0).float(), dev(ps)], -1),
        torch.cat([ftgt.unsqueeze(0).float(), dev(pt)], -1),
        dev(csrc) * ds, dev(ctgt) * ds,
        torch.ones(1, len(csrc), dtype=torch.bool, device=device),
        torch.ones(1, len(ctgt), dtype=torch.bool, device=device),
    )
    return out[0].float().cpu().numpy()


def link_frame_pair(src_um, tgt_um, prob, pred_um=None,
                    tight=TIGHT_UM, relaxed=RELAXED_UM, beta=LEARNED_BONUS):
    """One-to-one Hungarian, tight pass then relaxed on the leftovers.

    cost = ||tgt - predicted_src|| + 0.05*||tgt - src|| - beta*p, gated on the
    RAW distance. Returns a list of (i, j, p, distance_um, pass_index).
    """
    if not len(src_um) or not len(tgt_um):
        return []
    raw = np.linalg.norm(tgt_um[None] - src_um[:, None], axis=-1)
    anchor = src_um if pred_um is None else pred_um
    base = np.linalg.norm(tgt_um[None] - anchor[:, None], axis=-1) + 0.05 * raw - beta * prob

    out, used_i, used_j = [], set(), set()
    for p_idx, gate in enumerate((tight, relaxed)):
        ii = [i for i in range(len(src_um)) if i not in used_i]
        jj = [j for j in range(len(tgt_um)) if j not in used_j]
        if not ii or not jj:
            break
        sub, sub_raw = base[np.ix_(ii, jj)].copy(), raw[np.ix_(ii, jj)]
        forbidden = sub_raw > gate
        sub[forbidden] = 1000 * gate + 1
        r, c = linear_sum_assignment(sub)
        for a, b in zip(r, c):
            if forbidden[a, b]:
                continue
            i, j = ii[a], jj[b]
            out.append((i, j, float(prob[i, j]), float(raw[i, j]), p_idx))
            used_i.add(i); used_j.add(j)
    return out
