"""G3: oracle headroom decomposition. Designed to KILL the division thesis.

Everything runs off the cached detections + GT graphs, on the frozen LOEO folds.
Each oracle reports dJ_edge and 0.1*dJ_div separately, plus the multiplier.

Read O1 to three decimals at most: fold 44b6 has D=26, so one division event is
0.0038 of total score. O2 rests on ~129k labelled edges and is far better
resolved. If O1 and O2 are close, prefer O2 on measurement confidence alone.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from biohub import io

CACHE = Path("artifacts/cache")
RADIUS_UM, A, DIV_W = 7.0, 0.1, 0.1
GRID_UM = 1.625  # downsampled grid is isotropic


def match(gt_grid, gt_t, det_zyx, det_t):
    """Per-frame Hungarian on um distance, <= 7 um. Returns pred->gt and gt->pred."""
    p2g, g2p = {}, {}
    for t in np.unique(gt_t):
        gi = np.where(gt_t == t)[0]
        pi = np.where(det_t == t)[0]
        if not len(gi) or not len(pi):
            continue
        d = np.linalg.norm((gt_grid[gi][:, None] - det_zyx[pi][None]) * GRID_UM, axis=-1)
        r, c = linear_sum_assignment(d)
        for i, j in zip(r, c):
            if d[i, j] <= RADIUS_UM:
                g2p[int(gi[i])] = int(pi[j]); p2g[int(pi[j])] = int(gi[i])
    return p2g, g2p


def edge_confusion(pred_edges, gt_edge_set, gt_out, gt_in, p2g):
    tp = fp = 0
    matched_gt = set()
    for s, t in pred_edges:
        ms, mt = p2g.get(s), p2g.get(t)
        if ms is not None and mt is not None and (ms, mt) in gt_edge_set:
            tp += 1; matched_gt.add((ms, mt))
        elif (mt is not None and mt in gt_in) or (ms is not None and ms in gt_out):
            fp += 1        # contradicts known ground truth
        # else: between unannotated regions -> ignored, not penalised
    return tp, fp, len(gt_edge_set) - len(matched_gt)


def score(J, n_pred, n_est):
    mult = 1.0 - A * (n_pred - n_est) / n_est
    return max(0.0, J * mult), mult


def divisions(edges):
    """node -> children, for nodes with >= 2 children."""
    out = {}
    for s, t in edges:
        out.setdefault(s, []).append(t)
    return {k: v for k, v in out.items() if len(v) >= 2}, out
