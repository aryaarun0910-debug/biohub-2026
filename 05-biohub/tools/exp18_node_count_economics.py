#!/usr/bin/env python3
"""EXP-18 -- what does this competition actually PAY FOR?

    total_node_ratio = (N_pred - n_total) / n_total
    J_adj            = max(0, J * (1 - 0.1 * total_node_ratio))      <- no ceiling
    score            = J_adj + 0.1 * division_jaccard

n_total is the HOST'S estimate of the true cell count, ~35x our annotated count. Predicting
fewer nodes than that is rewarded without limit, so our oracle run scores adj_edge 1.0870 on an
edge Jaccard of 0.9957: a +9.2% bonus, worth more than the whole division term.

The strategic question this raises is the one that decides the whole competition for us:
A REAL DETECTOR FINDS UNANNOTATED CELLS TOO. Every extra detection pushes N_pred toward n_total
and erodes the bonus. Does better detection help or hurt?

This adds synthetic "extra" detections to an oracle run at increasing rates, links them normally,
and scores. Extra nodes are placed near real ones (as a real detector's extra finds would be),
each carried forward across frames so they form plausible tracks rather than isolated points.
"""
import sys
from pathlib import Path
import numpy as np, tracksdata as td

sys.path.insert(0, "src"); sys.path.insert(0, "reference/royerlab-baseline/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from biohub.contracts import Config, Graph, SCALE
from biohub.detect import detect_oracle
from biohub.edges import score_edges
from biohub.refine import refine
from biohub.resolve import resolve
from biohub.repair import repair
from tracking_cellmot.metrics import summarise
from geff import GeffMetadata
from _eval_common import load_gt, score_one

GT = Path("data/train_geff")
FILES = sorted(GT.glob("*.geff"))[:60]
cfg = Config()
rng = np.random.default_rng(0)

CACHE = {}
for p in FILES:
    gt = load_gt(p); n = gt.node_attrs()
    CACHE[p] = (np.array([r["t"] for r in n.iter_rows(named=True)]),
                np.array([[r["z"], r["y"], r["x"]] for r in n.iter_rows(named=True)], float),
                (GeffMetadata.read(p).extra or {}).get("estimated_number_of_nodes"))
    del gt, n


def inflate(t, zyx, mult, rng):
    """Add (mult-1)x extra detections: real-looking cells the annotation never labelled.

    Placed 8-25um from a real cell (far enough not to be a duplicate of it) and given the same
    per-frame trajectory, so they behave like genuine unannotated cells rather than noise.
    """
    if mult <= 1.0:
        return t, zyx
    n_extra = int(len(t) * (mult - 1.0))
    if n_extra <= 0:
        return t, zyx
    pick = rng.integers(0, len(t), n_extra)
    off_um = rng.normal(0, 1, (n_extra, 3))
    off_um /= np.linalg.norm(off_um, axis=1, keepdims=True) + 1e-9
    off_um *= rng.uniform(8.0, 25.0, (n_extra, 1))
    new = zyx[pick] + off_um / SCALE
    return np.concatenate([t, t[pick]]), np.concatenate([zyx, new])


print(f"  {len(FILES)} datasets; extra detections are UNANNOTATED-CELL-LIKE, not noise\n")
print(f"  {'N_pred/oracle':>14} {'node_ratio':>11} {'edgeJ':>8} {'adj_edge':>9} "
      f"{'divJ':>7} {'SCORE':>8}")
for mult in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0):
    rows = []
    for p in FILES:
        t, zyx, est = CACHE[p]
        t2, z2 = inflate(t, zyx, mult, np.random.default_rng(abs(hash(p.stem)) % 2**31))
        g = detect_oracle(t2, z2, p.stem)
        for st in (refine, score_edges, resolve, repair):
            g = st(g, cfg)
        rows.append(score_one(g, load_gt(p), est=est))
    s = summarise(rows)
    print(f"  {mult:>14.1f} {np.mean([r['total_node_ratio'] for r in rows]):>11.3f} {s['edge_jaccard']:>8.4f} "
          f"{s['adj_edge_jaccard']:>9.4f} {s['division_jaccard']:>7.4f} {s['score']:>8.4f}",
          flush=True)
