#!/usr/bin/env python3
"""Score a prediction on THE ACTUAL FOUR TEST MOVIES, using the released annotations.

Every proxy this campaign has used was the wrong substrate:
  - their in-kernel ppsweep: 8 held-out TRAIN stems carrying 12 divisions, over-reads LB +0.0041
  - our work/train_graphs:   199 TRAIN graphs from a different, fork-happy pipeline
  - our validator runs:      8 TRAIN stems

But the test set is only four datasets -- 44b6_0113de3b, 44b6_0b24845f, 6bba_05b6850b,
6bba_05db0fb1 -- and the organisers released sparse annotations for all four in train_geff,
along with estimated_number_of_nodes. The leaderboard scores held-out annotations on these SAME
movies. Same embryos, same timepoints, same density, same difficulty. It is by far the closest
proxy available, and nothing needed to be run on Kaggle to get it.

Usage:  python tools/testproxy.py [dir-of-prediction-geffs]
"""
import sys, warnings
from pathlib import Path
import numpy as np, polars as pl, tracksdata as td

sys.path.insert(0, "src"); sys.path.insert(0, "reference/royerlab-baseline/src")
warnings.filterwarnings("ignore")
from geff import GeffMetadata
from tracking_cellmot.metrics import evaluate, per_sample_metrics, node_recall, summarise
from biohub.contracts import SCALE

GT = Path("data/train_geff")
STEMS = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]
DEFAULT = Path("work/repro_out/tracking_repo/predictions/unknown/unet_transformer/split_0")

def load(p):
    g = td.graph.IndexedRXGraph.from_geff(p); return g[0] if isinstance(g, tuple) else g

def main(d):
    rows = []
    print(f"  {'stem':<16}{'N_pred':>9}{'n_total':>9}{'mult':>8}{'edgeJ':>8}{'adj':>8}"
          f"{'divTP':>7}{'divFP':>7}{'divFN':>7}{'weight':>8}")
    for s in STEMS:
        pp = Path(d) / f"{s}.geff"
        if not pp.exists(): print(f"  {s:<16} MISSING"); continue
        pred, gt = load(pp), load(GT / f"{s}.geff")
        est = (GeffMetadata.read(GT / f"{s}.geff").extra or {})["estimated_number_of_nodes"]
        er = evaluate(pred, gt, scale=tuple(SCALE), max_distance=7.0)
        nr = node_recall(pred, gt) if len(pred.edge_attrs()) else 0.0
        m = per_sample_metrics(er, float(est), nr)
        rows.append(m)
        w = m["edge_tp"] + m["edge_fp"] + m["edge_fn"]
        npred = er.num_pred_nodes
        print(f"  {s:<16}{npred:>9,}{est:>9,}{1-0.1*((npred-est)/est):>8.4f}"
              f"{m['edge_jaccard']:>8.4f}{m['adj_edge_jaccard']:>8.4f}"
              f"{m['division_tp']:>7}{m['division_fp']:>7}{m['division_fn']:>7}{w:>8}")
    if not rows: return
    S = summarise(rows)
    print(f"\n  adj_edge_jaccard {S['adj_edge_jaccard']:.5f}   division_jaccard "
          f"{S['division_jaccard']:.5f}   ->  SCORE {S['score']:.5f}")
    w = np.array([r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in rows], float)
    print(f"  per-dataset weights {w.astype(int).tolist()}  -> shares "
          f"{[f'{x:.1%}' for x in (w/w.sum())]}")

main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
