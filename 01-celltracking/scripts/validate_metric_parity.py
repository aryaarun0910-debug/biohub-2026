"""Metric hardening: numpy edge metric vs the AUTHORITATIVE tracksdata matcher on REAL crops.

Red-team #10 / Codex hardening: `biotrack.metric_numpy` (our fast local gate) was only validated on
7 hand-built adversarial cases, never against the organizer's `DistanceMatching` on a real dense crop
where crowded frames (hundreds of nuclei with overlapping 7um gates) can make the dense LSA and the
host's sparse min-weight-full-bipartite matcher disagree under ties.

This runs the SAME predicted graph (op_bright pipeline output) through BOTH scorers per crop and asserts
edge TP/FP/FN and adjusted edge Jaccard agree. Picks a few crowded 6bba crops by default.

Usage: .venv/Scripts/python.exe scripts/validate_metric_parity.py [crop ...]
"""
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from biotrack.metric import estimated_nodes, score_pred_graph  # noqa: E402
from biotrack.metric_numpy import Sample, score_sample  # noqa: E402
from biotrack.submission import submission_to_graphs  # noqa: E402
from run_phase1_ablation import cfg_op_bright  # noqa: E402
from run_v3_taxonomy import geff_to_sample  # noqa: E402

TRAIN = ROOT / "data" / "train"
# crowded-but-modest crops: real per-frame collisions, fast enough for a check
DEFAULT_CROPS = ["6bba_05b6850b", "6bba_062c8d37", "6bba_07477033", "44b6_0113de3b"]
ATOL_ADJ = 5e-4  # adj_edge_jaccard tolerance


def sample_to_graph(s: Sample, dataset: str):
    """Build a tracksdata graph from a numpy Sample via the submission round-trip."""
    id_to_sub = {int(n): i + 1 for i, n in enumerate(s.node_ids)}
    rows = [
        {"id": 0, "dataset": dataset, "row_type": "node", "node_id": id_to_sub[int(n)],
         "t": int(tt), "z": float(z), "y": float(y), "x": float(x), "source_id": -1, "target_id": -1}
        for n, tt, (z, y, x) in zip(s.node_ids, s.t, s.zyx)
    ]
    rows += [
        {"id": 0, "dataset": dataset, "row_type": "edge", "node_id": -1,
         "t": -1, "z": -1.0, "y": -1.0, "x": -1.0,
         "source_id": id_to_sub[int(a)], "target_id": id_to_sub[int(b)]}
        for a, b in s.edges
    ]
    return submission_to_graphs(pl.DataFrame(rows))[dataset]


def main():
    crops = sys.argv[1:] or DEFAULT_CROPS
    print(f"{'crop':<16} {'np TP/FP/FN':>18} {'td TP/FP/FN':>18} {'np adjJ':>9} {'td adjJ':>9} {'result':>7}")
    all_ok = True
    for crop in crops:
        pred = cfg_op_bright(crop)                       # numpy Sample (over-propose->dedup->link)
        gt = geff_to_sample(str(TRAIN / f"{crop}.geff"))
        n_est = estimated_nodes(str(TRAIN / f"{crop}.geff"))
        npm = score_sample(pred, gt, n_est)              # our fast numpy gate
        g = sample_to_graph(pred, crop)
        tdm = score_pred_graph(g, str(TRAIN / f"{crop}.geff"))  # authoritative tracksdata

        np_c = (npm["edge_tp"], npm["edge_fp"], npm["edge_fn"])
        td_c = (tdm["edge_tp"], tdm["edge_fp"], tdm["edge_fn"])
        counts_ok = np_c == td_c
        adj_ok = abs(npm["adj_edge_jaccard"] - tdm["adj_edge_jaccard"]) <= ATOL_ADJ
        ok = counts_ok and adj_ok
        all_ok &= ok
        print(f"{crop:<16} {str(np_c):>18} {str(td_c):>18} "
              f"{npm['adj_edge_jaccard']:>9.4f} {tdm['adj_edge_jaccard']:>9.4f} {'OK' if ok else 'DIFF':>7}")
    print("\nPARITY OK — numpy gate matches the authoritative matcher on real crops"
          if all_ok else "\nPARITY MISMATCH — investigate (dense-frame tie handling?)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
