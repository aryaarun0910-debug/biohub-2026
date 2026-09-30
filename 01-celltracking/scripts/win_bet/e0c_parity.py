"""E0c parity — validate the extracted wrapper against the saved 0.889 run.

Runs biotrack.wrapper.filter_output_graph on the four saved test-movie prediction
geffs with the EXACT submission config (min-track-len 7, effective 6 only on the
6bba_05b6850b public-test movie, image gap-refine ON over data/test) and compares
per-movie diagnostics to the saved run_stats.csv. Require exact/explained parity
before declaring the OOF baseline authoritative.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl  # noqa: E402

from biotrack import wrapper as W  # noqa: E402

PRED_DIR = ROOT / "artifacts/kaggle/lb897_calibration/tracking_repo/predictions/unknown/unet_transformer/split_0"
RUN_STATS = ROOT / "artifacts/kaggle/lb897_calibration/run_stats.csv"

# --- exact 0.889 submission output config ---
W.OUTPUT_MIN_TRACK_LEN = 7
W.SHORT_TRACK_MIN_LEN_BY_DATASET = {"6bba_05b6850b": 6}  # public-test-movie exception
W.GAP_REFINE_SYNTHETIC = True
W.TEST_DIR = ROOT / "data" / "test"

COMPARE = [
    "nodes", "edges", "gap_inserted_synthetic", "gap_added_nodes",
    "gap_refined_synthetic", "gap_refine_failed", "gap_refine_rejected_shift",
    "pruned_isolated_nodes", "motion_relink_edges", "safe_divisions_added",
    "short_track_components_removed", "short_track_nodes_removed",
    "short_track_edges_removed", "short_track_min_len_effective", "dropped_long_edges",
]


def build_nodes_edges(graph):
    nodes_by_id: dict[int, dict] = {}
    for row in graph.node_attrs().iter_rows(named=True):
        nid = int(row["node_id"])
        nodes_by_id[nid] = {
            "node_id": nid, "t": int(row["t"]),
            "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"]),
        }
    raw_edges: list[dict] = []
    for row in graph.edge_attrs().iter_rows(named=True):
        ep = row.get("edge_prob")
        raw_edges.append({
            "source_id": int(row["source_id"]), "target_id": int(row["target_id"]),
            "edge_prob": None if ep is None else float(ep),
        })
    return nodes_by_id, raw_edges


def main() -> None:
    ref = {r["dataset"]: r for r in pl.read_csv(RUN_STATS).iter_rows(named=True)}
    all_exact = True
    for pg in sorted(PRED_DIR.glob("*.geff")):
        ds = pg.stem
        g = W.graph_from_geff(pg)
        nbi, raw = build_nodes_edges(g)
        raw_nodes = len(nbi)
        nbi2, edges, stats = W.filter_output_graph(nbi, raw, dataset=ds)
        got = dict(stats)
        got["nodes"] = len(nbi2)
        got["edges"] = len(edges)
        r = ref.get(ds, {})
        print(f"\n=== {ds}  (raw_nodes mine={raw_nodes} saved={r.get('raw_nodes')}) ===")
        movie_ok = True
        for k in COMPARE:
            gv = got.get(k)
            rv = r.get(k)
            ok = (gv == rv)
            if not ok:
                movie_ok = False
                all_exact = False
            print(f"  {k:28s} mine={str(gv):>9} saved={str(rv):>9} {'ok' if ok else 'DIFF'}")
        print(f"  -> {ds}: {'EXACT' if movie_ok else 'MISMATCH'}")
    print(f"\n===== E0c PARITY: {'EXACT on all movies' if all_exact else 'MISMATCH — investigate above'} =====")


if __name__ == "__main__":
    main()
