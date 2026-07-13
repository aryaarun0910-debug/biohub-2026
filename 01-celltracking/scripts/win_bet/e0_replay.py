"""E0b — replay the pure-0.889 wrapper over raw OOF predictions and exact-score it.

Applies biotrack.wrapper.filter_output_graph (pure LB897 config, NO trackastra
fusion, gap-refine disabled) to each raw learned OOF geff, rebuilds a scorable
graph, and scores it embryo-held-out with the authoritative metric. This defines
THE production baseline every later repair delta is measured against.

Usage:
  .venv/Scripts/python.exe scripts/win_bet/e0_replay.py \
     --pred-dir artifacts/kaggle/oof_clean/pred_geffs_split_0 [--max-crops N]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl  # noqa: E402

from biotrack import wrapper as W  # noqa: E402
from biotrack.metric import score_pred_graph  # noqa: E402
from biotrack.submission import submission_to_graphs  # noqa: E402
from tracking_cellmot.metrics import summarise  # noqa: E402


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


def rows_for(dataset: str, nodes_by_id: dict, edges: list[dict]) -> list[dict]:
    ids = sorted(nodes_by_id)
    sub = {nid: i + 1 for i, nid in enumerate(ids)}  # 1-based submission-local ids
    rows = []
    for nid in ids:
        n = nodes_by_id[nid]
        rows.append({
            "dataset": dataset, "row_type": "node", "node_id": sub[nid],
            "t": int(n["t"]), "z": float(n["z"]), "y": float(n["y"]), "x": float(n["x"]),
            "source_id": -1, "target_id": -1,
        })
    for e in edges:
        rows.append({
            "dataset": dataset, "row_type": "edge", "node_id": -1,
            "t": -1, "z": -1.0, "y": -1.0, "x": -1.0,
            "source_id": sub[int(e["source_id"])], "target_id": sub[int(e["target_id"])],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-dir", default=str(ROOT / "data" / "train"))
    ap.add_argument("--image-dir", default=str(ROOT / "data" / "train"),
                    help="zarr image root for gap-refine (default data/train for OOF)")
    ap.add_argument("--min-track-len", type=int, default=7,
                    help="E0c: submission uses 7 uniformly on OOF (no public-test exception)")
    ap.add_argument("--no-gap-refine", action="store_true", help="disable image gap-refine")
    ap.add_argument("--max-crops", type=int)
    a = ap.parse_args()

    # E0c deployment-exact config (parity-verified vs saved 0.889 run_stats.csv):
    # min-track-len 7 UNIFORM (the 6bba_05b6850b=6 exception is a public-test movie,
    # not an OOF crop, so it is NOT applied here), image gap-refine ON.
    W.OUTPUT_MIN_TRACK_LEN = a.min_track_len
    W.SHORT_TRACK_MIN_LEN_BY_DATASET = {}
    W.GAP_REFINE_SYNTHETIC = not a.no_gap_refine
    W.TEST_DIR = Path(a.image_dir)
    print(f"[E0c config] min_track_len={W.OUTPUT_MIN_TRACK_LEN} uniform, "
          f"gap_refine={W.GAP_REFINE_SYNTHETIC}, image_dir={W.TEST_DIR}")

    preds = sorted(Path(a.pred_dir).glob("*.geff"))
    if a.max_crops:
        preds = preds[: a.max_crops]
    if not preds:
        sys.exit(f"no geffs in {a.pred_dir}")

    rows = []
    for pg in preds:
        stem = pg.stem
        gt = Path(a.gt_dir) / f"{stem}.geff"
        if not gt.exists():
            print(f"  skip {stem}: no GT"); continue
        g = W.graph_from_geff(pg)
        nbi, raw = build_nodes_edges(g)
        nbi, edges, stats = W.filter_output_graph(nbi, raw, dataset=stem)
        df = pl.DataFrame(rows_for(stem, nbi, edges)).with_row_index("id")
        pred_graph = submission_to_graphs(df)[stem]
        row = score_pred_graph(pred_graph, str(gt))
        rows.append(row)
        print(f"  {stem:16} nodes={row['num_pred_nodes']:>5} "
              f"adjJ={row['adj_edge_jaccard']:.4f} recall={row['node_recall']:.3f} "
              f"divTP/FP/FN={row['division_tp']}/{row['division_fp']}/{row['division_fn']}")

    s = summarise(rows)
    fam = preds[0].stem.split("_")[0]
    print(f"\n=== E0c deployment-exact wrapper OOF, held-out {fam} ({len(rows)} crops) ===")
    print(f"  adj_edge_jaccard = {s['adj_edge_jaccard']:.4f}")
    print(f"  division_jaccard = {s['division_jaccard']:.4f} "
          f"(TP={s['division_tp']} FP={s['division_fp']} FN={s['division_fn']})")
    print(f"  score            = {s['score']:.4f}")


if __name__ == "__main__":
    main()
