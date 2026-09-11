#!/usr/bin/env python3
"""Score a submission.csv against local .geff ground truth with the PATCHED scorer.

    score_submission.py <submission.csv> [--gt data/train_geff] [--label NAME]

Reports adj_edge_jaccard and division_jaccard SEPARATELY, which is the whole point:
the leaderboard only ever gives you their sum.
"""
import argparse, sys, json
from pathlib import Path
import polars as pl, tracksdata as td
from geff import GeffMetadata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference/royerlab-baseline/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference/royerlab-baseline"))
from tracking_cellmot.metrics import evaluate, per_sample_metrics, summarise, node_recall
from scripts.csv_to_geffs import build_graph_from_rows

SCALE = (1.625, 0.40625, 0.40625)

def load_graph(geff_path: Path):
    r = td.graph.IndexedRXGraph.from_geff(geff_path)
    return r[0] if isinstance(r, tuple) else r

def read_n_total(geff_path: Path) -> float:
    try:
        v = (GeffMetadata.read(geff_path).extra or {}).get("estimated_number_of_nodes")
        return float(v) if v is not None else float("nan")
    except Exception:
        return float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("--gt", default="data/train_geff")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    df = pl.read_csv(a.csv)
    gt_root = Path(a.gt)
    rows = []
    for ds in sorted(df["dataset"].unique().to_list()):
        d = df.filter(pl.col("dataset") == ds)
        geff = gt_root / f"{ds}.geff"
        if not geff.exists():
            print(f"  {ds}: NO GROUND TRUTH, skipped"); continue
        pred = build_graph_from_rows(d.filter(pl.col("row_type") == "node"),
                                     d.filter(pl.col("row_type") == "edge"))
        gt = load_graph(geff)
        er = evaluate(pred, gt, scale=SCALE, max_distance=7.0)
        n_total = read_n_total(geff)
        nr = node_recall(pred, gt)
        r = per_sample_metrics(er, n_total, nr); rows.append(r)
        print(f"  {ds:<18} edgeJ {r['edge_jaccard']:.4f}  adj {r['adj_edge_jaccard']:.4f}  "
              f"divJ tp/fp/fn {er.division_tp}/{er.division_fp}/{er.division_fn}  "
              f"N_pred {er.num_pred_nodes} vs n_total {n_total:.0f}")
    s = summarise(rows)
    tag = f" [{a.label}]" if a.label else ""
    print(f"\n== TOTAL{tag} ==")
    print(f"  adj_edge_jaccard : {s['adj_edge_jaccard']:.4f}")
    print(f"  division_jaccard : {s['division_jaccard']:.4f}  (tp/fp/fn "
          f"{s['division_tp']}/{s['division_fp']}/{s['division_fn']})")
    print(f"  score            : {s['score']:.4f}")

if __name__ == "__main__":
    main()
