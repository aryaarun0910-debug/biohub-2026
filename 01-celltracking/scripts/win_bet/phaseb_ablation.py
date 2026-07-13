"""Track-A exact ablations on the cached E0c graphs vs the frozen baseline (0.7595/0.6484).

Transforms (authoritative edge+division scoring, ProcessPool):
  baseline : E0c as-is (sanity: should reproduce 0.7595/0.6484)
  nofork   : keep only the nearest successor per source (drop 2nd-child edges) -> does the
             wrapper's forking help or hurt the exact composite?

Usage: phaseb_ablation.py --mode nofork [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

CACHE = ROOT / "artifacts/kaggle/e0c_cache"
SCALE = (1.625, 0.40625, 0.40625)


def cached_crops(split: int):
    return sorted(json.loads(p.read_text())["crop"]
                  for p in (CACHE / "status").glob(f"{split}__*.json")
                  if json.loads(p.read_text()).get("status") == "ok")


def score_transform(args) -> dict:
    split, crop, mode = args
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from biotrack.metric import score_pred_graph
    from biotrack.submission import submission_to_graphs
    df = pl.read_parquet(CACHE / "graphs" / str(split) / f"{crop}.parquet")
    if mode == "nofork":
        nodes = df.filter(pl.col("row_type") == "node")
        pos = {int(r["node_id"]): (r["z"], r["y"], r["x"]) for r in nodes.iter_rows(named=True)}
        edges = df.filter(pl.col("row_type") == "edge")
        keep_idx = []
        best: dict[int, tuple[float, int]] = {}
        erows = list(edges.iter_rows(named=True))
        for i, e in enumerate(erows):
            s, t = int(e["source_id"]), int(e["target_id"])
            ps, pt = pos[s], pos[t]
            d = ((ps[0]-pt[0])*SCALE[0])**2 + ((ps[1]-pt[1])*SCALE[1])**2 + ((ps[2]-pt[2])*SCALE[2])**2
            if s not in best or d < best[s][0]:
                best[s] = (d, i)
        keep = {v[1] for v in best.values()}
        kept_edges = edges[[i for i in range(edges.height) if i in keep]] if edges.height else edges
        df = pl.concat([nodes, kept_edges]) if kept_edges.height else nodes
    gdf = df.with_columns(pl.lit(crop).alias("dataset")).with_row_index("id")
    graph = submission_to_graphs(gdf)[crop]
    row = score_pred_graph(graph, str(ROOT / "data" / "train" / f"{crop}.geff"))
    return {"crop": crop, "split": split, **{k: row[k] for k in
            ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp",
             "division_fn", "node_recall", "num_pred_nodes")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="nofork", choices=["baseline", "nofork"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from biotrack.metric import estimated_nodes, per_sample_metrics
    from tracking_cellmot.metrics import EvaluationResult, summarise

    tasks = [(s, c, a.mode) for s in (0, 1) for c in cached_crops(s)]
    print(f"[ablation {a.mode}] {len(tasks)} crops, {a.workers} workers", flush=True)
    res = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(score_transform, tasks):
            res[(r["split"], r["crop"])] = r
    E0C = {0: 0.7595, 1: 0.6484}
    for fold, fam in ((0, "44b6"), (1, "6bba")):
        rows = []
        for c in cached_crops(fold):
            r = res[(fold, c)]
            er = EvaluationResult(r["edge_tp"], r["edge_fp"], r["edge_fn"], r["division_tp"],
                                  r["division_fp"], r["division_fn"], r["num_pred_nodes"])
            n_est = estimated_nodes(str(ROOT / "data" / "train" / f"{c}.geff"))
            rows.append(per_sample_metrics(er, n_est, r["node_recall"]))
        s = summarise(rows)
        d = s["score"] - E0C[fold]
        print(f"  {fam}: adjJ={s['adj_edge_jaccard']:.4f} divJ={s['division_jaccard']:.4f} "
              f"composite={s['score']:.4f}  (vs E0c {E0C[fold]:.4f}: {d:+.4f})")


if __name__ == "__main__":
    main()
