"""E0c Stages 2-4 — score the cached post-wrapper graphs (decoupled from the wrapper pass).

- Stage 2 (fast, single proc): numpy EDGE-only diagnostic via metric_numpy over the cache.
- Stage 3 (authoritative, process pool): exact edge + division score via tracksdata; aggregate
  weighted adj-edge-J + division-J -> composite score, per fold + min-fold.
- Stage 4 (parity): compare numpy vs authoritative EDGE stats across ALL cached crops --
  upgrades parity evidence from 1 crop to the E0c population.

metric_numpy is diagnostic only (no divisions); the authoritative score is the baseline.
Only crops with status=ok in the cache are scored; missing/failed crops are reported, never
silently dropped.

Usage: e0c_score.py [--workers 4]
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
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

CACHE = ROOT / "artifacts" / "kaggle" / "e0c_cache"


def _geff_to_sample(path: str | Path):
    """Convert a GEFF graph without depending on a retired experiment module."""
    from biotrack.metric import load_graph
    from biotrack.metric_numpy import Sample
    graph = load_graph(str(path))
    nodes = graph.node_attrs().sort("node_id")
    edges = graph.edge_attrs()
    return Sample(
        node_ids=nodes["node_id"].to_numpy().astype(np.int64),
        t=nodes["t"].to_numpy().astype(np.int64),
        zyx=nodes.select(["z", "y", "x"]).to_numpy().astype(float),
        edges=(edges.select(["source_id", "target_id"]).to_numpy().astype(np.int64)
               if len(edges) else np.empty((0, 2), dtype=np.int64)),
    )


def cached_crops(split: int, cache: Path = CACHE):
    out = []
    for sp in (cache / "status").glob(f"{split}__*.json"):
        st = json.loads(sp.read_text())
        if st.get("status") == "ok":
            out.append(st["crop"])
    return sorted(out)


def _pred_sample(split: int, crop: str, cache: Path = CACHE):
    from biotrack.metric_numpy import Sample
    df = pl.read_parquet(cache / "graphs" / str(split) / f"{crop}.parquet")
    nd = df.filter(pl.col("row_type") == "node")
    ids = nd["node_id"].to_numpy().astype(np.int64)
    t = nd["t"].to_numpy().astype(np.int64)
    zyx = np.stack([nd["z"].to_numpy(), nd["y"].to_numpy(), nd["x"].to_numpy()], 1).astype(float)
    ed = df.filter(pl.col("row_type") == "edge")
    edges = (np.stack([ed["source_id"].to_numpy(), ed["target_id"].to_numpy()], 1).astype(np.int64)
             if ed.height > 0 else np.zeros((0, 2), np.int64))
    return Sample(node_ids=ids, t=t, zyx=zyx, edges=edges)


def numpy_edge(split: int, crop: str, cache: Path = CACHE) -> dict:
    from biotrack.metric import estimated_nodes
    from biotrack.metric_numpy import score_sample
    gt_geff = ROOT / "data" / "train" / f"{crop}.geff"
    pred = _pred_sample(split, crop, cache)
    gt = _geff_to_sample(gt_geff)
    r = score_sample(pred, gt, estimated_nodes(str(gt_geff)))
    return {"crop": crop, "split": split, "np_tp": r["edge_tp"], "np_fp": r["edge_fp"],
            "np_fn": r["edge_fn"], "np_adj": r["adj_edge_jaccard"]}


def auth_score(args) -> dict:
    """Process-pool worker: authoritative edge + division score for one cached crop."""
    split, crop, cache_str = args
    cache = Path(cache_str)
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from biotrack.metric import score_pred_graph
    from biotrack.submission import submission_to_graphs
    gdf = (pl.read_parquet(cache / "graphs" / str(split) / f"{crop}.parquet")
           .with_columns(pl.lit(crop).alias("dataset")).with_row_index("id"))
    graph = submission_to_graphs(gdf)[crop]
    row = score_pred_graph(graph, str(ROOT / "data" / "train" / f"{crop}.geff"))
    return {"crop": crop, "split": split, **{k: row[k] for k in
            ("edge_tp", "edge_fp", "edge_fn", "adj_edge_jaccard", "division_tp",
             "division_fp", "division_fn", "node_recall", "num_pred_nodes")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", type=Path, default=CACHE)
    a = ap.parse_args()
    from tracking_cellmot.metrics import summarise

    cache = a.cache.resolve()
    pairs = [(s, c) for s in (0, 1) for c in cached_crops(s, cache)]
    tasks = [(s, c, str(cache)) for s, c in pairs]
    n0, n1 = len(cached_crops(0, cache)), len(cached_crops(1, cache))
    print(f"[e0c_score] cached ok: fold0(44b6)={n0}/71  fold1(6bba)={n1}/128  total={len(tasks)}")
    if not tasks:
        sys.exit("no cached crops with status=ok yet")

    print("Stage 2 (numpy edge, fast)...")
    np_rows = {(s, c): numpy_edge(s, c, cache) for s, c in pairs}

    print(f"Stage 3 (authoritative edge+division, {a.workers} workers)...")
    auth = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(auth_score, tasks):
            auth[(r["split"], r["crop"])] = r

    # aggregate authoritative per fold + composite (adj-edge-J needs n_est count-adjust,
    # already inside score_pred_graph's adj_edge_jaccard/summarise weighting)
    from biotrack.metric import estimated_nodes, per_sample_metrics
    from tracking_cellmot.metrics import EvaluationResult
    for fold, fam in ((0, "44b6"), (1, "6bba")):
        rows = []
        for c in cached_crops(fold, cache):
            r = auth[(fold, c)]
            er = EvaluationResult(r["edge_tp"], r["edge_fp"], r["edge_fn"],
                                  r["division_tp"], r["division_fp"], r["division_fn"], r["num_pred_nodes"])
            n_est = estimated_nodes(str(ROOT / "data" / "train" / f"{c}.geff"))
            rows.append(per_sample_metrics(er, n_est, r["node_recall"]))
        s = summarise(rows)
        print(f"\n=== E0c AUTHORITATIVE held-out {fam} ({len(rows)} crops) ===")
        print(f"  adj_edge_jaccard = {s['adj_edge_jaccard']:.4f}")
        print(f"  division_jaccard = {s['division_jaccard']:.4f} "
              f"(TP={s['division_tp']} FP={s['division_fp']} FN={s['division_fn']})")
        print(f"  composite score  = {s['score']:.4f}")

    # Stage 4: numpy vs authoritative EDGE parity across the population
    diffs = []
    for k in pairs:
        a_r, n_r = auth[k], np_rows[k]
        diffs.append((abs(a_r["edge_tp"] - n_r["np_tp"]) + abs(a_r["edge_fp"] - n_r["np_fp"])
                      + abs(a_r["edge_fn"] - n_r["np_fn"]), abs(a_r["adj_edge_jaccard"] - n_r["np_adj"])))
    n_edge_mismatch = sum(1 for d, _ in diffs if d > 0)
    max_adj = max(d for _, d in diffs)
    print(f"\n=== Stage 4 numpy-vs-authoritative EDGE parity over {len(tasks)} crops ===")
    print(f"  crops with edge TP/FP/FN mismatch: {n_edge_mismatch}/{len(tasks)}")
    print(f"  max |adj_edge_J diff|: {max_adj:.2e}")


if __name__ == "__main__":
    main()
