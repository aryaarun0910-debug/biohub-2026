"""Phase B step 1 — label the cached E0c candidate surface (scorer-consistent).

Operates on the cached candidate tables (no wrapper rerun). For each crop, matches the
RAW predicted nodes to GT with the scorer's optimal DistanceMatching, then labels every
candidate edge: label=1 iff BOTH endpoints match GT nodes AND the corresponding GT edge
exists (nearest-GT would be false supervision). Writes labeled candidate parquets +
per-fold summary: candidate recall and the wrapper COMPOSITE baseline (selected vs label)
that the breadth model must beat.

Output: artifacts/kaggle/e0c_cache/candidates_labeled/{split}/{crop}.parquet
"""
from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

CACHE = ROOT / "artifacts" / "kaggle" / "e0c_cache"
OUT = CACHE / "candidates_labeled"


def cached_crops(split: int):
    import json
    return sorted(json.loads(p.read_text())["crop"]
                  for p in (CACHE / "status").glob(f"{split}__*.json")
                  if json.loads(p.read_text()).get("status") == "ok")


def label_one(args) -> dict:
    split, crop = args
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, load_graph

    cand = pl.read_parquet(CACHE / "candidates" / str(split) / f"{crop}.parquet")
    if cand.height == 0:
        OUT.joinpath(str(split)).mkdir(parents=True, exist_ok=True)
        cand.with_columns(pl.lit(0).alias("label")).write_parquet(OUT / str(split) / f"{crop}.parquet")
        return {"split": split, "crop": crop, "n": 0, "pos": 0, "recall": float("nan"),
                "sel": 0, "sel_prec": float("nan")}

    # match RAW predicted nodes -> GT (scorer-consistent optimal matching)
    raw = load_graph(str(ROOT / "artifacts" / "kaggle" / "oof_clean" / f"pred_geffs_split_{split}" / f"{crop}.geff"))
    gt = load_graph(str(ROOT / "data" / "train" / f"{crop}.geff"))
    g = td.graph.InMemoryGraph()
    for k in ["z", "y", "x"]:
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    rn = raw.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t", "z", "y", "x"])
    raw_ids = rn[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list()
    internal = g.bulk_add_nodes([{"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
                                 for t, z, y, x in zip(rn["t"].to_list(), rn["z"].to_list(),
                                                       rn["y"].to_list(), rn["x"].to_list())])
    raw_to_int = {int(rid): internal[i] for i, rid in enumerate(raw_ids)}
    g.match(gt, matching=DistanceMatching(max_distance=MAX_DISTANCE, scale=DEFAULT_SCALE))
    na = g.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    int_to_gt = {int(r[td.DEFAULT_ATTR_KEYS.NODE_ID]):
                 (None if r[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID] in (None, -1)
                  else int(r[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]))
                 for r in na.iter_rows(named=True)}
    raw_to_gt = {rid: int_to_gt.get(raw_to_int[rid]) for rid in map(int, raw_ids)}
    gt_edges = set()
    if gt.num_edges() > 0:
        ea = gt.edge_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET])
        gt_edges = {(int(s), int(t)) for s, t in zip(ea[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list(),
                                                     ea[td.DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list())}

    srcs = cand["source_id"].to_list()
    tgts = cand["target_id"].to_list()
    labels = np.fromiter(
        (1 if (raw_to_gt.get(s) is not None and raw_to_gt.get(t) is not None
               and (raw_to_gt[s], raw_to_gt[t]) in gt_edges) else 0
         for s, t in zip(srcs, tgts)), dtype=np.int64, count=len(srcs))
    lab = cand.with_columns(pl.Series("label", labels))
    OUT.joinpath(str(split)).mkdir(parents=True, exist_ok=True)
    lab.write_parquet(OUT / str(split) / f"{crop}.parquet")

    # per-source: recall (true target present) + wrapper composite precision (selected vs label)
    per_src = lab.group_by("source_id").agg(pl.col("label").max().alias("hp"))
    recall = float(per_src["hp"].mean())
    sel = lab.filter(pl.col("selected") == 1)
    return {"split": split, "crop": crop, "n": lab.height, "pos": int(labels.sum()),
            "recall": recall, "sel": sel.height,
            "sel_prec": float(sel["label"].mean()) if sel.height else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    tasks = [(s, c) for s in (0, 1) for c in cached_crops(s)]
    print(f"[label] {len(tasks)} crops, {a.workers} workers")
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(label_one, tasks), 1):
            rows.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)
    df = pl.DataFrame(rows)
    for fold, fam in ((0, "44b6"), (1, "6bba")):
        d = df.filter(pl.col("split") == fold)
        n, pos = int(d["n"].sum()), int(d["pos"].sum())
        # candidate recall + wrapper composite precision, edge-volume-agnostic mean over crops
        rec = float(d["recall"].drop_nans().mean())
        selp = float((d["sel_prec"] * d["sel"]).sum() / max(1, d["sel"].sum()))
        print(f"\n=== labeled candidates held-out {fam} ===")
        print(f"  {n:,} candidates, {pos:,} positive ({100*pos/max(1,n):.1f}%)")
        print(f"  candidate recall (source has true target in pool) = {rec:.4f}")
        print(f"  WRAPPER COMPOSITE selected-edge precision vs label = {selp:.4f}  (the baseline to beat)")


if __name__ == "__main__":
    main()
