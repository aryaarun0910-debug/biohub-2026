"""Phase B step 1 — label the cached E0c candidate surface (scorer-consistent).

Operates on the cached candidate tables (no wrapper rerun). For each crop, matches the
RAW predicted nodes to GT with the scorer's optimal DistanceMatching, then labels every
candidate edge: label=1 iff BOTH endpoints match GT nodes AND the corresponding GT edge
exists (nearest-GT would be false supervision). Writes labeled candidate parquets +
per-fold summary: candidate recall and the wrapper COMPOSITE baseline (selected vs label)
that the breadth model must beat.

Competition GT is sparse, so non-positive candidates are NOT automatically negatives.
Each candidate receives a three-state supervision label:
  positive          both endpoints match an annotated GT edge
  reliable_negative source has an annotated outgoing edge and target matches a different GT node
  unlabeled         source/target is outside the annotated comparison surface

Output: artifacts/kaggle/e0c_cache/candidates_labeled_v2/{split}/{crop}.parquet
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
OUT = CACHE / "candidates_labeled_v2"


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
        return {"split": split, "crop": crop, "n": 0, "pos": 0, "reliable_neg": 0,
                "annotated_sources": 0, "answerable_sources": 0,
                "gt_annotated_sources": 0, "selected_answerable": 0,
                "selected_correct": 0}

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
    gt_outgoing: dict[int, set[int]] = {}
    for source, target in gt_edges:
        gt_outgoing.setdefault(source, set()).add(target)

    srcs = cand["source_id"].to_list()
    tgts = cand["target_id"].to_list()
    source_gt = [raw_to_gt.get(int(s)) for s in srcs]
    target_gt = [raw_to_gt.get(int(t)) for t in tgts]
    states: list[str] = []
    labels = np.zeros(len(srcs), dtype=np.int8)
    for i, (sg, tg) in enumerate(zip(source_gt, target_gt)):
        if sg is None or sg not in gt_outgoing or tg is None:
            states.append("unlabeled")
        elif tg in gt_outgoing[sg]:
            states.append("positive")
            labels[i] = 1
        else:
            states.append("reliable_negative")
    lab = cand.with_columns(
        pl.Series("source_gt_id", source_gt, dtype=pl.Int64),
        pl.Series("target_gt_id", target_gt, dtype=pl.Int64),
        pl.Series("supervision", states),
        pl.Series("label", labels),
    )
    # Ranking supervision is valid only within a source group containing its
    # annotated positive. Alternatives in such a group are decision negatives:
    # choosing one can steal the source assignment even when that alternative is
    # biologically unlabeled. Do not use candidates from other groups as negatives.
    answerable_source_ids = (
        lab.filter(pl.col("label") == 1)["source_id"].unique().to_list()
    )
    lab = lab.with_columns(
        pl.col("source_id").is_in(answerable_source_ids).alias("answerable_group"),
        pl.when(pl.col("source_id").is_in(answerable_source_ids))
        .then(pl.col("label"))
        .otherwise(None)
        .alias("rank_label"),
    )
    OUT.joinpath(str(split)).mkdir(parents=True, exist_ok=True)
    lab.write_parquet(OUT / str(split) / f"{crop}.parquet")

    # Decision-relevant accounting. Only matched predicted sources corresponding to a
    # GT source with an annotated outgoing edge are eligible association questions.
    annotated = lab.filter(pl.col("source_gt_id").is_in(list(gt_outgoing)))
    per_src = annotated.group_by("source_id").agg(
        pl.col("label").max().alias("answerable"),
        pl.col("source_gt_id").first(),
    )
    answerable_ids = per_src.filter(pl.col("answerable") == 1)["source_id"]
    selected = lab.filter((pl.col("selected") == 1) & pl.col("source_id").is_in(answerable_ids))
    selected_by_src = selected.group_by("source_id").agg(pl.col("label").max().alias("correct"))
    return {
        "split": split, "crop": crop, "n": lab.height, "pos": int(labels.sum()),
        "reliable_neg": int((lab["supervision"] == "reliable_negative").sum()),
        "annotated_sources": per_src.height,
        "answerable_sources": int(per_src["answerable"].sum()) if per_src.height else 0,
        "gt_annotated_sources": len(gt_outgoing),
        "selected_answerable": selected_by_src.height,
        "selected_correct": int(selected_by_src["correct"].sum()) if selected_by_src.height else 0,
    }


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
        rn = int(d["reliable_neg"].sum())
        ann = int(d["annotated_sources"].sum())
        ans = int(d["answerable_sources"].sum())
        gtann = int(d["gt_annotated_sources"].sum())
        sel = int(d["selected_answerable"].sum())
        selok = int(d["selected_correct"].sum())
        print(f"\n=== labeled candidates held-out {fam} ===")
        print(f"  {n:,} candidates: {pos:,} positive, {rn:,} reliable-negative, "
              f"{n-pos-rn:,} unlabeled")
        print(f"  candidate coverage | matched annotated source = {ans/max(1,ann):.4f} "
              f"({ans:,}/{ann:,})")
        print(f"  end-to-end candidate coverage | GT annotated source = {ans/max(1,gtann):.4f} "
              f"({ans:,}/{gtann:,})")
        print(f"  wrapper selected correct | answerable selected source = {selok/max(1,sel):.4f} "
              f"({selok:,}/{sel:,})")


if __name__ == "__main__":
    main()
