r"""Candidate reach as a function of probability floor AND parent rank.

WHY THIS RUNS BEFORE ANY TREATMENT IS CHOSEN
--------------------------------------------
``FACT-0370`` measured that 15.6% of fold-1 GT edges are never offered to any scorer, because
the deployed rule admits at most one parent per target (``FACT-0369``). LEVER-0037 proposes
widening that. But "widen it" is not an experiment until the two knobs have a measured shape:
a probability floor and a per-target rank cap trade recall against the number of candidates the
solver must then discriminate, and picking either from intuition would be fitting on a hunch.

So this maps the surface first. For every (floor, rank) pair it reports:

  reach          share of GT edges whose true pair is now offered, among those whose endpoints
                 are both detected - the quantity a better scorer could in principle convert
  candidates     how many candidate edges that costs, as a multiple of the deployed set - the
                 precision burden being handed to the linker
  rank_profile   where the true parent actually sits when it is offered. If true parents are
                 nearly always rank 1 or 2, a small cap captures most of the headroom and a
                 large one is pure cost.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not score anything and it does not choose. Reach is an upper bound on what widening can
buy, in exactly the sense ``FACT-0368``'s ladder is an upper bound - and ``FACT-0334``/``FACT-0337``
is the standing reminder that an oracle's headroom and a deployable gain are different numbers.
The treatment parameters are selected from this surface and then tested through the FULL chain,
with the division term reported separately (``FACT-0371`` host caution).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detpeak_curve import MAX_DISTANCE_UM, SCALE_UM, match_one_to_one_pairs  # noqa: E402

SCALE = np.asarray(SCALE_UM, dtype=np.float64)
DEFAULT_FLOORS = (0.5, 0.3, 0.2, 0.1, 0.05, 0.02)
DEFAULT_RANKS = (1, 2, 3, 4, 6, 8)


def gt_pairs_in_node_space(crop: str, gt_geff: Path, nodes: pl.DataFrame) -> tuple[set, int, int]:
    """GT edges expressed as (pre-ILP node id, pre-ILP node id), plus detection bookkeeping."""
    from biotrack.metric import load_graph

    graph = load_graph(gt_geff)
    gtn = graph.node_attrs().to_pandas()
    gt_ids = gtn["node_id"].to_numpy().astype(np.int64)
    row_of = {int(v): i for i, v in enumerate(gt_ids)}
    gt = gtn[["t", "z", "y", "x"]].to_numpy().astype(np.float64)
    gte = graph.edge_attrs().to_pandas()

    node_t = nodes["t"].to_numpy().astype(np.int64)
    node_xyz = nodes.select(["z", "y", "x"]).to_numpy().astype(np.float64) * SCALE
    node_ids = nodes["node_id"].to_numpy().astype(np.int64)

    matched: dict[int, int] = {}
    for frame in np.unique(gt[:, 0]).astype(np.int64):
        gm = np.nonzero(gt[:, 0] == frame)[0]
        pm = np.nonzero(node_t == frame)[0]
        if not len(pm):
            continue
        for g, p, _d in match_one_to_one_pairs(node_xyz[pm], gt[gm, 1:] * SCALE, MAX_DISTANCE_UM):
            matched[int(gm[g])] = int(node_ids[pm][p])

    pairs, total, detectable = set(), 0, 0
    for s, t in zip(gte["source_id"], gte["target_id"]):
        su, tv = row_of.get(int(s)), row_of.get(int(t))
        if su is None or tv is None:
            continue
        total += 1
        if su in matched and tv in matched:
            detectable += 1
            pairs.add((matched[su], matched[tv]))
    return pairs, total, detectable


def rank_within_target(source: np.ndarray, target: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """1-based rank of each candidate among the candidates sharing its target, best first."""
    order = np.lexsort((-prob, target))
    ranks = np.empty(len(source), dtype=np.int64)
    current, seen = None, 0
    for pos in order:
        if target[pos] != current:
            current, seen = target[pos], 0
        seen += 1
        ranks[pos] = seen
    return ranks


def crop_curve(crop: str, side: dict, pre: pl.DataFrame, gt_geff: Path,
               floors: tuple[float, ...], ranks: tuple[int, ...]) -> dict:
    nodes = pre.filter(pl.col("row_type") == "node")
    deployed_edges = pre.filter(pl.col("row_type") == "edge").height
    gt_set, gt_total, gt_detectable = gt_pairs_in_node_space(crop, gt_geff, nodes)
    if not gt_set:
        return {}

    src, tgt, prob = side["source_id"], side["target_id"], side["edge_prob"]
    rank = rank_within_target(src, tgt, prob)

    cells = []
    for floor in floors:
        for cap in ranks:
            keep = (prob > floor) & (rank <= cap)
            offered = set(zip(src[keep].tolist(), tgt[keep].tolist()))
            cells.append({
                "floor": floor,
                "rank_cap": cap,
                "reached": len(gt_set & offered),
                "candidates": int(keep.sum()),
            })

    # Where does the TRUE parent sit, when it is offered at all?
    pair_index = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(src.tolist(), tgt.tolist()))}
    true_ranks = [int(rank[pair_index[p]]) for p in gt_set if p in pair_index]
    return {
        "crop": crop,
        "gt_edges": gt_total,
        "gt_detectable": gt_detectable,
        "gt_in_node_space": len(gt_set),
        "deployed_edges": deployed_edges,
        "cells": cells,
        "true_parent_ranks": true_ranks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preilp", type=Path, required=True)
    ap.add_argument("--ecb-dir", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--floors", default=",".join(str(v) for v in DEFAULT_FLOORS))
    ap.add_argument("--ranks", default=",".join(str(v) for v in DEFAULT_RANKS))
    ap.add_argument("--max-crops", type=int)
    ap.add_argument("--crop-stride", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    floors = tuple(float(v) for v in args.floors.split(","))
    ranks = tuple(int(v) for v in args.ranks.split(","))
    pre = pl.read_parquet(args.preilp)
    crops = sorted(pre["dataset"].unique().to_list())[:: args.crop_stride]
    if args.max_crops:
        crops = crops[: args.max_crops]

    from audit_p30_acquisition import load_sidecar

    rows, all_ranks = [], []
    for i, crop in enumerate(crops, 1):
        side_path = args.ecb_dir / f"{crop}.npz"
        gt_geff = args.gt_dir / f"{crop}.geff"
        if not side_path.exists() or not gt_geff.exists():
            continue
        got = crop_curve(crop, load_sidecar(side_path), pre.filter(pl.col("dataset") == crop),
                         gt_geff, floors, ranks)
        if got:
            rows.append(got)
            all_ranks.extend(got.pop("true_parent_ranks"))
        print(f"  [{i}/{len(crops)}] {crop}", flush=True)

    detectable = sum(r["gt_detectable"] for r in rows)
    deployed_edges = sum(r["deployed_edges"] for r in rows)
    grid = {}
    for floor in floors:
        for cap in ranks:
            reached = sum(
                c["reached"] for r in rows for c in r["cells"]
                if c["floor"] == floor and c["rank_cap"] == cap
            )
            cands = sum(
                c["candidates"] for r in rows for c in r["cells"]
                if c["floor"] == floor and c["rank_cap"] == cap
            )
            grid[f"{floor}|{cap}"] = {
                "floor": floor,
                "rank_cap": cap,
                "reach": reached / max(detectable, 1),
                "candidates": cands,
                "candidates_vs_deployed": cands / max(deployed_edges, 1),
            }

    ranks_arr = np.asarray(all_ranks, dtype=np.int64)
    profile = {
        str(k): float((ranks_arr <= k).mean()) for k in ranks
    } if len(ranks_arr) else {}

    result = {
        "schema_version": 1,
        "n_crops": len(rows),
        "gt_detectable": detectable,
        "deployed_edges": deployed_edges,
        "grid": grid,
        "true_parent_rank_cdf": profile,
        "caveat": (
            "Reach is an UPPER BOUND on what widening can buy - the pair being offered is not "
            "the pair being chosen. Select treatment parameters here, then test them through "
            "the full chain with the division term reported separately."
        ),
        "per_crop": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")

    print(f"\nCANDIDATE_REACH_CURVE crops={len(rows)} detectable_gt_edges={detectable:,}")
    print(f"  {'floor':>7} " + " ".join(f"{('r<=%d' % k):>16}" for k in ranks))
    for floor in floors:
        cells = [grid[f"{floor}|{k}"] for k in ranks]
        print(f"  {floor:>7} " + " ".join(
            f"{c['reach']:.3f}/{c['candidates_vs_deployed']:>5.1f}x" for c in cells
        ))
    if profile:
        print("  true-parent rank CDF: " + "  ".join(f"<= {k}: {v:.3f}" for k, v in profile.items()))
    print("  (reach / candidate multiple vs the deployed set)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
