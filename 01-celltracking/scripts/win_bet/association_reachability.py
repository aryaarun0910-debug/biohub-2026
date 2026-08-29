r"""Is the association ceiling reachable from the candidate edges we already generate?

THE QUESTION
------------
``FACT-0368`` measured that association is 87% of the remaining gap on fold 0 - replacing only the
edges, on the deployed node set, closes most of the distance to a perfect graph. That makes a
better association model the campaign's main line. But it leaves one thing unmeasured, and it
decides WHICH kind of association work pays:

  a better SCORER on the candidate edges we already build, or better CANDIDATE GENERATION?

A learned edge head - HOCT (``FACT-0361``), RoPE-4D (``FACT-0340``) - only re-scores pairs that
are already candidates. Any GT edge with no candidate between its endpoints is invisible to it, no
matter how good it gets. So the association ceiling splits in two:

  reachable      GT edges whose endpoints are both detected AND which have a candidate edge
                 between them. A perfect scorer on today's candidate set would recover these.
  unreachable    GT edges whose endpoints are both detected but with NO candidate edge between
                 them. These need wider candidate generation - HOCT's own gate is 15 um with 64
                 neighbours (FACT-0361), which is a different rule from ours.
  undetected     GT edges with an unmatched endpoint. Not an association problem at all.

METHOD
------
Everything is measured in PRE-ILP space, because that is where the candidate set lives: GT nodes
are matched one-to-one at the official 7 um to the pre-ILP peaks, and each GT edge is then
classified by whether a candidate edge joins its two matched peaks. Direction is checked both ways
so an edge is not called unreachable merely for being stored the other way round.

SCOPE
-----
Needs a pre-ILP candidate export, which exists for fold 1 only. That is a real limit - fold 0 is
the promotion gate - but the question here is about the candidate-generation RULE, which is shared
by both folds, so a fold-1 answer is informative about the rule even though the rate is fold-1's.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detpeak_curve import MAX_DISTANCE_UM, SCALE_UM, match_one_to_one_pairs  # noqa: E402

SCALE = np.asarray(SCALE_UM, dtype=np.float64)


def crop_reachability(crop: str, gt_geff: Path, pre: pl.DataFrame) -> dict:
    from biotrack.metric import load_graph

    nodes = pre.filter(pl.col("row_type") == "node").sort("node_id")
    edges = pre.filter(pl.col("row_type") == "edge")
    if nodes.height == 0:
        return {}

    pre_t = nodes["t"].to_numpy().astype(np.int64)
    pre_xyz = nodes.select(["z", "y", "x"]).to_numpy().astype(np.float64) * SCALE
    pre_ids = nodes["node_id"].to_numpy().astype(np.int64)

    candidate: set[tuple[int, int]] = set()
    best_prob: dict[tuple[int, int], float] = {}
    for s, t, p in zip(edges["source_id"], edges["target_id"], edges["edge_prob"]):
        key = (int(s), int(t))
        candidate.add(key)
        best_prob[key] = max(best_prob.get(key, 0.0), float(p))

    gt_graph = load_graph(gt_geff)
    gtn = gt_graph.node_attrs().to_pandas()
    gt_ids = gtn["node_id"].to_numpy().astype(np.int64)
    gt_row = {int(v): i for i, v in enumerate(gt_ids)}
    gt = gtn[["t", "z", "y", "x"]].to_numpy().astype(np.float64)
    gte = gt_graph.edge_attrs().to_pandas()

    # GT node -> pre-ILP node id, official one-to-one per frame.
    matched: dict[int, int] = {}
    residual: dict[int, float] = {}
    for frame in np.unique(gt[:, 0]).astype(np.int64):
        gm = np.nonzero(gt[:, 0] == frame)[0]
        pm = np.nonzero(pre_t == frame)[0]
        if not len(pm):
            continue
        for g, p, dist in match_one_to_one_pairs(
            pre_xyz[pm], gt[gm, 1:] * SCALE, MAX_DISTANCE_UM
        ):
            matched[int(gm[g])] = int(pre_ids[pm][p])
            residual[int(gm[g])] = float(dist)

    counts = defaultdict(int)
    probs: list[float] = []
    gaps: list[float] = []
    for s, t in zip(gte["source_id"], gte["target_id"]):
        su, tv = gt_row.get(int(s)), gt_row.get(int(t))
        if su is None or tv is None:
            continue
        counts["gt_edges"] += 1
        if su not in matched or tv not in matched:
            counts["undetected_endpoint"] += 1
            continue
        a, b = matched[su], matched[tv]
        if (a, b) in candidate:
            counts["reachable"] += 1
            probs.append(best_prob[(a, b)])
        elif (b, a) in candidate:
            counts["reachable_reversed"] += 1
            probs.append(best_prob[(b, a)])
        else:
            counts["unreachable_no_candidate"] += 1
            gaps.append(float(np.linalg.norm(
                gt[su, 1:] * SCALE - gt[tv, 1:] * SCALE
            )))
    return {
        "crop": crop,
        **counts,
        "reachable_prob_median": float(np.median(probs)) if probs else None,
        "unreachable_gt_displacement_um_median": float(np.median(gaps)) if gaps else None,
        "unreachable_gt_displacement_um_p90": float(np.percentile(gaps, 90)) if gaps else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preilp", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--max-crops", type=int)
    ap.add_argument("--crop-stride", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pre = pl.read_parquet(args.preilp)
    crops = sorted(pre["dataset"].unique().to_list())[:: args.crop_stride]
    if args.max_crops:
        crops = crops[: args.max_crops]

    rows = []
    for i, crop in enumerate(crops, 1):
        gt_geff = args.gt_dir / f"{crop}.geff"
        if not gt_geff.exists():
            raise SystemExit(f"missing GT for {crop}")
        got = crop_reachability(crop, gt_geff, pre.filter(pl.col("dataset") == crop))
        if got:
            rows.append(got)
        print(f"  [{i}/{len(crops)}] {crop}", flush=True)

    table = pl.DataFrame(rows)

    def total_of(name: str) -> int:
        # Categories that never occur are absent from the per-crop dicts entirely.
        return int(table[name].sum()) if name in table.columns else 0

    total = total_of("gt_edges")
    reversed_reach = total_of("reachable_reversed")
    reach = total_of("reachable") + reversed_reach
    unreach = total_of("unreachable_no_candidate")
    undet = total_of("undetected_endpoint")
    result = {
        "schema_version": 1,
        "preilp": str(args.preilp),
        "n_crops": len(rows),
        "gt_edges": total,
        "reachable": reach,
        "reachable_reversed": reversed_reach,
        "unreachable_no_candidate": unreach,
        "undetected_endpoint": undet,
        "share_reachable": reach / max(total, 1),
        "share_unreachable_no_candidate": unreach / max(total, 1),
        "share_undetected_endpoint": undet / max(total, 1),
        "share_reachable_of_both_endpoints_detected": reach / max(reach + unreach, 1),
        "unreachable_gt_displacement_um_median": (
            float(table["unreachable_gt_displacement_um_median"].median())
            if unreach and "unreachable_gt_displacement_um_median" in table.columns else None
        ),
        "per_crop": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(
        f"\nASSOCIATION_REACHABILITY crops={len(rows)} gt_edges={total:,}\n"
        f"  reachable (a perfect scorer on today's candidates would get these) = "
        f"{reach:,} ({result['share_reachable']:.4f})\n"
        f"  unreachable, both endpoints detected but NO candidate edge          = "
        f"{unreach:,} ({result['share_unreachable_no_candidate']:.4f})\n"
        f"  endpoint not detected at all                                       = "
        f"{undet:,} ({result['share_undetected_endpoint']:.4f})\n"
        f"  => of GT edges with both endpoints detected, "
        f"{result['share_reachable_of_both_endpoints_detected']:.4f} are candidate-reachable"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
