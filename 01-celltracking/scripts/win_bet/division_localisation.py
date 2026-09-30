r"""Division recovery conditioned on localisation residual - Lime's observation, tested on CPU.

WHAT IS BEING TESTED
--------------------
``FACT-0348`` records an external competitor report that division recovery behaves as though it is
especially sensitive inside roughly 3 um even though the official matcher admits 7 um. ``FACT-0349``
verified at source WHY that is mechanically possible: the scorer matches each GT division window at
7 um and then requires a local directed fork whose two predicted child branches map to two distinct
GT daughter lineages, so a node can be admissible and still land on the wrong cell under one-to-one
assignment or feed the wrong branch. Neither fact measures the effect on our own data. This does.

METHOD
------
For every GT division in a fold:

  outcome    the OFFICIAL per-division score from ``tracking_cellmot.division_metrics.score_divisions``
             - the same function the leaderboard metric calls, not a re-derivation
  residual   the localisation residual of the division tuple - the parent (divider) and its two GT
             daughters - each measured as the distance from that GT node to the predicted node the
             one-to-one 7 um matcher assigned it, or unmatched

Recovery rate is then reported per residual band. If Lime's observation reproduces, recovery should
fall off well inside 7 um rather than only at the admissibility edge.

WHY THIS MATTERS FOR THE ROADMAP
--------------------------------
A dedicated mother-daughter pair head is a large build. It is worth it only if division recovery is
limited by something a pair head addresses - geometry and branch assignment near the division - as
opposed to the daughters simply not being detected. Splitting the failures by whether the tuple was
even matched separates those two, and the residual bands say how much precision a pair head needs.

CAVEAT, STATED UP FRONT
-----------------------
The outcome comes from the official scorer; the residual comes from this module's own one-to-one
match, the same implementation as the recall numbers in ``FACT-0354``/``FACT-0355``. They are the
same algorithm at the same radius but are computed independently, so treat the residual as a
descriptive covariate on the official outcome, not as the scorer's internal assignment.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tracksdata as td  # noqa: E402

from detpeak_curve import MAX_DISTANCE_UM, SCALE_UM, match_one_to_one_pairs  # noqa: E402

SCALE = np.asarray(SCALE_UM, dtype=np.float64)
DEFAULT_BANDS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0)


def build_pred_graph(rows: pl.DataFrame):
    """Predicted graph for one crop from submission rows, as biotrack.submission builds it."""
    nodes = rows.filter(pl.col("row_type") == "node").sort("node_id")
    edges = rows.filter(pl.col("row_type") == "edge")
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    internal = graph.bulk_add_nodes([
        {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
        for t, z, y, x in zip(nodes["t"], nodes["z"], nodes["y"], nodes["x"])
    ])
    remap = {int(o): int(n) for o, n in zip(nodes["node_id"], internal)}
    if edges.height:
        graph.bulk_add_edges([
            {"source_id": remap[int(s)], "target_id": remap[int(t)]}
            for s, t in zip(edges["source_id"], edges["target_id"])
            if int(s) in remap and int(t) in remap
        ])
    return graph, nodes


def gt_residuals(gt_nodes: pl.DataFrame, pred_nodes: pl.DataFrame) -> dict[int, float]:
    """GT node_id -> distance in microns to its one-to-one matched predicted node."""
    out: dict[int, float] = {}
    pred_t = pred_nodes["t"].to_numpy().astype(np.int64)
    pred_xyz = pred_nodes.select(["z", "y", "x"]).to_numpy().astype(np.float64) * SCALE
    gt_t = gt_nodes["t"].to_numpy().astype(np.int64)
    gt_xyz = gt_nodes.select(["z", "y", "x"]).to_numpy().astype(np.float64) * SCALE
    gt_ids = gt_nodes["node_id"].to_numpy().astype(np.int64)
    for frame in np.unique(gt_t):
        gm = gt_t == frame
        pm = pred_t == frame
        if not pm.any():
            continue
        for g, _, dist in match_one_to_one_pairs(pred_xyz[pm], gt_xyz[gm], MAX_DISTANCE_UM):
            out[int(gt_ids[gm][g])] = float(dist)
    return out


def division_tuples(gt_graph) -> dict[int, list[int]]:
    """GT divider node id -> its daughter node ids (out-degree >= 2)."""
    edges = gt_graph.edge_attrs().to_pandas()
    children: dict[int, list[int]] = defaultdict(list)
    for s, t in zip(edges["source_id"], edges["target_id"]):
        children[int(s)].append(int(t))
    return {k: v for k, v in children.items() if len(v) >= 2}


def crop_rows(crop: str, rows: pl.DataFrame, gt_geff: Path) -> list[dict]:
    from biotrack.metric import load_graph
    from tracking_cellmot.division_metrics import score_divisions

    gt_graph = load_graph(gt_geff)
    tuples = division_tuples(gt_graph)
    if not tuples:
        return []

    pred_graph, pred_nodes = build_pred_graph(rows)
    scores = score_divisions(pred_graph, gt_graph, tuple(SCALE_UM), MAX_DISTANCE_UM).scores

    gt_nodes = pl.from_pandas(gt_graph.node_attrs().to_pandas())
    residual = gt_residuals(gt_nodes, pred_nodes)

    out = []
    for divider, daughters in tuples.items():
        members = [divider] + list(daughters[:2])
        res = [residual.get(m) for m in members]
        matched = [r for r in res if r is not None]
        out.append({
            "crop": crop,
            "divider": int(divider),
            "recovered": int(scores.get(divider, 0)),
            "n_members": len(members),
            "n_members_matched": len(matched),
            "all_members_matched": len(matched) == len(members),
            "residual_max_um": float(max(matched)) if matched else None,
            "residual_mean_um": float(np.mean(matched)) if matched else None,
            "divider_residual_um": residual.get(divider),
        })
    return out


def summarise(table: pl.DataFrame, bands: tuple[float, ...]) -> dict:
    total = table.height
    recovered = int(table["recovered"].sum())
    full = table.filter(pl.col("all_members_matched"))
    partial = table.filter(~pl.col("all_members_matched"))
    result = {
        "n_gt_divisions": total,
        "n_recovered": recovered,
        "recovery_rate": recovered / max(total, 1),
        "detection_limited": {
            "n_tuples_with_a_member_unmatched": partial.height,
            "recovery_rate_when_incomplete": (
                float(partial["recovered"].mean()) if partial.height else None
            ),
            "n_tuples_fully_matched": full.height,
            "recovery_rate_when_fully_matched": (
                float(full["recovered"].mean()) if full.height else None
            ),
        },
        "bands": [],
    }
    # THE TEST. Among tuples whose three members are ALL matched inside the official radius,
    # does recovery still fall off with residual? If it does, admissibility is not sufficiency.
    for lo, hi in zip(bands[:-1], bands[1:]):
        band = full.filter(
            (pl.col("residual_max_um") >= lo) & (pl.col("residual_max_um") < hi)
        )
        result["bands"].append({
            "residual_max_um_lo": lo,
            "residual_max_um_hi": hi,
            "n": band.height,
            "recovery_rate": float(band["recovered"].mean()) if band.height else None,
        })
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred-csv", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--bands", default=",".join(str(b) for b in DEFAULT_BANDS))
    ap.add_argument("--max-crops", type=int)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--out-table", type=Path)
    args = ap.parse_args()

    bands = tuple(float(v) for v in args.bands.split(","))
    sub = pl.read_csv(args.pred_csv)
    crops = sorted(sub["dataset"].unique().to_list())
    if args.max_crops:
        crops = crops[: args.max_crops]

    rows: list[dict] = []
    for i, crop in enumerate(crops, 1):
        gt_geff = args.gt_dir / f"{crop}.geff"
        if not gt_geff.exists():
            raise SystemExit(f"missing GT for {crop}")
        rows.extend(crop_rows(crop, sub.filter(pl.col("dataset") == crop), gt_geff))
        print(f"  [{i}/{len(crops)}] {crop} (divisions so far {len(rows)})", flush=True)

    if not rows:
        raise SystemExit("no GT divisions found in the selected crops")
    table = pl.DataFrame(rows)
    if args.out_table:
        args.out_table.parent.mkdir(parents=True, exist_ok=True)
        table.write_parquet(args.out_table)

    result = {
        "schema_version": 1,
        "pred_csv": str(args.pred_csv),
        "n_crops": len(crops),
        "max_distance_um": MAX_DISTANCE_UM,
        "scale_um": list(SCALE_UM),
        "summary": summarise(table, bands),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    s = result["summary"]
    print(
        f"DIVISION_LOCALISATION crops={len(crops)} divisions={s['n_gt_divisions']:,} "
        f"recovered={s['n_recovered']:,} ({s['recovery_rate']:.4f}) | "
        f"fully_matched_rate={s['detection_limited']['recovery_rate_when_fully_matched']} "
        f"incomplete_rate={s['detection_limited']['recovery_rate_when_incomplete']}"
    )
    for b in s["bands"]:
        if b["n"]:
            print(f"   residual_max [{b['residual_max_um_lo']},{b['residual_max_um_hi']}) um: "
                  f"n={b['n']:,} recovery={b['recovery_rate']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
