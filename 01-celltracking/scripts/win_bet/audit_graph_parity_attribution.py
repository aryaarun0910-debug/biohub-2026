r"""Attribute a champion-graph parity difference to the STAGE that produced it, per crop.

WHY A ROW-COUNT DIFFERENCE IS NOT YET A VERDICT
------------------------------------------------
`audit_p30_acquisition.py` check A asks one binary question: is the acquisition's champion graph
identical to the control's? A difference there fails the check, and correctly - but it says
nothing about whether the ACQUISITION caused it. FACT-0352 already recorded one fold-1 crop
differing by 5 nodes and 4 edges under a completely different patch (the P29 DetPeak exporter),
and FACT-0363 established that the deployed ILP has multiple optima, so a difference can be the
substrate rather than the instrument. Deciding which requires the per-crop, per-stage counters,
and they are already written: `run_stats.csv` carries 70+ of them from detection through linefit.

WHAT IT REPORTS
---------------
  * per-crop graph identity on the decompressed CSVs, so "127 of 128 identical" is measured
    rather than inferred from a row total;
  * every run_stats column that differs, with the crops it differs on, in file order - which is
    pipeline order, so the FIRST differing column names the earliest stage that moved;
  * the detection verdict separately, because `raw_nodes` parity is the load-bearing one: if
    detection differs, an exporter patch is a live suspect; if it does not, the divergence is
    downstream of anything a passive export can touch.

FAIL CLOSED. A missing file, a column set that does not overlap, or an empty crop intersection
raises. Heartbeat `GRAPH_PARITY_ATTRIBUTION_COMPLETE`; its absence is the alarm.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

HEARTBEAT = "GRAPH_PARITY_ATTRIBUTION_COMPLETE"
SORT_KEYS = ["row_type", "node_id", "source_id", "target_id", "t", "z", "y", "x"]


def per_crop_identity(a_csv: Path, b_csv: Path) -> dict:
    a = pl.read_csv(a_csv).drop("id", strict=False)
    b = pl.read_csv(b_csv).drop("id", strict=False)
    crops = sorted(set(a["dataset"].unique().to_list()) | set(b["dataset"].unique().to_list()))
    if not crops:
        raise SystemExit("no crops in either graph - refusing to compare nothing")
    keys = [k for k in SORT_KEYS if k in a.columns and k in b.columns]
    divergent = []
    for c in crops:
        x = a.filter(pl.col("dataset") == c).sort(keys)
        y = b.filter(pl.col("dataset") == c).sort(keys)
        if x.shape != y.shape or not x.equals(y):
            divergent.append({"crop": c, "rows_a": x.height, "rows_b": y.height,
                              "row_delta": x.height - y.height})
    return {"crops": len(crops), "identical": len(crops) - len(divergent),
            "divergent": divergent, "passed": not divergent}


def stage_attribution(a_stats: Path, b_stats: Path, ignore: tuple[str, ...]) -> dict:
    ra, rb = pl.read_csv(a_stats), pl.read_csv(b_stats)
    shared = [c for c in ra.columns if c in rb.columns and c != "dataset" and c not in ignore]
    if not shared:
        raise SystemExit("no shared run_stats columns - refusing to attribute nothing")
    m = ra.join(rb, on="dataset", suffix="_b")
    if m.height == 0:
        raise SystemExit("run_stats join produced no rows - refusing to attribute nothing")
    diffs = {}
    for c in shared:                      # file order is pipeline order
        d = m.filter(pl.col(c) != pl.col(c + "_b"))
        if d.height:
            diffs[c] = {"n_crops": d.height, "crops": d["dataset"].to_list()[:8],
                        "example": {"crop": d["dataset"][0], "a": d[c][0], "b": d[c + "_b"][0]}}
    return {
        "columns_compared": len(shared),
        "crops_joined": m.height,
        "differing_columns_in_pipeline_order": diffs,
        "detection_identical": "raw_nodes" not in diffs,
        "earliest_differing_stage": next(iter(diffs), None),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv-a", type=Path, required=True, help="the run under audit")
    ap.add_argument("--csv-b", type=Path, required=True, help="the control")
    ap.add_argument("--stats-a", type=Path, required=True)
    ap.add_argument("--stats-b", type=Path, required=True)
    ap.add_argument("--ignore-columns", nargs="*", default=["predict_minutes_total"],
                    help="wall-clock columns differ on every run and carry no information")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    for p in (args.csv_a, args.csv_b, args.stats_a, args.stats_b):
        if not Path(p).is_file():
            raise SystemExit(f"missing input {p} - fail closed")

    identity = per_crop_identity(args.csv_a, args.csv_b)
    stages = stage_attribution(args.stats_a, args.stats_b, tuple(args.ignore_columns))
    result = {
        "schema_version": 1,
        "a": str(args.csv_a), "b": str(args.csv_b),
        "per_crop_identity": identity,
        "stage_attribution": stages,
        "reading": (
            "detection identical + a difference first appearing downstream means no passive "
            "exporter patch can be the cause; detection differing means it can be"
        ),
        "passed": bool(identity["passed"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"  per-crop identity: {identity['identical']}/{identity['crops']} identical; "
          f"divergent={[d['crop'] for d in identity['divergent']]}")
    print(f"  detection identical (raw_nodes): {stages['detection_identical']}")
    for c, d in stages["differing_columns_in_pipeline_order"].items():
        print(f"    {c}: {d['n_crops']} crop(s) {d['crops'][:4]}")
    print(f"{HEARTBEAT} passed={result['passed']} -> {args.out}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
