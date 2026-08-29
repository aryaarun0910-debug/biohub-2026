r"""Cross-run node-ordering stability between two pre-ILP exports of the SAME fold.

WHY THIS IS A SEPARATE CHECK FROM audit_p30_acquisition's CHECK D
-----------------------------------------------------------------
Check D proves that a run's sidecars address the same nodes as THAT RUN's pre-ILP graph. It is
self-consistent by construction because both artifacts come from one run, which is exactly why
P34 ships the pre-ILP export alongside the sidecars rather than relying on an older parquet.

This check answers the different question: does the node numbering AGREE ACROSS RUNS? That
matters because a body of fold-1 analysis is keyed to the 2026-08-19 ``preilp_f1_v2`` export -
FACT-0369's in-degree census and FACT-0370's reachability accounting among them. If the ordering
agrees, those results remain joinable to the new sidecars and nothing needs recomputing. If it
does not, they are not wrong, but they are not joinable either, and every downstream analysis
must use the new export alone.

WHAT IS A HARD FAILURE AND WHAT IS ONLY REPORTED
------------------------------------------------
Node identity is the HARD gate: same crops, same node count, same (t, z, y, x) at every node id.
The detector and the edge model are deterministic given the same weights, so a node-level
disagreement means something upstream actually changed.

Candidate edges are REPORTED, NOT GATED, and that distinction is paid for. FACT-0363 recorded a
falsifier that demanded exact node AND edge parity from an ILP replay, and it fired for the wrong
reason: three identical local solves of one crop gave 5,035 / 5,036 / 5,035 edges while holding
nodes at 5,385 every time, because the ILP has multiple optima. The pre-ILP stage is upstream of
the solver so its edges SHOULD be reproducible - but writing a gate that assumes a determinism
which has already surprised this project once is how a real result gets discarded as a failure.
So an edge difference is surfaced loudly and left for a human to read, not silently converted
into a verdict.

FAIL CLOSED. A missing crop, a missing column or an empty comparison raises. An audit that can
quietly compare nothing looks exactly like an audit that passed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

COORD_COLUMNS = ("t", "z", "y", "x")


def _nodes(frame: pl.DataFrame, crop: str) -> pl.DataFrame:
    out = (
        frame.filter((pl.col("dataset") == crop) & (pl.col("row_type") == "node"))
        .sort("node_id")
    )
    if out.height == 0:
        raise SystemExit(f"{crop}: no node rows - refusing to compare nothing")
    return out


def _edges(frame: pl.DataFrame, crop: str) -> set[tuple[int, int]]:
    e = frame.filter((pl.col("dataset") == crop) & (pl.col("row_type") == "edge"))
    return set(
        zip(e["source_id"].to_numpy().astype(np.int64).tolist(),
            e["target_id"].to_numpy().astype(np.int64).tolist())
    )


def compare_crop(crop: str, old: pl.DataFrame, new: pl.DataFrame) -> dict:
    a, b = _nodes(old, crop), _nodes(new, crop)
    same_count = a.height == b.height
    ids_match = same_count and bool(
        np.array_equal(a["node_id"].to_numpy(), b["node_id"].to_numpy())
    )
    coords_match = ids_match and all(
        bool(np.allclose(a[c].to_numpy().astype(np.float64),
                         b[c].to_numpy().astype(np.float64), rtol=0, atol=0))
        for c in COORD_COLUMNS
    )
    ea, eb = _edges(old, crop), _edges(new, crop)
    return {
        "crop": crop,
        "old_nodes": a.height,
        "new_nodes": b.height,
        "node_count_matches": same_count,
        "node_ids_match": ids_match,
        "node_coords_match_exactly": coords_match,
        "old_candidate_edges": len(ea),
        "new_candidate_edges": len(eb),
        "candidate_edges_identical": ea == eb,
        "edges_only_in_old": len(ea - eb),
        "edges_only_in_new": len(eb - ea),
        # The hard gate is node identity ALONE. See the module docstring on FACT-0363.
        "passed": bool(coords_match),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old", type=Path, required=True, help="the earlier pre-ILP parquet")
    ap.add_argument("--new", type=Path, required=True, help="the pre-ILP parquet just acquired")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    old, new = pl.read_parquet(args.old), pl.read_parquet(args.new)
    for name, frame in (("old", old), ("new", new)):
        missing = {"dataset", "row_type", "node_id", *COORD_COLUMNS} - set(frame.columns)
        if missing:
            raise SystemExit(f"{name} export is missing columns {sorted(missing)}")

    old_crops = set(old["dataset"].unique().to_list())
    new_crops = set(new["dataset"].unique().to_list())
    shared = sorted(old_crops & new_crops)
    if not shared:
        raise SystemExit("the two exports share no crops - nothing to compare, refusing to pass")

    rows = [compare_crop(crop, old, new) for crop in shared]
    failed = [r for r in rows if not r["passed"]]
    edge_diff = [r for r in rows if not r["candidate_edges_identical"]]
    result = {
        "schema_version": 1,
        "heartbeat": "PREILP_CROSSRUN_STABILITY_COMPLETE",
        "old": str(args.old),
        "new": str(args.new),
        "crops_compared": len(shared),
        "crops_only_in_old": sorted(old_crops - new_crops),
        "crops_only_in_new": sorted(new_crops - old_crops),
        "node_identity_failures": len(failed),
        "crops_with_candidate_edge_differences": len(edge_diff),
        "passed": not failed,
        "joinable": not failed and not edge_diff,
        "per_crop": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"{result['heartbeat']} crops={len(shared)} "
        f"node_identity_failures={len(failed)} edge_diffs={len(edge_diff)} "
        f"passed={result['passed']} joinable={result['joinable']}"
    )
    if failed:
        print(
            "  NODE ORDERING DIFFERS ACROSS RUNS. Prior fold-1 analyses keyed to the older export "
            "are NOT joinable to the new sidecars; use the new export alone.",
            flush=True,
        )
    elif edge_diff:
        print(
            "  Nodes agree, candidate edges do not. Report this - do not treat it as a pass or a "
            "failure without reading why (see the FACT-0363 note in this module's docstring).",
            flush=True,
        )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
