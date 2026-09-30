r"""Where do our divisions die between the pre-ILP graph and the scored output?

WHY
---
Measured 2026-08-25: the pre-ILP fold-1 graph contains **24 of 125** GT divisions
(``FACT-0083``) while the scored output contains **1** (``FACT-0080``). The pipeline
discards 23 of the 24 divisions it already had. Recovering those needs no new model and no
GPU, so it precedes training anything -- and it bounds any selector built on the current
candidate set at ~19% division recall.

This module localises the loss. For every GT division (a GT node with ``out_degree == 2``)
it asks the same question of both frames and classifies what changed.

CLASSIFICATION
--------------
A GT division is "held" in a frame when BOTH of its GT daughter edges are true positives
there. For divisions held pre-ILP but not in the output:

``node_lost``        an endpoint stopped being detected -- a DETECTION/filtering loss
``fork_collapsed``   all endpoints still detected, exactly ONE daughter edge survives --
                     the bijection signature. ``linear_sum_assignment`` cannot express
                     out-degree 2, so a fork is resolved by discarding one branch.
``both_edges_lost``  endpoints detected, neither daughter edge survives
``still_held``       survived to the output

The ``fork_collapsed`` count is the one that matters: it is the portion recoverable by
changing the association STRUCTURE alone, with the candidate set untouched.

USAGE
    python scripts/win_bet/division_loss_diff.py --atlas C:/temp/error_atlas \
        --upstream pre1 --output f1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def held_map(gtedges: pd.DataFrame) -> dict[tuple[str, int], int]:
    """(dataset, gt_source) -> number of its GT daughter edges that are true positives."""
    tp = gtedges[gtedges["is_tp"]].groupby(["dataset", "gt_source"]).size()
    return {k: int(v) for k, v in tp.items()}


def detect_map(gtedges: pd.DataFrame) -> dict[tuple[str, int], bool]:
    """(dataset, gt_source) -> whether every endpoint of its GT edges was detected."""
    out: dict[tuple[str, int], bool] = {}
    for r in gtedges.itertuples():
        key = (r.dataset, int(r.gt_source))
        ok = bool(r.src_detected) and bool(r.tgt_detected)
        out[key] = out.get(key, True) and ok
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="C:/temp/error_atlas")
    ap.add_argument("--upstream", default="pre1", help="frame BEFORE the ILP/relink")
    ap.add_argument("--output", default="f1", help="the SCORED frame")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    atlas = Path(args.atlas)
    gtn = pd.read_parquet(atlas / f"gtnodes_{args.output}.parquet")
    up = pd.read_parquet(atlas / f"gtedges_{args.upstream}.parquet")
    dn = pd.read_parquet(atlas / f"gtedges_{args.output}.parquet")

    gtdiv = gtn[gtn["out_degree"] == 2]
    print(f"GT divisions: {len(gtdiv):,}")

    up_held, dn_held = held_map(up), held_map(dn)
    dn_det = detect_map(dn)

    counts = {"still_held": 0, "fork_collapsed": 0, "node_lost": 0,
              "both_edges_lost": 0, "not_held_upstream": 0}
    collapsed: list[dict] = []

    for r in gtdiv.itertuples():
        key = (r.dataset, int(r.gt_id))
        if up_held.get(key, 0) != 2:
            counts["not_held_upstream"] += 1
            continue
        survived = dn_held.get(key, 0)
        if survived == 2:
            counts["still_held"] += 1
        elif not dn_det.get(key, True):
            counts["node_lost"] += 1
        elif survived == 1:
            counts["fork_collapsed"] += 1
            collapsed.append({"dataset": r.dataset, "gt_id": int(r.gt_id), "t": int(r.t)})
        else:
            counts["both_edges_lost"] += 1

    upstream_total = sum(v for k, v in counts.items() if k != "not_held_upstream")
    print(f"held upstream ({args.upstream}): {upstream_total}")
    print(f"\nFATE BETWEEN {args.upstream} AND {args.output}")
    for k in ("still_held", "fork_collapsed", "node_lost", "both_edges_lost"):
        pct = 100.0 * counts[k] / upstream_total if upstream_total else 0.0
        print(f"  {k:<18} {counts[k]:>4}  ({pct:5.1f}% of those held upstream)")

    print(f"\n  fork_collapsed is the BIJECTION signature: every endpoint still detected,")
    print(f"  exactly one daughter edge kept. Recoverable by changing association")
    print(f"  STRUCTURE alone, candidate set untouched.")

    if collapsed:
        print(f"\n  first collapsed divisions: "
              f"{[(c['dataset'][:13], c['t']) for c in collapsed[:6]]}")

    result = {"upstream": args.upstream, "output": args.output,
              "gt_divisions": int(len(gtdiv)), "held_upstream": upstream_total,
              "counts": counts, "collapsed": collapsed}
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
