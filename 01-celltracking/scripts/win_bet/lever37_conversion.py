r"""Did the extra candidates actually CONVERT into true edges?

WHY THIS IS A SEPARATE INSTRUMENT
---------------------------------
``PKT-0027``'s preregistered rule (3): added reach without converted true edges is a RANKING
FAILURE, not a null. Reach only says the true pair was OFFERED to the linker. Whether the linker
then SELECTED it is a different question, and it is the one that decides where the effort goes:

  reach up, true edges up      the widened candidate set is being used - the lever works
  reach up, true edges flat    the pipeline is offered the right parent and does not take it.
                               That is evidence about the edge model and the solver's
                               discrimination, and it redirects effort to the SCORER rather than
                               to widening further.
  reach up, true edges down    the extra candidates are actively displacing correct ones.

WHAT IT MEASURES, AND THE LIMIT OF IT
-------------------------------------
Per crop it computes the OPPORTUNITY - GT edges reachable under the treatment's (floor, rank)
but NOT under the deployed rule - and the REALISED change in true edges from the full-chain
replay. The conversion rate is the realised change over the opportunity.

That realised change is a NET count: the replay outputs retain per-crop metrics, not the final
graph, so a crop that gains ten new true edges and loses ten old ones is indistinguishable here
from one that changed nothing. The net is what moves the score, so it is the right quantity for
promotion; but if the net lands near zero with a large opportunity, identity-level attribution
needs a re-run that persists the final graphs, and this module says so rather than guessing.

``num_pred_nodes`` is carried through as well, because the packet requires node retention to be
reported separately - a widened candidate set can retain nodes the solver would otherwise drop,
and that is a different mechanism from converting edges.
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

from audit_p30_acquisition import load_sidecar  # noqa: E402
from candidate_reach_curve import gt_pairs_in_node_space, rank_within_target  # noqa: E402


def opportunity(crop: str, sidecar: Path, pre: pl.DataFrame, gt_geff: Path,
                deployed_floor: float, floor: float, rank: int) -> dict:
    """GT edges the treatment offers that the deployed rule did not."""
    nodes = pre.filter(pl.col("row_type") == "node")
    gt_set, _total, detectable = gt_pairs_in_node_space(crop, gt_geff, nodes)
    side = load_sidecar(sidecar)
    src, tgt, prob = side["source_id"], side["target_id"], side["edge_prob"]
    ranks = rank_within_target(src, tgt, prob)

    deployed_keep = prob > deployed_floor
    deployed_offered = set(zip(src[deployed_keep].tolist(), tgt[deployed_keep].tolist()))
    treat_keep = (prob > floor) & (ranks <= rank)
    treat_offered = set(zip(src[treat_keep].tolist(), tgt[treat_keep].tolist()))

    return {
        "crop": crop,
        "gt_detectable": detectable,
        "reached_deployed": len(gt_set & deployed_offered),
        "reached_treatment": len(gt_set & treat_offered),
        "newly_reachable": len((gt_set & treat_offered) - deployed_offered),
        "candidates_deployed": int(deployed_keep.sum()),
        "candidates_treatment": int(treat_keep.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preilp", type=Path, required=True)
    ap.add_argument("--ecb-dir", type=Path, required=True)
    ap.add_argument("--treatment-dir", type=Path, required=True)
    ap.add_argument("--treatment-pattern", default="l37_*.json")
    ap.add_argument("--control-csv", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--deployed-floor", type=float, default=0.5)
    ap.add_argument("--floor", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, estimated_nodes, load_graph
    from biotrack.submission import read_submission, submission_to_graphs
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics

    control_graphs = submission_to_graphs(read_submission(args.control_csv))
    pre = pl.read_parquet(args.preilp)

    rows = []
    for path in sorted(args.treatment_dir.glob(args.treatment_pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        crop = payload["crop"]
        gt_geff = args.gt_dir / f"{crop}.geff"
        if crop not in control_graphs or not gt_geff.exists():
            continue
        gt = load_graph(gt_geff)
        pred = control_graphs[crop]
        er = evaluate(pred, gt, scale=DEFAULT_SCALE, max_distance=MAX_DISTANCE)
        recall = node_recall(pred, gt) if pred.num_edges() and pred.num_nodes() else 0.0
        control = per_sample_metrics(er, estimated_nodes(gt_geff), recall)
        treat = payload["score"]

        opp = opportunity(
            crop, args.ecb_dir / f"{crop}.npz", pre.filter(pl.col("dataset") == crop),
            gt_geff, args.deployed_floor, args.floor, args.rank,
        )
        rows.append({
            **opp,
            "control_edge_tp": float(control["edge_tp"]),
            "treat_edge_tp": float(treat["edge_tp"]),
            "delta_edge_tp": float(treat["edge_tp"]) - float(control["edge_tp"]),
            "control_edge_fp": float(control["edge_fp"]),
            "treat_edge_fp": float(treat["edge_fp"]),
            "control_division_tp": float(control["division_tp"]),
            "treat_division_tp": float(treat["division_tp"]),
            "control_nodes": float(control["num_pred_nodes"]),
            "treat_nodes": float(treat["num_pred_nodes"]),
            "control_node_recall": float(control["node_recall"]),
            "treat_node_recall": float(treat["node_recall"]),
        })

    if not rows:
        raise SystemExit("no paired crops found")
    t = pl.DataFrame(rows)
    newly = int(t["newly_reachable"].sum())
    d_tp = float(t["delta_edge_tp"].sum())
    d_fp = float((t["treat_edge_fp"] - t["control_edge_fp"]).sum())
    d_div = float((t["treat_division_tp"] - t["control_division_tp"]).sum())

    result = {
        "schema_version": 1,
        "n_crops": t.height,
        "treatment": {"floor": args.floor, "rank": args.rank, "deployed_floor": args.deployed_floor},
        "opportunity": {
            "gt_detectable": int(t["gt_detectable"].sum()),
            "reached_deployed": int(t["reached_deployed"].sum()),
            "reached_treatment": int(t["reached_treatment"].sum()),
            "newly_reachable": newly,
            "candidates_deployed": int(t["candidates_deployed"].sum()),
            "candidates_treatment": int(t["candidates_treatment"].sum()),
        },
        "realised": {
            "delta_edge_tp": d_tp,
            "delta_edge_fp": d_fp,
            "delta_division_tp": d_div,
            "delta_nodes": float((t["treat_nodes"] - t["control_nodes"]).sum()),
            "control_node_recall": float(t["control_node_recall"].mean()),
            "treat_node_recall": float(t["treat_node_recall"].mean()),
        },
        "conversion_rate": (d_tp / newly) if newly else None,
        "caveat": (
            "delta_edge_tp is a NET count; the replay outputs do not retain final graphs, so "
            "gained and lost true edges cannot be separated here. A near-zero net against a "
            "large opportunity is the RANKING-FAILURE signature (PKT-0027 rule 3) and warrants "
            "an identity-level re-run that persists graphs before any further widening."
        ),
        "per_crop": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")

    o, r = result["opportunity"], result["realised"]
    print(
        f"\nLEVER37_CONVERSION crops={t.height} floor={args.floor} rank={args.rank}\n"
        f"  OPPORTUNITY  detectable={o['gt_detectable']:,}  reached deployed={o['reached_deployed']:,} "
        f"-> treatment={o['reached_treatment']:,}  NEWLY REACHABLE={newly:,}\n"
        f"               candidates {o['candidates_deployed']:,} -> {o['candidates_treatment']:,} "
        f"({o['candidates_treatment'] / max(o['candidates_deployed'], 1):.2f}x)\n"
        f"  REALISED     d_edge_TP={d_tp:+,.0f}   d_edge_FP={d_fp:+,.0f}   "
        f"d_division_TP={d_div:+,.0f}   d_nodes={r['delta_nodes']:+,.0f}\n"
        f"               node_recall {r['control_node_recall']:.4f} -> {r['treat_node_recall']:.4f}\n"
        f"  CONVERSION   {result['conversion_rate'] if newly else 'n/a'} "
        f"(realised true edges per newly reachable GT edge)"
    )
    if newly and d_tp <= 0:
        print("  => RANKING FAILURE signature: the true parent was offered and not selected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
