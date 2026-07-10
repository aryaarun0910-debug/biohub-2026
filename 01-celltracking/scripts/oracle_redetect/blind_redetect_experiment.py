"""Blind dangling-track redetection with exact post-hoc OOF scoring.

Query generation, image peak selection, confidence gating, and count budgeting do
not use GT. GT is loaded only after selection to score the modified graph. The
default mode is a dry run over one crop; pass --apply to construct and score.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from biotrack.metric import estimated_nodes  # noqa: E402
from biotrack.metric_numpy import score_sample  # noqa: E402
from core import (  # noqa: E402
    apply_redetection_proposals,
    blind_redetection_proposals,
    count_multiplier,
    load_sample,
    select_redetection_proposals,
)
from measure_oracles import discover  # noqa: E402
from redetect_diagnostic import open_volume  # noqa: E402


def parse_directions(value: str) -> tuple[str, ...]:
    return ("forward", "backward") if value == "both" else (value,)


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--pred-dir", action="append", required=True, type=Path)
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--image-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--fold", choices=["44b6", "6bba"], required=True)
    ap.add_argument("--max-crops", type=int, default=1)
    ap.add_argument("--max-queries", type=int, default=1000)
    ap.add_argument("--direction", choices=["forward", "backward", "both"], default="both")
    ap.add_argument("--no-stationary", action="store_true")
    ap.add_argument("--search-radius-um", type=float, default=7.0)
    ap.add_argument("--smooth-sigma-um", type=float, default=0.8)
    ap.add_argument("--top-k-per-query", type=int, default=3)
    ap.add_argument("--peak-nms-um", type=float, default=2.0)
    ap.add_argument("--min-confidence", type=float, default=1.0)
    ap.add_argument("--max-added-nodes", type=int, default=500)
    ap.add_argument("--max-added-ratio", type=float, default=0.02)
    ap.add_argument("--dedup-um", type=float, default=2.0)
    ap.add_argument("--apply", action="store_true", help="construct and exact-score graph; absent = dry run")
    ap.add_argument("--out-csv", type=Path)
    args = ap.parse_args()

    if args.max_crops < 1 or args.max_queries < 1:
        ap.error("--max-crops and --max-queries must be positive")
    preds = discover(args.pred_dir, {args.fold})
    selected_crops = list(sorted(preds.items()))[: args.max_crops]
    rows = []
    mode = "APPLY+SCORE" if args.apply else "DRY RUN"
    print(f"mode={mode}; crops={len(selected_crops)}; GT is not used for queries or selection")
    for i, (stem, pred_path) in enumerate(selected_crops, 1):
        pred = load_sample(pred_path)
        volume = open_volume(args.image_dir / f"{stem}.zarr")
        queries, proposals = blind_redetection_proposals(
            pred,
            len(volume),
            lambda t, v=volume: v[t],
            directions=parse_directions(args.direction),
            allow_stationary=not args.no_stationary,
            search_radius_um=args.search_radius_um,
            smooth_sigma_um=args.smooth_sigma_um,
            top_k_per_query=args.top_k_per_query,
            peak_nms_um=args.peak_nms_um,
            max_queries=args.max_queries,
        )
        chosen = select_redetection_proposals(
            pred,
            proposals,
            min_confidence=args.min_confidence,
            max_added_nodes=args.max_added_nodes,
            max_added_ratio=args.max_added_ratio,
            dedup_um=args.dedup_um,
        )
        row = {
            "crop": stem,
            "fold": args.fold,
            "mode": mode,
            "queries": len(queries),
            "raw_proposals": len(proposals),
            "added_nodes": len(chosen),
        }
        if args.apply:
            gt_path = args.gt_dir / f"{stem}.geff"
            if not gt_path.exists():
                raise FileNotFoundError(f"missing GT needed only for post-hoc score: {gt_path}")
            gt = load_sample(gt_path)
            n_est = estimated_nodes(gt_path)
            modified = apply_redetection_proposals(pred, chosen)
            before = score_sample(pred, gt, n_est)
            after = score_sample(modified, gt, n_est)
            row.update(
                {
                    "baseline_tp": before["edge_tp"],
                    "baseline_fp": before["edge_fp"],
                    "baseline_fn": before["edge_fn"],
                    "baseline_count_multiplier": count_multiplier(len(pred.node_ids), n_est),
                    "baseline_adjJ": before["adj_edge_jaccard"],
                    "after_tp": after["edge_tp"],
                    "after_fp": after["edge_fp"],
                    "after_fn": after["edge_fn"],
                    "after_count_multiplier": count_multiplier(len(modified.node_ids), n_est),
                    "after_adjJ": after["adj_edge_jaccard"],
                    "delta_tp": after["edge_tp"] - before["edge_tp"],
                    "delta_fp": after["edge_fp"] - before["edge_fp"],
                    "delta_fn": after["edge_fn"] - before["edge_fn"],
                    "delta_adjJ": after["adj_edge_jaccard"] - before["adj_edge_jaccard"],
                }
            )
            print(
                f"[{i}/{len(selected_crops)}] {stem}: queries={len(queries)} proposals={len(proposals)} "
                f"added={len(chosen)} TP/FP/FN {before['edge_tp']}/{before['edge_fp']}/{before['edge_fn']} -> "
                f"{after['edge_tp']}/{after['edge_fp']}/{after['edge_fn']}; "
                f"count-mult {row['baseline_count_multiplier']:.5f}->{row['after_count_multiplier']:.5f}; "
                f"adjJ {before['adj_edge_jaccard']:.5f}->{after['adj_edge_jaccard']:.5f} "
                f"(delta={row['delta_adjJ']:+.5f})"
            )
        else:
            print(
                f"[{i}/{len(selected_crops)}] {stem}: queries={len(queries)} raw-proposals={len(proposals)} "
                f"would-add={len(chosen)} (confidence/count/dedup gated)"
            )
        rows.append(row)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print(f"wrote {args.out_csv}")
    if not args.apply:
        print("Dry run made no graph and loaded no GT. Re-run identical arguments with --apply for exact OOF deltas.")


if __name__ == "__main__":
    main()
