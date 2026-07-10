"""Fast exact-edge sweep of organizer edge probability with frozen detections.

Node assignment is computed once per crop, then reused across thresholds. The
default outgoing cap of one also suppresses the learned baseline's noisy forks.
The output is meant for cross-embryo threshold selection, not per-crop tuning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "oracle_redetect"))

from biotrack.metric import estimated_nodes, load_graph  # noqa: E402
from biotrack.metric_numpy import ALPHA, match_nodes  # noqa: E402
from core import load_sample  # noqa: E402


def selected_edges(rows, threshold: float, max_children: int) -> list[tuple[int, int]]:
    by_source: dict[int, list[tuple[float, int]]] = {}
    for source, target, probability in rows:
        if probability >= threshold:
            by_source.setdefault(source, []).append((probability, target))
    output: list[tuple[int, int]] = []
    for source, candidates in by_source.items():
        candidates.sort(reverse=True)
        output.extend((source, target) for _, target in candidates[:max_children])
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", action="append", required=True)
    parser.add_argument("--gt-dir", default=str(ROOT / "data" / "train"))
    parser.add_argument(
        "--thresholds",
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95",
    )
    parser.add_argument("--max-children", type=int, default=1)
    args = parser.parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    if args.max_children < 1:
        parser.error("--max-children must be positive")

    totals: dict[str, dict[float, list[tuple[float, int]]]] = {}
    gt_dir = Path(args.gt_dir)
    for pred_dir in map(Path, args.pred_dir):
        for pred_path in sorted(pred_dir.glob("*.geff")):
            embryo = pred_path.stem.split("_", 1)[0]
            pred = load_sample(pred_path)
            gt_path = gt_dir / pred_path.name
            gt = load_sample(gt_path)
            matched = match_nodes(pred, gt)
            gt_edges = {(int(s), int(t)) for s, t in gt.edges}
            gt_out = {source for source, _ in gt_edges}
            gt_in = {target for _, target in gt_edges}
            attrs = load_graph(str(pred_path)).edge_attrs(attr_keys=["edge_prob"])
            rows = [
                (int(source), int(target), float(probability))
                for source, target, probability in attrs.select(
                    ["source_id", "target_id", "edge_prob"]
                ).iter_rows()
            ]
            n_est = estimated_nodes(gt_path)
            count_multiplier = max(0.0, 1.0 - ALPHA * (len(pred.node_ids) - n_est) / n_est)
            fold = totals.setdefault(embryo, {threshold: [] for threshold in thresholds})
            for threshold in thresholds:
                tp = valid = 0
                for source, target in selected_edges(rows, threshold, args.max_children):
                    gs, gt_target = matched.get(source), matched.get(target)
                    is_tp = gs is not None and gt_target is not None and (gs, gt_target) in gt_edges
                    is_valid = (gs is not None and gs in gt_out) or (
                        gt_target is not None and gt_target in gt_in
                    )
                    tp += int(is_tp)
                    valid += int(is_valid)
                fp = valid - tp
                fn = len(gt_edges) - tp
                denominator = tp + fp + fn
                adjusted = (tp / denominator if denominator else 0.0) * count_multiplier
                fold[threshold].append((adjusted, denominator))

    for embryo, threshold_rows in sorted(totals.items()):
        print(f"\n{embryo}")
        best = (-1.0, None)
        for threshold in thresholds:
            rows = threshold_rows[threshold]
            weight = sum(denominator for _, denominator in rows)
            score = sum(value * denominator for value, denominator in rows) / weight
            print(f"  threshold={threshold:>5.2f} edgeJ={score:.6f}")
            best = max(best, (score, threshold))
        print(f"  BEST threshold={best[1]:.2f} edgeJ={best[0]:.6f}")


if __name__ == "__main__":
    main()
