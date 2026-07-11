"""Cross-fold Trackastra greedy threshold/division sweep from cached candidates."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "oracle_redetect"))

from biotrack.metric import estimated_nodes  # noqa: E402
from biotrack.metric_numpy import ALPHA, match_nodes  # noqa: E402
from core import load_sample  # noqa: E402


def greedy_edges(
    candidates: list[tuple[int, int, float]],
    *,
    threshold: float,
    allow_divisions: bool,
) -> list[tuple[int, int]]:
    """Replicate Trackastra's global greedy capacity selection."""

    output: list[tuple[int, int]] = []
    out_degree: dict[int, int] = {}
    targets: set[int] = set()
    max_children = 2 if allow_divisions else 1
    for source, target, score in sorted(candidates, key=lambda row: row[2], reverse=True):
        if score < threshold:
            break
        if target in targets or out_degree.get(source, 0) >= max_children:
            continue
        output.append((source, target))
        targets.add(target)
        out_degree[source] = out_degree.get(source, 0) + 1
    return output


def agreement_bonus(
    candidates: list[tuple[int, int, float]],
    organizer_edges: set[tuple[int, int]],
    bonus: float,
) -> list[tuple[int, int, float]]:
    """Add a logit bonus when both association models propose the edge."""

    output = []
    for source, target, score in candidates:
        if (source, target) in organizer_edges and bonus:
            clipped = min(1 - 1e-6, max(1e-6, score))
            logit = math.log(clipped / (1 - clipped)) + bonus
            score = 1 / (1 + math.exp(-logit))
        output.append((source, target, score))
    return output


def find_named(roots: list[Path], name: str, suffix: str) -> Path:
    hits = [root / f"{name}{suffix}" for root in roots]
    hits = [path for path in hits if path.exists()]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one {name}{suffix}; found {hits}")
    return hits[0]


def read_candidates(path: Path) -> list[tuple[int, int, float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return [
        (int(row["source_node_id"]), int(row["target_node_id"]), float(row["score"]))
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", action="append", required=True, type=Path)
    parser.add_argument("--node-dir", action="append", required=True, type=Path)
    parser.add_argument("--organizer-dir", action="append", type=Path)
    parser.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    parser.add_argument(
        "--thresholds",
        default="0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.92,0.94,0.96,0.98",
    )
    parser.add_argument("--agreement-bonuses", default="0")
    parser.add_argument("--materialize-dir", type=Path)
    parser.add_argument("--materialize-threshold", type=float)
    parser.add_argument("--materialize-mode", choices=["div", "nodiv"], default="div")
    parser.add_argument("--materialize-agreement-bonus", type=float, default=0.0)
    parser.add_argument(
        "--prune-isolated",
        action="store_true",
        help="when materializing, retain only nodes incident to a selected edge",
    )
    parser.add_argument(
        "--materialize-only",
        action="store_true",
        help="export the requested configuration without computing the sweep table",
    )
    args = parser.parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    bonuses = [float(value) for value in args.agreement_bonuses.split(",")]
    modes = (True, False)
    totals: dict[str, dict[tuple[float, bool, float], list[tuple[float, int]]]] = {}

    candidate_paths = sorted(
        path for root in args.candidate_dir for path in root.glob("*.csv")
    )
    if not candidate_paths:
        parser.error("no candidate CSVs found")
    for candidate_path in candidate_paths:
        name = candidate_path.stem
        embryo = name.split("_", 1)[0]
        node_path = find_named(args.node_dir, name, ".geff")
        gt_path = args.gt_dir / f"{name}.geff"
        candidates = read_candidates(candidate_path)
        organizer_edges: set[tuple[int, int]] = set()
        if args.organizer_dir:
            from biotrack.metric import load_graph

            organizer_path = find_named(args.organizer_dir, name, ".geff")
            organizer_edges = {tuple(map(int, edge)) for edge in load_graph(str(organizer_path)).edge_list()}
        if args.materialize_dir is not None:
            if args.materialize_threshold is None:
                parser.error("--materialize-threshold is required with --materialize-dir")
            from scripts.trackastra_zero_shot.adapter import (
                export_frozen_edge_selection,
                load_frozen_detections,
            )

            materialize_candidates = agreement_bonus(
                candidates, organizer_edges, args.materialize_agreement_bonus
            )
            chosen = greedy_edges(
                materialize_candidates,
                threshold=args.materialize_threshold,
                allow_divisions=args.materialize_mode == "div",
            )
            score_by_edge = {
                (source, target): score for source, target, score in materialize_candidates
            }
            export_frozen_edge_selection(
                load_frozen_detections(node_path),
                chosen,
                score_by_edge,
                args.materialize_dir / f"{name}.geff",
                prune_isolated=args.prune_isolated,
            )
            if args.materialize_only:
                continue
        nodes = load_sample(node_path)
        gt = load_sample(gt_path)
        matched = match_nodes(nodes, gt)
        gt_edges = {(int(source), int(target)) for source, target in gt.edges}
        gt_out = {source for source, _ in gt_edges}
        gt_in = {target for _, target in gt_edges}
        n_est = estimated_nodes(gt_path)
        count_multiplier = max(0.0, 1.0 - ALPHA * (len(nodes.node_ids) - n_est) / n_est)
        fold = totals.setdefault(
            embryo,
            {
                (threshold, mode, bonus): []
                for threshold in thresholds
                for mode in modes
                for bonus in bonuses
            },
        )
        for bonus in bonuses:
            fused_candidates = agreement_bonus(candidates, organizer_edges, bonus)
            for threshold in thresholds:
                for allow_divisions in modes:
                    tp = valid = 0
                    for source, target in greedy_edges(
                        fused_candidates, threshold=threshold, allow_divisions=allow_divisions
                    ):
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
                    fold[(threshold, allow_divisions, bonus)].append((adjusted, denominator))

    for embryo, rows_by_config in sorted(totals.items()):
        print(f"\n{embryo}")
        best = (-1.0, None)
        for bonus in bonuses:
            for allow_divisions in modes:
                mode = "div" if allow_divisions else "nodiv"
                for threshold in thresholds:
                    rows = rows_by_config[(threshold, allow_divisions, bonus)]
                    weight = sum(denominator for _, denominator in rows)
                    score = sum(value * denominator for value, denominator in rows) / weight
                    print(
                        f"  bonus={bonus:>4.1f} mode={mode:5s} "
                        f"threshold={threshold:.2f} edgeJ={score:.6f}"
                    )
                    best = max(best, (score, (bonus, mode, threshold)))
        print(
            f"  BEST bonus={best[1][0]:.1f} mode={best[1][1]} "
            f"threshold={best[1][2]:.2f} edgeJ={best[0]:.6f}"
        )


if __name__ == "__main__":
    main()
