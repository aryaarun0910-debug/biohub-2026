"""Exact OOF ablation: cap outgoing children by organizer edge probability.

This mutates graphs in memory only. It is designed to answer whether the learned
baseline's extremely noisy forks should be suppressed before a stronger linker is
available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biotrack.metric import load_graph, score_pred_graph  # noqa: E402
from tracking_cellmot.metrics import summarise  # noqa: E402


def cap_children(graph, max_children: int = 1, score_attr: str = "edge_prob") -> int:
    """Keep the highest-probability outgoing edges for every source node."""

    if max_children < 1:
        raise ValueError("max_children must be at least one")
    attrs = graph.edge_attrs(attr_keys=[score_attr])
    by_source: dict[int, list[tuple[float, int]]] = {}
    for edge_id, source_id, probability in attrs.select(
        ["edge_id", "source_id", score_attr]
    ).iter_rows():
        by_source.setdefault(int(source_id), []).append((float(probability), int(edge_id)))
    remove: list[int] = []
    for edges in by_source.values():
        if len(edges) > max_children:
            edges.sort(reverse=True)
            remove.extend(edge_id for _, edge_id in edges[max_children:])
    if remove:
        graph.bulk_remove_edges(remove)
    return len(remove)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", action="append", required=True)
    parser.add_argument("--gt-dir", default=str(ROOT / "data" / "train"))
    parser.add_argument("--max-children", type=int, default=1)
    parser.add_argument("--score-attr", default="edge_prob")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    rows_by_embryo: dict[str, list[dict[str, float]]] = {}
    removed_by_embryo: dict[str, int] = {}
    for pred_root in map(Path, args.pred_dir):
        for path in sorted(pred_root.glob("*.geff")):
            graph = load_graph(str(path))
            removed = cap_children(graph, args.max_children, args.score_attr)
            embryo = path.stem.split("_", 1)[0]
            rows_by_embryo.setdefault(embryo, []).append(
                score_pred_graph(graph, str(gt_dir / path.name))
            )
            removed_by_embryo[embryo] = removed_by_embryo.get(embryo, 0) + removed

    for embryo, rows in sorted(rows_by_embryo.items()):
        result = summarise(rows)
        print(
            f"{embryo}: crops={len(rows)} removed={removed_by_embryo[embryo]} "
            f"edgeJ={result['adj_edge_jaccard']:.4f} "
            f"divJ={result['division_jaccard']:.4f} "
            f"score={result['score']:.4f}"
        )


if __name__ == "__main__":
    main()
