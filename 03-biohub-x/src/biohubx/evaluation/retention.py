"""The node-retention oracle: what the node-count adjustment is worth, measured.

The official score multiplies the edge Jaccard by ``1 - 0.1 * (N_pred - N_total)
/ N_total``. A system that emits roughly every cell receives a multiplier near
one; a system that emits roughly the annotated cells receives one near 1.09. The
second is only reachable by dropping predictions, and dropping predictions also
drops correct edges.

This module measures the trade directly. It builds oracle predictions out of a
dataset's own ground truth under two node budgets, scores them through the
pinned official scorer, and reports where the two curves cross. It is an upper
bound, not a strategy: it assumes the retained edges are the correct ones, which
no real filter can guarantee. What it establishes is the retention a real filter
would have to beat for the trade to be worth making at all.

Decoy nodes stand in for the predictions a dense system makes on unannotated
cells. F-0003 and F-0016 established that such predictions are free of edge and
division penalty, and ``verify_decoys_are_free`` re-checks that on real data
before the sweep relies on it.

Consumer: ``biohubx evaluate retention``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from biohubx.contracts.coordinates import VoxelCoordinateZYX
from biohubx.contracts.lineage import LineageGraph, LineageNode
from biohubx.data.competition import DatasetGroundTruth, edges_with_kinds
from biohubx.evaluation.official_metric import (
    EstimatedTotalNodes,
    FoldScore,
    metric_row,
    summarise_fold,
)

DECOY_ORIGIN_UM = 10_000.0
"""Where decoy nodes are placed: far enough that none can match within 7 um."""

DECOY_SPACING_VOXELS = 100.0
"""Spacing between decoys, so they cannot match one another either."""


@dataclass(frozen=True, slots=True)
class NodeBudget:
    """How many nodes a policy emits."""

    name: str

    ESTIMATED_TOTAL = "estimated_total"
    ANNOTATED = "annotated"


def retained_subgraph(truth: DatasetGroundTruth, *, edge_retention: float, seed: int) -> LineageGraph:
    """Keep a deterministic fraction of the annotated edges and their endpoints.

    This is the oracle's optimism: every retained edge is correct. A real filter
    keeps a mixture, so the retention reported here is a ceiling on what a real
    filter achieves at the same node count.
    """
    if not 0.0 < edge_retention <= 1.0:
        raise ValueError(f"edge retention must lie in (0, 1], got {edge_retention}")
    lineage = truth.lineage
    ordered = sorted(lineage.edges, key=lambda item: (item.source, item.target))
    keep_count = max(1, round(edge_retention * len(ordered)))
    order = np.random.default_rng(seed).permutation(len(ordered))
    kept = [ordered[index] for index in sorted(order[:keep_count].tolist())]

    endpoints = {edge.source for edge in kept} | {edge.target for edge in kept}
    nodes = tuple(node for node in lineage.nodes if node.node_id in endpoints)
    pairs = np.array([[edge.source, edge.target] for edge in kept], dtype=np.int64).reshape(-1, 2)
    return LineageGraph(dataset=lineage.dataset, nodes=nodes, edges=edges_with_kinds(lineage.dataset, pairs))


def with_decoys(graph: LineageGraph, *, target_nodes: int, frames: int) -> LineageGraph:
    """Pad a graph with unmatchable isolated nodes up to a declared node count.

    The decoys stand for a dense system's predictions on unannotated cells. They
    take part in no edge and sit far outside any annotated coordinate, so the
    only quantity they can move is the node count.
    """
    needed = target_nodes - len(graph.nodes)
    if needed <= 0:
        return graph
    next_id = max(node.node_id for node in graph.nodes) + 1
    decoys = tuple(
        LineageNode(
            dataset=graph.dataset,
            node_id=next_id + index,
            frame=index % max(1, frames),
            voxel=VoxelCoordinateZYX(z=0.0, y=DECOY_ORIGIN_UM + DECOY_SPACING_VOXELS * index, x=0.0),
        )
        for index in range(needed)
    )
    return LineageGraph(dataset=graph.dataset, nodes=(*graph.nodes, *decoys), edges=graph.edges)


def verify_decoys_are_free(truth: DatasetGroundTruth) -> dict[str, Any]:
    """Re-measure F-0003 and F-0016 on one real dataset before relying on them.

    Returns the two rows so the caller can record that the edge and division
    counts were identical and only the node count moved.
    """
    estimate = EstimatedTotalNodes.from_official_metadata(truth.estimated_total_nodes)
    bare = metric_row(truth.lineage, truth.lineage, estimated_total_nodes=estimate)
    dense = metric_row(
        with_decoys(truth.lineage, target_nodes=truth.estimated_total_nodes, frames=truth.frames),
        truth.lineage,
        estimated_total_nodes=estimate,
    )
    counted = ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn")
    return {
        "dataset": truth.dataset.value,
        "counts_unchanged": all(bare[key] == dense[key] for key in counted),
        "annotated_budget": {k: bare[k] for k in (*counted, "num_pred_nodes", "adj_edge_jaccard")},
        "estimated_total_budget": {k: dense[k] for k in (*counted, "num_pred_nodes", "adj_edge_jaccard")},
    }


def policy_row(
    truth: DatasetGroundTruth,
    *,
    edge_retention: float,
    node_budget: str,
    seed: int,
) -> dict[str, Any]:
    """Score one dataset under one retention policy through the official scorer."""
    prediction = retained_subgraph(truth, edge_retention=edge_retention, seed=seed)
    if node_budget == NodeBudget.ESTIMATED_TOTAL:
        prediction = with_decoys(prediction, target_nodes=truth.estimated_total_nodes, frames=truth.frames)
    elif node_budget != NodeBudget.ANNOTATED:
        raise ValueError(f"unknown node budget {node_budget!r}")
    return metric_row(
        prediction,
        truth.lineage,
        estimated_total_nodes=EstimatedTotalNodes.from_official_metadata(truth.estimated_total_nodes),
    )


@dataclass(frozen=True, slots=True)
class PolicyPoint:
    """One policy evaluated over one embryo fold."""

    policy: str
    node_budget: str
    edge_retention: float
    embryo: str
    fold: FoldScore

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "node_budget": self.node_budget,
            "edge_retention": self.edge_retention,
            "embryo": self.embryo,
            "official": self.fold.to_dict(),
        }


def sweep(
    corpus: list[DatasetGroundTruth],
    *,
    retentions: tuple[float, ...],
    seed: int,
) -> list[PolicyPoint]:
    """Score the dense baseline and the filtered policy over every embryo fold.

    Folds are reported separately and never pooled: F-0013 measured that the two
    embryos are different annotation regimes, so a pooled number would describe
    neither.
    """
    embryos = sorted({truth.embryo for truth in corpus})
    points: list[PolicyPoint] = []
    for embryo in embryos:
        fold_datasets = [truth for truth in corpus if truth.embryo == embryo]
        dense_rows = [
            policy_row(truth, edge_retention=1.0, node_budget=NodeBudget.ESTIMATED_TOTAL, seed=seed)
            for truth in fold_datasets
        ]
        points.append(
            PolicyPoint(
                policy="dense_baseline",
                node_budget=NodeBudget.ESTIMATED_TOTAL,
                edge_retention=1.0,
                embryo=embryo,
                fold=summarise_fold(dense_rows),
            )
        )
        for retention in retentions:
            rows = [
                policy_row(truth, edge_retention=retention, node_budget=NodeBudget.ANNOTATED, seed=seed)
                for truth in fold_datasets
            ]
            points.append(
                PolicyPoint(
                    policy="annotated_count_filtered",
                    node_budget=NodeBudget.ANNOTATED,
                    edge_retention=retention,
                    embryo=embryo,
                    fold=summarise_fold(rows),
                )
            )
    return points


def crossover(points: list[PolicyPoint], embryo: str) -> dict[str, Any]:
    """The retention at which the filtered policy stops beating the dense one.

    Reported on the adjusted edge Jaccard rather than the combined score,
    because the division term is additive and independent of the node budget:
    mixing it in would attribute division behaviour to a retention decision.
    """
    dense = next(p for p in points if p.embryo == embryo and p.policy == "dense_baseline")
    filtered = sorted(
        (p for p in points if p.embryo == embryo and p.policy == "annotated_count_filtered"),
        key=lambda p: p.edge_retention,
    )
    baseline = dense.fold.adjusted_edge_jaccard
    winning = [p for p in filtered if p.fold.adjusted_edge_jaccard >= baseline]
    breakeven = min((p.edge_retention for p in winning), default=None)
    headroom = [p for p in filtered if p.fold.adjusted_edge_jaccard >= baseline + 0.014]
    return {
        "embryo": embryo,
        "dense_adjusted_edge_jaccard": baseline,
        "lowest_retention_that_matches_dense": breakeven,
        "lowest_retention_worth_0_014": min((p.edge_retention for p in headroom), default=None),
        "filtered_adjusted_edge_jaccard": {p.edge_retention: p.fold.adjusted_edge_jaccard for p in filtered},
    }
