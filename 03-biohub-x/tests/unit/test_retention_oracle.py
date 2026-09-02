"""The retention oracle's two load-bearing assumptions, checked directly.

The sweep's conclusion rests on decoy nodes being free of edge and division
penalty while still moving the node count, and on the retained subgraph being a
legal lineage at every retention. Both are asserted here rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from biohubx.contracts.coordinates import VoxelCoordinateZYX
from biohubx.contracts.lineage import DatasetIdentity, LineageGraph, LineageNode
from biohubx.data.competition import DatasetGroundTruth, edges_with_kinds
from biohubx.evaluation.retention import (
    NodeBudget,
    policy_row,
    retained_subgraph,
    verify_decoys_are_free,
    with_decoys,
)

DATASET = DatasetIdentity(value="retention-fixture")


def track(length: int) -> LineageGraph:
    nodes = tuple(
        LineageNode(
            dataset=DATASET,
            node_id=index,
            frame=index,
            voxel=VoxelCoordinateZYX(z=0.0, y=float(index), x=0.0),
        )
        for index in range(length)
    )
    pairs = np.array([[index, index + 1] for index in range(length - 1)], dtype=np.int64)
    return LineageGraph(dataset=DATASET, nodes=nodes, edges=edges_with_kinds(DATASET, pairs))


def fixture(length: int = 12, estimate: int = 200) -> DatasetGroundTruth:
    return DatasetGroundTruth(
        dataset=DATASET,
        lineage=track(length),
        estimated_total_nodes=estimate,
        frames=length,
        embryo="fixture",
    )


def test_decoys_move_the_node_count_and_nothing_else() -> None:
    """The whole oracle rests on this: padding to a dense budget is free but not costless."""
    guard = verify_decoys_are_free(fixture())

    assert guard["counts_unchanged"] is True
    annotated = guard["annotated_budget"]
    dense = guard["estimated_total_budget"]
    assert annotated["num_pred_nodes"] == 12
    assert dense["num_pred_nodes"] == 200
    # Same edges, so the raw Jaccard is unchanged; only the multiplier moves,
    # and it moves the way the adjustment says it should.
    assert dense["adj_edge_jaccard"] < annotated["adj_edge_jaccard"]
    assert dense["adj_edge_jaccard"] == pytest.approx(1.0)
    assert annotated["adj_edge_jaccard"] == pytest.approx(1.0 * (1 - 0.1 * ((12 - 200) / 200)))


def test_padding_never_shrinks_a_graph() -> None:
    graph = track(12)
    assert with_decoys(graph, target_nodes=5, frames=12) is graph


def test_every_retention_yields_a_legal_lineage() -> None:
    truth = fixture(length=12)
    for retention in (1.0, 0.9, 0.75, 0.5, 0.25):
        kept = retained_subgraph(truth, edge_retention=retention, seed=0)
        # Constructing a LineageGraph validates it, so reaching here is the assertion.
        assert kept.edges
        endpoints = {edge.source for edge in kept.edges} | {edge.target for edge in kept.edges}
        assert {node.node_id for node in kept.nodes} == endpoints


def test_retention_is_deterministic_for_a_seed() -> None:
    truth = fixture()
    first = retained_subgraph(truth, edge_retention=0.6, seed=7).integer_export()
    second = retained_subgraph(truth, edge_retention=0.6, seed=7).integer_export()
    assert first == second


def test_a_retention_outside_the_unit_interval_is_refused() -> None:
    truth = fixture()
    with pytest.raises(ValueError, match="edge retention must lie"):
        retained_subgraph(truth, edge_retention=0.0, seed=0)
    with pytest.raises(ValueError, match="edge retention must lie"):
        retained_subgraph(truth, edge_retention=1.5, seed=0)


def test_an_unknown_node_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown node budget"):
        policy_row(fixture(), edge_retention=1.0, node_budget="whatever", seed=0)


def test_the_filtered_budget_beats_the_dense_one_at_full_retention() -> None:
    """The trade in one line: same edges, fewer nodes, higher adjusted Jaccard."""
    truth = fixture()
    dense = policy_row(truth, edge_retention=1.0, node_budget=NodeBudget.ESTIMATED_TOTAL, seed=0)
    filtered = policy_row(truth, edge_retention=1.0, node_budget=NodeBudget.ANNOTATED, seed=0)

    assert dense["edge_jaccard"] == filtered["edge_jaccard"]
    assert filtered["adj_edge_jaccard"] > dense["adj_edge_jaccard"]
