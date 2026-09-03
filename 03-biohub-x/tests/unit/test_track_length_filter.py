"""The filter must count what it removes correctly, or F-0023 measures another policy.

Small on purpose. AGENTS.md section 3 asks for a test that protects a reusable
invariant, not one per research source. What is protected here is the component
split and the division exemption: get either wrong and the retention number
describes a different filter from the one three public notebooks run.
"""

from __future__ import annotations

from biohubx.contracts.coordinates import VoxelCoordinateZYX
from biohubx.contracts.lineage import (
    DatasetIdentity,
    EdgeKind,
    LineageEdge,
    LineageGraph,
    LineageNode,
)
from biohubx.evaluation.track_length import components, filter_cost

DATASET = DatasetIdentity(value="track-length-fixture")


def node(node_id: int, frame: int, x: float = 0.0) -> LineageNode:
    return LineageNode(
        dataset=DATASET,
        node_id=node_id,
        frame=frame,
        voxel=VoxelCoordinateZYX(z=0.0, y=0.0, x=x),
    )


def edge(source: int, target: int, kind: EdgeKind = EdgeKind.CONTINUATION) -> LineageEdge:
    return LineageEdge(
        source_dataset=DATASET, source=source, target_dataset=DATASET, target=target, kind=kind
    )


def chain(first_id: int, frames: int) -> tuple[list[LineageNode], list[LineageEdge]]:
    nodes = [node(first_id + i, i) for i in range(frames)]
    edges = [edge(first_id + i, first_id + i + 1) for i in range(frames - 1)]
    return nodes, edges


def test_a_short_component_is_removed_and_a_long_one_is_kept() -> None:
    long_nodes, long_edges = chain(1, 6)
    short_nodes, short_edges = chain(100, 2)
    graph = LineageGraph(
        dataset=DATASET,
        nodes=tuple(long_nodes + short_nodes),
        edges=tuple(long_edges + short_edges),
    )

    assert len(components(graph)) == 2
    cost = filter_cost(graph, minimum_frames=4, keep_divisions=True)

    assert (cost.nodes_before, cost.nodes_after) == (8, 6)
    assert (cost.edges_before, cost.edges_after) == (6, 5)


def test_an_isolated_node_is_pruned_and_costs_no_edge() -> None:
    """Isolated-node pruning is the half of the policy that moves node count
    without touching edges, which is exactly the trade F-0018 measured."""
    long_nodes, long_edges = chain(1, 6)
    graph = LineageGraph(dataset=DATASET, nodes=tuple([*long_nodes, node(500, 0)]), edges=tuple(long_edges))

    cost = filter_cost(graph, minimum_frames=2, keep_divisions=True)

    assert cost.nodes_after == 6
    assert cost.edge_retention == 1.0


def test_the_division_exemption_keeps_a_short_dividing_component() -> None:
    """Divisions are scarce (F-0014) and carry their own metric term, so the
    exemption declines to trade one for node budget. Without it the same component
    is pruned, which is the whole difference the flag makes."""
    graph = LineageGraph(
        dataset=DATASET,
        nodes=(node(1, 0), node(2, 1, x=1.0), node(3, 1, x=-1.0)),
        edges=(edge(1, 2, EdgeKind.DIVISION), edge(1, 3, EdgeKind.DIVISION)),
    )

    assert components(graph)[0].has_division is True
    assert filter_cost(graph, minimum_frames=5, keep_divisions=True).nodes_after == 3
    assert filter_cost(graph, minimum_frames=5, keep_divisions=False).nodes_after == 0


def test_span_is_counted_in_frames_not_nodes() -> None:
    """A dividing component holds three nodes across two frames. Counting nodes
    would call it longer than it is and keep tracks this policy prunes."""
    graph = LineageGraph(
        dataset=DATASET,
        nodes=(node(1, 0), node(2, 1, x=1.0), node(3, 1, x=-1.0)),
        edges=(edge(1, 2, EdgeKind.DIVISION), edge(1, 3, EdgeKind.DIVISION)),
    )

    summary = components(graph)[0]

    assert len(summary.node_ids) == 3
    assert summary.frame_span == 2
