"""Legal-lineage contract.

The decoder must own the graph it emits. These are the structural failures the
mission names, each as its own mutation: a graph that violates one of them is
refused at construction, before anything downstream can score it or export it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from biohubx.contracts.coordinates import VoxelCoordinateZYX
from biohubx.contracts.lineage import (
    DatasetIdentity,
    EdgeKind,
    LineageEdge,
    LineageGraph,
    LineageNode,
)

DATASET = DatasetIdentity(value="contract-fixture")
OTHER = DatasetIdentity(value="other-dataset")


def node(node_id: int, frame: int, *, dataset: DatasetIdentity = DATASET, y: float = 0.0) -> LineageNode:
    return LineageNode(
        dataset=dataset,
        node_id=node_id,
        frame=frame,
        voxel=VoxelCoordinateZYX(z=0.0, y=y, x=0.0),
    )


def edge(
    source: int,
    target: int,
    kind: EdgeKind = EdgeKind.CONTINUATION,
    *,
    source_dataset: DatasetIdentity = DATASET,
    target_dataset: DatasetIdentity = DATASET,
) -> LineageEdge:
    return LineageEdge(
        source_dataset=source_dataset,
        source=source,
        target_dataset=target_dataset,
        target=target,
        kind=kind,
    )


# --- the legal shapes ------------------------------------------------------


def test_a_linear_track_is_legal() -> None:
    graph = LineageGraph(
        dataset=DATASET,
        nodes=(node(0, 0), node(1, 1), node(2, 2)),
        edges=(edge(0, 1), edge(1, 2)),
    )
    assert len(graph.edges) == 2


def test_a_division_is_legal() -> None:
    graph = LineageGraph(
        dataset=DATASET,
        nodes=(node(0, 0), node(1, 1, y=5.0), node(2, 1, y=-5.0)),
        edges=(edge(0, 1, EdgeKind.DIVISION), edge(0, 2, EdgeKind.DIVISION)),
    )
    assert sum(1 for e in graph.edges if e.kind is EdgeKind.DIVISION) == 2


def test_births_and_deaths_are_legal() -> None:
    # A node with no parent is a birth; a node with no child is a death. Neither
    # needs an explicit marker, but neither may be silently dropped either.
    graph = LineageGraph(dataset=DATASET, nodes=(node(0, 0), node(1, 1), node(2, 0)), edges=(edge(0, 1),))
    assert len(graph.nodes) == 3


def test_a_graph_must_contain_at_least_one_node() -> None:
    with pytest.raises(ValidationError):
        LineageGraph(dataset=DATASET, nodes=(), edges=())


# --- one mutation per required refusal -------------------------------------


def test_duplicate_node_identity_is_refused() -> None:
    with pytest.raises(ValidationError, match="duplicate node identities"):
        LineageGraph(dataset=DATASET, nodes=(node(0, 0), node(0, 1)), edges=())


def test_a_dangling_endpoint_is_refused() -> None:
    with pytest.raises(ValidationError, match="edge endpoint is missing"):
        LineageGraph(dataset=DATASET, nodes=(node(0, 0),), edges=(edge(0, 99),))


def test_a_duplicate_candidate_pair_is_refused() -> None:
    with pytest.raises(ValidationError, match="duplicate edge"):
        LineageGraph(dataset=DATASET, nodes=(node(0, 0), node(1, 1)), edges=(edge(0, 1), edge(0, 1)))


def test_in_degree_above_one_is_refused() -> None:
    with pytest.raises(ValidationError, match="in-degree"):
        LineageGraph(
            dataset=DATASET,
            nodes=(node(0, 0), node(1, 0, y=10.0), node(2, 1)),
            edges=(edge(0, 2), edge(1, 2)),
        )


def test_out_degree_above_two_is_refused() -> None:
    with pytest.raises(ValidationError, match="out-degree"):
        LineageGraph(
            dataset=DATASET,
            nodes=(node(0, 0), node(1, 1), node(2, 1, y=10.0), node(3, 1, y=20.0)),
            edges=(
                edge(0, 1, EdgeKind.DIVISION),
                edge(0, 2, EdgeKind.DIVISION),
                edge(0, 3, EdgeKind.DIVISION),
            ),
        )


def test_a_cross_dataset_edge_is_refused() -> None:
    with pytest.raises(ValidationError, match="cross-dataset"):
        LineageGraph(
            dataset=DATASET,
            nodes=(node(0, 0), node(1, 1)),
            edges=(edge(0, 1, target_dataset=OTHER),),
        )


def test_a_node_from_another_dataset_is_refused() -> None:
    with pytest.raises(ValidationError, match="crosses dataset identity"):
        LineageGraph(dataset=DATASET, nodes=(node(0, 0), node(1, 1, dataset=OTHER)), edges=())


def test_a_frame_skipping_edge_is_refused() -> None:
    # The official scorer drops multi-frame edges silently. Biohub-X refuses
    # them at construction so they can never reach the scorer in the first place.
    with pytest.raises(ValidationError, match="advance exactly one frame"):
        LineageGraph(dataset=DATASET, nodes=(node(0, 0), node(1, 2)), edges=(edge(0, 1),))


def test_a_backward_time_edge_is_refused() -> None:
    with pytest.raises(ValidationError, match="advance exactly one frame"):
        LineageGraph(dataset=DATASET, nodes=(node(0, 0), node(1, 1)), edges=(edge(1, 0),))


def test_a_same_frame_edge_is_refused() -> None:
    with pytest.raises(ValidationError, match="advance exactly one frame"):
        LineageGraph(dataset=DATASET, nodes=(node(0, 0), node(1, 0, y=5.0)), edges=(edge(0, 1),))


@pytest.mark.parametrize("bad_id", [0.5, "1", True, -1])
def test_a_non_integer_or_negative_identity_is_refused(bad_id: object) -> None:
    with pytest.raises(ValidationError):
        LineageNode.model_validate(
            {
                "dataset": DATASET,
                "node_id": bad_id,
                "frame": 0,
                "voxel": {"z": 0.0, "y": 0.0, "x": 0.0},
            }
        )


@pytest.mark.parametrize("bad_frame", [0.5, "0", -1])
def test_a_non_integer_or_negative_frame_is_refused(bad_frame: object) -> None:
    with pytest.raises(ValidationError):
        LineageNode.model_validate(
            {
                "dataset": DATASET,
                "node_id": 0,
                "frame": bad_frame,
                "voxel": {"z": 0.0, "y": 0.0, "x": 0.0},
            }
        )


# --- edge kind must agree with topology ------------------------------------


def test_a_lone_edge_labelled_division_is_refused() -> None:
    # A fork claimed in the label but absent from the topology is exactly the
    # kind of disagreement the official division metric punishes.
    with pytest.raises(ValidationError, match="edge kind disagrees"):
        LineageGraph(
            dataset=DATASET,
            nodes=(node(0, 0), node(1, 1)),
            edges=(edge(0, 1, EdgeKind.DIVISION),),
        )


def test_a_fork_labelled_continuation_is_refused() -> None:
    with pytest.raises(ValidationError, match="edge kind disagrees"):
        LineageGraph(
            dataset=DATASET,
            nodes=(node(0, 0), node(1, 1), node(2, 1, y=5.0)),
            edges=(edge(0, 1), edge(0, 2)),
        )


def test_daughter_order_does_not_change_the_graph() -> None:
    # Division is a set of two daughters, not an ordered pair. The export must
    # be identical whichever order the decoder happened to emit them in.
    forward = LineageGraph(
        dataset=DATASET,
        nodes=(node(0, 0), node(1, 1, y=5.0), node(2, 1, y=-5.0)),
        edges=(edge(0, 1, EdgeKind.DIVISION), edge(0, 2, EdgeKind.DIVISION)),
    )
    swapped = LineageGraph(
        dataset=DATASET,
        nodes=(node(0, 0), node(2, 1, y=-5.0), node(1, 1, y=5.0)),
        edges=(edge(0, 2, EdgeKind.DIVISION), edge(0, 1, EdgeKind.DIVISION)),
    )
    assert forward.integer_export() == swapped.integer_export()


# --- export ----------------------------------------------------------------


def test_integer_export_keeps_identities_and_frames_integral() -> None:
    graph = LineageGraph(
        dataset=DATASET,
        nodes=(node(2, 2), node(0, 0), node(1, 1)),
        edges=(edge(1, 2), edge(0, 1)),
    )
    exported = graph.integer_export()
    assert [record["node_id"] for record in exported["nodes"]] == [0, 1, 2]
    assert [record["source"] for record in exported["edges"]] == [0, 1]
    for record in exported["nodes"]:
        assert isinstance(record["node_id"], int) and not isinstance(record["node_id"], bool)
        assert isinstance(record["frame"], int)


def test_export_is_deterministic_regardless_of_input_order() -> None:
    nodes = (node(0, 0), node(1, 1), node(2, 2))
    edges = (edge(0, 1), edge(1, 2))
    first = LineageGraph(dataset=DATASET, nodes=nodes, edges=edges).integer_export()
    second = LineageGraph(
        dataset=DATASET, nodes=tuple(reversed(nodes)), edges=tuple(reversed(edges))
    ).integer_export()
    assert first == second


def test_unknown_fields_are_refused() -> None:
    with pytest.raises(ValidationError):
        LineageGraph.model_validate(
            {"dataset": DATASET, "nodes": [node(0, 0)], "edges": [], "confidence": 0.9}
        )
