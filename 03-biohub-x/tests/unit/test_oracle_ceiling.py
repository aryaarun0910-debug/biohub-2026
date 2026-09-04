"""The ceiling graph must charge every proposal and emit only matched ground truth.

Two things this protects. The oracle may only rewrite a ground-truth edge onto
proposals when both endpoints matched one-to-one, otherwise it would be inventing
an edge and the ceiling would lie upward. And every proposal must appear as a
node whether or not it matched, otherwise the node-count term would be evaded
and the ceiling would lie upward again.
"""

from __future__ import annotations

from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE, VoxelCoordinateZYX
from biohubx.contracts.instances import CandidateInstance, InstanceSet, ProposalSource
from biohubx.contracts.lineage import DatasetIdentity, EdgeKind, LineageEdge, LineageGraph, LineageNode
from biohubx.evaluation.oracle import match_one_to_one, oracle_graph

DATASET = DatasetIdentity(value="oracle-fixture")


def node(node_id: int, frame: int, x: float) -> LineageNode:
    return LineageNode(
        dataset=DATASET, node_id=node_id, frame=frame, voxel=VoxelCoordinateZYX(z=0.0, y=0.0, x=x)
    )


def edge(source: int, target: int) -> LineageEdge:
    return LineageEdge(
        source_dataset=DATASET,
        source=source,
        target_dataset=DATASET,
        target=target,
        kind=EdgeKind.CONTINUATION,
    )


def proposal(instance_id: int, frame: int, x: float) -> CandidateInstance:
    return CandidateInstance.from_voxel(
        dataset=DATASET,
        instance_id=instance_id,
        frame=frame,
        voxel=VoxelCoordinateZYX(z=0.0, y=0.0, x=x),
        source=ProposalSource.DOG_MULTISCALE,
        confidence=1.0,
        scale=OFFICIAL_VOXEL_SCALE,
    )


def test_only_edges_with_both_endpoints_matched_survive_and_every_proposal_is_a_node() -> None:
    """Three annotated nodes in a chain; proposals near the first two only, plus
    one stray proposal. The chain's first edge survives, its second does not,
    and the stray is charged as a node."""
    annotated = LineageGraph(
        dataset=DATASET,
        nodes=(node(1, 0, 0.0), node(2, 1, 2.0), node(3, 2, 4.0)),
        edges=(edge(1, 2), edge(2, 3)),
    )
    # x is in voxels; 0.40625 um each, so 4 voxels is 1.6 um, inside the radius.
    proposals = InstanceSet(
        dataset=DATASET,
        instances=(proposal(10, 0, 4.0), proposal(11, 1, 6.0), proposal(12, 2, 200.0)),
    )

    matches = match_one_to_one(proposals, annotated)
    ceiling = oracle_graph(proposals, annotated)

    assert matches == {1: 10, 2: 11}
    assert ceiling.graph is not None
    assert ceiling.retained_edges == 1
    assert ceiling.annotated_edges == 2
    assert {n.node_id for n in ceiling.graph.nodes} == {10, 11, 12}
    assert [(e.source, e.target) for e in ceiling.graph.edges] == [(10, 11)]


def test_one_proposal_cannot_serve_two_annotated_nodes() -> None:
    """Two annotated cells in one frame, one proposal between them within radius
    of both. One-to-one matching gives it to one cell only."""
    annotated = LineageGraph(dataset=DATASET, nodes=(node(1, 0, 0.0), node(2, 0, 20.0)), edges=())
    proposals = InstanceSet(dataset=DATASET, instances=(proposal(5, 0, 10.0),))

    matches = match_one_to_one(proposals, annotated)

    assert len(matches) == 1


def test_a_window_with_no_retained_edge_reports_itself_instead_of_a_graph() -> None:
    annotated = LineageGraph(dataset=DATASET, nodes=(node(1, 0, 0.0), node(2, 1, 2.0)), edges=(edge(1, 2),))
    proposals = InstanceSet(dataset=DATASET, instances=(proposal(9, 0, 4.0),))

    ceiling = oracle_graph(proposals, annotated)

    assert ceiling.graph is None
    assert ceiling.annotated_edges == 1
    assert ceiling.retained_edges == 0
    assert ceiling.matched_nodes == 1
