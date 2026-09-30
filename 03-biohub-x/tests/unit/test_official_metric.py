"""Adapter around the pinned official metric.

These tests cover the adapter's own behaviour: that it refuses what it must,
that it passes Biohub-X contracts through to the authoritative implementation
unchanged, and that the node-count denominator is always an explicit,
provenance-tagged quantity. The metric's own arithmetic belongs to the vendored
upstream source and is characterised separately.
"""

from __future__ import annotations

import pytest

from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE, VoxelCoordinateZYX, VoxelScaleZYX
from biohubx.contracts.lineage import (
    DatasetIdentity,
    EdgeKind,
    LineageEdge,
    LineageGraph,
    LineageNode,
)
from biohubx.evaluation.official_metric import (
    CALIBRATION_FIXTURES,
    EstimatedTotalNodes,
    NodeCountProvenance,
    evaluate_calibration_fixtures,
    evaluate_graph,
)

DATASET = DatasetIdentity(value="adapter-fixture")


def node(node_id: int, frame: int, *, z: float = 0.0, y: float = 0.0, x: float = 0.0) -> LineageNode:
    return LineageNode(dataset=DATASET, node_id=node_id, frame=frame, voxel=VoxelCoordinateZYX(z=z, y=y, x=x))


def edge(source: int, target: int, kind: EdgeKind = EdgeKind.CONTINUATION) -> LineageEdge:
    return LineageEdge(
        source_dataset=DATASET, source=source, target_dataset=DATASET, target=target, kind=kind
    )


def linear_graph() -> LineageGraph:
    return LineageGraph(
        dataset=DATASET,
        nodes=(node(0, 0), node(1, 1), node(2, 2)),
        edges=(edge(0, 1), edge(1, 2)),
    )


def complete(graph: LineageGraph) -> EstimatedTotalNodes:
    return EstimatedTotalNodes.from_complete_synthetic_ground_truth(graph)


# --- the node-count denominator is never implicit --------------------------


def test_estimated_total_nodes_is_a_required_argument() -> None:
    # The adapter must not choose this number for the caller. Defaulting it to
    # the annotated count would silently report an adjusted Jaccard that is not
    # the competition's, because annotations are sparse.
    graph = linear_graph()
    with pytest.raises(TypeError):
        evaluate_graph(graph, graph.model_copy(deep=True))  # type: ignore[call-arg]


def test_every_estimate_carries_its_provenance() -> None:
    assert (
        EstimatedTotalNodes.from_official_metadata(1000.0).provenance
        is NodeCountProvenance.OFFICIAL_GEFF_METADATA
    )
    assert EstimatedTotalNodes.declared(50.0).provenance is NodeCountProvenance.DECLARED
    assert complete(linear_graph()).provenance is NodeCountProvenance.COMPLETE_SYNTHETIC_GROUND_TRUTH


def test_a_complete_synthetic_estimate_equals_the_annotated_count() -> None:
    graph = linear_graph()
    assert complete(graph).value == float(len(graph.nodes))


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_a_non_positive_or_non_finite_estimate_is_refused(bad: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        EstimatedTotalNodes.declared(bad)


def test_the_result_reports_the_annotated_count_and_the_estimate_separately() -> None:
    graph = linear_graph()
    result = evaluate_graph(
        graph,
        graph.model_copy(deep=True),
        estimated_total_nodes=EstimatedTotalNodes.from_official_metadata(30.0),
    )
    assert result.annotated_gt_nodes == 3
    assert result.estimated_total_nodes == 30.0
    assert result.node_count_provenance == NodeCountProvenance.OFFICIAL_GEFF_METADATA.value
    # Distinguishable: conflating them would make the ratio 0.0 instead of -0.9.
    assert result.total_node_ratio == pytest.approx((3 - 30.0) / 30.0)


# --- refusals ---------------------------------------------------------------


def test_a_missing_ground_truth_artifact_is_refused() -> None:
    graph = linear_graph()
    with pytest.raises(FileNotFoundError, match="ground-truth graph artifact is required"):
        evaluate_graph(graph, None, estimated_total_nodes=complete(graph))


def test_a_dataset_mismatch_is_refused() -> None:
    # Each graph is internally consistent; they simply describe different movies.
    # Scoring one against the other would silently compare unrelated embryos.
    elsewhere = DatasetIdentity(value="a-different-dataset")
    other = LineageGraph(
        dataset=elsewhere,
        nodes=(
            LineageNode(dataset=elsewhere, node_id=0, frame=0, voxel=VoxelCoordinateZYX(z=0, y=0, x=0)),
            LineageNode(dataset=elsewhere, node_id=1, frame=1, voxel=VoxelCoordinateZYX(z=0, y=0, x=0)),
        ),
        edges=(
            LineageEdge(
                source_dataset=elsewhere,
                source=0,
                target_dataset=elsewhere,
                target=1,
                kind=EdgeKind.CONTINUATION,
            ),
        ),
    )
    graph = linear_graph()
    with pytest.raises(ValueError, match="different datasets"):
        evaluate_graph(graph, other, estimated_total_nodes=complete(graph))


def test_an_isotropic_scale_cannot_reach_the_metric() -> None:
    # There is no code path that constructs one, so the refusal happens at the
    # coordinate contract rather than inside the adapter.
    with pytest.raises(ValueError):
        VoxelScaleZYX(z_um=0.40625, y_um=0.40625, x_um=0.40625)


# --- known-good and known-bad graphs ---------------------------------------


def test_a_perfect_prediction_scores_one() -> None:
    graph = linear_graph()
    result = evaluate_graph(graph, graph.model_copy(deep=True), estimated_total_nodes=complete(graph))
    assert (result.edge_tp, result.edge_fp, result.edge_fn) == (2, 0, 0)
    assert result.edge_jaccard == pytest.approx(1.0)
    assert result.adjusted_edge_jaccard == pytest.approx(1.0)
    assert result.node_recall == pytest.approx(1.0)
    assert result.score == pytest.approx(1.0)
    assert result.division_jaccard is None


def test_a_missing_edge_lowers_the_jaccard_by_exactly_one_false_negative() -> None:
    gt = linear_graph()
    pred = LineageGraph(dataset=DATASET, nodes=gt.nodes, edges=gt.edges[:1])
    result = evaluate_graph(pred, gt, estimated_total_nodes=complete(gt))
    assert (result.edge_tp, result.edge_fp, result.edge_fn) == (1, 0, 1)
    assert result.edge_jaccard == pytest.approx(0.5)


def test_a_perfect_division_adds_the_weighted_division_term() -> None:
    dataset = DatasetIdentity(value="division-fixture")
    nodes = (
        LineageNode(dataset=dataset, node_id=0, frame=0, voxel=VoxelCoordinateZYX(z=0, y=0, x=0)),
        LineageNode(dataset=dataset, node_id=1, frame=1, voxel=VoxelCoordinateZYX(z=0, y=5, x=0)),
        LineageNode(dataset=dataset, node_id=2, frame=1, voxel=VoxelCoordinateZYX(z=0, y=-5, x=0)),
    )
    edges = (
        LineageEdge(
            source_dataset=dataset, source=0, target_dataset=dataset, target=1, kind=EdgeKind.DIVISION
        ),
        LineageEdge(
            source_dataset=dataset, source=0, target_dataset=dataset, target=2, kind=EdgeKind.DIVISION
        ),
    )
    graph = LineageGraph(dataset=dataset, nodes=nodes, edges=edges)
    result = evaluate_graph(
        graph,
        graph.model_copy(deep=True),
        estimated_total_nodes=EstimatedTotalNodes.from_complete_synthetic_ground_truth(graph),
    )
    assert result.division_tp == 1
    assert result.division_jaccard == pytest.approx(1.0)
    # The division term is weighted 0.1, so a perfect graph with a division
    # scores above 1.0. That is the metric's design, not a defect.
    assert result.score == pytest.approx(1.1)


def test_the_official_scale_is_used_by_default() -> None:
    gt = LineageGraph(dataset=DATASET, nodes=(node(0, 0), node(1, 1)), edges=(edge(0, 1),))
    # Five voxels along z is 8.125 um, beyond the 7 um radius: no match.
    far_in_z = LineageGraph(dataset=DATASET, nodes=(node(0, 0), node(1, 1, z=5.0)), edges=(edge(0, 1),))
    # The same five voxels along x is 2.03 um, inside the radius: matched.
    near_in_x = LineageGraph(dataset=DATASET, nodes=(node(0, 0), node(1, 1, x=5.0)), edges=(edge(0, 1),))
    assert evaluate_graph(far_in_z, gt, estimated_total_nodes=complete(gt)).edge_tp == 0
    assert evaluate_graph(near_in_x, gt, estimated_total_nodes=complete(gt)).edge_tp == 1
    assert OFFICIAL_VOXEL_SCALE.z_um == 4 * OFFICIAL_VOXEL_SCALE.x_um


# --- the fixture suite the CLI exposes -------------------------------------


def test_every_declared_fixture_runs_and_lands_in_one_behavior() -> None:
    fixtures = evaluate_calibration_fixtures()
    assert set(fixtures) == set(CALIBRATION_FIXTURES)
    for name, outcome in fixtures.items():
        assert outcome["behavior"] in {
            "evaluated",
            "evaluated_official_sparse_gt",
            "refused",
        }, f"{name} produced an undeclared behavior"


def test_malformed_graph_fixtures_are_refused_not_scored() -> None:
    fixtures = evaluate_calibration_fixtures()
    must_refuse = {
        "missing_node_endpoint",
        "duplicate_node",
        "illegal_in_degree",
        "illegal_out_degree",
        "invalid_three_daughter_fork",
        "cross_dataset_edge",
        "backward_time_edge",
        "non_integer_identity",
        "missing_gt_artifact",
    }
    for name in must_refuse:
        assert fixtures[name]["behavior"] == "refused", f"{name} was scored instead of refused"


def test_the_fixture_suite_is_deterministic() -> None:
    assert evaluate_calibration_fixtures() == evaluate_calibration_fixtures()
