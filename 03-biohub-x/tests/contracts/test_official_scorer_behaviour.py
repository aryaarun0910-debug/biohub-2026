"""Measured behaviour of the pinned official scorer.

These are characterisation tests, not assertions about what the metric ought to
do. Each one measures a property of the authoritative implementation that
changes what a system should try to emit, so that the property is recorded as a
Biohub-X measurement rather than believed from an external reading.

Every claim proved here is registered in registry/findings.yaml. If the pinned
official source is ever moved to a different commit, these tests are the
instrument that says whether the finding survived.
"""

from __future__ import annotations

import pytest

from biohubx.contracts.coordinates import VoxelCoordinateZYX
from biohubx.contracts.lineage import (
    DatasetIdentity,
    EdgeKind,
    LineageEdge,
    LineageGraph,
    LineageNode,
)
from biohubx.evaluation.official_metric import (
    EstimatedTotalNodes,
    OfficialMetricResult,
    evaluate_graph,
)

DATASET = DatasetIdentity(value="scorer-behaviour")


def node(node_id: int, frame: int, *, z: float = 0.0, y: float = 0.0, x: float = 0.0) -> LineageNode:
    return LineageNode(dataset=DATASET, node_id=node_id, frame=frame, voxel=VoxelCoordinateZYX(z=z, y=y, x=x))


def edge(source: int, target: int, kind: EdgeKind = EdgeKind.CONTINUATION) -> LineageEdge:
    return LineageEdge(
        source_dataset=DATASET, source=source, target_dataset=DATASET, target=target, kind=kind
    )


def annotated_track() -> LineageGraph:
    """Two annotated cells forming one annotated edge across frames 0 and 1."""
    return LineageGraph(dataset=DATASET, nodes=(node(0, 0), node(1, 1)), edges=(edge(0, 1),))


def score(prediction: LineageGraph, ground_truth: LineageGraph, estimate: float) -> OfficialMetricResult:
    return evaluate_graph(
        prediction,
        ground_truth,
        estimated_total_nodes=EstimatedTotalNodes.declared(estimate),
    )


# --- F1: an unmatched isolated prediction is not an edge false positive ----


def test_an_unmatched_isolated_prediction_adds_no_edge_false_positive() -> None:
    """Predicting a cell the annotations do not cover costs nothing in edge terms.

    The graph gains an isolated node far outside the 7 um matching radius, so it
    matches no ground-truth node and takes part in no edge. Edge TP, FP and FN
    are unchanged and the raw edge Jaccard is unchanged.

    This is only the edge accounting. The same node still enters the node-count
    adjustment, which the companion test below measures, so "free" means free of
    edge penalty, not free of consequence.
    """
    gt = annotated_track()
    baseline = score(gt, gt, estimate=2.0)

    # 100 voxels along y is 40.6 um: far outside the matching radius, and in a
    # third frame so it cannot be an endpoint of any annotated edge.
    with_spectator = LineageGraph(
        dataset=DATASET,
        nodes=(*gt.nodes, node(2, 2, y=100.0)),
        edges=gt.edges,
    )
    observed = score(with_spectator, gt, estimate=2.0)

    assert (observed.edge_tp, observed.edge_fp, observed.edge_fn) == (
        baseline.edge_tp,
        baseline.edge_fp,
        baseline.edge_fn,
    )
    assert observed.edge_jaccard == pytest.approx(baseline.edge_jaccard)
    assert observed.node_recall == pytest.approx(baseline.node_recall)
    # Unchanged in edge terms, but counted as a predicted node.
    assert observed.num_pred_nodes == baseline.num_pred_nodes + 1


def test_the_unmatched_prediction_still_moves_the_node_count_adjustment() -> None:
    """The companion measurement: an edge-free node is not consequence-free.

    Holding the estimate fixed, adding one unmatched node raises num_pred_nodes,
    which raises total_node_ratio, which lowers the adjusted edge Jaccard even
    though the raw Jaccard did not move.
    """
    gt = annotated_track()
    baseline = score(gt, gt, estimate=2.0)
    with_spectator = LineageGraph(dataset=DATASET, nodes=(*gt.nodes, node(2, 2, y=100.0)), edges=gt.edges)
    observed = score(with_spectator, gt, estimate=2.0)

    assert observed.edge_jaccard == pytest.approx(baseline.edge_jaccard)
    assert observed.total_node_ratio > baseline.total_node_ratio
    assert observed.adjusted_edge_jaccard < baseline.adjusted_edge_jaccard


def test_an_unmatched_pair_of_predictions_may_carry_a_free_edge() -> None:
    """An edge between two cells the annotations do not cover is also free.

    Both endpoints lie outside the matching radius of any annotated node, so the
    edge is neither a true positive nor a false positive.
    """
    gt = annotated_track()
    baseline = score(gt, gt, estimate=2.0)
    with_unannotated_track = LineageGraph(
        dataset=DATASET,
        nodes=(*gt.nodes, node(2, 0, y=100.0), node(3, 1, y=100.0)),
        edges=(*gt.edges, edge(2, 3)),
    )
    observed = score(with_unannotated_track, gt, estimate=2.0)

    assert observed.edge_fp == baseline.edge_fp
    assert observed.edge_tp == baseline.edge_tp
    assert observed.edge_jaccard == pytest.approx(baseline.edge_jaccard)


def test_a_wrong_edge_on_an_annotated_cell_is_a_false_positive() -> None:
    """The boundary of the above: touching an annotated cell is not free.

    The predicted source matches an annotated node that has an annotated
    successor, and the edge goes somewhere else, so it is charged.
    """
    gt = annotated_track()
    wrong = LineageGraph(
        dataset=DATASET,
        nodes=(node(0, 0), node(1, 1, y=100.0)),
        edges=(edge(0, 1),),
    )
    observed = score(wrong, gt, estimate=2.0)
    assert observed.edge_fp == 1
    assert observed.edge_tp == 0
    assert observed.edge_fn == 1


# --- F2: under-prediction lifts the adjustment multiplier above one --------


def test_under_prediction_makes_the_adjustment_multiplier_exceed_one() -> None:
    """Predicting fewer nodes than the estimate multiplies the Jaccard upward.

    The multiplier is ``1 - 0.1 * (N_pred - N_total) / N_total``. It exceeds one
    exactly when N_pred < N_total, and the official implementation clamps only
    at zero, so there is no upper bound at one.

    Demonstrating an adjusted Jaccard above one therefore requires an estimated
    total that is larger than the annotated ground-truth count. With a complete
    ground truth the two are equal, the ratio is zero, and the multiplier is
    exactly one, which is why this test declares the estimate explicitly.
    """
    gt = annotated_track()
    assert len(gt.nodes) == 2

    # An estimate ten times the annotated count: the annotations cover a tenth
    # of the cells, which is the sparse regime the competition actually has.
    observed = score(gt, gt, estimate=20.0)

    assert observed.annotated_gt_nodes == 2
    assert observed.estimated_total_nodes == 20.0
    assert observed.num_pred_nodes == 2
    assert observed.total_node_ratio == pytest.approx((2 - 20.0) / 20.0)
    assert observed.edge_jaccard == pytest.approx(1.0)

    expected_multiplier = 1 - 0.1 * observed.total_node_ratio
    assert expected_multiplier > 1.0
    assert observed.adjusted_edge_jaccard == pytest.approx(observed.edge_jaccard * expected_multiplier)
    assert observed.adjusted_edge_jaccard > 1.0


def test_matching_the_estimate_exactly_leaves_the_jaccard_untouched() -> None:
    gt = annotated_track()
    observed = score(gt, gt, estimate=float(len(gt.nodes)))
    assert observed.total_node_ratio == pytest.approx(0.0)
    assert observed.adjusted_edge_jaccard == pytest.approx(observed.edge_jaccard)


def test_over_prediction_lowers_the_adjusted_jaccard() -> None:
    gt = annotated_track()
    observed = score(gt, gt, estimate=1.0)
    assert observed.total_node_ratio > 0
    assert observed.adjusted_edge_jaccard < observed.edge_jaccard


def test_a_complete_ground_truth_cannot_show_the_effect() -> None:
    """Why the estimate must be declared, stated as a measurement.

    Using the annotated count as the estimate pins the ratio at zero whatever
    the sparsity really is, which hides the entire lever.
    """
    gt = annotated_track()
    observed = evaluate_graph(
        gt,
        gt,
        estimated_total_nodes=EstimatedTotalNodes.from_complete_synthetic_ground_truth(gt),
    )
    assert observed.total_node_ratio == pytest.approx(0.0)
    assert observed.adjusted_edge_jaccard == pytest.approx(observed.edge_jaccard)


# --- F3: multi-frame edges, on both sides of the boundary ------------------


def test_the_lineage_contract_rejects_a_multi_frame_edge_before_evaluation() -> None:
    """Biohub-X boundary: an edge spanning two frames never reaches the scorer.

    The graph is refused at construction, so the question of how the official
    scorer would treat it does not arise inside this system.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="advance exactly one frame"):
        LineageGraph(
            dataset=DATASET,
            nodes=(node(0, 0), node(1, 2)),
            edges=(edge(0, 1),),
        )


def test_the_official_scorer_drops_a_multi_frame_edge_rather_than_charging_it() -> None:
    """Official boundary, measured directly on the vendored implementation.

    Constructed below the Biohub-X contract, because no legal Biohub-X graph can
    express this. The measurement matters because a system that emitted such an
    edge would be neither rewarded nor penalised for it: the edge would simply
    vanish, and the annotated edge it failed to reproduce would count as a false
    negative. Silence is the danger, and it is why the contract above refuses.
    """
    import polars as pl
    import tracksdata as td  # type: ignore[import-untyped]

    from biohubx._vendor.official_competition.tracking_cellmot import metrics as authoritative

    def build(edges: tuple[tuple[int, int], ...], frames: tuple[int, ...]) -> object:
        graph = td.graph.InMemoryGraph()
        for axis in ("z", "y", "x"):
            graph.add_node_attr_key(axis, pl.Float64, 0.0)
        ids = [
            graph.add_node(attrs={td.DEFAULT_ATTR_KEYS.T: frame, "z": 0.0, "y": 0.0, "x": 0.0})
            for frame in frames
        ]
        for source, target in edges:
            graph.add_edge(ids[source], ids[target], {})
        return graph

    # Ground truth: three frames, two consecutive annotated edges.
    gt = build(edges=((0, 1), (1, 2)), frames=(0, 1, 2))
    # Prediction: one edge skipping from frame 0 straight to frame 2.
    skipping = build(edges=((0, 2),), frames=(0, 1, 2))

    counts = authoritative.evaluate(skipping, gt, scale=(1.625, 0.40625, 0.40625), max_distance=7.0)

    # Not charged as a false positive: the edge is dropped before counting.
    assert counts.edge_fp == 0
    assert counts.edge_tp == 0
    # Both annotated edges remain unreproduced.
    assert counts.edge_fn == 2


def test_a_backward_edge_is_also_refused_by_the_contract() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="advance exactly one frame"):
        LineageGraph(dataset=DATASET, nodes=(node(0, 0), node(1, 1)), edges=(edge(1, 0),))


# --- F4: the division counter shares the unmatched-prediction exemption ----


def annotated_division() -> LineageGraph:
    """One annotated division: a parent in frame 0 with two daughters in frame 1."""
    return LineageGraph(
        dataset=DATASET,
        nodes=(node(0, 0), node(1, 1, y=1.0), node(2, 1, y=-1.0)),
        edges=(edge(0, 1, EdgeKind.DIVISION), edge(0, 2, EdgeKind.DIVISION)),
    )


def test_an_unmatched_fork_adds_no_division_false_positive() -> None:
    """A fork invented where the annotations do not reach costs no division penalty.

    F1 measured this exemption for edges. The division counter turns out to
    share it: a predicted fork is only chargeable once its own node matches a
    ground-truth node, so a fork built entirely from unmatched nodes is not
    counted at all. Division TP, FP and FN are unchanged and the division
    Jaccard stays at 1.0.

    As with F1, free of division penalty is not free of consequence. The three
    extra nodes still enter the node-count adjustment, which is the only term
    that moves here.
    """
    gt = annotated_division()
    baseline = score(gt, gt, estimate=3.0)
    assert (baseline.division_tp, baseline.division_fp, baseline.division_fn) == (1, 0, 0)

    # 100 voxels along y is 40.6 um, far outside the 7 um matching radius, so
    # neither the fork nor either of its daughters can match a ground-truth node.
    with_spurious_fork = LineageGraph(
        dataset=DATASET,
        nodes=(*gt.nodes, node(10, 0, y=100.0), node(11, 1, y=100.0, x=1.0), node(12, 1, y=100.0, x=-1.0)),
        edges=(*gt.edges, edge(10, 11, EdgeKind.DIVISION), edge(10, 12, EdgeKind.DIVISION)),
    )
    measured = score(with_spurious_fork, gt, estimate=3.0)

    # The invented division is uncharged, and so are its two edges.
    assert (measured.division_tp, measured.division_fp, measured.division_fn) == (1, 0, 0)
    assert measured.division_jaccard == 1.0
    assert (measured.edge_tp, measured.edge_fp, measured.edge_fn) == (2, 0, 0)
    assert measured.edge_jaccard == baseline.edge_jaccard == 1.0

    # The node count is the only thing that moved: 3 predicted nodes became 6.
    assert baseline.num_pred_nodes == 3
    assert measured.num_pred_nodes == 6
    assert measured.adjusted_edge_jaccard < baseline.adjusted_edge_jaccard


def test_a_fork_on_an_annotated_cell_is_a_division_false_positive() -> None:
    """The exemption ends where the annotations begin, exactly as it does for edges.

    Against a ground truth holding no divisions at all, a fork whose own node
    matches an annotated cell is evaluable and is charged. That single false
    positive takes the division Jaccard from undefined to zero, and the spurious
    branch is charged a second time as an edge false positive.
    """
    gt = LineageGraph(
        dataset=DATASET,
        nodes=(node(0, 0), node(1, 1), node(2, 2)),
        edges=(edge(0, 1), edge(1, 2)),
    )
    baseline = score(gt, gt, estimate=3.0)
    # No divisions anywhere means the division term is dropped, not scored zero.
    assert (baseline.division_tp, baseline.division_fp, baseline.division_fn) == (0, 0, 0)
    assert baseline.division_jaccard is None

    # The fork sits on node 0, which matches an annotated cell that has a successor.
    forked = LineageGraph(
        dataset=DATASET,
        nodes=(node(0, 0), node(1, 1), node(2, 2), node(3, 1, y=2.0)),
        edges=(edge(0, 1, EdgeKind.DIVISION), edge(0, 3, EdgeKind.DIVISION), edge(1, 2)),
    )
    measured = score(forked, gt, estimate=3.0)

    assert (measured.division_tp, measured.division_fp, measured.division_fn) == (0, 1, 0)
    assert measured.division_jaccard == 0.0
    assert measured.edge_fp == 1
