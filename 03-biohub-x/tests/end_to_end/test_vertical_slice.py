"""The Phase-2 vertical slice, end to end.

volume -> instances -> representation -> candidate graph -> scored options
-> legal lineage graph -> official score

The point of these tests is causal ownership, not accuracy. They check that
every stage is reached through the one in front of it, that the emitted graph is
legal by construction, that the numbers the slice reports are attributable to a
stage, and that the parts which could pass vacuously are shown to fire.

Nothing measured here is evidence about the competition. The fixture is
synthetic and the matcher is untrained.
"""

from __future__ import annotations

import pytest

from biohubx.contracts.lineage import EdgeKind
from biohubx.evaluation.official_metric import NodeCountProvenance
from biohubx.tracking.pipeline import SliceConfig, SliceResult, run_slice


@pytest.fixture(scope="module")
def clean() -> SliceResult:
    return run_slice()


@pytest.fixture(scope="module")
def noisy() -> SliceResult:
    """Enough noise that the detector invents cells and abstention has work to do."""
    return run_slice(SliceConfig(noise=0.10))


# --- the chain completes and owns its output -------------------------------


def test_the_slice_runs_end_to_end(clean: SliceResult) -> None:
    assert clean.instances.instances
    assert clean.representation.features
    assert clean.candidates.edges
    assert clean.prediction.targets
    assert clean.emitted.nodes
    assert clean.score.score == pytest.approx(1.1)


def test_the_emitted_graph_is_legal_by_construction(clean: SliceResult) -> None:
    # LineageGraph validates on construction, so reaching this point already
    # proves in-degree, out-degree, one-frame edges and endpoint presence.
    graph = clean.emitted
    in_degree: dict[int, int] = {}
    out_degree: dict[int, int] = {}
    node_ids = {node.node_id for node in graph.nodes}
    frames = {node.node_id: node.frame for node in graph.nodes}
    for edge in graph.edges:
        assert edge.source in node_ids and edge.target in node_ids
        assert frames[edge.target] == frames[edge.source] + 1
        in_degree[edge.target] = in_degree.get(edge.target, 0) + 1
        out_degree[edge.source] = out_degree.get(edge.source, 0) + 1
    assert max(in_degree.values()) <= 1
    assert max(out_degree.values()) <= 2


def test_every_offered_target_receives_a_prediction(clean: SliceResult) -> None:
    assert clean.prediction.covers(frozenset(clean.candidates.targets))


def test_the_representation_covers_every_candidate_instance(clean: SliceResult) -> None:
    referenced = {edge.source for edge in clean.candidates.edges}
    referenced |= {edge.target for edge in clean.candidates.edges}
    assert clean.representation.covers(frozenset(referenced))


def test_every_detected_instance_becomes_a_node(clean: SliceResult) -> None:
    # Dropping isolated detections would change the node count, which the metric
    # charges for. That is a calibration decision, not the decoder's to make.
    assert {node.node_id for node in clean.emitted.nodes} == {
        item.instance_id for item in clean.instances.instances
    }


def test_the_division_is_recovered_and_labelled(clean: SliceResult) -> None:
    forks = [
        node.node_id
        for node in clean.emitted.nodes
        if sum(1 for edge in clean.emitted.edges if edge.source == node.node_id) == 2
    ]
    assert len(forks) == 1
    assert all(edge.kind is EdgeKind.DIVISION for edge in clean.emitted.edges if edge.source == forks[0])
    assert clean.score.division_tp == 1


def test_the_slice_is_deterministic() -> None:
    first, second = run_slice(), run_slice()
    assert first.emitted.integer_export() == second.emitted.integer_export()
    assert first.report() == second.report()


def test_the_report_measures_reach_rather_than_assuming_it(clean: SliceResult) -> None:
    reach = clean.candidates.reach
    assert reach is not None
    assert reach.reach == pytest.approx(1.0)
    assert clean.report()["candidates"]["reach"]["fraction"] == pytest.approx(1.0)


# --- the parts that could pass vacuously are shown to fire ------------------


def test_a_tighter_candidate_radius_lowers_reach_and_caps_the_score() -> None:
    """Reach is a real ceiling, not a number that is always one.

    True displacements in this fixture are one to two micrometres, so a radius
    below that starves the matcher. The edge Jaccard cannot exceed reach, which
    is the whole reason reach is reported separately.
    """
    starved = run_slice(SliceConfig(candidate_radius_um=1.0))
    reach = starved.candidates.reach
    assert reach is not None
    assert reach.reach < 1.0
    assert reach.unreachable_outside_radius > 0
    assert reach.unreachable_missing_endpoint == 0
    assert starved.score.edge_jaccard <= reach.reach + 1e-9


def test_a_harsher_detection_threshold_shows_up_as_a_missing_endpoint() -> None:
    """The other reach failure, attributed to the detector rather than the radius."""
    strict = run_slice(SliceConfig(detection_threshold=0.99))
    reach = strict.candidates.reach
    assert reach is not None
    assert reach.unreachable_missing_endpoint > 0
    assert reach.unreachable_outside_radius == 0


def test_abstention_actually_fires_under_noise(noisy: SliceResult) -> None:
    # In the clean fixture no target ever prefers having no parent, which would
    # make the no-parent contract untested in practice.
    assert noisy.decode_report.abstentions > 0
    assert noisy.decode_report.births > noisy.decode_report.abstentions - 1


def test_spurious_detections_cost_nothing_in_edge_terms_but_collapse_the_score() -> None:
    """The metric's node-count lever, visible end to end in a running system.

    Two noise levels that lose the same single true edge, so the raw edge
    Jaccard is identical in both. Between them the detector invents an order of
    magnitude more cells, none of which matches an annotated node and none of
    which costs an edge false positive. The adjusted Jaccard nevertheless falls
    to zero, entirely through the node-count term.

    This is the behaviour measured in the scorer characterisation tests, now
    arriving through a whole pipeline instead of a hand-built graph, which is
    what makes it a statement about a system rather than about an equation.
    """
    mild = run_slice(SliceConfig(noise=0.06))
    severe = run_slice(SliceConfig(noise=0.15))

    assert len(severe.instances.instances) > len(mild.instances.instances) * 10
    assert severe.score.edge_jaccard == pytest.approx(mild.score.edge_jaccard)
    assert severe.score.edge_fp == mild.score.edge_fp
    assert severe.score.total_node_ratio > mild.score.total_node_ratio
    assert severe.score.adjusted_edge_jaccard < mild.score.adjusted_edge_jaccard
    assert severe.score.adjusted_edge_jaccard == pytest.approx(0.0)


def test_noise_costs_a_true_edge_before_it_costs_the_node_count(clean: SliceResult) -> None:
    # Separating the two failure modes: the first thing noise breaks is a real
    # link, and only afterwards does the flood of spurious nodes take over.
    mild = run_slice(SliceConfig(noise=0.06))
    assert mild.score.edge_jaccard < clean.score.edge_jaccard
    assert mild.score.edge_fn > clean.score.edge_fn


def test_the_matcher_reads_every_feature_channel_it_claims_to_use() -> None:
    """No dead channel: changing a weight that scales a channel must move a score.

    A channel that is wired up but never affects the output is worse than an
    absent one, because it looks like evidence in every diagram and report.
    """
    from biohubx.tracking.matcher import MatcherWeights

    baseline = run_slice()
    no_distance = run_slice(SliceConfig(weights=MatcherWeights(displacement_um_penalty=0.0)))
    no_intensity = run_slice(SliceConfig(weights=MatcherWeights(intensity_change_penalty=0.0)))
    never_born = run_slice(SliceConfig(weights=MatcherWeights(birth_base=-100.0)))

    # Compared across every offered pair, not just the first target: an earlier
    # version of this test looked at one target only, and reported the intensity
    # channel as dead when it was merely inactive on that one pair.
    baseline_scores = _all_parent_scores(baseline)
    assert baseline_scores != _all_parent_scores(no_distance)
    assert baseline_scores != _all_parent_scores(no_intensity)
    # The abstention channel is live in the direction that still produces a
    # scorable graph. The opposite extreme is exercised by the test below.
    assert never_born.decode_report.abstentions == 0


def test_abstaining_everywhere_is_refused_rather_than_scored_as_zero() -> None:
    """A system that abstains on every target emits a graph with no edges.

    That is a real possible output, and the official scorer cannot report node
    recall for it: `evaluate` returns early without matching an edgeless graph,
    and the `node_recall` helper it documents as requiring a matched graph then
    raises from inside the dependency. Biohub-X refuses first and says why,
    rather than passing off a key error as a result or inventing a zero.
    """
    from biohubx.evaluation.official_metric import UnscorablePredictionError
    from biohubx.tracking.matcher import MatcherWeights

    with pytest.raises(UnscorablePredictionError, match="no edges"):
        run_slice(SliceConfig(weights=MatcherWeights(birth_base=100.0)))


def _all_parent_scores(result: SliceResult) -> list[float]:
    return [parent.score for target in result.prediction.targets for parent in target.parents]


def test_the_fixture_gives_cells_different_brightness() -> None:
    # A fixture where every cell looks identical cannot test any appearance
    # reasoning, and would make the intensity channel permanently inert.
    result = run_slice()
    intensities = {round(item.peak_intensity, 3) for item in result.representation.features}
    assert len(intensities) > 1


# --- the sparse regime -----------------------------------------------------


def test_sparse_annotation_uses_a_declared_estimate_not_the_annotated_count() -> None:
    """The regime the competition actually has, running through the whole slice.

    Only some cells are annotated, but the node-count estimate refers to every
    cell. The system predicts all of them, so it is not penalised for detecting
    cells nobody labelled, and the edges among them are free.
    """
    sparse = run_slice(SliceConfig(annotated_fraction=0.4))
    assert len(sparse.truth.annotated.nodes) < len(sparse.truth.lineage.nodes)
    assert sparse.score.node_count_provenance == NodeCountProvenance.DECLARED.value
    assert sparse.score.annotated_gt_nodes == len(sparse.truth.annotated.nodes)
    assert sparse.score.estimated_total_nodes == float(len(sparse.truth.lineage.nodes))
    assert sparse.score.edge_jaccard == pytest.approx(1.0)


def test_a_complete_fixture_declares_its_provenance_honestly(clean: SliceResult) -> None:
    assert clean.score.node_count_provenance == NodeCountProvenance.COMPLETE_SYNTHETIC_GROUND_TRUTH.value


def test_the_report_never_restates_the_official_score(clean: SliceResult) -> None:
    # Scores live in one place in the report so a reader cannot find two copies
    # that disagree after an edit.
    report = clean.report()
    assert "score" in report["official"]
    assert "score" not in report["decode"]
    assert "score" not in report["candidates"]
