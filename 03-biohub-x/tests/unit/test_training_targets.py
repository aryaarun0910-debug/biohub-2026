"""The positive-unlabelled objective, the band audit that rejected the band, and peaks.

The load-bearing claim is that no voxel is ever supervised as background. These
check it directly, check the audit that forced it, and check that a heatmap
becomes proposals on the right grid.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the model-cpu dependency group is not installed")

from biohubx.contracts.coordinates import VoxelCoordinateZYX  # noqa: E402
from biohubx.contracts.lineage import (  # noqa: E402
    DatasetIdentity,
    EdgeKind,
    LineageEdge,
    LineageGraph,
    LineageNode,
)
from biohubx.proposals.peaks import HeatmapProposalError, instances_from_heatmap  # noqa: E402
from biohubx.training.targets import (  # noqa: E402
    CANDIDATE_BANDS,
    PriorError,
    TargetConstructionError,
    class_prior_from_estimate,
    positive_mask,
    positive_unlabelled_loss,
    report_bands,
    select_band,
)

DATASET = DatasetIdentity(value="target-fixture")
DOWNSAMPLE = (1, 4, 4)


def graph(coords: list[tuple[int, float, float, float]]) -> LineageGraph:
    nodes = tuple(
        LineageNode(dataset=DATASET, node_id=i, frame=f, voxel=VoxelCoordinateZYX(z=z, y=y, x=x))
        for i, (f, z, y, x) in enumerate(coords)
    )
    edges = tuple(
        LineageEdge(
            source_dataset=DATASET,
            source=i,
            target_dataset=DATASET,
            target=i + 1,
            kind=EdgeKind.CONTINUATION,
        )
        for i in range(len(coords) - 1)
        if coords[i + 1][0] == coords[i][0] + 1
    )
    return LineageGraph(dataset=DATASET, nodes=nodes, edges=edges)


TRACK = [(0, 1.0, 0.0, 0.0), (1, 1.0, 0.0, 0.0)]


# --- no voxel is ever background --------------------------------------------


def test_only_annotated_voxels_are_positive_and_nothing_is_background() -> None:
    mask, placed = positive_mask((2, 4, 4, 4), graph(TRACK), downsample=DOWNSAMPLE)
    assert placed == 2
    assert int(mask.sum()) == 2
    # Everything else is unlabelled. There is no third state to check, which is
    # the point: this objective cannot express "background" at all.
    assert mask.dtype == torch.bool


def test_the_class_prior_comes_from_the_dataset_not_a_constant() -> None:
    prior = class_prior_from_estimate(3783, 100 * 64 * 64 * 64)
    assert prior == pytest.approx(3783 / (100 * 64 * 64 * 64))
    with pytest.raises(PriorError, match="not a probability"):
        class_prior_from_estimate(10, 5)
    with pytest.raises(PriorError, match="grid must have voxels"):
        class_prior_from_estimate(10, 0)


def test_confident_predictions_on_unlabelled_voxels_still_cost_something() -> None:
    """Unlabelled is not ignored. It is a mixture, and the risk knows it."""
    logits = torch.zeros((2, 4, 4, 4))
    mask, _ = positive_mask((2, 4, 4, 4), graph(TRACK), downsample=DOWNSAMPLE)
    baseline, _ = positive_unlabelled_loss(logits, mask, prior=1e-3)

    shouting = logits.clone()
    shouting[~mask] = 10.0
    louder, _ = positive_unlabelled_loss(shouting, mask, prior=1e-3)
    assert float(louder) > float(baseline), (
        "calling every unlabelled voxel a cell must cost more than staying neutral"
    )


def test_finding_the_annotated_cells_lowers_the_risk() -> None:
    logits = torch.zeros((2, 4, 4, 4))
    mask, _ = positive_mask((2, 4, 4, 4), graph(TRACK), downsample=DOWNSAMPLE)
    baseline, _ = positive_unlabelled_loss(logits, mask, prior=1e-3)

    found = logits.clone()
    found[mask] = 6.0
    better, terms = positive_unlabelled_loss(found, mask, prior=1e-3)
    assert float(better) < float(baseline)
    assert terms["risk_positive"] < 0.01


def test_the_negative_risk_clamp_is_reported_when_it_fires() -> None:
    logits = torch.zeros((2, 4, 4, 4))
    mask, _ = positive_mask((2, 4, 4, 4), graph(TRACK), downsample=DOWNSAMPLE)
    confident = logits.clone()
    confident[mask] = 40.0
    _, terms = positive_unlabelled_loss(confident, mask, prior=0.9)
    assert terms["negative_risk"] < 0
    assert terms["negative_risk_clamped"] == 1.0


def test_a_risk_with_no_positive_or_no_unlabelled_voxel_is_refused() -> None:
    logits = torch.zeros((2, 2, 2, 2))
    with pytest.raises(TargetConstructionError, match="needs at least one positive"):
        positive_unlabelled_loss(logits, torch.zeros_like(logits, dtype=torch.bool), prior=1e-3)
    with pytest.raises(TargetConstructionError, match="nothing to contrast against"):
        positive_unlabelled_loss(logits, torch.ones_like(logits, dtype=torch.bool), prior=1e-3)
    with pytest.raises(TargetConstructionError, match="differ"):
        positive_unlabelled_loss(logits, torch.ones((3, 3, 3, 3), dtype=torch.bool), prior=1e-3)


def test_a_node_outside_the_grid_is_refused_rather_than_clamped() -> None:
    outside = graph([(0, 99.0, 99.0, 99.0), (1, 99.0, 99.0, 99.0)])
    with pytest.raises(TargetConstructionError, match="no annotated node landed"):
        positive_mask((2, 4, 4, 4), outside, downsample=DOWNSAMPLE)


# --- the audit that rejected the band ---------------------------------------


def test_a_band_is_contradicted_by_any_annotated_node_below_it() -> None:
    reports = {r.quantile: r for r in report_bands(np.array([0.99, 0.95, 0.40]), CANDIDATE_BANDS)}
    assert reports[0.90].contradicted_nodes == 1
    assert reports[0.90].coverage == pytest.approx(2 / 3)
    assert reports[0.50].contradicted_nodes == 1
    assert reports[0.90].worst_node_percentile == pytest.approx(0.40)


def test_select_band_returns_none_when_no_band_covers_every_node() -> None:
    """The answer that forced the objective change."""
    assert select_band(np.array([0.99, 0.95, 0.05])) is None
    assert select_band(np.array([0.999, 0.995])) == 0.99


def test_select_band_prefers_the_tightest_band_that_still_covers() -> None:
    assert select_band(np.array([0.92, 0.97])) == 0.90


def test_an_empty_population_is_refused() -> None:
    with pytest.raises(TargetConstructionError, match="no annotated node percentiles"):
        report_bands(np.array([]))


# --- heatmap to proposals ---------------------------------------------------


def test_peaks_are_returned_in_full_resolution_coordinates() -> None:
    heatmap = np.zeros((1, 4, 8, 8), dtype=np.float32)
    heatmap[0, 2, 3, 5] = 0.99
    instances = instances_from_heatmap(heatmap, dataset=DATASET, downsample=DOWNSAMPLE, threshold=0.5)
    assert len(instances.instances) == 1
    voxel = instances.instances[0].voxel
    assert (voxel.z, voxel.y, voxel.x) == (2.0, 12.0, 20.0)


def test_adjacent_peaks_are_suppressed_into_one() -> None:
    heatmap = np.zeros((1, 4, 8, 8), dtype=np.float32)
    heatmap[0, 2, 3, 5] = 0.99
    heatmap[0, 2, 3, 6] = 0.98
    instances = instances_from_heatmap(heatmap, dataset=DATASET, downsample=DOWNSAMPLE, threshold=0.5)
    assert len(instances.instances) == 1
    assert instances.instances[0].confidence == pytest.approx(0.99, abs=1e-6)


def test_an_empty_heatmap_is_a_failure_not_an_empty_result() -> None:
    with pytest.raises(HeatmapProposalError, match="proposed nothing"):
        instances_from_heatmap(
            np.zeros((1, 4, 8, 8), dtype=np.float32),
            dataset=DATASET,
            downsample=DOWNSAMPLE,
            threshold=0.5,
        )


def test_a_threshold_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(HeatmapProposalError, match=r"threshold must lie in \(0, 1\)"):
        instances_from_heatmap(
            np.ones((1, 4, 8, 8), dtype=np.float32),
            dataset=DATASET,
            downsample=DOWNSAMPLE,
            threshold=1.0,
        )


# --- the corpus measurement that decided it ---------------------------------


def test_the_recorded_audit_shows_no_band_survives_on_both_embryos() -> None:
    """Instrument for F-0020, read back from the run that produced it.

    Kept as a test so the conclusion cannot quietly drift from the artifact. If a
    later audit does find a band covering every annotated cell, this fails and
    the objective decision gets revisited on purpose rather than by accident.
    """
    report = pathlib.Path(__file__).resolve().parents[2] / "artifacts/mask-audit.json"
    if not report.is_file():
        pytest.skip("run `biohubx evaluate mask-audit` on a machine holding the corpus")
    audit = json.loads(report.read_text(encoding="utf-8"))

    folds = {entry["fold"]: entry for entry in audit["folds"]}
    assert folds["train_6bba"]["selected_band"] is None, (
        "a band now covers every 6bba annotation; the positive-unlabelled decision needs revisiting"
    )
    for embryo, stats in audit["by_embryo"].items():
        q90 = next(b for b in stats["bands"] if b["quantile"] == 0.90)
        assert q90["contradicted_nodes"] > 0, f"q90 no longer contradicts any {embryo} annotation"
