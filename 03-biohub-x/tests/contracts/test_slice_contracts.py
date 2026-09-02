"""Contracts introduced by the Phase-2 vertical slice.

One mutation per rule, for the four contracts that carry evidence between the
detector and the decoder: instances, representation, candidates and predictions.
The rules they enforce are the ones the mission names as required mutation
tests, and each is written so that removing the rule makes a test fail rather
than makes a test pass vacuously.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from biohubx.contracts.candidates import CandidateEdge, CandidateGraph, ReachReport
from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE, VoxelCoordinateZYX
from biohubx.contracts.instances import CandidateInstance, InstanceSet, ProposalSource
from biohubx.contracts.lineage import DatasetIdentity
from biohubx.contracts.predictions import (
    NO_PARENT,
    AssociationPrediction,
    ParentScore,
    TargetPrediction,
)
from biohubx.contracts.representation import InstanceFeatures, RepresentationSet

DATASET = DatasetIdentity(value="slice-contracts")
OTHER = DatasetIdentity(value="somewhere-else")


def instance(
    instance_id: int, frame: int, *, z: float = 0.0, y: float = 0.0, x: float = 0.0
) -> CandidateInstance:
    return CandidateInstance.from_voxel(
        dataset=DATASET,
        instance_id=instance_id,
        frame=frame,
        voxel=VoxelCoordinateZYX(z=z, y=y, x=x),
        source=ProposalSource.CLASSICAL_LOCAL_MAXIMUM,
        confidence=0.9,
    )


def features(instance_id: int, **overrides: object) -> InstanceFeatures:
    base: dict[str, object] = {
        "instance_id": instance_id,
        "peak_intensity": 0.8,
        "local_mean_intensity": 0.4,
        "neighbour_count": 1,
        "nearest_neighbour_um": 6.0,
        "is_isolated": False,
        "border_distance_um": 10.0,
    }
    base.update(overrides)
    return InstanceFeatures.model_validate(base)


def candidate(source: int, target: int, *, displacement_um: float = 2.0) -> CandidateEdge:
    return CandidateEdge(
        source=source,
        target=target,
        source_frame=0,
        target_frame=1,
        displacement_um=displacement_um,
    )


# --- instances --------------------------------------------------------------


def test_an_instance_carries_both_coordinate_systems() -> None:
    made = instance(0, 0, z=2.0, y=4.0, x=8.0)
    assert made.physical.z_um == pytest.approx(2.0 * OFFICIAL_VOXEL_SCALE.z_um)
    assert made.physical.x_um == pytest.approx(8.0 * OFFICIAL_VOXEL_SCALE.x_um)


def test_a_physical_coordinate_that_disagrees_with_its_voxel_is_refused() -> None:
    # Carrying two representations is only safe while they agree. A transposed
    # or rescaled physical coordinate would otherwise travel silently.
    with pytest.raises(ValidationError, match="disagrees with the voxel coordinate"):
        CandidateInstance(
            dataset=DATASET,
            instance_id=0,
            frame=0,
            voxel=VoxelCoordinateZYX(z=2.0, y=0.0, x=0.0),
            physical=VoxelCoordinateZYX(z=2.0, y=0.0, x=0.0)
            .to_physical(OFFICIAL_VOXEL_SCALE)
            .model_copy(update={"z_um": 99.0}),
            source=ProposalSource.CLASSICAL_LOCAL_MAXIMUM,
            confidence=0.5,
        )


def test_every_instance_records_which_detector_proposed_it() -> None:
    assert instance(0, 0).source is ProposalSource.CLASSICAL_LOCAL_MAXIMUM


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_a_confidence_outside_the_unit_interval_is_refused(bad: float) -> None:
    with pytest.raises(ValidationError):
        CandidateInstance.from_voxel(
            dataset=DATASET,
            instance_id=0,
            frame=0,
            voxel=VoxelCoordinateZYX(z=0.0, y=0.0, x=0.0),
            source=ProposalSource.CLASSICAL_LOCAL_MAXIMUM,
            confidence=bad,
        )


def test_duplicate_instance_identities_are_refused() -> None:
    with pytest.raises(ValidationError, match="duplicate instance identities"):
        InstanceSet(dataset=DATASET, instances=(instance(0, 0), instance(0, 1)))


def test_an_empty_proposal_set_is_a_failure_not_a_result() -> None:
    with pytest.raises(ValidationError, match="failure, not an empty result"):
        InstanceSet(dataset=DATASET, instances=())


def test_an_instance_from_another_dataset_is_refused() -> None:
    stray = CandidateInstance.from_voxel(
        dataset=OTHER,
        instance_id=1,
        frame=0,
        voxel=VoxelCoordinateZYX(z=0.0, y=0.0, x=0.0),
        source=ProposalSource.CLASSICAL_LOCAL_MAXIMUM,
        confidence=0.5,
    )
    with pytest.raises(ValidationError, match="crosses dataset identity"):
        InstanceSet(dataset=DATASET, instances=(instance(0, 0), stray))


# --- representation ---------------------------------------------------------


def test_a_missing_feature_raises_rather_than_returning_zeros() -> None:
    # Zero-filling would make an unmeasured channel indistinguishable from a
    # genuine measurement of zero, and quietly teach a model that absence is small.
    representation = RepresentationSet(dataset=DATASET, features=(features(0),))
    with pytest.raises(KeyError, match="no representation for instance 7"):
        representation.get(7)


def test_an_incomplete_feature_row_is_refused() -> None:
    with pytest.raises(ValidationError):
        InstanceFeatures.model_validate({"instance_id": 0, "peak_intensity": 0.5})


def test_an_isolated_instance_must_declare_its_sentinel() -> None:
    # The honest distance for a lone instance is infinite, which would propagate
    # silently through arithmetic, so isolation is a flag the caller must check.
    with pytest.raises(ValidationError, match="sentinel"):
        features(0, is_isolated=True, nearest_neighbour_um=5.0)
    assert features(0, is_isolated=True, nearest_neighbour_um=0.0, neighbour_count=0).is_isolated


def test_coverage_is_reported_rather_than_assumed() -> None:
    representation = RepresentationSet(dataset=DATASET, features=(features(0), features(1)))
    assert representation.covers(frozenset({0, 1}))
    assert not representation.covers(frozenset({0, 1, 2}))


def test_duplicate_feature_rows_are_refused() -> None:
    with pytest.raises(ValidationError, match="duplicate feature rows"):
        RepresentationSet(dataset=DATASET, features=(features(0), features(0)))


# --- candidates -------------------------------------------------------------


def test_a_duplicate_candidate_pair_is_refused() -> None:
    with pytest.raises(ValidationError, match="duplicate candidate pairs"):
        CandidateGraph(dataset=DATASET, radius_um=10.0, edges=(candidate(0, 1), candidate(0, 1)))


def test_a_candidate_beyond_the_declared_radius_is_refused() -> None:
    # The radius is a claim about what was offered. A graph that violates its own
    # claim makes the reach report meaningless.
    with pytest.raises(ValidationError, match="exceeds the declared radius"):
        CandidateGraph(dataset=DATASET, radius_um=5.0, edges=(candidate(0, 1, displacement_um=6.0),))


def test_a_candidate_that_does_not_advance_one_frame_is_refused() -> None:
    with pytest.raises(ValidationError, match="advance exactly one frame"):
        CandidateEdge(source=0, target=1, source_frame=0, target_frame=2, displacement_um=1.0)


def test_an_instance_cannot_be_its_own_parent() -> None:
    with pytest.raises(ValidationError, match="own parent"):
        CandidateEdge(source=3, target=3, source_frame=0, target_frame=1, displacement_um=0.0)


def test_an_empty_candidate_graph_is_a_failure() -> None:
    with pytest.raises(ValidationError, match="offers the matcher nothing"):
        CandidateGraph(dataset=DATASET, radius_um=10.0, edges=())


def test_reach_accounting_must_close() -> None:
    # Every true edge is either reachable or unreachable for exactly one stated
    # reason. Numbers that do not add up would hide a whole failure mode.
    with pytest.raises(ValidationError, match="does not close"):
        ReachReport(
            true_edges=10,
            reachable_true_edges=4,
            unreachable_missing_endpoint=2,
            unreachable_outside_radius=1,
        )


def test_reach_distinguishes_a_detector_miss_from_a_radius_miss() -> None:
    report = ReachReport(
        true_edges=10,
        reachable_true_edges=6,
        unreachable_missing_endpoint=3,
        unreachable_outside_radius=1,
    )
    assert report.reach == pytest.approx(0.6)
    assert report.unreachable_missing_endpoint != report.unreachable_outside_radius


def test_reach_with_no_true_edges_refuses_rather_than_returning_a_default() -> None:
    report = ReachReport(
        true_edges=0,
        reachable_true_edges=0,
        unreachable_missing_endpoint=0,
        unreachable_outside_radius=0,
    )
    with pytest.raises(ValueError, match="undefined"):
        _ = report.reach


def test_an_absent_reach_report_is_not_a_reach_of_zero() -> None:
    graph = CandidateGraph(dataset=DATASET, radius_um=10.0, edges=(candidate(0, 1),))
    assert graph.reach is None


# --- predictions ------------------------------------------------------------


def test_a_prediction_without_a_no_parent_row_is_refused() -> None:
    # The required mutation: abstention is a scored option, not an afterthought.
    with pytest.raises(ValidationError):
        TargetPrediction.model_validate(
            {"target": 1, "target_frame": 1, "parents": [{"source": 0, "score": -1.0}]}
        )


def test_abstention_competes_on_the_same_axis_as_the_parents() -> None:
    losing = TargetPrediction(
        target=1,
        target_frame=1,
        parents=(ParentScore(source=0, score=-3.0),),
        no_parent_score=-1.0,
    )
    winning = TargetPrediction(
        target=1,
        target_frame=1,
        parents=(ParentScore(source=0, score=-0.5),),
        no_parent_score=-1.0,
    )
    assert losing.best is None, "abstention should win when every parent scores worse"
    assert winning.best is not None and winning.best.source == 0


def test_a_tie_goes_to_abstention() -> None:
    # A tie means the evidence does not distinguish a link from a birth, and
    # inventing a link there is exactly what the no-parent row exists to prevent.
    tied = TargetPrediction(
        target=1,
        target_frame=1,
        parents=(ParentScore(source=0, score=-1.0),),
        no_parent_score=-1.0,
    )
    assert tied.best is None


def test_the_margin_covers_abstention_too() -> None:
    prediction = TargetPrediction(
        target=1,
        target_frame=1,
        parents=(ParentScore(source=0, score=-0.5), ParentScore(source=2, score=-2.0)),
        no_parent_score=-1.0,
    )
    # Top two overall are -0.5 (a parent) and -1.0 (abstention), not the two parents.
    assert prediction.margin == pytest.approx(0.5)


def test_a_target_offered_as_its_own_parent_is_refused() -> None:
    with pytest.raises(ValidationError, match="its own parent"):
        TargetPrediction(
            target=1,
            target_frame=1,
            parents=(ParentScore(source=1, score=-1.0),),
            no_parent_score=-2.0,
        )


def test_duplicate_parent_rows_are_refused() -> None:
    with pytest.raises(ValidationError, match="duplicate parent rows"):
        TargetPrediction(
            target=1,
            target_frame=1,
            parents=(ParentScore(source=0, score=-1.0), ParentScore(source=0, score=-2.0)),
            no_parent_score=-3.0,
        )


def test_a_skipped_target_is_not_an_abstention() -> None:
    # A target the matcher never scored is a gap. The decoder has to be able to
    # tell that apart from a target that was scored and chose to have no parent.
    prediction = AssociationPrediction(
        dataset=DATASET,
        targets=(
            TargetPrediction(
                target=1,
                target_frame=1,
                parents=(ParentScore(source=0, score=-1.0),),
                no_parent_score=-2.0,
            ),
        ),
    )
    assert prediction.covers(frozenset({1}))
    assert not prediction.covers(frozenset({1, 2}))


def test_the_abstention_option_has_a_stable_name() -> None:
    # Reports and manifests name it; a renamed constant would silently break them.
    assert NO_PARENT == "no_parent"
