"""The re-scorer's shared invariants: patch geometry, channel contracts, count-tied thresholds, labels.

These are the parts every E07 arm relies on and no arm may bend: a patch is
always (2h+1)^3 and zero-padded at the grid's edge, the channel count is the
arm's declaration, the threshold that keeps k candidates keeps exactly k, and
a kept set is renumbered densely with probabilities as confidences.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from biohubx.contracts.coordinates import PhysicalCoordinateZYX, VoxelCoordinateZYX
from biohubx.contracts.instances import CandidateInstance, InstanceSet, ProposalSource
from biohubx.contracts.lineage import DatasetIdentity
from biohubx.proposals.rescore import (
    ARMS,
    PATCH_HALF,
    PatchScorer,
    RescoreError,
    _cut,
    build_scorer,
    candidate_patches,
    candidate_prior,
    keep_by_threshold,
    threshold_for_count,
)

DATASET = DatasetIdentity(value="rescore-fixture")


def instance(
    instance_id: int, frame: int, z: float, y: float, x: float, confidence: float = 0.5
) -> CandidateInstance:
    return CandidateInstance(
        dataset=DATASET,
        instance_id=instance_id,
        frame=frame,
        voxel=VoxelCoordinateZYX(z=z, y=y, x=x),
        physical=PhysicalCoordinateZYX(z_um=z * 1.625, y_um=y * 0.40625, x_um=x * 0.40625),
        source=ProposalSource.DOG_MULTISCALE,
        confidence=confidence,
    )


def test_cut_is_always_the_declared_size_and_zero_padded_at_the_edge() -> None:
    grid = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    corner = _cut(grid, (0, 0, 0), 2)
    assert corner.shape == (5, 5, 5)
    assert corner[:2].sum() == 0.0 and corner[2, 2, 2] == grid[0, 0, 0]
    centre = _cut(grid, (2, 2, 3), 1)
    assert np.array_equal(centre, grid[1:4, 1:4, 2:5])


def test_patches_carry_the_arm_channel_count_and_follow_instance_order() -> None:
    torch.manual_seed(0)
    volume = np.random.default_rng(0).random((3, 8, 64, 64)).astype(np.float32)
    instances = InstanceSet(
        dataset=DATASET, instances=(instance(0, 1, 4.0, 32.0, 32.0), instance(1, 2, 1.0, 8.0, 60.0))
    )
    for arm, spec in ARMS.items():
        patches = candidate_patches(volume, instances, arm=arm)
        assert patches.shape == (
            2,
            spec["channels"],
            2 * PATCH_HALF + 1,
            2 * PATCH_HALF + 1,
            2 * PATCH_HALF + 1,
        )
    # An InstanceSet refuses to be empty (a proposal step returning nothing is a failure),
    # so the empty-pool branch is unreachable through the contract and not tested here.
    with pytest.raises(RescoreError):
        candidate_patches(volume, instances, arm="A9")


def test_scorer_returns_one_logit_per_candidate_and_refuses_wrong_channels() -> None:
    model = build_scorer("A4", seed=1)
    assert isinstance(model, PatchScorer)
    logits = model(torch.zeros(3, 5, 9, 9, 9))
    assert logits.shape == (3,)
    with pytest.raises(RescoreError):
        model(torch.zeros(3, 3, 9, 9, 9))


def test_threshold_for_count_keeps_exactly_that_many() -> None:
    logits = torch.tensor([2.0, -1.0, 0.5, 3.0, -2.0])
    instances = InstanceSet(
        dataset=DATASET, instances=tuple(instance(i, 0, 1.0, 10.0 * i, 10.0) for i in range(5))
    )
    threshold = threshold_for_count(logits, 2)
    kept = keep_by_threshold(instances, logits, threshold)
    assert [k.instance_id for k in kept.instances] == [0, 1]
    assert sorted(round(k.confidence, 4) for k in kept.instances) == sorted(
        round(float(torch.sigmoid(logits[i])), 4) for i in (0, 3)
    )
    assert threshold_for_count(logits, 0) == 1.0
    assert keep_by_threshold(instances, logits, threshold_for_count(logits, 99)).instances.__len__() == 5


def test_prior_is_the_estimate_over_the_pool_and_stays_inside_the_open_interval() -> None:
    assert candidate_prior(50.0, 200) == 0.25
    assert 0.0 < candidate_prior(1000.0, 10) < 1.0
    with pytest.raises(RescoreError):
        candidate_prior(10.0, 0)
