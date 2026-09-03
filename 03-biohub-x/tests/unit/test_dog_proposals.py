"""The DoG source must land proposals in full-resolution voxels, or F-0025 is wrong.

The plane is strided to make the grid isotropic, so every coordinate has to be
scaled back before an instance is built. Getting that wrong puts every proposal at
a quarter of its true y and x while every shape still matches, which no shape check
would catch and which would quietly destroy recall.
"""

from __future__ import annotations

import numpy as np
import pytest

from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE, VoxelScaleZYX
from biohubx.contracts.instances import ProposalSource
from biohubx.contracts.lineage import DatasetIdentity
from biohubx.proposals.dog import detect_instances, dog_response, isotropic_plane_stride

DATASET = DatasetIdentity(value="dog-fixture")


def volume_with_blob(
    centre: tuple[int, int, int], shape: tuple[int, int, int, int] = (1, 24, 96, 96)
) -> np.ndarray:
    """One Gaussian blob of roughly nucleus size, at a known full-resolution voxel."""
    volume = np.zeros(shape, dtype=np.float32)
    z, y, x = np.ogrid[: shape[1], : shape[2], : shape[3]]
    cz, cy, cx = centre
    # Isotropic in micrometres: z spacing is four times y and x.
    squared = ((z - cz) * 4.0) ** 2 + (y - cy) ** 2 + (x - cx) ** 2
    volume[0] = np.exp(-squared / (2 * 7.0**2))
    return volume


def test_the_plane_stride_comes_from_the_official_scale() -> None:
    assert isotropic_plane_stride(OFFICIAL_VOXEL_SCALE) == 4


def test_a_non_integral_anisotropy_refuses_rather_than_rounding() -> None:
    """Nearly isotropic is not isotropic, and one sigma would then mean different
    distances along different axes."""
    with pytest.raises(ValueError, match="not integral"):
        isotropic_plane_stride(VoxelScaleZYX(z_um=1.5, y_um=0.4, x_um=0.4))


def test_a_blob_is_proposed_at_its_full_resolution_position() -> None:
    centre = (12, 48, 60)
    instances = detect_instances(volume_with_blob(centre), dataset=DATASET, response_quantile=0.999)

    assert instances.instances
    best = min(
        instances.instances,
        key=lambda i: abs(i.voxel.y - centre[1]) + abs(i.voxel.x - centre[2]),
    )
    # Within one strided step in the plane, which is four full-resolution voxels.
    assert abs(best.voxel.y - centre[1]) <= 4
    assert abs(best.voxel.x - centre[2]) <= 4
    assert abs(best.voxel.z - centre[0]) <= 2
    assert best.source is ProposalSource.DOG_MULTISCALE


def test_the_scale_bank_may_not_be_empty() -> None:
    with pytest.raises(ValueError, match="scale bank is empty"):
        dog_response(np.zeros((4, 8, 8), dtype=np.float32), radii_um=(), voxel_um=1.625)


def test_a_flat_volume_proposes_nothing_and_says_so() -> None:
    """An empty proposal set is a failure, not a result, and the other sources
    already refuse it."""
    with pytest.raises(ValueError, match="proposed nothing"):
        detect_instances(np.zeros((1, 8, 32, 32), dtype=np.float32), dataset=DATASET)
