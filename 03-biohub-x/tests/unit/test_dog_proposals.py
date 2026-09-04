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
from biohubx.contracts.instances import InstanceSet, ProposalSource
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


def test_bucketed_suppression_agrees_exactly_with_brute_force() -> None:
    """The bucket search must be an optimisation, not a different algorithm.

    F-0006, F-0017, F-0019 and F-0025 all rest on this detector's output, so a
    suppression that returns a different set would silently invalidate them. The
    bucketing is exact because any point within the radius lies in one of the 27
    surrounding cells; this asserts that rather than trusting the argument.
    """
    from biohubx.proposals.classical import suppressed_maxima

    rng = np.random.default_rng(20260904)
    frame = rng.random((12, 40, 40)).astype(np.float32)
    spacing = (1.625, 1.625, 1.625)
    radius = 4.0

    fast = suppressed_maxima(frame, threshold=0.9, radius_um=radius, spacing_um=spacing)

    # Brute force, written out here so the reference is independent of the code
    # under test rather than a refactor of it.
    above = np.argwhere(frame >= 0.9)
    values = frame[above[:, 0], above[:, 1], above[:, 2]]
    order = np.lexsort((above[:, 2], above[:, 1], above[:, 0], -values))
    taken: list[np.ndarray] = []
    slow: list[tuple[tuple[float, float, float], float]] = []
    for index in order:
        voxel = above[index].astype(np.float64)
        physical = voxel * np.array(spacing)
        if any(float(np.linalg.norm(physical - other)) < radius for other in taken):
            continue
        taken.append(physical)
        slow.append(((float(voxel[0]), float(voxel[1]), float(voxel[2])), float(values[index])))

    assert len(fast) == len(slow)
    assert [point for point, _ in fast] == [point for point, _ in slow]


def test_refinement_moves_a_peak_toward_a_blob_centred_between_grid_voxels() -> None:
    """A blob centred at a half-voxel offset in the plane lands on a grid voxel
    without refinement and closer to its true centre with it. If refinement made
    it worse the H-11 probe would be measuring the wrong thing."""
    from biohubx.proposals.dog import detect_instances as detect

    # Centre at y = 50, x = 62: on the stride-4 grid that is 12.5 and 15.5, so
    # exactly between voxels in both plane axes.
    volume = volume_with_blob((12, 50, 62))
    plain = detect(volume, dataset=DATASET, response_quantile=0.999)
    refined = detect(volume, dataset=DATASET, response_quantile=0.999, refine_centroids=True)

    def nearest(instances: InstanceSet) -> tuple[float, float]:
        best = min(instances.instances, key=lambda i: abs(i.voxel.y - 50) + abs(i.voxel.x - 62))
        return abs(best.voxel.y - 50), abs(best.voxel.x - 62)

    plain_dy, plain_dx = nearest(plain)
    refined_dy, refined_dx = nearest(refined)
    # Refinement must move toward the true centre on each plane axis, not merely
    # reduce a sum that one axis could dominate. A one-voxel cube on a stride-4
    # grid is deliberately conservative, so the gain is bounded and no absolute
    # target is asserted; the direction is the claim.
    assert refined_dy < plain_dy
    assert refined_dx < plain_dx


def test_local_maxima_only_yields_one_proposal_per_blob_where_packing_yields_several() -> None:
    """A single wide blob. Greedy suppression packs its shoulders as separate
    proposals once the threshold admits them; requiring a true local maximum
    leaves exactly the peak."""
    volume = volume_with_blob((12, 48, 48))
    packed = detect_instances(volume, dataset=DATASET, response_quantile=0.95)
    peaks = detect_instances(volume, dataset=DATASET, response_quantile=0.95, local_maxima_only=True)

    assert len(packed.instances) > 1
    assert len(peaks.instances) == 1


def test_a_peak_on_the_volume_face_is_still_a_peak() -> None:
    """Replicated padding makes a face voxel tie with its own copy, so a strict
    maximum on the first slice is silently dropped. A blob whose centre sits on
    z=0 must still yield exactly one peak proposal."""
    volume = volume_with_blob((0, 48, 48))
    peaks = detect_instances(volume, dataset=DATASET, response_quantile=0.95, local_maxima_only=True)

    assert len(peaks.instances) == 1
    assert peaks.instances[0].voxel.z <= 1.0


def test_per_scale_union_keeps_two_neighbours_a_large_scale_fuses() -> None:
    """Two blobs 6 um apart in the plane. A 4.5 um scale fuses them into one
    maximum that wins the pointwise max; strict peaks per scale keep both."""
    volume = volume_with_blob((12, 48, 40))
    volume = np.maximum(volume, volume_with_blob((12, 48, 55)))
    bank = (2.0, 4.5)
    fused = detect_instances(
        volume, dataset=DATASET, radii_um=bank, response_quantile=0.95, local_maxima_only=True
    )
    union = detect_instances(
        volume, dataset=DATASET, radii_um=bank, response_quantile=0.95, per_scale_union=True
    )

    assert len(union.instances) >= len(fused.instances)
    assert len(union.instances) == 2
