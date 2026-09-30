"""Coordinate contract.

The official metric measures distance in micrometres on an anisotropic grid
where z voxels are exactly four times the y and x voxels. Every downstream
component that compares positions inherits that asymmetry, so the contract has
to make the axis order explicit, round-trip exactly, and refuse the isotropic
substitution outright.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from biohubx.contracts.coordinates import (
    OFFICIAL_VOXEL_SCALE,
    PhysicalCoordinateZYX,
    VoxelCoordinateZYX,
    VoxelScaleZYX,
)

# The official spacing, restated here so a silent edit to the constant fails a
# test rather than propagating into every recorded score.
OFFICIAL_Z_UM = 1.625
OFFICIAL_YX_UM = 0.40625
OFFICIAL_MATCHING_RADIUS_UM = 7.0


def test_the_official_scale_is_the_published_spacing() -> None:
    assert OFFICIAL_VOXEL_SCALE.values == (OFFICIAL_Z_UM, OFFICIAL_YX_UM, OFFICIAL_YX_UM)
    assert OFFICIAL_VOXEL_SCALE.axis_order == "zyx"


def test_axis_order_is_declared_not_positional() -> None:
    # A three-column tensor cannot say which column is z. These objects can.
    voxel = VoxelCoordinateZYX(z=1.0, y=2.0, x=3.0)
    assert voxel.axis_order == "zyx"
    assert (voxel.z, voxel.y, voxel.x) == (1.0, 2.0, 3.0)


def test_voxel_to_physical_uses_the_matching_axis_scale() -> None:
    physical = VoxelCoordinateZYX(z=1.0, y=1.0, x=1.0).to_physical(OFFICIAL_VOXEL_SCALE)
    assert physical.z_um == pytest.approx(OFFICIAL_Z_UM)
    assert physical.y_um == pytest.approx(OFFICIAL_YX_UM)
    assert physical.x_um == pytest.approx(OFFICIAL_YX_UM)


@pytest.mark.parametrize(
    ("z", "y", "x"),
    [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (12.5, -7.25, 300.75), (-4.0, 0.5, 0.125)],
)
def test_round_trip_is_exact_within_tolerance(z: float, y: float, x: float) -> None:
    voxel = VoxelCoordinateZYX(z=z, y=y, x=x)
    returned = voxel.to_physical(OFFICIAL_VOXEL_SCALE).to_voxel(OFFICIAL_VOXEL_SCALE)
    tolerance = PhysicalCoordinateZYX.round_trip_tolerance_voxels
    assert math.isclose(returned.z, z, abs_tol=tolerance)
    assert math.isclose(returned.y, y, abs_tol=tolerance)
    assert math.isclose(returned.x, x, abs_tol=tolerance)


def test_an_isotropic_scale_cannot_be_constructed() -> None:
    with pytest.raises(ValidationError):
        VoxelScaleZYX(z_um=0.40625, y_um=0.40625, x_um=0.40625)


@pytest.mark.parametrize(
    ("z", "y", "x"), [(0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 2.0)]
)
def test_a_non_positive_scale_is_refused(z: float, y: float, x: float) -> None:
    with pytest.raises(ValidationError):
        VoxelScaleZYX(z_um=z, y_um=y, x_um=x)


def test_a_transposed_scale_is_a_different_object() -> None:
    # Swapping z and x produces a scale that is valid in isolation but wrong for
    # this dataset. The contract cannot detect intent; it can only make the
    # difference visible, which is what the negative control below relies on.
    transposed = VoxelScaleZYX(z_um=OFFICIAL_YX_UM, y_um=OFFICIAL_YX_UM, x_um=OFFICIAL_Z_UM)
    assert transposed.values != OFFICIAL_VOXEL_SCALE.values


# --- negative control: the isotropic convention must fail ------------------


def _separation_um(offset: tuple[float, float, float], scale: VoxelScaleZYX) -> float:
    origin = VoxelCoordinateZYX(z=0.0, y=0.0, x=0.0).to_physical(scale)
    other = VoxelCoordinateZYX(z=offset[0], y=offset[1], x=offset[2]).to_physical(scale)
    return math.dist(
        (origin.z_um, origin.y_um, origin.x_um),
        (other.z_um, other.y_um, other.x_um),
    )


def test_isotropic_substitution_changes_which_pairs_are_within_the_matching_radius() -> None:
    """The negative control the mission requires.

    Five voxels along z is 8.125 um and lies OUTSIDE the 7 um matching radius.
    The identical five voxels along x is 2.03 um and lies INSIDE it. Under an
    isotropic scale the two are indistinguishable, so a system built on the
    isotropic convention would match a pair the official metric rejects and
    would be scored on a different graph than the one it thinks it emitted.
    """
    five_along_z = (5.0, 0.0, 0.0)
    five_along_x = (0.0, 0.0, 5.0)

    z_true = _separation_um(five_along_z, OFFICIAL_VOXEL_SCALE)
    x_true = _separation_um(five_along_x, OFFICIAL_VOXEL_SCALE)
    assert z_true == pytest.approx(8.125)
    assert x_true == pytest.approx(2.03125)
    assert z_true > OFFICIAL_MATCHING_RADIUS_UM > x_true

    # The isotropic convention cannot be built with this contract, so the
    # control constructs the distance it WOULD produce and shows it collapses
    # the two cases onto one another and onto the wrong side of the radius.
    z_isotropic = math.dist((0.0, 0.0, 0.0), tuple(v * OFFICIAL_YX_UM for v in five_along_z))
    x_isotropic = math.dist((0.0, 0.0, 0.0), tuple(v * OFFICIAL_YX_UM for v in five_along_x))
    assert z_isotropic == pytest.approx(x_isotropic)
    assert z_isotropic < OFFICIAL_MATCHING_RADIUS_UM
    assert (z_true > OFFICIAL_MATCHING_RADIUS_UM) is not (z_isotropic > OFFICIAL_MATCHING_RADIUS_UM)


def test_voxel_distance_alone_is_not_a_physical_distance() -> None:
    # Equal voxel displacements along different axes are not equal separations.
    # Any component that gates on raw voxel distance is measuring the wrong thing.
    assert _separation_um((1.0, 0.0, 0.0), OFFICIAL_VOXEL_SCALE) == pytest.approx(
        4.0 * _separation_um((0.0, 0.0, 1.0), OFFICIAL_VOXEL_SCALE)
    )


def test_coordinates_are_frozen() -> None:
    voxel = VoxelCoordinateZYX(z=1.0, y=2.0, x=3.0)
    with pytest.raises(ValidationError):
        voxel.z = 9.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_coordinates_are_refused(bad: float) -> None:
    with pytest.raises(ValidationError):
        VoxelCoordinateZYX(z=bad, y=0.0, x=0.0)


def test_unknown_fields_are_refused() -> None:
    with pytest.raises(ValidationError):
        VoxelCoordinateZYX(z=0.0, y=0.0, x=0.0, t=3)  # type: ignore[call-arg]
