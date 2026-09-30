"""Physical geometry and intensity features for candidate instances.

Everything here is computed in micrometres. Nothing is defaulted: if a feature
cannot be measured for an instance, this module raises rather than filling the
channel with zero, because a zero density and an unmeasured density are
different claims and only one of them is true.

Consumer: :mod:`biohubx.tracking.matcher`.
"""

from __future__ import annotations

import numpy as np

from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE, VoxelScaleZYX
from biohubx.contracts.instances import CandidateInstance, InstanceSet
from biohubx.contracts.representation import InstanceFeatures, RepresentationSet

DEFAULT_DENSITY_RADIUS_UM = 12.0


def describe_instances(
    instances: InstanceSet,
    volume: np.ndarray,
    *,
    density_radius_um: float = DEFAULT_DENSITY_RADIUS_UM,
    scale: VoxelScaleZYX = OFFICIAL_VOXEL_SCALE,
) -> RepresentationSet:
    """Measure every instance. Every instance gets a full row or the call fails."""
    if volume.ndim != 4:
        raise ValueError(f"expected a (T, Z, Y, X) volume, got shape {volume.shape}")
    if density_radius_um <= 0:
        raise ValueError(f"density radius must be positive, got {density_radius_um}")

    shape = volume.shape[1:]
    spacing = np.array(scale.values, dtype=np.float64)
    features: list[InstanceFeatures] = []

    for frame in instances.frames:
        in_frame = instances.by_frame(frame)
        positions = np.array(
            [[item.physical.z_um, item.physical.y_um, item.physical.x_um] for item in in_frame],
            dtype=np.float64,
        )
        for index, instance in enumerate(in_frame):
            separations = np.linalg.norm(positions - positions[index], axis=1)
            others = np.delete(separations, index)
            isolated = others.size == 0
            features.append(
                InstanceFeatures(
                    instance_id=instance.instance_id,
                    peak_intensity=_sample(volume, instance),
                    local_mean_intensity=_neighbourhood_mean(volume, instance, scale=scale),
                    neighbour_count=int(np.count_nonzero(others <= density_radius_um)),
                    nearest_neighbour_um=0.0 if isolated else float(others.min()),
                    is_isolated=isolated,
                    border_distance_um=_border_distance_um(instance, shape, spacing),
                )
            )
    return RepresentationSet(dataset=instances.dataset, features=tuple(features))


def _voxel_index(instance: CandidateInstance, shape: tuple[int, ...]) -> tuple[int, int, int]:
    coordinates = (instance.voxel.z, instance.voxel.y, instance.voxel.x)
    return tuple(  # type: ignore[return-value]
        int(min(max(round(value), 0), size - 1)) for value, size in zip(coordinates, shape, strict=True)
    )


def _sample(volume: np.ndarray, instance: CandidateInstance) -> float:
    z, y, x = _voxel_index(instance, volume.shape[1:])
    return float(volume[instance.frame, z, y, x])


def _neighbourhood_mean(volume: np.ndarray, instance: CandidateInstance, *, scale: VoxelScaleZYX) -> float:
    """Mean intensity over a box that is roughly cubic in micrometres.

    Half-widths are derived per axis from the physical spacing, so the box is not
    four times taller in micrometres than it is wide.
    """
    extent_um = 2.0
    shape = volume.shape[1:]
    centre = _voxel_index(instance, shape)
    half = [max(1, round(extent_um / axis_um)) for axis_um in scale.values]
    slices = tuple(
        slice(max(0, centre[axis] - half[axis]), min(shape[axis], centre[axis] + half[axis] + 1))
        for axis in range(3)
    )
    window = volume[instance.frame][slices]
    if window.size == 0:
        raise ValueError(f"instance {instance.instance_id} has an empty neighbourhood window")
    return float(window.mean())


def _border_distance_um(instance: CandidateInstance, shape: tuple[int, ...], spacing: np.ndarray) -> float:
    position = np.array([instance.voxel.z, instance.voxel.y, instance.voxel.x], dtype=np.float64)
    extent = np.array(shape, dtype=np.float64) - 1.0
    to_low = position * spacing
    to_high = (extent - position) * spacing
    return float(max(0.0, min(np.concatenate([to_low, to_high]))))
