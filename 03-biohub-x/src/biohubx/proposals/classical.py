"""A deterministic local-maximum proposal source.

No learning, no weights, no randomness. It exists so the vertical slice has a
real detector rather than the ground truth in disguise: it genuinely misses
cells and genuinely invents them, which is what makes the reach report and the
node-count accounting downstream mean anything.

Suppression happens in micrometres, not voxels. On this grid a z voxel is four
times a y or x voxel, so a voxel-radius neighbourhood would suppress four times
too aggressively along z and let duplicates through in the plane.

Consumers: ``biohubx infer synthetic``, and ``biohubx.proposals.peaks`` for the
suppression it shares.
"""

from __future__ import annotations

import numpy as np

from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE, VoxelCoordinateZYX, VoxelScaleZYX
from biohubx.contracts.instances import (
    CandidateInstance,
    InstanceSet,
    ProposalSource,
)
from biohubx.contracts.lineage import DatasetIdentity

DEFAULT_DETECTION_THRESHOLD = 0.30
DEFAULT_SUPPRESSION_RADIUS_UM = 4.0


def detect_instances(
    volume: np.ndarray,
    *,
    dataset: DatasetIdentity,
    threshold: float = DEFAULT_DETECTION_THRESHOLD,
    suppression_radius_um: float = DEFAULT_SUPPRESSION_RADIUS_UM,
    scale: VoxelScaleZYX = OFFICIAL_VOXEL_SCALE,
) -> InstanceSet:
    """Propose one instance per suppressed local maximum, frame by frame.

    Identities are assigned in a fixed order (frame, then descending intensity,
    then voxel position) so that two runs over the same volume produce byte-
    identical output. A detector whose identities move between runs makes every
    downstream artifact digest meaningless.
    """
    if volume.ndim != 4:
        raise ValueError(f"expected a (T, Z, Y, X) volume, got shape {volume.shape}")
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must lie in (0, 1), got {threshold}")
    if suppression_radius_um <= 0:
        raise ValueError(f"suppression radius must be positive, got {suppression_radius_um}")

    instances: list[CandidateInstance] = []
    next_id = 0
    for frame in range(volume.shape[0]):
        for centre, peak in suppressed_maxima(
            volume[frame],
            threshold=threshold,
            radius_um=suppression_radius_um,
            spacing_um=scale.values,
        ):
            instances.append(
                CandidateInstance.from_voxel(
                    dataset=dataset,
                    instance_id=next_id,
                    frame=frame,
                    voxel=VoxelCoordinateZYX(z=centre[0], y=centre[1], x=centre[2]),
                    source=ProposalSource.CLASSICAL_LOCAL_MAXIMUM,
                    confidence=float(min(1.0, max(0.0, peak))),
                    scale=scale,
                )
            )
            next_id += 1
    if not instances:
        raise ValueError("the detector proposed nothing; an empty proposal set is a failure, not a result")
    return InstanceSet(dataset=dataset, instances=tuple(instances))


def suppressed_maxima(
    frame: np.ndarray,
    *,
    threshold: float,
    radius_um: float,
    spacing_um: tuple[float, float, float],
) -> list[tuple[tuple[float, float, float], float]]:
    """Greedy physical-radius suppression over voxels above the threshold.

    Takes a spacing rather than a :class:`VoxelScaleZYX` because it suppresses on
    whatever grid it is handed, and a learned detector's strided grid is not the
    official voxel grid. On this data the reference's (1, 4, 4) striding makes
    that grid isotropic at 1.625 um, which the coordinate contract refuses to
    represent, and rightly: that contract exists to stop an isotropic scale being
    substituted for the anisotropic official one. Here the isotropy is a real
    property of a derived grid, not a substitution, so the spacing travels as
    plain micrometres and the contract keeps guarding what it was written for.
    """
    above = np.argwhere(frame >= threshold)
    if above.size == 0:
        return []

    values = frame[above[:, 0], above[:, 1], above[:, 2]]
    # Sort by descending intensity, breaking ties by position so the result does
    # not depend on the order argwhere happened to return.
    order = np.lexsort((above[:, 2], above[:, 1], above[:, 0], -values))

    spacing = np.array(spacing_um, dtype=np.float64)
    accepted_physical: list[np.ndarray] = []
    accepted: list[tuple[tuple[float, float, float], float]] = []
    for index in order:
        voxel = above[index].astype(np.float64)
        physical = voxel * spacing
        if any(float(np.linalg.norm(physical - taken)) < radius_um for taken in accepted_physical):
            continue
        accepted_physical.append(physical)
        accepted.append(((float(voxel[0]), float(voxel[1]), float(voxel[2])), float(values[index])))
    return accepted
