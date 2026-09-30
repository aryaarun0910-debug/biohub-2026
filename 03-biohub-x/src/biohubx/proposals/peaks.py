"""Turn a detection heatmap into proposals the rest of the system already accepts.

A per-voxel heatmap is not a proposal set. Between them sit two steps the
reference performs and Biohub-X had not: reading peaks out of the map, and
suppressing the duplicates that a blob of adjacent bright voxels produces. The
E01 preflight measured the consequence of not having them, with candidate reach
limited entirely by endpoints that were never proposed ([[F-0017]]).

Suppression is not reimplemented here. It is the same greedy physical-radius
pass the classical detector uses, imported rather than copied, so the two
proposal sources cannot drift into disagreeing about what a duplicate is.

The heatmap is produced on the model's strided grid, so coordinates are scaled
back to full-resolution voxels before an instance is built. Getting that wrong
would put every proposal in the wrong place while every shape still matched,
which is exactly the kind of error a shape check cannot catch.

Consumer: ``biohubx train preflight``.
"""

from __future__ import annotations

import numpy as np

from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE, VoxelCoordinateZYX, VoxelScaleZYX
from biohubx.contracts.instances import CandidateInstance, InstanceSet, ProposalSource
from biohubx.contracts.lineage import DatasetIdentity
from biohubx.proposals.classical import DEFAULT_SUPPRESSION_RADIUS_UM, suppressed_maxima


class HeatmapProposalError(ValueError):
    """The heatmap cannot yield a proposal set."""


def instances_from_heatmap(
    heatmap: np.ndarray,
    *,
    dataset: DatasetIdentity,
    downsample: tuple[int, ...],
    threshold: float,
    suppression_radius_um: float = DEFAULT_SUPPRESSION_RADIUS_UM,
    scale: VoxelScaleZYX = OFFICIAL_VOXEL_SCALE,
) -> InstanceSet:
    """Read peaks from a ``(T, Z, Y, X)`` probability map and suppress duplicates.

    ``threshold`` is a probability, matching the reference's own detector
    threshold rather than a logit. Identities are assigned in the same fixed
    order the classical detector uses, so two runs over the same heatmap produce
    byte-identical output.
    """
    if heatmap.ndim != 4:
        raise HeatmapProposalError(f"expected a (T, Z, Y, X) heatmap, got shape {heatmap.shape}")
    if not 0.0 < threshold < 1.0:
        raise HeatmapProposalError(f"threshold must lie in (0, 1), got {threshold}")

    dz, dy, dx = (float(d) for d in downsample)
    # Suppression happens in micrometres on the grid it is given. Striding by
    # (1, 4, 4) makes that grid isotropic at 1.625 um, which is exactly why the
    # suppressor takes a spacing and not a VoxelScaleZYX.
    grid_spacing = (scale.z_um * dz, scale.y_um * dy, scale.x_um * dx)

    instances: list[CandidateInstance] = []
    next_id = 0
    for frame in range(heatmap.shape[0]):
        for centre, peak in suppressed_maxima(
            heatmap[frame],
            threshold=threshold,
            radius_um=suppression_radius_um,
            spacing_um=grid_spacing,
        ):
            instances.append(
                CandidateInstance.from_voxel(
                    dataset=dataset,
                    instance_id=next_id,
                    frame=frame,
                    voxel=VoxelCoordinateZYX(z=centre[0] * dz, y=centre[1] * dy, x=centre[2] * dx),
                    source=ProposalSource.CLASSICAL_LOCAL_MAXIMUM,
                    confidence=float(min(1.0, max(0.0, peak))),
                    scale=scale,
                )
            )
            next_id += 1

    if not instances:
        raise HeatmapProposalError(
            f"no voxel reached the {threshold} threshold, so the heatmap proposed nothing; "
            "an empty proposal set is a failure, not a result"
        )
    return InstanceSet(dataset=dataset, instances=tuple(instances))
