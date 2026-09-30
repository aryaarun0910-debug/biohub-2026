"""The audit's two counts must mean what F-0027 and F-0028 say they mean.

One-to-one recall is a capped assignment, so one proposal cannot satisfy two
annotated cells; reachability is not, so it can. And a near miss must report its
displacement with the sign pointing from the cell to the proposal, or the z-versus-
plane question H-11 asks would be answered backwards.
"""

from __future__ import annotations

import numpy as np

from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE, VoxelCoordinateZYX
from biohubx.contracts.instances import CandidateInstance, InstanceSet, ProposalSource
from biohubx.contracts.lineage import DatasetIdentity, LineageGraph, LineageNode
from biohubx.evaluation.proposals import measure_reachability
from biohubx.evaluation.residuals import audit_nodes, one_to_one_recall

DATASET = DatasetIdentity(value="residual-fixture")


def node(node_id: int, z: float, y: float, x: float, frame: int = 0) -> LineageNode:
    return LineageNode(dataset=DATASET, node_id=node_id, frame=frame, voxel=VoxelCoordinateZYX(z=z, y=y, x=x))


def proposal(instance_id: int, z: float, y: float, x: float, frame: int = 0) -> CandidateInstance:
    return CandidateInstance.from_voxel(
        dataset=DATASET,
        instance_id=instance_id,
        frame=frame,
        voxel=VoxelCoordinateZYX(z=z, y=y, x=x),
        source=ProposalSource.DOG_MULTISCALE,
        confidence=1.0,
        scale=OFFICIAL_VOXEL_SCALE,
    )


def test_one_proposal_reaches_two_cells_but_can_only_match_one() -> None:
    """Two annotated cells 4 um apart in the plane, one proposal between them."""
    annotated = LineageGraph(dataset=DATASET, nodes=(node(1, 0, 0, 0), node(2, 0, 0, 9.8)), edges=())
    proposals = InstanceSet(dataset=DATASET, instances=(proposal(0, 0, 0, 4.9),))

    reach = measure_reachability(proposals, annotated, estimated_total_nodes=2.0)
    matched, total = one_to_one_recall(proposals, annotated)

    assert reach.reached == 2
    assert (matched, total) == (1, 2)


def test_a_near_miss_reports_its_displacement_toward_the_proposal() -> None:
    """Cell at the origin, proposal 6 voxels deeper in z, which is 9.75 um: a miss
    by distance, and the displacement must say it was along z and positive."""
    annotated = LineageGraph(dataset=DATASET, nodes=(node(1, 10, 40, 40),), edges=())
    proposals = InstanceSet(dataset=DATASET, instances=(proposal(0, 16, 40, 40),))
    volume = np.zeros((1, 32, 96, 96), dtype=np.float32)

    [audit] = audit_nodes(
        dataset_id="44b6_fixture",
        embryo="44b6",
        volume=volume,
        instances=proposals,
        annotated=annotated,
        response_per_frame={},
        plane_stride=4,
        first_frame=0,
    )

    assert audit.reached is False
    assert abs(audit.nearest_proposal_um - 9.75) < 1e-6
    assert abs(audit.nearest_dz_um - 9.75) < 1e-6
    assert audit.nearest_dy_um == 0.0 and audit.nearest_dx_um == 0.0
