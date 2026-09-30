"""Candidate instance contract.

A candidate instance is one hypothesis that a cell exists at a place and time.
It is not yet a node in the emitted graph: the decoder decides that later, which
is the whole point of keeping proposals and decisions separate.

Every instance records which detector proposed it and how confident that
detector was, so that a later multi-source bank can keep competing hypotheses
without first destroying the evidence that distinguishes them.

Consumers: :mod:`biohubx.proposals.classical`, :mod:`biohubx.tracking.candidate_graph`.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from biohubx.contracts.coordinates import (
    OFFICIAL_VOXEL_SCALE,
    PhysicalCoordinateZYX,
    VoxelCoordinateZYX,
    VoxelScaleZYX,
)
from biohubx.contracts.lineage import DatasetIdentity

Identity = Annotated[StrictInt, Field(ge=0)]
Frame = Annotated[StrictInt, Field(ge=0)]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class ProposalSource(StrEnum):
    """Which detector family proposed an instance.

    Recorded from the first instance onward, even while only one source exists,
    because a bitmask added later cannot recover provenance that was never
    written down.
    """

    CLASSICAL_LOCAL_MAXIMUM = "classical_local_maximum"
    """Deterministic local-maximum detector with physical-radius suppression."""

    DOG_MULTISCALE = "dog_multiscale"
    """Multi-scale Difference-of-Gaussians blob detector on the isotropic grid."""


class CandidateInstance(BaseModel):
    """One proposed cell, in both coordinate systems, with its provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: DatasetIdentity
    instance_id: Identity
    frame: Frame
    voxel: VoxelCoordinateZYX
    physical: PhysicalCoordinateZYX
    source: ProposalSource
    confidence: UnitInterval

    @model_validator(mode="after")
    def physical_matches_voxel(self) -> CandidateInstance:
        """Both coordinate systems must describe the same point.

        Carrying two representations is what stops a downstream component from
        having to guess which one it holds. That is only safe while they agree.
        """
        expected = self.voxel.to_physical(OFFICIAL_VOXEL_SCALE)
        tolerance = 1e-9
        for axis, got, want in (
            ("z", self.physical.z_um, expected.z_um),
            ("y", self.physical.y_um, expected.y_um),
            ("x", self.physical.x_um, expected.x_um),
        ):
            if abs(got - want) > tolerance:
                raise ValueError(
                    f"physical {axis} disagrees with the voxel coordinate under the official scale: "
                    f"{got} != {want}"
                )
        return self

    @classmethod
    def from_voxel(
        cls,
        *,
        dataset: DatasetIdentity,
        instance_id: int,
        frame: int,
        voxel: VoxelCoordinateZYX,
        source: ProposalSource,
        confidence: float,
        scale: VoxelScaleZYX = OFFICIAL_VOXEL_SCALE,
    ) -> CandidateInstance:
        """Build an instance, deriving the physical coordinate from the scale."""
        return cls(
            dataset=dataset,
            instance_id=instance_id,
            frame=frame,
            voxel=voxel,
            physical=voxel.to_physical(scale),
            source=source,
            confidence=confidence,
        )


class InstanceSet(BaseModel):
    """Every candidate instance proposed for one dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: DatasetIdentity
    instances: tuple[CandidateInstance, ...]

    @field_validator("instances")
    @classmethod
    def require_instances(cls, value: tuple[CandidateInstance, ...]) -> tuple[CandidateInstance, ...]:
        if not value:
            raise ValueError("a proposal step that returns nothing is a failure, not an empty result")
        return value

    @model_validator(mode="after")
    def validate_set(self) -> InstanceSet:
        ids = [instance.instance_id for instance in self.instances]
        duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate instance identities: {duplicates}")
        for instance in self.instances:
            if instance.dataset != self.dataset:
                raise ValueError(f"instance {instance.instance_id} crosses dataset identity")
        return self

    def by_frame(self, frame: int) -> tuple[CandidateInstance, ...]:
        return tuple(
            sorted(
                (instance for instance in self.instances if instance.frame == frame),
                key=lambda instance: instance.instance_id,
            )
        )

    @property
    def frames(self) -> tuple[int, ...]:
        return tuple(sorted({instance.frame for instance in self.instances}))

    def get(self, instance_id: int) -> CandidateInstance:
        for instance in self.instances:
            if instance.instance_id == instance_id:
                return instance
        raise KeyError(f"no instance with identity {instance_id}")
