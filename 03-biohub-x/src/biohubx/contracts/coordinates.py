"""Explicit voxel and physical coordinate contracts.

Consumer: :mod:`biohubx.evaluation.official_metric`.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

AxisOrder = Literal["zyx"]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class VoxelScaleZYX(BaseModel):
    """Physical micrometres per voxel in explicit ``(z, y, x)`` order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    axis_order: AxisOrder = "zyx"
    z_um: PositiveFloat
    y_um: PositiveFloat
    x_um: PositiveFloat

    @property
    def values(self) -> tuple[float, float, float]:
        return (self.z_um, self.y_um, self.x_um)

    @model_validator(mode="after")
    def refuse_isotropic_substitution(self) -> VoxelScaleZYX:
        if self.z_um == self.y_um == self.x_um:
            raise ValueError("isotropic voxel scale is forbidden for the anisotropic official contract")
        return self


class VoxelCoordinateZYX(BaseModel):
    """A voxel coordinate whose axes cannot be anonymous or transposed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    axis_order: AxisOrder = "zyx"
    z: FiniteFloat
    y: FiniteFloat
    x: FiniteFloat

    def to_physical(self, scale: VoxelScaleZYX) -> PhysicalCoordinateZYX:
        return PhysicalCoordinateZYX(
            z_um=self.z * scale.z_um, y_um=self.y * scale.y_um, x_um=self.x * scale.x_um
        )


class PhysicalCoordinateZYX(BaseModel):
    """A physical coordinate in micrometres, explicitly ``(z, y, x)``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    axis_order: AxisOrder = "zyx"
    z_um: FiniteFloat
    y_um: FiniteFloat
    x_um: FiniteFloat
    round_trip_tolerance_voxels: ClassVar[float] = 1e-9

    def to_voxel(self, scale: VoxelScaleZYX) -> VoxelCoordinateZYX:
        return VoxelCoordinateZYX(
            z=self.z_um / scale.z_um, y=self.y_um / scale.y_um, x=self.x_um / scale.x_um
        )


OFFICIAL_VOXEL_SCALE = VoxelScaleZYX(z_um=1.625, y_um=0.40625, x_um=0.40625)
