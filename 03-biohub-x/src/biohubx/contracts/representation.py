"""Per-instance representation contract.

The representation is what the matcher is allowed to reason from. Phase 2 fills
it with geometry and intensity only; appearance and morphology embeddings arrive
with the encoder that produces them.

Two rules make this contract worth having rather than passing a bare vector:

* every channel is a named field, so a matcher cannot silently read the wrong
  column when a new feature is inserted;
* a missing feature is an error. Nothing here is defaulted to zero, because a
  zero-filled channel is indistinguishable from a genuine measurement of zero
  and quietly teaches a model that absence means "small".

Consumers: :mod:`biohubx.representation.geometry`, :mod:`biohubx.tracking.matcher`.
"""

from __future__ import annotations

from collections import Counter
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from biohubx.contracts.lineage import DatasetIdentity

Identity = Annotated[StrictInt, Field(ge=0)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]


class InstanceFeatures(BaseModel):
    """The evidence one candidate instance carries into association.

    Every field is required. There is no constructor that fills a channel it was
    not given.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    instance_id: Identity

    peak_intensity: FiniteFloat
    """Normalised intensity at the proposed centre."""

    local_mean_intensity: FiniteFloat
    """Mean normalised intensity in the neighbourhood used for suppression."""

    neighbour_count: Annotated[StrictInt, Field(ge=0)]
    """Instances within the density radius, in physical units, excluding itself."""

    nearest_neighbour_um: NonNegative
    """Physical distance to the closest other instance in the same frame.

    Infinite would be the honest value for a lone instance, but an infinity
    propagates silently through arithmetic, so a lone instance is represented by
    the sentinel below and callers must check it.
    """

    is_isolated: bool
    """True when the frame holds no other instance, making the distance a sentinel."""

    border_distance_um: NonNegative
    """Physical distance to the nearest volume face.

    A cell close to a face can leave the field of view, which is a legitimate
    reason for a track to end and therefore evidence for the no-parent state.
    """

    @model_validator(mode="after")
    def isolation_and_distance_agree(self) -> InstanceFeatures:
        if self.is_isolated and self.nearest_neighbour_um != 0.0:
            raise ValueError("an isolated instance must record a sentinel nearest-neighbour of 0.0")
        if not self.is_isolated and self.neighbour_count == 0 and self.nearest_neighbour_um == 0.0:
            raise ValueError("a non-isolated instance cannot sit exactly on its nearest neighbour")
        return self


class RepresentationSet(BaseModel):
    """Features for every instance in one dataset, covering the set exactly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: DatasetIdentity
    features: tuple[InstanceFeatures, ...]

    @field_validator("features")
    @classmethod
    def require_features(cls, value: tuple[InstanceFeatures, ...]) -> tuple[InstanceFeatures, ...]:
        if not value:
            raise ValueError("an empty representation is a failure, not a result")
        return value

    @model_validator(mode="after")
    def unique_instances(self) -> RepresentationSet:
        ids = [item.instance_id for item in self.features]
        duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate feature rows: {duplicates}")
        return self

    def get(self, instance_id: int) -> InstanceFeatures:
        """Return one instance's features, or fail.

        Deliberately raises rather than returning a zero-filled default: a
        matcher that silently scores an instance it has no evidence for is
        scoring noise.
        """
        for item in self.features:
            if item.instance_id == instance_id:
                return item
        raise KeyError(f"no representation for instance {instance_id}")

    def covers(self, instance_ids: frozenset[int]) -> bool:
        return {item.instance_id for item in self.features} >= instance_ids
