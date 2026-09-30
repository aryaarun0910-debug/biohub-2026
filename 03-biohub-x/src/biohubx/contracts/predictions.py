"""Association prediction contract, with abstention as a first-class option.

Every target carries one score per offered parent and one score for having no
parent at all. The no-parent score is a row in the same table, competing on the
same axis, not a threshold applied afterwards to the best edge. A prediction
that omits it is refused.

The distinction matters because a threshold cannot express "this cell is new".
It can only express "the best parent is not good enough", which is a different
claim and one that collapses at exactly the moments a birth is most likely: a
cell entering the field of view often has a plausible-looking neighbour.

Phase 2 fills these scores with an untrained deterministic rule. The structure
is what is being fixed here; the weights are not evidence of anything.

Consumers: :mod:`biohubx.tracking.matcher`, :mod:`biohubx.tracking.graph_decoder`.
"""

from __future__ import annotations

from collections import Counter
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from biohubx.contracts.lineage import DatasetIdentity

Identity = Annotated[StrictInt, Field(ge=0)]
Frame = Annotated[StrictInt, Field(ge=0)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

NO_PARENT = "no_parent"
"""The name of the abstention option, used in reports and manifests."""


class ParentScore(BaseModel):
    """One offered parent and the evidence for it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Identity
    score: FiniteFloat


class TargetPrediction(BaseModel):
    """What the matcher believes about one target's parentage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: Identity
    target_frame: Frame
    parents: tuple[ParentScore, ...]
    no_parent_score: FiniteFloat
    """Scored from the target's own evidence, never derived from the parents."""

    @model_validator(mode="after")
    def no_duplicate_parents(self) -> TargetPrediction:
        sources = [parent.source for parent in self.parents]
        duplicates = sorted(key for key, count in Counter(sources).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate parent rows for target {self.target}: {duplicates}")
        if self.target in sources:
            raise ValueError(f"target {self.target} is offered as its own parent")
        return self

    @property
    def best(self) -> ParentScore | None:
        """The highest-scoring parent, or None when abstention wins.

        Abstention wins ties. A tie means the evidence does not distinguish a
        link from a birth, and inventing a link in that case is the failure mode
        the no-parent row exists to prevent.
        """
        if not self.parents:
            return None
        best = max(self.parents, key=lambda parent: (parent.score, -parent.source))
        return best if best.score > self.no_parent_score else None

    @property
    def margin(self) -> float:
        """Top-1 minus top-2 over all options including abstention.

        A small margin is the ambiguity signal a cascade would route on. It is
        reported here rather than computed by each consumer so that everyone
        means the same thing by it.
        """
        options = sorted([parent.score for parent in self.parents] + [self.no_parent_score], reverse=True)
        if len(options) == 1:
            raise ValueError("margin is undefined with a single option; do not report a default")
        return options[0] - options[1]


class AssociationPrediction(BaseModel):
    """Predictions for every target the candidate graph offered."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: DatasetIdentity
    targets: tuple[TargetPrediction, ...]

    @field_validator("targets")
    @classmethod
    def require_targets(cls, value: tuple[TargetPrediction, ...]) -> tuple[TargetPrediction, ...]:
        if not value:
            raise ValueError("an empty prediction is a failure, not a result")
        return value

    @model_validator(mode="after")
    def unique_targets(self) -> AssociationPrediction:
        ids = [target.target for target in self.targets]
        duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate target predictions: {duplicates}")
        return self

    def covers(self, targets: frozenset[int]) -> bool:
        """Whether every offered target received a prediction.

        A target the matcher skipped is not an abstention. It is a gap, and the
        decoder must be able to tell the two apart.
        """
        return {target.target for target in self.targets} >= targets
