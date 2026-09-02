"""Sparse temporal candidate graph contract.

Candidate generation and candidate scoring are separate contracts on purpose. A
scorer cannot be blamed for an edge it was never offered, so this module also
carries the reach report: the fraction of true edges that survived into the
candidate set. Reach is the ceiling every downstream number sits under, and it
has to be measured rather than assumed.

Consumers: :mod:`biohubx.tracking.candidate_graph`, :mod:`biohubx.tracking.matcher`.
"""

from __future__ import annotations

from collections import Counter
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from biohubx.contracts.lineage import DatasetIdentity

Identity = Annotated[StrictInt, Field(ge=0)]
Frame = Annotated[StrictInt, Field(ge=0)]
NonNegative = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]


class CandidateEdge(BaseModel):
    """One offered parent-to-target hypothesis across exactly one frame."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Identity
    target: Identity
    source_frame: Frame
    target_frame: Frame
    displacement_um: NonNegative
    """Physical separation of the two instances, in micrometres."""

    @model_validator(mode="after")
    def advances_exactly_one_frame(self) -> CandidateEdge:
        if self.target_frame != self.source_frame + 1:
            raise ValueError(
                f"a candidate edge must advance exactly one frame: {self.source_frame} -> {self.target_frame}"
            )
        if self.source == self.target:
            raise ValueError("an instance cannot be its own parent")
        return self


class ReachReport(BaseModel):
    """How much of the true answer the candidate set still contains.

    Reported for the frames the graph actually spans. A true edge whose endpoints
    were never detected is counted as unreachable rather than excused, because
    the detector's miss is still a miss from the graph's point of view.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    true_edges: Annotated[StrictInt, Field(ge=0)]
    reachable_true_edges: Annotated[StrictInt, Field(ge=0)]
    unreachable_missing_endpoint: Annotated[StrictInt, Field(ge=0)]
    unreachable_outside_radius: Annotated[StrictInt, Field(ge=0)]

    @model_validator(mode="after")
    def counts_are_consistent(self) -> ReachReport:
        accounted = (
            self.reachable_true_edges + self.unreachable_missing_endpoint + self.unreachable_outside_radius
        )
        if accounted != self.true_edges:
            raise ValueError(
                f"reach accounting does not close: {accounted} accounted for {self.true_edges} true edges"
            )
        return self

    @property
    def reach(self) -> float:
        """Fraction of true edges the scorer will get the chance to choose."""
        if self.true_edges == 0:
            raise ValueError("reach is undefined with no true edges; do not report a default")
        return self.reachable_true_edges / self.true_edges


class CandidateGraph(BaseModel):
    """Every edge the matcher will be offered, and nothing it will not."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: DatasetIdentity
    radius_um: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    edges: tuple[CandidateEdge, ...]
    reach: ReachReport | None = None
    """Present only when a ground truth was supplied. Absent is not zero."""

    @field_validator("edges")
    @classmethod
    def require_edges(cls, value: tuple[CandidateEdge, ...]) -> tuple[CandidateEdge, ...]:
        if not value:
            raise ValueError("an empty candidate graph offers the matcher nothing; this is a failure")
        return value

    @model_validator(mode="after")
    def no_duplicate_pairs(self) -> CandidateGraph:
        pairs = [(edge.source, edge.target) for edge in self.edges]
        duplicates = sorted(pair for pair, count in Counter(pairs).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate candidate pairs: {duplicates}")
        for edge in self.edges:
            if edge.displacement_um > self.radius_um:
                raise ValueError(
                    f"candidate {edge.source}->{edge.target} at {edge.displacement_um} um "
                    f"exceeds the declared radius of {self.radius_um} um"
                )
        return self

    def parents_of(self, target: int) -> tuple[CandidateEdge, ...]:
        return tuple(
            sorted(
                (edge for edge in self.edges if edge.target == target),
                key=lambda edge: edge.source,
            )
        )

    @property
    def targets(self) -> tuple[int, ...]:
        return tuple(sorted({edge.target for edge in self.edges}))
