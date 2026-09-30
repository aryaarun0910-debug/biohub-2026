"""A deterministic, untrained matcher with an explicit no-parent option.

What is being fixed here is the SHAPE of the prediction, not its quality. Each
target receives one score per offered parent and one score for having no parent,
all on the same axis, and the no-parent score is computed from the target's own
evidence rather than derived from how good its best parent looked. A threshold
on the best edge cannot express "this cell is new"; it can only express "the best
parent is not good enough", and those two claims come apart exactly where births
happen.

The weights below are hand-chosen constants. They are not a result, they were
not fitted to anything, and no number produced with them is evidence about the
competition. They exist so the vertical slice runs end to end and so the learned
matcher that replaces them has a contract to satisfy and a baseline to beat.

Consumer: :mod:`biohubx.tracking.graph_decoder`.
"""

from __future__ import annotations

from dataclasses import dataclass

from biohubx.contracts.candidates import CandidateGraph
from biohubx.contracts.predictions import (
    AssociationPrediction,
    ParentScore,
    TargetPrediction,
)
from biohubx.contracts.representation import RepresentationSet


@dataclass(frozen=True, slots=True)
class MatcherWeights:
    """Hand-set constants for the untrained matcher.

    Named and grouped so that the learned matcher's first job is to make this
    object unnecessary, and so that nobody mistakes these for fitted values.
    """

    displacement_um_penalty: float = 0.35
    """Score lost per micrometre of physical displacement."""

    intensity_change_penalty: float = 1.0
    """Score lost per unit of normalised peak-intensity change."""

    birth_base: float = -2.4
    """Baseline plausibility of a target having no parent at all."""

    birth_border_bonus_per_um: float = 0.20
    """Added to the no-parent score as a target sits closer to a volume face.

    A cell near the edge of the field of view may have entered it, which is
    positive evidence for a birth and is exactly the evidence a threshold on the
    best edge has no way to use.
    """

    birth_border_reference_um: float = 8.0
    """Distance beyond which proximity to a face stops counting as evidence."""


UNTRAINED_WEIGHTS = MatcherWeights()
"""The single shared default. Named so that a caller relying on it is visible."""


def score_candidates(
    candidates: CandidateGraph,
    representation: RepresentationSet,
    *,
    weights: MatcherWeights = UNTRAINED_WEIGHTS,
) -> AssociationPrediction:
    """Score every offered parent and the no-parent option for every target.

    Fails rather than guessing when the representation does not cover an
    instance the candidate graph refers to: a matcher scoring an instance it has
    no evidence for is scoring noise.
    """
    referenced = frozenset(
        {edge.source for edge in candidates.edges} | {edge.target for edge in candidates.edges}
    )
    if not representation.covers(referenced):
        missing = sorted(referenced - {item.instance_id for item in representation.features})
        raise ValueError(f"representation does not cover candidate instances: {missing}")

    predictions: list[TargetPrediction] = []
    for target in candidates.targets:
        target_features = representation.get(target)
        parents = tuple(
            ParentScore(
                source=edge.source,
                score=_link_score(
                    displacement_um=edge.displacement_um,
                    source_intensity=representation.get(edge.source).peak_intensity,
                    target_intensity=target_features.peak_intensity,
                    weights=weights,
                ),
            )
            for edge in candidates.parents_of(target)
        )
        target_frame = candidates.parents_of(target)[0].target_frame
        predictions.append(
            TargetPrediction(
                target=target,
                target_frame=target_frame,
                parents=parents,
                no_parent_score=_no_parent_score(
                    border_distance_um=target_features.border_distance_um, weights=weights
                ),
            )
        )
    return AssociationPrediction(dataset=candidates.dataset, targets=tuple(predictions))


def _link_score(
    *,
    displacement_um: float,
    source_intensity: float,
    target_intensity: float,
    weights: MatcherWeights,
) -> float:
    """Evidence that this parent explains this target.

    Physical displacement dominates, with an intensity-consistency term so the
    score is not a pure distance gate. Both are measured, neither is defaulted.
    """
    return -(
        weights.displacement_um_penalty * displacement_um
        + weights.intensity_change_penalty * abs(source_intensity - target_intensity)
    )


def _no_parent_score(*, border_distance_um: float, weights: MatcherWeights) -> float:
    """Evidence that this target has no parent, from the target alone.

    Deliberately independent of the parent scores. It is computed from where the
    target sits, so a cell at the edge of the volume can win the no-parent option
    even when a plausible-looking parent is nearby, and a cell in the middle
    needs its parents to be genuinely bad before abstention wins.
    """
    proximity = max(0.0, weights.birth_border_reference_um - border_distance_um)
    return weights.birth_base + weights.birth_border_bonus_per_um * proximity
