"""A detection target that refuses to call an unlabelled cell background.

The published CC0 objective builds its target as ``torch.zeros_like(logits)``,
marks the annotated node voxels positive, and lightly penalises every other
voxel. It has no ignore class. Measured against the real corpus that means the
44b6 movie ``44b6_0c582fdc`` supervises 71 cells as positive and 27,887 visually
identical cells as background, and the ratio differs by two orders of magnitude
between the embryos, so the two folds would not even be solving the same
problem. AGENTS.md section 5 forbids this directly: unlabelled and ignore
regions are respected, never treated as background.

So the target here is three-valued. Positives are the annotated node voxels.
Negatives are voxels the image itself says are empty. Everything bright but
unlabelled is ignored, because that is where the unannotated cells are and
Biohub-X cannot tell which of them is one.

The band is chosen by intensity quantile from the volume being trained on, not
by a tuned constant. It is checkable, and the check is the reason to believe it:
at the ninetieth percentile the band covers about a tenth of voxels and contains
every annotated node in the measured sample. What it costs is supervision on
bright non-cell structure, which the published objective did get. That is the
trade, and it is taken deliberately.

Consumer: ``biohubx train preflight``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from biohubx.contracts.lineage import LineageGraph

POSITIVE = 1
IGNORE = -1
NEGATIVE = 0

DEFAULT_IGNORE_QUANTILE = 0.90
"""Everything above this intensity is unlabelled-cell territory, not background."""


class TargetConstructionError(ValueError):
    """The target could not be built without violating the ignore contract."""


@dataclass(frozen=True, slots=True)
class DetectionTarget:
    """Three-valued supervision, with the counts that make it auditable."""

    labels: torch.Tensor
    """(T, Z, Y, X) int8, one of POSITIVE, NEGATIVE or IGNORE."""

    ignore_threshold: float
    ignore_quantile: float
    positives: int
    negatives: int
    ignored: int
    annotated_nodes: int
    unplaceable_nodes: int

    def to_dict(self) -> dict[str, Any]:
        total = self.positives + self.negatives + self.ignored
        return {
            "ignore_quantile": self.ignore_quantile,
            "ignore_threshold": self.ignore_threshold,
            "positive_voxels": self.positives,
            "negative_voxels": self.negatives,
            "ignored_voxels": self.ignored,
            "total_voxels": total,
            "ignored_fraction": self.ignored / total if total else 0.0,
            "annotated_nodes": self.annotated_nodes,
            "unplaceable_nodes": self.unplaceable_nodes,
        }


def build_detection_target(
    volume: np.ndarray,
    annotated: LineageGraph,
    *,
    downsample: tuple[int, ...],
    ignore_quantile: float = DEFAULT_IGNORE_QUANTILE,
) -> DetectionTarget:
    """Label every voxel positive, negative or ignored.

    ``volume`` is the normalised ``(T, Z, Y, X)`` window at full resolution;
    ``annotated`` carries window-relative coordinates. Both are reduced to the
    model's grid by the same striding the reference uses, so a node and its
    voxel cannot end up on different grids.
    """
    if volume.ndim != 4:
        raise TargetConstructionError(f"expected a (T, Z, Y, X) volume, got shape {volume.shape}")
    if not 0.0 < ignore_quantile < 1.0:
        raise TargetConstructionError(f"ignore quantile must lie in (0, 1), got {ignore_quantile}")

    dz, dy, dx = (int(d) for d in downsample)
    grid = volume[:, ::dz, ::dy, ::dx]
    threshold = float(np.quantile(grid, ignore_quantile))

    labels = np.where(grid >= threshold, IGNORE, NEGATIVE).astype(np.int8)

    placed = 0
    for node in annotated.nodes:
        z, y, x = int(node.voxel.z) // dz, int(node.voxel.y) // dy, int(node.voxel.x) // dx
        if not (0 <= node.frame < grid.shape[0]):
            continue
        if not (0 <= z < grid.shape[1] and 0 <= y < grid.shape[2] and 0 <= x < grid.shape[3]):
            continue
        labels[node.frame, z, y, x] = POSITIVE
        placed += 1

    if placed == 0:
        raise TargetConstructionError(
            "no annotated node landed inside the window's grid, so the target would be "
            "entirely negative and ignored; that is not something to train on"
        )

    positives = int((labels == POSITIVE).sum())
    ignored = int((labels == IGNORE).sum())
    negatives = int((labels == NEGATIVE).sum())
    return DetectionTarget(
        labels=torch.from_numpy(labels),
        ignore_threshold=threshold,
        ignore_quantile=ignore_quantile,
        positives=positives,
        negatives=negatives,
        ignored=ignored,
        annotated_nodes=len(annotated.nodes),
        unplaceable_nodes=len(annotated.nodes) - placed,
    )


def masked_detection_loss(logits: torch.Tensor, target: DetectionTarget) -> torch.Tensor:
    """Balanced BCE over the supervised voxels only.

    Ignored voxels contribute no loss and no gradient. Positives and negatives
    are each normalised by their own count, so the enormous negative majority
    cannot drown the handful of annotated cells, and changing the ignore
    quantile does not silently rescale the objective.
    """
    labels = target.labels.to(logits.device)
    if logits.shape != labels.shape:
        raise TargetConstructionError(
            f"logits {tuple(logits.shape)} and target {tuple(labels.shape)} describe different grids"
        )
    positive = labels == POSITIVE
    negative = labels == NEGATIVE
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        raise TargetConstructionError(
            f"a balanced loss needs both classes, got {n_pos} positive and {n_neg} negative voxels"
        )

    weight = torch.zeros_like(logits)
    weight[positive] = 1.0 / n_pos
    weight[negative] = 1.0 / n_neg
    supervised = torch.where(positive, torch.ones_like(logits), torch.zeros_like(logits))
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, supervised, weight=weight, reduction="sum"
    )
