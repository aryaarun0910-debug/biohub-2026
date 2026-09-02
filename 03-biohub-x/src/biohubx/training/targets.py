"""Supervision for a detector trained on sparse annotations.

The published CC0 objective builds its target as ``torch.zeros_like(logits)``,
marks the annotated node voxels positive, and calls every other voxel a
negative. Measured against this corpus that means ``44b6_0c582fdc`` supervises
71 cells as positive and 27,887 visually identical cells as background (R-0005).
AGENTS.md section 5 forbids it: unlabelled regions are respected, never treated
as background.

The obvious repair was an intensity band, calling only dim voxels background.
E03-MASK-AUDIT measured that band at every annotated node in the corpus and
falsified it ([[F-0020]]). No quantile covers every annotated cell on 6bba; the
tightest one that covers 44b6 withholds half the volume from supervision and
still misfires on the held-out embryo. Max-pooling instead of striding does not
rescue it, so the dim annotated cells are real and not a grid artefact.

So there is no negative class. Voxels are positive or unlabelled, and the
unlabelled ones are modelled as a mixture whose positive share is the dataset's
own recorded cell estimate. The band survives here only as the audit that
rejected it, because a rejected rule is worth being able to re-measure.

Consumers: ``biohubx train preflight`` and ``biohubx evaluate mask-audit``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from biohubx.contracts.lineage import LineageGraph

CANDIDATE_BANDS: tuple[float, ...] = (0.50, 0.70, 0.80, 0.90, 0.95, 0.99)
"""Bands the audit evaluates. Higher means fewer voxels withheld and more risk."""


class TargetConstructionError(ValueError):
    """Supervision could not be built from this window."""


class PriorError(ValueError):
    """The class prior is not a usable probability."""


# --- the audit that rejected the band ---------------------------------------


@dataclass(frozen=True, slots=True)
class BandReport:
    """One candidate band, measured against one population of annotated nodes."""

    quantile: float
    nodes: int
    nodes_in_candidate_band: int
    contradicted_nodes: int
    worst_node_percentile: float
    voxel_fraction_candidate: float

    @property
    def coverage(self) -> float:
        return self.nodes_in_candidate_band / self.nodes if self.nodes else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantile": self.quantile,
            "nodes": self.nodes,
            "nodes_in_candidate_band": self.nodes_in_candidate_band,
            "contradicted_nodes": self.contradicted_nodes,
            "coverage": self.coverage,
            "worst_node_percentile": self.worst_node_percentile,
            "voxel_fraction_candidate": self.voxel_fraction_candidate,
            "voxel_fraction_confident_background": 1.0 - self.voxel_fraction_candidate,
        }


def report_bands(
    node_percentiles: np.ndarray, bands: tuple[float, ...] = CANDIDATE_BANDS
) -> list[BandReport]:
    """Measure each band against annotated nodes' own intensity percentile ranks.

    A node below the band is a node that band would have called background. That
    is the falsifier for calling anything confident background, counted rather
    than argued about.
    """
    ranks = np.asarray(node_percentiles, dtype=np.float64)
    if ranks.size == 0:
        raise TargetConstructionError("no annotated node percentiles to report on")
    reports: list[BandReport] = []
    for quantile in bands:
        inside = int((ranks >= quantile).sum())
        reports.append(
            BandReport(
                quantile=quantile,
                nodes=int(ranks.size),
                nodes_in_candidate_band=inside,
                contradicted_nodes=int(ranks.size) - inside,
                worst_node_percentile=float(ranks.min()),
                # A quantile band leaves exactly 1 - q of voxels above it.
                voxel_fraction_candidate=1.0 - quantile,
            )
        )
    return reports


def select_band(
    training_percentiles: np.ndarray,
    bands: tuple[float, ...] = CANDIDATE_BANDS,
    *,
    required_coverage: float = 1.0,
) -> float | None:
    """Pick the tightest band a fold's own training annotations permit, or None.

    None is the answer that matters: it means no band can be justified and the
    negative class must go, rather than the band being widened until it stops
    contradicting anything.

    Only ever called with a fold's TRAINING embryo. Passing the evaluation
    embryo here would let held-out data choose the rule.
    """
    chosen: float | None = None
    for report in report_bands(training_percentiles, bands):
        if report.coverage >= required_coverage:
            chosen = report.quantile if chosen is None else max(chosen, report.quantile)
    return chosen


# --- what E03 actually trains against ---------------------------------------


def class_prior_from_estimate(estimated_total_nodes: float, grid_voxels: int) -> float:
    """The fraction of grid voxels expected to be a cell centre.

    Read from the dataset's own ``estimated_number_of_nodes``, official metadata
    rather than a tuned constant, over the voxels of the grid the model predicts
    on. It counts every cell, annotated or not, because that is what the
    detector is asked to find; which of them the metric scores is a later
    question for the retention filter ([[F-0018]]), not for the loss.
    """
    if grid_voxels <= 0:
        raise PriorError(f"grid must have voxels, got {grid_voxels}")
    prior = float(estimated_total_nodes) / float(grid_voxels)
    if not 0.0 < prior < 1.0:
        raise PriorError(
            f"class prior {prior} is not a probability; "
            f"{estimated_total_nodes} cells over {grid_voxels} voxels"
        )
    return prior


def positive_mask(
    grid_shape: tuple[int, ...],
    annotated: LineageGraph,
    *,
    downsample: tuple[int, ...],
) -> tuple[torch.Tensor, int]:
    """Just the annotated node voxels. Everything else is unlabelled, not background."""
    dz, dy, dx = (int(d) for d in downsample)
    mask = torch.zeros(grid_shape, dtype=torch.bool)
    placed = 0
    for node in annotated.nodes:
        z, y, x = int(node.voxel.z) // dz, int(node.voxel.y) // dy, int(node.voxel.x) // dx
        if not 0 <= node.frame < grid_shape[0]:
            continue
        if not (0 <= z < grid_shape[1] and 0 <= y < grid_shape[2] and 0 <= x < grid_shape[3]):
            continue
        mask[node.frame, z, y, x] = True
        placed += 1
    if placed == 0:
        raise TargetConstructionError("no annotated node landed inside the grid")
    return mask, placed


def positive_unlabelled_loss(
    logits: torch.Tensor,
    positive: torch.Tensor,
    *,
    prior: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Non-negative positive-unlabelled risk. No voxel is ever called background.

    Returns the loss and its three risk terms. The clamp firing is the signal
    that the model has started to memorise the positives, and it belongs in a
    log rather than being discovered afterwards.
    """
    if logits.shape != positive.shape:
        raise TargetConstructionError(
            f"logits {tuple(logits.shape)} and positives {tuple(positive.shape)} differ"
        )
    if not 0.0 < prior < 1.0:
        raise PriorError(f"class prior must lie in (0, 1), got {prior}")
    mask = positive.bool()
    n_pos = int(mask.sum())
    if n_pos == 0:
        raise TargetConstructionError("a positive-unlabelled risk needs at least one positive")
    unlabelled = ~mask
    if int(unlabelled.sum()) == 0:
        raise TargetConstructionError("every voxel is positive; there is nothing to contrast against")

    softplus = torch.nn.functional.softplus
    risk_positive = softplus(-logits[mask]).mean()
    risk_positive_as_negative = softplus(logits[mask]).mean()
    risk_unlabelled_as_negative = softplus(logits[unlabelled]).mean()

    negative_risk = risk_unlabelled_as_negative - prior * risk_positive_as_negative
    loss = prior * risk_positive + torch.clamp(negative_risk, min=0.0)
    return loss, {
        "prior": prior,
        "positives": float(n_pos),
        "risk_positive": float(risk_positive.item()),
        "risk_unlabelled_as_negative": float(risk_unlabelled_as_negative.item()),
        "negative_risk": float(negative_risk.item()),
        "negative_risk_clamped": float(negative_risk < 0),
    }
