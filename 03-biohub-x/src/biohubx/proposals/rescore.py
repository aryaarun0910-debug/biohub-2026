"""Candidate re-scoring: keep every strict per-scale DoG peak and learn which ones are cells.

The 6bba miss atlas (R-0022) found that 587 of 706 unreached cells sit within
7 micrometres of a strict per-scale maximum the frame-level response quantile
discarded, while the same source proposes a third more cells than the embryo
holds. The lever is therefore ranking, not detection: take the complete peak
set as candidates, score each from a small patch of image and per-scale
response around it, optionally with the neighbouring frames, and keep by score
with the count tied to the movie's own estimate. Supervision is positive and
unlabelled only, as everywhere in Biohub-X: a candidate matched one-to-one to
an annotated cell is positive, every other candidate is unlabelled, and no
candidate is ever called a negative.

Two arms share everything but their input channels. ``A3`` sees intensity and
the two per-scale responses of the candidate's own frame. ``A4`` adds the
intensity of the previous and following frames at the same grid location.

Consumers: ``biohubx model inspect --arm``, ``biohubx train probe``, and the E07
family that scores the kept candidates through the oracle ceiling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE, VoxelScaleZYX
from biohubx.contracts.instances import CandidateInstance, InstanceSet
from biohubx.contracts.lineage import DatasetIdentity, LineageGraph
from biohubx.evaluation.oracle import match_one_to_one
from biohubx.proposals import dog

DEFAULT_POOL_QUANTILE = 0.80
"""The response quantile that keeps every annotated cell in the pool on the pilot
(node match 1.000 on 44b6, 0.999 on 6bba, 12 movies each) at 3.4 and 14.7 times the
estimated count; 0.50 adds candidates and no cells. Configured per family and checked
per embryo on its own rows, so no held-out number selects it."""

PATCH_HALF = 4
"""Half-width in isotropic grid voxels: a 9^3 patch at 1.625 um spans 14.6 um,
twice the official match radius, so a candidate's neighbours are in view."""

ARMS: dict[str, dict[str, Any]] = {
    "A3": {
        "temporal": False,
        "channels": 3,
        "description": "intensity + per-scale DoG responses of the candidate's frame",
    },
    "A4": {
        "temporal": True,
        "channels": 5,
        "description": "A3 plus the previous and following frames' intensity at the same location",
    },
}


class RescoreError(ValueError):
    """The re-scorer cannot be built or fed as asked."""


class PatchScorer(nn.Module):
    """A small 3D convolutional scorer over one patch per candidate: logits, one per candidate."""

    def __init__(self, channels: int, width: int = 16) -> None:
        super().__init__()
        self.channels = channels
        self.features = nn.Sequential(
            nn.Conv3d(channels, width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(width, 2 * width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(2 * width, 2 * width, kernel_size=3, padding=0),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(2 * width, 1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 5 or patches.shape[1] != self.channels:
            raise RescoreError(f"expected (N, {self.channels}, D, H, W) patches, got {tuple(patches.shape)}")
        pooled = self.pool(self.features(patches)).flatten(1)
        logits: torch.Tensor = self.head(pooled).squeeze(1)
        return logits


def build_scorer(arm: str, *, seed: int = 0) -> PatchScorer:
    if arm not in ARMS:
        raise RescoreError(f"unknown arm {arm!r}; known: {sorted(ARMS)}")
    torch.manual_seed(seed)
    return PatchScorer(int(ARMS[arm]["channels"]))


def extract_candidates(
    volume: np.ndarray,
    *,
    dataset: DatasetIdentity,
    radii_um: tuple[float, ...] = (2.0, 3.0),
    suppression_radius_um: float = 4.0,
    response_quantile: float = DEFAULT_POOL_QUANTILE,
    scale: VoxelScaleZYX = OFFICIAL_VOXEL_SCALE,
) -> InstanceSet:
    """Every strict per-scale peak, deduplicated by the shared physical suppression."""
    return dog.detect_instances(
        volume,
        dataset=dataset,
        radii_um=radii_um,
        response_quantile=response_quantile,
        suppression_radius_um=suppression_radius_um,
        local_maxima_only=True,
        per_scale_union=True,
        scale=scale,
    )


@dataclass
class FrameGrids:
    """The isotropic grids one frame's patches are cut from."""

    intensity: np.ndarray
    responses: list[np.ndarray]


def frame_grids(
    volume: np.ndarray,
    frame: int,
    *,
    radii_um: tuple[float, ...],
    scale: VoxelScaleZYX = OFFICIAL_VOXEL_SCALE,
) -> FrameGrids:
    stride = dog.isotropic_plane_stride(scale)
    plane = volume[frame][:, ::stride, ::stride]
    normalised = dog.normalise_frame(plane)
    responses = [dog.dog_response(normalised, radii_um=(r,), voxel_um=float(scale.z_um)) for r in radii_um]
    return FrameGrids(
        intensity=normalised.astype(np.float32), responses=[r.astype(np.float32) for r in responses]
    )


def _cut(grid: np.ndarray, centre: tuple[int, int, int], half: int) -> np.ndarray:
    """A (2h+1)^3 patch around ``centre``, zero-padded where the grid ends."""
    out = np.zeros((2 * half + 1,) * 3, dtype=np.float32)
    lo = [c - half for c in centre]
    hi = [c + half + 1 for c in centre]
    src = tuple(slice(max(0, lo[i]), min(grid.shape[i], hi[i])) for i in range(3))
    dst = tuple(
        slice(src[i].start - lo[i], src[i].start - lo[i] + (src[i].stop - src[i].start)) for i in range(3)
    )
    out[dst] = grid[src]
    return out


def candidate_patches(
    volume: np.ndarray,
    instances: InstanceSet,
    *,
    arm: str,
    radii_um: tuple[float, ...] = (2.0, 3.0),
    half: int = PATCH_HALF,
    scale: VoxelScaleZYX = OFFICIAL_VOXEL_SCALE,
    standardise: bool = True,
) -> torch.Tensor:
    """One patch per candidate, channels as the arm declares, in instance order, standardised per channel."""
    if arm not in ARMS:
        raise RescoreError(f"unknown arm {arm!r}")
    temporal = bool(ARMS[arm]["temporal"])
    stride = dog.isotropic_plane_stride(scale)
    frames = volume.shape[0]
    grids: dict[int, FrameGrids] = {}

    def grid_for(frame: int) -> FrameGrids:
        if frame not in grids:
            grids[frame] = frame_grids(volume, frame, radii_um=radii_um, scale=scale)
        return grids[frame]

    patches = []
    for instance in instances.instances:
        centre = (
            round(instance.voxel.z),
            round(instance.voxel.y) // stride,
            round(instance.voxel.x) // stride,
        )
        own = grid_for(instance.frame)
        channels = [_cut(own.intensity, centre, half)] + [_cut(r, centre, half) for r in own.responses]
        if temporal:
            for neighbour in (instance.frame - 1, instance.frame + 1):
                clamped = min(max(neighbour, 0), frames - 1)
                channels.append(_cut(grid_for(clamped).intensity, centre, half))
        patches.append(np.stack(channels))
    if not patches:
        return torch.zeros((0, int(ARMS[arm]["channels"]), 2 * half + 1, 2 * half + 1, 2 * half + 1))
    stacked = np.stack(patches)
    if standardise:
        # Per channel over this window's own pool. The DoG responses are two orders
        # of magnitude smaller than normalised intensity, and Stage 0 measured the
        # scorer barely moving with them at initialisation; a scale the network
        # can see is a preprocessing choice, recorded in the family config, not a
        # learned constant and not anything read from the held-out embryo.
        mean = stacked.mean(axis=(0, 2, 3, 4), keepdims=True)
        std = stacked.std(axis=(0, 2, 3, 4), keepdims=True) + 1e-6
        stacked = (stacked - mean) / std
    return torch.from_numpy(stacked.astype(np.float32))


def candidate_labels(instances: InstanceSet, annotated: LineageGraph) -> torch.Tensor:
    """Positive where a candidate is the one-to-one match of an annotated cell; everything else unlabelled."""
    matched = set(match_one_to_one(instances, annotated).values())
    return torch.tensor(
        [instance.instance_id in matched for instance in instances.instances], dtype=torch.bool
    )


def candidate_prior(estimated_nodes: float, candidates: int) -> float:
    """The fraction of candidates expected to be cells: the movie's own estimate over the pool size."""
    if candidates <= 0:
        raise RescoreError("no candidates, so no prior")
    return float(min(max(estimated_nodes / candidates, 1e-4), 1.0 - 1e-4))


def keep_by_threshold(instances: InstanceSet, logits: torch.Tensor, threshold: float) -> InstanceSet:
    """The candidates whose score clears the threshold, renumbered densely, confidence = probability."""
    probabilities = torch.sigmoid(logits.detach().float()).cpu().numpy()
    kept: list[CandidateInstance] = []
    for instance, p in zip(instances.instances, probabilities, strict=True):
        if p >= threshold:
            kept.append(instance.model_copy(update={"instance_id": len(kept), "confidence": float(p)}))
    return InstanceSet(dataset=instances.dataset, instances=tuple(kept))


def keep_top(instances: InstanceSet, logits: torch.Tensor, count: int) -> InstanceSet:
    """The ``count`` highest-scoring candidates, renumbered densely, confidence = probability.

    Ties break on instance identity so two runs keep the same set. ``count`` is
    round(ratio x the movie's own official estimate), never anything read from
    ground truth.
    """
    probabilities = torch.sigmoid(logits.detach().float()).cpu().numpy()
    order = sorted(range(len(instances.instances)), key=lambda i: (-float(probabilities[i]), i))[
        : max(count, 0)
    ]
    kept: list[CandidateInstance] = []
    for index in sorted(order):
        instance = instances.instances[index]
        kept.append(
            instance.model_copy(update={"instance_id": len(kept), "confidence": float(probabilities[index])})
        )
    return InstanceSet(dataset=instances.dataset, instances=tuple(kept))


def threshold_for_count(logits: torch.Tensor, target_count: int) -> float:
    """The probability that keeps exactly ``target_count`` candidates: the training-embryo constant.

    Chosen on the training embryo only, so the held-out count is what the
    scorer produces, never a truncation against held-out ground truth.
    """
    probabilities = torch.sigmoid(logits.detach().float()).cpu().numpy()
    if target_count <= 0:
        return 1.0
    if target_count >= probabilities.size:
        return float(probabilities.min()) if probabilities.size else 0.0
    ordered = np.sort(probabilities)[::-1]
    return float(ordered[target_count - 1])


def describe(arm: str) -> dict[str, Any]:
    spec = dict(ARMS[arm])
    spec.update({"arm": arm, "patch_half": PATCH_HALF, "default_pool_quantile": DEFAULT_POOL_QUANTILE})
    return spec


def batches(tensor: torch.Tensor, size: int) -> Sequence[torch.Tensor]:
    return [tensor[i : i + size] for i in range(0, tensor.shape[0], size)]
