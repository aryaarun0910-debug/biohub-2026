"""Training-side adapters for the Biohub-X T=2 sanity system.

This module is intentionally narrower than a production trainer.  It turns the committed
``biohubx_synthetic_t2_v1`` point substrate into the sparse tensors consumed by
``BiohubXMatcher`` and defines the supervised objective.  It does not claim that point-derived
features replace FOCUS-3D masks; the real-mask adapter remains a separate pre-GPU gate.

The key invariant is that source dropout is physical.  Dropped sources are removed from the
model input and candidate indices are remapped to the compact visible-source array.  Leaving a
dropped source in the token set while merely hiding its positive edge would leak the answer into
the contextual encoder and make the abstention task fictitious.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .contract import CLASSES
from .model import MatcherOutput


@dataclass(frozen=True)
class PreparedT2Batch:
    """Compact, identity-free tensors for one synthetic T=2 matcher call."""

    source_features: Tensor
    target_features: Tensor
    source_positions_um: Tensor
    target_positions_um: Tensor
    candidate_source_index: Tensor
    candidate_target_index: Tensor
    candidate_is_positive: Tensor
    no_parent_label: Tensor
    visible_source_row: np.ndarray

    def to(self, device: torch.device | str) -> "PreparedT2Batch":
        return PreparedT2Batch(
            source_features=self.source_features.to(device),
            target_features=self.target_features.to(device),
            source_positions_um=self.source_positions_um.to(device),
            target_positions_um=self.target_positions_um.to(device),
            candidate_source_index=self.candidate_source_index.to(device),
            candidate_target_index=self.candidate_target_index.to(device),
            candidate_is_positive=self.candidate_is_positive.to(device),
            no_parent_label=self.no_parent_label.to(device),
            visible_source_row=self.visible_source_row,
        )


@dataclass(frozen=True)
class T2Loss:
    total: Tensor
    pair: Tensor
    no_parent: Tensor
    n_positive_pairs: int
    n_negative_pairs: int
    n_abstentions: int


def _require(batch: Mapping[str, object], name: str) -> np.ndarray:
    if name not in batch:
        raise ValueError(f"synthetic T=2 batch is missing {name!r}")
    return np.asarray(batch[name])


def _identity_free_features(positions_um: np.ndarray) -> np.ndarray:
    """A deterministic software-smoke representation with no lineage or row identifier.

    These eight functions are deliberately non-constant on ordinary point clouds.  They exist to
    exercise the trainable matcher before a real FOCUS representation is connected; they are not
    a substitute for that representation and must not be reported as model evidence.
    """
    p = np.asarray(positions_um, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError(f"positions must be finite (N,3) physical coordinates, got {p.shape}")
    scale = np.maximum(np.std(p, axis=0, keepdims=True), 1e-6)
    centred = (p - np.mean(p, axis=0, keepdims=True)) / scale
    radius = np.linalg.norm(centred, axis=1, keepdims=True)
    cross = np.column_stack(
        (centred[:, 0] * centred[:, 1], centred[:, 1] * centred[:, 2],
         centred[:, 2] * centred[:, 0], np.sin(radius[:, 0]))
    )
    return np.concatenate((centred, radius, cross), axis=1).astype(np.float32)


def prepare_synthetic_t2(batch: Mapping[str, object]) -> PreparedT2Batch:
    """Remove dropped sources and remap the offered sparse candidate surface fail-closed."""
    if batch.get("schema") != "biohubx_synthetic_t2_v1" or batch.get("temporal_window") != 2:
        raise ValueError("prepare_synthetic_t2 requires the exact biohubx_synthetic_t2_v1 schema")
    source_um = _require(batch, "source_um_zyx").astype(np.float64)
    target_um = _require(batch, "target_um_zyx").astype(np.float64)
    visible = _require(batch, "source_visible")
    source_index = _require(batch, "candidate_source_index")
    target_index = _require(batch, "candidate_target_index")
    positive = _require(batch, "candidate_is_positive")
    no_parent = _require(batch, "no_parent_label")
    n = len(target_um)
    if source_um.shape != (n, 3) or visible.shape != (n,) or visible.dtype.kind != "b":
        raise ValueError("source positions and boolean visibility must align with every target")
    if target_um.shape != (n, 3) or no_parent.shape != (n,) or no_parent.dtype.kind != "b":
        raise ValueError("target positions and boolean no-parent labels must align")
    if source_index.dtype.kind not in "iu" or target_index.dtype.kind not in "iu":
        raise ValueError("candidate endpoint indices must be integer")
    if positive.dtype.kind != "b" or source_index.shape != target_index.shape \
            or source_index.shape != positive.shape:
        raise ValueError("candidate endpoints and boolean labels must have equal one-dimensional shape")
    if np.any(source_index < 0) or np.any(source_index >= n) \
            or np.any(target_index < 0) or np.any(target_index >= n):
        raise ValueError("candidate endpoint is out of bounds")
    if np.any(~visible[source_index]):
        raise ValueError("a candidate references a source that source dropout removed")

    visible_rows = np.flatnonzero(visible).astype(np.int64)
    old_to_new = np.full(n, -1, dtype=np.int64)
    old_to_new[visible_rows] = np.arange(len(visible_rows), dtype=np.int64)
    compact_source = old_to_new[source_index]
    if np.any(compact_source < 0):
        raise ValueError("source compaction produced an uncovered candidate")

    # Enforce the abstention semantics independently of the generator's own audit.
    for target in range(n):
        rows = target_index == target
        if int(positive[rows].sum()) != int(not no_parent[target]):
            raise ValueError(f"target {target} pair labels contradict its no-parent label")

    return PreparedT2Batch(
        source_features=torch.from_numpy(_identity_free_features(source_um[visible_rows])),
        target_features=torch.from_numpy(_identity_free_features(target_um)),
        source_positions_um=torch.from_numpy(source_um[visible_rows].astype(np.float32)),
        target_positions_um=torch.from_numpy(target_um.astype(np.float32)),
        candidate_source_index=torch.from_numpy(compact_source),
        candidate_target_index=torch.from_numpy(target_index.astype(np.int64, copy=False)),
        candidate_is_positive=torch.from_numpy(positive.astype(bool, copy=False)),
        no_parent_label=torch.from_numpy(no_parent.astype(np.float32, copy=False)),
        visible_source_row=visible_rows,
    )


def supervised_t2_loss(output: MatcherOutput, batch: PreparedT2Batch,
                       *, no_parent_weight: float = 1.0) -> T2Loss:
    """Continuation-vs-neither pair loss plus an independent learned abstention loss.

    The point generator contains no divisions, so it never fabricates division positives.  The
    division logit participates as a negative class here; real or mask-derived fork supervision
    must be added by a separately audited generator before any division-capability claim.
    """
    if no_parent_weight <= 0 or not np.isfinite(no_parent_weight):
        raise ValueError("no_parent_weight must be finite and positive")
    if output.candidate_source_index.shape != batch.candidate_source_index.shape \
            or not torch.equal(output.candidate_source_index, batch.candidate_source_index) \
            or not torch.equal(output.candidate_target_index, batch.candidate_target_index):
        raise ValueError("loss batch candidate surface does not match the model output")
    if output.no_parent_logits.shape != batch.no_parent_label.shape:
        raise ValueError("loss batch no-parent labels do not match the model output")

    labels = torch.full(
        (output.pair_logits.shape[0],), CLASSES.index("neither"),
        dtype=torch.long, device=output.pair_logits.device,
    )
    positive = batch.candidate_is_positive.to(output.pair_logits.device)
    labels[positive] = CLASSES.index("continuation")
    pair_loss = F.cross_entropy(output.pair_logits, labels)
    abstain_loss = F.binary_cross_entropy_with_logits(
        output.no_parent_logits,
        batch.no_parent_label.to(output.no_parent_logits.device),
    )
    total = pair_loss + float(no_parent_weight) * abstain_loss
    return T2Loss(
        total=total,
        pair=pair_loss,
        no_parent=abstain_loss,
        n_positive_pairs=int(positive.sum().item()),
        n_negative_pairs=int((~positive).sum().item()),
        n_abstentions=int(batch.no_parent_label.sum().item()),
    )


__all__ = [
    "PreparedT2Batch",
    "T2Loss",
    "prepare_synthetic_t2",
    "supervised_t2_loss",
]
