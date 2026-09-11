"""Stage 2 — refine: move nodes, and rank them for pruning.

    refine_coord = coord + dzyx        # sparse Kaggle supervision, at detected peaks only
    refine_logit = logit + dlogit      # a RANKER, not a threshold

Why this stage carries the most leverage in the detector (EXP-3, our own corpus): centroid
error costs -4.9% of edge Jaccard at sigma 2.0um but -20.6% at 2.5um and -41.9% at 3.0um. The
cliff sits inside half a micron. Its author measured error falling 1.18 -> 0.90-0.97 voxels, an
18-24% reduction, with 70-75% of points improved, 10-14% of peaks pruned, recall held at 99.5%.

GATE: ship only if centroid error falls >=15% against the un-refined detector at >=99.0%
recall. Otherwise keep the plain detector and spend the time on the linker.

Note EXP-2: duplicates cost roughly in PROPORTION to the duplicated fraction (2% -> -2.8%,
20% -> -23.0%), and non-monotonically, duplicating 100% costs only -9.6% because uniform
duplication lets the matching pair consistently. So prune for LOCALISATION, not for count.
"""
from __future__ import annotations
import numpy as np
from .contracts import Graph, Config

def apply_refinement(g: Graph, dzyx: np.ndarray, dlogit: np.ndarray, cfg: Config):
    """Returns (graph, nodes_moved). The count is returned rather than inferred because this
    stage both moves and prunes, and the ledger cannot reconstruct it afterwards."""
    """Move every node by its predicted offset, then prune by the refined rank.

    Nodes are MUTABLE by design: many false positives sit very close to a true node and are
    recoverable by moving them — 'you cannot detect and fix a location'.
    """
    zyx = g.zyx + dzyx
    score = (g.score if g.score is not None else np.zeros(len(g.t))) + dlogit
    keep = score >= np.quantile(score, cfg.refine_prune_quantile) if len(score) else slice(None)
    moved = int((np.abs(dzyx[keep]) > 1e-9).any(axis=1).sum())
    return Graph(t=g.t[keep], zyx=zyx[keep], score=score[keep],
                 edges=np.empty((0, 2), int), dataset=g.dataset), moved

def refine(g: Graph, cfg: Config, model=None, volume=None, device: str = "cpu") -> Graph:
    """No model yet -> pass through unchanged. The pipeline stays runnable, and the stage
    ledger will show `nodes_moved 0`, which is the honest reading: refinement is off."""
    if model is None or not cfg.refine_enabled:
        return g
    raise NotImplementedError("wire refine_head + apply_refinement here")
