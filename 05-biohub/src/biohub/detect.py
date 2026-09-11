"""Stage 1 — detect: volume -> candidate nodes.

TARGET ARCHITECTURE (not yet trained; needs the Mac and FOCUS-3D pseudo-labels):

    node_logit = node_head(x)      # supervised DENSELY  on FOCUS-3D pseudo-centres
    refinement = refine_head(x)    # supervised SPARSELY on Kaggle, at detected peaks only

Sparse labels must NOT supervise detection. A loss on single hard voxels teaches every
unannotated real cell to be a hard negative, which is the documented cause of loss blowing up
at ~10 epochs; the dense-supervised route trains 200 epochs without overfitting.

Build rules, all measured:
  * `bias=True` on every Conv3d — the MPS fast path needs it (torch 2.14, ~16x).
  * decoders are Upsample+Conv3d, never ConvTranspose3d, which is unusable on MPS.
  * do NOT augment: removing augmentation improved peak pruning 8%->10-14% AND error
    reduction 18%->22-24% while halving epoch time.
  * the (1,4,4) downsample destroys nothing — 0 of 7,584 GT nodes collapsed.

Until then `detect_oracle` substitutes ground-truth positions so the downstream stages can be
developed, gated and measured. Everything after this stage is already validated against it.
"""
from __future__ import annotations
import numpy as np
from .contracts import Graph, Config, SCALE

def local_maxima(heat: np.ndarray, cfg: Config) -> np.ndarray:
    """Peak extraction with physical-radius suppression, in voxels for a (1,4,4) grid.

    Suppression must be STRICT local maxima, not a packed above-threshold region: greedy
    physical-radius suppression returned 686 points of which only 46 were strict maxima.
    """
    from scipy import ndimage
    rad = np.maximum(1, np.rint(cfg.nms_radius_um / SCALE).astype(int))
    size = tuple(2 * r + 1 for r in rad)
    mx = ndimage.maximum_filter(heat, size=size, mode="nearest")
    return np.argwhere((heat == mx) & (heat >= cfg.det_threshold))

def detect(volume, cfg: Config, model=None, device: str = "cpu") -> Graph:
    if model is None:
        raise NotImplementedError(
            "detect() needs the dual-head model. Use detect_oracle() until it is trained — "
            "every downstream stage is already validated against oracle detections.")
    raise NotImplementedError("wire node_head + peak extraction here")

def detect_oracle(t: np.ndarray, zyx: np.ndarray, dataset: str = "") -> Graph:
    """Ground-truth positions as perfect detections. The substitute the whole pipeline was
    developed against, and the baseline every real detector is measured relative to."""
    return Graph(t=np.asarray(t), zyx=np.asarray(zyx, float),
                 score=np.ones(len(t)), dataset=dataset)
