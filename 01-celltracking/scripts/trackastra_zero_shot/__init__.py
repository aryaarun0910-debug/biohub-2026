"""Offline Trackastra zero-shot bridge for frozen Biohub detections."""

from .adapter import (
    FrozenDetections,
    MaskLabel,
    export_trackastra_result,
    load_frozen_detections,
    rasterize_ellipsoid_masks,
)

__all__ = [
    "FrozenDetections",
    "MaskLabel",
    "export_trackastra_result",
    "load_frozen_detections",
    "rasterize_ellipsoid_masks",
]
