"""Per-crop candidate caching for the Phase-1 ablation ladder.

Over-proposal (Phase 1 step 3) is the expensive stage (~seconds/crop x100 frames). We run it
ONCE per crop and cache the candidate set + per-candidate features to disk, so arbitration,
linking, and redetection experiments (steps 4-6) reuse it and run in milliseconds.

Cache layout: data/cache/candidates/<crop>.npz with, per frame, a variable-length list of
candidates flattened + an index. Stored as an object array of per-frame (M_t, D) arrays where
columns are [z, y, x, response, scale, intensity, contrast, density] (raw voxel coords).
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "candidates"

# candidate feature columns (raw voxel coords + features)
FEATURES = ["z", "y", "x", "response", "scale", "intensity", "contrast", "density"]


def path(crop: str) -> Path:
    return CACHE / f"{crop}.npz"


def exists(crop: str) -> bool:
    return path(crop).exists()


def save(crop: str, frames: list[np.ndarray]) -> Path:
    """frames: list over t of (M_t, len(FEATURES)) float arrays."""
    CACHE.mkdir(parents=True, exist_ok=True)
    obj = np.empty(len(frames), dtype=object)
    for t, f in enumerate(frames):
        obj[t] = np.asarray(f, np.float32).reshape(-1, len(FEATURES))
    np.savez_compressed(path(crop), frames=obj, columns=np.array(FEATURES))
    return path(crop)


def load(crop: str) -> list[np.ndarray]:
    """Return list over t of (M_t, len(FEATURES)) arrays, or raise FileNotFoundError."""
    d = np.load(path(crop), allow_pickle=True)
    return list(d["frames"])


def coords(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Extract (M_t, 3) raw (z,y,x) per frame from cached candidate frames."""
    return [f[:, :3] if len(f) else f.reshape(0, 3) for f in frames]
