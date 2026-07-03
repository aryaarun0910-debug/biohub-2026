"""Phase-1 step 3: high-recall over-proposal with per-candidate features.

Generates MANY candidate nuclei per frame (low DoG threshold, multi-scale) with features, so a
downstream same-cell arbitration (step 4-5) can emit one representative per cell. We propose ONCE
at a low base threshold and keep the response, so a recall-vs-count frontier is a cheap in-memory
filter on the response column (no re-detection).

Candidate columns (see biotrack.cache.FEATURES), RAW voxel coords:
  [z, y, x, response, scale, intensity, contrast, density]

Volume reading mirrors the Kaggle notebook (zarr locally, tensorstore on Kaggle).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

SCALE = (1.625, 0.40625, 0.40625)
ISO_UM = 1.625


@dataclass
class ProposeConfig:
    xy_downsample: int = 4
    norm_q: tuple = (0.01, 0.997)
    scale_pairs_um: list = field(default_factory=lambda: [(1.5, 4.0), (2.2, 5.5)])
    base_rel_threshold: float = 0.02    # LOW: over-propose; frontier filters response afterwards
    nms_dist_um: float = 1.0            # LOOSE NMS is the recall lever (finding step 1a); same-cell
                                        # dedup is the later smart stage. Finer scales HURT -> keep 2.
    max_peaks: int = 60000
    density_radius_um: float = 8.0


# --------------------------------------------------------------------------- volume IO
def open_volume(zarr_path: Path):
    try:
        import zarr
        return zarr.open_group(str(zarr_path), mode="r")["0"]
    except ImportError:
        import tensorstore as ts
        return ts.open({"driver": "zarr3",
                        "kvstore": {"driver": "file", "path": str(Path(zarr_path) / "0")}}).result()


def read_frame(arr, t) -> np.ndarray:
    fr = arr[t]
    return np.asarray(fr.read().result()) if hasattr(fr, "read") else np.asarray(fr)


def downsample_xy(vol: np.ndarray, f: int) -> np.ndarray:
    Z, Y, X = vol.shape
    Y2, X2 = (Y // f) * f, (X // f) * f
    return vol[:, :Y2, :X2].astype(np.float32).reshape(Z, Y2 // f, f, X2 // f, f).mean(axis=(2, 4))


def normalize_frame(vol: np.ndarray, q) -> np.ndarray:
    lo, hi = (float(v) for v in np.quantile(vol, q))
    return np.clip((vol - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- proposal
def _multiscale_dog_and_scale(vol: np.ndarray, scale_pairs, iso=ISO_UM):
    """Return (max response over bands, index of winning band per voxel)."""
    resp = None
    which = None
    for i, (s_small, s_large) in enumerate(scale_pairs):
        d = gaussian_filter(vol, s_small / iso) - gaussian_filter(vol, s_large / iso)
        if resp is None:
            resp = d.copy(); which = np.zeros_like(d, np.int8)
        else:
            m = d > resp
            resp = np.where(m, d, resp); which = np.where(m, i, which)
    return resp, which


def propose_frame(raw_iso: np.ndarray, cfg: ProposeConfig) -> np.ndarray:
    """Over-propose candidates in one isotropic (downsampled) frame. Returns (M,8), ISO coords."""
    norm = normalize_frame(raw_iso, cfg.norm_q)
    resp, which = _multiscale_dog_and_scale(norm, cfg.scale_pairs_um)
    thr = cfg.base_rel_threshold * float(resp.max()) if resp.max() > 0 else cfg.base_rel_threshold
    min_vox = max(1, int(round(cfg.nms_dist_um / ISO_UM)))
    peaks = peak_local_max(resp, min_distance=min_vox, threshold_abs=thr,
                           exclude_border=False, num_peaks=cfg.max_peaks)
    if len(peaks) == 0:
        return np.zeros((0, 8), np.float32)
    zz, yy, xx = peaks.T
    response = resp[zz, yy, xx]
    scale_id = which[zz, yy, xx].astype(np.float32)
    intensity = norm[zz, yy, xx]
    # local contrast: peak minus mean of a small neighborhood (via maximum/mean proxy)
    local_mean = gaussian_filter(norm, 1.0)[zz, yy, xx]
    contrast = intensity - local_mean
    # local density: candidates within density_radius (physical) -> crowding feature
    phys = peaks * np.array(SCALE, np.float32)
    tree = cKDTree(phys)
    density = np.array([len(tree.query_ball_point(p, cfg.density_radius_um)) - 1 for p in phys], np.float32)
    return np.stack([zz, yy, xx, response, scale_id, intensity, contrast, density], axis=1).astype(np.float32)


def to_raw(cands_iso: np.ndarray, f: int) -> np.ndarray:
    """Map ISO-frame candidate coords to RAW voxel coords (undo XY downsample + center offset)."""
    if len(cands_iso) == 0:
        return cands_iso
    out = cands_iso.copy()
    off = (f - 1) / 2.0
    out[:, 1] = out[:, 1] * f + off
    out[:, 2] = out[:, 2] * f + off
    return out


def propose_volume(zarr_path: Path, cfg: ProposeConfig = ProposeConfig()) -> list[np.ndarray]:
    """Over-propose for every timepoint. Returns list over t of (M_t, 8) RAW-coord candidates."""
    arr = open_volume(zarr_path)
    frames = []
    for t in range(arr.shape[0]):
        iso = downsample_xy(read_frame(arr, t), cfg.xy_downsample)
        frames.append(to_raw(propose_frame(iso, cfg), cfg.xy_downsample))
    return frames


def filter_response(frames: list[np.ndarray], rel_threshold: float) -> list[np.ndarray]:
    """Keep candidates whose response >= rel_threshold * (per-frame max). Cheap frontier filter."""
    out = []
    for f in frames:
        if len(f) == 0:
            out.append(f); continue
        thr = rel_threshold * f[:, 3].max()
        out.append(f[f[:, 3] >= thr])
    return out
