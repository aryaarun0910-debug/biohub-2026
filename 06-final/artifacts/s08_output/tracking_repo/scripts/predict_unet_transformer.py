#!/usr/bin/env python
"""Run UNet + transformer edge prediction on datasets and export to .geff.

Usage:
    uv run scripts/predict_unet_transformer.py --split 0
"""

import argparse
import contextlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import zarr
from tqdm import tqdm

import tracksdata as td

from biohub_tracking.io import open_dataset, save_graph

# Import model and helpers from companion training script.
sys.path.insert(0, str(Path(__file__).parent))
from train_unet_transformer import (
    DEFAULT_METHOD,
    UNetNodeTransformer,
    extract_pos_features,
    _POS_EMBED_DIM,
)
from biohub_tracking.models import TemporalUNet3D

from dataspec import USERNAME, INTERACTIVE, WEIGHTS_PATH
from evaluate import evaluate_run
from biohub_tracking.metrics import summarise


# =============================================================================
# Prediction config
# =============================================================================

@dataclass
class PredictConfig:
    """All hyperparameters that can affect prediction quality / score.

    Detection
    ---------
    det_threshold : float
        Minimum sigmoid probability for a local-max peak to be kept.

    Edge filtering
    --------------
    edge_activation : str
        Activation applied to raw edge logits: ``"sigmoid"`` (independent
        per-edge scores) or ``"softmax"`` (row-normalised over t+1 nodes).
    threshold : float
        Minimum edge probability to consider a link at all.
    max_parents_per_node : int
        Maximum number of incoming edges per node (typically 1).
    max_children_per_node : int
        Maximum number of outgoing edges per node (1 = no divisions, 2 = divisions allowed).
    """
    # Detection
    det_threshold: float = 0.5
    det_tta: bool = True  # flip-xy TTA for detection logits
    pool_kernel_um: float = 3.0  # max-pool kernel size in µm for detection peak extraction
    # Edge filtering
    edge_activation: str = "softmax"  # "sigmoid" or "softmax"
    threshold: float = 0.5

    # ILP post-processing
    use_ilp: bool = False
    ilp_edge_weight: float = -1.0
    ilp_appearance_weight: float = 0.1
    ilp_disappearance_weight: float = 0.1
    ilp_division_weight: float = 1.0

    max_parents_per_node: int | None = None
    max_children_per_node: int | None = None

    def __post_init__(self) -> None:
        # When ILP is enabled it handles parent/children constraints itself,
        # so greedy limits are left unconstrained (None).  When ILP is
        # disabled, default to 1/1 to avoid unconstrained edge assignment.
        if not self.use_ilp:
            if self.max_parents_per_node is None:
                self.max_parents_per_node = 1
            if self.max_children_per_node is None:
                self.max_children_per_node = 2



# =============================================================================
# Helpers
# =============================================================================


@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


# =============================================================================
# Graph building
# =============================================================================

def build_graph(
    coords: np.ndarray,
    edges: list[tuple[int, int, float, float]],
) -> td.graph.InMemoryGraph:
    """Build a tracksdata graph from detection coords and predicted edges.

    Avoids ``add_node_attr_key`` to sidestep a tracksdata/Polars compatibility
    issue where the float default value is mistakenly used as a dtype.
    Probabilities are passed as-is (softmax output, already in [0, 1]).
    """
    graph = td.graph.InMemoryGraph()

    for key in ["z", "y", "x"]:
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    node_ids = graph.bulk_add_nodes([
        {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
        for t, z, y, x in coords
    ])

    if edges:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        graph.bulk_add_edges([
            {
                "source_id": node_ids[src],
                "target_id": node_ids[tgt],
                "edge_prob": prob,
                "edge_dist": dist,
            }
            for src, tgt, prob, dist in edges
        ])

    return graph


# =============================================================================
# Model loading
# =============================================================================

_DEFAULT_CONFIG = {
    "unet_out_channels": 32,
    "unet_layers": [32, 64, 128],
    "downsample": [1, 4, 4],
    "window_size": 2,
}


def load_model(
    weights_path: Path, device: torch.device,
) -> tuple[UNetNodeTransformer, int, tuple[int, ...]]:
    """Reconstruct UNetNodeTransformer from saved config + weights.

    Reads ``config.json`` from the same directory as the weights file.
    Falls back to ``_DEFAULT_CONFIG`` if the file is missing.

    Returns ``(model, window_size, downsample)``.
    """
    config_path = weights_path.parent / "config.json"
    if config_path.exists():
        config = {**_DEFAULT_CONFIG, **json.loads(config_path.read_text())}
    else:
        print(f"Warning: config.json not found at {config_path}, using defaults.", flush=True)
        config = _DEFAULT_CONFIG

    # Support legacy configs that used "downsample_factor" (scalar).
    if "downsample_factor" in config and "downsample" not in config:
        df = config["downsample_factor"]
        config["downsample"] = [df, df, df]

    downsample = tuple(config["downsample"])

    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=config["unet_out_channels"],
        layers=config["unet_layers"],
    )
    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=config["unet_out_channels"],
        pos_feat_dim=4 * _POS_EMBED_DIM,
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config["window_size"], downsample


# =============================================================================
# Per-frame loading
# =============================================================================

def _load_frame(
    zarr_arr,
    t: int,
    target_shape: list[int],
    downsample: tuple[int, ...] = (1, 1, 1),
) -> torch.Tensor:
    """Load one frame from zarr with strided spatial downsample (no normalisation)."""
    dz, dy, dx = downsample
    raw = zarr_arr[t, ::dz, ::dy, ::dx].astype(np.float32)
    frame = torch.from_numpy(raw)
    if list(frame.shape) != target_shape:
        frame = F.interpolate(
            frame[None, None], size=target_shape,
            mode="trilinear", align_corners=False,
        )[0, 0]
    return frame


# =============================================================================
# Inference
# =============================================================================

def pool_kernel_from_um(
    um: float,
    voxel_size: tuple[float, ...],
) -> tuple[int, ...]:
    """Convert a physical suppression distance (microns) to a per-axis voxel kernel.

    Each axis gets ``round(um / voxel_size_axis)`` voxels, forced to odd
    (for symmetric padding) and at least 1.

    Parameters
    ----------
    um : float
        Desired suppression distance in microns.
    voxel_size : tuple[float, ...]
        Per-axis voxel sizes in microns, e.g. ``(1.625, 0.40625, 0.40625)``.
    """
    kernel = []
    for s in voxel_size:
        k = max(1, round(um / s))
        if k % 2 == 0:
            k += 1
        kernel.append(k)
    return tuple(kernel)


def _detect_cells_pooled(
    det_logits: torch.Tensor,
    t: int,
    det_threshold: float = 0.5,
    pool_kernel: tuple[int, ...] = (3, 3, 3),
) -> np.ndarray:
    """Extract cell coordinates via max-pool local-max (same as training).

    Coordinates are returned in the downsampled grid.  The caller is
    responsible for scaling back to original resolution if needed.

    Parameters
    ----------
    det_logits : torch.Tensor
        (1, Z, Y, X) raw logits.
    t : int
        Time index to prepend as the first column.
    det_threshold : float
        Minimum sigmoid probability for a peak to be considered (default 0.5).
    pool_kernel : tuple[int, ...]
        Per-axis kernel size for local-max pooling,
        e.g. ``(3, 11, 11)`` for anisotropic data.

    Returns
    -------
    np.ndarray
        (N, 4) int16 array with columns [t, z, y, x] in downsampled space.
    """
    logits = det_logits.unsqueeze(0)  # (1, 1, Z, Y, X)
    pad = tuple(k // 2 for k in pool_kernel)
    pooled = F.max_pool3d(logits, pool_kernel, stride=1, padding=pad)
    is_peak = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)
    peak_idx = torch.nonzero(is_peak[0, 0])  # (N, 3)

    if peak_idx.shape[0] == 0:
        return np.empty((0, 4), dtype=np.int16)

    coords = peak_idx.float().cpu().numpy()
    t_col = np.full((len(coords), 1), t, dtype=np.float32)
    return np.concatenate([t_col, coords], axis=1).astype(np.int16)


@torch.no_grad()
def predict_video(
    model: UNetNodeTransformer,
    ds_path: Path,
    device: torch.device,
    cfg: PredictConfig,
    window_size: int = 2,
    max_frames: int | None = None,
    unet_batch_size: int = 4,
    downsample: tuple[int, ...] = (1, 4, 4),
    secondary_model: UNetNodeTransformer | None = None,
    secondary_edge_weight: float = 0.0,
    secondary_detection_weight: float = 0.0,
    secondary_link_mode: str = "raw",
    secondary_mix_temperature: float = 1.0,
    secondary_low_margin_max: float = 0.2,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    """Run inference on a single video using sliding windows of W frames.

    Windows slide with stride ``W - 1`` so every consecutive pair is covered
    exactly once.  UNet features from each window are reused for edge
    prediction on all ``W - 1`` consecutive pairs within the window.

    Returns
    -------
    coords : np.ndarray
        Shape (N, 4) — columns [t, z, y, x] in original resolution.
    edges : list of (src_idx, tgt_idx, prob, distance) tuples
    """
    ds = open_dataset(ds_path, normalize=False, load_image=False, downsample=downsample)
    if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
        raise ValueError(f"Zarr attrs missing image_statistics.quantiles for {ds_path}")
    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low = float(ds.quantiles["0.001"])
    q_high = float(ds.quantiles["0.999"])

    T = ds.image_shape[0] if max_frames is None else min(ds.image_shape[0], max_frames)
    image_shape = (T,) + ds.image_shape[1:]
    target_shape = list(image_shape[1:])

    ds_arr = np.array(downsample, dtype=np.float32)  # for coord rescaling at the end
    ds_arr_t = torch.from_numpy(ds_arr).to(device)   # for predict_edges (original-space coords)
    pos_feat_dim = 4 * _POS_EMBED_DIM
    W = window_size
    voxel_size = tuple(s * d for s, d in zip(ds.scale, downsample))
    pool_k = pool_kernel_from_um(cfg.pool_kernel_um, voxel_size)

    # Running node registry — each entry records the frame-t detections.
    # coord_offset[t] = (start, end) half-open range into the stacked array.
    seen_frames: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    coord_lists: list[np.ndarray] = []
    coord_offset: dict[int, tuple[int, int]] = {}
    global_node_count: int = 0
    all_edges: list[tuple[int, int, float, float]] = []

    # Sliding windows with stride W-1 cover every consecutive pair exactly once.
    stride = max(W - 1, 1)
    window_starts = list(range(0, T - W + 1, stride))
    # Ensure the very last pair (T-2 → T-1) is covered.
    if not window_starts or window_starts[-1] + W < T:
        last = max(T - W, 0)
        if not window_starts or last != window_starts[-1]:
            window_starts.append(last)

    for ws in tqdm(
        window_starts,
        desc="  windows",
        leave=False,
        disable=not INTERACTIVE,
    ):
        frame_indices = list(range(ws, ws + W))

        # --- UNet encode (single window, batch_size=1) ---
        imgs = torch.stack([
            _load_frame(zarr_arr, t, target_shape, downsample)
            for t in frame_indices
        ])  # (W, *spatial)
        # Quantile normalisation (0.1%–99.9%) to match training pipeline.
        imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
        imgs = imgs.unsqueeze(0).to(device)   # (1, W, *spatial)

        unet_out, det_logits = model.encode(imgs)
        # unet_out: (1, W, C, *spatial_down), det_logits: list of W × (1, 1, *spatial_down)

        # Detection TTA: original + flip-x + flip-y + flip-xy, average logits.
        # TTA: flip along Y (-2) and X (-1) only.  Z is excluded because
        # the data is highly anisotropic (Z resolution ~4x coarser than XY),
        # so Z-flips would produce out-of-distribution inputs.
        if cfg.det_tta:
            _edge_tta = os.environ.get('BIOHUB_EDGE_FEATURE_TTA', '0') != '0'
            _unet_acc = unet_out.clone() if _edge_tta else None
            _nv = 1

            for dims in [(-1,), (-2,), (-2, -1)]:
                imgs_flip = imgs.flip(dims)
                _u_flip, det_flip = model.encode(imgs_flip)

                for f in range(W):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)

                if _edge_tta:
                    _unet_acc = _unet_acc + _u_flip.flip(dims)
                del imgs_flip, det_flip, _u_flip
                _nv += 1

            for _k in (1, 3):
                imgs_rot = torch.rot90(imgs, _k, dims = (-2, -1))
                _u_rot, det_rot = model.encode(imgs_rot)

                for f in range(W):
                    det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims = (-2, -1))

                if _edge_tta:
                    _unet_acc = _unet_acc + torch.rot90(_u_rot, -_k, dims = (-2, -1))
                del imgs_rot, det_rot, _u_rot
                _nv += 1

            imgs_t = imgs.transpose(-1, -2)
            _u_t, det_t = model.encode(imgs_t)

            for f in range(W):
                det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)

            if _edge_tta:
                _unet_acc = _unet_acc + _u_t.transpose(-1, -2)
            del imgs_t, det_t, _u_t
            _nv += 1

            imgs_at = torch.rot90(imgs, 1, dims = (-2, -1)).transpose(-1, -2)
            _u_at, det_at = model.encode(imgs_at)

            for f in range(W):
                det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims = (-2, -1))

            if _edge_tta:
                _unet_acc = _unet_acc + torch.rot90(_u_at.transpose(-1, -2), -1, dims = (-2, -1))
            del imgs_at, det_at, _u_at
            _nv += 1

            for f in range(W):
                det_logits[f] = det_logits[f] / _nv

            if _edge_tta:
                if _unet_acc.shape != unet_out.shape:
                    raise RuntimeError(f'Edge-feature TTA shape mismatch: {tuple(_unet_acc.shape)} vs {tuple(unet_out.shape)}')
                _delta = float((_unet_acc / _nv - unet_out).abs().mean())

                if _delta == 0.0:
                    raise RuntimeError('Edge-feature TTA produced no feature change')
                unet_out = _unet_acc / _nv
                print('EDGE_TTA_ACTIVE views =', _nv, 'mean_abs_feat_delta =', round(_delta, 6), flush = True)
                del _unet_acc

        secondary_unet_out = None

        if secondary_model is not None:
            secondary_unet_out, secondary_det_logits = secondary_model.encode(imgs)
            _secondary_edge_tta = os.environ.get('BIOHUB_SECONDARY_EDGE_FEATURE_TTA', '0') != '0'
            _secondary_unet_acc = secondary_unet_out.clone() if _secondary_edge_tta else None

            if secondary_detection_weight > 0.0:
                if cfg.det_tta:
                    _secondary_nv = 1

                    for dims in [(-1,), (-2,), (-2, -1)]:
                        secondary_imgs_flip = imgs.flip(dims)
                        _secondary_u_flip, secondary_det_flip = secondary_model.encode(secondary_imgs_flip)

                        for f in range(W):
                            secondary_det_logits[f] = secondary_det_logits[f] + secondary_det_flip[f].flip(dims)

                        if _secondary_edge_tta:
                            _secondary_unet_acc = _secondary_unet_acc + _secondary_u_flip.flip(dims)
                        del secondary_imgs_flip, secondary_det_flip, _secondary_u_flip
                        _secondary_nv += 1

                    for _k in (1, 3):
                        secondary_imgs_rot = torch.rot90(imgs, _k, dims = (-2, -1))
                        _secondary_u_rot, secondary_det_rot = secondary_model.encode(secondary_imgs_rot)

                        for f in range(W):
                            secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(secondary_det_rot[f], -_k, dims = (-2, -1))

                        if _secondary_edge_tta:
                            _secondary_unet_acc = _secondary_unet_acc + torch.rot90(_secondary_u_rot, -_k, dims = (-2, -1))
                        del secondary_imgs_rot, secondary_det_rot, _secondary_u_rot
                        _secondary_nv += 1
                    secondary_imgs_t = imgs.transpose(-1, -2)
                    _secondary_u_t, secondary_det_t = secondary_model.encode(secondary_imgs_t)

                    for f in range(W):
                        secondary_det_logits[f] = secondary_det_logits[f] + secondary_det_t[f].transpose(-1, -2)

                    if _secondary_edge_tta:
                        _secondary_unet_acc = _secondary_unet_acc + _secondary_u_t.transpose(-1, -2)
                    del secondary_imgs_t, secondary_det_t, _secondary_u_t
                    _secondary_nv += 1
                    secondary_imgs_at = torch.rot90(imgs, 1, dims = (-2, -1)).transpose(-1, -2)
                    _secondary_u_at, secondary_det_at = secondary_model.encode(secondary_imgs_at)

                    for f in range(W):
                        secondary_det_logits[f] = secondary_det_logits[f] + torch.rot90(secondary_det_at[f].transpose(-1, -2), -1, dims = (-2, -1))

                    if _secondary_edge_tta:
                        _secondary_unet_acc = _secondary_unet_acc + torch.rot90(_secondary_u_at.transpose(-1, -2), -1, dims = (-2, -1))
                    del secondary_imgs_at, secondary_det_at, _secondary_u_at
                    _secondary_nv += 1

                    for f in range(W):
                        secondary_det_logits[f] = secondary_det_logits[f] / _secondary_nv

                    if _secondary_edge_tta:
                        if _secondary_unet_acc.shape != secondary_unet_out.shape:
                            raise RuntimeError(f'Secondary edge-feature TTA shape mismatch: {tuple(_secondary_unet_acc.shape)} vs {tuple(secondary_unet_out.shape)}')
                        _secondary_delta = float((_secondary_unet_acc / _secondary_nv - secondary_unet_out).abs().mean())

                        if _secondary_delta == 0.0:
                            raise RuntimeError('Secondary edge-feature TTA produced no feature change')
                        _secondary_edge_tta_weight = float(os.environ.get('BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT', '1.0'))

                        if not 0.0 < _secondary_edge_tta_weight <= 1.0:
                            raise RuntimeError(f'Secondary edge-feature TTA weight must be in (0, 1], got {_secondary_edge_tta_weight}')
                        _secondary_tta_mean = _secondary_unet_acc / _secondary_nv
                        secondary_unet_out = (1.0 - _secondary_edge_tta_weight) * secondary_unet_out + _secondary_edge_tta_weight * _secondary_tta_mean
                        print('SECONDARY_EDGE_TTA_ACTIVE views =', _secondary_nv, 'weight =', _secondary_edge_tta_weight, 'mean_abs_feat_delta =', round(_secondary_delta, 6), flush = True)
                        del _secondary_unet_acc

                for f in range(W):
                    primary_det = det_logits[f]
                    secondary_det = secondary_det_logits[f]
                    primary_mean = primary_det.mean()
                    secondary_mean = secondary_det.mean()
                    primary_scale = primary_det.float().std(unbiased = False).clamp_min(1e-4)
                    secondary_scale = secondary_det.float().std(unbiased = False).clamp_min(1e-4)
                    scale_ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)
                    secondary_det_aligned = (secondary_det - secondary_mean) * scale_ratio + primary_mean
                    blended_det = (1.0 - secondary_detection_weight) * primary_det + secondary_detection_weight * secondary_det_aligned
                    primary_candidates = len(_detect_cells_pooled(primary_det[0], int(frame_indices[f]), cfg.det_threshold, pool_k))
                    blended_candidates = len(_detect_cells_pooled(blended_det[0], int(frame_indices[f]), cfg.det_threshold, pool_k))
                    minimum_retention = float(os.environ.get('BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION', '0.90'))
                    candidate_retention = blended_candidates / primary_candidates if primary_candidates else 1.0
                    use_primary_detection = bool(primary_candidates > 0 and candidate_retention < minimum_retention)
                    det_logits[f] = primary_det if use_primary_detection else blended_det

                    if int(frame_indices[f]) not in seen_frames:
                        shard = os.environ.get('BIOHUB_GPU_SHARD', 'single').replace('/', '_')
                        guard_log = Path('/kaggle/working') / f'retention_guard_{shard}.jsonl'
                        guard_record = {'dataset': ds_path.stem, 'frame': int(frame_indices[f]), 'primary_candidates': int(primary_candidates), 'blended_candidates': int(blended_candidates), 'retention': float(candidate_retention), 'minimum_retention': float(minimum_retention), 'use_primary': bool(use_primary_detection)}

                        with guard_log.open('a') as guard_handle:
                            guard_handle.write(json.dumps(guard_record, sort_keys = True) + '\n')

                        if use_primary_detection:
                            print('BIOHUB_RETENTION_GUARD ' + json.dumps(guard_record, sort_keys = True), flush = True)

            del secondary_det_logits

        del imgs

        # --- Detect cells in each frame (dedup across windows) ---
        for f_idx, t in enumerate(frame_indices):
            if t not in seen_frames:
                arr = _detect_cells_pooled(
                    det_logits[f_idx][0], t, cfg.det_threshold, pool_k,
                )
                coord_offset[t] = (global_node_count, global_node_count + len(arr))
                global_node_count += len(arr)
                coord_lists.append(arr)
                seen_frames.add(t)

        coords_so_far = (
            np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
        )

        # --- Edge prediction for each consecutive pair in the window ---
        for f_idx in range(W - 1):
            t_src, t_tgt = frame_indices[f_idx], frame_indices[f_idx + 1]
            if (t_src, t_tgt) in seen_pairs:
                continue
            seen_pairs.add((t_src, t_tgt))

            if t_src not in coord_offset or t_tgt not in coord_offset:
                continue
            s_src, e_src = coord_offset[t_src]
            s_tgt, e_tgt = coord_offset[t_tgt]
            if e_src == s_src or e_tgt == s_tgt:
                continue

            c_src = coords_so_far[s_src:e_src]
            c_tgt = coords_so_far[s_tgt:e_tgt]
            n_src, n_tgt = len(c_src), len(c_tgt)
            idx_src = np.arange(s_src, e_src, dtype=np.int64)
            idx_tgt = np.arange(s_tgt, e_tgt, dtype=np.int64)

            # Build tensors (batch_size=1).
            p_coords_src = torch.from_numpy(c_src[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            p_coords_tgt = torch.from_numpy(c_tgt[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            # Use window-relative time (f_idx, f_idx+1) normalised by W, not absolute frame index.
            window_shape = (W,) + image_shape[1:]
            c_src_rel = c_src.copy()
            c_src_rel[:, 0] = f_idx
            c_tgt_rel = c_tgt.copy()
            c_tgt_rel[:, 0] = f_idx + 1
            p_pos_src = torch.from_numpy(extract_pos_features(c_src_rel, window_shape)).unsqueeze(0).to(device)
            p_pos_tgt = torch.from_numpy(extract_pos_features(c_tgt_rel, window_shape)).unsqueeze(0).to(device)
            p_mask_src = torch.ones(1, n_src, dtype=torch.bool, device=device)
            p_mask_tgt = torch.ones(1, n_tgt, dtype=torch.bool, device=device)

            unet_feat_src = model._index_features(
                unet_out[:, f_idx], p_coords_src, p_mask_src,
            )
            unet_feat_tgt = model._index_features(
                unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt,
            )
            edge_logits_pair = model.predict_edges(unet_feat_src, unet_feat_tgt, p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t, p_pos_src, p_pos_tgt, p_mask_src, p_mask_tgt,)

            _bidirectional_weight = float(os.environ.get("BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT", "0"))

            if _bidirectional_weight > 0.0:
                reverse_logits_native = model.predict_edges(unet_feat_tgt, unet_feat_src, p_coords_tgt * ds_arr_t, p_coords_src * ds_arr_t, p_pos_tgt, p_pos_src, p_mask_tgt, p_mask_src,)
                reverse_logits_pair = reverse_logits_native.transpose(1, 2)

                forward_center = edge_logits_pair.mean(dim = 1, keepdim = True)
                forward_scale = edge_logits_pair.float().std(dim = 1, keepdim = True, unbiased = False).clamp_min(1e-4)
                reverse_center = reverse_logits_pair.mean(dim = 1, keepdim = True)
                reverse_scale = reverse_logits_pair.float().std(dim = 1, keepdim = True, unbiased = False).clamp_min(1e-4)
                reverse_scale_ratio = (forward_scale / reverse_scale).clamp(0.5, 2.0)
                reverse_scale_ratio = reverse_scale_ratio.to(reverse_logits_pair.dtype)
                reverse_aligned = (reverse_logits_pair - reverse_center) * reverse_scale_ratio + forward_center
                forward_prob = torch.softmax(edge_logits_pair.float(), dim = 1).clamp_min(1e-8)
                reverse_prob = torch.softmax(reverse_aligned.float(), dim = 1).clamp_min(1e-8)
                harmonic_prob = 1.0 / ((1.0 - _bidirectional_weight) / forward_prob + _bidirectional_weight / reverse_prob)
                harmonic_prob = harmonic_prob / harmonic_prob.sum(dim = 1, keepdim = True).clamp_min(1e-8)
                harmonic_logits = torch.log(harmonic_prob.clamp_min(1e-8))
                harmonic_center = harmonic_logits.mean(dim = 1, keepdim = True)
                harmonic_scale = harmonic_logits.std(dim = 1, keepdim = True, unbiased = False).clamp_min(1e-4)
                harmonic_scale_ratio = (forward_scale / harmonic_scale).clamp(0.5, 2.0)
                edge_logits_pair = ((harmonic_logits - harmonic_center) * harmonic_scale_ratio + forward_center).to(reverse_aligned.dtype)
                del (reverse_logits_native, reverse_logits_pair, reverse_aligned, forward_prob, reverse_prob, harmonic_prob, harmonic_logits,)

            if secondary_model is not None:
                if secondary_unet_out is None:
                    raise RuntimeError('Secondary model is loaded but its feature map is missing')
                secondary_feat_src = secondary_model._index_features(secondary_unet_out[:, f_idx], p_coords_src, p_mask_src)
                secondary_feat_tgt = secondary_model._index_features(secondary_unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt)
                secondary_logits_pair = secondary_model.predict_edges(secondary_feat_src, secondary_feat_tgt, p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t, p_pos_src, p_pos_tgt, p_mask_src, p_mask_tgt)

                if secondary_link_mode == 'raw':
                    secondary_for_mix = secondary_logits_pair
                    blend_weight = secondary_edge_weight
                elif secondary_link_mode in {'calibrated', 'adaptive', 'low_margin_consensus'}:
                    primary_center = edge_logits_pair.mean(dim = 1, keepdim = True)
                    primary_scale = edge_logits_pair.float().std(dim = 1, keepdim = True, unbiased = False).clamp_min(0.0001)
                    secondary_center = secondary_logits_pair.mean(dim = 1, keepdim = True)
                    secondary_scale = secondary_logits_pair.float().std(dim = 1, keepdim = True, unbiased = False).clamp_min(0.0001)
                    secondary_scale_ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)
                    secondary_for_mix = (secondary_logits_pair - secondary_center) * secondary_scale_ratio + primary_center

                    if secondary_link_mode == 'calibrated':
                        blend_weight = secondary_edge_weight
                    elif secondary_link_mode == 'adaptive':
                        if n_src >= 2:
                            primary_probs = torch.softmax(edge_logits_pair[0], dim = 0)
                            secondary_probs = torch.softmax(secondary_for_mix[0], dim = 0)
                            primary_top2 = torch.topk(primary_probs, k = 2, dim = 0)
                            secondary_top2 = torch.topk(secondary_probs, k = 2, dim = 0)
                            primary_margin = primary_top2.values[0] - primary_top2.values[1]
                            secondary_margin = secondary_top2.values[0] - secondary_top2.values[1]
                            local_weight = (secondary_edge_weight + secondary_margin - primary_margin).clamp(0.15, 0.75)
                            same_parent = primary_top2.indices[0].eq(secondary_top2.indices[0])
                            local_weight = torch.where(same_parent, torch.maximum(local_weight, torch.full_like(local_weight, secondary_edge_weight)), local_weight)
                            blend_weight = local_weight.view(1, 1, -1)
                        else:
                            blend_weight = secondary_edge_weight
                    elif n_src >= 2:
                        primary_probs = torch.softmax(edge_logits_pair[0], dim = 0)
                        secondary_probs = torch.softmax(secondary_for_mix[0], dim = 0)
                        primary_top2 = torch.topk(primary_probs, k = 2, dim = 0)
                        secondary_top2 = torch.topk(secondary_probs, k = 2, dim = 0)
                        primary_margin = primary_top2.values[0] - primary_top2.values[1]
                        same_parent = primary_top2.indices[0].eq(secondary_top2.indices[0])
                        uncertainty = ((secondary_low_margin_max - primary_margin) / secondary_low_margin_max).clamp(0.0, 1.0)
                        local_weight = secondary_edge_weight * uncertainty
                        local_weight = torch.where(same_parent, local_weight, torch.zeros_like(local_weight))
                        blend_weight = local_weight.view(1, 1, -1)
                    else:
                        blend_weight = 0.0
                else:
                    raise ValueError(f'Unsupported secondary link mode: {secondary_link_mode}')
                edge_logits_pair = (1.0 - blend_weight) * edge_logits_pair + blend_weight * secondary_for_mix

                if secondary_mix_temperature != 1.0:
                    mixed_center = edge_logits_pair.mean(dim = 1, keepdim = True)
                    edge_logits_pair = mixed_center + (edge_logits_pair - mixed_center) / secondary_mix_temperature

            raw = edge_logits_pair[0]
            if cfg.edge_activation == "softmax":
                probs = torch.softmax(raw, dim=0).cpu().numpy()
            else:
                probs = torch.sigmoid(raw).cpu().numpy()

            candidates = sorted(
                [
                    (probs[i, j], i, j)
                    for i in range(n_src)
                    for j in range(n_tgt)
                    if probs[i, j] > cfg.threshold
                ],
                reverse=True,
            )

            children_count: dict[int, int] = {}
            parents_count: dict[int, int] = {}

            for prob, i, j in candidates:
                n_ch = children_count.get(i, 0)
                n_pa = parents_count.get(j, 0)
                if cfg.max_children_per_node is not None and n_ch >= cfg.max_children_per_node:
                    continue
                if cfg.max_parents_per_node is not None and n_pa >= cfg.max_parents_per_node:
                    continue

                gi, gj = int(idx_src[i]), int(idx_tgt[j])
                dist = float(np.linalg.norm(
                    coords_so_far[gi, 1:].astype(np.float32)
                    - coords_so_far[gj, 1:].astype(np.float32)
                ))
                all_edges.append((gi, gj, float(prob), dist))
                children_count[i] = n_ch + 1
                parents_count[j] = n_pa + 1

        del unet_out

        if secondary_unet_out is not None:
            del secondary_unet_out

    coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
    # Scale spatial coords back to original resolution.
    coords = coords.astype(np.float32)
    coords[:, 1:] *= ds_arr
    coords = coords.astype(np.int16)
    _coordinate_manifest_arm = os.environ.get('BIOHUB_DIAGNOSTIC_ARM', '').strip()

    if _coordinate_manifest_arm:
        import hashlib as _coordinate_hashlib
        _coordinate_shard = os.environ.get('BIOHUB_GPU_SHARD', 'single').replace('/', '_')
        _coordinate_array = np.ascontiguousarray(coords.astype('<i2', copy = False))
        _coordinate_frame_counts = [[int(_coordinate_t), int((_coordinate_array[:, 0] == _coordinate_t).sum())] for _coordinate_t in np.unique(_coordinate_array[:, 0])]
        _coordinate_record = {'columns': ['t', 'z', 'y', 'x'], 'coordinate_sha256': _coordinate_hashlib.sha256(_coordinate_array.tobytes(order = 'C')).hexdigest(), 'dataset': ds_path.stem, 'dtype': '<i2', 'frame_counts': _coordinate_frame_counts, 'rows': int(len(_coordinate_array)), 'stage': 'post_detection_pre_graph_pre_ilp'}
        _coordinate_manifest_path = Path('/kaggle/working') / f'detector_coordinates_{_coordinate_manifest_arm}_{_coordinate_shard}.jsonl'

        with _coordinate_manifest_path.open('a') as _coordinate_handle:
            _coordinate_handle.write(json.dumps(_coordinate_record, sort_keys = True) + '\n')

    return (coords, all_edges)


# =============================================================================
# Prediction loop
# =============================================================================

def predict(
    data_dir: Path,
    fold: int,
    splits_file: Path,
    weights_path: Path,
    cfg: PredictConfig,
    method: str = DEFAULT_METHOD,
    debug_video: Path | None = None,
    unet_batch_size: int = 4,
    video_slice: slice | None = None,
    evaluate: bool = False,
) -> None:
    """Run inference on the test split and save predictions as .geff files."""
    if debug_video is not None:
        test_names = [debug_video.name]
        data_dir = debug_video.parent
    else:
        folds = json.loads(splits_file.read_text())
        test_names = folds[fold]["test"]
        if video_slice is not None:
            test_names = test_names[video_slice]

    from dataspec import PREDICTIONS_PATH
    output_dir = PREDICTIONS_PATH / USERNAME / method / f"split_{fold}"
    if output_dir.exists():
        import shutil
        for old in output_dir.glob("*.geff"):
            if old.is_dir():
                shutil.rmtree(old)
            else:
                old.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, window_size, downsample = load_model(weights_path, device)

    secondary_model = None
    secondary_weights_text = os.environ.get("BIOHUB_SECONDARY_WEIGHTS", "").strip()
    secondary_edge_weight = float(os.environ.get("BIOHUB_SECONDARY_EDGE_WEIGHT", "0"))
    secondary_detection_weight = float(os.environ.get("BIOHUB_SECONDARY_DETECTION_WEIGHT", "0"))
    secondary_link_mode = os.environ.get("BIOHUB_SECONDARY_LINK_MODE", "raw").strip()
    secondary_mix_temperature = float(os.environ.get("BIOHUB_SECONDARY_MIX_TEMPERATURE", "1"))
    secondary_low_margin_max = float(os.environ.get("BIOHUB_SECONDARY_LOW_MARGIN_MAX", "0.2"))
    edge_candidate_threshold = float(os.environ.get("BIOHUB_DUAL_SEED_EDGE_THRESHOLD", str(cfg.threshold)))

    if secondary_weights_text:
        if not 0.0 < secondary_edge_weight < 1.0:
            raise ValueError("BIOHUB_SECONDARY_EDGE_WEIGHT must be strictly between 0 and 1")

        if not 0.0 <= secondary_detection_weight < 1.0:
            raise ValueError("BIOHUB_SECONDARY_DETECTION_WEIGHT must be in the half-open interval [0, 1)")

        if secondary_link_mode not in { "raw", "calibrated", "adaptive", "low_margin_consensus" }:
            raise ValueError("BIOHUB_SECONDARY_LINK_MODE must be raw, calibrated, adaptive, " "or low_margin_consensus")

        if not 0.5 <= secondary_mix_temperature <= 2.0:
            raise ValueError("BIOHUB_SECONDARY_MIX_TEMPERATURE must be in [0.5, 2.0]")

        if not 0.0 < edge_candidate_threshold < 1.0:
            raise ValueError("BIOHUB_DUAL_SEED_EDGE_THRESHOLD must be strictly between 0 and 1")

        if not 0.0 < secondary_low_margin_max <= 1.0:
            raise ValueError("BIOHUB_SECONDARY_LOW_MARGIN_MAX must be in (0, 1]")
        secondary_model, secondary_window_size, secondary_downsample = load_model(Path(secondary_weights_text), device,)

        if secondary_window_size != window_size or secondary_downsample != downsample:
            raise ValueError("Primary and secondary models have incompatible inference grids: " f"primary: (window = {window_size}, downsample = {downsample}), " f"secondary: (window = {secondary_window_size}, downsample = {secondary_downsample})")
        cfg.threshold = edge_candidate_threshold
        print(f"Secondary model: {secondary_weights_text} | " f"edge weight = {secondary_edge_weight:.3f} | " f"detection weight = {secondary_detection_weight:.3f} | " f"link mode = {secondary_link_mode} | " f"temperature = {secondary_mix_temperature:.3f} | " f"low-margin max = {secondary_low_margin_max:.3f} | " f"edge threshold = {cfg.threshold:.3f}", flush = True,)

    print(
        f"Fold {fold}: {len(test_names)} datasets | "
        f"weights={weights_path} | device={device} | window_size={window_size} | pool_kernel_um={cfg.pool_kernel_um}",
        flush=True,
    )

    for name in tqdm(test_names, desc="Predicting", disable=not INTERACTIVE):
        ds_path = data_dir / name
        coords, edges = predict_video(
                model, ds_path, device,
                cfg=cfg,
                window_size=window_size,
                unet_batch_size = unet_batch_size,
                downsample = downsample,
                secondary_model = secondary_model,
                secondary_edge_weight = secondary_edge_weight,
                secondary_detection_weight = secondary_detection_weight,
                secondary_link_mode = secondary_link_mode,
                secondary_mix_temperature = secondary_mix_temperature,
                secondary_low_margin_max = secondary_low_margin_max)
        graph = build_graph(coords, edges)
        if cfg.use_ilp and graph.num_edges() > 0:
            solver = td.solvers.ILPSolver(
                edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
                appearance_weight=cfg.ilp_appearance_weight,
                disappearance_weight=cfg.ilp_disappearance_weight,
                division_weight=cfg.ilp_division_weight,
            )
            with suppress_output():
                graph = solver.solve(graph)
        save_graph(graph, output_dir / f"{name}.geff")

    print(f"Saved {len(test_names)} predictions to {output_dir}", flush=True)

    if evaluate:
        run = {
            "username": USERNAME,
            "method": method,
            "split": f"split_{fold}",
            "dir": output_dir,
            "geffs": sorted(output_dir.glob("*.geff")),
        }
        results = evaluate_run(run)
        s = summarise(results)
        print(
            f"Evaluation ({len(results)} videos): "
            f"score={s['score']:.4f}  "
            f"edge_jaccard={s['edge_jaccard']:.4f}  "
            f"adj_edge_jaccard={s['adj_edge_jaccard']:.4f} (n_adj={s['n_adj']})  "
            f"division_jaccard={s['division_jaccard']:.4f} "
            f"(TP={s['division_tp']} FP={s['division_fp']} FN={s['division_fn']})  "
            f"node_recall={s['node_recall']:.4f}  (n={s['n']})",
            flush=True,
        )


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run UNet + transformer edge prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Default: DATASET_PATH")
    parser.add_argument("--splits", type=str, default=None,
                        help="Default: DATASET_PATH/dataset_splits.json")
    parser.add_argument("--split", type=str, default="0",
                        help="Split index (0-4) or 'all'.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to weights file. "
                             "Default: weights/{method}/split_{split}/edge_predictor_best.pth")
    parser.add_argument("--debug-video", type=str, default=None,
                        help="Path to a single dataset. Ignores fold/splits.")
    parser.add_argument("--slice", type=str, default=None,
                        help="Python slice of the test list, e.g. ':1' for first video, "
                             "'2:5' for videos 2-4.")
    parser.add_argument("--unet-batch-size", type=int, default=4,
                        help="Number of frame pairs per UNet forward pass (default: 4).")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation against GT after saving predictions.")
    parser.add_argument("--det-threshold", type=float, default=0.99,
                        help="Min sigmoid probability for a detection peak to be kept. "
                             "Default 0.99: the detector is poorly calibrated because the "
                             "ground truth is sparse (only some cells annotated), so a high "
                             "threshold keeps precision up. Sweep it for your model.")
    parser.add_argument("--use-ilp", action="store_true",
                        help="Post-process the predicted graph with the tracksdata ILP "
                             "solver (global, flow-consistent linking) instead of greedy "
                             "assignment. Needs pyscipopt; produces cleaner tracks.")
    parser.add_argument("--ilp-edge-weight", type=float, default=-1.0,
                        help="ILP: weight on edge_prob (default -1.0).")
    parser.add_argument("--ilp-appearance-weight", type=float, default=0.1,
                        help="ILP: cost of a track appearing (default 0.1).")
    parser.add_argument("--ilp-disappearance-weight", type=float, default=0.1,
                        help="ILP: cost of a track disappearing (default 0.1).")
    parser.add_argument("--ilp-division-weight", type=float, default=1.0,
                        help="ILP: cost of a division; lower to allow more splits (default 1.0).")

    args = parser.parse_args()

    from dataspec import DATASET_PATH
    data_dir = Path(args.data_dir) if args.data_dir else Path(DATASET_PATH)
    splits_file = Path(args.splits) if args.splits else data_dir / "dataset_splits.json"
    debug_video = Path(args.debug_video) if args.debug_video else None
    video_slice = (
        slice(*[int(x) if x else None for x in args.slice.split(":")])
        if args.slice else None
    )
    cfg = PredictConfig(
        det_threshold=args.det_threshold,
        use_ilp=args.use_ilp,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance_weight=args.ilp_appearance_weight,
        ilp_disappearance_weight=args.ilp_disappearance_weight,
        ilp_division_weight=args.ilp_division_weight,
    )

    folds = range(5) if args.split == "all" else [int(args.split)]

    for fold in folds:
        weights_path = (
            Path(args.weights) if args.weights
            else WEIGHTS_PATH / args.method / f"split_{fold}" / "edge_predictor_best.pth"
        )
        predict(
            data_dir=data_dir,
            fold=fold,
            splits_file=splits_file,
            weights_path=weights_path,
            cfg=cfg,
            method=args.method,
            debug_video=debug_video,
            unet_batch_size=args.unet_batch_size,
            video_slice=video_slice,
            evaluate=args.evaluate,
        )


if __name__ == "__main__":
    main()
