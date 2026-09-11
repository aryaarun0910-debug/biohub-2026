#!/usr/bin/env python
"""Train a full-frame 3D U-Net center-heatmap detector for Biohub tracking.

This is a separate detector backend from the UNet+transformer edge model. It
learns only cell centers:

    frame volume -> XY pooled volume -> center heatmap

The loss is positive-unlabelled: labelled centers are strong positives, dark
background is a normal negative, and bright unlabelled voxels receive a small
weight so sparse labels do not become hard false negatives.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)


@dataclass
class FullFrameTrainingConfig:
    seed: int = 2026
    pool_factor: int = 4
    base_channels: int = 24

    gauss_sigma: float = 1.0
    pos_thresh: float = 0.05
    bg_quantile: float = 0.40
    w_pos: float = 12.0
    w_bg: float = 1.0
    w_ignore: float = 0.05

    norm_lo_pct: float = 50.0
    norm_hi_pct: float = 99.5
    norm_clip_lo: float = -0.5
    norm_clip_hi: float = 6.0

    batch_size: int = 8
    epochs: int = 50
    frames_per_movie: int = 0
    movie_limit: int | None = None
    val_fraction: float = 0.10
    num_workers: int = 4

    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    grad_clip_norm: float | None = None

    random_flip: bool = True
    brightness_jitter: float = 0.0


def config_from_args(args: argparse.Namespace) -> FullFrameTrainingConfig:
    return FullFrameTrainingConfig(
        seed=args.seed,
        pool_factor=args.pool_factor,
        base_channels=args.base_channels,
        gauss_sigma=args.gauss_sigma,
        pos_thresh=args.pos_thresh,
        bg_quantile=args.bg_quantile,
        w_pos=args.w_pos,
        w_bg=args.w_bg,
        w_ignore=args.w_ignore,
        norm_lo_pct=args.norm_lo_pct,
        norm_hi_pct=args.norm_hi_pct,
        norm_clip_lo=args.norm_clip_lo,
        norm_clip_hi=args.norm_clip_hi,
        batch_size=args.batch_size,
        epochs=args.epochs,
        frames_per_movie=args.frames_per_movie,
        movie_limit=args.movie_limit,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        random_flip=not args.no_random_flip,
        brightness_jitter=args.brightness_jitter,
    )


def config_from_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> FullFrameTrainingConfig:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = dict(checkpoint.get("config", {}))
    valid_keys = {field.name for field in fields(FullFrameTrainingConfig)}
    defaults = asdict(FullFrameTrainingConfig())
    merged = {**defaults, **{k: v for k, v in saved.items() if k in valid_keys}}

    # A resume run should keep the trained model/loss/data geometry contract.
    # These runtime controls are safe to change when extending a run.
    merged["epochs"] = args.epochs
    merged["batch_size"] = args.batch_size
    merged["num_workers"] = args.num_workers
    return FullFrameTrainingConfig(**merged)


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeepCenterUNet3D(nn.Module):
    """Three-level 3D U-Net producing one center-heatmap logit volume."""

    def __init__(self, in_channels: int = 1, base_channels: int = 24) -> None:
        super().__init__()
        c = int(base_channels)
        self.enc1 = ConvBlock3d(in_channels, c)
        self.down1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock3d(c, c * 2)
        self.down2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock3d(c * 2, c * 4)
        self.down3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock3d(c * 4, c * 8)
        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3d(c * 8, c * 4)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3d(c * 4, c * 2)
        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3d(c * 2, c)
        self.head = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def read_zarr_meta(zarr_path: Path) -> tuple[tuple[int, ...], np.dtype]:
    meta = json.loads((zarr_path / "0" / "zarr.json").read_text())
    return tuple(int(v) for v in meta["shape"]), np.dtype(meta["data_type"]).newbyteorder("<")


def decompress_blosc(raw: bytes) -> bytes:
    try:
        import blosc2  # type: ignore

        return blosc2.decompress(raw)
    except Exception:
        from numcodecs import blosc  # type: ignore

        return blosc.decompress(raw)


def read_frame(zarr_path: Path, t: int, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    frame_shape = shape[1:]
    chunk = zarr_path / "0" / "c" / str(t) / "0" / "0" / "0"
    try:
        arr = np.frombuffer(decompress_blosc(chunk.read_bytes()), dtype=dtype)
        if arr.size == int(np.prod(frame_shape)):
            return arr.reshape(frame_shape).copy()
    except Exception:
        pass
    import zarr  # type: ignore

    return np.asarray(zarr.open(zarr_path / "0", mode="r")[t])


def block_mean_xy(volume: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return volume.astype(np.float32, copy=False)
    z, y, x = volume.shape
    y2 = (y // factor) * factor
    x2 = (x // factor) * factor
    cropped = volume[:, :y2, :x2].astype(np.float32, copy=False)
    return cropped.reshape(z, y2 // factor, factor, x2 // factor, factor).mean(axis=(2, 4))


def normalize_dynamic_range(
    volume: np.ndarray,
    lo_pct: float,
    hi_pct: float,
    clip_lo: float,
    clip_hi: float,
) -> np.ndarray:
    vol = np.asarray(volume, dtype=np.float32)
    lo, hi = np.percentile(vol, [lo_pct, hi_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    ratio = (vol - lo) / (hi - lo)
    return np.clip(ratio, clip_lo, clip_hi).astype(np.float32)


def read_geff_nodes(geff_path: Path) -> dict[int, np.ndarray]:
    """Return sparse labelled centers grouped by time."""

    try:
        import tracksdata as td  # type: ignore

        graph = td.graph.IndexedRXGraph.from_geff(geff_path)
        graph = graph[0] if isinstance(graph, tuple) else graph
        out: dict[int, list[list[float]]] = {}
        for row in graph.node_attrs().iter_rows(named=True):
            out.setdefault(int(row["t"]), []).append(
                [float(row["z"]), float(row["y"]), float(row["x"])]
            )
        return {
            t: np.asarray(coords, dtype=np.float32)
            for t, coords in out.items()
            if len(coords)
        }
    except Exception:
        pass

    import zarr  # type: ignore

    root = zarr.open_group(str(geff_path), mode="r")
    t_values = np.asarray(root["nodes/props/t/values"])
    z_values = np.asarray(root["nodes/props/z/values"])
    y_values = np.asarray(root["nodes/props/y/values"])
    x_values = np.asarray(root["nodes/props/x/values"])
    out_list: dict[int, list[list[float]]] = {}
    for t, z, y, x in zip(t_values, z_values, y_values, x_values):
        out_list.setdefault(int(t), []).append([float(z), float(y), float(x)])
    return {
        t: np.asarray(coords, dtype=np.float32)
        for t, coords in out_list.items()
        if len(coords)
    }


def make_heatmap(
    pooled_shape: tuple[int, int, int],
    centers_zyx: np.ndarray,
    pool_factor: int,
    sigma: float,
) -> np.ndarray:
    heatmap = np.zeros(pooled_shape, dtype=np.float32)
    if centers_zyx.size == 0:
        return heatmap
    radius = max(1, int(math.ceil(3.0 * sigma)))
    sigma2 = float(sigma) ** 2
    z_max, y_max, x_max = pooled_shape
    for z0, y0, x0 in centers_zyx:
        center = np.array([z0, y0 / pool_factor, x0 / pool_factor], dtype=np.float32)
        zc, yc, xc = [float(v) for v in center]
        z_start = max(0, int(math.floor(zc)) - radius)
        z_stop = min(z_max, int(math.floor(zc)) + radius + 2)
        y_start = max(0, int(math.floor(yc)) - radius)
        y_stop = min(y_max, int(math.floor(yc)) + radius + 2)
        x_start = max(0, int(math.floor(xc)) - radius)
        x_stop = min(x_max, int(math.floor(xc)) + radius + 2)
        if z_start >= z_stop or y_start >= y_stop or x_start >= x_stop:
            continue
        zz = np.arange(z_start, z_stop, dtype=np.float32)[:, None, None]
        yy = np.arange(y_start, y_stop, dtype=np.float32)[None, :, None]
        xx = np.arange(x_start, x_stop, dtype=np.float32)[None, None, :]
        d2 = (zz - zc) ** 2 + (yy - yc) ** 2 + (xx - xc) ** 2
        blob = np.exp(-0.5 * d2 / max(sigma2, 1e-6)).astype(np.float32)
        view = heatmap[z_start:z_stop, y_start:y_stop, x_start:x_stop]
        np.maximum(view, blob, out=view)
    return heatmap


def positive_unlabeled_weight_map(
    image: np.ndarray,
    heatmap: np.ndarray,
    cfg: FullFrameTrainingConfig,
) -> np.ndarray:
    weights = np.full(heatmap.shape, cfg.w_ignore, dtype=np.float32)
    bg_cutoff = float(np.quantile(image, cfg.bg_quantile))
    weights[image < bg_cutoff] = cfg.w_bg
    weights[heatmap > cfg.pos_thresh] = cfg.w_pos
    return weights


def flip_together(rng: np.random.Generator, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    shape = arrays[0].shape
    axes = tuple(axis for axis in range(len(shape)) if rng.random() < 0.5)
    if axes:
        arrays = tuple(np.flip(arr, axis=axes) for arr in arrays)
    return tuple(np.ascontiguousarray(arr, dtype=np.float32) for arr in arrays)


class FullFrameDataset(Dataset):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        cfg: FullFrameTrainingConfig,
        training: bool,
    ) -> None:
        self.samples = samples
        self.cfg = cfg
        self.training = training
        self.items: list[tuple[int, int]] = []
        rng = np.random.default_rng(cfg.seed + (0 if training else 10_000))
        for sample_idx, sample in enumerate(samples):
            n_t = int(sample["shape"][0])
            if cfg.frames_per_movie and cfg.frames_per_movie > 0 and cfg.frames_per_movie < n_t:
                frames = sorted(rng.choice(n_t, size=cfg.frames_per_movie, replace=False).tolist())
            else:
                frames = list(range(n_t))
            self.items.extend((sample_idx, int(t)) for t in frames)
        if not self.items:
            raise ValueError("No training frames were selected.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample_idx, t = self.items[index]
        sample = self.samples[sample_idx]
        frame = read_frame(sample["zarr"], t, sample["shape"], sample["dtype"])
        pooled = block_mean_xy(frame, self.cfg.pool_factor)
        image = normalize_dynamic_range(
            pooled,
            self.cfg.norm_lo_pct,
            self.cfg.norm_hi_pct,
            self.cfg.norm_clip_lo,
            self.cfg.norm_clip_hi,
        )
        target = make_heatmap(
            image.shape,
            sample["centers_by_t"].get(t, np.empty((0, 3), dtype=np.float32)),
            self.cfg.pool_factor,
            self.cfg.gauss_sigma,
        )
        weights = positive_unlabeled_weight_map(image, target, self.cfg)

        if self.training:
            rng = np.random.default_rng((self.cfg.seed + 1_000_003 * index) & 0xFFFFFFFF)
            if self.cfg.random_flip:
                image, target, weights = flip_together(rng, image, target, weights)
            if self.cfg.brightness_jitter > 0:
                scale = float(rng.uniform(1.0 - self.cfg.brightness_jitter, 1.0 + self.cfg.brightness_jitter))
                image = np.ascontiguousarray(image * scale, dtype=np.float32)

        return (
            torch.from_numpy(image[None, ...]),
            torch.from_numpy(target[None, ...]),
            torch.from_numpy(weights[None, ...]),
        )


def discover_samples(data_dir: Path, cfg: FullFrameTrainingConfig) -> list[dict[str, Any]]:
    zarrs = sorted(data_dir.glob("*.zarr"))
    if cfg.movie_limit is not None:
        zarrs = zarrs[: int(cfg.movie_limit)]
    samples: list[dict[str, Any]] = []
    for zarr_path in zarrs:
        geff_path = data_dir / f"{zarr_path.stem}.geff"
        if not geff_path.exists():
            continue
        shape, dtype = read_zarr_meta(zarr_path)
        centers_by_t = read_geff_nodes(geff_path)
        if not centers_by_t:
            continue
        samples.append(
            {
                "name": zarr_path.stem,
                "zarr": zarr_path,
                "geff": geff_path,
                "shape": shape,
                "dtype": dtype,
                "centers_by_t": centers_by_t,
            }
        )
    if not samples:
        raise FileNotFoundError(f"No paired .zarr/.geff samples with labels found in {data_dir}")
    return samples


def split_samples(
    samples: list[dict[str, Any]],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_embryo: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        embryo = str(sample["name"]).split("_", 1)[0]
        by_embryo.setdefault(embryo, []).append(sample)
    embryos = sorted(by_embryo)
    rng.shuffle(embryos)
    n_val = max(1, int(round(len(embryos) * val_fraction))) if len(embryos) > 1 else 0
    val_embryos = set(embryos[:n_val])
    train = [sample for sample in samples if str(sample["name"]).split("_", 1)[0] not in val_embryos]
    val = [sample for sample in samples if str(sample["name"]).split("_", 1)[0] in val_embryos]
    if not train and val:
        train, val = val, []
    return train, val


def weighted_bce_loss(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted = loss * weights
    return weighted.sum() / torch.clamp(weights.sum(), min=1.0)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: FullFrameTrainingConfig,
    epoch: int,
    best_score: float,
    history: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "config": asdict(cfg),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_score": float(best_score),
            "history": history,
        },
        tmp,
    )
    os.replace(tmp, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, list[dict[str, float]]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return (
        int(checkpoint.get("epoch", 0)),
        float(checkpoint.get("best_score", float("-inf"))),
        list(checkpoint.get("history", [])),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_idx, (image, target, weights) in enumerate(loader, start=1):
        image = image.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        weights = weights.to(device=device, dtype=torch.float32)
        logits = model(image)
        losses.append(float(weighted_bce_loss(logits, target, weights).detach().cpu()))
        if max_batches is not None and batch_idx >= max_batches:
            break
    return float(np.mean(losses)) if losses else float("nan")


def parse_float_list(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("Expected at least one comma-separated threshold value.")
    return sorted(set(values))


def peak_distance_um(
    peak_zyx: np.ndarray,
    gt_zyx: np.ndarray,
    pool_factor: int,
) -> np.ndarray:
    if gt_zyx.size == 0:
        return np.empty((0,), dtype=np.float32)
    xy_offset = (pool_factor - 1) / 2.0 if pool_factor > 1 else 0.0
    peak_orig = np.asarray(
        [
            peak_zyx[0],
            peak_zyx[1] * pool_factor + xy_offset,
            peak_zyx[2] * pool_factor + xy_offset,
        ],
        dtype=np.float32,
    )
    delta = (gt_zyx.astype(np.float32) - peak_orig[None, :]) * np.asarray(VOXEL_SCALE_UM, dtype=np.float32)
    return np.sqrt(np.sum(delta * delta, axis=1))


def find_heatmap_peaks(
    heatmap: np.ndarray,
    threshold: float,
    min_distance: int,
) -> tuple[np.ndarray, np.ndarray]:
    radius = max(0, int(min_distance))
    tensor = torch.from_numpy(np.asarray(heatmap, dtype=np.float32))[None, None]
    if radius > 0:
        kernel = 2 * radius + 1
        local = F.max_pool3d(tensor, kernel_size=kernel, stride=1, padding=radius)
    else:
        local = tensor
    mask = (tensor == local) & (tensor >= float(threshold))
    coords = torch.nonzero(mask[0, 0], as_tuple=False).cpu().numpy()
    if coords.size == 0:
        return coords.reshape(0, 3).astype(np.float32), np.empty((0,), dtype=np.float32)
    scores = heatmap[coords[:, 0], coords[:, 1], coords[:, 2]].astype(np.float32)
    order = np.argsort(-scores)
    return coords[order].astype(np.float32), scores[order].astype(np.float32)


def greedy_match_peaks(
    peaks: np.ndarray,
    scores: np.ndarray,
    gt_zyx: np.ndarray,
    pool_factor: int,
    match_radius_um: float,
) -> tuple[int, list[float], list[bool]]:
    if len(peaks) == 0:
        return 0, [], []
    used_gt: set[int] = set()
    nearest_distances: list[float] = []
    matched_flags: list[bool] = []
    for peak in peaks:
        distances = peak_distance_um(peak, gt_zyx, pool_factor)
        if distances.size == 0:
            nearest_distances.append(float("nan"))
            matched_flags.append(False)
            continue
        nearest_distances.append(float(np.min(distances)))
        order = np.argsort(distances)
        chosen = None
        for gt_idx in order:
            if int(gt_idx) in used_gt:
                continue
            if float(distances[gt_idx]) <= match_radius_um:
                chosen = int(gt_idx)
            break
        if chosen is None:
            matched_flags.append(False)
        else:
            used_gt.add(chosen)
            matched_flags.append(True)
    return len(used_gt), nearest_distances, matched_flags


@torch.no_grad()
def evaluate_gate_metrics(
    model: nn.Module,
    dataset: FullFrameDataset,
    cfg: FullFrameTrainingConfig,
    device: torch.device,
    output_dir: Path,
    thresholds: list[float],
    max_frames: int,
    peak_min_distance: int,
    match_radius_um: float,
    peak_sample_limit: int,
) -> dict[str, Any]:
    """Write threshold/peak diagnostics for later full-frame gate design.

    The sparse labels are not a complete cell inventory, so these metrics are
    calibration signals rather than leaderboard estimates. They are most useful
    for choosing conservative rescue thresholds and for comparing detector
    checkpoints under the same validation split.
    """

    if max_frames <= 0 or len(dataset) == 0:
        return {}

    model.eval()
    n_eval = min(int(max_frames), len(dataset))
    if n_eval == len(dataset):
        indices = list(range(len(dataset)))
    else:
        indices = np.linspace(0, len(dataset) - 1, n_eval).round().astype(int).tolist()

    min_threshold = min(thresholds)
    xy_offset = (cfg.pool_factor - 1) / 2.0 if cfg.pool_factor > 1 else 0.0
    accum = {
        threshold: {"frames": 0, "gt": 0, "pred": 0, "matched": 0}
        for threshold in thresholds
    }
    frame_rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []

    for eval_idx, dataset_idx in enumerate(indices, start=1):
        sample_idx, t = dataset.items[int(dataset_idx)]
        sample = dataset.samples[sample_idx]
        image, _, _ = dataset[int(dataset_idx)]
        logits = model(image[None].to(device=device, dtype=torch.float32))
        heatmap = torch.sigmoid(logits[0, 0]).detach().cpu().numpy().astype(np.float32)
        gt_zyx = sample["centers_by_t"].get(t, np.empty((0, 3), dtype=np.float32))

        base_peaks, base_scores = find_heatmap_peaks(heatmap, min_threshold, peak_min_distance)
        base_match_count, base_nearest, base_matched = greedy_match_peaks(
            base_peaks,
            base_scores,
            gt_zyx,
            cfg.pool_factor,
            match_radius_um,
        )
        if len(peak_rows) < peak_sample_limit:
            remaining = peak_sample_limit - len(peak_rows)
            for peak, score, nearest, matched in zip(
                base_peaks[:remaining],
                base_scores[:remaining],
                base_nearest[:remaining],
                base_matched[:remaining],
            ):
                peak_rows.append(
                    {
                        "dataset": sample["name"],
                        "t": int(t),
                        "z": float(peak[0]),
                        "y": float(peak[1]),
                        "x": float(peak[2]),
                        "z_orig": float(peak[0]),
                        "y_orig": float(peak[1] * cfg.pool_factor + xy_offset),
                        "x_orig": float(peak[2] * cfg.pool_factor + xy_offset),
                        "score": float(score),
                        "nearest_label_um": nearest,
                        "matched_within_radius": int(matched),
                    }
                )

        for threshold in thresholds:
            keep = base_scores >= float(threshold)
            peaks = base_peaks[keep]
            scores = base_scores[keep]
            matched_count, nearest_distances, matched_flags = greedy_match_peaks(
                peaks,
                scores,
                gt_zyx,
                cfg.pool_factor,
                match_radius_um,
            )
            n_gt = int(len(gt_zyx))
            n_pred = int(len(peaks))
            precision = matched_count / n_pred if n_pred else float("nan")
            recall = matched_count / n_gt if n_gt else float("nan")
            nearest_arr = np.asarray(
                [v for v in nearest_distances if np.isfinite(v)],
                dtype=np.float32,
            )
            frame_rows.append(
                {
                    "dataset": sample["name"],
                    "t": int(t),
                    "threshold": float(threshold),
                    "n_gt_sparse": n_gt,
                    "n_pred": n_pred,
                    "n_matched": int(matched_count),
                    "precision_sparse": precision,
                    "recall_sparse": recall,
                    "score_mean": float(np.mean(scores)) if len(scores) else float("nan"),
                    "score_p90": float(np.percentile(scores, 90)) if len(scores) else float("nan"),
                    "nearest_um_median": float(np.median(nearest_arr)) if nearest_arr.size else float("nan"),
                }
            )
            accum[threshold]["frames"] += 1
            accum[threshold]["gt"] += n_gt
            accum[threshold]["pred"] += n_pred
            accum[threshold]["matched"] += int(matched_count)

        if eval_idx == 1 or eval_idx % 25 == 0 or eval_idx == n_eval:
            print(f"gate-eval frame={eval_idx}/{n_eval}", flush=True)

    threshold_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        row = accum[threshold]
        n_pred = int(row["pred"])
        n_gt = int(row["gt"])
        matched = int(row["matched"])
        precision = matched / n_pred if n_pred else float("nan")
        recall = matched / n_gt if n_gt else float("nan")
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
            else float("nan")
        )
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "frames": int(row["frames"]),
                "n_gt_sparse": n_gt,
                "n_pred": n_pred,
                "n_matched": matched,
                "precision_sparse": precision,
                "recall_sparse": recall,
                "f1_sparse": f1,
                "pred_per_frame": n_pred / max(int(row["frames"]), 1),
                "gt_per_frame": n_gt / max(int(row["frames"]), 1),
            }
        )

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_dir / "gate_threshold_metrics.csv", threshold_rows)
    write_csv(output_dir / "gate_frame_metrics.csv", frame_rows)
    write_csv(output_dir / "gate_peak_samples.csv", peak_rows)

    summary = {
        "thresholds": thresholds,
        "max_frames": int(max_frames),
        "evaluated_frames": int(n_eval),
        "peak_min_distance": int(peak_min_distance),
        "match_radius_um": float(match_radius_um),
        "peak_sample_limit": int(peak_sample_limit),
        "threshold_metrics": threshold_rows,
        "notes": [
            "Sparse-label precision/recall are calibration features, not complete-cell metrics.",
            "Use high precision thresholds as conservative node-rescue gates.",
        ],
    }
    (output_dir / "gate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def append_history_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "val_loss", "score", "minutes"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def copy_if_exists(src: Path, dst: Path) -> dict[str, Any] | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "path": dst.name,
        "bytes": dst.stat().st_size,
    }


def evaluate_best_checkpoint_gate(
    checkpoint_path: Path,
    cfg: FullFrameTrainingConfig,
    val_ds: FullFrameDataset,
    device: torch.device,
    output_dir: Path,
    args: argparse.Namespace,
    max_frames: int,
) -> dict[str, Any]:
    if max_frames <= 0:
        return {}
    if not checkpoint_path.exists():
        print(f"Gate evaluation skipped: {checkpoint_path} was not found.")
        return {}

    eval_model = DeepCenterUNet3D(base_channels=cfg.base_channels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    eval_model.load_state_dict(checkpoint["model_state"])
    return evaluate_gate_metrics(
        model=eval_model,
        dataset=val_ds,
        cfg=cfg,
        device=device,
        output_dir=output_dir,
        thresholds=parse_float_list(args.gate_thresholds),
        max_frames=max_frames,
        peak_min_distance=args.gate_peak_min_distance,
        match_radius_um=args.gate_match_radius_um,
        peak_sample_limit=args.gate_peak_sample_limit,
    )


def write_epoch_snapshot(
    snapshot_root: Path,
    snapshot_prefix: str,
    output_dir: Path,
    epoch: int,
    cfg: FullFrameTrainingConfig,
    best_score: float,
    val_ds: FullFrameDataset,
    device: torch.device,
    args: argparse.Namespace,
) -> Path:
    snapshot_dir = snapshot_root / f"{snapshot_prefix}{epoch:04d}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, Any] = {}
    for name in [
        "best.pt",
        "checkpoint_last.pt",
        "config.json",
        "history.csv",
        "split_manifest.json",
    ]:
        info = copy_if_exists(output_dir / name, snapshot_dir / name)
        if info is not None:
            copied[name] = info

    gate_frames = (
        args.snapshot_gate_eval_frames
        if args.snapshot_gate_eval_frames is not None
        else args.gate_eval_frames
    )
    gate_summary = evaluate_best_checkpoint_gate(
        checkpoint_path=output_dir / "best.pt",
        cfg=cfg,
        val_ds=val_ds,
        device=device,
        output_dir=snapshot_dir,
        args=args,
        max_frames=int(gate_frames),
    )
    for name in [
        "gate_summary.json",
        "gate_threshold_metrics.csv",
        "gate_frame_metrics.csv",
        "gate_peak_samples.csv",
    ]:
        path = snapshot_dir / name
        if path.exists():
            copied[name] = {
                "path": name,
                "bytes": path.stat().st_size,
            }

    snapshot_manifest = {
        "snapshot_epoch": int(epoch),
        "snapshot_prefix": snapshot_prefix,
        "source_output_dir": str(output_dir),
        "best_score_so_far": float(best_score),
        "config": asdict(cfg),
        "gate_eval_frames": int(gate_frames),
        "gate_summary_written": bool(gate_summary),
        "files": copied,
        "notes": [
            "best.pt is the best validation-loss checkpoint observed up to this snapshot epoch.",
            "checkpoint_last.pt is the exact training state at this snapshot epoch.",
            "Gate diagnostics are computed from best.pt on the fixed validation split.",
        ],
    }
    (snapshot_dir / "SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(snapshot_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"snapshot written: {snapshot_dir}", flush=True)
    return snapshot_dir


def train(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    last_path = output_dir / "checkpoint_last.pt"
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"--resume requires {last_path}")
        cfg = config_from_checkpoint(last_path, args)
    else:
        cfg = config_from_args(args)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    data_dir = args.data_dir.expanduser().resolve()
    if not args.resume and output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty. Use --resume or --overwrite intentionally."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True) + "\n")

    print("== Full-frame center detector training ==")
    print("data_dir:", data_dir)
    print("output_dir:", output_dir)
    print(json.dumps(asdict(cfg), indent=2, sort_keys=True))

    samples = discover_samples(data_dir, cfg)
    train_samples, val_samples = split_samples(samples, cfg.val_fraction, cfg.seed)
    print(f"samples: total={len(samples)} train={len(train_samples)} val={len(val_samples)}")
    print("train sample examples:", [s["name"] for s in train_samples[:5]])
    print("val sample examples:", [s["name"] for s in val_samples[:5]])
    split_manifest = {
        "seed": int(cfg.seed),
        "val_fraction": float(cfg.val_fraction),
        "train": [str(sample["name"]) for sample in train_samples],
        "val": [str(sample["name"]) for sample in val_samples],
        "all": [str(sample["name"]) for sample in samples],
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n"
    )

    train_ds = FullFrameDataset(train_samples, cfg, training=True)
    val_ds = FullFrameDataset(val_samples or train_samples[:1], cfg, training=False)
    print(f"frames: train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=max(0, min(cfg.num_workers, 2)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        print("device:", torch.cuda.get_device_name(0))
    else:
        print("device:", device)

    model = DeepCenterUNet3D(base_channels=cfg.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.csv"
    snapshot_root = (
        args.snapshot_dir.expanduser().resolve()
        if args.snapshot_dir is not None
        else output_dir.parent / f"{output_dir.name}_snapshots"
    )

    start_epoch = 0
    best_score = float("-inf")
    history: list[dict[str, float]] = []
    if args.resume:
        start_epoch, best_score, history = load_checkpoint(last_path, model, optimizer, device)
        print(f"Resumed from epoch {start_epoch}; best_score={best_score:.6f}")

    total_batches = len(train_loader)
    last_completed_epoch = start_epoch
    snapshotted_epochs: set[int] = set()
    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        model.train()
        epoch_start = time.time()
        running = 0.0
        seen_batches = 0
        for batch_idx, (image, target, weights) in enumerate(train_loader, start=1):
            image = image.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)
            weights = weights.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = weighted_bce_loss(logits, target, weights)
            loss.backward()
            if cfg.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            running += float(loss.detach().cpu())
            seen_batches += 1
            if batch_idx == 1 or batch_idx % args.progress_interval == 0 or batch_idx == total_batches:
                avg = running / max(seen_batches, 1)
                pct = 100.0 * batch_idx / max(total_batches, 1)
                print(
                    f"epoch={epoch}/{cfg.epochs} batch={batch_idx}/{total_batches} "
                    f"({pct:.1f}%) train_loss={avg:.6f}",
                    flush=True,
                )

        train_loss = running / max(seen_batches, 1)
        val_loss = evaluate(model, val_loader, device, args.val_batches)
        score = -val_loss if np.isfinite(val_loss) else -train_loss
        minutes = (time.time() - epoch_start) / 60.0
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "score": float(score),
            "minutes": float(minutes),
        }
        history.append(row)
        append_history_csv(history_path, history)

        if score > best_score:
            best_score = score
            save_checkpoint(best_path, model, optimizer, cfg, epoch, best_score, history)
            print(f"new best epoch={epoch} score={best_score:.6f} val_loss={val_loss:.6f}")

        save_checkpoint(last_path, model, optimizer, cfg, epoch, best_score, history)
        last_completed_epoch = epoch
        print(
            f"epoch={epoch} done train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"best_score={best_score:.6f} minutes={minutes:.2f}",
            flush=True,
        )
        if args.snapshot_interval > 0 and epoch % args.snapshot_interval == 0:
            write_epoch_snapshot(
                snapshot_root=snapshot_root,
                snapshot_prefix=args.snapshot_prefix,
                output_dir=output_dir,
                epoch=epoch,
                cfg=cfg,
                best_score=best_score,
                val_ds=val_ds,
                device=device,
                args=args,
            )
            snapshotted_epochs.add(epoch)

    print("Training complete.")
    print("best:", best_path)
    print("last:", last_path)

    if args.gate_eval_frames > 0:
        if not best_path.exists():
            print("Gate evaluation skipped: best.pt was not found.")
        else:
            print("Loading best checkpoint for gate evaluation:", best_path)
            gate_summary = evaluate_best_checkpoint_gate(
                checkpoint_path=best_path,
                cfg=cfg,
                val_ds=val_ds,
                device=device,
                output_dir=output_dir,
                args=args,
                max_frames=args.gate_eval_frames,
            )
            if gate_summary:
                print("Gate evaluation complete:")
                print(json.dumps(gate_summary["threshold_metrics"], indent=2))

    if args.snapshot_final and last_completed_epoch > 0 and last_completed_epoch not in snapshotted_epochs:
        write_epoch_snapshot(
            snapshot_root=snapshot_root,
            snapshot_prefix=args.snapshot_prefix,
            output_dir=output_dir,
            epoch=last_completed_epoch,
            cfg=cfg,
            best_score=best_score,
            val_ds=val_ds,
            device=device,
            args=args,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pool-factor", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--gauss-sigma", type=float, default=1.0)
    parser.add_argument("--pos-thresh", type=float, default=0.05)
    parser.add_argument("--bg-quantile", type=float, default=0.40)
    parser.add_argument("--w-pos", type=float, default=12.0)
    parser.add_argument("--w-bg", type=float, default=1.0)
    parser.add_argument("--w-ignore", type=float, default=0.05)
    parser.add_argument("--norm-lo-pct", type=float, default=50.0)
    parser.add_argument("--norm-hi-pct", type=float, default=99.5)
    parser.add_argument("--norm-clip-lo", type=float, default=-0.5)
    parser.add_argument("--norm-clip-hi", type=float, default=6.0)

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frames-per-movie", type=int, default=0)
    parser.add_argument("--movie-limit", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--no-random-flip", action="store_true")
    parser.add_argument("--brightness-jitter", type=float, default=0.0)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--val-batches", type=int, default=24)
    parser.add_argument(
        "--gate-eval-frames",
        type=int,
        default=240,
        help="Validation frames to use for post-training gate calibration; set 0 to disable.",
    )
    parser.add_argument(
        "--gate-thresholds",
        default="0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80",
        help="Comma-separated heatmap thresholds for sparse GT calibration.",
    )
    parser.add_argument("--gate-peak-min-distance", type=int, default=1)
    parser.add_argument("--gate-match-radius-um", type=float, default=7.0)
    parser.add_argument("--gate-peak-sample-limit", type=int, default=50000)
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=0,
        help="Write best/last/config/history/split/gate diagnostics every N epochs; 0 disables.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Root directory for interval snapshots. Defaults to <output_dir>_snapshots beside output_dir.",
    )
    parser.add_argument("--snapshot-prefix", default="ep")
    parser.add_argument(
        "--snapshot-final",
        action="store_true",
        help="Also snapshot the final/resumed epoch if it was not already captured by --snapshot-interval.",
    )
    parser.add_argument(
        "--snapshot-gate-eval-frames",
        type=int,
        default=None,
        help="Validation frames for snapshot gate diagnostics. Defaults to --gate-eval-frames.",
    )
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
