"""Load the borrowed UNetNodeTransformer and run it on MPS.

Contract recovered from the support pack's train/predict scripts:
  downsample     (1, 4, 4)  -> spatial grid (64, 64, 64), isotropic 1.625 um
  normalisation  (x - q[0.001]) / (q[0.999] - q[0.001] + 1e-6), clamped >= 0
  window_size    2
  encode(imgs)   -> unet_out (B, W, 32, Z, Y, X), det_logits list of W x (B,1,Z,Y,X)
  peaks          max_pool3d local-max, kernel from 5.0 um -> (3,3,3), sigmoid > thr
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = ROOT / "weights" / "biohub-tracking-support-pack-50ep-v1"
SECONDARY = ROOT / "weights" / "biohub-temporal-unet3d-seed314159-v1"

DOWNSAMPLE = (1, 4, 4)
WINDOW = 2
POOL_KERNEL_UM = 5.0
DET_THRESHOLD = 0.965
# after (1,4,4) striding the grid is isotropic at the z pitch
VOXEL_DOWN_UM = (1.625, 1.625, 1.625)


def _import_models():
    p = str(SUPPORT / "repo" / "src")
    if p not in sys.path:
        sys.path.insert(0, p)
    from biohub_tracking.models.temporal_unet import TemporalUNet3D
    from biohub_tracking.models.simple_node_transformer import SimpleNodeTransformer
    return TemporalUNet3D, SimpleNodeTransformer


class UNetNodeTransformer(torch.nn.Module):
    """Reimplementation of the support pack's wrapper, inference paths only."""

    def __init__(self, cfg: dict, pos_feat_dim: int):
        super().__init__()
        TemporalUNet3D, SimpleNodeTransformer = _import_models()
        c = cfg["unet_out_channels"]
        self.unet = TemporalUNet3D(
            in_channels=1, out_channels=c, layers=tuple(cfg["unet_layers"]),
            gradient_checkpointing=False,
        )
        self.detect_head = torch.nn.Conv3d(c, 1, kernel_size=1)
        self.transformer = SimpleNodeTransformer(
            feat_dim=c + pos_feat_dim, hidden_dim=128, n_heads=4, n_blocks=4, dropout=0.3
        )

    @torch.no_grad()
    def encode(self, imgs: torch.Tensor):
        """imgs (B, W, Z, Y, X) already downsampled + normalised."""
        unet_out = self.unet(imgs.unsqueeze(2))            # (B, W, C, Z, Y, X)
        det = [self.detect_head(unet_out[:, i]) for i in range(unet_out.shape[1])]
        return unet_out, det


def load(which: str = "primary", device: str = "mps") -> tuple[UNetNodeTransformer, dict]:
    root = SUPPORT if which == "primary" else SECONDARY
    wdir = root / "weights" / "unet_transformer" / "split_0"
    cfg = json.loads((wdir / "config.json").read_text())
    sd = torch.load(wdir / "edge_predictor_best.pth", map_location="cpu", weights_only=False)
    if "model_state" in sd:
        sd = sd["model_state"]
    # derive the positional-embedding width from the checkpoint rather than
    # hardcoding it: proj takes [unet features | positional features]
    pos_feat_dim = int(sd["transformer.proj.weight"].shape[1]) - cfg["unet_out_channels"]
    cfg = {**cfg, "pos_feat_dim": pos_feat_dim}
    model = UNetNodeTransformer(cfg, pos_feat_dim)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")
    return model.to(device).eval(), cfg


def pool_kernel(um: float = POOL_KERNEL_UM, voxel=VOXEL_DOWN_UM) -> tuple[int, ...]:
    out = []
    for s in voxel:
        k = max(1, round(um / s))
        out.append(k + 1 if k % 2 == 0 else k)
    return tuple(out)


def normalise(raw: np.ndarray, q_low: float, q_high: float) -> torch.Tensor:
    """Strided-downsample a full-res frame and quantile-normalise it."""
    dz, dy, dx = DOWNSAMPLE
    f = torch.from_numpy(raw[::dz, ::dy, ::dx].astype(np.float32))
    return ((f - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)


def peaks(det_logits: torch.Tensor, threshold: float = DET_THRESHOLD,
          kernel: tuple[int, ...] | None = None) -> torch.Tensor:
    """(1, Z, Y, X) logits -> (N, 3) integer peak coords in the downsampled grid."""
    kernel = kernel or pool_kernel()
    lg = det_logits.reshape(1, 1, *det_logits.shape[-3:])
    pooled = F.max_pool3d(lg, kernel, stride=1, padding=tuple(k // 2 for k in kernel))
    is_peak = (lg == pooled) & (torch.sigmoid(lg) > threshold)
    return torch.nonzero(is_peak[0, 0])


def sample_features(unet_out: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Sub-voxel feature sampling at (N,3) float coords in the downsampled grid.

    grid_sample rather than integer indexing: the annotation carries its own
    localisation error, so we want the feature at the refined position. Native
    on MPS, verified no fallback.
    """
    C, Z, Y, X = unet_out.shape[-4:]
    dev, dt = unet_out.device, unet_out.dtype
    c = coords.to(dev, dt)
    # normalise to [-1,1] in (x, y, z) order, which is what grid_sample expects
    g = torch.stack([
        2 * c[:, 2] / max(X - 1, 1) - 1,
        2 * c[:, 1] / max(Y - 1, 1) - 1,
        2 * c[:, 0] / max(Z - 1, 1) - 1,
    ], dim=-1).reshape(1, 1, 1, -1, 3)
    out = F.grid_sample(unet_out.reshape(1, C, Z, Y, X), g,
                        mode="bilinear", align_corners=True, padding_mode="border")
    return out.reshape(C, -1).T  # (N, C)
