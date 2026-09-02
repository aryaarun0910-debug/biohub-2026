"""The minimum Biohub-X definition needed to load the reference checkpoints.

The two published checkpoints are flat state dicts whose keys are ``unet.*``,
``detect_head.*`` and ``transformer.*``. The first and third belong to the
vendored CC0 modules; the second is a single 1x1x1 convolution that the upstream
training script composes around them. That composition is what this module
supplies, and it supplies nothing else.

It exists to answer one question: do the quarantined weights load and run. It is
not a model zoo and must not grow into one. When Biohub-X trains its own folds,
those models get their own definitions and their own registry entries; this
module is not where they go.

The weights it loads are ``reference_only``. One of the two trained on all 199
annotated movies, which include the four whose volumes are byte-identical to the
public-test volumes, so nothing this module produces may support a held-out
finding (D-0020, R-0002).

Consumer: ``biohubx infer reference``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from biohubx._vendor.reference_baseline.simple_node_transformer import SimpleNodeTransformer
from biohubx._vendor.reference_baseline.temporal_unet import TemporalUNet3D

REFERENCE_UNET_OUT_CHANNELS = 32
REFERENCE_UNET_LAYERS: tuple[int, ...] = (32, 64, 128)
REFERENCE_DOWNSAMPLE: tuple[int, ...] = (1, 4, 4)
REFERENCE_WINDOW_SIZE = 2
"""The published config.json, recorded here rather than read from an external file."""

REFERENCE_HIDDEN_DIM = 128
REFERENCE_HEADS = 4
REFERENCE_BLOCKS = 4
REFERENCE_DROPOUT = 0.3
"""Upstream transformer defaults. Wrong values fail strict loading rather than degrade quietly."""

PROJECTION_KEY = "transformer.proj.weight"
DETECT_HEAD_KEY = "detect_head.weight"


class ReferenceArchitectureError(ValueError):
    """A checkpoint does not describe the architecture this module can build."""


@dataclass(frozen=True, slots=True)
class ReferenceSpec:
    """Everything needed to instantiate the reference, derived not assumed."""

    unet_out_channels: int
    unet_layers: tuple[int, ...]
    pos_feat_dim: int
    window_size: int
    downsample: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "unet_out_channels": self.unet_out_channels,
            "unet_layers": list(self.unet_layers),
            "pos_feat_dim": self.pos_feat_dim,
            "window_size": self.window_size,
            "downsample": list(self.downsample),
        }


class ReferenceEdgeModel(nn.Module):
    """The vendored UNet and node transformer, plus the detection head between them."""

    def __init__(self, spec: ReferenceSpec) -> None:
        super().__init__()
        self.spec = spec
        self.unet = TemporalUNet3D(
            in_channels=1,
            out_channels=spec.unet_out_channels,
            layers=list(spec.unet_layers),
        )
        self.detect_head = nn.Conv3d(spec.unet_out_channels, 1, kernel_size=1)
        self.transformer = SimpleNodeTransformer(
            feat_dim=spec.unet_out_channels + spec.pos_feat_dim,
            hidden_dim=REFERENCE_HIDDEN_DIM,
            n_heads=REFERENCE_HEADS,
            n_blocks=REFERENCE_BLOCKS,
            dropout=REFERENCE_DROPOUT,
        )

    def detect(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the detection path only.

        ``window`` is ``(B, W, Z, Y, X)``. Returns the UNet feature volume
        ``(B, W, C, Z, Y, X)`` and the detection logits ``(B, W, 1, Z, Y, X)``.
        The edge transformer is deliberately not run: this repository has no
        proposal set to feed it, and pretending otherwise would produce a number
        that describes nothing.
        """
        if window.ndim != 5:
            raise ReferenceArchitectureError(
                f"expected a (B, W, Z, Y, X) window, got shape {tuple(window.shape)}"
            )
        features = self.unet(window.unsqueeze(2))
        logits = torch.stack(
            [self.detect_head(features[:, index]) for index in range(features.shape[1])], dim=1
        )
        return features, logits


def spec_from_state_dict(state: dict[str, torch.Tensor]) -> ReferenceSpec:
    """Read the architecture out of the checkpoint instead of assuming it.

    ``pos_feat_dim`` is not recorded anywhere in the published config; upstream
    computes it as four times a positional embedding constant. Deriving it from
    the projection width means a checkpoint built with a different embedding is
    detected here rather than at a shape error twenty frames into inference.
    """
    for key in (PROJECTION_KEY, DETECT_HEAD_KEY):
        if key not in state:
            raise ReferenceArchitectureError(f"checkpoint has no {key!r}; it is not this architecture")
    channels = int(state[DETECT_HEAD_KEY].shape[1])
    feat_dim = int(state[PROJECTION_KEY].shape[1])
    pos_feat_dim = feat_dim - channels
    if pos_feat_dim <= 0:
        raise ReferenceArchitectureError(
            f"projection width {feat_dim} does not exceed the {channels} UNet channels, "
            "so there is no room for a positional embedding"
        )
    return ReferenceSpec(
        unet_out_channels=channels,
        unet_layers=REFERENCE_UNET_LAYERS,
        pos_feat_dim=pos_feat_dim,
        window_size=REFERENCE_WINDOW_SIZE,
        downsample=REFERENCE_DOWNSAMPLE,
    )


def load_reference_model(weights: Path) -> tuple[ReferenceEdgeModel, ReferenceSpec]:
    """Load a quarantined checkpoint with ``strict=True`` and no gradients.

    Strict is the point. A tolerant load would accept a checkpoint whose keys
    only partly match and leave the rest randomly initialised, which is how a
    reference reproduction quietly becomes a different model.
    """
    if not weights.is_file():
        raise ReferenceArchitectureError(f"no checkpoint at {weights}")
    state: dict[str, torch.Tensor] = torch.load(weights, map_location="cpu", weights_only=True)
    spec = spec_from_state_dict(state)
    model = ReferenceEdgeModel(spec)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, spec
