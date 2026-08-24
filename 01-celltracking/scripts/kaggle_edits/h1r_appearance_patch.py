r"""H1-R appearance-embedding patch for the vendored association model.

This module patches COPIES of ``train_unet_transformer.py`` and
``predict_unet_transformer.py``.  It never edits the vendored files itself.  Every edit is
an exact-string replacement with an asserted occurrence count followed by a compile check.

The legacy path is intentionally the default: ``appearance_dim=0`` creates no parameters
and leaves edge logits unchanged.  Passing ``--appearance-dim 64`` adds a small projector
over the already-indexed 32-channel U-Net node features and a zero-initialized, bounded
cosine-similarity residual on the existing direct edge logits.

The inserted ``continuation_spatial_knn_triplet_loss`` helper is deliberately separate
from the vendored primary edge loss.  It learns only same-track continuations, excludes
unknown identities and division daughters from the negative pool, and mines the hardest
embedding negative among spatially nearby candidates.

Usage::

    python scripts/kaggle_edits/h1r_appearance_patch.py \
        --trainer /tmp/train_unet_transformer.py \
        --predict /tmp/predict_unet_transformer.py
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_APPEARANCE_SOURCE = r'''
class H1RAppearanceEmbedding(nn.Module):
    """Optional normalized node appearance head plus a bounded logit residual.

    ``appearance_dim=0`` is a parameter-free legacy path.  For an enabled head, the
    scalar gate starts at zero, so adding the module cannot perturb the initial logits.
    """

    def __init__(self, input_dim: int, appearance_dim: int = 0):
        super().__init__()
        if appearance_dim < 0:
            raise ValueError("appearance_dim must be non-negative")
        self.appearance_dim = int(appearance_dim)
        self.head: nn.Module | None = None
        self.register_parameter("gate", None)
        if self.appearance_dim:
            self.head = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, 64),
                nn.GELU(),
                nn.Linear(64, self.appearance_dim),
            )
            self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        base_logits: torch.Tensor,
        feat_src: torch.Tensor,
        feat_tgt: torch.Tensor,
        mask_src: torch.Tensor | None = None,
        mask_tgt: torch.Tensor | None = None,
        return_embeddings: bool = False,
    ):
        if self.head is None:
            if return_embeddings:
                return base_logits, None, None
            return base_logits

        raw_src = self.head(feat_src)
        raw_tgt = self.head(feat_tgt)
        # Normalize in fp32 under AMP, then return to the model activation dtype.
        emb_src = F.normalize(raw_src.float(), p=2, dim=-1).to(raw_src.dtype)
        emb_tgt = F.normalize(raw_tgt.float(), p=2, dim=-1).to(raw_tgt.dtype)
        if mask_src is not None:
            emb_src = emb_src * mask_src.unsqueeze(-1).to(emb_src.dtype)
        if mask_tgt is not None:
            emb_tgt = emb_tgt * mask_tgt.unsqueeze(-1).to(emb_tgt.dtype)

        similarity = torch.bmm(emb_src, emb_tgt.transpose(1, 2))
        # Exactly zero at initialization and bounded to [-2, 2] thereafter.
        residual_scale = 2.0 * torch.tanh(self.gate)
        logits = base_logits + residual_scale * similarity
        if return_embeddings:
            return logits, emb_src, emb_tgt
        return logits


def continuation_spatial_knn_triplet_loss(
    emb_src: torch.Tensor,
    emb_tgt: torch.Tensor,
    coords_src: torch.Tensor,
    coords_tgt: torch.Tensor,
    track_id_src: torch.Tensor,
    track_id_tgt: torch.Tensor,
    parent_track_id_tgt: torch.Tensor,
    mask_src: torch.Tensor,
    mask_tgt: torch.Tensor,
    *,
    k: int = 16,
    margin: float = 0.3,
    max_distance: float | None = None,
) -> torch.Tensor:
    """Symmetric continuation-only batch-hard triplet loss.

    Coordinates must already be in one physical coordinate system.  For every anchor
    that has a same-track continuation, the farthest same-ID positive is contrasted with
    the closest embedding negative among the ``k`` spatially nearest eligible candidates.
    A target whose ``parent_track_id`` equals the source ID is a daughter: it is excluded
    rather than incorrectly taught as a negative.  IDs below zero and padded nodes never
    participate.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if max_distance is not None and max_distance <= 0:
        raise ValueError("max_distance must be positive when provided")
    if emb_src.ndim != 3 or emb_tgt.ndim != 3:
        raise ValueError("embeddings must have shape (B, N, E)")
    if emb_src.shape[0] != emb_tgt.shape[0] or emb_src.shape[2] != emb_tgt.shape[2]:
        raise ValueError("source and target embedding batch/feature dimensions must match")

    expected = (
        (coords_src, emb_src.shape[:2] + (3,), "coords_src"),
        (coords_tgt, emb_tgt.shape[:2] + (3,), "coords_tgt"),
        (track_id_src, emb_src.shape[:2], "track_id_src"),
        (track_id_tgt, emb_tgt.shape[:2], "track_id_tgt"),
        (parent_track_id_tgt, emb_tgt.shape[:2], "parent_track_id_tgt"),
        (mask_src, emb_src.shape[:2], "mask_src"),
        (mask_tgt, emb_tgt.shape[:2], "mask_tgt"),
    )
    for value, shape, name in expected:
        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {tuple(shape)}")

    src = F.normalize(emb_src.float(), p=2, dim=-1)
    tgt = F.normalize(emb_tgt.float(), p=2, dim=-1)
    # Squared Euclidean distance for normalized embeddings, matching triplet-loss form.
    embedding_distance = (2.0 - 2.0 * torch.bmm(src, tgt.transpose(1, 2))).clamp_min(0.0)
    spatial_distance = torch.cdist(coords_src.float().detach(), coords_tgt.float().detach())

    def _mine_direction(
        distance: torch.Tensor,
        spatial: torch.Tensor,
        anchor_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        anchor_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        candidate_parent_ids: torch.Tensor | None = None,
        anchor_parent_ids: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        losses: list[torch.Tensor] = []
        for batch_index in range(distance.shape[0]):
            valid_candidates = candidate_mask[batch_index].bool() & (
                candidate_ids[batch_index] >= 0
            )
            valid_anchors = anchor_mask[batch_index].bool() & (
                anchor_ids[batch_index] >= 0
            )
            for anchor_index in torch.where(valid_anchors)[0].tolist():
                anchor_id = anchor_ids[batch_index, anchor_index]
                positive = valid_candidates & (candidate_ids[batch_index] == anchor_id)
                if not bool(positive.any()):
                    continue

                negative = valid_candidates & (candidate_ids[batch_index] != anchor_id)
                if candidate_parent_ids is not None:
                    # Forward direction: daughters of this source are lineage-related.
                    negative = negative & (
                        candidate_parent_ids[batch_index] != anchor_id
                    )
                if anchor_parent_ids is not None:
                    # Reverse direction: the candidate source matching this target's parent
                    # is lineage-related and must not be mined as a negative.
                    parent_id = anchor_parent_ids[batch_index, anchor_index]
                    if bool(parent_id >= 0):
                        negative = negative & (
                            candidate_ids[batch_index] != parent_id
                        )
                if max_distance is not None:
                    negative = negative & (
                        spatial[batch_index, anchor_index] <= max_distance
                    )
                negative_index = torch.where(negative)[0]
                if negative_index.numel() == 0:
                    continue

                nearest_count = min(k, int(negative_index.numel()))
                nearest_order = torch.topk(
                    spatial[batch_index, anchor_index, negative_index],
                    nearest_count,
                    largest=False,
                ).indices
                hard_pool = negative_index[nearest_order]
                hardest_positive = distance[batch_index, anchor_index, positive].max()
                hardest_negative = distance[batch_index, anchor_index, hard_pool].min()
                losses.append(F.relu(hardest_positive - hardest_negative + margin))
        return losses

    forward = _mine_direction(
        embedding_distance,
        spatial_distance,
        track_id_src,
        track_id_tgt,
        mask_src,
        mask_tgt,
        candidate_parent_ids=parent_track_id_tgt,
    )
    reverse = _mine_direction(
        embedding_distance.transpose(1, 2),
        spatial_distance.transpose(1, 2),
        track_id_tgt,
        track_id_src,
        mask_tgt,
        mask_src,
        anchor_parent_ids=parent_track_id_tgt,
    )

    # Keep an autograd-connected zero for batches with no usable triplets.
    zero = (emb_src.sum() + emb_tgt.sum()) * 0.0
    directional_means = []
    if forward:
        directional_means.append(torch.stack(forward).mean())
    if reverse:
        directional_means.append(torch.stack(reverse).mean())
    if not directional_means:
        return zero
    return torch.stack(directional_means).mean() + zero
'''

# Make the exact implementation inserted into the trainer available to unit tests and to
# callers that want to prepare the loss before patching a notebook copy.
exec(_APPEARANCE_SOURCE, globals())


_CLASS_ANCHOR = "class UNetNodeTransformer(nn.Module):\n"
_CLASS_INSERT = _APPEARANCE_SOURCE + "\n\n" + _CLASS_ANCHOR

_INIT_SIG_OLD = '''        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        dropout: float = 0.3,
    ):
'''
_INIT_SIG_NEW = '''        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        dropout: float = 0.3,
        appearance_dim: int = 0,
    ):
'''

_INIT_BODY_OLD = '''        self.unet = unet
        self.unet_out_channels = unet_out_channels

        self.detect_head = nn.Conv3d(unet_out_channels, 1, kernel_size=1)
'''
_INIT_BODY_NEW = '''        self.unet = unet
        self.unet_out_channels = unet_out_channels
        # appearance_dim=0 is the parameter-free, checkpoint-compatible legacy path.
        self.appearance = H1RAppearanceEmbedding(unet_out_channels, appearance_dim)

        self.detect_head = nn.Conv3d(unet_out_channels, 1, kernel_size=1)
'''

_PREDICT_SIG_OLD = '''        mask_src: torch.Tensor,       # (B, N_src) bool
        mask_tgt: torch.Tensor,       # (B, N_tgt) bool
    ) -> torch.Tensor:
'''
_PREDICT_SIG_NEW = '''        mask_src: torch.Tensor,       # (B, N_src) bool
        mask_tgt: torch.Tensor,       # (B, N_tgt) bool
        return_appearance: bool = False,
    ):
'''

_PREDICT_BODY_OLD = '''        feat_src = torch.cat([unet_feat_src, pos_feat_src], dim=-1)
        feat_tgt = torch.cat([unet_feat_tgt, pos_feat_tgt], dim=-1)
        return self.transformer(feat_src, feat_tgt, coords_src, coords_tgt, mask_src, mask_tgt)
'''
_PREDICT_BODY_NEW = '''        feat_src = torch.cat([unet_feat_src, pos_feat_src], dim=-1)
        feat_tgt = torch.cat([unet_feat_tgt, pos_feat_tgt], dim=-1)
        base_logits = self.transformer(
            feat_src, feat_tgt, coords_src, coords_tgt, mask_src, mask_tgt,
        )
        return self.appearance(
            base_logits, unet_feat_src, unet_feat_tgt, mask_src, mask_tgt,
            return_embeddings=return_appearance,
        )
'''

_TRAIN_SIG_OLD = '''    pool_kernel_um: float = 5.0,
    data_parallel: bool = True,
) -> UNetNodeTransformer:
'''
_TRAIN_SIG_NEW = '''    pool_kernel_um: float = 5.0,
    data_parallel: bool = True,
    appearance_dim: int = 0,
) -> UNetNodeTransformer:
'''

_CONFIG_OLD = '''        "window_size": window_size,
        "pool_kernel_um": pool_kernel_um,
'''
_CONFIG_NEW = '''        "window_size": window_size,
        "pool_kernel_um": pool_kernel_um,
        "appearance_dim": appearance_dim,
'''

_MODEL_BUILD_OLD = '''        unet=unet,
        unet_out_channels=unet_out_channels,
        pos_feat_dim=pos_feat_dim,
    ).to(device)
'''
_MODEL_BUILD_NEW = '''        unet=unet,
        unet_out_channels=unet_out_channels,
        pos_feat_dim=pos_feat_dim,
        appearance_dim=appearance_dim,
    ).to(device)
'''

_PARSER_OLD = '''    parser.add_argument("--unet-out-channels", type=int, default=32)
'''
_PARSER_NEW = '''    parser.add_argument("--unet-out-channels", type=int, default=32)
    parser.add_argument("--appearance-dim", type=int, default=0,
                        help="Appearance embedding width; 0=legacy/off, 64=H1-R opt-in.")
'''

_MAIN_CALL_OLD = '''            unet_out_channels=args.unet_out_channels,
            unet_layers=unet_layers,
'''
_MAIN_CALL_NEW = '''            unet_out_channels=args.unet_out_channels,
            unet_layers=unet_layers,
            appearance_dim=args.appearance_dim,
'''

TRAINER_PATCHES = (
    (_CLASS_ANCHOR, _CLASS_INSERT, 1),
    (_INIT_SIG_OLD, _INIT_SIG_NEW, 1),
    (_INIT_BODY_OLD, _INIT_BODY_NEW, 1),
    (_PREDICT_SIG_OLD, _PREDICT_SIG_NEW, 1),
    (_PREDICT_BODY_OLD, _PREDICT_BODY_NEW, 1),
    (_TRAIN_SIG_OLD, _TRAIN_SIG_NEW, 1),
    (_CONFIG_OLD, _CONFIG_NEW, 1),
    (_MODEL_BUILD_OLD, _MODEL_BUILD_NEW, 1),
    (_PARSER_OLD, _PARSER_NEW, 1),
    (_MAIN_CALL_OLD, _MAIN_CALL_NEW, 1),
)

_INFER_CONFIG_OLD = '''    "window_size": 2,
}
'''
_INFER_CONFIG_NEW = '''    "window_size": 2,
    "appearance_dim": 0,
}
'''

_INFER_MODEL_OLD = '''        unet=unet,
        unet_out_channels=config["unet_out_channels"],
        pos_feat_dim=4 * _POS_EMBED_DIM,
    )
'''
_INFER_MODEL_NEW = '''        unet=unet,
        unet_out_channels=config["unet_out_channels"],
        pos_feat_dim=4 * _POS_EMBED_DIM,
        appearance_dim=config.get("appearance_dim", 0),
    )
'''

INFERENCE_PATCHES = (
    (_INFER_CONFIG_OLD, _INFER_CONFIG_NEW, 1),
    (_INFER_MODEL_OLD, _INFER_MODEL_NEW, 1),
)


def _apply_exact(path: Path | str, patches, marker: str, label: str) -> None:
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    if marker in source:
        print(f"h1r_appearance_patch: {path} already patched -- skipping")
        return
    for index, (old, new, expected) in enumerate(patches):
        count = source.count(old)
        if count != expected:
            raise AssertionError(
                f"h1r_appearance_patch: {label} patch {index} matched {count} times "
                f"(expected {expected}) in {path}"
            )
        source = source.replace(old, new, expected)
    compile(source, str(path), "exec")
    path.write_text(source, encoding="utf-8")
    print(f"h1r_appearance_patch: {len(patches)} {label} patches applied to {path}")


def apply_h1r_appearance_patch(trainer_path: Path | str) -> None:
    """Patch a COPY of the vendored trainer/model in place."""
    _apply_exact(
        trainer_path,
        TRAINER_PATCHES,
        "class H1RAppearanceEmbedding",
        "trainer",
    )


def apply_h1r_appearance_inference_patch(predict_path: Path | str) -> None:
    """Patch a COPY of the inference config and model constructor in place."""
    _apply_exact(
        predict_path,
        INFERENCE_PATCHES,
        '"appearance_dim": 0',
        "inference",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", help="path to a COPY of train_unet_transformer.py")
    parser.add_argument("--predict", help="path to a COPY of predict_unet_transformer.py")
    args = parser.parse_args()
    if args.trainer:
        apply_h1r_appearance_patch(args.trainer)
    if args.predict:
        apply_h1r_appearance_inference_patch(args.predict)
    if not (args.trainer or args.predict):
        parser.error("nothing to do: pass --trainer and/or --predict")
