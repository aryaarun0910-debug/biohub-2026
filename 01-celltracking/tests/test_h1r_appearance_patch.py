"""Contracts for the opt-in H1-R appearance embedding patch."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.kaggle_edits.h1r_appearance_patch import (
    H1RAppearanceEmbedding,
    apply_h1r_appearance_inference_patch,
    apply_h1r_appearance_patch,
    continuation_spatial_knn_triplet_loss,
)
from scripts.kaggle_edits.h1r_edge_loss_patch import apply_h1r_edge_loss_patch
from scripts.kaggle_edits.h1r_trainer_patch import apply_h1r_trainer_patch


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "vendor" / "kaggle-cell-tracking" / "scripts" / "train_unet_transformer.py"
PREDICT = ROOT / "vendor" / "kaggle-cell-tracking" / "scripts" / "predict_unet_transformer.py"


def _loss(
    src: torch.Tensor,
    tgt: torch.Tensor,
    src_tid: torch.Tensor,
    tgt_tid: torch.Tensor,
    tgt_pid: torch.Tensor,
    src_mask: torch.Tensor | None = None,
    tgt_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, n_src = src_tid.shape
    n_tgt = tgt_tid.shape[1]
    if src_mask is None:
        src_mask = torch.ones(batch, n_src, dtype=torch.bool)
    if tgt_mask is None:
        tgt_mask = torch.ones(batch, n_tgt, dtype=torch.bool)
    src_coords = torch.arange(n_src, dtype=torch.float32).view(1, n_src, 1).expand(batch, -1, 3)
    tgt_coords = torch.arange(n_tgt, dtype=torch.float32).view(1, n_tgt, 1).expand(batch, -1, 3)
    return continuation_spatial_knn_triplet_loss(
        src,
        tgt,
        src_coords,
        tgt_coords,
        src_tid,
        tgt_tid,
        tgt_pid,
        src_mask,
        tgt_mask,
        k=16,
        margin=0.3,
    )


def test_exact_string_patch_is_idempotent_and_legacy_config_defaults_off(tmp_path: Path) -> None:
    trainer = tmp_path / TRAINER.name
    predict = tmp_path / PREDICT.name
    trainer.write_text(TRAINER.read_text(encoding="utf-8"), encoding="utf-8")
    predict.write_text(PREDICT.read_text(encoding="utf-8"), encoding="utf-8")

    apply_h1r_appearance_patch(trainer)
    apply_h1r_appearance_inference_patch(predict)
    first_trainer = trainer.read_text(encoding="utf-8")
    first_predict = predict.read_text(encoding="utf-8")
    compile(first_trainer, str(trainer), "exec")
    compile(first_predict, str(predict), "exec")

    assert '"appearance_dim": appearance_dim' in first_trainer
    assert 'parser.add_argument("--appearance-dim", type=int, default=0' in first_trainer
    assert '"appearance_dim": 0' in first_predict
    assert 'appearance_dim=config.get("appearance_dim", 0)' in first_predict

    apply_h1r_appearance_patch(trainer)
    apply_h1r_appearance_inference_patch(predict)
    assert trainer.read_text(encoding="utf-8") == first_trainer
    assert predict.read_text(encoding="utf-8") == first_predict


def test_patch_stacks_after_existing_h1r_trainer_and_edge_loss_patches(tmp_path: Path) -> None:
    trainer = tmp_path / TRAINER.name
    trainer.write_text(TRAINER.read_text(encoding="utf-8"), encoding="utf-8")
    apply_h1r_trainer_patch(trainer)
    apply_h1r_edge_loss_patch(trainer)
    apply_h1r_appearance_patch(trainer)
    source = trainer.read_text(encoding="utf-8")
    compile(source, str(trainer), "exec")
    assert "_H1R_BG_TERM" in source
    assert "class H1RAppearanceEmbedding" in source
    assert '"appearance_dim": appearance_dim' in source


def test_legacy_noop_has_no_state_and_gate_zero_preserves_logits_exactly() -> None:
    base = torch.randn(2, 3, 4)
    src = torch.randn(2, 3, 32)
    tgt = torch.randn(2, 4, 32)

    legacy = H1RAppearanceEmbedding(32, 0)
    assert legacy.state_dict() == {}
    assert torch.equal(legacy(base, src, tgt), base)

    enabled = H1RAppearanceEmbedding(32, 64)
    out, emb_src, emb_tgt = enabled(base, src, tgt, return_embeddings=True)
    assert enabled.gate.item() == 0.0
    assert torch.equal(out, base)
    assert emb_src.shape == (2, 3, 64)
    assert emb_tgt.shape == (2, 4, 64)
    assert torch.allclose(emb_src.norm(dim=-1), torch.ones(2, 3), atol=1e-5)


def test_gate_and_triplet_loss_have_valid_gradients() -> None:
    torch.manual_seed(4)
    module = H1RAppearanceEmbedding(4, 4)
    base = torch.zeros(1, 2, 3, requires_grad=True)
    feat_src = torch.randn(1, 2, 4, requires_grad=True)
    feat_tgt = torch.randn(1, 3, 4, requires_grad=True)
    logits, emb_src, emb_tgt = module(base, feat_src, feat_tgt, return_embeddings=True)

    # The residual has a live gate gradient even though its initialized value is zero.
    logits.sum().backward(retain_graph=True)
    assert module.gate.grad is not None
    assert torch.isfinite(module.gate.grad)
    module.zero_grad(set_to_none=True)
    feat_src.grad = None
    feat_tgt.grad = None

    loss = _loss(
        emb_src,
        emb_tgt,
        torch.tensor([[10, 20]]),
        torch.tensor([[10, 20, 30]]),
        torch.tensor([[-1, -1, -1]]),
    )
    loss.backward()
    head_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in module.head.parameters()
        if parameter.grad is not None
    )
    assert torch.isfinite(loss)
    assert head_grad > 0
    assert feat_src.grad is not None and float(feat_src.grad.abs().sum()) > 0
    assert feat_tgt.grad is not None and float(feat_tgt.grad.abs().sum()) > 0


def test_padding_and_unknown_identities_are_excluded() -> None:
    src_valid = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    tgt_valid = torch.tensor([[[0.8, 0.2], [0.2, 0.8], [1.0, 1.0]]])
    src_tid_valid = torch.tensor([[10, 20]])
    tgt_tid_valid = torch.tensor([[10, 20, 30]])
    tgt_pid_valid = torch.full((1, 3), -1)
    reference = _loss(
        src_valid,
        tgt_valid,
        src_tid_valid,
        tgt_tid_valid,
        tgt_pid_valid,
    )

    # Padded and unknown nodes are deliberately chosen to be pathological hard negatives.
    src = torch.cat([src_valid, torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])], dim=1)
    tgt = torch.cat([tgt_valid, torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])], dim=1)
    src_tid = torch.tensor([[10, 20, -1, 999]])
    tgt_tid = torch.tensor([[10, 20, 30, -1, 999]])
    tgt_pid = torch.full((1, 5), -1)
    src_mask = torch.tensor([[True, True, True, False]])
    tgt_mask = torch.tensor([[True, True, True, True, False]])
    augmented = _loss(src, tgt, src_tid, tgt_tid, tgt_pid, src_mask, tgt_mask)
    assert torch.allclose(augmented, reference, atol=1e-6)


def test_division_daughter_is_neither_continuation_positive_nor_hard_negative() -> None:
    src = torch.tensor([[[1.0, 0.0]]])
    # continuation, real negative, daughter (daughter is identical to anchor on purpose)
    tgt_a = torch.tensor([[[0.8, 0.2], [0.0, 1.0], [1.0, 0.0]]])
    tgt_b = tgt_a.clone()
    tgt_b[0, 2] = torch.tensor([-1.0, 0.0])
    src_tid = torch.tensor([[10]])
    tgt_tid = torch.tensor([[10, 30, 21]])
    tgt_pid = torch.tensor([[-1, -1, 10]])

    loss_a = _loss(src, tgt_a, src_tid, tgt_tid, tgt_pid)
    loss_b = _loss(src, tgt_b, src_tid, tgt_tid, tgt_pid)
    assert torch.allclose(loss_a, loss_b, atol=1e-6)


def test_no_usable_triplet_returns_differentiable_zero() -> None:
    src = torch.randn(1, 1, 3, requires_grad=True)
    tgt = torch.randn(1, 1, 3, requires_grad=True)
    loss = _loss(
        src,
        tgt,
        torch.tensor([[-1]]),
        torch.tensor([[-1]]),
        torch.tensor([[-1]]),
    )
    assert loss.item() == 0.0
    loss.backward()
    assert src.grad is not None and torch.equal(src.grad, torch.zeros_like(src))
    assert tgt.grad is not None and torch.equal(tgt.grad, torch.zeros_like(tgt))


@pytest.mark.parametrize("bad_k", [0, -1])
def test_invalid_knn_size_is_rejected(bad_k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        continuation_spatial_knn_triplet_loss(
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 3),
            torch.zeros(1, 1, 3),
            torch.zeros(1, 1, dtype=torch.long),
            torch.zeros(1, 1, dtype=torch.long),
            torch.full((1, 1), -1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.bool),
            torch.ones(1, 1, dtype=torch.bool),
            k=bad_k,
        )
