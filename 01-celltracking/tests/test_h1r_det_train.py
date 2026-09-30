from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.kaggle_edits.h1r_det_train import (
    DEPLOYED_SIGMOID_THR,
    Zh001rWindows,
    _train_step,
    assert_trunk_unchanged,
    best_operating_point,
    configure_trunk_contract,
    encode_detection_logits,
    enforce_trunk_mode,
    eval_thresholds,
    global_max_nodes,
    selection_at_threshold,
    trunk_snapshot,
)


def test_eval_thresholds_includes_deployed_and_rejects_invalid() -> None:
    thresholds = eval_thresholds("0.5,0.9")
    assert "deployed" in thresholds
    assert len(thresholds) == 3
    assert np.isclose(
        thresholds["deployed"],
        np.log(DEPLOYED_SIGMOID_THR / (1.0 - DEPLOYED_SIGMOID_THR)),
    )
    with pytest.raises(ValueError):
        eval_thresholds("0.5,1.0")


def test_best_operating_point_breaks_f1_ties_on_precision() -> None:
    name, stats = best_operating_point({
        "loose": {"f1": 0.8, "precision": 0.7},
        "strict": {"f1": 0.8, "precision": 0.9},
    })
    assert name == "strict"
    assert stats["precision"] == 0.9


def test_checkpoint_selection_is_fixed_to_declared_threshold() -> None:
    val = {
        "p0.5": {"f1": 0.9, "precision": 0.8},
        "deployed": {"f1": 0.7, "precision": 0.95},
    }
    name, stats = selection_at_threshold(val)
    assert name == "deployed"
    assert stats["f1"] == 0.7
    with pytest.raises(KeyError):
        selection_at_threshold(val, "missing")


class _EquivariantDetector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def encode(self, imgs):
        self.calls += 1
        return None, [imgs[:, f].clone() for f in range(imgs.shape[1])]


def test_eight_view_tta_matches_identity_for_equivariant_detector() -> None:
    model = _EquivariantDetector()
    imgs = torch.arange(2 * 2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 2, 3, 4, 4)
    logits = encode_detection_logits(model, imgs, use_tta=True)
    assert model.calls == 8
    assert all(torch.equal(logits[f], imgs[:, f]) for f in range(2))


class _DetectorForStep(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def encode(self, imgs):
        return None, [self.weight * imgs[:, f] for f in range(imgs.shape[1])]


class _LossModule:
    @staticmethod
    def compute_detection_loss(logits, coords, masks, neg_weight):
        return (logits - 1.0).square().mean()


def test_train_step_proves_adamw_ran_even_with_zero_lr() -> None:
    model = _DetectorForStep()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    batch = {
        "imgs": torch.ones(2, 2, 2, 2, 2),
        "coords": torch.zeros(2, 2, 1, 3),
        "masks": torch.ones(2, 2, 1, dtype=torch.bool),
    }
    record = _train_step(
        _LossModule, model, batch, torch.device("cpu"), 0.1, optimizer, scaler
    )
    assert record["finite_loss"]
    assert record["finite_gradients"]
    assert record["optimizer_step_occurred"]
    assert record["optimizer_step_before"] == 0
    assert record["optimizer_step_after"] == 1
    assert model.weight.item() == 0.25


class _SharedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unet = torch.nn.Sequential(
            torch.nn.Conv3d(1, 2, 1), torch.nn.BatchNorm3d(2)
        )
        self.detect_head = torch.nn.Conv3d(2, 1, 1)


def test_freeze_contract_preserves_parameters_and_bn_buffers() -> None:
    model = _SharedModel()
    configure_trunk_contract(model, "freeze", 1e-4)
    model.train()
    enforce_trunk_mode(model, "freeze")
    assert not model.unet.training
    assert not any(p.requires_grad for p in model.unet.parameters())
    assert all(p.requires_grad for p in model.detect_head.parameters())
    before = trunk_snapshot(model)
    with torch.no_grad():
        model.detect_head.weight.add_(1)
    assert_trunk_unchanged(model, before)
    with torch.no_grad():
        model.unet[1].running_mean.add_(1)
    with pytest.raises(RuntimeError, match="frozen trunk contract violated"):
        assert_trunk_unchanged(model, before)


def test_adabn_control_requires_zero_lr() -> None:
    with pytest.raises(ValueError, match="requires H1R_LR=0"):
        configure_trunk_contract(_SharedModel(), "adabn_control", 1e-4)


def test_global_max_prevents_validation_truncation(tmp_path) -> None:
    np.save(tmp_path / "zh001r_iso.npy", np.zeros((2, 2, 4, 4, 4), dtype=np.uint8))
    np.savez(
        tmp_path / "zh001r_nodes.npz",
        f0=np.array([[0, 0, 0, 0]], dtype=np.float32),
        f1=np.array([[1, 1, 1, 1]], dtype=np.float32),
        f2=np.array([[0, 0, 0, 0]], dtype=np.float32),
        f3=np.array([
            [1, 0, 0, 0], [1, 1, 1, 1], [1, 2, 2, 2],
        ], dtype=np.float32),
    )
    max_nodes = global_max_nodes(tmp_path)
    assert max_nodes == 3
    validation = Zh001rWindows(tmp_path, [1], max_nodes=max_nodes)
    item = validation[0]
    assert item["coords"].shape == (2, 3, 3)
    assert item["masks"].sum().item() == 4


def test_detector_resume_discovery_searches_the_kaggle_input_mount(tmp_path, monkeypatch):
    """H1R_RESUME=1 was dead code on Kaggle and read as a safety net.

    /kaggle/working starts EMPTY on every batch run, so <out>/detector_last.pth cannot exist
    on a fresh push. A preempted 12h kernel therefore lost everything and silently restarted
    from epoch 0. A prior run's state can only arrive as an ATTACHED INPUT.
    """
    import scripts.kaggle_edits.h1r_det_train as mod

    out = tmp_path / "work"; out.mkdir()
    assert mod.discover_detector_resume(out) is None      # fresh run: nothing to resume

    local = out / "detector_last.pth"; local.write_bytes(b"x")
    assert mod.discover_detector_resume(out) == local     # prefers local when present

    # Fall back to an attached dataset when /kaggle/input is the only place state exists.
    fake_root = tmp_path / "input"
    attached = fake_root / "prior-run" / "detector_last.pth"
    attached.parent.mkdir(parents=True)
    attached.write_bytes(b"y")

    real_path = mod.Path

    class RoutedPath(type(real_path())):
        def __new__(cls, *a, **kw):
            if a and str(a[0]) == "/kaggle/input":
                return real_path(fake_root)
            return real_path(*a, **kw)

    monkeypatch.setattr(mod, "Path", RoutedPath)
    out2 = tmp_path / "work2"; out2.mkdir()
    assert mod.discover_detector_resume(out2) == attached
