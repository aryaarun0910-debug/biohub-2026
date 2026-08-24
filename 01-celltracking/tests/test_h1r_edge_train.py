from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "kaggle_edits"))
sys.path.insert(0, str(ROOT / "vendor" / "kaggle-cell-tracking" / "scripts"))
import h1r_edge_data as D  # noqa: E402
import h1r_edge_train as E  # noqa: E402
import train_unet_transformer as T  # noqa: E402


def _fixture(tmp_path):
    nodes = {
        "f0": np.array([[0, 2, 2, 2], [0, 5, 5, 5], [0, 2, 5, 2]], np.float32),
        "f1": np.array([[1, 2, 2, 3], [1, 5, 4, 5], [1, 5, 6, 5], [1, 6, 1, 6]], np.float32),
    }
    ident = {"tid_0_0": np.array([10, 20, 30]), "pid_0_0": np.array([-1, -1, -1]),
             "tid_0_1": np.array([10, 21, 22, 40]), "pid_0_1": np.array([-1, 20, 20, -1])}
    np.savez(tmp_path / "zh001r_nodes.npz", **nodes)
    np.savez(tmp_path / "zh001r_identity.npz", **ident)
    rng = np.random.default_rng(1)
    np.save(tmp_path / "zh001r_iso.npy", rng.integers(0, 255, (1, 2, 8, 8, 8), dtype=np.uint8))
    data = D.Zh001rEdgeData(tmp_path / "zh001r_nodes.npz", tmp_path / "zh001r_identity.npz", n_frames=2)
    return E.Zh001rEdgeWindows(data, tmp_path / "zh001r_iso.npy", [0], node_cap=4, seed=9)


def test_epoch_sampling_is_reproducible_and_epoch_aware():
    a = E.epoch_sample_indices(20, 6, seed=4, epoch=2, crop=1, frame=7)
    assert np.array_equal(a, E.epoch_sample_indices(20, 6, seed=4, epoch=2, crop=1, frame=7))
    assert not np.array_equal(a, E.epoch_sample_indices(20, 6, seed=4, epoch=3, crop=1, frame=7))


def test_dataset_excludes_division_daughter_columns_from_continuation_targets(tmp_path):
    row = _fixture(tmp_path)[0]
    assert row["target"].sum().item() == 1
    assert row["division_cols"].tolist() == [False, True, True, False]


def test_division_columns_stay_masked_when_node_cap_drops_the_mother(tmp_path, monkeypatch):
    ds = _fixture(tmp_path)
    ds.node_cap = 2
    original = E.epoch_sample_indices

    def forced(n, cap, *, seed, epoch, crop, frame):
        # Source keeps tids 10 and 30, deliberately dropping dividing mother 20.
        if frame == 0:
            return np.array([0, 2])
        if frame == 1:
            return np.array([1, 2])
        return original(n, cap, seed=seed, epoch=epoch, crop=crop, frame=frame)

    monkeypatch.setattr(E, "epoch_sample_indices", forced)
    row = ds[0]
    selected_tids = row["tid1"].tolist()
    assert selected_tids == [21, 22]
    for j, tid in enumerate(selected_tids):
        if tid in (21, 22):
            assert bool(row["division_cols"][j])


def test_division_columns_have_exactly_zero_loss_gradient():
    logits = torch.randn(1, 2, 3, requires_grad=True)
    target = torch.zeros_like(logits); target[0, 0, 0] = 1
    masks = torch.ones(1, 2, dtype=torch.bool)
    target_mask = torch.ones(1, 3, dtype=torch.bool)
    division = torch.tensor([[False, True, False]])
    E.continuation_loss(logits, target, masks, target_mask, division).backward()
    assert torch.equal(logits.grad[0, :, 1], torch.zeros(2))


def test_public_model_initialization_is_strict_and_freezes_detector(tmp_path):
    cfg = {"unet_out_channels": 4, "unet_layers": [4, 8]}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    base = T.UNetNodeTransformer(T.TemporalUNet3D(1, 4, [4, 8]), 4,
                                 4 * T._POS_EMBED_DIM)
    weights = tmp_path / "edge_predictor_best.pth"
    torch.save(base.state_dict(), weights)
    loaded = E.load_public_full_model(T, weights, torch.device("cpu"), appearance_dim=8)
    assert all(not p.requires_grad for p in loaded.base.detect_head.parameters())
    assert all(torch.equal(v, loaded.base.state_dict()[k]) for k, v in base.state_dict().items())

    broken = dict(base.state_dict()); broken.pop(next(iter(broken)))
    torch.save(broken, weights)
    try:
        E.load_public_full_model(T, weights, torch.device("cpu"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("partial public checkpoint was accepted")


def test_synthetic_end_to_end_step_trains_unet_transformer_projector_only(tmp_path):
    ds = _fixture(tmp_path); batch = E.collate_edge_windows([ds[0]])
    unet = T.TemporalUNet3D(in_channels=1, out_channels=4, layers=[4, 8])
    base = T.UNetNodeTransformer(unet=unet, unet_out_channels=4,
                                 pos_feat_dim=4 * T._POS_EMBED_DIM,
                                 hidden_dim=16, n_heads=4, n_blocks=1, dropout=0.0)
    model = E.EdgeTrainingModel(base, appearance_dim=64)
    before = {k: v.clone() for k, v in model.base.detect_head.state_dict().items()}
    loss, _ = E.forward_batch(T, model, batch, torch.device("cpu"), triplet_weight=1.0)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.base.unet.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.base.transformer.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.projector.parameters())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3); opt.step()
    assert all(torch.equal(v, model.base.detect_head.state_dict()[k]) for k, v in before.items())


def test_resume_roundtrip_restores_every_training_state(tmp_path):
    base = T.UNetNodeTransformer(T.TemporalUNet3D(1, 4, [4, 8]), 4, 4 * T._POS_EMBED_DIM,
                                 hidden_dim=16, n_heads=4, n_blocks=1)
    model = E.EdgeTrainingModel(base, 8); opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = tmp_path / "resume.pth"
    E.save_resume(path, model, opt, sched, scaler, epoch=2, best=.4, history=[{"epoch": 2}])
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for p in model.parameters(): p.add_(1)
    start, best, history = E.load_resume(path, model, opt, sched, scaler, torch.device("cpu"))
    assert start == 3 and best == .4 and history == [{"epoch": 2}]
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in expected.items())


def test_s5_specs_attach_nodes_and_identity_and_never_submit():
    for name in ("h1r_edge_s5.json", "h1r_edge_s5_smoke.json"):
        spec = json.loads((ROOT / "scripts" / "kaggle_specs" / name).read_text())
        assert "kkunizaw/biohub-zh001r" in spec["datasets"]
        assert "aryaarun07/biohub-zh001r-identity" in spec["datasets"]
        assert spec["expects_submission"] is False
        assert spec["enable_gpu"] is True
        embedded = {edit.get("code_file") for edit in spec["edits"]}
        assert "scripts/kaggle_edits/h1r_edge_data.py" in embedded
        assert "scripts/kaggle_edits/h1r_edge_train.py" in embedded
        assert "scripts/kaggle_edits/h1r_edge_kernel_run.py" in embedded


def test_edge_trainer_recreates_vendored_src_import_path():
    source = (ROOT / "scripts" / "kaggle_edits" / "h1r_edge_train.py").read_text()
    assert 'str(trainer_dir.parent / "src")' in source
