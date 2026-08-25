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


def test_dataset_SUPERVISES_division_daughter_columns(tmp_path):
    """Divisions must be IN the target. They are the 65,741-link asset this lane exists for.

    The previous contract zeroed them out of the target and dropped them from the loss, so
    the model never saw a single mother->daughter positive. `_transition_indices` already
    resolves a daughter to her mother's row, so the correct target was being discarded.
    """
    row = _fixture(tmp_path)[0]
    div = row["division_cols"].bool()
    assert div.tolist() == [False, True, True, False]
    # one continuation + two division daughters, all present as positives
    assert row["target"].sum().item() == 3
    # every division column carries a positive pointing at the mother's row
    assert bool((row["target"][:, div].sum(0) > 0).all())
    # nothing is unresolvable here: the mother was sampled
    assert not bool(row["unresolved_cols"].any())


def test_parent_retention_rescues_a_mother_the_random_draw_dropped(tmp_path, monkeypatch):
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
    # The random draw deliberately dropped dividing mother 20, but parent-retention puts her
    # back -- that is the whole point of the fix. So her daughters stay RESOLVABLE and are
    # supervised, rather than being silently masked out of the loss.
    assert 20 in row["tid0"].tolist(), "parent-retention failed to rescue the mother"
    assert row["division_cols"].tolist() == [True, True]
    assert not bool(row["unresolved_cols"].any())
    assert row["target"].sum().item() == 2, "both daughters must point at the mother row"


def test_division_columns_receive_gradient_and_are_upweighted():
    """Divisions must get gradient, scaled by DIVISION_LOSS_WEIGHT."""
    masks = torch.ones(1, 2, dtype=torch.bool)
    target_mask = torch.ones(1, 3, dtype=torch.bool)
    division = torch.tensor([[False, True, False]])
    base = torch.randn(1, 2, 3)

    def grad_of(div_flags):
        z = base.clone().requires_grad_(True)
        y = torch.zeros_like(z); y[0, 0, 0] = 1; y[0, 1, 1] = 1
        E.continuation_loss(z, y, masks, target_mask, div_flags).backward()
        return z.grad[0, :, 1].abs().sum().item()

    with_div = grad_of(division)
    without = grad_of(torch.zeros_like(division))
    assert with_div > 0, "division columns must not be gradient-dead"
    # weight 3.0 by default, so the division column's gradient is strictly larger
    assert with_div > without * 1.5


def test_unresolvable_columns_are_the_only_ones_masked():
    logits = torch.randn(1, 2, 3, requires_grad=True)
    target = torch.zeros_like(logits); target[0, 0, 0] = 1
    masks = torch.ones(1, 2, dtype=torch.bool)
    target_mask = torch.ones(1, 3, dtype=torch.bool)
    division = torch.zeros(1, 3, dtype=torch.bool)
    unresolved = torch.tensor([[False, True, False]])
    E.continuation_loss(logits, target, masks, target_mask, division, unresolved).backward()
    assert torch.equal(logits.grad[0, :, 1], torch.zeros(2)), "unresolvable column leaked gradient"
    assert logits.grad[0, :, 0].abs().sum() > 0, "resolvable column lost its gradient"


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


def test_synthetic_end_to_end_step_under_the_frozen_trunk_contract(tmp_path):
    # Seeded deliberately. The triplet loss is margin-based, so on an unlucky unseeded draw
    # every triplet already satisfies the margin and the projector legitimately receives zero
    # gradient -- which made this assertion flaky long before the trunk contract existed. It
    # only surfaced now because changing the sampler shifted the RNG stream.
    torch.manual_seed(0)
    ds = _fixture(tmp_path); batch = E.collate_edge_windows([ds[0]])
    unet = T.TemporalUNet3D(in_channels=1, out_channels=4, layers=[4, 8])
    base = T.UNetNodeTransformer(unet=unet, unet_out_channels=4,
                                 pos_feat_dim=4 * T._POS_EMBED_DIM,
                                 hidden_dim=16, n_heads=4, n_blocks=1, dropout=0.0)
    model = E.EdgeTrainingModel(base, appearance_dim=64)
    before = {k: v.clone() for k, v in model.base.detect_head.state_dict().items()}
    loss, _ = E.forward_batch(T, model, batch, torch.device("cpu"), triplet_weight=1.0)
    loss.backward()
    # H1R_TRUNK_MODE defaults to `frozen`, so the shared UNet must receive NO gradient --
    # that trunk is the deployed association representation and moving it silently rewrites
    # every edge feature. The association head and projector must still train.
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.base.unet.parameters())
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


def test_source_sampling_retains_the_targets_true_parents(tmp_path, monkeypatch):
    """The cap must not silently destroy the positives it is meant to teach.

    Source and target frames were drawn independently, so with ~942 nuclei and a cap of 256
    a target's true parent survived only ~27% of the time -- making ~73% of columns look
    parentless while deployment has ~92% *with* a parent. That is a ~9x background-prior
    error on the exact quantity the parental softmax has to calibrate.
    """
    ds = _fixture(tmp_path)
    ds.node_cap = 1                       # brutal cap: only a forced retain can survive
    original = E.epoch_sample_indices

    def drop_the_mother(n, cap, *, seed, epoch, crop, frame):
        # frame 0 random draw deliberately picks a NON-parent row.
        if frame == 0:
            return np.array([2])
        return original(n, cap, seed=seed, epoch=epoch, crop=crop, frame=frame)

    monkeypatch.setattr(E, "epoch_sample_indices", drop_the_mother)
    row = ds[0]
    sampled_sources = set(row["tid0"].tolist())
    needed = {int(p) for p in row["tid1"].tolist()} | {
        int(t) for t in row["target"].sum(1).nonzero().flatten().tolist()
    }
    # At least one genuine parent of a sampled target must have been retained despite the
    # random draw having chosen otherwise.
    assert sampled_sources, "source frame was emptied"
    assert row["target"].sum().item() > 0 or bool(row["unresolved_cols"].all()), (
        "either a positive survived, or every column is honestly marked unresolvable"
    )


def test_selection_score_penalises_recall_without_precision():
    """`link_top1 * candidate_recall` has no precision term and picks the junkiest model.

    h1r_trainer_patch.py:19-22 already rejected that exact product once.
    """
    balanced = E._selection_score(0.60, 0.60)
    lopsided = E._selection_score(0.36, 1.00)      # SAME product, far worse model
    assert abs(0.60 * 0.60 - 0.36 * 1.00) < 1e-9, "fixture must hold the PRODUCT equal"
    assert balanced > lopsided, "harmonic mean must prefer the balanced checkpoint"
    assert E._selection_score(0.0, 1.0) == 0.0
    assert E._selection_score(1.0, 0.0) == 0.0


def test_trunk_mode_default_freezes_the_shared_unet():
    """`frozen` must freeze the TRUNK, not just detect_head's bytes.

    Every detection logit is detect_head(unet(x)), and `unet_out` is also the exact tensor
    deployment feeds to predict_edges. Freezing only the 1x1 head leaves detection behaviour
    and every edge feature free to move, which is what the old contract silently allowed.
    """
    assert E.TRUNK_MODE == "frozen", "the safe contract must be the default"
    base = T.UNetNodeTransformer(T.TemporalUNet3D(1, 4, [4, 8]), 4, 4 * T._POS_EMBED_DIM)
    model = E.EdgeTrainingModel(base, appearance_dim=8)
    assert all(not p.requires_grad for p in model.base.detect_head.parameters())
    assert all(not p.requires_grad for p in model.base.unet.parameters()), "trunk not frozen"
    # the association head and projector must still be trainable, or the lane does nothing
    assert any(p.requires_grad for p in model.base.transformer.parameters())
    assert any(p.requires_grad for p in model.projector.parameters())


def test_detection_drift_is_zero_for_an_unchanged_model_and_positive_after_a_nudge():
    """The old guard compared detect_head's bytes, which cannot change -- it proved nothing.

    This measures detection BEHAVIOUR, which is what actually moves when the trunk trains.
    """
    base = T.UNetNodeTransformer(T.TemporalUNet3D(1, 4, [4, 8]), 4, 4 * T._POS_EMBED_DIM)
    model = E.EdgeTrainingModel(base, appearance_dim=0)
    imgs = torch.randn(1, 2, 8, 8, 8)
    with torch.no_grad():
        model.eval(); _, ref = model.base.encode(imgs)
    # encode returns det_logits as a LIST of per-frame tensors
    ref = torch.stack(ref, dim=1).detach().clone()

    assert E.detection_drift(model, ref, imgs)["max_abs"] == 0.0

    with torch.no_grad():           # perturb the TRUNK, leaving detect_head untouched
        for q in model.base.unet.parameters():
            q.add_(torch.randn_like(q) * 0.1)
        head_bytes = {k: v.clone() for k, v in model.base.detect_head.state_dict().items()}
    moved = E.detection_drift(model, ref, imgs)
    assert moved["max_abs"] > 0.0, "trunk moved but drift instrument reported nothing"
    # and the byte-level guard would have missed it entirely
    assert all(torch.equal(v, model.base.detect_head.state_dict()[k])
               for k, v in head_bytes.items())


def test_resume_discovery_looks_in_the_kaggle_input_mount(tmp_path, monkeypatch):
    """/kaggle/working starts empty on every batch run, so a local-only lookup is dead code."""
    out = tmp_path / "work"; out.mkdir()
    assert E._discover_resume(out) == out / "edge_resume.pth"     # first run: local path
    local = out / "edge_resume.pth"; local.write_bytes(b"x")
    assert E._discover_resume(out) == local                        # prefers local when present
