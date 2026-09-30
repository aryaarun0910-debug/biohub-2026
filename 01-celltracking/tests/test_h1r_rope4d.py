"""Software contracts for the 4-D RoPE option of the S5 association trainer.

``H1R_EDGE_POS_ENCODING=sinusoidal|rope4d`` (scripts/kaggle_edits/h1r_rope4d.py). These tests pin
(a) that sinusoidal is the untouched vendored model bit for bit, (b) that with projected q/k inputs
held fixed the rotary attention factor is invariant to a global translation/time shift, (c) that
they DO change under a relative displacement, (d) that the option round-trips env -> loader ->
config.json -> inference install -> specs, and (e) that the frozen trunk / zero-drift contract is
untouched by the rotation. They say nothing about whether rope4d helps - that is the job of gate 2.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "kaggle_edits"))
sys.path.insert(0, str(ROOT / "scripts" / "core"))
sys.path.insert(0, str(ROOT / "vendor" / "kaggle-cell-tracking" / "scripts"))

# See the note in tests/test_h1r_edge_train.py: this module imports from the PINNED EXTERNAL
# CHECKOUT `vendor/kaggle-cell-tracking`, absent from a fresh clone by design. Unguarded it broke
# COLLECTION and interrupted the entire suite. The skip reason names the repair, not the symptom.
import bootstrap_vendor as BV  # noqa: E402

if not BV.is_ok("vendor/kaggle-cell-tracking"):
    pytest.skip(BV.diagnostic("vendor/kaggle-cell-tracking"), allow_module_level=True)

import h1r_edge_data as D  # noqa: E402
import h1r_edge_train as E  # noqa: E402
import h1r_rope4d as R  # noqa: E402
import h1r_rope4d_inference_patch as P  # noqa: E402
import train_unet_transformer as T  # noqa: E402

SPECS = ROOT / "scripts" / "kaggle_specs"
RUNNER = ROOT / "scripts" / "kaggle_edits" / "h1r_edge_kernel_run.py"
PREDICT = ROOT / "vendor" / "kaggle-cell-tracking" / "scripts" / "predict_unet_transformer.py"
VOXEL = torch.tensor(R.VOXEL_UM)
# The deployed transformer geometry (hidden 128, 4 heads -> head_dim 32, 4 blocks; the vendored
# defaults load_public_full_model rebuilds) on a tiny UNet. Dropout is inert in eval mode.
PUBLIC_TRANSFORMER = dict(hidden_dim=128, n_heads=4, n_blocks=4, dropout=0.3)


def _public_like(tmp_path, **kw):
    cfg = {"unet_out_channels": 4, "unet_layers": [4, 8], "downsample": [1, 4, 4], "window_size": 2}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    torch.manual_seed(0)
    base = T.UNetNodeTransformer(T.TemporalUNet3D(1, 4, [4, 8]), 4, 4 * T._POS_EMBED_DIM, **kw)
    weights = tmp_path / "edge_predictor_best.pth"
    torch.save(base.state_dict(), weights)
    return base, weights


def _edge_inputs(seed=0, b=2, n0=7, n1=9, c=4):
    """Pre-indexed UNet features, FULL-RES voxel coords, fixed sinusoidal pos feats, padded masks."""
    g = torch.Generator().manual_seed(seed)
    f0 = torch.randn(b, n0, c, generator=g)
    f1 = torch.randn(b, n1, c, generator=g)
    extent = torch.tensor([30.0, 800.0, 800.0])
    c0 = torch.rand(b, n0, 3, generator=g) * extent
    c1 = torch.rand(b, n1, 3, generator=g) * extent
    m0 = torch.ones(b, n0, dtype=torch.bool); m0[1, -1] = False
    m1 = torch.ones(b, n1, dtype=torch.bool); m1[1, -2:] = False
    shape = (2, 32, 3200, 3200)
    p0 = T._pos_embed_torch(torch.cat([torch.zeros(b, n0, 1), c0], -1), shape)
    p1 = T._pos_embed_torch(torch.cat([torch.ones(b, n1, 1), c1], -1), shape)
    return f0, f1, c0, c1, p0, p1, m0, m1


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


# ----------------------------------------------------------------------------- (a) sinusoidal
def test_sinusoidal_is_the_untouched_vendored_model_bit_for_bit(tmp_path):
    base, weights = _public_like(tmp_path, **PUBLIC_TRANSFORMER)
    loaded = E.load_public_full_model(T, weights, torch.device("cpu"), pos_encoding="sinusoidal")
    assert type(loaded.base.transformer) is T.SimpleNodeTransformer
    assert not isinstance(loaded.base.transformer, R.Rope4DNodeTransformer)
    base.eval(); loaded.eval()
    inputs = _edge_inputs()
    with torch.no_grad():
        assert torch.equal(base.predict_edges(*inputs), loaded.base.predict_edges(*inputs))
    # the default knob IS sinusoidal, so an unset env gives the same untouched model
    assert R.resolve_pos_encoding() == "sinusoidal" or os.environ.get("H1R_EDGE_POS_ENCODING")


def test_rope_attention_without_rotation_matches_nn_multiheadattention():
    """Parity of the functional re-implementation with the vendored nn.MultiheadAttention."""
    torch.manual_seed(1)
    mha = torch.nn.MultiheadAttention(32, 4, batch_first=True, dropout=0.0).eval()
    xq, xkv = torch.randn(2, 5, 32), torch.randn(2, 8, 32)
    kpm = torch.zeros(2, 8, dtype=torch.bool); kpm[0, -3:] = True
    with torch.no_grad():
        want, _ = mha(xq, xkv, xkv, key_padding_mask=kpm)
        got = R.rope_cross_attention(mha, xq, xkv, kpm, None, None, None, None, training=False)
    assert torch.allclose(want, got, atol=1e-5)


# ------------------------------------------------------------------ (b)/(c) attention logits
def _logits(rope, q, k, c4_q, c4_k):
    cos_q, sin_q = rope.cos_sin(c4_q)
    cos_k, sin_k = rope.cos_sin(c4_k)
    return R.Rope4D.rotate(q, cos_q, sin_q) @ R.Rope4D.rotate(k, cos_k, sin_k).transpose(-1, -2)


def _coords4(g, b, n, frame):
    vox = torch.rand(b, n, 3, generator=g) * torch.tensor([30.0, 800.0, 800.0])
    return torch.cat([torch.full((b, n, 1), float(frame)), vox], -1)


def test_rope4d_attention_logits_are_invariant_to_global_translation_and_time_shift():
    g = torch.Generator().manual_seed(2)
    rope = R.Rope4D(head_dim=32, bands=4)
    q, k = torch.randn(2, 4, 6, 32, generator=g), torch.randn(2, 4, 9, 32, generator=g)
    c4_q, c4_k = _coords4(g, 2, 6, 0), _coords4(g, 2, 9, 1)
    base = _logits(rope, q, k, c4_q, c4_k)
    # a global translation of EVERY node by (12.5, -40.0, 33.0) um and 5 frames
    shift_um = torch.tensor([12.5, -40.0, 33.0])
    shift = torch.cat([torch.tensor([5.0]), shift_um / VOXEL])
    moved = _logits(rope, q, k, c4_q + shift, c4_k + shift)
    assert torch.allclose(base, moved, atol=1e-4), float((base - moved).abs().max())
    assert base.abs().max() > 1e-2  # not a degenerate zero test


def test_rope4d_attention_logits_change_under_a_relative_displacement():
    g = torch.Generator().manual_seed(3)
    rope = R.Rope4D(head_dim=32, bands=4)
    q, k = torch.randn(2, 4, 6, 32, generator=g), torch.randn(2, 4, 9, 32, generator=g)
    c4_q, c4_k = _coords4(g, 2, 6, 0), _coords4(g, 2, 9, 1)
    base = _logits(rope, q, k, c4_q, c4_k)
    # move ONLY the keys by 3 um in x (a typical inter-frame displacement, FACT-0040 scale)
    only_keys = c4_k + torch.tensor([0.0, 0.0, 0.0, 3.0 / R.VOXEL_UM[2]])
    assert float((base - _logits(rope, q, k, c4_q, only_keys)).abs().max()) > 1e-2
    # and the rotation is applied at all: rotated logits differ from the raw dot product
    assert float((base - q @ k.transpose(-1, -2)).abs().max()) > 1e-2


def test_rope4d_forward_with_fixed_absolute_inputs_is_translation_invariant_and_differs_from_sinusoidal(tmp_path):
    """Through the real predict_edges path: RoPE is in the forward, and the whole forward stays
    invariant to a global translation / time offset while the absolute inputs are held fixed."""
    base, weights = _public_like(tmp_path, **PUBLIC_TRANSFORMER)
    plain = E.load_public_full_model(T, weights, torch.device("cpu"), pos_encoding="sinusoidal").eval()
    rope_model = E.load_public_full_model(T, weights, torch.device("cpu"), pos_encoding="rope4d").eval()
    assert isinstance(rope_model.base.transformer, R.Rope4DNodeTransformer)
    f0, f1, c0, c1, p0, p1, m0, m1 = _edge_inputs()
    with torch.no_grad():
        ref = rope_model.base.predict_edges(f0, f1, c0, c1, p0, p1, m0, m1)
        assert not torch.allclose(ref, plain.base.predict_edges(f0, f1, c0, c1, p0, p1, m0, m1), atol=1e-3)
        shift = torch.tensor([12.5, -40.0, 33.0]) / VOXEL
        moved = rope_model.base.predict_edges(f0, f1, c0 + shift, c1 + shift, p0, p1, m0, m1)
        assert torch.allclose(ref, moved, atol=1e-4)
        # a global shift in t: the same rotation with frame offsets (9, 10) instead of (0, 1)
        cfg = {**R.rope4d_config_from_env({}), "frame_offsets": [9.0, 10.0]}
        shifted = E.load_public_full_model(T, weights, torch.device("cpu"), pos_encoding="sinusoidal").eval()
        R.install_rope4d(shifted.base, cfg)
        assert torch.allclose(ref, shifted.base.predict_edges(f0, f1, c0, c1, p0, p1, m0, m1), atol=1e-4)
        # padded key slots must not leak into real rows
        f1_leak = f1.clone(); f1_leak[1, -1] += 100.0
        leak = rope_model.base.predict_edges(f0, f1_leak, c0, c1, p0, p1, m0, m1)
        assert torch.allclose(ref[:, :, :-2], leak[:, :, :-2], atol=1e-5)


# ------------------------------------------------------------------------ (d) round trips
def test_env_knob_round_trips_through_loader_config_and_inference_install(tmp_path, monkeypatch):
    base, weights = _public_like(tmp_path, **PUBLIC_TRANSFORMER)
    monkeypatch.setenv("H1R_EDGE_POS_ENCODING", "rope4d")
    monkeypatch.setenv("H1R_ROPE_BANDS", "2")
    monkeypatch.setenv("H1R_ROPE_SPACE_WAVELENGTHS_UM", "8,160")
    monkeypatch.setenv("H1R_ROPE_TIME_WAVELENGTHS", "2,16")
    try:
        importlib.reload(E)
        assert E.POS_ENCODING == "rope4d"
        assert E.ROPE4D_CONFIG == {"bands": 2, "space_wavelengths_um": [8.0, 160.0],
                                   "time_wavelengths_frames": [2.0, 16.0],
                                   "voxel_um": list(R.VOXEL_UM), "frame_offsets": [0.0, 1.0]}
        trained = E.load_public_full_model(T, weights, torch.device("cpu"))
        rope = trained.base.transformer.rope
        assert isinstance(trained.base.transformer, R.Rope4DNodeTransformer)
        assert rope.bands == 2 and 2 * rope.pairs == 16  # partial RoPE: 16 of 32 dims rotated
        assert rope.theta.shape == (4, 2)
        assert torch.allclose(rope.theta[1:], torch.tensor([[2 * np.pi / 160.0, 2 * np.pi / 8.0]]).expand(3, 2))
        assert torch.allclose(rope.theta[0], torch.tensor([2 * np.pi / 16.0, 2 * np.pi / 2.0]))
        # every pretrained parameter survives, and the export is key-identical to the public model
        assert list(trained.base.state_dict().keys()) == list(base.state_dict().keys())
        assert all(torch.equal(v, trained.base.state_dict()[k]) for k, v in base.state_dict().items())
        # config.json round trip: the inference side rebuilds the public class, loads the export
        # strictly, and re-installs the rotation from the config -> identical logits
        exported = {**json.loads((tmp_path / "config.json").read_text()), **rope.export_config()}
        assert exported["h1r_pos_encoding"] == "rope4d"
        fresh = T.UNetNodeTransformer(T.TemporalUNet3D(1, 4, [4, 8]), 4, 4 * T._POS_EMBED_DIM,
                                      **PUBLIC_TRANSFORMER)
        fresh.load_state_dict(trained.base.state_dict(), strict=True)
        R.install_rope4d_from_config(fresh, exported)
        trained.eval(); fresh.eval()
        inputs = _edge_inputs(seed=4)
        with torch.no_grad():
            assert torch.allclose(trained.base.predict_edges(*inputs), fresh.predict_edges(*inputs), atol=1e-6)
        with pytest.raises(RuntimeError):
            R.install_rope4d_from_config(fresh, exported)  # never installed twice
    finally:
        for key in ("H1R_EDGE_POS_ENCODING", "H1R_ROPE_BANDS", "H1R_ROPE_SPACE_WAVELENGTHS_UM",
                    "H1R_ROPE_TIME_WAVELENGTHS"):
            monkeypatch.delenv(key, raising=False)
        importlib.reload(E)
    assert E.POS_ENCODING == "sinusoidal" and E.ROPE4D_CONFIG is None
    with pytest.raises(ValueError):
        R.resolve_pos_encoding("banana")
    with pytest.raises(ValueError):
        R.Rope4D(head_dim=32, bands=5)  # 8 * bands must fit the head
    assert R.geometric_wavelengths(10.0, 200.0, 4)[0] == 200.0
    assert abs(R.geometric_wavelengths(10.0, 200.0, 4)[-1] - 10.0) < 1e-9


def _env(spec: dict) -> dict:
    out: dict = {}
    for e in spec.get("edits", []):
        if e.get("kind") == "env":
            out.update(e.get("vars", {}))
    return out


def test_specs_carry_the_knob_and_the_consumer_globs_the_rope_basenames():
    full = json.loads((SPECS / "h1r_edge_s5_rope4d.json").read_text(encoding="utf-8"))
    smoke = json.loads((SPECS / "h1r_edge_s5_rope4d_smoke.json").read_text(encoding="utf-8"))
    control = json.loads((SPECS / "h1r_edge_s5.json").read_text(encoding="utf-8"))
    cons = json.loads((SPECS / "deploy_h1r_edge_s5_rope4d_loeo_f0.json").read_text(encoding="utf-8"))
    sib = json.loads((SPECS / "deploy_h1r_edge_s5_loeo_f0.json").read_text(encoding="utf-8"))
    for spec in (full, smoke):
        env = _env(spec)
        assert env["H1R_EDGE_POS_ENCODING"] == "rope4d"
        assert env["H1R_ROPE_BANDS"] == "4"
        assert env["H1R_ROPE_SPACE_WAVELENGTHS_UM"] == "10,200" and env["H1R_ROPE_TIME_WAVELENGTHS"] == "2,20"
        assert "H1R_KERNEL" in env  # the spliced module registers itself only under the kernel flag
        files = [e.get("code_file") for e in spec["edits"]]
        assert files.index("scripts/kaggle_edits/h1r_edge_train.py") + 1 == files.index("scripts/kaggle_edits/h1r_rope4d.py")
        assert files.index("scripts/kaggle_edits/h1r_rope4d.py") < files.index("scripts/kaggle_edits/h1r_edge_kernel_run.py")
    # the control stays sinusoidal, explicitly, and splices the SAME (inert) module cell so the
    # two kernels run identical code and differ only in the env knob
    assert _env(control)["H1R_EDGE_POS_ENCODING"] == "sinusoidal"
    assert "scripts/kaggle_edits/h1r_rope4d.py" in [e.get("code_file") for e in control["edits"]]
    assert full["slug"] == "biohub-h1r-edge-s5-rope4d" and full["out_dir"] == "notebooks/kaggle_h1r_edge_s5_rope4d"
    assert full["artifact_role"] == "training"
    assert _env(full)["H1R_EDGE_INIT_WEIGHTS_GLOB"] == _env(control)["H1R_EDGE_INIT_WEIGHTS_GLOB"]
    consumers = {row["artifact"]: row for row in full["deploy_consumers"]}
    assert consumers["edge_predictor_best.pth"]["spec"] == "scripts/kaggle_specs/deploy_h1r_edge_s5_rope4d_loeo_f0.json"
    assert smoke["artifact_role"] == "diagnostic" and _env(smoke)["H1R_EDGE_MAX_STEPS"] == "2"
    assert "H1R_EDGE_INIT_WEIGHTS_GLOB" not in _env(smoke)
    assert cons["kernel_sources"] == ["aryaarun07/biohub-h1r-edge-s5-rope4d"]
    assert cons["out_dir"] == "notebooks/kaggle_deploy_h1r_edge_s5_rope4d_loeo_f0" and cons["out_dir"] != sib["out_dir"]
    assert {"producer_spec": "scripts/kaggle_specs/h1r_edge_s5_rope4d.json", "artifact": "edge_predictor_best.pth"} in cons["consumes_artifacts"]
    ce = _env(cons)
    assert ce["BIOHUB_LOEO_WEIGHTS_GLOB"] == "/kaggle/input/*/edge_predictor_best_h1r_s5_rope4d.pth"
    assert ce["BIOHUB_LOEO_CONFIG_GLOB"] == "/kaggle/input/*/config_h1r_s5_rope4d.json"
    for key in ("BIOHUB_SAFE_DIV_MAX_UM", "BIOHUB_LOEO_STEMS", "BIOHUB_LOEO_ARM", "BIOHUB_LOEO_FOLD"):
        assert ce[key] == _env(sib)[key], key
    files = [e.get("code_file") for e in cons["edits"]]
    # the inference patch runs AFTER loeo_retarget (which resolves WEIGHTS_RELATIVE) and before predict
    assert files.index("scripts/kaggle_edits/loeo_retarget.py") + 1 == files.index("scripts/kaggle_edits/h1r_rope4d_inference_patch.py")


def test_runner_exports_rope_basenames_only_under_rope4d():
    src = RUNNER.read_text(encoding="utf-8")
    assert '"rope4d": ("edge_predictor_best_h1r_s5_rope4d.pth", "config_h1r_s5_rope4d.json")' in src
    assert '"sinusoidal": ("edge_predictor_best_h1r_s5.pth", "config_h1r_s5.json")' in src
    assert "}[POS_ENCODING]" in src and '"pos_encoding": POS_ENCODING' in src


def test_inference_patch_is_a_verbatim_module_copy_and_applies_exactly_once(tmp_path):
    module = (ROOT / "scripts" / "kaggle_edits" / "h1r_rope4d.py").read_text(encoding="utf-8")
    assert P.ROPE4D_MODULE_SOURCE == module, (
        "h1r_rope4d_inference_patch.py embeds a stale copy of h1r_rope4d.py; run "
        "python scripts/kaggle_edits/h1r_rope4d_inference_patch.py --sync")
    assert not re.search(r"^from __future__ import", module, re.M)  # spliced mid-cell by the factory
    predictor = tmp_path / PREDICT.name
    original = PREDICT.read_text(encoding="utf-8")
    predictor.write_text(original, encoding="utf-8")
    cfg = tmp_path / "config.json"
    # sinusoidal checkpoint: byte-for-byte untouched, no module written
    cfg.write_text(json.dumps({"unet_out_channels": 32}))
    report = P.apply_rope4d_inference_patch(predictor, cfg, tmp_path / "h1r_rope4d.py")
    assert report["patched"] is False and predictor.read_text(encoding="utf-8") == original
    assert not (tmp_path / "h1r_rope4d.py").exists()
    # rope4d checkpoint: module written, load_model patched once, still compiles
    cfg.write_text(json.dumps({"unet_out_channels": 32, **R.Rope4D(32).export_config()}))
    report = P.apply_rope4d_inference_patch(predictor, cfg, tmp_path / "h1r_rope4d.py")
    patched = predictor.read_text(encoding="utf-8")
    assert report["patched"] is True and (tmp_path / "h1r_rope4d.py").read_text(encoding="utf-8") == module
    assert patched.count("install_rope4d_from_config(model, config)") == 1
    assert patched.index("install_rope4d_from_config") < patched.index("    model.to(device)\n    model.eval()")
    compile(patched, str(predictor), "exec")
    with pytest.raises(RuntimeError):
        P.apply_rope4d_inference_patch(predictor, cfg, tmp_path / "h1r_rope4d.py")  # anchor consumed


# ------------------------------------------------------------- (e) frozen trunk / drift
def test_frozen_trunk_and_zero_detection_drift_hold_under_rope4d(tmp_path):
    torch.manual_seed(0)
    ds = _fixture(tmp_path); batch = E.collate_edge_windows([ds[0]])
    base, weights = _public_like(tmp_path, hidden_dim=32, n_heads=4, n_blocks=1, dropout=0.0)
    unet_before = base.unet
    cfg = {**R.rope4d_config_from_env({}), "bands": 1}  # head_dim 8 -> one band per axis
    R.install_rope4d(base, cfg)
    assert base.unet is unet_before and base.detect_head is not None
    model = E.EdgeTrainingModel(base, appearance_dim=8)
    imgs = batch["imgs"]
    with torch.no_grad():
        model.eval(); _, ref = model.base.encode(imgs); model.train()
    reference = torch.stack(ref, dim=1).detach().clone()
    loss, _ = E.forward_batch(T, model, batch, torch.device("cpu"), triplet_weight=1.0)
    loss.backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.base.unet.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.base.transformer.parameters())
    assert all(p.grad is None for p in model.base.detect_head.parameters())
    params = [p for p in model.parameters() if p.requires_grad]
    torch.optim.AdamW(params, lr=1e-2).step()
    assert E.detection_drift(model, reference, imgs)["max_abs"] == 0.0
    # the rotary module adds NO state: resume/export dictionaries keep the public key set
    assert list(model.base.state_dict().keys()) == list(base.state_dict().keys())
    assert not any(k.startswith("transformer.rope") for k in model.state_dict())
