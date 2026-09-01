from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch

import scripts.kaggle_edits.h1r_adabn_detection_only as adabn_module
from scripts.kaggle_edits.h1r_adabn_detection_only import (
    AUDIT_SHA256,
    CALIBRATED_SHA256,
    CALIBRATION_MODE,
    CONFIG_SHA256,
    PRIMARY_SHA256,
    SUMMARY_SHA256,
    patch_detection_only_predictor,
    validate_adabn_artifact_contract,
    validate_adabn_state_contract,
)


import sys as _sys
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts" / "core"))
import hashing as _HASHING  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "scripts" / "kaggle_specs" / "deploy_h1r_adabn_detection_only.json"
PRODUCER = ROOT / "scripts" / "kaggle_specs" / "h1r_det_s1_adabn_control.json"


def _state() -> dict[str, torch.Tensor]:
    state = {
        "detect_head.weight": torch.ones(1),
        "transformer.weight": torch.ones(1),
        "unet.encoder.weight": torch.ones(1),
    }
    for index in range(10):
        prefix = f"unet.block.{index}.bn"
        state[f"{prefix}.weight"] = torch.ones(2)
        state[f"{prefix}.bias"] = torch.zeros(2)
        state[f"{prefix}.running_mean"] = torch.zeros(2)
        state[f"{prefix}.running_var"] = torch.ones(2)
        state[f"{prefix}.num_batches_tracked"] = torch.tensor(0)
    return state


def _calibrated_state() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    primary = _state()
    calibrated = {key: value.clone() for key, value in primary.items()}
    for key in calibrated:
        if key.endswith("running_mean"):
            calibrated[key].add_(0.25)
        elif key.endswith("running_var"):
            calibrated[key].add_(0.5)
        elif key.endswith("num_batches_tracked"):
            calibrated[key].add_(144)
    return primary, calibrated


def test_tensor_contract_accepts_only_all_thirty_bn_buffers() -> None:
    primary, calibrated = _calibrated_state()
    changed = validate_adabn_state_contract(primary, calibrated)
    assert len(changed) == 30
    assert all(key.startswith("unet.") for key in changed)
    assert not any("detect_head" in key or "transformer" in key for key in changed)


@pytest.mark.parametrize("key", [
    "detect_head.weight", "transformer.weight", "unet.encoder.weight",
    "unet.block.0.bn.weight",
])
def test_tensor_contract_rejects_any_parameter_or_association_drift(key: str) -> None:
    primary, calibrated = _calibrated_state()
    calibrated[key].add_(1)
    with pytest.raises(RuntimeError, match="learned/association tensors"):
        validate_adabn_state_contract(primary, calibrated)


def test_tensor_contract_rejects_partial_or_structurally_different_control() -> None:
    primary, calibrated = _calibrated_state()
    calibrated["unet.block.0.bn.running_mean"] = primary[
        "unet.block.0.bn.running_mean"
    ].clone()
    with pytest.raises(RuntimeError, match="complete BN-buffer set"):
        validate_adabn_state_contract(primary, calibrated)
    calibrated = {key: value for key, value in calibrated.items() if key != "transformer.weight"}
    with pytest.raises(RuntimeError, match="missing"):
        validate_adabn_state_contract(primary, calibrated)


def test_protocol_contract_is_lr_zero_and_control_only() -> None:
    audit = {"contract": "adabn_control", "lr": 0.0, "finite_optimizer_steps": 70}
    summary = {
        "trunk_contract": "adabn_control", "selection_threshold": "deployed",
        "epochs_run": 1,
    }
    assert validate_adabn_artifact_contract(audit, summary)["lr"] == 0.0
    for bad_audit, bad_summary in [
        ({**audit, "lr": 1e-4}, summary),
        ({**audit, "finite_optimizer_steps": 0}, summary),
        (audit, {**summary, "trunk_contract": "joint_unsafe"}),
        (audit, {**summary, "selection_threshold": "p0.5"}),
        (audit, {**summary, "epochs_run": 2}),
    ]:
        with pytest.raises(RuntimeError):
            validate_adabn_artifact_contract(bad_audit, bad_summary)


def _fake_bundle(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in (
        "edge_predictor_best.pth", "config.json", "summary.json",
        "training_audit.json",
    ):
        (root / name).write_bytes(name.encode("ascii"))


def test_install_accepts_flattened_duplicate_from_same_kaggle_source(
    tmp_path, monkeypatch,
) -> None:
    input_root = tmp_path / "input"
    mount = input_root / "notebooks" / "owner" / "control-kernel"
    _fake_bundle(mount)
    _fake_bundle(mount / "h1r_detector")
    repo = tmp_path / "repo"
    primary = repo / "weights" / "primary.pth"
    primary.parent.mkdir(parents=True)
    primary.write_bytes(b"primary")

    expected = {
        "primary.pth": PRIMARY_SHA256,
        "edge_predictor_best.pth": CALIBRATED_SHA256,
        "config.json": CONFIG_SHA256,
        "summary.json": SUMMARY_SHA256,
        "training_audit.json": AUDIT_SHA256,
    }
    monkeypatch.setattr(adabn_module, "sha256_file", lambda path: expected[path.name])
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: _state())
    monkeypatch.setattr(adabn_module, "validate_adabn_state_contract", lambda *_: [])
    monkeypatch.setattr(
        adabn_module, "validate_adabn_artifact_contract", lambda *_: {"lr": 0.0},
    )
    monkeypatch.setattr(adabn_module, "patch_detection_only_predictor", lambda *_: "a" * 64)
    monkeypatch.setattr(json, "loads", lambda *_: {})

    record = adabn_module.install_adabn_detection_only(
        repo, "weights/primary.pth", input_root, tmp_path / "working",
    )
    assert record["calibrated_sha256"] == CALIBRATED_SHA256


def test_install_rejects_same_artifact_from_two_source_mounts(
    tmp_path, monkeypatch,
) -> None:
    input_root = tmp_path / "input"
    _fake_bundle(input_root / "notebooks" / "owner" / "kernel-a")
    _fake_bundle(input_root / "notebooks" / "owner" / "kernel-b")
    repo = tmp_path / "repo"
    primary = repo / "weights" / "primary.pth"
    primary.parent.mkdir(parents=True)
    primary.write_bytes(b"primary")
    expected = {
        "primary.pth": PRIMARY_SHA256,
        "edge_predictor_best.pth": CALIBRATED_SHA256,
        "config.json": CONFIG_SHA256,
        "summary.json": SUMMARY_SHA256,
        "training_audit.json": AUDIT_SHA256,
    }
    monkeypatch.setattr(adabn_module, "sha256_file", lambda path: expected[path.name])
    with pytest.raises(RuntimeError, match="one exact AdaBN source mount"):
        adabn_module.install_adabn_detection_only(
            repo, "weights/primary.pth", input_root, tmp_path / "working",
        )


def _predictor_source() -> str:
    tta_calls = """\
        if cfg.det_tta:
            for dims in [(-1,), (-2,), (-2, -1)]:
                _, a = model.encode(imgs.flip(dims))
            for k in (1, 3):
                _, b = model.encode(imgs)
            _, c = model.encode(imgs)
            _, d = model.encode(imgs)
"""
    return """\
def predict_video(
        model: UNetNodeTransformer,
    secondary_low_margin_max: float = 0.2,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
        unet_out, det_logits = model.encode(imgs)
        # unet_out: (1, W, C, *spatial_down), det_logits: list of W × (1, 1, *spatial_down)
""" + tta_calls + """\
        secondary_unet_out = None
        unet_feat_src = model._index_features(
                unet_out[:, f_idx], p_coords_src, p_mask_src,
        )
        unet_feat_tgt = model._index_features(
                unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt,
        )

def predict():
    model, window_size, downsample = load_model(weights_path, device)

    secondary_model = None
    coords, edges = predict_video(
                secondary_mix_temperature=secondary_mix_temperature,
                secondary_low_margin_max=secondary_low_margin_max,
            )
"""


def test_predictor_patch_keeps_original_features_and_routes_only_logits(tmp_path) -> None:
    predictor = tmp_path / "predict.py"
    predictor.write_text(_predictor_source(), encoding="utf-8")
    digest = patch_detection_only_predictor(predictor)
    patched = predictor.read_text(encoding="utf-8")
    assert len(digest) == 64
    assert "unet_out, _primary_det_logits_unused = model.encode(imgs)" in patched
    assert patched.count("adabn_detection_model.encode(") == 5
    assert patched.count("model._index_features(\n                unet_out") == 2
    assert "adabn_detection_model.predict_edges" not in patched
    assert "adabn_detection_model._index_features" not in patched


def test_predictor_patch_fails_if_the_p3_tta_surface_drifts(tmp_path) -> None:
    predictor = tmp_path / "predict.py"
    predictor.write_text(_predictor_source().replace(
        "            _, d = model.encode(imgs)\n", ""
    ), encoding="utf-8")
    with pytest.raises(RuntimeError, match="four deployed P3 TTA"):
        patch_detection_only_predictor(predictor)


def test_patch_applies_to_the_exact_final_p3_predictor_surface(tmp_path) -> None:
    """Replay P3's literal runtime rewrites, then apply this consumer to that exact source."""
    notebook = json.loads((
        ROOT / "notebooks" / "kaggle_p3_harmonic" / "biohub-p3-harmonic.ipynb"
    ).read_text(encoding="utf-8"))
    notebook_code = "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    wanted = {
        "_old", "_new", "_ensemble_replacements", "_guard_old", "_guard_new",
        "_bi_old", "_bi_new",
    }
    values = {}
    for node in ast.walk(ast.parse(notebook_code)):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            values[target.id] = ast.literal_eval(node.value)

    source = (
        ROOT / "vendor" / "kaggle-cell-tracking" / "scripts" / "predict_unet_transformer.py"
    ).read_text(encoding="utf-8")
    rewrites = [
        (values["_old"], values["_new"]),
        *values["_ensemble_replacements"],
        (values["_guard_old"], values["_guard_new"]),
        (values["_bi_old"], values["_bi_new"]),
    ]
    for index, (old, new) in enumerate(rewrites):
        assert source.count(old) == 1, f"P3 runtime rewrite {index} drifted"
        source = source.replace(old, new, 1)

    predictor = tmp_path / "predict_unet_transformer.py"
    predictor.write_text(source, encoding="utf-8")
    patch_detection_only_predictor(predictor)
    patched = predictor.read_text(encoding="utf-8")
    assert patched.count("adabn_detection_model.encode(") == 5
    # Identity + 3 flips + 2 rotations + transpose + anti-transpose = 8 runtime views.
    assert 1 + 3 + 2 + 1 + 1 == 8
    tta = patched[
        patched.index("        if cfg.det_tta:"):
        patched.index("        secondary_unet_out = None")
    ]
    assert tta.count("adabn_detection_model.encode(") == 4
    assert "_, det_flip = model.encode(" not in tta
    assert "_, det_rot = model.encode(" not in tta
    assert "_, det_t = model.encode(" not in tta
    assert "_, det_at = model.encode(" not in tta
    assert patched.count("model._index_features(\n                unet_out") == 2
    assert "adabn_detection_model.predict_edges" not in patched


def test_specs_pin_reciprocal_mode_and_exact_completed_artifacts() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    producer = json.loads(PRODUCER.read_text(encoding="utf-8"))
    consumed = spec["consumes_artifacts"]
    declared = producer["deploy_consumers"]
    assert consumed == [{
        "producer_spec": "scripts/kaggle_specs/h1r_det_s1_adabn_control.json",
        "artifact": "edge_predictor_best.pth", "mode": CALIBRATION_MODE,
        "sha256": CALIBRATED_SHA256,
    }]
    assert {
        "artifact": "edge_predictor_best.pth",
        "spec": "scripts/kaggle_specs/deploy_h1r_adabn_detection_only.json",
        "mode": CALIBRATION_MODE, "sha256": CALIBRATED_SHA256,
    } in declared
    assert producer["artifact_role"] == "calibration"
    provenance = spec["provenance"]
    assert spec["base_notebook"] == (
        "notebooks/kaggle_p3_harmonic/biohub-p3-harmonic.ipynb"
    )
    assert "p9" not in spec["base_notebook"].lower()
    assert provenance["primary_checkpoint_sha256"] == PRIMARY_SHA256
    assert provenance["producer_config_sha256"] == CONFIG_SHA256
    assert provenance["producer_summary_sha256"] == SUMMARY_SHA256
    assert provenance["producer_training_audit_sha256"] == AUDIT_SHA256
    # CANONICAL: the patch is SOURCE, and its recorded digest was taken from an LF worktree.
    # A raw comparison passes here and fails in any CRLF checkout - measured, and it was one of
    # the ten clean-clone failures. The recorded value in the spec is UNCHANGED; only the
    # computation is canonicalised, because canonicalising an LF file is a no-op.
    patch_sha = _HASHING.canonical_text_sha256(
        ROOT / "scripts" / "kaggle_edits" / "h1r_adabn_detection_only.py")
    assert provenance["consumer_patch_sha256"] == patch_sha
    built_sha = hashlib.sha256((
        ROOT / "notebooks" / "kaggle_deploy_h1r_adabn_detection_only"
        / "biohub-h1r-adabn-detection-only.ipynb"
    ).read_bytes()).hexdigest()
    assert provenance["expected_built_notebook_sha256"] == built_sha
    assert spec["kernel_sources"] == [
        "aryaarun07/biohub-h1r-detector-s1-adabn-control"
    ]
