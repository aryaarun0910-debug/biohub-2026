"""Regression tests for the factory's machine-enforced experiment defect ledger."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "core"))
import kaggle_factory as KF  # noqa: E402


def notebook(source: str) -> dict:
    return {"cells": [{"cell_type": "code", "source": source.splitlines(keepends=True)}]}


def spec(name: str, *, datasets=(), env=None, source_path="scripts/kaggle_specs/unit.json") -> dict:
    edits = []
    if env:
        edits.append({"kind": "env", "vars": env})
    return {
        "name": name,
        "datasets": list(datasets),
        "edits": edits,
        "_spec_path": source_path,
    }


def defect_ids(given_spec: dict, source: str) -> set[str]:
    _, violations = KF.validate_defect_gate(given_spec, notebook(source))
    return {row.split(" [", 1)[0] for row in violations}


def test_ledger_has_exactly_the_eight_known_defect_classes():
    ledger = KF.load_defect_ledger()
    assert [d["id"] for d in ledger["defects"]] == [
        "DG-001-cuda-bce-under-autocast",
        "DG-002-shared-unet-detector-drift",
        "DG-003-division-supervision-deleted",
        "DG-004-any-threshold-checkpoint-selection",
        "DG-005-unreachable-resume",
        "DG-006-identity-sidecar-missing",
        "DG-007-produced-artifact-has-no-deployer",
        "DG-008-calibration-crosses-into-association",
    ]


def test_cuda_probability_bce_signature_is_blocked():
    ids = defect_ids(spec("h1r_edge_s5_unit"), "loss = F.binary_cross_entropy(prob, y)")
    assert "DG-001-cuda-bce-under-autocast" in ids


def test_shared_unet_without_freeze_teacher_or_joint_loss_is_blocked():
    ids = defect_ids(spec("h1r_edge_s5_unit"), "for p in self.base.detect_head.parameters():\n    p.requires_grad_(False)\n")
    assert "DG-002-shared-unet-detector-drift" in ids


def test_deleting_division_columns_is_blocked_even_if_named_in_prose():
    source = """\
def fake(continuation, division_cols):
    division_loss = 0.0
    continuation[:, torch.from_numpy(division_cols)] = 0.0
    keep = ~division_cols
"""
    ids = defect_ids(spec("h1r_edge_s5_unit"), source)
    assert "DG-003-division-supervision-deleted" in ids


def test_max_f1_checkpoint_selection_is_blocked():
    source = "selected_name, selected = best_operating_point(val)\nis_best = selected['f1'] > best\n"
    ids = defect_ids(spec("h1r_det_s1_unit"), source)
    assert "DG-004-any-threshold-checkpoint-selection" in ids


def test_deployed_selection_must_be_explicit_in_spec_and_code():
    source = """\
selected_name, selected = selection_at_threshold(
    val, selection_threshold
)
"""
    given = spec("h1r_det_s1_unit")
    assert "DG-004-any-threshold-checkpoint-selection" in defect_ids(given, source)
    given["edits"] = [{
        "kind": "env", "vars": {"H1R_SELECT_THRESHOLD": "deployed"},
    }]
    assert "DG-004-any-threshold-checkpoint-selection" not in defect_ids(given, source)


def test_resume_enabled_without_external_source_is_blocked():
    given = spec("h1r_det_s1", env={
        "H1R_RESUME": "1", "H1R_SELECT_THRESHOLD": "deployed",
    })
    ids = defect_ids(given, "selection_metric = 'deployed'\n")
    assert "DG-005-unreachable-resume" in ids
    given["resume_sources"] = [{"dataset": "owner/prior-run", "artifact": "detector_last.pth"}]
    assert "DG-005-unreachable-resume" not in defect_ids(given, "selection_metric = 'deployed'\n")


def test_edge_spec_without_identity_sidecar_is_blocked():
    ids = defect_ids(spec("h1r_edge_s5_unit"), "division_loss = joint_detection_loss = 0\n")
    assert "DG-006-identity-sidecar-missing" in ids
    given = spec("h1r_edge_s5_unit", datasets=["aryaarun07/biohub-zh001r-identity"])
    assert "DG-006-identity-sidecar-missing" not in defect_ids(
        given, "division_loss = joint_detection_loss = 0\n"
    )


def test_produced_artifact_requires_real_reciprocal_deployment_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(KF, "REPO", tmp_path)
    producer_rel = "scripts/kaggle_specs/train.json"
    consumer_rel = "scripts/kaggle_specs/deploy.json"
    given = spec("h1r_det_s1", source_path=str(tmp_path / producer_rel),
                 env={"H1R_SELECT_THRESHOLD": "deployed"})
    given["artifact_role"] = "training"
    source = "selection_metric = 'deployed'\ntorch.save(state, 'edge_predictor_best.pth')\n"
    assert "DG-007-produced-artifact-has-no-deployer" in defect_ids(given, source)

    given["deploy_consumers"] = [{"artifact": "edge_predictor_best.pth", "spec": consumer_rel}]
    consumer_path = tmp_path / consumer_rel
    consumer_path.parent.mkdir(parents=True)
    consumer_path.write_text(json.dumps({
        "artifact_role": "deployment",
        "consumes_artifacts": [{
            "producer_spec": producer_rel,
            "artifact": "edge_predictor_best.pth",
        }],
    }), encoding="utf-8")
    assert "DG-007-produced-artifact-has-no-deployer" not in defect_ids(given, source)


def test_diagnostic_artifact_requires_reason_and_cannot_name_deployer():
    source = "selection_metric = 'deployed'\ntorch.save(state, 'edge_predictor_best.pth')\n"
    given = spec("h1r_det_s1_smoke", env={"H1R_SELECT_THRESHOLD": "deployed"})
    given["artifact_role"] = "diagnostic"
    assert "DG-007-produced-artifact-has-no-deployer" in defect_ids(given, source)
    given["diagnostic_reason"] = "Smoke output validates mechanics and must never be promoted."
    assert "DG-007-produced-artifact-has-no-deployer" not in defect_ids(given, source)
    given["deploy_consumers"] = [{"artifact": "edge_predictor_best.pth", "spec": "x.json"}]
    assert "DG-007-produced-artifact-has-no-deployer" in defect_ids(given, source)


def test_calibration_artifact_requires_exact_mode_hash_and_reciprocity(tmp_path, monkeypatch):
    monkeypatch.setattr(KF, "REPO", tmp_path)
    producer_rel = "scripts/kaggle_specs/calibrate.json"
    consumer_rel = "scripts/kaggle_specs/deploy.json"
    source = "torch.save(state, 'edge_predictor_best.pth')\n"
    given = spec("h1r_det_s1_adabn_control", source_path=str(tmp_path / producer_rel),
                 env={"H1R_SELECT_THRESHOLD": "deployed"})
    given.update({
        "artifact_role": "calibration",
        "calibration_reason": "Only BatchNorm buffers may reach detector logits.",
        "deploy_consumers": [{
            "artifact": "edge_predictor_best.pth", "spec": consumer_rel,
            "mode": "detection_only_bn_buffers", "sha256": "ab" * 32,
        }],
    })
    consumer = tmp_path / consumer_rel
    consumer.parent.mkdir(parents=True)
    consumer.write_text(json.dumps({
        "artifact_role": "deployment",
        "consumes_artifacts": [{
            "producer_spec": producer_rel, "artifact": "edge_predictor_best.pth",
            "mode": "detection_only_bn_buffers", "sha256": "ab" * 32,
        }],
    }), encoding="utf-8")
    assert "DG-007-produced-artifact-has-no-deployer" not in defect_ids(given, source)

    given["deploy_consumers"][0]["mode"] = "whole_checkpoint"
    assert "DG-007-produced-artifact-has-no-deployer" in defect_ids(given, source)
    given["deploy_consumers"][0]["mode"] = "detection_only_bn_buffers"
    given["deploy_consumers"][0]["sha256"] = "unpinned"
    assert "DG-007-produced-artifact-has-no-deployer" in defect_ids(given, source)


def test_calibration_consumer_blocks_association_access():
    safe = """\
CALIBRATION_MODE = "detection_only_bn_buffers"
validate_adabn_state_contract(primary, calibrated)
unet_out, _primary_det_logits_unused = model.encode(imgs)
"""
    given = spec("deploy_h1r_adabn_detection_only")
    assert "DG-008-calibration-crosses-into-association" not in defect_ids(given, safe)
    unsafe = safe + "adabn_detection_model.predict_edges(a, b)\n"
    assert "DG-008-calibration-crosses-into-association" in defect_ids(given, unsafe)


@pytest.mark.parametrize("missing", [
    'CALIBRATION_MODE = "detection_only_bn_buffers"\n',
    "validate_adabn_state_contract(primary, calibrated)\n",
])
def test_calibration_consumer_requires_both_whitelist_and_mode(missing):
    safe = """\
CALIBRATION_MODE = "detection_only_bn_buffers"
validate_adabn_state_contract(primary, calibrated)
unet_out, _primary_det_logits_unused = model.encode(imgs)
"""
    given = spec("deploy_h1r_adabn_detection_only")
    assert "DG-008-calibration-crosses-into-association" in defect_ids(
        given, safe.replace(missing, "")
    )


def test_cmd_build_fails_before_writing_a_known_bad_notebook(tmp_path, monkeypatch):
    monkeypatch.setattr(KF, "REPO", tmp_path)
    base = tmp_path / "base.ipynb"
    base.write_text(json.dumps(notebook(
        "selected_name, selected = best_operating_point(val)\n"
    )), encoding="utf-8")
    given = {
        **spec("h1r_det_s1_unit", source_path=str(tmp_path / "spec.json"),
               env={"H1R_SELECT_THRESHOLD": "deployed"}),
        "slug": "unit", "title": "Unit", "code_file": "unit.ipynb",
        "out_dir": "out", "base_notebook": "base.ipynb", "edits": [],
    }
    with pytest.raises(SystemExit, match="DG-004-any-threshold-checkpoint-selection"):
        KF.cmd_build(given)
    assert not (tmp_path / "out").exists(), "a failed defect gate must not create build artifacts"


def test_cmd_verify_rechecks_the_built_notebook(tmp_path, monkeypatch):
    monkeypatch.setattr(KF, "REPO", tmp_path)
    bad = notebook("selected_name, selected = best_operating_point(val)\n")
    (tmp_path / "base.ipynb").write_text(json.dumps(bad), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    (out / "unit.ipynb").write_text(json.dumps(bad), encoding="utf-8")
    given = {
        **spec("h1r_det_s1_unit", source_path=str(tmp_path / "spec.json"),
               env={"H1R_SELECT_THRESHOLD": "deployed"}),
        "slug": "unit", "title": "Unit", "code_file": "unit.ipynb",
        "out_dir": "out", "base_notebook": "base.ipynb",
    }
    with pytest.raises(SystemExit, match="DG-004-any-threshold-checkpoint-selection"):
        KF.cmd_verify(given)


def test_non_h1r_specs_are_unaffected_by_scoped_rules():
    applicable, violations = KF.validate_defect_gate(
        spec("p3_deployed"), notebook("F.binary_cross_entropy(prob, y)\n")
    )
    assert applicable == []
    assert violations == []
