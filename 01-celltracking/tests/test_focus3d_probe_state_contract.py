"""Mutation tests for FOCUS-3D checkpoint/runtime state comparison."""

from __future__ import annotations

from pathlib import Path

import torch

from scripts.win_bet.focus3d_probe import (
    _attest_training_only_criterion_buffer,
    _state_compatibility,
    _write_payload,
)


def test_exact_state_is_clean():
    state = {"backbone.weight": torch.zeros(2, 3), "head.bias": torch.ones(4)}
    report = _state_compatibility(state, dict(state))
    assert report["missing"] == []
    assert report["unexpected"] == []
    assert report["shape_mismatch"] == []
    assert report["dtype_mismatch"] == []


def test_training_only_buffer_is_visible_not_silently_discarded():
    model = {"backbone.weight": torch.zeros(2, 3)}
    checkpoint = {**model, "criterion.empty_weight": torch.ones(2)}
    report = _state_compatibility(model, checkpoint)
    assert report["unexpected"] == ["criterion.empty_weight"]


def test_shape_dtype_missing_and_arbitrary_extra_are_independently_detected():
    model = {
        "same": torch.zeros(2),
        "wrong_shape": torch.zeros(2, 3),
        "wrong_dtype": torch.zeros(2, dtype=torch.float32),
        "missing": torch.zeros(1),
    }
    checkpoint = {
        "same": torch.zeros(2),
        "wrong_shape": torch.zeros(3, 2),
        "wrong_dtype": torch.zeros(2, dtype=torch.float16),
        "arbitrary": torch.zeros(1),
    }
    report = _state_compatibility(model, checkpoint)
    assert report["missing"] == ["missing"]
    assert report["unexpected"] == ["arbitrary"]
    assert report["shape_mismatch"] == ["wrong_shape"]
    assert report["dtype_mismatch"] == ["wrong_dtype"]


def _publisher_fixture(tmp_path: Path) -> tuple[Path, Path]:
    runtime = tmp_path / "runtime"
    criterion = runtime / "pkg" / "criterion_win.py"
    model = runtime / "pkg" / "maskformer_model_win.py"
    criterion.parent.mkdir(parents=True)
    criterion.write_text(
        "empty_weight = torch.ones(self.num_classes + 1)\n"
        "empty_weight[-1] = self.eos_coef\n"
        "self.register_buffer('empty_weight', empty_weight)\n",
        encoding="utf-8",
    )
    model.write_text(
        "if build_criterion:\n    criterion = build()\nelse:\n    criterion = None\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "MODEL:\n  SEM_SEG_HEAD:\n    NUM_CLASSES: 1\n"
        "  MASK_FORMER:\n    NO_OBJECT_WEIGHT: 0.1\n",
        encoding="utf-8",
    )
    return runtime, config


def test_training_only_buffer_is_source_config_and_value_bound(tmp_path):
    runtime, config = _publisher_fixture(tmp_path)
    report = _attest_training_only_criterion_buffer(
        runtime, config, {"criterion.empty_weight": torch.tensor([1.0, 0.1])})
    assert report["source_registers_buffer"] is True
    assert report["inference_explicitly_omits_criterion"] is True
    assert report["value_matches_config"] is True


def test_training_only_attestation_refuses_value_or_source_drift(tmp_path):
    runtime, config = _publisher_fixture(tmp_path)
    with torch.no_grad():
        bad = {"criterion.empty_weight": torch.tensor([1.0, 0.2])}
    try:
        _attest_training_only_criterion_buffer(runtime, config, bad)
    except RuntimeError as exc:
        assert "does not equal" in str(exc)
    else:
        raise AssertionError("wrong training-only buffer value was accepted")

    (runtime / "pkg" / "criterion_win.py").write_text("pass\n", encoding="utf-8")
    try:
        _attest_training_only_criterion_buffer(
            runtime, config, {"criterion.empty_weight": torch.tensor([1.0, 0.1])})
    except RuntimeError as exc:
        assert "no longer registers" in str(exc)
    else:
        raise AssertionError("publisher source drift was accepted")


def test_evidence_payload_is_atomic_and_carries_heartbeat(tmp_path):
    output = tmp_path / "nested" / "receipt.json"
    _write_payload(str(output), {"checkpoint_runtime_contract_clean": True})
    text = output.read_text(encoding="utf-8")
    assert '"heartbeat": "FOCUS3D_PROBE_COMPLETE"' in text
    assert not output.with_name(output.name + ".tmp").exists()
