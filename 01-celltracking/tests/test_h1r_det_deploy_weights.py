from __future__ import annotations

import json

import pytest

from scripts.kaggle_edits.h1r_det_deploy_weights import (
    deploy_h1r_s1,
    validate_h1r_deploy_contract,
)


GOOD_AUDIT = {"finite_optimizer_steps": 3}
GOOD_SUMMARY = {
    "trunk_contract": "freeze",
    "selection_threshold": "deployed",
    "eval_tta": True,
}


@pytest.mark.parametrize(
    ("audit", "summary", "message"),
    [
        ({"finite_optimizer_steps": 0}, GOOD_SUMMARY, "finite AdamW"),
        (GOOD_AUDIT, {**GOOD_SUMMARY, "trunk_contract": "joint_unsafe"}, "trunk drift"),
        (GOOD_AUDIT, {**GOOD_SUMMARY, "selection_threshold": "p0.5"}, "mismatched"),
        (GOOD_AUDIT, {**GOOD_SUMMARY, "eval_tta": False}, "mismatched"),
    ],
)
def test_deployment_contract_fails_closed(audit, summary, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        validate_h1r_deploy_contract(audit, summary)


def test_deployment_bridge_copies_only_audited_distinct_weight(tmp_path) -> None:
    repo = tmp_path / "repo"
    destination = repo / "weights" / "edge_predictor_best.pth"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old")
    source = tmp_path / "input" / "producer"
    source.mkdir(parents=True)
    (source / "edge_predictor_best.pth").write_bytes(b"new-valid-weight")
    (source / "training_audit.json").write_text(json.dumps(GOOD_AUDIT))
    (source / "summary.json").write_text(json.dumps(GOOD_SUMMARY))

    result = deploy_h1r_s1(
        repo, "weights/edge_predictor_best.pth", input_root=tmp_path / "input"
    )
    assert destination.read_bytes() == b"new-valid-weight"
    assert result["finite_optimizer_steps"] == 3
    assert result["old_sha256"] != result["new_sha256"]
