"""Fail-closed deployment bridge for the full S1 detector artifact.

Runs after the support pack is unpacked and before prediction starts. A checkpoint is accepted
only with colocated scientific and optimizer-step audits. The P3 detector threshold and
eight-view inference code remain unchanged.
"""
import hashlib as _h1r_hashlib
import json as _h1r_json
import shutil as _h1r_shutil
from pathlib import Path as _H1R_Path


def _h1r_sha256(path: _H1R_Path) -> str:
    digest = _h1r_hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_h1r_deploy_contract(audit: dict, summary: dict) -> dict:
    """Validate the scientific contract before any checkpoint bytes are copied."""
    contract = {
        "trunk_contract": summary.get("trunk_contract"),
        "selection_threshold": summary.get("selection_threshold"),
        "eval_tta": summary.get("eval_tta"),
        "finite_optimizer_steps": int(audit.get("finite_optimizer_steps", 0)),
    }
    if contract["trunk_contract"] != "freeze":
        raise RuntimeError(f"S1 deploy rejects shared-trunk drift: {contract}")
    if contract["selection_threshold"] != "deployed" or not contract["eval_tta"]:
        raise RuntimeError(f"S1 deploy rejects mismatched selection/evaluation: {contract}")
    if contract["finite_optimizer_steps"] <= 0:
        raise RuntimeError(f"S1 deploy rejects a checkpoint without finite AdamW steps: {contract}")
    return contract


def deploy_h1r_s1(repo_dir: _H1R_Path, weights_relative: str,
                   input_root: _H1R_Path = _H1R_Path("/kaggle/input")) -> dict:
    audits = sorted(input_root.rglob("training_audit.json"))
    if len(audits) != 1:
        raise RuntimeError(f"S1 deploy expected exactly one training_audit.json, got {audits}")
    source = audits[0].parent
    weight = source / "edge_predictor_best.pth"
    summary_path = source / "summary.json"
    for required in (weight, summary_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    audit = _h1r_json.loads(audits[0].read_text())
    summary = _h1r_json.loads(summary_path.read_text())
    contract = validate_h1r_deploy_contract(audit, summary)
    destination = repo_dir / weights_relative
    if not destination.is_file():
        raise FileNotFoundError(destination)
    old_sha = _h1r_sha256(destination)
    new_sha = _h1r_sha256(weight)
    if new_sha == old_sha:
        raise RuntimeError("S1 deploy checkpoint is byte-identical to the original primary weight")
    _h1r_shutil.copy2(weight, destination)
    if _h1r_sha256(destination) != new_sha:
        raise RuntimeError("S1 deploy checkpoint copy failed byte verification")
    result = {**contract, "old_sha256": old_sha, "new_sha256": new_sha}
    print("H1R_S1_DEPLOY", _h1r_json.dumps(result, sort_keys=True))
    return result


# Importing this file for unit tests stays inert; the deployment notebook defines both names.
if "REPO_DIR" in globals() and "WEIGHTS_RELATIVE" in globals():
    deploy_h1r_s1(REPO_DIR, WEIGHTS_RELATIVE)
