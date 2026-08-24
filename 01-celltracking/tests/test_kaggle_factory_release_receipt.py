"""Release receipts make factory submit commands byte- and version-specific."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "core"))
import kaggle_factory as KF  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def submission_frame() -> pd.DataFrame:
    datasets = [
        "44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1",
    ]
    rows = []
    for dataset in datasets:
        rows.extend([
            {
                "dataset": dataset, "row_type": "node", "node_id": 0,
                "t": 0, "z": 1, "y": 2, "x": 3, "source_id": -1,
                "target_id": -1,
            },
            {
                "dataset": dataset, "row_type": "node", "node_id": 1,
                "t": 1, "z": 2, "y": 3, "x": 4, "source_id": -1,
                "target_id": -1,
            },
            {
                "dataset": dataset, "row_type": "edge", "node_id": -1,
                "t": -1, "z": -1, "y": -1, "x": -1, "source_id": 0,
                "target_id": 1,
            },
        ])
    frame = pd.DataFrame(rows)
    frame.insert(0, "id", range(len(frame)))
    return frame[[
        "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
        "source_id", "target_id",
    ]]


@pytest.fixture
def release_repo(tmp_path: Path, monkeypatch) -> dict:
    auditor_bytes = KF.audit_script_path().read_bytes()
    monkeypatch.setattr(KF, "REPO", tmp_path)
    monkeypatch.setattr(
        KF, "DEFECT_LEDGER", tmp_path / "scripts" / "core" / "experiment_defects.json",
    )

    auditor = KF.audit_script_path()
    auditor.parent.mkdir(parents=True)
    auditor.write_bytes(auditor_bytes)
    KF.DEFECT_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    KF.DEFECT_LEDGER.write_text(
        json.dumps({"schema_version": 1, "defects": []}) + "\n", encoding="utf-8",
    )

    spec_path = tmp_path / "scripts" / "kaggle_specs" / "candidate.json"
    spec_path.parent.mkdir(parents=True)
    raw_spec = {
        "name": "candidate", "slug": "candidate-kernel", "title": "Candidate",
        "code_file": "candidate.ipynb", "out_dir": "notebooks/candidate",
        "base_notebook": "base.ipynb", "expects_submission": True,
    }
    spec_path.write_text(json.dumps(raw_spec), encoding="utf-8")
    spec = KF.load_spec(spec_path)

    out = KF.out_dir(spec)
    out.mkdir(parents=True)
    notebook = KF.built_nb(spec)
    notebook.write_text('{"cells": []}\n', encoding="utf-8")
    manifest = KF.manifest_path(spec)
    manifest.write_text(json.dumps({
        "built_sha256": sha256(notebook),
        "declared_slug": spec["slug"],
    }) + "\n", encoding="utf-8")

    dest = tmp_path / "fetched"
    dest.mkdir()
    submission = dest / "submission.csv"
    submission_frame().to_csv(submission, index=False)
    return {
        "spec": spec, "spec_path": spec_path, "dest": dest,
        "submission": submission, "notebook": notebook, "manifest": manifest,
        "auditor": auditor, "ledger": KF.DEFECT_LEDGER,
    }


def issue_receipt(case: dict, version: int = 7) -> dict:
    assert KF.cmd_audit(case["spec"], case["dest"], version) == 0
    return json.loads(KF.receipt_path(case["dest"]).read_text(encoding="utf-8"))


def test_audit_writes_complete_durable_release_receipt(release_repo: dict) -> None:
    receipt = issue_receipt(release_repo)
    report = release_repo["dest"] / KF.AUDIT_REPORT

    assert receipt["schema_version"] == 1
    assert receipt["verdict"] == "PASS"
    assert receipt["spec"]["path"] == "scripts/kaggle_specs/candidate.json"
    assert receipt["spec"]["sha256"] == KF.normalized_spec_sha256(release_repo["spec"])
    assert receipt["build"]["notebook_sha256"] == sha256(release_repo["notebook"])
    assert receipt["defect_ledger"]["sha256"] == sha256(release_repo["ledger"])
    assert receipt["submission"]["sha256"] == sha256(release_repo["submission"])
    assert receipt["auditor"]["sha256"] == sha256(release_repo["auditor"])
    assert receipt["audit_report"]["sha256"] == sha256(report)
    assert receipt["kernel"] == {
        "owner": KF.OWNER, "slug": "candidate-kernel", "version": 7,
    }


def test_submitcmd_emits_only_the_recorded_exact_version(
    release_repo: dict, capsys,
) -> None:
    issue_receipt(release_repo, version=7)
    assert KF.cmd_submitcmd(
        release_repo["spec"], "candidate", release_repo["dest"], 7,
    ) == 0
    output = capsys.readouterr().out
    assert "-v 7" in output
    assert "<VERSION>" not in output
    assert sha256(release_repo["submission"]) in output


def test_submitcmd_refuses_non_submission_spec(release_repo: dict) -> None:
    release_repo["spec"]["expects_submission"] = False
    with pytest.raises(SystemExit, match="not submission-authorized"):
        KF.cmd_submitcmd(release_repo["spec"], "x", release_repo["dest"], 7)


def test_submitcmd_refuses_absent_receipt(release_repo: dict) -> None:
    with pytest.raises(SystemExit, match="run `audit"):
        KF.cmd_submitcmd(release_repo["spec"], "x", release_repo["dest"], 7)


def test_submitcmd_refuses_tampered_receipt(release_repo: dict) -> None:
    receipt = issue_receipt(release_repo)
    receipt["submission"]["sha256"] = "0" * 64
    KF.receipt_path(release_repo["dest"]).write_text(
        json.dumps(receipt), encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="no longer matches.*submission"):
        KF.cmd_submitcmd(release_repo["spec"], "x", release_repo["dest"], 7)


def test_submitcmd_refuses_changed_audit_report(release_repo: dict) -> None:
    issue_receipt(release_repo)
    report = release_repo["dest"] / KF.AUDIT_REPORT
    report.write_bytes(report.read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="audit report is absent or changed"):
        KF.cmd_submitcmd(release_repo["spec"], "x", release_repo["dest"], 7)


def test_audit_and_submitcmd_require_positive_explicit_version(release_repo: dict) -> None:
    with pytest.raises(SystemExit, match="explicit positive"):
        KF.cmd_audit(release_repo["spec"], release_repo["dest"], None)
    issue_receipt(release_repo, version=7)
    with pytest.raises(SystemExit, match="explicit positive"):
        KF.cmd_submitcmd(release_repo["spec"], "x", release_repo["dest"], None)
    with pytest.raises(SystemExit, match="no longer matches.*kernel"):
        KF.cmd_submitcmd(release_repo["spec"], "x", release_repo["dest"], 8)


@pytest.mark.parametrize(
    "changed",
    ["submission", "notebook", "manifest", "ledger", "auditor", "spec"],
)
def test_submitcmd_refuses_every_changed_bound_input(
    release_repo: dict, changed: str,
) -> None:
    issue_receipt(release_repo)
    if changed == "spec":
        release_repo["spec"]["title"] = "Changed Candidate"
    else:
        path = release_repo[changed]
        if changed == "notebook":
            path.write_text('{"cells": [{"cell_type": "code"}]}\n', encoding="utf-8")
            # Keep the manifest internally consistent; its receipt hash must still expose drift.
            manifest = release_repo["manifest"]
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["built_sha256"] = sha256(path)
            manifest.write_text(json.dumps(data) + "\n", encoding="utf-8")
        else:
            path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(SystemExit, match="no longer matches|built notebook"):
        KF.cmd_submitcmd(release_repo["spec"], "x", release_repo["dest"], 7)


def test_failed_reaudit_removes_prior_receipt(release_repo: dict) -> None:
    issue_receipt(release_repo)
    frame = submission_frame()
    frame["t"] = frame["t"].astype(float)
    frame.loc[0, "t"] = 0.5
    frame.to_csv(release_repo["submission"], index=False)

    assert KF.cmd_audit(release_repo["spec"], release_repo["dest"], 7) == 1
    assert not KF.receipt_path(release_repo["dest"]).exists()
    assert not (release_repo["dest"] / KF.AUDIT_REPORT).exists()
