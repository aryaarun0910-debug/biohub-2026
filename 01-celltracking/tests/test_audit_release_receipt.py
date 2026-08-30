"""Self-test for the release-receipt auditor: every condition must be able to FAIL.

A PASS is only worth what the instrument can catch. Each test here plants ONE defect in an
otherwise-clean sandbox release and asserts (a) the overall verdict flips to FAIL and (b) the
NAMED check is the one that flipped - so a defect cannot be masked by an unrelated failure.

The sandbox is a real git repository with a real commit, so the git-identity checks are exercised
rather than skipped.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "win_bet" / "audit_release_receipt.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_release_receipt", MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _nb(cells: list[str]) -> dict:
    return {"cells": [{"cell_type": "code", "source": [c], "metadata": {}, "outputs": [],
                       "execution_count": None} for c in cells],
            "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def _structural(sha: str, *, fractional: int = 0, failing: bool = False,
                indegree: int = 1) -> list:
    return [{
        "sha256": sha, "rows": 30, "nodes": 20, "edges": 10, "n_datasets": 1,
        "datasets": ["44b6_x"], "divisions": 2,
        "max_indegree": indegree, "max_outdegree": 2,
        "checks": [
            {"check": "A1 schema", "pass": not failing, "detail": "columns match"},
            {"check": "A1 numeric", "pass": True,
             "detail": f"non_numeric_or_nonfinite=0; fractional={fractional}"},
            {"check": "A8 in_degree", "pass": True, "detail": f"max in-degree={indegree}"},
        ],
        "verdict": "PASS" if not failing else "FAIL",
    }]


@pytest.fixture()
def sandbox(tmp_path: Path):
    """A clean, passing release: spec, notebook, manifest, metadata, fetched graph, registry."""
    mod = _load_module()
    root = tmp_path / "repo"
    (root / "scripts" / "kaggle_specs").mkdir(parents=True)
    (root / "scripts" / "core").mkdir(parents=True)
    (root / "notebooks" / "kaggle_demo" / "_out").mkdir(parents=True)
    (root / "research" / "00-system" / "registry").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    nb_path = root / "notebooks" / "kaggle_demo" / "demo.ipynb"
    nb_path.write_text(json.dumps(_nb(["print('a')", "print('b')"]), indent=1), encoding="utf-8")
    nb_sha = _sha(nb_path.read_bytes())

    spec = {
        "name": "demo", "slug": "biohub-demo", "title": "Demo", "code_file": "demo.ipynb",
        "out_dir": "notebooks/kaggle_demo",
        "datasets": ["owner/weights-a", "owner/weights-b"],
        "competition_sources": ["comp"], "enable_gpu": True, "enable_internet": False,
        "expects_submission": True, "edits": [],
    }
    spec_path = root / "scripts" / "kaggle_specs" / "demo.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    spec_norm = mod.normalized_spec_sha256(spec)

    (root / "notebooks" / "kaggle_demo" / "build_manifest.json").write_text(json.dumps({
        "name": "demo", "base_sha256": nb_sha, "built_sha256": nb_sha, "n_cells": 2, "edits": [],
        "declared_slug": "biohub-demo", "pushed_slug": "biohub-demo",
        "defect_gate": {"ledger": "scripts/core/experiment_defects.json", "passed": []},
    }, indent=2), encoding="utf-8")
    (root / "notebooks" / "kaggle_demo" / "kernel-metadata.json").write_text(json.dumps({
        "id": "aryaarun07/biohub-demo", "code_file": "demo.ipynb",
        "dataset_sources": ["owner/weights-a", "owner/weights-b"],
        "competition_sources": ["comp"], "enable_gpu": True, "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
    }, indent=2), encoding="utf-8")
    (root / "scripts" / "core" / "experiment_defects.json").write_text(
        json.dumps({"schema_version": 1, "defects": [
            {"id": "DG-999-other", "scope": {"spec_name_globs": ["something_else*"]},
             "checks": []}]}, indent=2), encoding="utf-8")

    csv_path = root / "notebooks" / "kaggle_demo" / "_out" / "submission.csv"
    csv_path.write_text("id,dataset,row_type\n0,44b6_x,node\n", encoding="utf-8")
    csv_sha = _sha(csv_path.read_bytes())
    (root / "notebooks" / "kaggle_demo" / "_out" / "structural_audit.json").write_text(
        json.dumps(_structural(csv_sha), indent=2), encoding="utf-8")
    (root / "notebooks" / "kaggle_demo" / "_out" / "audit_receipt.json").write_text(json.dumps({
        "schema_version": 1, "verdict": "PASS",
        "spec": {"path": "scripts/kaggle_specs/demo.json", "sha256": spec_norm},
        "build": {"notebook_sha256": nb_sha},
        "submission": {"path": "submission.csv", "sha256": csv_sha, "bytes": 30},
        "kernel": {"owner": "aryaarun07", "slug": "biohub-demo", "version": 1},
    }, indent=2), encoding="utf-8")

    reg = root / "research" / "00-system" / "registry"
    reg.joinpath("experiments.yaml").write_text(
        "experiments:\n"
        "  - id: EXP-9999\n"
        "    spec: scripts/kaggle_specs/demo.json\n"
        "    kernel: aryaarun07/biohub-demo\n"
        "    kernel_version: 1\n"
        "    submission: 12345\n"
        "    lb: 0.931\n"
        "    status: scored\n", encoding="utf-8")
    reg.joinpath("facts.yaml").write_text(
        "facts:\n"
        "  - id: FACT-9999\n"
        "    statement: demo\n"
        "    value: 0.931\n"
        "    provenance: MEASURED\n"
        "    validity: VALID\n"
        "    experiment: EXP-9999\n"
        "    scope:\n"
        "      submission: 12345\n"
        "      preregistered:\n"
        "        band: [0.925, 0.932]\n"
        "        central: 0.929\n"
        "        falsifier: at or below 0.928\n"
        "      outcome:\n"
        "        scored: 0.931\n"
        "        band_hit: true\n"
        "        falsifier_fired: false\n", encoding="utf-8")

    ref = tmp_path / "upstream.ipynb"
    ref.write_text(json.dumps(_nb(["print('a')", "print('b')"]), indent=4), encoding="utf-8")

    g = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
    subprocess.run([*g, "commit", "-q", "-m", "sandbox"], cwd=root, check=True, capture_output=True)

    mod.ROOT = root
    mod.REGISTRY = reg
    return {"mod": mod, "root": root, "spec_path": spec_path, "nb": nb_path, "csv": csv_path,
            "ref": ref, "out": root / "notebooks" / "kaggle_demo" / "_out",
            "nbdir": root / "notebooks" / "kaggle_demo"}


def _run(sb, *, reference=True):
    return sb["mod"].audit_release(sb["spec_path"], "EXP-9999",
                                   sb["ref"] if reference else None, "HEAD")


def _write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def test_clean_release_passes(sandbox):
    res = _run(sandbox)
    failed = [k for k, v in res["checks"].items() if not v["passed"]]
    assert res["passed"] is True, failed
    assert len(res["checks"]) >= 20


def test_missing_fetched_graph_fails(sandbox):
    for p in sandbox["out"].iterdir():
        p.unlink()
    res = _run(sandbox)
    assert res["passed"] is False
    assert res["checks"]["graph_fetched"]["passed"] is False


def test_float_coordinates_fail(sandbox):
    csv_sha = _sha(sandbox["csv"].read_bytes())
    _write(sandbox["out"] / "structural_audit.json", _structural(csv_sha, fractional=17))
    res = _run(sandbox)
    assert res["passed"] is False
    assert res["checks"]["graph_coordinates_integer"]["passed"] is False
    assert res["checks"]["graph_structurally_legal"]["passed"] is True


def test_failing_structural_check_fails(sandbox):
    csv_sha = _sha(sandbox["csv"].read_bytes())
    _write(sandbox["out"] / "structural_audit.json", _structural(csv_sha, failing=True))
    res = _run(sandbox)
    assert res["passed"] is False
    assert res["checks"]["graph_structurally_legal"]["passed"] is False


def test_illegal_indegree_fails(sandbox):
    csv_sha = _sha(sandbox["csv"].read_bytes())
    _write(sandbox["out"] / "structural_audit.json", _structural(csv_sha, indegree=2))
    res = _run(sandbox)
    assert res["passed"] is False
    assert res["checks"]["graph_non_empty_and_degree_legal"]["passed"] is False


def test_graph_mutated_after_audit_fails(sandbox):
    sandbox["csv"].write_text("id,dataset,row_type\n0,44b6_x,node\n1,44b6_x,node\n",
                              encoding="utf-8")
    res = _run(sandbox)
    assert res["passed"] is False
    assert res["checks"]["graph_matches_audit_receipt"]["passed"] is False


def test_notebook_edited_after_build_fails(sandbox):
    sandbox["nb"].write_text(json.dumps(_nb(["print('a')", "print('MUTATED')"]), indent=1),
                             encoding="utf-8")
    res = _run(sandbox)
    assert res["passed"] is False
    assert res["checks"]["notebook_matches_manifest"]["passed"] is False
    assert res["checks"]["notebook_identical_to_commit"]["passed"] is False


def test_slug_divergence_fails(sandbox):
    man = json.loads((sandbox["nbdir"] / "build_manifest.json").read_text(encoding="utf-8"))
    man["pushed_slug"] = "biohub-demo-2"
    _write(sandbox["nbdir"] / "build_manifest.json", man)
    res = _run(sandbox)
    assert res["passed"] is False
    assert res["checks"]["slug_binding"]["passed"] is False


def test_undeclared_weights_dataset_fails(sandbox):
    meta = json.loads((sandbox["nbdir"] / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["dataset_sources"].append("owner/weights-SMUGGLED")
    _write(sandbox["nbdir"] / "kernel-metadata.json", meta)
    res = _run(sandbox)
    assert res["passed"] is False
    assert res["checks"]["weights_binding"]["passed"] is False


def test_spec_edited_after_audit_fails(sandbox):
    spec = json.loads(sandbox["spec_path"].read_text(encoding="utf-8"))
    spec["enable_internet"] = True
    _write(sandbox["spec_path"], spec)
    res = _run(sandbox)
    assert res["passed"] is False
    assert res["checks"]["audit_receipt_binds_this_notebook"]["passed"] is False


def test_score_outside_band_fails(sandbox):
    reg = sandbox["root"] / "research" / "00-system" / "registry" / "facts.yaml"
    reg.write_text(reg.read_text(encoding="utf-8").replace("scored: 0.931", "scored: 0.9"),
                   encoding="utf-8")
    res = _run(sandbox)
    assert res["passed"] is False
    assert res["checks"]["score_inside_preregistered_band"]["passed"] is False


def test_fired_falsifier_fails(sandbox):
    reg = sandbox["root"] / "research" / "00-system" / "registry" / "facts.yaml"
    reg.write_text(reg.read_text(encoding="utf-8").replace("falsifier_fired: false",
                                                           "falsifier_fired: true"),
                   encoding="utf-8")
    res = _run(sandbox)
    assert res["passed"] is False
    assert res["checks"]["falsifier_did_not_fire"]["passed"] is False


def test_registry_score_disagreement_fails(sandbox):
    reg = sandbox["root"] / "research" / "00-system" / "registry" / "experiments.yaml"
    reg.write_text(reg.read_text(encoding="utf-8").replace("lb: 0.931", "lb: 0.928"),
                   encoding="utf-8")
    res = _run(sandbox)
    assert res["passed"] is False
    assert res["checks"]["score_matches_registry_experiment"]["passed"] is False


def test_upstream_source_divergence_fails(sandbox):
    sandbox["ref"].write_text(json.dumps(_nb(["print('a')", "print('DIFFERENT')"]), indent=4),
                              encoding="utf-8")
    res = _run(sandbox)
    assert res["passed"] is False
    assert res["checks"]["reproduces_upstream_source"]["passed"] is False


def test_upstream_equality_ignores_json_formatting(sandbox):
    """The factory re-serialises the notebook at build time, so byte equality is the wrong test;
    cell-source equality is the right one. This asserts the instrument uses the right one."""
    assert sandbox["ref"].read_bytes() != sandbox["nb"].read_bytes()
    res = _run(sandbox)
    assert res["checks"]["reproduces_upstream_source"]["passed"] is True


def test_vacuous_defect_gate_is_reported_not_hidden(sandbox):
    res = _run(sandbox)
    gate = res["checks"]["defect_gate_not_silently_vacuous"]
    assert gate["passed"] is True and gate["vacuous"] is True
    assert gate["applicable_rules"] == []


def test_applicable_defect_rule_not_recorded_fails(sandbox):
    ledger = sandbox["root"] / "scripts" / "core" / "experiment_defects.json"
    _write(ledger, {"schema_version": 1, "defects": [
        {"id": "DG-777-applies", "scope": {"spec_name_globs": ["demo*"]}, "checks": []}]})
    res = _run(sandbox)
    assert res["passed"] is False
    gate = res["checks"]["defect_gate_not_silently_vacuous"]
    assert gate["passed"] is False and gate["applicable_rules"] == ["DG-777-applies"]


def test_missing_notebook_fails_closed(sandbox):
    sandbox["nb"].unlink()
    res = _run(sandbox)
    assert res["passed"] is False
    assert res["checks"]["notebook_present"]["passed"] is False


def test_no_commit_named_is_a_failure_not_a_skip(sandbox):
    res = sandbox["mod"].audit_release(sandbox["spec_path"], "EXP-9999", None, None)
    assert res["passed"] is False
    assert res["checks"]["notebook_identical_to_commit"]["passed"] is False
