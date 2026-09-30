"""Registration as a command rather than a hand edit.

Registration is the step that turns a measured identity into a claim the
repository makes. Doing it by editing YAML would mean retyping digests, which
is exactly the manual duplication the repository forbids everywhere else.

The rule these tests fix is that registration is safe to repeat and refuses to
overwrite a claim it did not make.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from biohubx.artifacts import load_artifact_registry, verify_registry
from biohubx.cli import app

runner = CliRunner()


def official_root(base: Path) -> Path:
    train = base / "train"
    (train / "d1.zarr" / "0").mkdir(parents=True)
    (train / "d1.zarr" / "0" / "chunk").write_bytes(b"volume-bytes")
    (train / "d1.geff" / "nodes").mkdir(parents=True)
    (train / "d1.geff" / "nodes" / "ids").write_bytes(b"node-ids")
    test = base / "test"
    (test / "d2.zarr" / "0").mkdir(parents=True)
    (test / "d2.zarr" / "0" / "chunk").write_bytes(b"held-out")
    return base


def empty_registry(path: Path) -> Path:
    path.write_text("schema_version: 1\nartifacts: []\n", encoding="utf-8")
    return path


def fingerprint(root: Path) -> Path:
    from biohubx.cli import repository_root

    assert runner.invoke(app, ["data", "fingerprint", "--root", str(root)]).exit_code == 0
    return repository_root() / "artifacts/data-fingerprint.json"


def register(report: Path, registry: Path) -> int:
    result = runner.invoke(app, ["artifacts", "register", "--from", str(report), "--registry", str(registry)])
    return result.exit_code


def test_registration_adds_every_fingerprinted_artifact(tmp_path: Path) -> None:
    root = official_root(tmp_path / "data")
    registry = empty_registry(tmp_path / "artifacts.yaml")
    assert register(fingerprint(root), registry) == 0

    loaded = load_artifact_registry(registry)
    assert {record.id for record in loaded.artifacts} == {
        "competition.train.d1.zarr",
        "competition.train.d1.geff",
        "competition.test.d2.zarr",
    }
    for record in loaded.artifacts:
        assert record.is_tree
        assert record.external_path is not None
        assert record.shape is not None


def test_a_registered_artifact_verifies_immediately(tmp_path: Path) -> None:
    # Registration that produced something unverifiable would be worse than none.
    root = official_root(tmp_path / "data")
    registry = empty_registry(tmp_path / "artifacts.yaml")
    assert register(fingerprint(root), registry) == 0

    loaded = load_artifact_registry(registry)
    for deep in (False, True):
        checks = verify_registry(loaded, tmp_path, deep=deep)
        assert checks and all(check.ok for check in checks)


def test_registering_twice_changes_nothing(tmp_path: Path) -> None:
    root = official_root(tmp_path / "data")
    registry = empty_registry(tmp_path / "artifacts.yaml")
    report = fingerprint(root)

    assert register(report, registry) == 0
    first = registry.read_text(encoding="utf-8")
    assert register(report, registry) == 0
    assert registry.read_text(encoding="utf-8") == first

    document = yaml.safe_load(first)
    assert len(document["artifacts"]) == 3


def test_a_changed_dataset_under_a_registered_id_is_refused(tmp_path: Path) -> None:
    """The conflict that must not be resolved automatically.

    The same id now carries a different identity, which means the data changed
    underneath a claim the registry already makes. Whether that is a re-download
    or a problem is not something a command can know.
    """
    root = official_root(tmp_path / "data")
    registry = empty_registry(tmp_path / "artifacts.yaml")
    assert register(fingerprint(root), registry) == 0
    before = registry.read_text(encoding="utf-8")

    (root / "train" / "d1.zarr" / "0" / "chunk").write_bytes(b"different-bytes-entirely")
    assert register(fingerprint(root), registry) == 1
    assert registry.read_text(encoding="utf-8") == before, "a refused merge must write nothing"


def test_a_plan_report_carries_no_identity_and_is_refused(tmp_path: Path) -> None:
    from biohubx.cli import repository_root

    root = official_root(tmp_path / "data")
    registry = empty_registry(tmp_path / "artifacts.yaml")
    assert runner.invoke(app, ["data", "fingerprint", "--root", str(root), "--plan"]).exit_code == 0
    plan = repository_root() / "artifacts/data-fingerprint-plan.json"
    assert register(plan, registry) == 2


def test_a_missing_report_is_refused(tmp_path: Path) -> None:
    registry = empty_registry(tmp_path / "artifacts.yaml")
    assert register(tmp_path / "absent.json", registry) == 2


def test_registration_preserves_artifacts_it_did_not_add(tmp_path: Path) -> None:
    root = official_root(tmp_path / "data")
    registry = tmp_path / "artifacts.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "artifacts": [
                    {
                        "id": "pre.existing",
                        "kind": "document",
                        "path": "README.md",
                        "schema": "markdown",
                        "digests": {"raw": "raw_artifact_sha256:sha256:" + "0" * 64},
                        "provenance": {"status": "measured"},
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    assert register(fingerprint(root), registry) == 0
    loaded = load_artifact_registry(registry)
    assert "pre.existing" in {record.id for record in loaded.artifacts}
    assert len(loaded.artifacts) == 4


def test_the_registered_provenance_is_honest_about_not_being_cleared(tmp_path: Path) -> None:
    # Fingerprinting establishes identity, not licence or eligibility. The
    # record must not imply a review that nobody has done.
    root = official_root(tmp_path / "data")
    registry = empty_registry(tmp_path / "artifacts.yaml")
    assert register(fingerprint(root), registry) == 0
    for record in load_artifact_registry(registry).artifacts:
        assert record.provenance.status.value == "external_uncleared"
        assert record.provenance.competition_eligible is None


def test_the_run_manifest_records_what_was_added(tmp_path: Path) -> None:
    from biohubx.cli import repository_root

    root = official_root(tmp_path / "data")
    registry = empty_registry(tmp_path / "artifacts.yaml")
    assert register(fingerprint(root), registry) == 0
    manifest = json.loads(
        (repository_root() / "artifacts/manifests/artifacts-register.json").read_text(encoding="utf-8")
    )
    assert sorted(manifest["added"]) == [
        "competition.test.d2.zarr",
        "competition.train.d1.geff",
        "competition.train.d1.zarr",
    ]
    assert manifest["unchanged"] == []
