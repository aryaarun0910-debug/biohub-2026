"""The CLI contract: heartbeat, refusal, atomic manifest, no silent defaults."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from biohubx import __version__
from biohubx.cli import app, repository_root

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_repository_root_is_derived_from_the_package_not_the_cwd() -> None:
    # A root that follows the process CWD is how a command silently operates on
    # the wrong repository.
    assert repository_root() == REPO_ROOT


def test_no_args_shows_help_rather_than_doing_something_default() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code != 0
    assert "Usage" in result.output


def test_digest_requires_an_explicit_kind() -> None:
    result = runner.invoke(app, ["artifacts", "digest", "README.md"])
    assert result.exit_code != 0, "a digest whose kind was chosen for you is not an identity"


def test_digest_prints_a_typed_token_and_a_heartbeat() -> None:
    target = REPO_ROOT / "research/primitives-dossier.md"
    result = runner.invoke(app, ["artifacts", "digest", str(target), "--kind", "raw_artifact_sha256"])
    assert result.exit_code == 0
    assert result.stdout.strip().startswith("raw_artifact_sha256:sha256:")


def test_digest_refuses_a_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["artifacts", "digest", str(tmp_path / "absent.md"), "--kind", "raw_artifact_sha256"]
    )
    assert result.exit_code == 2


def test_verify_passes_on_the_repository_registry_and_writes_a_manifest() -> None:
    result = runner.invoke(app, ["artifacts", "verify"])
    assert result.exit_code == 0, result.output

    manifest_path = REPO_ROOT / "artifacts/manifests/artifacts-verify.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["command"] == "artifacts verify"
    assert manifest["failed"] == 0
    assert manifest["checked"] >= 1
    assert manifest["canonicalization_version"] == "v1"
    assert manifest["biohubx_version"] == __version__
    # The manifest records what was checked, not just that something was.
    assert all(check["ok"] for check in manifest["checks"])


def test_verify_refuses_a_missing_registry(tmp_path: Path) -> None:
    result = runner.invoke(app, ["artifacts", "verify", "--registry", str(tmp_path / "absent.yaml")])
    assert result.exit_code == 2


def test_verify_fails_when_a_recorded_identity_no_longer_holds(tmp_path: Path) -> None:
    subject = tmp_path / "subject.md"
    subject.write_bytes(b"original\n")
    registry = tmp_path / "artifacts.yaml"
    registry.write_text(
        "schema_version: 1\n"
        "artifacts:\n"
        "  - id: subject\n"
        "    kind: document\n"
        "    path: subject.md\n"
        "    schema: markdown\n"
        "    digests:\n"
        f"      raw: raw_artifact_sha256:sha256:{'0' * 64}\n"
        "    provenance:\n"
        "      status: measured\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["artifacts", "verify", "--registry", str(registry)])
    assert result.exit_code == 1
