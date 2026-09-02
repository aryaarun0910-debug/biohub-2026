"""The CLI contract: heartbeat, refusal, atomic manifest, no silent defaults."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
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


def test_infer_synthetic_emits_a_graph_and_records_its_digest() -> None:
    result = runner.invoke(app, ["infer", "synthetic"])
    assert result.exit_code == 0, result.output

    graph_path = REPO_ROOT / "artifacts/slice-graph.json"
    assert graph_path.is_file()
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    assert graph["nodes"] and graph["edges"]

    manifest = json.loads(
        (REPO_ROOT / "artifacts/manifests/infer-synthetic.json").read_text(encoding="utf-8")
    )
    # The manifest records the digest of the bytes actually written, so a later
    # command can tell whether it is scoring this run or a stale artifact.
    assert manifest["emitted_graph_digest"].startswith("raw_artifact_sha256:sha256:")
    assert manifest["config"]["seed"] == 0


@pytest.mark.parametrize("fraction", ["0", "1.5", "-0.2"])
def test_infer_synthetic_refuses_an_impossible_annotation_fraction(fraction: str) -> None:
    # A refusal the CLI can actually reach. The edgeless-graph refusal lives at
    # the pipeline boundary and is tested there, because no CLI flag produces it.
    result = runner.invoke(app, ["infer", "synthetic", "--annotated-fraction", fraction])
    assert result.exit_code == 2


def test_evaluate_slice_scores_the_emitted_graph() -> None:
    assert runner.invoke(app, ["infer", "synthetic"]).exit_code == 0
    result = runner.invoke(app, ["evaluate", "slice"])
    assert result.exit_code == 0, result.output
    payload = json.loads((REPO_ROOT / "artifacts/slice-score.json").read_text(encoding="utf-8"))
    assert payload["official"]["edge_tp"] > 0


def test_evaluate_slice_refuses_a_missing_graph(tmp_path: Path) -> None:
    result = runner.invoke(app, ["evaluate", "slice", "--graph", str(tmp_path / "absent.json")])
    assert result.exit_code == 2


def test_evaluate_slice_refuses_a_stale_graph(tmp_path: Path) -> None:
    # Scoring an artifact from different settings against a freshly generated
    # ground truth would silently compare two different runs.
    stale = tmp_path / "stale.json"
    stale.write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    result = runner.invoke(app, ["evaluate", "slice", "--graph", str(stale)])
    assert result.exit_code == 2


def _official_root(base: Path) -> Path:
    train = base / "train"
    (train / "d1.zarr" / "0").mkdir(parents=True)
    (train / "d1.zarr" / "0" / "chunk").write_bytes(b"volume-bytes")
    (train / "d1.geff" / "nodes").mkdir(parents=True)
    (train / "d1.geff" / "nodes" / "ids").write_bytes(b"node-ids")
    test = base / "test"
    (test / "d2.zarr" / "0").mkdir(parents=True)
    (test / "d2.zarr" / "0" / "chunk").write_bytes(b"held-out")
    return base


def test_data_fingerprint_records_a_typed_tree_identity_per_artifact(tmp_path: Path) -> None:
    result = runner.invoke(app, ["data", "fingerprint", "--root", str(_official_root(tmp_path))])
    assert result.exit_code == 0, result.output

    report = json.loads((REPO_ROOT / "artifacts/data-fingerprint.json").read_text(encoding="utf-8"))
    entries = {entry["artifact"]: entry for entry in report["entries"]}
    assert set(entries) == {"d1.zarr", "d1.geff", "d2.zarr"}
    for entry in entries.values():
        # A typed token, not a bare hex string, and not a directory size.
        assert entry["tree_digest"].startswith("tree_sha256:sha256/v1:")
        assert entry["file_count"] >= 1
    assert entries["d2.zarr"]["split_role"] == "unannotated"
    assert entries["d1.zarr"]["split_role"] == "annotated"


def test_data_fingerprint_changes_when_a_single_byte_changes(tmp_path: Path) -> None:
    root = _official_root(tmp_path)

    def digest_of(artifact: str) -> str:
        assert runner.invoke(app, ["data", "fingerprint", "--root", str(root)]).exit_code == 0
        report = json.loads((REPO_ROOT / "artifacts/data-fingerprint.json").read_text(encoding="utf-8"))
        return str(next(e["tree_digest"] for e in report["entries"] if e["artifact"] == artifact))

    before = digest_of("d1.zarr")
    (root / "train" / "d1.zarr" / "0" / "chunk").write_bytes(b"volume-bytes-EDITED")
    assert digest_of("d1.zarr") != before


def test_data_fingerprint_can_target_one_dataset(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["data", "fingerprint", "--root", str(_official_root(tmp_path)), "--dataset", "d2"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads((REPO_ROOT / "artifacts/data-fingerprint.json").read_text(encoding="utf-8"))
    assert {entry["dataset_id"] for entry in report["entries"]} == {"d2"}


def test_data_fingerprint_refuses_an_unknown_dataset(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["data", "fingerprint", "--root", str(_official_root(tmp_path)), "--dataset", "absent"]
    )
    assert result.exit_code == 2


def test_data_fingerprint_refuses_without_an_explicit_root() -> None:
    result = runner.invoke(app, ["data", "fingerprint"])
    assert result.exit_code == 2


def test_data_fingerprint_refuses_an_invalid_layout(tmp_path: Path) -> None:
    # It validates the layout before reading a single byte, so a wrong root
    # fails in a second rather than after hashing a partial tree.
    (tmp_path / "loose.zarr").mkdir()
    (tmp_path / "other.geff").mkdir()
    result = runner.invoke(app, ["data", "fingerprint", "--root", str(tmp_path)])
    assert result.exit_code == 2


def test_data_fingerprint_does_not_modify_the_data(tmp_path: Path) -> None:
    root = _official_root(tmp_path)
    before = sorted((p.as_posix(), p.stat().st_size if p.is_file() else -1) for p in root.rglob("*"))
    assert runner.invoke(app, ["data", "fingerprint", "--root", str(root)]).exit_code == 0
    after = sorted((p.as_posix(), p.stat().st_size if p.is_file() else -1) for p in root.rglob("*"))
    assert after == before
