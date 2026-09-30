"""Artifact registry contract.

A recorded hash is only an assertion if something re-derives it. These tests are
what make `registry/artifacts.yaml` binding rather than decorative.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from biohubx.artifacts import (
    ARTIFACT_REGISTRY_PATH,
    ArtifactRecord,
    ArtifactRegistry,
    ProvenanceStatus,
    load_artifact_registry,
    verify_registry,
)
from biohubx.hashing import Digest, DigestKind

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def registry() -> ArtifactRegistry:
    return load_artifact_registry(REPO_ROOT / ARTIFACT_REGISTRY_PATH)


def test_every_recorded_digest_still_holds(registry: ArtifactRegistry) -> None:
    checks = verify_registry(registry, REPO_ROOT)
    assert checks, "the registry asserted nothing"
    failures = [f"{c.artifact_id}/{c.slot}: {c.detail}" for c in checks if not c.ok]
    assert not failures, "recorded identities no longer hold:\n" + "\n".join(failures)


def test_every_digest_is_a_typed_token(registry: ArtifactRegistry) -> None:
    slot_kind = {
        "raw": DigestKind.RAW_ARTIFACT,
        "canonical_text": DigestKind.CANONICAL_TEXT,
        "tree": DigestKind.TREE,
    }
    for record in registry.artifacts:
        for slot, token in record.digests.items():
            digest = Digest.parse(token)
            assert digest.kind is slot_kind[slot], f"{record.id}: slot {slot} holds {digest.kind.value}"


def test_the_dossier_is_registered_as_reference_only(registry: ArtifactRegistry) -> None:
    # The dossier is a research input. If its status ever became anything else,
    # its contents would start conferring standing on primitives that no
    # Biohub-X measurement supports (DECISIONS.md D-0006).
    dossier = next(r for r in registry.artifacts if r.path == "research/primitives-dossier.md")
    assert dossier.provenance.status is ProvenanceStatus.REFERENCE_ONLY
    assert dossier.provenance.original_path, "a hash-bound snapshot must record where it came from"
    assert dossier.provenance.source_digest, "a hash-bound snapshot must record the source identity"


def test_the_shortlist_only_admits_primitives_with_a_live_component() -> None:
    # Phase 0 has no live component, so the shortlist must be empty. Once
    # components exist this test tightens to "every entry names one".
    document = yaml.safe_load((REPO_ROOT / "research/shortlist.yaml").read_text(encoding="utf-8"))
    entries = document["shortlist"]
    for entry in entries:
        assert entry.get("component"), f"shortlist entry {entry.get('id')!r} names no live component"


# --- the registry refuses what it should ----------------------------------


def _record(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "x",
        "kind": "document",
        "path": "research/primitives-dossier.md",
        "schema": "markdown",
        "digests": {"raw": "raw_artifact_sha256:sha256:" + "0" * 64},
        "provenance": {"status": "reference_only"},
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path", "C:/Users/someone/weights.pt"),
        ("path", "/home/someone/weights.pt"),
        ("path", "../outside/weights.pt"),
    ],
    ids=["windows-absolute", "posix-absolute", "escapes-repository"],
)
def test_artifact_paths_outside_the_repository_are_refused(field: str, value: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactRecord.model_validate(_record(**{field: value}))


def test_a_bare_hex_digest_is_refused() -> None:
    with pytest.raises(ValidationError):
        ArtifactRecord.model_validate(_record(digests={"raw": "0" * 64}))


def test_an_unknown_digest_slot_is_refused() -> None:
    with pytest.raises(ValidationError):
        ArtifactRecord.model_validate(_record(digests={"sha": "raw_artifact_sha256:sha256:" + "0" * 64}))


def test_an_artifact_without_provenance_is_refused() -> None:
    incomplete = _record()
    del incomplete["provenance"]
    with pytest.raises(ValidationError):
        ArtifactRecord.model_validate(incomplete)


def test_unknown_fields_are_refused() -> None:
    # extra="forbid" everywhere: a typo in a registry key must fail loudly
    # rather than being silently dropped and quietly believed.
    with pytest.raises(ValidationError):
        ArtifactRecord.model_validate(_record(licence="MIT"))


def test_duplicate_artifact_ids_are_refused() -> None:
    with pytest.raises(ValidationError):
        ArtifactRegistry.model_validate(
            {"schema_version": 1, "artifacts": [_record(id="dup"), _record(id="dup")]}
        )


def test_a_missing_registry_is_a_failure_not_an_empty_result(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_artifact_registry(tmp_path / "absent.yaml")


def test_a_missing_artifact_file_is_a_failed_check_not_a_skip(tmp_path: Path) -> None:
    registry = ArtifactRegistry.model_validate(
        {"schema_version": 1, "artifacts": [_record(path="does/not/exist.md")]}
    )
    checks = verify_registry(registry, tmp_path)
    assert len(checks) == 1
    assert not checks[0].ok
    assert "not found" in (checks[0].detail or "")


def test_a_digest_filed_under_the_wrong_slot_is_a_failed_check(tmp_path: Path) -> None:
    target = tmp_path / "a.md"
    target.write_bytes(b"a\n")
    registry = ArtifactRegistry.model_validate(
        {
            "schema_version": 1,
            "artifacts": [
                _record(path="a.md", digests={"raw": "canonical_text_sha256:sha256/v1:" + "0" * 64})
            ],
        }
    )
    checks = verify_registry(registry, tmp_path)
    assert not checks[0].ok
