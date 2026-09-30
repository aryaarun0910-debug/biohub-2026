"""Mutation test: line-ending drift must be caught by raw identity and absorbed
by canonical identity.

The failure this guards against is concrete. A checkout on a machine with
different line-ending settings rewrites every text file. Under a single hash
that is indistinguishable from someone having edited the model configuration,
so either every verification fails for no reason, or verification gets relaxed
until it stops catching real edits. D-0002 splits the two questions; this test
mutates a file both ways and asserts the split actually works.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biohubx.artifacts import ArtifactRegistry, atomic_write_bytes, verify_registry
from biohubx.hashing import canonical_text_digest, raw_digest


def _registry(
    path: str, raw_token: str | None = None, canonical_token: str | None = None
) -> ArtifactRegistry:
    digests: dict[str, str] = {}
    if raw_token is not None:
        digests["raw"] = raw_token
    if canonical_token is not None:
        digests["canonical_text"] = canonical_token
    return ArtifactRegistry.model_validate(
        {
            "schema_version": 1,
            "artifacts": [
                {
                    "id": "subject",
                    "kind": "config",
                    "path": path,
                    "schema": "yaml",
                    "digests": digests,
                    "provenance": {"status": "measured"},
                }
            ],
        }
    )


LF = b"depth: 4\nabstention: true\n"
CRLF = b"depth: 4\r\nabstention: true\r\n"
EDITED = b"depth: 6\nabstention: true\n"


def test_line_ending_mutation_breaks_raw_identity(tmp_path: Path) -> None:
    target = tmp_path / "model.yaml"
    target.write_bytes(LF)
    registry = _registry("model.yaml", raw_token=raw_digest(LF).token)
    assert all(c.ok for c in verify_registry(registry, tmp_path))

    target.write_bytes(CRLF)  # the mutation
    checks = verify_registry(registry, tmp_path)
    assert not checks[0].ok, "raw identity must notice that the bytes changed"
    assert checks[0].detail == "digest mismatch"


def test_line_ending_mutation_does_not_break_canonical_identity(tmp_path: Path) -> None:
    target = tmp_path / "model.yaml"
    target.write_bytes(LF)
    registry = _registry("model.yaml", canonical_token=canonical_text_digest(LF).token)
    assert all(c.ok for c in verify_registry(registry, tmp_path))

    target.write_bytes(CRLF)  # the same mutation
    assert all(c.ok for c in verify_registry(registry, tmp_path)), (
        "canonical identity must survive a pure line-ending rewrite"
    )


def test_a_real_content_edit_breaks_both_identities(tmp_path: Path) -> None:
    # The complement: canonical identity must not be so forgiving that it stops
    # catching an actual change. Depth 4 becoming depth 6 is a different system.
    target = tmp_path / "model.yaml"
    target.write_bytes(LF)
    registry = _registry(
        "model.yaml",
        raw_token=raw_digest(LF).token,
        canonical_token=canonical_text_digest(LF).token,
    )
    assert all(c.ok for c in verify_registry(registry, tmp_path))

    target.write_bytes(EDITED)
    checks = verify_registry(registry, tmp_path)
    assert not any(c.ok for c in checks), "a content edit must break both identities"


def test_atomic_write_leaves_no_partial_file_when_serialisation_fails(tmp_path: Path) -> None:
    target = tmp_path / "out" / "graph.json"
    # The write fails after the temporary file has been created. The contract is
    # that a reader never sees a partial file, so on failure the destination must
    # not exist at all and no .partial may be left behind.
    with pytest.raises(TypeError):
        atomic_write_bytes(target, "not bytes")  # type: ignore[arg-type]
    assert not target.exists()
    assert not list(target.parent.glob("*.partial")), "temporary file was left behind"


def test_atomic_write_replaces_previous_content_completely(tmp_path: Path) -> None:
    target = tmp_path / "graph.json"
    atomic_write_bytes(target, b"a very long previous version\n")
    atomic_write_bytes(target, b"short\n")
    assert target.read_bytes() == b"short\n"
