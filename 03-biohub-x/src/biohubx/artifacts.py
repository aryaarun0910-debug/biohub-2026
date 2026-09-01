"""Artifact records, atomic materialisation and registry verification.

Consumer: :mod:`biohubx.cli` (``biohubx artifacts digest``, ``biohubx artifacts
verify``).

An artifact is anything whose identity Biohub-X asserts: a hash-bound research
snapshot, an exported candidate graph, a checkpoint, a built notebook, a
submission. The registry at ``registry/artifacts.yaml`` is the only place those
assertions live, and every digest in it is a typed token (see
:mod:`biohubx.hashing`).
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from biohubx.hashing import Digest, DigestKind, NotTextError, digest_file

ARTIFACT_REGISTRY_PATH = Path("registry/artifacts.yaml")
MANIFEST_DIR = Path("artifacts/manifests")


class ProvenanceStatus(StrEnum):
    """What standing an artifact has inside the system."""

    REFERENCE_ONLY = "reference_only"
    """Read as input by humans and agents. Confers no belief; never imported."""

    MEASURED = "measured"
    """Produced by a Biohub-X experiment inside this repository."""

    EXTERNAL_UNCLEARED = "external_uncleared"
    """From outside; licence and training-data provenance not yet cleared."""

    EXTERNAL_CLEARED = "external_cleared"
    """From outside; provenance recorded and cleared for use."""


DigestToken = Annotated[str, Field(pattern=r"^[a-z0-9_]+:[a-z0-9]+(?:/v\d+)?:[0-9a-f]{64}$")]

DigestSlot = Literal["raw", "canonical_text"]

_SLOT_KIND: dict[str, DigestKind] = {
    "raw": DigestKind.RAW_ARTIFACT,
    "canonical_text": DigestKind.CANONICAL_TEXT,
}


class ArtifactProvenance(BaseModel):
    """Where an artifact came from. Absent provenance is a hard failure."""

    model_config = ConfigDict(extra="forbid")

    status: ProvenanceStatus
    imported_utc: datetime | None = None
    original_path: str | None = None
    source_url: str | None = None
    source_commit: str | None = None
    source_digest: DigestToken | None = None
    code_license: str | None = None
    weight_license: str | None = None
    training_data: str | None = None
    competition_eligible: bool | None = None
    note: str | None = None


class ArtifactRecord(BaseModel):
    """One artifact and the identities Biohub-X asserts about it."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    path: str = Field(min_length=1)
    schema_: str = Field(alias="schema", min_length=1)
    digests: dict[DigestSlot, DigestToken] = Field(min_length=1)
    produced_by: str | None = None
    provenance: ArtifactProvenance

    @field_validator("path")
    @classmethod
    def _refuse_absolute(cls, value: str) -> str:
        # Both path flavours are checked, because `pathlib` only understands the
        # host convention: a POSIX-rooted path is not absolute to a Windows
        # interpreter, and a drive-letter path is not absolute to a POSIX one.
        # A registry written on one machine must be refused identically on the
        # other, so neither flavour is allowed to slip through.
        if PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute():
            raise ValueError(f"artifact path must be repository-relative, got {value!r}")
        if ".." in re.split(r"[\\/]", value):
            raise ValueError(f"artifact path must not escape the repository, got {value!r}")
        return value

    def digest(self, slot: DigestSlot) -> Digest:
        return Digest.parse(self.digests[slot])


class ArtifactRegistry(BaseModel):
    """The whole of ``registry/artifacts.yaml``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    artifacts: list[ArtifactRecord]

    @field_validator("artifacts")
    @classmethod
    def _unique_ids(cls, value: list[ArtifactRecord]) -> list[ArtifactRecord]:
        seen: set[str] = set()
        for record in value:
            if record.id in seen:
                raise ValueError(f"duplicate artifact id: {record.id!r}")
            seen.add(record.id)
        return value


class DigestCheck(BaseModel):
    """Result of comparing one recorded digest against the bytes on disk."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    slot: str
    expected: str
    observed: str | None
    ok: bool
    detail: str | None = None


def load_artifact_registry(path: Path) -> ArtifactRegistry:
    """Load and validate the artifact registry. A missing registry is a failure."""
    if not path.is_file():
        raise FileNotFoundError(f"artifact registry not found: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if document is None:
        raise ValueError(f"artifact registry is empty: {path}")
    return ArtifactRegistry.model_validate(document)


def verify_registry(registry: ArtifactRegistry, root: Path) -> list[DigestCheck]:
    """Re-derive every recorded digest from the bytes on disk.

    A missing file, an unreadable file, a mismatched digest or a digest filed
    under the wrong slot is a failed check. Nothing is skipped and nothing falls
    back to a default location.
    """
    checks: list[DigestCheck] = []
    for record in registry.artifacts:
        target = root / record.path
        for slot, token in record.digests.items():
            expected = Digest.parse(token)
            kind = _SLOT_KIND[slot]
            if expected.kind is not kind:
                checks.append(
                    DigestCheck(
                        artifact_id=record.id,
                        slot=slot,
                        expected=token,
                        observed=None,
                        ok=False,
                        detail=f"slot {slot} holds a {expected.kind.value} digest",
                    )
                )
                continue
            if not target.is_file():
                checks.append(
                    DigestCheck(
                        artifact_id=record.id,
                        slot=slot,
                        expected=token,
                        observed=None,
                        ok=False,
                        detail=f"file not found: {record.path}",
                    )
                )
                continue
            try:
                observed = digest_file(target, kind)
            except NotTextError as exc:
                checks.append(
                    DigestCheck(
                        artifact_id=record.id,
                        slot=slot,
                        expected=token,
                        observed=None,
                        ok=False,
                        detail=str(exc),
                    )
                )
                continue
            matched = observed.token == token
            checks.append(
                DigestCheck(
                    artifact_id=record.id,
                    slot=slot,
                    expected=token,
                    observed=observed.token,
                    ok=matched,
                    detail=None if matched else "digest mismatch",
                )
            )
    return checks


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes so that a reader never observes a partial file.

    The temporary file is created in the destination directory so that
    :func:`os.replace` is a same-filesystem rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".partial")
    tmp = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # Same-directory rename, so the replacement is atomic on both platforms.
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text with LF line endings, matching repository policy."""
    atomic_write_bytes(path, text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"))
