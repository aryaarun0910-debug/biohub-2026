"""Artifact records, atomic materialisation and registry verification.

Consumer: :mod:`biohubx.cli` (``biohubx artifacts digest``, ``biohubx artifacts
verify``).

An artifact is anything whose identity Biohub-X asserts: a hash-bound research
snapshot, an exported candidate graph, a checkpoint, a built notebook, a
submission, a competition dataset. The registry at ``registry/artifacts.yaml``
is the only place those assertions live, and every digest in it is a typed token
(see :mod:`biohubx.hashing`).

Verification is tiered, because the artifacts are not alike. A file inside the
repository is re-derived from its bytes on every run: it is small, always
present, and the repository must contain what it claims. A dataset tree is
outside the repository, is machine-local, and takes minutes to read, so by
default it is checked for presence and shape only and is REPORTED as not deeply
verified. ``--deep`` re-derives it. Nothing is ever skipped silently; a check
that was not made says so.
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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from biohubx.hashing import (
    Digest,
    DigestKind,
    NotATreeError,
    NotTextError,
    TreeNotQuiescentError,
    digest_file,
    tree_digest,
    tree_shape,
)

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

DigestSlot = Literal["raw", "canonical_text", "tree"]

_SLOT_KIND: dict[str, DigestKind] = {
    "raw": DigestKind.RAW_ARTIFACT,
    "canonical_text": DigestKind.CANONICAL_TEXT,
    "tree": DigestKind.TREE,
}


class VerificationDepth(StrEnum):
    """How thoroughly one recorded identity was checked on this run."""

    DEEP = "deep"
    """Re-derived from the bytes. The recorded identity either holds or it does not."""

    SHAPE = "shape"
    """Presence, file count and total size only. Says nothing about a flipped byte."""

    ABSENT = "absent"
    """The artifact is not on this machine. Reported, never counted as verified."""


class ArtifactShape(BaseModel):
    """What a tree artifact contains, for the cheap tier of verification.

    Recorded alongside a tree digest so a shape check has something to compare
    against. Matching shape is not identity, and the report never says it is.
    """

    model_config = ConfigDict(extra="forbid")

    file_count: Annotated[int, Field(ge=0)]
    empty_directory_count: Annotated[int, Field(ge=0)]
    total_bytes: Annotated[int, Field(ge=0)]


class ArtifactProvenance(BaseModel):
    """Where an artifact came from, and whether anyone has checked the terms.

    Absent provenance is a hard failure. So is a clearance with no evidence: see
    the validator below. Licence, access restrictions and eligibility are three
    separate questions and are recorded in three separate fields, because a
    permissive licence on the bytes says nothing about who is allowed to hold
    them or about whether using them is within a competition's rules.

    None of this is established by hashing. Byte identity answers what the data
    is; clearance answers whether it may be used, and the two are recorded
    independently so that neither can be mistaken for the other.
    """

    model_config = ConfigDict(extra="forbid")

    status: ProvenanceStatus
    imported_utc: datetime | None = None
    original_path: str | None = None
    source_url: str | None = None
    source_commit: str | None = None
    source_digest: DigestToken | None = None
    code_license: str | None = None
    weight_license: str | None = None
    data_license: str | None = None
    """Terms on the data itself. For a dataset this is the licence that matters;
    neither a code licence nor a weight licence is a substitute for it."""

    access_restrictions: str | None = None
    """What the holder may not do, independently of the licence.

    Competition data can be permissively licensed and still carry a rule against
    redistributing it to non-participants. A licence field alone would record
    the first and silently lose the second.
    """

    training_data: str | None = None
    competition_eligible: bool | None = None
    reviewed_by: str | None = None
    reviewed_utc: datetime | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _clearance_needs_evidence(self) -> ArtifactProvenance:
        """A cleared status must name what was checked.

        Without this the status is decorative: anything could be marked cleared
        with no licence, no restrictions and no eligibility decision recorded,
        and a later reader would have no way to tell a real review from a
        forgotten default.
        """
        if self.status is not ProvenanceStatus.EXTERNAL_CLEARED:
            return self
        missing: list[str] = []
        if not self.source_url:
            missing.append("source_url")
        if not (self.code_license or self.weight_license or self.data_license):
            missing.append("a licence (code_license, weight_license or data_license)")
        if not self.access_restrictions:
            missing.append("access_restrictions")
        if self.competition_eligible is None:
            missing.append("competition_eligible")
        if not self.reviewed_by:
            missing.append("reviewed_by")
        if missing:
            raise ValueError(
                "external_cleared asserts that someone checked the terms, so it requires "
                f"the evidence of that check. Missing: {missing}. Use external_uncleared "
                "until the review has actually happened"
            )
        return self


class ArtifactRecord(BaseModel):
    """One artifact and the identities Biohub-X asserts about it."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    path: str | None = Field(default=None, min_length=1)
    """Repository-relative location. Set for artifacts the repository contains."""

    external_path: str | None = Field(default=None, min_length=1)
    """Machine-local location for an artifact the repository does not contain.

    Evidence, not identity. The digest is what travels between machines; this
    only says where the bytes happened to be when they were read.
    """

    schema_: str = Field(alias="schema", min_length=1)
    digests: dict[DigestSlot, DigestToken] = Field(min_length=1)
    shape: ArtifactShape | None = None
    produced_by: str | None = None
    provenance: ArtifactProvenance

    @model_validator(mode="after")
    def _one_location_and_a_matching_identity(self) -> ArtifactRecord:
        if (self.path is None) == (self.external_path is None):
            raise ValueError(
                f"artifact {self.id!r} must record exactly one location: `path` for something the "
                "repository contains, or `external_path` for something it does not"
            )
        if "tree" in self.digests:
            if len(self.digests) != 1:
                raise ValueError(
                    f"artifact {self.id!r} mixes a tree digest with file digests; a directory has "
                    "no single-file identity and a file has no tree identity"
                )
            if self.shape is None:
                raise ValueError(
                    f"artifact {self.id!r} records a tree digest with no shape, so it could never "
                    "be checked without reading every byte"
                )
        elif self.shape is not None:
            raise ValueError(f"artifact {self.id!r} records a shape but is not a tree artifact")
        return self

    @field_validator("path")
    @classmethod
    def _refuse_absolute(cls, value: str | None) -> str | None:
        if value is None:
            return None
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

    @property
    def is_tree(self) -> bool:
        return "tree" in self.digests

    def locate(self, root: Path) -> Path:
        """Where to look for this artifact on this machine."""
        if self.path is not None:
            return root / self.path
        assert self.external_path is not None
        return Path(self.external_path)


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
    depth: VerificationDepth
    detail: str | None = None


def load_artifact_registry(path: Path) -> ArtifactRegistry:
    """Load and validate the artifact registry. A missing registry is a failure."""
    if not path.is_file():
        raise FileNotFoundError(f"artifact registry not found: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if document is None:
        raise ValueError(f"artifact registry is empty: {path}")
    return ArtifactRegistry.model_validate(document)


def _check(
    record: ArtifactRecord,
    slot: str,
    token: str,
    *,
    observed: str | None,
    ok: bool,
    depth: VerificationDepth,
    detail: str | None = None,
) -> DigestCheck:
    return DigestCheck(
        artifact_id=record.id,
        slot=slot,
        expected=token,
        observed=observed,
        ok=ok,
        depth=depth,
        detail=detail,
    )


def _verify_tree(record: ArtifactRecord, target: Path, token: str, *, deep: bool) -> DigestCheck:
    """Check a directory artifact, deeply or by shape.

    A shape check compares file count, empty-directory count and total size. It
    is minutes cheaper and it is NOT identity: it cannot see a flipped byte. The
    depth travels with the result so a caller can never read one as the other.
    """
    if not target.is_dir():
        # An external artifact that is not on this machine is reported, not
        # failed: the repository must contain what it claims, but a machine need
        # not hold every dataset. An in-repository tree that is missing IS a
        # failure, because the repository did claim it.
        if record.external_path is not None:
            return _check(
                record,
                "tree",
                token,
                observed=None,
                ok=True,
                depth=VerificationDepth.ABSENT,
                detail=f"not on this machine: {target}",
            )
        return _check(
            record,
            "tree",
            token,
            observed=None,
            ok=False,
            depth=VerificationDepth.ABSENT,
            detail=f"directory not found: {target}",
        )

    assert record.shape is not None  # enforced by ArtifactRecord
    try:
        if deep:
            observed = tree_digest(target).digest.token
            return _check(
                record,
                "tree",
                token,
                observed=observed,
                ok=observed == token,
                depth=VerificationDepth.DEEP,
                detail=None if observed == token else "tree digest mismatch",
            )
        shape = tree_shape(target)
    except (NotATreeError, TreeNotQuiescentError, OSError) as exc:
        return _check(
            record,
            "tree",
            token,
            observed=None,
            ok=False,
            depth=VerificationDepth.SHAPE,
            detail=str(exc),
        )

    differences = [
        f"{name}: recorded {expected}, found {found}"
        for name, expected, found in (
            ("file_count", record.shape.file_count, shape.file_count),
            (
                "empty_directory_count",
                record.shape.empty_directory_count,
                shape.empty_directory_count,
            ),
            ("total_bytes", record.shape.total_bytes, shape.total_bytes),
        )
        if expected != found
    ]
    summary = f"files={shape.file_count} bytes={shape.total_bytes}"
    return _check(
        record,
        "tree",
        token,
        observed=summary,
        ok=not differences,
        depth=VerificationDepth.SHAPE,
        detail="; ".join(differences) if differences else "shape only; not deeply verified",
    )


def verify_registry(registry: ArtifactRegistry, root: Path, *, deep: bool = False) -> list[DigestCheck]:
    """Check every recorded identity, at the depth each artifact warrants.

    File artifacts inside the repository are always re-derived from their bytes.
    Tree artifacts are re-derived only when ``deep`` is set, and are otherwise
    checked for presence and shape, with the depth recorded on every result so
    that a shape check is never read as an identity check.

    A missing file, a mismatched digest, a digest filed under the wrong slot or
    a changed shape is a failure. Nothing is skipped, and a check that was not
    made says which one it was.
    """
    checks: list[DigestCheck] = []
    for record in registry.artifacts:
        target = record.locate(root)
        for slot, token in record.digests.items():
            expected = Digest.parse(token)
            kind = _SLOT_KIND[slot]
            if expected.kind is not kind:
                checks.append(
                    _check(
                        record,
                        slot,
                        token,
                        observed=None,
                        ok=False,
                        depth=VerificationDepth.DEEP,
                        detail=f"slot {slot} holds a {expected.kind.value} digest",
                    )
                )
                continue

            if kind is DigestKind.TREE:
                checks.append(_verify_tree(record, target, token, deep=deep))
                continue

            if not target.is_file():
                checks.append(
                    _check(
                        record,
                        slot,
                        token,
                        observed=None,
                        ok=False,
                        depth=VerificationDepth.ABSENT,
                        detail=f"file not found: {target}",
                    )
                )
                continue
            try:
                observed_digest = digest_file(target, kind)
            except (NotTextError, NotATreeError) as exc:
                checks.append(
                    _check(
                        record,
                        slot,
                        token,
                        observed=None,
                        ok=False,
                        depth=VerificationDepth.DEEP,
                        detail=str(exc),
                    )
                )
                continue
            matched = observed_digest.token == token
            checks.append(
                _check(
                    record,
                    slot,
                    token,
                    observed=observed_digest.token,
                    ok=matched,
                    depth=VerificationDepth.DEEP,
                    detail=None if matched else "digest mismatch",
                )
            )
    return checks


def summarise_checks(checks: list[DigestCheck]) -> dict[str, int]:
    """Count results by depth, so a report states what was actually established."""
    return {
        "checked": len(checks),
        "failed": sum(1 for check in checks if not check.ok),
        "deep": sum(1 for check in checks if check.depth is VerificationDepth.DEEP),
        "shape_only": sum(1 for check in checks if check.depth is VerificationDepth.SHAPE),
        "absent": sum(1 for check in checks if check.depth is VerificationDepth.ABSENT),
    }


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
