"""Content identity for Biohub-X.

Biohub-X maintains two identities for every piece of content and never treats
them as interchangeable.

``RAW_ARTIFACT_SHA256``
    SHA-256 of the exact bytes on disk. This is the identity of weights,
    exported graphs, built notebooks, submissions, and of any hash-bound
    snapshot whose point is to assert "these are the original bytes".

``CANONICAL_TEXT_SHA256``
    SHA-256 of the bytes produced by :func:`canonicalize_text`. This is the
    identity used to detect source and configuration drift. It is stable across
    platform line-ending conventions, so it answers "did the content change?"
    rather than "did the file change?".

Every digest produced here carries its kind and, for canonical text, the
canonicalization version. A bare hexadecimal string is not a valid identity
anywhere in this repository: it does not say what it is the digest *of*.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

ALGORITHM = "sha256"

CANONICALIZATION_VERSION = "v1"
"""Canonical-text rules, version 1.

1. Decode as strict UTF-8. Bytes that are not valid UTF-8 have no canonical
   text identity; use the raw identity instead.
2. Remove a leading UTF-8 byte-order mark.
3. Replace CRLF and lone CR with LF.
4. If the result is non-empty and does not end with LF, append one LF.
5. Re-encode as UTF-8.

Trailing whitespace, blank lines and indentation are deliberately preserved.
Canonicalization removes platform convention, not content. Bumping this
version invalidates every recorded canonical digest and requires a
DECISIONS.md entry.
"""

_BOM = "﻿"
_READ_CHUNK = 1 << 20

_TOKEN_RE = re.compile(
    r"^(?P<kind>[a-z0-9_]+):(?P<algorithm>[a-z0-9]+)"
    r"(?:/(?P<canonicalization>v\d+))?:(?P<hexdigest>[0-9a-f]{64})$"
)


class NotTextError(ValueError):
    """Canonical-text identity was requested for bytes that are not UTF-8."""


class DigestKind(StrEnum):
    """The two identities. There is no third, and no default."""

    RAW_ARTIFACT = "raw_artifact_sha256"
    CANONICAL_TEXT = "canonical_text_sha256"


@dataclass(frozen=True, slots=True)
class Digest:
    """A digest that states what it is a digest of.

    Serialised form (``token``)::

        raw_artifact_sha256:sha256:<64 hex>
        canonical_text_sha256:sha256/v1:<64 hex>

    The canonicalization segment is present exactly when the kind is
    ``CANONICAL_TEXT``, so a raw digest can never be silently read as a
    canonical one or the reverse.
    """

    kind: DigestKind
    hexdigest: str
    algorithm: str = ALGORITHM
    canonicalization: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.hexdigest):
            raise ValueError(f"not a lowercase sha256 hexdigest: {self.hexdigest!r}")
        if self.algorithm != ALGORITHM:
            raise ValueError(f"unsupported algorithm: {self.algorithm!r}")
        if self.kind is DigestKind.RAW_ARTIFACT and self.canonicalization is not None:
            raise ValueError("a raw artifact digest must not declare a canonicalization")
        if self.kind is DigestKind.CANONICAL_TEXT and self.canonicalization is None:
            raise ValueError("a canonical text digest must declare a canonicalization")

    @property
    def token(self) -> str:
        """The only form in which a digest may be written to a registry."""
        algorithm = self.algorithm
        if self.canonicalization is not None:
            algorithm = f"{algorithm}/{self.canonicalization}"
        return f"{self.kind.value}:{algorithm}:{self.hexdigest}"

    def __str__(self) -> str:
        return self.token

    @classmethod
    def parse(cls, token: str) -> Digest:
        """Parse a digest token. Untyped or malformed input is refused."""
        match = _TOKEN_RE.match(token)
        if match is None:
            raise ValueError(f"not a digest token (expected '<kind>:<algorithm>[/<canon>]:<hex>'): {token!r}")
        try:
            kind = DigestKind(match["kind"])
        except ValueError as exc:
            raise ValueError(f"unknown digest kind in token: {token!r}") from exc
        return cls(
            kind=kind,
            hexdigest=match["hexdigest"],
            algorithm=match["algorithm"],
            canonicalization=match["canonicalization"],
        )


def canonicalize_text(data: bytes) -> bytes:
    """Apply canonical-text rules ``v1``. See :data:`CANONICALIZATION_VERSION`."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NotTextError("content is not valid UTF-8 and has no canonical text identity") from exc
    if text.startswith(_BOM):
        text = text[len(_BOM) :]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if text and not text.endswith("\n"):
        text += "\n"
    return text.encode("utf-8")


def raw_digest(data: bytes) -> Digest:
    """Identity of the exact bytes."""
    return Digest(kind=DigestKind.RAW_ARTIFACT, hexdigest=hashlib.sha256(data).hexdigest())


def canonical_text_digest(data: bytes) -> Digest:
    """Identity of the content, independent of platform line endings."""
    canonical = canonicalize_text(data)
    return Digest(
        kind=DigestKind.CANONICAL_TEXT,
        hexdigest=hashlib.sha256(canonical).hexdigest(),
        canonicalization=CANONICALIZATION_VERSION,
    )


def raw_digest_file(path: Path) -> Digest:
    """Streamed raw identity, so large weights never need to be held in memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_CHUNK):
            hasher.update(chunk)
    return Digest(kind=DigestKind.RAW_ARTIFACT, hexdigest=hasher.hexdigest())


def canonical_text_digest_file(path: Path) -> Digest:
    """Canonical text identity of a file. Refuses non-UTF-8 content."""
    return canonical_text_digest(path.read_bytes())


def digest_file(path: Path, kind: DigestKind) -> Digest:
    """Digest a file under an explicitly named identity. There is no default kind."""
    if kind is DigestKind.RAW_ARTIFACT:
        return raw_digest_file(path)
    return canonical_text_digest_file(path)
