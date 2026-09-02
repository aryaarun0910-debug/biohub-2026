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

``TREE_SHA256``
    SHA-256 over a canonical listing of a whole directory tree: every file's
    path, size and content digest, plus every empty directory. This is the
    identity of a dataset artifact, because a Zarr-backed volume or graph is a
    directory of many thousands of chunk files and has no single-file digest.
    A layout check answers "is this the right shape"; only this answers "is this
    the same data".

Every digest produced here carries its kind and, for canonical text, the
canonicalization version. A bare hexadecimal string is not a valid identity
anywhere in this repository: it does not say what it is the digest *of*.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
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

TREE_CANONICALIZATION_VERSION = "v1"
"""Canonical-tree rules, version 1.

The listing is built from the tree root, and every path is recorded relative to
it, so moving or renaming the root does not change the identity while moving a
file inside it does.

1. Walk the tree. Only regular files and directories are permitted; a symlink,
   device node or anything else raises, because following one could leave the
   tree and not following one would silently lose information.
2. Record each relative path with forward slashes, encoded as UTF-8.
3. A regular file contributes ``f <sha256> <size> <path>``, where the digest is
   the raw identity of its bytes.
4. An EMPTY directory contributes ``d - - <path>``. A non-empty directory is
   already implied by the files inside it, but an empty one is structure that
   would otherwise vanish from the listing entirely.
5. Sort the records by their UTF-8 encoded path.
6. Join with LF, append one trailing LF, encode UTF-8, and take the SHA-256.

A path containing a line feed or a null byte is refused rather than escaped: the
listing is a line format, and such names are pathological rather than expected.

Bumping this version invalidates every recorded tree digest and requires a
DECISIONS.md entry.
"""

_BOM = "\ufeff"
_READ_CHUNK = 1 << 20

_TOKEN_RE = re.compile(
    r"^(?P<kind>[a-z0-9_]+):(?P<algorithm>[a-z0-9]+)"
    r"(?:/(?P<canonicalization>v\d+))?:(?P<hexdigest>[0-9a-f]{64})$"
)


class NotTextError(ValueError):
    """Canonical-text identity was requested for bytes that are not UTF-8."""


class NotATreeError(ValueError):
    """A tree digest was requested for something that is not a walkable directory."""


class DigestKind(StrEnum):
    """The three identities. There is no fourth, and no default."""

    RAW_ARTIFACT = "raw_artifact_sha256"
    """The exact bytes of one file."""

    CANONICAL_TEXT = "canonical_text_sha256"
    """The content of one text file, independent of platform line endings."""

    TREE = "tree_sha256"
    """The structure and content of a whole directory tree."""


@dataclass(frozen=True, slots=True)
class Digest:
    """A digest that states what it is a digest of.

    Serialised form (``token``)::

        raw_artifact_sha256:sha256:<64 hex>
        canonical_text_sha256:sha256/v1:<64 hex>
        tree_sha256:sha256/v1:<64 hex>

    A raw digest never carries a canonicalization segment and the other two
    always do, so no kind can be silently read as another. The canonicalization
    versions of text and tree are independent and are not interchangeable, which
    is why the kind is part of the token rather than implied by the version.
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
        if self.kind is not DigestKind.RAW_ARTIFACT and self.canonicalization is None:
            raise ValueError(f"a {self.kind.value} digest must declare a canonicalization")

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
    if kind is DigestKind.TREE:
        raise NotATreeError("a tree identity belongs to a directory, not a file; use tree_digest instead")
    if kind is DigestKind.RAW_ARTIFACT:
        return raw_digest_file(path)
    return canonical_text_digest_file(path)


# --- tree identity ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TreeRecord:
    """One line of the canonical listing, kept so a mismatch can be localised."""

    kind: str
    relative_path: str
    size_bytes: int | None
    content_sha256: str | None

    @property
    def line(self) -> str:
        size = "-" if self.size_bytes is None else str(self.size_bytes)
        content = "-" if self.content_sha256 is None else self.content_sha256
        return f"{self.kind} {content} {size} {self.relative_path}"


@dataclass(frozen=True, slots=True)
class TreeManifest:
    """A tree digest together with the listing it was computed from.

    Keeping the listing is what turns a mismatch from "something changed" into
    "this file changed", which is the difference between a usable alarm and an
    unusable one.
    """

    root_name: str
    records: tuple[TreeRecord, ...]
    file_count: int
    empty_directory_count: int
    total_bytes: int
    digest: Digest

    def canonical_listing(self) -> str:
        return "".join(f"{record.line}\n" for record in self.records)


def _relative_posix(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    if "\n" in relative or "\x00" in relative:
        raise NotATreeError(f"path contains a line feed or null byte and cannot be listed: {relative!r}")
    return relative


def tree_digest(
    root: Path,
    *,
    on_file: Callable[[int, int], None] | None = None,
) -> TreeManifest:
    """Identity of a whole directory tree under canonicalization ``v1``.

    Reads every byte of every file exactly once. On a large dataset that is the
    dominant cost of registration, and it is the cost of being able to say that
    the data an experiment used is the data that was registered.

    ``on_file`` receives ``(files_done, bytes_done)`` after each file, so a
    caller can emit a heartbeat rather than appearing to hang.
    """
    if not root.exists():
        raise NotATreeError(f"tree root does not exist: {root}")
    if root.is_symlink() or not root.is_dir():
        raise NotATreeError(f"tree root is not a directory: {root}")

    records: list[TreeRecord] = []
    files = 0
    empty_directories = 0
    total_bytes = 0

    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise NotATreeError(
                f"symlink inside the tree: {_relative_posix(root, path)}. Following it could leave "
                "the tree and skipping it would lose information, so neither is done silently"
            )
        if path.is_dir():
            if not any(path.iterdir()):
                records.append(
                    TreeRecord(
                        kind="d",
                        relative_path=_relative_posix(root, path),
                        size_bytes=None,
                        content_sha256=None,
                    )
                )
                empty_directories += 1
            continue
        if not path.is_file():
            raise NotATreeError(f"neither a regular file nor a directory: {_relative_posix(root, path)}")
        size = path.stat().st_size
        records.append(
            TreeRecord(
                kind="f",
                relative_path=_relative_posix(root, path),
                size_bytes=size,
                content_sha256=raw_digest_file(path).hexdigest,
            )
        )
        files += 1
        total_bytes += size
        if on_file is not None:
            on_file(files, total_bytes)

    records.sort(key=lambda record: record.relative_path.encode("utf-8"))
    listing = "".join(f"{record.line}\n" for record in records)
    return TreeManifest(
        root_name=root.name,
        records=tuple(records),
        file_count=files,
        empty_directory_count=empty_directories,
        total_bytes=total_bytes,
        digest=Digest(
            kind=DigestKind.TREE,
            hexdigest=hashlib.sha256(listing.encode("utf-8")).hexdigest(),
            canonicalization=TREE_CANONICALIZATION_VERSION,
        ),
    )
