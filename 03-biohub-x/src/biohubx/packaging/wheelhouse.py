"""Two identities for one wheelhouse, because Kaggle stores less than it is sent.

A wheelhouse is uploaded as a directory and published as a dataset, and those are
not the same tree. ``dataset-metadata.json`` is configuration: the Kaggle CLI
reads it to learn what to create and does not keep it among the data, so the
published dataset holds one file fewer than the bundle that produced it. One
recorded digest would therefore name a tree that exists on neither side, and the
difference would surface later as an inventory mismatch with no way to tell a
benign platform behaviour from a substituted file.

The distinction is not bookkeeping. The upload bundle is what this machine sends
and can re-derive from its own staging directory. The published payload is what a
kernel mounts, so it is the identity an authorisation has to name and the only one
a running preflight could ever check.

Consumer: ``biohubx package wheelhouse``.
"""

from __future__ import annotations

import re
import shutil
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from biohubx.hashing import TreeManifest

KAGGLE_CONFIGURATION_FILENAME = "dataset-metadata.json"
"""Read by the Kaggle CLI to create the dataset, and never stored as data.

It sits at the root of the upload bundle by the CLI's requirement. Nested files of
the same name would be ordinary data, which is why the exclusion is anchored to
the root rather than matched by name anywhere in the tree.
"""


class WheelhouseError(Exception):
    """The wheelhouse is not something a published identity may be asserted about."""


@dataclass(frozen=True, slots=True)
class LockedWheel:
    """One wheel as ``uv.lock`` records it, which is the root authority for it."""

    filename: str
    package: str
    version: str
    sha256: str
    size_bytes: int | None
    """Absent for some entries. uv.lock does not promise a size, and the hash is
    the authority anyway, so a missing size is recorded as unknown rather than
    defaulted to a number nothing measured."""


@dataclass(frozen=True, slots=True)
class TreeDifference:
    """One way two trees disagree, named at the path where they do.

    A count of differences says something changed. A path says which file, which
    is the difference between an alarm someone can act on and one they cannot.
    """

    relative_path: str
    reason: str
    expected: str
    observed: str


def canonical_package_name(name: str) -> str:
    """PEP 503 normalisation, because ``google_crc32c`` and ``google-crc32c`` are one package.

    A wheel filename escapes the name with underscores while a lock file and a
    requirements file spell it with hyphens. Comparing the two spellings literally
    would report a missing requirement for a package that is present.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_wheels(lock_text: str, filenames: Iterable[str]) -> dict[str, LockedWheel]:
    """Look every staged wheel up in ``uv.lock`` by its exact filename.

    Filename rather than package name, because a package resolves to many wheels
    across platforms and only one of them is the file on disk. Matching by package
    would accept a macOS wheel as evidence for a manylinux one.
    """
    lock = tomllib.loads(lock_text)
    index: dict[str, LockedWheel] = {}
    for package in lock.get("package", []):
        for wheel in package.get("wheels", []):
            filename = str(wheel["url"]).rsplit("/", 1)[-1]
            size = wheel.get("size")
            index[filename] = LockedWheel(
                filename=filename,
                package=str(package["name"]),
                version=str(package["version"]),
                sha256=str(wheel["hash"]).removeprefix("sha256:"),
                size_bytes=None if size is None else int(size),
            )

    resolved: dict[str, LockedWheel] = {}
    for filename in filenames:
        if filename not in index:
            raise WheelhouseError(
                f"{filename} appears in no uv.lock entry, so no locked hash covers it. "
                "A wheel the lock does not name is a wheel nobody chose"
            )
        resolved[filename] = index[filename]
    return resolved


_REQUIREMENT_LINE = re.compile(
    r"^(?P<package>[A-Za-z0-9._-]+)==(?P<version>[A-Za-z0-9._+!-]+) --hash=sha256:(?P<sha256>[0-9a-f]{64})$"
)


def requirement_hashes(requirements_text: str) -> dict[str, tuple[str, str]]:
    """``{canonical name: (version, sha256)}`` exactly as pip will enforce it.

    ``--require-hashes`` is the only thing refusing a substituted wheel on a
    machine with no index to consult, so the file is parsed rather than trusted. A
    hash that is present but unreadable protects nothing, and a line this parser
    cannot read is refused rather than skipped: a silently skipped requirement is
    an unpinned install that still looks pinned.
    """
    joined = requirements_text.replace("\\\n", " ")
    hashes: dict[str, tuple[str, str]] = {}
    for raw in joined.splitlines():
        line = " ".join(raw.split())
        if not line or line.startswith("#"):
            continue
        match = _REQUIREMENT_LINE.match(line)
        if match is None:
            raise WheelhouseError(
                f"cannot read this requirement line, so what pip would enforce is unknown: {line!r}"
            )
        hashes[canonical_package_name(match["package"])] = (match["version"], match["sha256"])
    return hashes


def published_relative_paths(bundle: TreeManifest) -> tuple[str, ...]:
    """The relative paths Kaggle stores, taken from the bundle's own verified listing.

    Read out of the manifest rather than from a second walk, so both identities
    describe one enumeration of one tree and cannot drift between two of them.
    """
    return tuple(
        record.relative_path
        for record in bundle.records
        if record.kind == "f" and record.relative_path != KAGGLE_CONFIGURATION_FILENAME
    )


def stage_published_payload(root: Path, destination: Path, relative_paths: Iterable[str]) -> int:
    """Copy the published payload into ``destination``, preserving relative paths.

    The payload is digested as a tree in its own right rather than by deleting a
    line from the bundle's canonical listing. Subtracting a line is wrong in
    general: removing the last file from a directory turns that directory into an
    empty one, which tree canonicalization ``v1`` records in its own right, so an
    edited listing can describe a tree that could not exist. Walking a real tree
    has no such case, and it reuses the contract-tested walk instead of a second
    implementation of it.
    """
    copied = 0
    for relative in relative_paths:
        source = root.joinpath(*relative.split("/"))
        target = destination.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied += 1
    return copied


def compare_manifests(expected: TreeManifest, observed: TreeManifest) -> tuple[TreeDifference, ...]:
    """Every way two trees disagree: names, sizes and contents, in one pass.

    Content is compared as well as size because a same-size substitution changes
    neither the file count nor the total bytes, which is exactly the case a shape
    check cannot see.
    """
    expected_files = {record.relative_path: record for record in expected.records if record.kind == "f"}
    observed_files = {record.relative_path: record for record in observed.records if record.kind == "f"}
    differences: list[TreeDifference] = []

    for relative in sorted(expected_files.keys() - observed_files.keys()):
        differences.append(
            TreeDifference(relative_path=relative, reason="missing", expected="present", observed="absent")
        )
    for relative in sorted(observed_files.keys() - expected_files.keys()):
        differences.append(
            TreeDifference(relative_path=relative, reason="unexpected", expected="absent", observed="present")
        )
    for relative in sorted(expected_files.keys() & observed_files.keys()):
        want, got = expected_files[relative], observed_files[relative]
        if want.size_bytes != got.size_bytes:
            differences.append(
                TreeDifference(
                    relative_path=relative,
                    reason="size",
                    expected=str(want.size_bytes),
                    observed=str(got.size_bytes),
                )
            )
        if want.content_sha256 != got.content_sha256:
            differences.append(
                TreeDifference(
                    relative_path=relative,
                    reason="content",
                    expected=str(want.content_sha256),
                    observed=str(got.content_sha256),
                )
            )
    return tuple(differences)
