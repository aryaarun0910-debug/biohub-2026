"""Build the Kaggle package for a fold, and the guards it carries with it.

A GPU run is the one place where Biohub-X's checks stop applying, because the
code runs somewhere this repository cannot watch. So the guards travel with it.
The package pins the commit and the configuration digest it was built from, and
its entry point refuses to train until it has re-verified, on the machine that
will do the training, that the mounted inputs are the ones that were registered,
that the fold it was asked for excludes its own evaluation embryo, and that no
quarantined checkpoint or public-test path is reachable.

The package carries no competition bytes and no external weights. It carries
code and constants; everything else is mounted by Kaggle and checked on arrival.

Notebooks are generated here rather than written by hand, per AGENTS.md section
3: the notebook contains no model logic, only a call into the tested package.

Consumer: ``biohubx package kaggle``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARCHIVE_EPOCH = (1980, 1, 1, 0, 0, 0)
"""Fixed timestamp for every archive entry, so the payload digest is reproducible."""

ARCHIVE_SUFFIXES = frozenset({".py"})
"""Only source enters the payload. Nothing else has a reason to be there."""

PACKAGE_ROOT_ENV = "BIOHUBX_PACKAGE_ROOT"
DEFAULT_PACKAGE_ROOT = "/kaggle/working/biohubx-package"
"""Kaggle documents /kaggle/working as the writable runtime directory.

The payload extracts there and is imported from there. Nothing is imported from
/kaggle/input: no dataset or model source supplies this package, so no such path
is promised to exist, and depending on one is how a run dies before it starts.
"""

PUBLIC_TEST_MARKERS = ("/test/", "/test")
"""Path fragments meaning the run is reading the public-test split.

Compared against forward-slash-normalised paths so a Windows separator cannot
slip one past the check.
"""


def _normalise(path: str) -> str:
    return path.replace("\\", "/")


QUARANTINED_ARTIFACT_PREFIX = "reference.pilkwang"
"""Artifact ids the training run may never load."""


class PackagingError(ValueError):
    """The package cannot be built or does not satisfy its own guards."""


@dataclass(frozen=True, slots=True)
class FoldSpec:
    """One leave-one-embryo-out fold, named so a run cannot silently swap them."""

    fold_id: str
    train_embryo: str
    evaluate_embryo: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "train_embryo": self.train_embryo,
            "evaluate_embryo": self.evaluate_embryo,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class PackageSpec:
    """Everything the package asserts about itself before it is allowed to train."""

    commit: str
    config_path: str
    config_digest: str
    fold: FoldSpec
    epochs: int
    batch_size: int
    learning_rate: float
    accelerator: str
    expected_gpu_count: int
    smoke: bool
    max_movies: int | None
    runtime_ceiling_seconds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "config_path": self.config_path,
            "config_digest": self.config_digest,
            "fold": self.fold.to_dict(),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "accelerator": self.accelerator,
            "expected_gpu_count": self.expected_gpu_count,
            "smoke": self.smoke,
            "max_movies": self.max_movies,
            "runtime_ceiling_seconds": self.runtime_ceiling_seconds,
        }


def slug_of(title: str) -> str:
    """How Kaggle derives a kernel slug from a title.

    Lowercased, non-alphanumerics collapsed to single hyphens, trimmed. If the
    title does not slugify to the requested id, Kaggle warns and uses the title's
    slug instead, so the kernel that gets created is not the one that was named.
    """
    import re

    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", title.lower())).strip("-")


def kernel_id_for(owner: str, title: str) -> str:
    """``owner/slug`` with the slug derived from the title, so they cannot disagree.

    Kaggle slugifies the title and prefers the result over any id it was given,
    with a warning. E03-SMOKE was pushed to an address nobody named because that
    warning was treated as noise. Deriving the slug removes the possibility.
    """
    if not owner or "/" in owner:
        raise PackagingError(f"owner must be a bare Kaggle account slug, got {owner!r}")
    return f"{owner}/{slug_of(title)}"


def kernel_metadata(spec: PackageSpec, *, slug: str, title: str) -> dict[str, Any]:
    """The Kaggle kernel manifest, with internet off and no dataset sources.

    ``enable_internet`` is false because a run that can reach the network can
    also acquire a checkpoint nobody approved. ``dataset_sources`` is empty
    because E03 uses no external weights at all, and an empty list is a claim a
    reviewer can check at a glance.
    """
    owner, _, name = slug.rpartition("/")
    if not owner:
        raise PackagingError(f"a Kaggle kernel id is owner/slug; got {slug!r}")
    if slug_of(title) != name:
        raise PackagingError(
            f"title {title!r} slugifies to {slug_of(title)!r}, not {name!r}; Kaggle would create a "
            "kernel at a different address than the one named"
        )
    return {
        "id": slug,
        "title": title,
        "code_file": "run.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "accelerator": spec.accelerator,
        "dataset_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "competition_sources": ["biohub-cell-tracking-during-development"],
    }


def guard_report(
    spec: PackageSpec,
    *,
    mounted_digests: dict[str, str],
    registered_digests: dict[str, str],
    dataset_ids: list[str],
    reachable_paths: list[str],
    opened_paths: list[str],
    gpu_count: int,
) -> dict[str, Any]:
    """Run every guard and report each one, refusing on the first real failure.

    Returns the report so a passing run records what it checked, not only that
    it started. A guard that passes silently is a guard nobody can audit later.
    """
    failures: list[str] = []

    # An identity check over an empty set verifies nothing and would pass
    # silently, which is worse than not having one.
    if not registered_digests:
        failures.append(
            "the package shipped no input digests, so mounted identity cannot be verified; "
            "a run that cannot check what it trained on is not the run that was approved"
        )

    missing = sorted(set(registered_digests) - set(mounted_digests))
    mismatched = sorted(
        name
        for name, digest in registered_digests.items()
        if name in mounted_digests and mounted_digests[name] != digest
    )
    if missing:
        failures.append(f"inputs registered but not mounted: {missing[:5]}")
    if mismatched:
        failures.append(f"mounted inputs whose identity differs from the registry: {mismatched[:5]}")

    wrong_fold = sorted(d for d in dataset_ids if d.split("_", 1)[0] == spec.fold.evaluate_embryo)
    if wrong_fold:
        failures.append(
            f"fold {spec.fold.fold_id} would train on {len(wrong_fold)} movies from its own "
            f"evaluation embryo {spec.fold.evaluate_embryo}: {wrong_fold[:5]}"
        )
    if not any(d.split("_", 1)[0] == spec.fold.train_embryo for d in dataset_ids):
        failures.append(f"no movie from the training embryo {spec.fold.train_embryo} was mounted")

    # The public-test directory is mounted by Kaggle whether or not this run
    # wants it, so its existence is not the failure. Opening it would be. The
    # check is therefore on the paths the run will actually read.
    public_test = sorted(
        path for path in opened_paths if any(marker in _normalise(path) for marker in PUBLIC_TEST_MARKERS)
    )
    if public_test:
        failures.append(f"the run would read public-test paths: {public_test[:3]}")

    quarantined = sorted(
        path
        for path in list(opened_paths) + list(reachable_paths)
        if QUARANTINED_ARTIFACT_PREFIX in path or _normalise(path).endswith(".pth")
    )
    if quarantined:
        failures.append(f"a quarantined checkpoint is reachable to this run: {quarantined[:3]}")

    if gpu_count != spec.expected_gpu_count:
        failures.append(
            f"requested {spec.expected_gpu_count} GPU(s) and found {gpu_count}; "
            "the run would not cost what was approved"
        )

    return {
        "inputs_verified": len(registered_digests),
        "inputs_missing": missing,
        "inputs_mismatched": mismatched,
        "fold_membership_asserted": not wrong_fold,
        "training_movies": len(dataset_ids),
        "public_test_paths_opened": public_test,
        "paths_the_run_opens": len(opened_paths),
        "quarantined_checkpoints_reachable": quarantined,
        "gpu_count": gpu_count,
        "expected_gpu_count": spec.expected_gpu_count,
        "passed": not failures,
        "failures": failures,
    }


NOTEBOOK_CELL = """\
# Generated by `biohubx package kaggle`. Do not edit here.
#
# Self-contained on purpose. The tested package travels inside this notebook as a
# deterministic archive, is verified against a digest recorded at build time, and
# is extracted into the writable runtime directory before anything imports it.
#
# Nothing is imported from /kaggle/input. No dataset or model source supplies
# this package, so no such path is promised to exist, and an earlier run died
# before executing a single line because it assumed one did.
import base64
import hashlib
import io
import json
import os
import pathlib
import sys
import zipfile


def stage(name, detail=""):
    print(("BIOHUBX_STAGE " + name + " " + str(detail)).rstrip(), flush=True)


PAYLOAD_SHA256 = "__PAYLOAD_SHA256__"
PAYLOAD_B64 = \"\"\"__PAYLOAD_B64__\"\"\"
SPEC = json.loads(r\"\"\"__SPEC_JSON__\"\"\")

stage("bootstrap-verify", "expecting " + PAYLOAD_SHA256[:23])
payload = base64.b64decode("".join(PAYLOAD_B64.split()))
actual = hashlib.sha256(payload).hexdigest()
if actual != PAYLOAD_SHA256:
    raise SystemExit(
        "embedded package digest mismatch: expected " + PAYLOAD_SHA256 + " got " + actual
    )
stage("bootstrap-verify", "ok bytes=" + str(len(payload)))

target = pathlib.Path(os.environ.get("BIOHUBX_PACKAGE_ROOT", "__PACKAGE_ROOT__"))
target.mkdir(parents=True, exist_ok=True)
extracted = target / "src" / "biohubx"
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    names = archive.namelist()
    for name in names:
        if name.startswith("/") or ".." in pathlib.PurePosixPath(name).parts:
            raise SystemExit("refusing an archive entry that escapes its root: " + name)
    archive.extractall(extracted)
stage("bootstrap-extract", str(extracted) + " files=" + str(len(names)))

sys.path.insert(0, str(target / "src"))
import biohubx

origin = pathlib.Path(biohubx.__file__).resolve()
if not str(origin).startswith(str(extracted.resolve())):
    raise SystemExit("biohubx was imported from " + str(origin) + ", not the verified payload")
stage("bootstrap-import", str(origin))
stage("start", json.dumps(SPEC["fold"]))
stage("pinned", "commit=" + SPEC["commit"] + " config=" + SPEC["config_digest"])

from biohubx.packaging.entry import run_fold

run_fold(SPEC)
"""


def build_notebook(
    spec: PackageSpec,
    *,
    shipped: dict[str, Any] | None = None,
    payload: bytes | None = None,
    package_root: str = DEFAULT_PACKAGE_ROOT,
) -> dict[str, Any]:
    """A one-cell notebook carrying the package it needs and nothing else.

    ``shipped`` is the spec plus the input digests the run must verify against.
    ``payload`` is the deterministic source archive; the notebook checks its
    digest before extracting it and refuses to import from anywhere else.

    nbformat 4.5 validates before conversion and requires a cell ``id`` and a
    kernelspec ``display_name``. Omitting either fails the notebook at import
    time, which costs a GPU session and produces no output at all. Learned the
    expensive way; a pre-push gate now checks both.
    """
    if payload is None:
        raise PackagingError("a notebook without its package payload could not import anything")
    check_payload_contents(payload)
    source = (
        NOTEBOOK_CELL.replace("__PAYLOAD_SHA256__", hashlib.sha256(payload).hexdigest())
        .replace("__PAYLOAD_B64__", encode_payload(payload))
        .replace("__PACKAGE_ROOT__", package_root)
        .replace("__SPEC_JSON__", json.dumps(shipped or spec.to_dict(), sort_keys=True))
    )
    if "/kaggle/input/biohubx-package" in source:
        raise PackagingError("the notebook still references the unmounted input path")
    return {
        "cells": [
            {
                "id": "biohubx-entry",
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def forbidden_content(root: Path) -> list[str]:
    """Anything in the staged package that must not be shipped.

    Competition bytes and model weights are the two categories that matter. The
    check is on what is actually staged, not on what the builder intended.
    """
    offenders: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if (
            path.suffix.lower() in {".pth", ".pt", ".ckpt", ".safetensors", ".npy", ".npz"}
            or ".zarr" in str(path).lower()
            or ".geff" in str(path).lower()
            or (name.endswith(".csv") and "submission" in name)
        ):
            offenders.append(str(path.relative_to(root)))
    return offenders


def deterministic_archive(root: Path) -> bytes:
    """Zip the source tree so that identical input always yields identical bytes.

    Entries are sorted, timestamps fixed and permissions normalised, because a
    payload whose digest moves between builds cannot be verified at runtime
    against a digest recorded at build time.
    """
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix in ARCHIVE_SUFFIXES),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise PackagingError(f"no source files to archive under {root}")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            name = path.relative_to(root).as_posix()
            if "__pycache__" in name:
                continue
            info = zipfile.ZipInfo(filename=name, date_time=ARCHIVE_EPOCH)
            info.external_attr = 0o644 << 16
            info.create_system = 0
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def archive_inventory(payload: bytes) -> list[str]:
    """Every path inside the payload, so what ships can be read rather than trusted."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return sorted(archive.namelist())


def archive_digest(payload: bytes) -> str:
    """The typed digest the notebook verifies before it extracts anything."""
    return f"raw_artifact_sha256:sha256:{hashlib.sha256(payload).hexdigest()}"


def check_payload_contents(payload: bytes) -> None:
    """Refuse a payload carrying anything but source.

    Competition bytes, checkpoints and external weights must never travel inside
    a notebook. The check reads the archive rather than trusting the filter that
    built it.
    """
    offenders: list[str] = []
    for name in archive_inventory(payload):
        lower = name.lower()
        binary = (".zarr", ".geff", ".pth", ".pt", ".ckpt", ".npy")
        if not lower.endswith(".py") or any(marker in lower for marker in binary):
            offenders.append(name)
    if offenders:
        raise PackagingError(f"the embedded payload carries non-source content: {offenders[:5]}")


def encode_payload(payload: bytes) -> str:
    """Base64 in fixed-width lines, so the notebook diffs readably."""
    encoded = base64.b64encode(payload).decode("ascii")
    return chr(10).join(encoded[index : index + 120] for index in range(0, len(encoded), 120))
