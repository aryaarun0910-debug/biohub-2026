"""Does the wheelhouse install offline, and can the result read a real chunk.

Two questions in that order, because a wheelhouse that installs but cannot decode
blosc has answered neither. The competition is attached for exactly one read:
readability is the whole point, and a synthetic round trip alone would prove only
that zarr can read what zarr just wrote.

Like the environment audit, this imports no part of Biohub-X, so it runs on an
image where Biohub-X cannot.

Consumer: ``biohubx package preflight``.
"""

from __future__ import annotations

import json
from typing import Any

from biohubx.packaging.audit import AuditError, AuditSpec

PREFLIGHT_ID = "E03-WHEELHOUSE-PREFLIGHT-01"
PREFLIGHT_KERNEL = "aryaarun07/biohub-x-wheelhouse-preflight"
PREFLIGHT_TITLE = "Biohub X wheelhouse preflight"
WHEELHOUSE_SLUG = "aryaarun07/biohubx-wheelhouse-zarr-cp312-linux"
INPUT_ROOT_DEFAULT = "/kaggle/input"
"""Where the search starts. The wheelhouse is found inside it by identity.

E03-WHEELHOUSE-PREFLIGHT-01 assumed /kaggle/input/<slug> and refused, having
measured only that its guess was wrong (R-0007). A path is a guess; the tree
digest the package already carries is not, so it locates as well as verifies."""
PREFLIGHT_RUNTIME_CEILING_SECONDS = 600

EXPECTED_PREFLIGHT_STAGES: tuple[str, ...] = (
    "preflight-start",
    "wheelhouse",
    "verify",
    "install",
    "import",
    "roundtrip",
    "competition-read",
    "manifest",
    "done",
)

TREE_VERIFIER_SOURCE = '''\
def biohubx_relative(root, path):
    relative = path.relative_to(root).as_posix()
    if chr(10) in relative or chr(0) in relative:
        raise RuntimeError("path contains a line feed or null byte: " + repr(relative))
    return relative


def biohubx_canonical_tree(root):
    """Tree canonicalization v1, carried by this notebook rather than imported.

    The package may not import Biohub-X, because it has to run on an image where
    Biohub-X cannot, so the authoritative walk cannot travel as code. It travels
    as behaviour instead: a test requires this function to return exactly what
    biohubx.hashing.tree_digest returns for the same tree, so a divergence between
    the two fails the gate here rather than passing silently over there.

    The walk is explicit and judges every entry before descending into it. Listing
    a tree with rglob would follow a symlink and record foreign files under paths
    that look local before any check could run.
    """
    records = []
    pending = [root]
    while pending:
        directory = pending.pop()
        children = sorted(directory.iterdir(), key=lambda item: item.name)
        if not children:
            if directory != root:
                records.append(("d", "-", "-", biohubx_relative(root, directory)))
            continue
        for child in children:
            if child.is_symlink():
                raise RuntimeError(
                    "symlink inside the mounted tree at " + biohubx_relative(root, child)
                )
            if child.is_dir():
                pending.append(child)
            elif child.is_file():
                digest = hashlib.sha256()
                with open(child, "rb") as handle:
                    while True:
                        block = handle.read(1048576)
                        if not block:
                            break
                        digest.update(block)
                records.append(
                    ("f", digest.hexdigest(), str(child.stat().st_size), biohubx_relative(root, child))
                )
            else:
                raise RuntimeError(
                    "neither a regular file nor a directory: " + biohubx_relative(root, child)
                )
    records.sort(key=lambda record: record[3].encode("utf-8"))
    listing = "".join(
        record[0] + " " + record[1] + " " + record[2] + " " + record[3] + chr(10) for record in records
    )
    token = "tree_sha256:sha256/v1:" + hashlib.sha256(listing.encode("utf-8")).hexdigest()
    return token, records


def biohubx_compare(expected, observed):
    """Name every difference at the path where it happens.

    A digest says the tree is wrong. A path says which file, which is the
    difference between an alarm someone can act on and one they cannot. A rename
    appears as one missing path and one extra path, which is what a rename is.
    """
    want = dict((record[3], record) for record in expected)
    got = dict((record[3], record) for record in observed)
    differences = []
    for relative in sorted(set(want) - set(got)):
        differences.append("missing " + relative)
    for relative in sorted(set(got) - set(want)):
        differences.append("extra " + relative)
    for relative in sorted(set(want) & set(got)):
        if want[relative][2] != got[relative][2]:
            differences.append(
                "size " + relative + " expected=" + want[relative][2] + " observed=" + got[relative][2]
            )
        if want[relative][1] != got[relative][1]:
            differences.append(
                "content " + relative + " expected=" + want[relative][1] + " observed=" + got[relative][1]
            )
    return differences


BIOHUBX_MAX_DEPTH = 4


def biohubx_candidate_roots(input_root, max_depth):
    """Directories under the input root, shallowest first, bounded, one level at a time.

    Never a recursive glob. The competition mount is a corpus of hundreds of
    chunked volumes, and descending all of it to find a directory of four wheels
    would cost more than the run it is part of. Four levels covers a plain mount,
    a mount under a kind, and a mount under a kind and an owner, with one spare.
    """
    # The root counts as a candidate. Otherwise pointing the search straight at
    # the wheelhouse would fail to find it, which is a surprising edge and costs
    # nothing to remove: an oversized root is rejected by shape like any other.
    roots = [input_root]
    frontier = [input_root]
    depth = 0
    while frontier and depth < max_depth:
        following = []
        for parent in frontier:
            try:
                children = sorted(parent.iterdir(), key=lambda item: item.name)
            except OSError:
                continue
            for child in children:
                if child.is_symlink() or not child.is_dir():
                    continue
                roots.append(child)
                following.append(child)
        frontier = following
        depth += 1
    return roots


def biohubx_shape(root, max_files, max_bytes):
    """File count and total bytes, abandoned the moment either cap is passed.

    The cap is what keeps the search cheap. A candidate already larger than the
    tree being looked for cannot be that tree, so there is no reason to finish
    counting it and every reason not to hash it.
    """
    files = 0
    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError:
            return None
        for child in children:
            if child.is_symlink():
                return None
            if child.is_dir():
                pending.append(child)
            elif child.is_file():
                files += 1
                if files > max_files:
                    return None
                try:
                    total += child.stat().st_size
                except OSError:
                    return None
                if total > max_bytes:
                    return None
    return (files, total)


def biohubx_find_wheelhouse(input_root, expected):
    """Locate the authorised tree by its identity rather than by a guessed path.

    A name says where to look. A digest says when you have actually found it,
    which is the question that matters, and it is the same anchor the package
    already carries, so locating and verifying rest on one fact instead of two.
    Shape is checked first, so only a tree of exactly the right size is ever
    hashed. A tree that is the right size and the wrong content is reported as a
    near miss with its differences, because that is the interesting failure.
    """
    records = [tuple(item) for item in expected["records"]]
    want_files = len([record for record in records if record[0] == "f"])
    want_bytes = sum(int(record[2]) for record in records if record[0] == "f")
    near_misses = []
    hashed = 0
    for candidate in biohubx_candidate_roots(input_root, BIOHUBX_MAX_DEPTH):
        if biohubx_shape(candidate, want_files, want_bytes) != (want_files, want_bytes):
            continue
        hashed += 1
        token, observed = biohubx_canonical_tree(candidate)
        if token == expected["tree"]:
            return candidate, near_misses, hashed
        near_misses.append((str(candidate), token, biohubx_compare(records, observed)))
    return None, near_misses, hashed


def biohubx_listing(input_root, limit=60):
    """What is actually mounted, so that a refusal diagnoses itself.

    The previous refusal named the path it wanted and not the paths it had, which
    left a missing dataset and a dataset mounted elsewhere indistinguishable.
    """
    entries = []
    frontier = [input_root]
    depth = 0
    while frontier and depth < 3:
        following = []
        for parent in frontier:
            try:
                children = sorted(parent.iterdir(), key=lambda item: item.name)
            except OSError:
                continue
            for child in children:
                is_directory = child.is_dir()
                entries.append(str(child) + ("/" if is_directory else ""))
                if len(entries) >= limit:
                    return entries
                if is_directory and not child.is_symlink():
                    following.append(child)
        frontier = following
        depth += 1
    return entries
'''


def preflight_kernel_metadata(kernel: str = PREFLIGHT_KERNEL, title: str = PREFLIGHT_TITLE) -> dict[str, Any]:
    """CPU only, internet disabled, the wheelhouse attached and nothing else added."""
    return {
        "id": kernel,
        "title": title,
        "code_file": "run.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [WHEELHOUSE_SLUG],
        "kernel_sources": [],
        "model_sources": [],
        "competition_sources": ["biohub-cell-tracking-during-development"],
    }


PREFLIGHT_CELL = '''\
# Generated by `biohubx package preflight`. Do not edit here.
#
# Two questions, in order. Does the wheelhouse install offline on this image, and
# can the result decode a real competition chunk. Imports no part of Biohub-X, so
# it runs on an image where Biohub-X cannot.
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time


def stage(name, detail=""):
    print(("BIOHUBX_STAGE " + name + " " + str(detail)).rstrip(), flush=True)


__TREE_VERIFIER__

SPEC = json.loads(r"""__SPEC_JSON__""")
# The authorised identity of the tree this kernel expects to mount, carried here
# rather than read from the mount. requirements-offline.txt cannot authenticate
# itself: a substituted dataset shipping its own matching requirements file would
# satisfy --require-hashes exactly. The anchor has to arrive with the code.
EXPECTED = json.loads(r"""__PUBLISHED_JSON__""")
INPUT_ROOT = pathlib.Path(os.environ.get("BIOHUBX_INPUT_ROOT", "__INPUT_ROOT__"))
COMPETITION = pathlib.Path(
    os.environ.get(
        "BIOHUB_DATA_ROOT", "/kaggle/input/competitions/biohub-cell-tracking-during-development"
    )
)
OUT = pathlib.Path(os.environ.get("BIOHUBX_OUTPUT", "/kaggle/working"))

started = time.time()
stage("preflight-start", SPEC["preflight_id"] + " commit=" + SPEC["commit"])
report = {"schema_version": 1, "preflight_id": SPEC["preflight_id"], "commit": SPEC["commit"]}


def write_report():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "wheelhouse-preflight.json"
    partial = manifest.with_suffix(".json.partial")
    partial.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + chr(10), encoding="utf-8"
    )
    partial.replace(manifest)
    return manifest


stage("wheelhouse", "searching " + str(INPUT_ROOT) + " for the authorised tree")
resolved, near_misses, hashed = biohubx_find_wheelhouse(INPUT_ROOT, EXPECTED)
report["wheelhouse"] = {
    "input_root": str(INPUT_ROOT),
    "located_by": "the identity the package carries, not a guessed path",
    "candidates_hashed": hashed,
    "near_misses": [
        {"path": path, "observed_tree": token, "differences": differences[:20]}
        for path, token, differences in near_misses
    ],
    "resolved": None if resolved is None else str(resolved),
}
if resolved is None:
    listing = biohubx_listing(INPUT_ROOT)
    report["wheelhouse"]["mounted"] = listing
    report["fatal"] = "the authorised wheelhouse is not mounted under " + str(INPUT_ROOT)
    stage("wheelhouse", "NOT FOUND expected=" + EXPECTED["tree"])
    for entry in listing[:40]:
        stage("wheelhouse", "mounted " + entry)
    for path, token, _ in near_misses:
        stage("wheelhouse", "near miss " + path + " observed=" + token)
    write_report()
    raise SystemExit("the authorised wheelhouse is not mounted under " + str(INPUT_ROOT))
WHEELHOUSE = resolved
wheels = sorted(p.name for p in (WHEELHOUSE / "wheels").glob("*.whl"))
report["wheelhouse"]["root"] = str(WHEELHOUSE)
report["wheelhouse"]["wheels"] = wheels
stage("wheelhouse", "resolved " + str(WHEELHOUSE) + " wheels=" + str(len(wheels)))

stage("verify", "recomputing the mounted tree under canonicalization v1")
expected_records = [tuple(item) for item in EXPECTED["records"]]
observed_tree, observed_records = biohubx_canonical_tree(WHEELHOUSE)
differences = biohubx_compare(expected_records, observed_records)
matches = observed_tree == EXPECTED["tree"] and not differences
report["verify"] = {
    "expected_tree": EXPECTED["tree"],
    "observed_tree": observed_tree,
    "expected_file_count": len(expected_records),
    "observed_file_count": len(observed_records),
    "differences": differences,
    "matches": matches,
}
stage("verify", "expected=" + EXPECTED["tree"])
stage("verify", "observed=" + observed_tree)
for difference in differences[:20]:
    stage("verify", "difference " + difference)
if not matches:
    report["fatal"] = "the mounted wheelhouse is not the authorised published payload"
    write_report()
    raise SystemExit("the mounted wheelhouse is not the authorised published payload")
stage("verify", "the mounted tree is the authorised published payload; nothing extra, missing or altered")

stage("install", "offline, no index, no deps, hashes required")
command = [
    sys.executable, "-m", "pip", "install",
    "--no-index", "--no-deps", "--require-hashes",
    "-r", str(WHEELHOUSE / "requirements-offline.txt"),
    "--find-links", str(WHEELHOUSE / "wheels"),
]
completed = subprocess.run(command, capture_output=True, text=True, check=False)
report["install"] = {
    "command": command,
    "returncode": completed.returncode,
    "stdout_tail": completed.stdout[-2000:],
    "stderr_tail": completed.stderr[-2000:],
}
stage("install", "returncode=" + str(completed.returncode))
if completed.returncode != 0:
    report["fatal"] = "offline install failed"
    write_report()
    raise SystemExit("offline install failed")

stage("import", "begin")
imports = {}
for name in ("zarr", "numcodecs", "donfig", "google_crc32c"):
    entry = {"importable": False}
    try:
        module = __import__(name)
        entry["importable"] = True
        entry["version"] = str(getattr(module, "__version__", "unknown"))
        entry["file"] = str(getattr(module, "__file__", "unknown"))
    except BaseException as exc:
        entry["error"] = repr(exc)[:300]
    imports[name] = entry
    stage("import", name + " importable=" + str(entry["importable"]))
report["imports"] = imports

stage("roundtrip", "begin")
roundtrip = {"codec": "blosc zstd bitshuffle typesize 2, the competition's own"}
try:
    import numpy as np
    import zarr

    target = OUT / "roundtrip.zarr"
    array = zarr.create_array(
        store=str(target),
        shape=(2, 4, 8, 8),
        chunks=(1, 4, 8, 8),
        dtype="uint16",
        compressors=zarr.codecs.BloscCodec(
            cname="zstd", clevel=1, shuffle=zarr.codecs.BloscShuffle.bitshuffle, typesize=2
        ),
        overwrite=True,
    )
    written = np.arange(2 * 4 * 8 * 8, dtype="uint16").reshape(2, 4, 8, 8)
    array[:] = written
    roundtrip["ok"] = bool(np.array_equal(written, zarr.open_array(str(target), mode="r")[:]))
except BaseException as exc:
    roundtrip["error"] = repr(exc)[:400]
    roundtrip["ok"] = False
report["roundtrip"] = roundtrip
stage("roundtrip", json.dumps(roundtrip)[:200])

stage("competition-read", "begin")
read = {"root": str(COMPETITION), "mounted": COMPETITION.is_dir(), "decoded": False}
try:
    import numpy as np
    import zarr

    train = COMPETITION / "train"
    visible = sorted(train.glob("*.zarr")) if train.is_dir() else []
    read["volumes_visible"] = len(visible)
    if visible:
        data = zarr.open(str(visible[0]), mode="r")["0"]
        chunk = np.asarray(data[0, 0])
        read["dataset"] = visible[0].name
        read["shape"] = list(data.shape)
        read["dtype"] = str(data.dtype)
        read["chunk_shape"] = list(chunk.shape)
        read["chunk_min"] = int(chunk.min())
        read["chunk_max"] = int(chunk.max())
        read["decoded"] = True
except BaseException as exc:
    read["error"] = repr(exc)[:400]
report["competition_read"] = read
stage("competition-read", "decoded=" + str(read["decoded"]))

report["elapsed_seconds"] = round(time.time() - started, 3)
manifest = write_report()
stage("manifest", str(manifest) + " bytes=" + str(manifest.stat().st_size))
stage("done", "elapsed=" + str(report["elapsed_seconds"]) + "s")
'''


def build_preflight_notebook(
    spec: AuditSpec,
    *,
    published_payload: dict[str, Any],
    input_root: str = INPUT_ROOT_DEFAULT,
) -> dict[str, Any]:
    """One cell, no Biohub-X import, and the identity it will check travelling inside it.

    ``published_payload`` carries the expected tree token and the canonical records
    behind it. Both are required and neither has a default: a preflight that would
    install from whatever it happened to mount is the thing this argument exists to
    make unbuildable.
    """
    token = str(published_payload.get("tree", ""))
    records = published_payload.get("records") or []
    if not token.startswith("tree_sha256:sha256/v1:"):
        raise AuditError(f"the expected published payload must be a canonical tree identity, got {token!r}")
    if not records:
        raise AuditError(
            "the expected published payload carries no records, so a mismatch could name no file "
            "and the check would report only that something changed"
        )
    payload = {**spec.to_dict(), "preflight_id": spec.audit_id}
    source = (
        PREFLIGHT_CELL.replace("__TREE_VERIFIER__", TREE_VERIFIER_SOURCE)
        .replace("__SPEC_JSON__", json.dumps(payload, sort_keys=True))
        .replace(
            "__PUBLISHED_JSON__",
            json.dumps({"tree": token, "records": [list(record) for record in records]}, sort_keys=True),
        )
        .replace("__INPUT_ROOT__", input_root)
    )
    for forbidden in ("import biohubx", "from biohubx"):
        if forbidden in source:
            raise AuditError(f"the preflight must not contain {forbidden!r}")
    return {
        "cells": [
            {
                "id": "biohubx-wheelhouse-preflight",
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
