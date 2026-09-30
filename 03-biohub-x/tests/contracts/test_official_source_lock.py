"""Official-source lock contract.

The metric mathematics are not reimplemented in Biohub-X. They are byte-identical
vendored upstream code, pinned to one commit, with typed digests recorded in
`registry/official_source.yaml`. These tests make that pin binding: if the
vendored bytes drift, or the adapter's pin and the lock disagree, the build fails
rather than quietly scoring against a different authority.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from biohubx.evaluation.official_metric import OFFICIAL_MAX_DISTANCE_UM, OFFICIAL_SOURCE_COMMIT
from biohubx.hashing import Digest, DigestKind, digest_file

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / "registry/official_source.yaml"
VENDOR_ROOT = REPO_ROOT / "src/biohubx/_vendor/official_competition"


@pytest.fixture(scope="module")
def lock() -> dict[str, Any]:
    document: dict[str, Any] = yaml.safe_load(LOCK_PATH.read_text(encoding="utf-8"))
    return document


def test_every_vendored_digest_still_holds(lock: dict[str, Any]) -> None:
    sources = lock["sources"]
    assert sources, "the lock asserts nothing"
    slot_kind = {"raw": DigestKind.RAW_ARTIFACT, "canonical_text": DigestKind.CANONICAL_TEXT}
    failures: list[str] = []
    for source in sources:
        path = REPO_ROOT / source["vendored_path"]
        if not path.is_file():
            failures.append(f"{source['vendored_path']}: missing")
            continue
        for slot, token in source["digests"].items():
            observed = digest_file(path, slot_kind[slot]).token
            if observed != token:
                failures.append(f"{source['vendored_path']} [{slot}]: {observed} != {token}")
    assert not failures, "vendored official source drifted:\n" + "\n".join(failures)


def test_every_digest_is_a_typed_token(lock: dict[str, Any]) -> None:
    slot_kind = {"raw": DigestKind.RAW_ARTIFACT, "canonical_text": DigestKind.CANONICAL_TEXT}
    for source in lock["sources"]:
        for slot, token in source["digests"].items():
            assert Digest.parse(token).kind is slot_kind[slot]


def test_the_adapter_pin_and_the_lock_name_the_same_commit(lock: dict[str, Any]) -> None:
    # Two places record the commit: the lock, and the constant the adapter
    # exposes to reports. If they ever disagree, a report would cite provenance
    # the code does not actually have.
    assert lock["repository"]["commit"] == OFFICIAL_SOURCE_COMMIT
    assert len(OFFICIAL_SOURCE_COMMIT) == 40
    assert lock["repository"]["license"] == "BSD-3-Clause"


def test_the_upstream_license_is_retained_beside_the_vendored_code() -> None:
    # BSD-3-Clause requires the notice to travel with the copy.
    license_path = VENDOR_ROOT / "LICENSE"
    assert license_path.is_file()
    text = license_path.read_text(encoding="utf-8")
    assert "Redistributions of source code must retain" in text


def test_the_lock_covers_every_upstream_module(lock: dict[str, Any]) -> None:
    # A vendored upstream file with no recorded digest is unpinned code that
    # looks pinned. The check is scoped to the copied upstream package; the two
    # package markers above it are Biohub-X shims, covered by the test below.
    locked = {source["vendored_path"] for source in lock["sources"]}
    upstream = {
        str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        for path in (VENDOR_ROOT / "tracking_cellmot").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    assert upstream, "no upstream modules found; the vendored copy is missing"
    assert upstream <= locked, f"vendored but unlocked: {sorted(upstream - locked)}"


def test_the_package_markers_around_the_upstream_copy_contain_no_logic() -> None:
    """The two shims that make the vendored copy importable must stay inert.

    They are Biohub-X files, not upstream ones, so they carry no recorded digest.
    That is only safe while they hold nothing but a docstring: any code there
    would be unpinned behaviour sitting on the authoritative scoring path.
    """
    markers = (
        REPO_ROOT / "src/biohubx/_vendor/__init__.py",
        VENDOR_ROOT / "__init__.py",
    )
    for marker in markers:
        assert marker.is_file(), f"missing package marker: {marker}"
        body = [
            line
            for line in marker.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith('"""') and not line.lstrip().startswith("#")
        ]
        assert not body, f"package marker contains logic: {marker}"


def test_vendored_files_are_tracked_by_git() -> None:
    # The vendored copy is the authority at scoring time. If it were ignored, a
    # clone would score against nothing at all.
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "src/biohubx/_vendor/official_competition/LICENSE"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, "vendored official source is not tracked by git"


def test_the_matching_radius_is_the_official_seven_micrometres() -> None:
    assert OFFICIAL_MAX_DISTANCE_UM == 7.0


def test_runtime_dependencies_are_pinned_exactly(lock: dict[str, Any]) -> None:
    # The official matching runs inside tracksdata. A range would let the
    # authoritative behaviour move underneath a recorded score.
    dependencies = lock["runtime_dependencies"]
    assert dependencies["tracksdata"]["commit"] == "7bfeaf845ceb951226f19b72fe5b80e01601018a"
    for name in ("tracksdata", "polars", "scipy"):
        assert name in dependencies, f"{name} is used by the metric path but is not pinned"


def test_installed_runtime_matches_the_pinned_versions(lock: dict[str, Any]) -> None:
    import polars
    import scipy  # type: ignore[import-untyped]
    import tracksdata  # type: ignore[import-untyped]

    dependencies = lock["runtime_dependencies"]
    assert polars.__version__ == dependencies["polars"]["version"]
    assert scipy.__version__ == dependencies["scipy"]["version"]
    assert tracksdata.__version__ == dependencies["tracksdata"]["version"]
