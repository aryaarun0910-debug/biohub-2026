"""Tree identity contract.

A Zarr-backed volume or graph is a directory of many thousands of chunk files
and has no single-file digest. A layout check answers whether something is the
right shape; only this answers whether it is the same data.

These tests fix the properties that make a tree digest an identity rather than a
checksum: it must change when anything about the tree changes, it must not change
when nothing does, and it must be independent of the machine that computed it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from biohubx.hashing import (
    TREE_CANONICALIZATION_VERSION,
    Digest,
    DigestKind,
    NotATreeError,
    digest_file,
    tree_digest,
)


def build(root: Path) -> Path:
    """A miniature of the real artifact shape: chunked data plus metadata."""
    (root / "0" / "c").mkdir(parents=True)
    (root / "0" / "c" / "0.0.0").write_bytes(b"chunk-a")
    (root / "0" / "c" / "0.0.1").write_bytes(b"chunk-b")
    (root / "zarr.json").write_bytes(b'{"shape": [4, 10, 40, 40]}')
    return root


# --- it is a typed identity -------------------------------------------------


def test_the_token_declares_its_kind_and_canonicalization(tmp_path: Path) -> None:
    manifest = tree_digest(build(tmp_path / "a.zarr"))
    assert manifest.digest.token.startswith(f"tree_sha256:sha256/{TREE_CANONICALIZATION_VERSION}:")
    assert Digest.parse(manifest.digest.token) == manifest.digest
    assert Digest.parse(manifest.digest.token).kind is DigestKind.TREE


def test_a_tree_digest_is_not_interchangeable_with_a_file_digest(tmp_path: Path) -> None:
    # Same underlying bytes, different question. The kinds must not collide even
    # when a tree happens to hold exactly one file.
    root = tmp_path / "single.zarr"
    root.mkdir()
    (root / "only").write_bytes(b"payload")
    tree = tree_digest(root).digest
    raw = digest_file(root / "only", DigestKind.RAW_ARTIFACT)
    assert tree.kind is not raw.kind
    assert tree.token != raw.token
    assert tree.hexdigest != raw.hexdigest


def test_a_tree_kind_is_refused_for_a_file(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"x\n")
    with pytest.raises(NotATreeError, match="belongs to a directory"):
        digest_file(target, DigestKind.TREE)


def test_a_tree_digest_must_declare_a_canonicalization() -> None:
    with pytest.raises(ValueError, match="must declare a canonicalization"):
        Digest(kind=DigestKind.TREE, hexdigest="0" * 64)


# --- it does not change when nothing changes -------------------------------


def test_the_digest_is_stable_across_repeated_runs(tmp_path: Path) -> None:
    root = build(tmp_path / "a.zarr")
    assert tree_digest(root).digest == tree_digest(root).digest


def test_the_digest_does_not_depend_on_the_root_name_or_location(tmp_path: Path) -> None:
    # Paths are recorded relative to the root, so moving or renaming the whole
    # artifact does not change what the data is.
    first = build(tmp_path / "here" / "a.zarr")
    second = build(tmp_path / "elsewhere" / "renamed.zarr")
    assert tree_digest(first).digest == tree_digest(second).digest
    assert tree_digest(first).root_name != tree_digest(second).root_name


def test_the_digest_does_not_depend_on_creation_order(tmp_path: Path) -> None:
    forward = tmp_path / "forward.zarr"
    forward.mkdir()
    (forward / "a").write_bytes(b"one")
    (forward / "b").write_bytes(b"two")

    backward = tmp_path / "backward.zarr"
    backward.mkdir()
    (backward / "b").write_bytes(b"two")
    (backward / "a").write_bytes(b"one")

    assert tree_digest(forward).digest == tree_digest(backward).digest


# --- it changes when anything changes ---------------------------------------


def test_changing_one_byte_in_one_chunk_changes_the_digest(tmp_path: Path) -> None:
    root = build(tmp_path / "a.zarr")
    before = tree_digest(root).digest
    (root / "0" / "c" / "0.0.1").write_bytes(b"chunk-B")
    assert tree_digest(root).digest != before


def test_moving_a_file_within_the_tree_changes_the_digest(tmp_path: Path) -> None:
    # Content-only hashing would miss this. The path is part of the identity
    # because a chunk in the wrong place is different data.
    root = build(tmp_path / "a.zarr")
    before = tree_digest(root).digest
    (root / "0" / "c" / "0.0.1").rename(root / "0" / "c" / "0.1.0")
    assert tree_digest(root).digest != before


def test_adding_or_removing_a_file_changes_the_digest(tmp_path: Path) -> None:
    root = build(tmp_path / "a.zarr")
    before = tree_digest(root).digest
    (root / "0" / "c" / "0.0.2").write_bytes(b"chunk-c")
    added = tree_digest(root).digest
    assert added != before
    (root / "0" / "c" / "0.0.2").unlink()
    assert tree_digest(root).digest == before


def test_an_empty_directory_is_part_of_the_identity(tmp_path: Path) -> None:
    # An empty directory contributes no files, so a file-only listing would let
    # it appear and disappear silently.
    root = build(tmp_path / "a.zarr")
    before = tree_digest(root).digest
    (root / "unused").mkdir()
    assert tree_digest(root).digest != before
    assert tree_digest(root).empty_directory_count == 1


def test_two_files_swapping_contents_changes_the_digest(tmp_path: Path) -> None:
    # The multiset of contents is unchanged; only the pairing moved. A digest
    # over sorted contents alone would call these identical.
    root = tmp_path / "a.zarr"
    root.mkdir()
    (root / "a").write_bytes(b"one")
    (root / "b").write_bytes(b"two")
    before = tree_digest(root).digest
    (root / "a").write_bytes(b"two")
    (root / "b").write_bytes(b"one")
    assert tree_digest(root).digest != before


def test_truncating_a_file_to_empty_changes_the_digest(tmp_path: Path) -> None:
    root = build(tmp_path / "a.zarr")
    before = tree_digest(root).digest
    (root / "zarr.json").write_bytes(b"")
    assert tree_digest(root).digest != before


# --- the listing localises a mismatch --------------------------------------


def test_the_manifest_keeps_the_listing_so_a_mismatch_can_be_localised(tmp_path: Path) -> None:
    """A bare digest says something changed. The listing says which file."""
    root = build(tmp_path / "a.zarr")
    before = tree_digest(root)
    (root / "0" / "c" / "0.0.1").write_bytes(b"tampered")
    after = tree_digest(root)

    assert after.digest != before.digest
    changed = {record.relative_path for record in set(after.records) - set(before.records)}
    assert changed == {"0/c/0.0.1"}


def test_the_listing_reproduces_the_digest(tmp_path: Path) -> None:
    import hashlib

    manifest = tree_digest(build(tmp_path / "a.zarr"))
    recomputed = hashlib.sha256(manifest.canonical_listing().encode("utf-8")).hexdigest()
    assert recomputed == manifest.digest.hexdigest


def test_paths_are_recorded_with_forward_slashes(tmp_path: Path) -> None:
    # A Windows checkout and a POSIX one must agree, so the separator is fixed
    # rather than inherited from the host.
    manifest = tree_digest(build(tmp_path / "a.zarr"))
    assert any(record.relative_path == "0/c/0.0.0" for record in manifest.records)
    assert all("\\" not in record.relative_path for record in manifest.records)


def test_counts_and_sizes_are_reported(tmp_path: Path) -> None:
    manifest = tree_digest(build(tmp_path / "a.zarr"))
    assert manifest.file_count == 3
    assert manifest.total_bytes == 7 + 7 + 26


def test_progress_is_reported_so_a_long_run_does_not_look_hung(tmp_path: Path) -> None:
    seen: list[tuple[int, int]] = []
    tree_digest(build(tmp_path / "a.zarr"), on_file=lambda files, size: seen.append((files, size)))
    assert [files for files, _ in seen] == [1, 2, 3]
    assert seen[-1][1] == 40


# --- refusals ---------------------------------------------------------------


def test_a_missing_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NotATreeError, match="does not exist"):
        tree_digest(tmp_path / "absent")


def test_a_file_given_as_a_tree_root_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "a.zarr"
    target.write_bytes(b"not a directory")
    with pytest.raises(NotATreeError, match="not a directory"):
        tree_digest(target)


def test_an_empty_tree_has_an_identity_rather_than_an_error(tmp_path: Path) -> None:
    # An empty artifact is a real state worth recording, distinct from a missing
    # one, which is refused above.
    root = tmp_path / "empty.zarr"
    root.mkdir()
    manifest = tree_digest(root)
    assert manifest.file_count == 0
    assert manifest.digest.kind is DigestKind.TREE


@pytest.mark.skipif(sys.platform == "win32", reason="creating symlinks needs privilege on Windows")
def test_a_symlink_inside_the_tree_is_refused(tmp_path: Path) -> None:
    # Following it could leave the tree; skipping it would lose information.
    # Neither happens silently.
    root = build(tmp_path / "a.zarr")
    (root / "link").symlink_to(root / "zarr.json")
    with pytest.raises(NotATreeError, match="symlink inside the tree"):
        tree_digest(root)
