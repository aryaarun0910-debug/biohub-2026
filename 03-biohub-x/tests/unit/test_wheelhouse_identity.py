"""A wheelhouse has two identities, and recording one of them is how they get confused.

Kaggle consumes ``dataset-metadata.json`` to create a dataset and does not store
it, so the tree that leaves this machine and the tree a kernel mounts differ by
exactly that file. The authorisation for E03-WHEELHOUSE-PREFLIGHT-01 named the
first and the platform published the second, which read as an inventory mismatch
and stopped a push.

The load-bearing test here is that the published payload's digest equals the
digest of a tree genuinely built without that file. If it did not, the recorded
published identity would be an arithmetic result rather than an assertion about
any tree that exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biohubx.hashing import TreeManifest, tree_digest
from biohubx.packaging.wheelhouse import (
    KAGGLE_CONFIGURATION_FILENAME,
    WheelhouseError,
    canonical_package_name,
    compare_manifests,
    locked_wheels,
    published_relative_paths,
    requirement_hashes,
    stage_published_payload,
)

LOCK = """
[[package]]
name = "zarr"
version = "3.3.0"

[[package.wheels]]
url = "https://files.pythonhosted.org/packages/aa/bb/zarr-3.3.0-py3-none-any.whl"
hash = "sha256:323bf5366d4f909052ef6e2e03e7481a7434c3ee75d3a981eb3a71fc1ae22cef"
size = 363685

[[package]]
name = "google-crc32c"
version = "1.8.0"

[[package.wheels]]
url = "https://files.pythonhosted.org/packages/cc/dd/google_crc32c-1.8.0-cp312-cp312-manylinux_2_17_x86_64.whl"
hash = "sha256:14f87e04d613dfa218d6135e81b78272c3b904e2a7053b841481b38a7d901411"
"""
"""Table syntax where uv.lock uses inline tables. Both parse to the same
structure, and the real lock is exercised by its own test below."""

REQUIREMENTS = """\
# Generated from uv.lock. Offline install only.
zarr==3.3.0 \\
    --hash=sha256:323bf5366d4f909052ef6e2e03e7481a7434c3ee75d3a981eb3a71fc1ae22cef
google-crc32c==1.8.0 \\
    --hash=sha256:14f87e04d613dfa218d6135e81b78272c3b904e2a7053b841481b38a7d901411
"""


def build_bundle(root: Path, *, configuration: bool = True) -> Path:
    """A wheelhouse shaped like the real one: wheels in a subdirectory, files at the root."""
    (root / "wheels").mkdir(parents=True)
    (root / "wheels/zarr-3.3.0-py3-none-any.whl").write_bytes(b"zarr wheel bytes")
    (root / "wheels/donfig-0.8.1.post1-py3-none-any.whl").write_bytes(b"donfig wheel bytes")
    (root / "requirements-offline.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (root / "README.md").write_text("what this is\n", encoding="utf-8")
    if configuration:
        (root / KAGGLE_CONFIGURATION_FILENAME).write_text('{"id": "owner/slug"}\n', encoding="utf-8")
    return root


def published_manifest(bundle_root: Path, destination: Path) -> TreeManifest:
    bundle = tree_digest(bundle_root)
    destination.mkdir(parents=True)
    stage_published_payload(bundle_root, destination, published_relative_paths(bundle))
    return tree_digest(destination)


def test_the_published_payload_digest_equals_a_tree_actually_built_without_the_configuration(
    tmp_path: Path,
) -> None:
    """The whole point. A staged payload must be indistinguishable from the real thing.

    Built two independent ways: once by staging the bundle's payload, and once by
    creating the same wheelhouse with no configuration file at all. Equal digests
    mean the recorded published identity describes a tree that exists rather than
    a listing with a line removed.
    """
    staged = published_manifest(build_bundle(tmp_path / "bundle"), tmp_path / "staged")
    never_had_one = tree_digest(build_bundle(tmp_path / "without", configuration=False))

    assert staged.digest == never_had_one.digest
    assert staged.file_count == never_had_one.file_count


def test_the_two_identities_differ_by_exactly_the_configuration_file(tmp_path: Path) -> None:
    bundle_root = build_bundle(tmp_path / "bundle")
    bundle = tree_digest(bundle_root)
    published = published_manifest(bundle_root, tmp_path / "staged")

    assert bundle.digest != published.digest
    assert bundle.file_count - published.file_count == 1
    configuration_bytes = (bundle_root / KAGGLE_CONFIGURATION_FILENAME).stat().st_size
    assert bundle.total_bytes - published.total_bytes == configuration_bytes


def test_only_the_root_configuration_is_excluded_not_a_nested_file_of_that_name(
    tmp_path: Path,
) -> None:
    """A nested file of the same name is data, and dropping it would corrupt the payload."""
    bundle_root = build_bundle(tmp_path / "bundle")
    nested = bundle_root / "wheels" / KAGGLE_CONFIGURATION_FILENAME
    nested.write_text("payload, not configuration\n", encoding="utf-8")

    relative_paths = published_relative_paths(tree_digest(bundle_root))

    assert f"wheels/{KAGGLE_CONFIGURATION_FILENAME}" in relative_paths
    assert KAGGLE_CONFIGURATION_FILENAME not in relative_paths


def test_a_same_size_substitution_is_caught_because_content_is_compared(tmp_path: Path) -> None:
    """The negative control. A flipped byte moves no count and no total.

    This is the case a filename-and-size check passes, which is why the published
    dataset is compared by content and not by the inventory Kaggle reports.
    """
    expected = tree_digest(build_bundle(tmp_path / "expected", configuration=False))
    tampered_root = build_bundle(tmp_path / "tampered", configuration=False)
    (tampered_root / "wheels/zarr-3.3.0-py3-none-any.whl").write_bytes(b"zarr wheel bytez")
    observed = tree_digest(tampered_root)

    assert expected.file_count == observed.file_count
    assert expected.total_bytes == observed.total_bytes
    differences = compare_manifests(expected, observed)
    assert [difference.reason for difference in differences] == ["content"]
    assert differences[0].relative_path == "wheels/zarr-3.3.0-py3-none-any.whl"


def test_compare_names_a_missing_file_an_extra_file_and_a_size_change(tmp_path: Path) -> None:
    expected = tree_digest(build_bundle(tmp_path / "expected", configuration=False))
    observed_root = build_bundle(tmp_path / "observed", configuration=False)
    (observed_root / "README.md").unlink()
    (observed_root / "EXTRA.txt").write_text("uninvited\n", encoding="utf-8")
    (observed_root / "wheels/donfig-0.8.1.post1-py3-none-any.whl").write_bytes(b"longer donfig bytes")

    differences = compare_manifests(expected, tree_digest(observed_root))
    reasons = {(difference.relative_path, difference.reason) for difference in differences}

    assert ("README.md", "missing") in reasons
    assert ("EXTRA.txt", "unexpected") in reasons
    assert ("wheels/donfig-0.8.1.post1-py3-none-any.whl", "size") in reasons


def test_identical_trees_report_no_differences(tmp_path: Path) -> None:
    expected = tree_digest(build_bundle(tmp_path / "expected", configuration=False))
    observed = tree_digest(build_bundle(tmp_path / "observed", configuration=False))

    assert compare_manifests(expected, observed) == ()


def test_requirement_hashes_reads_continuations_and_normalises_the_name() -> None:
    hashes = requirement_hashes(REQUIREMENTS)

    assert hashes["zarr"][0] == "3.3.0"
    assert hashes["google-crc32c"][1] == "14f87e04d613dfa218d6135e81b78272c3b904e2a7053b841481b38a7d901411"
    assert canonical_package_name("google_crc32c") == "google-crc32c"


def test_an_unreadable_requirement_line_is_refused_rather_than_skipped() -> None:
    """A skipped line is an unpinned install that still looks pinned."""
    with pytest.raises(WheelhouseError, match="cannot read this requirement line"):
        requirement_hashes("zarr==3.3.0\n")


def test_a_wheel_the_lock_does_not_name_is_refused() -> None:
    with pytest.raises(WheelhouseError, match=r"appears in no uv\.lock entry"):
        locked_wheels(LOCK, ["numcodecs-0.15.1-cp312-cp312-manylinux_2_17_x86_64.whl"])


def test_a_lock_entry_without_a_size_records_an_unknown_size_not_a_zero() -> None:
    """uv.lock does not promise a size, and a defaulted zero would compare as a mismatch."""
    resolved = locked_wheels(
        LOCK,
        [
            "zarr-3.3.0-py3-none-any.whl",
            "google_crc32c-1.8.0-cp312-cp312-manylinux_2_17_x86_64.whl",
        ],
    )

    assert resolved["zarr-3.3.0-py3-none-any.whl"].size_bytes == 363685
    assert resolved["google_crc32c-1.8.0-cp312-cp312-manylinux_2_17_x86_64.whl"].size_bytes is None


def test_the_repositorys_own_uv_lock_parses_under_this_reader() -> None:
    """A fixture proves the reader. Only the real lock proves it reads uv's format.

    No digest is asserted here, because a hash copied into a test is a hash
    duplicated outside the registry. What this holds is that uv.lock still states
    wheels in a shape this reader understands, which is what would break silently.
    """
    lock_text = (Path(__file__).resolve().parents[2] / "uv.lock").read_text(encoding="utf-8")

    resolved = locked_wheels(lock_text, ["zarr-3.3.0-py3-none-any.whl"])

    assert resolved["zarr-3.3.0-py3-none-any.whl"].package == "zarr"
    assert resolved["zarr-3.3.0-py3-none-any.whl"].version == "3.3.0"
    assert len(resolved["zarr-3.3.0-py3-none-any.whl"].sha256) == 64


def test_wheels_are_matched_by_filename_so_a_sibling_platform_wheel_is_not_evidence() -> None:
    """One package resolves to many wheels; only the file on disk is the one staged."""
    lock = LOCK.replace(
        "google_crc32c-1.8.0-cp312-cp312-manylinux_2_17_x86_64.whl",
        "google_crc32c-1.8.0-cp312-cp312-macosx_12_0_arm64.whl",
    )

    with pytest.raises(WheelhouseError, match=r"appears in no uv\.lock entry"):
        locked_wheels(lock, ["google_crc32c-1.8.0-cp312-cp312-manylinux_2_17_x86_64.whl"])
