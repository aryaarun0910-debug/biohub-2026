"""Read-only data validation.

`biohubx data validate` exists to answer one question honestly: is the root I
was handed the layout I expect? It must never guess a location, never write into
the data, and never treat an ambiguous layout as good enough.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biohubx.data.validation import (
    synthetic_validation_report,
    validate_data_root,
)


def make_pair(root: Path, dataset_id: str) -> None:
    """Create the sibling pair the official layout uses.

    The .zarr is a directory upstream; the validator only inspects names and
    sizes, so a directory is what the fixture creates. This differs from
    production in cost, not in kind.
    """
    (root / f"{dataset_id}.zarr").mkdir(parents=True)
    (root / f"{dataset_id}.geff").write_bytes(b"")


def test_a_paired_root_validates(tmp_path: Path) -> None:
    make_pair(tmp_path, "44b6_0b24845f")
    make_pair(tmp_path, "6bba_1c93aa02")
    manifests = validate_data_root(tmp_path)
    assert [manifest.dataset_id for manifest in manifests] == ["44b6_0b24845f", "6bba_1c93aa02"]
    assert all(manifest.schema == "official-zarr-geff-sibling-pair/v1" for manifest in manifests)


def test_the_split_identity_is_the_dataset_not_the_path(tmp_path: Path) -> None:
    # Splits must be groupable by source movie. A path would not survive being
    # moved; the dataset id is the stable identity.
    make_pair(tmp_path, "44b6_0b24845f")
    manifest = validate_data_root(tmp_path)[0]
    assert manifest.split_identity == "44b6_0b24845f"
    assert manifest.relative_path == "."


def test_a_missing_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_data_root(tmp_path / "absent")


def test_a_file_given_as_a_root_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "not-a-directory"
    target.write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        validate_data_root(target)


def test_an_empty_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"no \.zarr/\.geff"):
        validate_data_root(tmp_path)


def test_an_unpaired_zarr_is_refused(tmp_path: Path) -> None:
    # A movie with no annotations cannot be silently treated as trainable.
    (tmp_path / "lonely.zarr").mkdir()
    with pytest.raises(ValueError, match="unpaired"):
        validate_data_root(tmp_path)


def test_an_unpaired_geff_is_refused(tmp_path: Path) -> None:
    (tmp_path / "lonely.geff").write_bytes(b"")
    with pytest.raises(ValueError, match="unpaired"):
        validate_data_root(tmp_path)


def test_a_nested_layout_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    # Upstream ships train/ and test/ subdirectories. Pointing the validator at
    # the parent is ambiguous, and picking one for the caller would be a guess.
    make_pair(tmp_path, "top")
    make_pair(tmp_path / "train", "nested")
    with pytest.raises(ValueError, match="ambiguous data layout"):
        validate_data_root(tmp_path)


def test_validation_does_not_modify_the_root(tmp_path: Path) -> None:
    make_pair(tmp_path, "44b6_0b24845f")
    before = sorted(path.name for path in tmp_path.iterdir())
    validate_data_root(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_the_synthetic_report_touches_no_real_data() -> None:
    report = synthetic_validation_report()
    assert report["mode"] == "synthetic"
    datasets = report["datasets"]
    assert isinstance(datasets, list)
    assert len(datasets) == 1
    entry = datasets[0]
    assert entry["provenance"] == "generated-in-memory; no source data accessed"
    # Sizes are None, not 0: an unknown size is recorded as unknown rather than
    # zero-filled into something a reader could mistake for a measurement.
    assert set(entry["sizes_bytes"].values()) == {None}
