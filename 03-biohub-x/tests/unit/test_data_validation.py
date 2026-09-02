"""Read-only data validation.

`biohubx data validate` answers one question honestly: is the root I was handed
the layout I expect? It must never guess a location, never write into the data,
and never treat an ambiguous layout as good enough. It must also not treat the
documented absence of ground truth in the test split as a fault.

Fixtures build the artifacts as directories, because a `.zarr` and a `.geff` are
Zarr-backed directory trees on disk, not files. Building them as files would let
these tests pass against a validator that could not read the real thing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biohubx.data.validation import (
    ArtifactKind,
    RoleProvenance,
    SplitRole,
    synthetic_validation_report,
    validate_data_root,
)


def make_volume(directory: Path, dataset_id: str) -> None:
    """A .zarr as it really is: a directory with chunk content inside."""
    chunks = directory / f"{dataset_id}.zarr" / "0"
    chunks.mkdir(parents=True)
    (chunks / "c").write_bytes(b"")


def make_truth(directory: Path, dataset_id: str) -> None:
    """A .geff as it really is: a directory with nodes/ and edges/ inside."""
    root = directory / f"{dataset_id}.geff"
    (root / "nodes").mkdir(parents=True)
    (root / "edges").mkdir(parents=True)


def make_official_root(root: Path, *, with_test: bool = True) -> None:
    train = root / "train"
    train.mkdir(parents=True)
    for dataset_id in ("44b6_0b24845f", "6bba_283bf9f1"):
        make_volume(train, dataset_id)
        make_truth(train, dataset_id)
    if with_test:
        test = root / "test"
        test.mkdir(parents=True)
        make_volume(test, "9fd1_aa77bb01")
    (root / "sample_submission.csv").write_bytes(b"")


# --- the official split layout ---------------------------------------------


def test_the_official_root_validates(tmp_path: Path) -> None:
    make_official_root(tmp_path)
    manifests = validate_data_root(tmp_path)
    assert sorted(manifest.dataset_id for manifest in manifests) == [
        "44b6_0b24845f",
        "6bba_283bf9f1",
        "9fd1_aa77bb01",
    ]


def test_the_test_split_is_unannotated_by_design_not_broken(tmp_path: Path) -> None:
    # The point of the change: a volume with no ground truth in `test/` is the
    # documented layout, and refusing it would refuse the real dataset.
    make_official_root(tmp_path)
    by_id = {manifest.dataset_id: manifest for manifest in validate_data_root(tmp_path)}
    submission = by_id["9fd1_aa77bb01"]
    assert submission.split == "test"
    assert submission.split_role is SplitRole.UNANNOTATED
    assert [artifact.name for artifact in submission.artifacts] == ["9fd1_aa77bb01.zarr"]


def test_the_train_split_requires_ground_truth(tmp_path: Path) -> None:
    make_official_root(tmp_path)
    by_id = {manifest.dataset_id: manifest for manifest in validate_data_root(tmp_path)}
    annotated = by_id["44b6_0b24845f"]
    assert annotated.split_role is SplitRole.ANNOTATED
    assert [artifact.name for artifact in annotated.artifacts] == [
        "44b6_0b24845f.zarr",
        "44b6_0b24845f.geff",
    ]


def test_a_role_taken_from_an_official_split_name_says_so(tmp_path: Path) -> None:
    # A declared role and an inferred one are different kinds of knowledge, and
    # the manifest records which one it holds.
    make_official_root(tmp_path)
    assert all(
        manifest.role_provenance is RoleProvenance.OFFICIAL_SPLIT_NAME
        for manifest in validate_data_root(tmp_path)
    )


def test_a_missing_ground_truth_in_train_is_refused(tmp_path: Path) -> None:
    make_official_root(tmp_path)
    make_volume(tmp_path / "train", "lonely")
    with pytest.raises(ValueError, match=r"Volumes with no \.geff"):
        validate_data_root(tmp_path)


def test_ground_truth_appearing_in_the_test_split_is_refused(tmp_path: Path) -> None:
    # Labels in the submission split would mean the layout is not what it claims.
    make_official_root(tmp_path)
    make_truth(tmp_path / "test", "9fd1_aa77bb01")
    with pytest.raises(ValueError, match="unannotated by design"):
        validate_data_root(tmp_path)


def test_a_root_that_is_both_a_split_parent_and_a_dataset_holder_is_refused(tmp_path: Path) -> None:
    make_official_root(tmp_path)
    make_volume(tmp_path, "stray")
    with pytest.raises(ValueError, match="one or the other"):
        validate_data_root(tmp_path)


def test_artifacts_nested_below_a_split_are_refused(tmp_path: Path) -> None:
    make_official_root(tmp_path)
    make_volume(tmp_path / "train" / "extra", "buried")
    with pytest.raises(ValueError, match="nests dataset artifacts"):
        validate_data_root(tmp_path)


def test_a_split_with_no_volumes_is_refused(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    with pytest.raises(ValueError, match=r"contains no \.zarr volumes"):
        validate_data_root(tmp_path)


# --- presence is unambiguous ------------------------------------------------


def test_a_present_directory_is_distinguishable_from_an_absent_artifact(tmp_path: Path) -> None:
    """The defect this replaced: both used to report a size of None.

    A Zarr-backed artifact is a directory, so a size-only record made a present
    dataset indistinguishable from a missing one.
    """
    make_official_root(tmp_path)
    present = validate_data_root(tmp_path)[0].artifacts[0]
    assert present.kind is ArtifactKind.DIRECTORY
    assert present.entry_count is not None and present.entry_count > 0
    assert present.size_bytes is None

    datasets = synthetic_validation_report()["datasets"]
    assert isinstance(datasets, list)
    missing = datasets[0]["artifacts"][0]
    assert missing["kind"] is ArtifactKind.ABSENT
    assert missing["entry_count"] is None
    # Both records carry the same size field. Only `kind` separates them, which
    # is the whole repair: previously there was nothing but the size to look at.
    assert missing["size_bytes"] == present.size_bytes


def test_a_regular_file_reports_bytes_not_an_entry_count(tmp_path: Path) -> None:
    train = tmp_path / "train"
    train.mkdir()
    (train / "flat.zarr").write_bytes(b"0123456789")
    make_truth(train, "flat")
    artifact = validate_data_root(tmp_path)[0].artifacts[0]
    assert artifact.kind is ArtifactKind.FILE
    assert artifact.size_bytes == 10
    assert artifact.entry_count is None


# --- a single leaf split ----------------------------------------------------


def test_a_leaf_directory_of_paired_datasets_validates(tmp_path: Path) -> None:
    for dataset_id in ("a", "b"):
        make_volume(tmp_path, dataset_id)
        make_truth(tmp_path, dataset_id)
    manifests = validate_data_root(tmp_path)
    assert [manifest.dataset_id for manifest in manifests] == ["a", "b"]
    assert all(manifest.split == "." for manifest in manifests)
    assert all(manifest.split_role is SplitRole.ANNOTATED for manifest in manifests)


def test_a_leaf_role_records_that_it_was_inferred(tmp_path: Path) -> None:
    make_volume(tmp_path, "a")
    make_truth(tmp_path, "a")
    assert validate_data_root(tmp_path)[0].role_provenance is RoleProvenance.INFERRED_FROM_CONTENTS


def test_a_leaf_directory_of_volumes_alone_is_read_as_unannotated(tmp_path: Path) -> None:
    make_volume(tmp_path, "a")
    assert validate_data_root(tmp_path)[0].split_role is SplitRole.UNANNOTATED


def test_a_half_annotated_leaf_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    # Neither role fits, and picking one would be a guess about whether the
    # missing ground truth is intentional.
    make_volume(tmp_path, "a")
    make_truth(tmp_path, "a")
    make_volume(tmp_path, "b")
    with pytest.raises(ValueError, match="partly annotated"):
        validate_data_root(tmp_path)


def test_the_refusal_names_which_side_is_missing(tmp_path: Path) -> None:
    # The message this replaced labelled the two sets in a way that read
    # backwards: it printed `geff=[id]` for a dataset whose geff was absent.
    make_volume(tmp_path, "has-volume-only")
    make_volume(tmp_path, "paired")
    make_truth(tmp_path, "paired")
    make_truth(tmp_path, "has-truth-only")
    with pytest.raises(ValueError) as caught:
        validate_data_root(tmp_path)
    message = str(caught.value)
    assert "volume but no ground truth: ['has-volume-only']" in message
    assert "Ground truth with no volume: ['has-truth-only']" in message


# --- refusals that have nothing to do with splits ---------------------------


def test_a_missing_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_data_root(tmp_path / "absent")


def test_a_file_given_as_a_root_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "not-a-directory"
    target.write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        validate_data_root(target)


def test_an_empty_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no train/ or test/ split directory"):
        validate_data_root(tmp_path)


def test_validation_does_not_modify_the_root(tmp_path: Path) -> None:
    make_official_root(tmp_path)
    before = sorted(path.as_posix() for path in tmp_path.rglob("*"))
    validate_data_root(tmp_path)
    assert sorted(path.as_posix() for path in tmp_path.rglob("*")) == before


def test_the_synthetic_report_touches_no_real_data() -> None:
    report = synthetic_validation_report()
    assert report["mode"] == "synthetic"
    datasets = report["datasets"]
    assert isinstance(datasets, list)
    assert len(datasets) == 1
    assert datasets[0]["provenance"] == "generated-in-memory; no source data accessed"
    assert all(artifact["kind"] is ArtifactKind.ABSENT for artifact in datasets[0]["artifacts"])
