"""External tree artifacts and tiered verification.

A dataset tree is unlike everything else in the registry: it lives outside the
repository, its location is machine-local, and reading it takes minutes. The
registry holds its identity anyway, and verification is tiered so the gate stays
fast without any check being skipped in silence.

The rule these tests fix is that a shape check is never allowed to read as an
identity check. Every result carries the depth at which it was made.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from biohubx.artifacts import (
    ARTIFACT_REGISTRY_PATH,
    ArtifactRecord,
    ArtifactRegistry,
    VerificationDepth,
    load_artifact_registry,
    summarise_checks,
    verify_registry,
)
from biohubx.hashing import tree_digest

REPO_ROOT = Path(__file__).resolve().parents[2]
TREE_TOKEN = "tree_sha256:sha256/v1:" + "a" * 64
RAW_TOKEN = "raw_artifact_sha256:sha256:" + "0" * 64


def tree_record(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "competition.train.d1.zarr",
        "kind": "competition_dataset_zarr",
        "external_path": "C:/elsewhere/d1.zarr",
        "schema": "official-split-layout/v1",
        "digests": {"tree": TREE_TOKEN},
        "shape": {"file_count": 3, "empty_directory_count": 0, "total_bytes": 40},
        "provenance": {"status": "external_uncleared"},
    }
    base.update(overrides)
    return base


def build_tree(root: Path) -> Path:
    (root / "0").mkdir(parents=True)
    (root / "0" / "a").write_bytes(b"chunk-a")
    (root / "0" / "b").write_bytes(b"chunk-b")
    (root / "meta").write_bytes(b"{}")
    return root


def registered(target: Path, **overrides: object) -> ArtifactRegistry:
    """A registry holding the true identity of a tree that exists on disk."""
    measured = tree_digest(target)
    record = tree_record(
        external_path=str(target),
        digests={"tree": measured.digest.token},
        shape={
            "file_count": measured.file_count,
            "empty_directory_count": measured.empty_directory_count,
            "total_bytes": measured.total_bytes,
        },
    )
    record.update(overrides)
    return ArtifactRegistry.model_validate({"schema_version": 1, "artifacts": [record]})


# --- the record can express an external tree -------------------------------


def test_an_external_tree_artifact_is_representable() -> None:
    record = ArtifactRecord.model_validate(tree_record())
    assert record.is_tree
    assert record.path is None
    assert record.external_path == "C:/elsewhere/d1.zarr"


def test_an_artifact_must_record_exactly_one_location() -> None:
    with pytest.raises(ValidationError, match="exactly one location"):
        ArtifactRecord.model_validate(tree_record(path="data/d1.zarr"))
    nowhere = tree_record()
    del nowhere["external_path"]
    with pytest.raises(ValidationError, match="exactly one location"):
        ArtifactRecord.model_validate(nowhere)


def test_a_tree_digest_cannot_be_mixed_with_a_file_digest() -> None:
    # A directory has no single-file identity and a file has no tree identity.
    with pytest.raises(ValidationError, match="mixes a tree digest"):
        ArtifactRecord.model_validate(tree_record(digests={"tree": TREE_TOKEN, "raw": RAW_TOKEN}))


def test_a_tree_digest_without_a_shape_is_refused() -> None:
    # Without a recorded shape the cheap tier has nothing to compare against, so
    # the artifact could only ever be checked by reading every byte.
    shapeless = tree_record()
    del shapeless["shape"]
    with pytest.raises(ValidationError, match="tree digest with no shape"):
        ArtifactRecord.model_validate(shapeless)


def test_a_shape_on_a_non_tree_artifact_is_refused() -> None:
    with pytest.raises(ValidationError, match="records a shape but is not a tree"):
        ArtifactRecord.model_validate(
            {
                "id": "x",
                "kind": "document",
                "path": "README.md",
                "schema": "markdown",
                "digests": {"raw": RAW_TOKEN},
                "shape": {"file_count": 1, "empty_directory_count": 0, "total_bytes": 1},
                "provenance": {"status": "measured"},
            }
        )


# --- the two tiers, and the difference between them ------------------------


def test_the_default_tier_checks_shape_and_says_so(tmp_path: Path) -> None:
    target = build_tree(tmp_path / "d1.zarr")
    checks = verify_registry(registered(target), tmp_path)
    assert len(checks) == 1
    assert checks[0].ok
    assert checks[0].depth is VerificationDepth.SHAPE
    assert "not deeply verified" in (checks[0].detail or "")


def test_the_deep_tier_re_derives_the_identity(tmp_path: Path) -> None:
    target = build_tree(tmp_path / "d1.zarr")
    checks = verify_registry(registered(target), tmp_path, deep=True)
    assert checks[0].ok
    assert checks[0].depth is VerificationDepth.DEEP
    assert checks[0].observed == checks[0].expected


def test_a_same_size_byte_flip_passes_shape_and_fails_deep(tmp_path: Path) -> None:
    """The honest limit of the cheap tier, asserted rather than assumed.

    Shape compares counts and total size. A byte flipped in place changes
    neither, so the shape tier passes it and only the deep tier catches it. That
    is exactly why a shape pass is reported as not deeply verified rather than
    as a clean bill of health.
    """
    target = build_tree(tmp_path / "d1.zarr")
    registry = registered(target)
    (target / "0" / "a").write_bytes(b"chunk-X")  # same length, different content

    shallow = verify_registry(registry, tmp_path)
    assert shallow[0].ok, "the shape tier cannot see a same-size change"
    assert shallow[0].depth is VerificationDepth.SHAPE

    deep = verify_registry(registry, tmp_path, deep=True)
    assert not deep[0].ok
    assert deep[0].detail == "tree digest mismatch"


def test_a_changed_shape_fails_even_the_cheap_tier(tmp_path: Path) -> None:
    target = build_tree(tmp_path / "d1.zarr")
    registry = registered(target)
    (target / "0" / "c").write_bytes(b"an extra chunk")

    checks = verify_registry(registry, tmp_path)
    assert not checks[0].ok
    assert "file_count" in (checks[0].detail or "")


def test_a_changed_total_size_fails_the_cheap_tier(tmp_path: Path) -> None:
    target = build_tree(tmp_path / "d1.zarr")
    registry = registered(target)
    (target / "0" / "a").write_bytes(b"chunk-a-with-more-bytes")

    checks = verify_registry(registry, tmp_path)
    assert not checks[0].ok
    assert "total_bytes" in (checks[0].detail or "")


# --- presence, and who is entitled to be missing ---------------------------


def test_an_external_artifact_absent_from_this_machine_is_reported_not_failed(
    tmp_path: Path,
) -> None:
    # The repository must contain what it claims; a machine need not hold every
    # dataset. Absent is reported so it can never be mistaken for verified.
    target = build_tree(tmp_path / "d1.zarr")
    registry = registered(target)
    shutil.rmtree(target)

    checks = verify_registry(registry, tmp_path)
    assert checks[0].ok
    assert checks[0].depth is VerificationDepth.ABSENT
    assert "not on this machine" in (checks[0].detail or "")


def test_an_in_repository_tree_that_is_absent_is_a_failure(tmp_path: Path) -> None:
    # The repository did claim this one, so its absence is a broken claim.
    target = build_tree(tmp_path / "d1.zarr")
    record = registered(target).artifacts[0].model_dump(by_alias=True, exclude_none=True)
    record["path"] = "absent/d1.zarr"
    del record["external_path"]
    in_repo = ArtifactRegistry.model_validate({"schema_version": 1, "artifacts": [record]})

    checks = verify_registry(in_repo, tmp_path)
    assert not checks[0].ok
    assert checks[0].depth is VerificationDepth.ABSENT


# --- the report states what was established --------------------------------


def test_the_summary_counts_results_by_depth(tmp_path: Path) -> None:
    target = build_tree(tmp_path / "d1.zarr")
    totals = summarise_checks(verify_registry(registered(target), tmp_path))
    assert totals == {"checked": 1, "failed": 0, "deep": 0, "shape_only": 1, "absent": 0}


def test_the_repository_registry_is_still_verified_deeply() -> None:
    # The in-repository artifacts must not have been quietly demoted to the
    # cheap tier by the arrival of a kind of artifact that needs it.
    loaded = load_artifact_registry(REPO_ROOT / ARTIFACT_REGISTRY_PATH)
    checks = verify_registry(loaded, REPO_ROOT)
    repository_ids = {record.id for record in loaded.artifacts if record.path is not None}
    repository_checks = [check for check in checks if check.artifact_id in repository_ids]
    assert repository_checks
    assert all(check.depth is VerificationDepth.DEEP for check in repository_checks)
    assert all(check.ok for check in repository_checks)
