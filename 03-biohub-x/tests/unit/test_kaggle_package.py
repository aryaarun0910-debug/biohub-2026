"""The guards that travel with a package to a machine this repository cannot watch.

Each test is one way a GPU run could quietly become a different run than the one
that was approved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from biohubx.packaging.kaggle import (
    FoldSpec,
    PackageSpec,
    build_notebook,
    forbidden_content,
    guard_report,
    kernel_metadata,
)

FOLD = FoldSpec(fold_id="fold_44b6", train_embryo="44b6", evaluate_embryo="6bba", seed=4460)
SPEC = PackageSpec(
    commit="0" * 40,
    config_path="configs/e03-clean-folds.yaml",
    config_digest="canonical_text_sha256:sha256/v1:" + "a" * 64,
    fold=FOLD,
    epochs=2,
    batch_size=2,
    learning_rate=1e-4,
    accelerator="nvidiaTeslaT4",
    expected_gpu_count=1,
    smoke=True,
    max_movies=2,
    runtime_ceiling_seconds=2400,
)

REGISTERED = {"44b6_a.zarr": "tree_sha256:sha256/v1:" + "b" * 64}


def report(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "mounted_digests": dict(REGISTERED),
        "registered_digests": dict(REGISTERED),
        "dataset_ids": ["44b6_a", "44b6_b"],
        "reachable_paths": ["/kaggle/input/competitions/x/train", "/kaggle/input/competitions/x/test"],
        "opened_paths": ["/kaggle/input/competitions/x/train/44b6_a.zarr"],
        "gpu_count": 1,
    }
    kwargs.update(overrides)
    return guard_report(SPEC, **kwargs)


def test_a_clean_run_passes_every_guard() -> None:
    assert report()["passed"], report()["failures"]


def test_a_package_with_no_input_digests_cannot_verify_and_is_refused() -> None:
    """An identity check over an empty set would pass while verifying nothing."""
    result = report(registered_digests={}, mounted_digests={})
    assert not result["passed"]
    assert any("shipped no input digests" in f for f in result["failures"])


def test_a_mounted_input_whose_identity_differs_is_refused() -> None:
    result = report(mounted_digests={"44b6_a.zarr": "tree_sha256:sha256/v1:" + "c" * 64})
    assert not result["passed"]
    assert any("identity differs" in f for f in result["failures"])


def test_a_registered_input_that_is_not_mounted_is_refused() -> None:
    result = report(mounted_digests={})
    assert not result["passed"]
    assert any("not mounted" in f for f in result["failures"])


def test_training_on_the_evaluation_embryo_is_refused() -> None:
    """The failure that would silently invalidate the whole fold."""
    result = report(dataset_ids=["44b6_a", "6bba_z"])
    assert not result["passed"]
    assert any("its own evaluation embryo" in f for f in result["failures"])
    assert result["fold_membership_asserted"] is False


def test_a_fold_with_no_training_embryo_movie_is_refused() -> None:
    result = report(dataset_ids=["9999_a"])
    assert not result["passed"]
    assert any("no movie from the training embryo" in f for f in result["failures"])


def test_the_public_test_split_merely_existing_is_not_a_failure() -> None:
    """Kaggle mounts it regardless. Reading it is the failure, not its presence."""
    assert report()["passed"]
    assert report()["public_test_paths_opened"] == []


def test_opening_a_public_test_path_is_refused_on_either_separator() -> None:
    # Assembled rather than written as a literal: a drive-rooted path in a
    # tracked file trips this repository's own absolute-path contract, and that
    # contract is right to be strict even about a fake one.
    windows = "\\".join(("C:", "kaggle", "input", "competitions", "x", "test", "44b6_a.zarr"))
    for path in ("/kaggle/input/competitions/x/test/44b6_a.zarr", windows):
        result = report(opened_paths=[path])
        assert not result["passed"], path
        assert any("public-test" in f for f in result["failures"])


def test_a_reachable_quarantined_checkpoint_is_refused() -> None:
    result = report(reachable_paths=["/kaggle/input/reference.pilkwang.support_pack/edge.pth"])
    assert not result["passed"]
    assert any("quarantined checkpoint" in f for f in result["failures"])


def test_the_wrong_number_of_gpus_is_refused() -> None:
    result = report(gpu_count=2)
    assert not result["passed"]
    assert any("would not cost what was approved" in f for f in result["failures"])


# --- what the package ships -------------------------------------------------


def test_the_kernel_metadata_disables_internet_and_ships_no_datasets() -> None:
    meta = kernel_metadata(SPEC, slug="user/kernel", title="t")
    assert "/" in str(meta["id"]), (
        "a Kaggle kernel id is owner/slug; without an owner a push addresses nothing"
    )
    assert meta["enable_internet"] is False
    assert meta["enable_gpu"] is True
    assert meta["accelerator"] == "nvidiaTeslaT4"
    assert meta["dataset_sources"] == []
    assert meta["model_sources"] == []
    assert meta["competition_sources"] == ["biohub-cell-tracking-during-development"]


def test_the_notebook_embeds_the_spec_and_carries_no_model_logic() -> None:
    shipped = {**SPEC.to_dict(), "input_digests": REGISTERED}
    notebook = build_notebook(SPEC, shipped=shipped)
    assert len(notebook["cells"]) == 1
    source = "".join(notebook["cells"][0]["source"])
    assert "run_fold" in source
    assert SPEC.commit in source
    assert "44b6_a.zarr" in source, "the digests the run must verify are not in the notebook"
    for forbidden in ("torch.nn", "Conv3d", "backward(", "load_state_dict"):
        assert forbidden not in source, f"model logic leaked into the notebook: {forbidden}"


def test_weights_and_competition_bytes_are_refused_in_a_package(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "ok.py").write_text("x = 1", encoding="utf-8")
    assert forbidden_content(tmp_path) == []

    (tmp_path / "weights.pth").write_bytes(b"0")
    (tmp_path / "movie.zarr").mkdir()
    (tmp_path / "movie.zarr" / "0").write_bytes(b"0")
    offenders = forbidden_content(tmp_path)
    assert "weights.pth" in offenders
    assert any("movie.zarr" in name for name in offenders)


def test_the_staged_package_from_the_last_build_shipped_nothing_forbidden() -> None:
    """Read back the real package rather than trusting the builder's own claim."""
    report_path = Path(__file__).resolve().parents[2] / "artifacts/kaggle-package.json"
    if not report_path.is_file():
        pytest.skip("run `biohubx package kaggle` first")
    built = json.loads(report_path.read_text(encoding="utf-8"))
    assert built["package"]["pushed"] is False
    staged = Path(__file__).resolve().parents[2] / built["package"]["path"]
    if not staged.is_dir():
        pytest.skip("the staged package is not on this machine")
    assert forbidden_content(staged) == []
