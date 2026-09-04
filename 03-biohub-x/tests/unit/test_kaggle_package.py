"""The guards that travel with a package to a machine this repository cannot watch.

Each test is one way a GPU run could quietly become a different run than the one
that was approved.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from biohubx.packaging.kaggle import (
    KERNEL_SOURCE_LIMIT_BYTES,
    FoldSpec,
    PackageSpec,
    PackagingError,
    archive_digest,
    archive_inventory,
    build_notebook,
    check_payload_contents,
    deterministic_archive,
    forbidden_content,
    guard_report,
    kernel_id_for,
    kernel_metadata,
    kernel_source_bytes,
    slug_of,
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
    accelerator="NvidiaTeslaT4",
    allowed_gpu_counts=(1, 2),
    expected_device_substring="Tesla T4",
    smoke=True,
    smoke_id="E03-SMOKE-03",
    wheelhouse_slug="aryaarun07/biohubx-wheelhouse-zarr-cp312-linux",
    wheelhouse_tree="tree_sha256:sha256/v1:" + "c" * 64,
    max_movies=2,
    runtime_ceiling_seconds=2400,
)

WHEELHOUSE = {
    "tree": "tree_sha256:sha256/v1:" + "c" * 64,
    "records": [["f", "d" * 64, "9", "requirements-offline.txt"]],
}

REGISTERED = {"44b6_a.zarr": "tree_sha256:sha256/v1:" + "b" * 64}


def repo_source() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "biohubx"


def report(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "mounted_digests": dict(REGISTERED),
        "registered_digests": dict(REGISTERED),
        "dataset_ids": ["44b6_a", "44b6_b"],
        "reachable_paths": ["/kaggle/input/competitions/x/train", "/kaggle/input/competitions/x/test"],
        "opened_paths": ["/kaggle/input/competitions/x/train/44b6_a.zarr"],
        "gpu_count": 1,
        "device_names": ["Tesla T4"],
        "device_vram_bytes": [15636037632],
        "training_device": "cuda:0",
        "dataset_sources": ["aryaarun07/biohubx-wheelhouse-zarr-cp312-linux"],
        "wheelhouse_verified": True,
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


def test_a_gpu_count_outside_the_permitted_set_is_refused() -> None:
    result = report(gpu_count=3, device_names=["Tesla T4"] * 3, device_vram_bytes=[1] * 3)
    assert not result["passed"]
    assert any("would not cost what was approved" in f for f in result["failures"])


# --- what the package ships -------------------------------------------------


def test_the_kernel_metadata_disables_internet_and_ships_no_datasets() -> None:
    meta = kernel_metadata(SPEC, slug="user/kernel", title="kernel")
    assert "/" in str(meta["id"]), (
        "a Kaggle kernel id is owner/slug; without an owner a push addresses nothing"
    )
    assert meta["enable_internet"] is False
    assert meta["enable_gpu"] is True
    # Canonical casing. The lower-cased form was requested for attempt 2 and
    # Kaggle allocated a P100 (D-0028).
    assert meta["accelerator"] == "NvidiaTeslaT4"
    # Both keys. R-0010 observed that public T4 notebooks pin machine_shape and
    # that Kaggle echoed that key for our own preflight; the accelerator key alone
    # did not take effect for E03-SMOKE-02.
    assert meta["machine_shape"] == "NvidiaTeslaT4"
    # Exactly the wheelhouse and nothing else. E03 needs zarr, which the image
    # lacks; it needs no external weights, and an extra source is how one would
    # arrive.
    assert meta["dataset_sources"] == ["aryaarun07/biohubx-wheelhouse-zarr-cp312-linux"]
    assert meta["model_sources"] == []
    assert meta["competition_sources"] == ["biohub-cell-tracking-during-development"]


def test_the_notebook_embeds_the_spec_and_carries_no_model_logic() -> None:
    shipped = {**SPEC.to_dict(), "input_digests": REGISTERED}
    notebook = build_notebook(
        SPEC,
        shipped=shipped,
        payload=deterministic_archive(repo_source()),
        wheelhouse_payload=WHEELHOUSE,
    )
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


# --- the two defects that cost a GPU session --------------------------------


def test_the_notebook_carries_the_fields_nbformat_validates_before_running() -> None:
    """A missing display_name or cell id fails at conversion, before any guard runs.

    That is the worst possible failure: a GPU session spent, no heartbeat, no
    result, and nothing measured. It happened once.
    """
    notebook = build_notebook(
        SPEC,
        shipped={**SPEC.to_dict(), "input_digests": REGISTERED},
        payload=deterministic_archive(repo_source()),
        wheelhouse_payload=WHEELHOUSE,
    )

    kernelspec = notebook["metadata"]["kernelspec"]
    for required in ("display_name", "language", "name"):
        assert required in kernelspec, f"nbformat requires kernelspec.{required}"
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    for cell in notebook["cells"]:
        assert cell.get("id"), "nbformat 4.5 requires every cell to carry an id"
        for required in ("cell_type", "metadata", "source"):
            assert required in cell


def test_a_title_that_does_not_slugify_to_the_id_is_refused() -> None:
    """Kaggle slugifies the title and uses that, so a mismatch silently relocates the kernel."""
    assert slug_of("biohubx e03 fold 44b6") == "biohubx-e03-fold-44b6"
    assert slug_of("Biohub-X E03 fold_44b6") == "biohub-x-e03-fold-44b6"

    with pytest.raises(PackagingError, match="slugifies to"):
        kernel_metadata(SPEC, slug="owner/biohubx-e03-fold-44b6", title="Biohub-X E03 fold_44b6")

    ok = kernel_metadata(SPEC, slug="owner/biohubx-e03-fold-44b6", title="biohubx e03 fold 44b6")
    assert ok["id"] == "owner/biohubx-e03-fold-44b6"


def test_a_kernel_id_without_an_owner_is_refused() -> None:
    with pytest.raises(PackagingError, match="owner/slug"):
        kernel_metadata(SPEC, slug="biohubx-e03-fold-44b6", title="biohubx e03 fold 44b6")


# --- the pre-push gate ------------------------------------------------------


def test_the_kernel_id_is_derived_from_the_title_so_they_cannot_disagree() -> None:
    assert kernel_id_for("aryaarun07", "Biohub-X E03 fold 44b6") == "aryaarun07/biohub-x-e03-fold-44b6"
    with pytest.raises(PackagingError, match="bare Kaggle account slug"):
        kernel_id_for("owner/extra", "t")
    with pytest.raises(PackagingError, match="bare Kaggle account slug"):
        kernel_id_for("", "t")


def test_the_gate_rejects_the_notebook_that_failed_on_kaggle() -> None:
    """The exact shape E03-SMOKE pushed must now fail locally, before any network call."""
    prepush = pytest.importorskip(
        "biohubx.packaging.prepush", reason="the package-gate dependency group is not installed"
    )
    pytest.importorskip("nbformat", reason="the package-gate dependency group is not installed")

    broken = {
        "cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "x = 1"}
        ],
        "metadata": {"kernelspec": {"language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with pytest.raises(prepush.PrePushError, match="kernelspec is missing"):
        prepush.check_required_metadata(broken)
    with pytest.raises(prepush.PrePushError, match="rejected the generated notebook"):
        prepush.validate_notebook(broken)


def test_the_generated_notebook_validates_and_converts() -> None:
    prepush = pytest.importorskip(
        "biohubx.packaging.prepush", reason="the package-gate dependency group is not installed"
    )
    pytest.importorskip("nbconvert", reason="the package-gate dependency group is not installed")

    notebook = build_notebook(
        SPEC,
        shipped={**SPEC.to_dict(), "input_digests": REGISTERED},
        payload=deterministic_archive(repo_source()),
        wheelhouse_payload=WHEELHOUSE,
    )
    prepush.check_required_metadata(notebook)
    prepush.validate_notebook(notebook)
    assert prepush.convert_notebook(notebook) > 0


def test_a_gate_report_with_no_heartbeat_does_not_pass() -> None:
    prepush = pytest.importorskip(
        "biohubx.packaging.prepush", reason="the package-gate dependency group is not installed"
    )
    silent = prepush.PrePushReport(
        nbformat_validated=True,
        nbconvert_converted=True,
        converted_bytes=1,
        metadata_fields_present=True,
        entry_point_exercised=True,
        stage_lines_seen=0,
    )
    assert not silent.passed


# --- the notebook carries its own package -----------------------------------


def test_the_archive_is_byte_identical_across_builds() -> None:
    """A payload whose digest moves cannot be verified at runtime against a recorded one."""
    first = deterministic_archive(repo_source())
    second = deterministic_archive(repo_source())
    assert first == second
    assert archive_digest(first) == archive_digest(second)
    assert archive_digest(first).startswith("raw_artifact_sha256:sha256:")


def test_the_payload_carries_source_and_nothing_else() -> None:
    payload = deterministic_archive(repo_source())
    inventory = archive_inventory(payload)
    assert inventory, "an empty payload could not import anything"
    assert all(name.endswith(".py") for name in inventory)
    assert not any("__pycache__" in name for name in inventory)
    check_payload_contents(payload)


def test_a_payload_holding_a_checkpoint_is_refused(tmp_path: Path) -> None:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("biohubx/__init__.py", "x = 1")
        archive.writestr("weights/best.pth", "0")
    with pytest.raises(PackagingError, match="non-source content"):
        check_payload_contents(buffer.getvalue())


def test_the_notebook_never_imports_from_the_unmounted_input_path() -> None:
    """No dataset or model source supplies the package, so that path is not promised."""
    notebook = build_notebook(
        SPEC,
        shipped={**SPEC.to_dict(), "input_digests": REGISTERED},
        payload=deterministic_archive(repo_source()),
        wheelhouse_payload=WHEELHOUSE,
    )
    source = "".join(notebook["cells"][0]["source"])
    assert "/kaggle/input/biohubx-package" not in source
    assert "/kaggle/working/biohubx-package" in source
    assert "hashlib.sha256" in source, "the notebook must verify its payload before extracting"


def test_a_notebook_without_a_payload_is_refused() -> None:
    with pytest.raises(PackagingError, match="could not import anything"):
        build_notebook(SPEC, shipped=SPEC.to_dict(), payload=None, wheelhouse_payload=WHEELHOUSE)


def test_the_embedded_cell_is_valid_python() -> None:
    import ast

    notebook = build_notebook(
        SPEC,
        shipped={**SPEC.to_dict(), "input_digests": REGISTERED},
        payload=deterministic_archive(repo_source()),
        wheelhouse_payload=WHEELHOUSE,
    )
    ast.parse("".join(notebook["cells"][0]["source"]))


def test_the_expected_stage_sequence_is_checked_as_a_subsequence() -> None:
    prepush = pytest.importorskip(
        "biohubx.packaging.prepush", reason="the package-gate dependency group is not installed"
    )
    complete = list(prepush.EXPECTED_STAGES)
    assert prepush.missing_from_sequence(complete) == ()
    assert prepush.missing_from_sequence([*complete, "extra"]) == ()
    without_done = [name for name in complete if name != "done"]
    assert "done" in prepush.missing_from_sequence(without_done)
    assert prepush.missing_from_sequence(list(reversed(complete)))


def test_an_unapproved_gpu_model_is_refused() -> None:
    """Counting devices is not checking hardware.

    Attempt 2 was authorised for a T4, Kaggle allocated a P100, device_count was
    1, and the count guard passed while the run proceeded on hardware nobody
    approved.
    """
    assert report(device_names=["Tesla T4"])["passed"]

    wrong = report(device_names=["Tesla P100-PCIE-16GB"])
    assert not wrong["passed"]
    assert any("did not take effect" in f for f in wrong["failures"])


# --- E03-SMOKE-03: the hardware and the dependency source -------------------


def test_an_accelerator_run_that_finds_no_accelerator_refuses() -> None:
    """Attempt 2 passed a count check and ran on a P100. A missing card must refuse too.

    Guarding the model only when a device is present leaves the case where none
    is, which is the same adjacent-check shape: the check sits next to the thing
    that matters instead of on it.
    """
    failures = report(gpu_count=0, device_names=[], device_vram_bytes=[], training_device="cpu")["failures"]

    assert any("Tesla T4" in line for line in failures)


def test_a_p100_is_refused_by_name() -> None:
    outcome = report(device_names=["Tesla P100-PCIE-16GB"])

    assert not outcome["passed"]
    assert any("P100" in line for line in outcome["failures"])
    assert outcome["device_names"] == ["Tesla P100-PCIE-16GB"]


def test_a_dataset_source_that_is_not_the_wheelhouse_is_refused_by_name() -> None:
    """An external weight pack is what an extra source would smuggle in."""
    outcome = report(
        dataset_sources=[
            "aryaarun07/biohubx-wheelhouse-zarr-cp312-linux",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
        ]
    )

    assert not outcome["passed"]
    assert any("pilkwang" in line for line in outcome["failures"])


def test_an_unverified_wheelhouse_refuses_even_when_everything_else_passes() -> None:
    """Saying nothing must not pass. The default is False for exactly this case."""
    outcome = report(wheelhouse_verified=False)

    assert not outcome["passed"]
    assert any("not verified" in line for line in outcome["failures"])


def test_the_notebook_verifies_and_installs_the_wheelhouse_before_importing_biohubx() -> None:
    """biohubx imports zarr, so installing after the import installs too late."""
    source = "".join(
        build_notebook(
            SPEC,
            shipped={**SPEC.to_dict(), "input_digests": REGISTERED},
            payload=deterministic_archive(repo_source()),
            wheelhouse_payload=WHEELHOUSE,
        )["cells"][0]["source"]
    )

    verified = source.index("biohubx_canonical_tree(wheelhouse_root)")
    installed = source.index('"pip", "install"')
    imported = source.index("import biohubx")
    assert verified < installed < imported
    assert str(WHEELHOUSE["tree"]) in source


def test_a_package_cannot_be_built_without_the_wheelhouse_identity_it_will_check() -> None:
    with pytest.raises(PackagingError, match="canonical tree token"):
        build_notebook(
            SPEC,
            shipped=SPEC.to_dict(),
            payload=deterministic_archive(repo_source()),
            wheelhouse_payload={"tree": "", "records": [["f", "a", "1", "b"]]},
        )


def test_the_attempt_id_travels_in_the_package() -> None:
    """E03-SMOKE-01 and -02 are separate immutable records; -03 must be nameable too."""
    source = "".join(
        build_notebook(
            SPEC,
            shipped={**SPEC.to_dict(), "input_digests": REGISTERED},
            payload=deterministic_archive(repo_source()),
            wheelhouse_payload=WHEELHOUSE,
        )["cells"][0]["source"]
    )

    assert "E03-SMOKE-03" in source
    assert SPEC.to_dict()["smoke_id"] == "E03-SMOKE-03"
    assert report()["smoke_id"] == "E03-SMOKE-03"


def test_the_gpu_count_guard_compares_the_observed_count_not_the_expected_one() -> None:
    """E03-SMOKE-03 was approved for one T4 and Kaggle allocated two.

    The guard passed, because smoke mode handed it `expected` in place of
    `observed` before comparing them, so the check could not fail in the only mode
    that used it. The observed count now reaches the guard, and the tolerance
    Arya Arun authorised on 2026-09-04 is a declared set rather than a widened
    comparison: a count outside the set still refuses and still names what it saw.
    """
    outcome = report(gpu_count=4, device_names=["Tesla T4"] * 4, device_vram_bytes=[1] * 4)

    assert not outcome["passed"]
    assert any("reports 4" in line for line in outcome["failures"])
    assert outcome["gpu_count"] == 4
    assert outcome["allowed_gpu_counts"] == [1, 2]


def test_two_visible_t4s_are_accepted_and_recorded() -> None:
    """[[R-0011]]: Kaggle allocated two cards against a request for one.

    With the count guard corrected by [[D-0038]] that allocation would now refuse
    and cost an attempt, so the envelope permits one or two. What must not
    happen is the second card being used, so the guard records the whole
    allocation and the pinned device beside it.
    """
    outcome = report(
        gpu_count=2,
        device_names=["Tesla T4", "Tesla T4"],
        device_vram_bytes=[15636037632, 15636037632],
    )

    assert outcome["passed"], outcome["failures"]
    assert outcome["gpu_count"] == 2
    assert outcome["device_names"] == ["Tesla T4", "Tesla T4"]
    assert outcome["device_vram_bytes"] == [15636037632, 15636037632]
    assert outcome["training_device"] == "cuda:0"


def test_a_second_visible_device_that_is_not_a_t4_refuses() -> None:
    """R-0011 recorded the model of device 0 alone, so a mixed allocation was
    invisible. Every visible card is now checked by name."""
    outcome = report(
        gpu_count=2,
        device_names=["Tesla T4", "Tesla P100-PCIE-16GB"],
        device_vram_bytes=[15636037632, 17071734784],
    )

    assert not outcome["passed"]
    assert any("cuda:1" in line and "P100" in line for line in outcome["failures"])


def test_a_second_worker_may_train_on_the_second_visible_card_and_nowhere_else() -> None:
    """D-0045: two cards are two isolated workers. cuda:1 is a worker's device when two
    are visible; cuda:2 is nobody's, and a name that is not one device refuses."""
    two = dict(gpu_count=2, device_names=["Tesla T4", "Tesla T4"], device_vram_bytes=[15636037632] * 2)

    assert report(**two, training_device="cuda:1")["passed"]
    outside = report(**two, training_device="cuda:2")
    assert not outside["passed"]
    assert any("pinned to one visible device" in line for line in outside["failures"])
    spread = report(**two, training_device="cuda")
    assert not spread["passed"]
    one = report(
        gpu_count=1, device_names=["Tesla T4"], device_vram_bytes=[15636037632], training_device="cuda:1"
    )
    assert not one["passed"]


def test_a_vram_reading_per_visible_device_is_required() -> None:
    """Recording less hardware than was checked would leave the manifest unable
    to say what the run actually ran on."""
    outcome = report(gpu_count=2, device_names=["Tesla T4", "Tesla T4"], device_vram_bytes=[15636037632])

    assert not outcome["passed"]
    assert any("VRAM reading" in line for line in outcome["failures"])


def test_a_local_exercise_skips_the_hardware_checks_and_records_that_it_did() -> None:
    """Skipping is fine on a machine with no accelerator. Skipping silently is not:
    a manifest from a local exercise must not look like one whose hardware was
    actually checked."""
    outcome = report(
        gpu_count=0, device_names=[], device_vram_bytes=[], training_device="cpu", local_exercise=True
    )

    assert outcome["passed"], outcome["failures"]
    assert outcome["local_exercise"] is True
    assert any("gpu_count" in line for line in outcome["checks_skipped"])


def test_a_remote_run_cannot_acquire_the_local_exemption_by_default() -> None:
    outcome = report(gpu_count=0, device_names=[], device_vram_bytes=[], training_device="cpu")

    assert not outcome["passed"]
    assert outcome["checks_skipped"] == []


# --- E07-SMOKE-01 attempt 1: Kaggle's one-megabyte kernel source limit -------


def test_the_source_archive_is_actually_compressed() -> None:
    """`writestr` honours the ZipInfo over the archive, so the ZIP_DEFLATED the
    builder asked for was silently ignored and the source shipped stored.

    Nobody noticed until Kaggle refused the E07 package for exceeding its source
    limit ([[R-0025]]). The assertion is on the effect, not the argument.
    """
    payload = deterministic_archive(repo_source())

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = archive.infolist()
    assert entries
    assert all(entry.compress_type == zipfile.ZIP_DEFLATED for entry in entries)
    stored = sum(entry.file_size for entry in entries)
    assert len(payload) < stored / 2, (
        f"{stored} bytes of source became a {len(payload)} byte archive; it is not being compressed"
    )


def test_the_archive_is_still_byte_identical_between_builds() -> None:
    """Compression may not cost reproducibility: the runtime verifies the payload
    against a digest recorded at build time."""
    assert deterministic_archive(repo_source()) == deterministic_archive(repo_source())


def test_the_measured_source_is_what_kaggle_receives_not_the_file() -> None:
    """The Kaggle CLI strips code-cell outputs and joins each source list before
    sending, so the file on disk overstates what the limit applies to."""
    notebook = build_notebook(
        SPEC,
        shipped={**SPEC.to_dict(), "input_digests": REGISTERED},
        payload=deterministic_archive(repo_source()),
        wheelhouse_payload=WHEELHOUSE,
    )
    notebook["cells"][0]["outputs"] = [{"output_type": "stream", "text": "x" * 50_000}]

    measured = kernel_source_bytes(notebook)

    assert measured < len(json.dumps(notebook).encode("utf-8"))


def test_the_real_package_fits_under_the_kaggle_source_limit() -> None:
    """The regression that cost attempt 1 of GPU-CAMPAIGN-E07-01."""
    notebook = build_notebook(
        SPEC,
        shipped={**SPEC.to_dict(), "input_digests": REGISTERED},
        payload=deterministic_archive(repo_source()),
        wheelhouse_payload=WHEELHOUSE,
    )

    assert kernel_source_bytes(notebook) < KERNEL_SOURCE_LIMIT_BYTES
