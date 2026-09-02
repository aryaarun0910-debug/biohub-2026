"""Withdrawing a terms review.

Used when a clearance turns out to rest on something that was not established:
an attestation from someone not in a position to give it, terms paraphrased
rather than read, a reviewer who did not review.

The end state is `external_uncleared`, not `not-eligible`. Recording
not-eligible would assert that a review happened and reached a negative
conclusion. Uncleared asserts only that no review stands, which is what is
actually true after a withdrawal.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from biohubx.artifacts import ProvenanceStatus, load_artifact_registry
from biohubx.cli import app, repository_root

runner = CliRunner()

REVIEW = [
    "--reviewed-by",
    "a-named-person",
    "--source-url",
    "https://www.kaggle.com/competitions/example/rules",
    "--access-restrictions",
    "not to be redistributed to non-participants",
    "--data-license",
    "CC0-1.0",
    "--eligibility",
    "eligible",
]


def official_root(base: Path) -> Path:
    train = base / "train"
    (train / "d1.zarr" / "0").mkdir(parents=True)
    (train / "d1.zarr" / "0" / "chunk").write_bytes(b"volume-bytes")
    (train / "d1.geff" / "nodes").mkdir(parents=True)
    (train / "d1.geff" / "nodes" / "ids").write_bytes(b"node-ids")
    return base


def cleared_registry(tmp_path: Path) -> Path:
    root = official_root(tmp_path / "data")
    registry = tmp_path / "artifacts.yaml"
    registry.write_text("schema_version: 1\nartifacts: []\n", encoding="utf-8")
    assert runner.invoke(app, ["data", "fingerprint", "--root", str(root)]).exit_code == 0
    report = repository_root() / "artifacts/data-fingerprint.json"
    assert (
        runner.invoke(
            app, ["artifacts", "register", "--from", str(report), "--registry", str(registry)]
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            app,
            [
                "artifacts",
                "clear",
                "--registry",
                str(registry),
                "--id-prefix",
                "competition.",
                *REVIEW,
            ],
        ).exit_code
        == 0
    )
    return registry


def revoke(registry: Path, reason: str = "the attestation was not from a person able to give it") -> int:
    return runner.invoke(
        app,
        [
            "artifacts",
            "revoke",
            "--registry",
            str(registry),
            "--id-prefix",
            "competition.",
            "--reason",
            reason,
        ],
    ).exit_code


def test_revocation_returns_every_record_to_uncleared(tmp_path: Path) -> None:
    registry = cleared_registry(tmp_path)
    assert revoke(registry) == 0
    for record in load_artifact_registry(registry).artifacts:
        assert record.provenance.status is ProvenanceStatus.EXTERNAL_UNCLEARED


def test_every_part_of_the_review_claim_is_removed(tmp_path: Path) -> None:
    # Leaving a licence or a reviewer behind would let a later reader reassemble
    # a clearance that nobody stands behind.
    registry = cleared_registry(tmp_path)
    assert revoke(registry) == 0
    for record in load_artifact_registry(registry).artifacts:
        provenance = record.provenance
        assert provenance.reviewed_by is None
        assert provenance.reviewed_utc is None
        assert provenance.competition_eligible is None
        assert provenance.data_license is None
        assert provenance.access_restrictions is None
        assert provenance.source_url is None


def test_the_withdrawal_is_recorded_rather_than_silent(tmp_path: Path) -> None:
    """The registry must not look as though no clearance was ever attempted.

    A silent revert would lose the fact that a wrong claim was once recorded,
    which is exactly the history a later reader needs.
    """
    registry = cleared_registry(tmp_path)
    assert revoke(registry, "a specific stated reason") == 0
    for record in load_artifact_registry(registry).artifacts:
        note = record.provenance.note or ""
        assert "Terms review withdrawn" in note
        assert "a specific stated reason" in note


def test_the_dataset_description_survives_the_withdrawal(tmp_path: Path) -> None:
    registry = cleared_registry(tmp_path)
    assert revoke(registry) == 0
    for record in load_artifact_registry(registry).artifacts:
        assert "Competition dataset" in (record.provenance.note or "")


def test_identity_and_shape_are_untouched(tmp_path: Path) -> None:
    # Withdrawing a review changes what was said about the terms, never what the
    # artifact is.
    registry = cleared_registry(tmp_path)
    before = {r.id: (r.digests, r.shape, r.external_path) for r in load_artifact_registry(registry).artifacts}
    assert revoke(registry) == 0
    after = {r.id: (r.digests, r.shape, r.external_path) for r in load_artifact_registry(registry).artifacts}
    assert after == before


def test_revocation_is_idempotent(tmp_path: Path) -> None:
    registry = cleared_registry(tmp_path)
    assert revoke(registry) == 0
    assert revoke(registry, "second withdrawal") == 0
    for record in load_artifact_registry(registry).artifacts:
        assert record.provenance.status is ProvenanceStatus.EXTERNAL_UNCLEARED


def test_an_unmatched_prefix_is_refused(tmp_path: Path) -> None:
    registry = cleared_registry(tmp_path)
    result = runner.invoke(
        app,
        [
            "artifacts",
            "revoke",
            "--registry",
            str(registry),
            "--id-prefix",
            "nothing-matches.",
            "--reason",
            "x",
        ],
    )
    assert result.exit_code == 2


def test_a_reason_is_required(tmp_path: Path) -> None:
    # A withdrawal with no stated reason is as unhelpful as a silent one.
    registry = cleared_registry(tmp_path)
    result = runner.invoke(
        app,
        ["artifacts", "revoke", "--registry", str(registry), "--id-prefix", "competition."],
    )
    assert result.exit_code != 0


def test_the_manifest_records_the_withdrawal(tmp_path: Path) -> None:
    registry = cleared_registry(tmp_path)
    assert revoke(registry, "a stated reason") == 0
    manifest = json.loads(
        (repository_root() / "artifacts/manifests/artifacts-revoke.json").read_text(encoding="utf-8")
    )
    assert manifest["withdrawn"] == 2
    assert manifest["reason"] == "a stated reason"


def test_a_withdrawn_artifact_can_be_cleared_again_by_a_real_review(tmp_path: Path) -> None:
    # Withdrawal is not a permanent bar. It returns the artifact to the state
    # where a genuine review can be recorded against it.
    registry = cleared_registry(tmp_path)
    assert revoke(registry) == 0
    assert (
        runner.invoke(
            app,
            [
                "artifacts",
                "clear",
                "--registry",
                str(registry),
                "--id-prefix",
                "competition.",
                *REVIEW,
            ],
        ).exit_code
        == 0
    )
    for record in load_artifact_registry(registry).artifacts:
        assert record.provenance.status is ProvenanceStatus.EXTERNAL_CLEARED
