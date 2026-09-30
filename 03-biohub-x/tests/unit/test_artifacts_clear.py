"""Recording a completed terms review.

The command writes down a review that has happened. It does not perform one and
it cannot judge eligibility, so every piece of evidence is a required option and
none of it has a default.

Two properties matter most. A review is about specific bytes, so it may not be
attached to an artifact whose recorded identity no longer holds. And the
eligibility decision must be stated explicitly, because a defaulted answer to
"may we use this" is the one answer nobody should be able to give by accident.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
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
]


def official_root(base: Path) -> Path:
    train = base / "train"
    (train / "d1.zarr" / "0").mkdir(parents=True)
    (train / "d1.zarr" / "0" / "chunk").write_bytes(b"volume-bytes")
    (train / "d1.geff" / "nodes").mkdir(parents=True)
    (train / "d1.geff" / "nodes" / "ids").write_bytes(b"node-ids")
    test = base / "test"
    (test / "d2.zarr" / "0").mkdir(parents=True)
    (test / "d2.zarr" / "0" / "chunk").write_bytes(b"held-out")
    return base


@pytest.fixture
def registered(tmp_path: Path) -> tuple[Path, Path]:
    """A registry holding three freshly fingerprinted, uncleared artifacts."""
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
    return root, registry


def clear(registry: Path, *extra: str) -> int:
    return runner.invoke(
        app,
        [
            "artifacts",
            "clear",
            "--registry",
            str(registry),
            "--id-prefix",
            "competition.",
            *REVIEW,
            *extra,
        ],
    ).exit_code


# --- the decision must be explicit -----------------------------------------


def test_eligibility_has_no_default(registered: tuple[Path, Path]) -> None:
    # A defaulted answer to whether the data may be used is the one answer
    # nobody should be able to give by accident.
    _, registry = registered
    assert clear(registry) != 0


@pytest.mark.parametrize("omit", ["--reviewed-by", "--source-url", "--access-restrictions"])
def test_every_piece_of_evidence_is_required(registered: tuple[Path, Path], omit: str) -> None:
    _, registry = registered
    partial = list(REVIEW)
    index = partial.index(omit)
    del partial[index : index + 2]
    result = runner.invoke(
        app,
        [
            "artifacts",
            "clear",
            "--registry",
            str(registry),
            "--id-prefix",
            "competition.",
            "--eligibility",
            "eligible",
            *partial,
        ],
    )
    assert result.exit_code != 0


def test_a_clearance_without_any_licence_is_refused(registered: tuple[Path, Path]) -> None:
    _, registry = registered
    partial = list(REVIEW)
    index = partial.index("--data-license")
    del partial[index : index + 2]
    result = runner.invoke(
        app,
        [
            "artifacts",
            "clear",
            "--registry",
            str(registry),
            "--id-prefix",
            "competition.",
            "--eligibility",
            "eligible",
            *partial,
        ],
    )
    assert result.exit_code == 2


# --- a completed review is recorded in full --------------------------------


def test_a_completed_review_is_recorded_against_every_selected_artifact(
    registered: tuple[Path, Path],
) -> None:
    _, registry = registered
    assert clear(registry, "--eligibility", "eligible") == 0

    for record in load_artifact_registry(registry).artifacts:
        provenance = record.provenance
        assert provenance.status is ProvenanceStatus.EXTERNAL_CLEARED
        assert provenance.reviewed_by == "a-named-person"
        assert provenance.data_license == "CC0-1.0"
        assert provenance.access_restrictions == "not to be redistributed to non-participants"
        assert provenance.competition_eligible is True
        assert provenance.reviewed_utc is not None


def test_a_not_eligible_decision_is_recorded_as_such(registered: tuple[Path, Path]) -> None:
    # Recording that something may NOT be used is as important as the reverse,
    # and must not be expressible only by omission.
    _, registry = registered
    assert clear(registry, "--eligibility", "not-eligible") == 0
    for record in load_artifact_registry(registry).artifacts:
        assert record.provenance.competition_eligible is False
        assert record.provenance.status is ProvenanceStatus.EXTERNAL_CLEARED


def test_the_identity_and_shape_survive_clearance(registered: tuple[Path, Path]) -> None:
    _, registry = registered
    before = {r.id: (r.digests, r.shape) for r in load_artifact_registry(registry).artifacts}
    assert clear(registry, "--eligibility", "eligible") == 0
    after = {r.id: (r.digests, r.shape) for r in load_artifact_registry(registry).artifacts}
    assert after == before, "clearing terms must not disturb what the artifact is"


def test_only_the_selected_prefix_is_cleared(registered: tuple[Path, Path]) -> None:
    _, registry = registered
    result = runner.invoke(
        app,
        [
            "artifacts",
            "clear",
            "--registry",
            str(registry),
            "--id-prefix",
            "competition.test.",
            "--eligibility",
            "eligible",
            *REVIEW,
        ],
    )
    assert result.exit_code == 0
    by_id = {r.id: r for r in load_artifact_registry(registry).artifacts}
    assert by_id["competition.test.d2.zarr"].provenance.status is ProvenanceStatus.EXTERNAL_CLEARED
    assert by_id["competition.train.d1.zarr"].provenance.status is ProvenanceStatus.EXTERNAL_UNCLEARED


# --- a review is about specific bytes --------------------------------------


def test_a_clearance_is_refused_when_the_data_no_longer_matches(
    registered: tuple[Path, Path],
) -> None:
    """The guard that keeps a real review honest.

    A clearance says someone read the terms covering these bytes. If the
    recorded identity no longer holds, attaching the review would bind it to
    data it was never about.
    """
    root, registry = registered
    (root / "train" / "d1.zarr" / "0" / "chunk").write_bytes(b"a-different-and-longer-payload")

    assert clear(registry, "--eligibility", "eligible") == 1
    for record in load_artifact_registry(registry).artifacts:
        assert record.provenance.status is ProvenanceStatus.EXTERNAL_UNCLEARED


def test_a_refused_clearance_writes_nothing(registered: tuple[Path, Path]) -> None:
    root, registry = registered
    before = registry.read_text(encoding="utf-8")
    (root / "train" / "d1.zarr" / "0" / "chunk").write_bytes(b"a-different-and-longer-payload")
    assert clear(registry, "--eligibility", "eligible") == 1
    assert registry.read_text(encoding="utf-8") == before


def test_an_unmatched_prefix_is_refused(registered: tuple[Path, Path]) -> None:
    _, registry = registered
    result = runner.invoke(
        app,
        [
            "artifacts",
            "clear",
            "--registry",
            str(registry),
            "--id-prefix",
            "nothing-matches-this.",
            "--eligibility",
            "eligible",
            *REVIEW,
        ],
    )
    assert result.exit_code == 2


def test_the_manifest_records_who_reviewed_and_what_they_decided(
    registered: tuple[Path, Path],
) -> None:
    _, registry = registered
    assert clear(registry, "--eligibility", "not-eligible") == 0
    manifest = json.loads(
        (repository_root() / "artifacts/manifests/artifacts-clear.json").read_text(encoding="utf-8")
    )
    assert manifest["reviewed_by"] == "a-named-person"
    assert manifest["competition_eligible"] is False
    assert manifest["cleared"] == 3


def test_the_registry_stays_loadable_and_verifiable(registered: tuple[Path, Path]) -> None:
    _, registry = registered
    assert clear(registry, "--eligibility", "eligible") == 0
    document = yaml.safe_load(registry.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert runner.invoke(app, ["artifacts", "verify", "--registry", str(registry)]).exit_code == 0


# --- prose must not outlive the fact it described --------------------------


def test_clearing_removes_a_note_that_says_the_review_is_outstanding(
    registered: tuple[Path, Path],
) -> None:
    """The defect this closes, reproduced and then prevented.

    Earlier fingerprint runs wrote the review status into the note as well as
    into the status field. Prose does not update, so once a clearance was
    recorded the note contradicted the record carrying it, and a reader could
    believe the sentence over the field.
    """
    from biohubx.artifacts import PROVISIONAL_REVIEW_NOTE

    _, registry = registered
    document = yaml.safe_load(registry.read_text(encoding="utf-8"))
    for record in document["artifacts"]:
        record["provenance"]["note"] = f"Some description. {PROVISIONAL_REVIEW_NOTE}"
    registry.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    assert clear(registry, "--eligibility", "eligible") == 0
    for record in load_artifact_registry(registry).artifacts:
        note = record.provenance.note or ""
        assert PROVISIONAL_REVIEW_NOTE not in note
        assert "Some description." in note, "the rest of the note must survive"


def test_a_fingerprint_note_no_longer_duplicates_the_status(
    registered: tuple[Path, Path],
) -> None:
    # The status field already answers whether the terms were reviewed. Saying
    # it twice is how the two came to disagree.
    from biohubx.artifacts import PROVISIONAL_REVIEW_NOTE

    _, registry = registered
    for record in load_artifact_registry(registry).artifacts:
        assert PROVISIONAL_REVIEW_NOTE not in (record.provenance.note or "")
        assert record.provenance.status is ProvenanceStatus.EXTERNAL_UNCLEARED


def test_stripping_a_note_that_was_only_the_stale_sentence_leaves_nothing() -> None:
    from biohubx.artifacts import PROVISIONAL_REVIEW_NOTE, strip_provisional_review_note

    assert strip_provisional_review_note(PROVISIONAL_REVIEW_NOTE) is None
    assert strip_provisional_review_note(None) is None
    assert strip_provisional_review_note("kept") == "kept"
