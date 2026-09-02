"""Clearance is a separate act from identity, and it must survive.

Hashing answers what the data is. Clearance answers whether it may be used, and
nothing about a digest establishes it. The two are recorded independently so
neither can be mistaken for the other, and a cleared status has to name the
evidence of the review that produced it.

Licence, access restrictions and eligibility are three separate questions. A
permissive licence on the bytes says nothing about who may hold them, and
neither says anything about whether a competition's own rules allow their use.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from biohubx.artifacts import ArtifactProvenance, ProvenanceStatus, load_artifact_registry
from biohubx.cli import app, repository_root

runner = CliRunner()

COMPLETE_REVIEW: dict[str, object] = {
    "status": "external_cleared",
    "source_url": "https://www.kaggle.com/competitions/example",
    "data_license": "CC0-1.0",
    "access_restrictions": "not to be redistributed to non-participants",
    "competition_eligible": True,
    "reviewed_by": "a-named-person",
}


# --- a cleared status must name what was checked ---------------------------


def test_a_complete_review_is_accepted() -> None:
    # The positive control: the guard must not simply refuse every clearance.
    provenance = ArtifactProvenance.model_validate(COMPLETE_REVIEW)
    assert provenance.status is ProvenanceStatus.EXTERNAL_CLEARED


@pytest.mark.parametrize(
    "omitted",
    ["source_url", "data_license", "access_restrictions", "competition_eligible", "reviewed_by"],
)
def test_a_clearance_missing_any_piece_of_evidence_is_refused(omitted: str) -> None:
    incomplete = dict(COMPLETE_REVIEW)
    del incomplete[omitted]
    with pytest.raises(ValidationError, match="external_cleared asserts"):
        ArtifactProvenance.model_validate(incomplete)


def test_the_refusal_names_what_is_missing() -> None:
    with pytest.raises(ValidationError) as caught:
        ArtifactProvenance.model_validate({"status": "external_cleared"})
    message = str(caught.value)
    for expected in ("source_url", "access_restrictions", "competition_eligible", "reviewed_by"):
        assert expected in message
    assert "external_uncleared" in message, "the refusal should say what to record instead"


def test_an_uncleared_status_claims_nothing_and_needs_nothing() -> None:
    # Uncleared is the honest default. Requiring evidence for it would push
    # people to assert a review they have not done.
    provenance = ArtifactProvenance.model_validate({"status": "external_uncleared"})
    assert provenance.status is ProvenanceStatus.EXTERNAL_UNCLEARED
    assert provenance.competition_eligible is None


def test_a_data_licence_is_not_a_code_or_weight_licence() -> None:
    """Three separate fields because they are three separate facts.

    For a dataset the licence that matters is on the data. A code licence would
    be answering a question nobody asked, and accepting it in place of a data
    licence would let a cleared record cite irrelevant evidence.
    """
    fields = set(ArtifactProvenance.model_fields)
    assert {"code_license", "weight_license", "data_license"} <= fields
    assert "access_restrictions" in fields


def test_a_licence_alone_does_not_clear_an_artifact() -> None:
    # The specific confusion this guards against: competition data can be CC0
    # and still carry a rule against passing it to non-participants.
    licence_only = {
        "status": "external_cleared",
        "source_url": "https://www.kaggle.com/competitions/example",
        "data_license": "CC0-1.0",
        "reviewed_by": "a-named-person",
        "competition_eligible": True,
    }
    with pytest.raises(ValidationError, match="access_restrictions"):
        ArtifactProvenance.model_validate(licence_only)


# --- clearance survives re-registration ------------------------------------


def official_root(base: Path) -> Path:
    train = base / "train"
    (train / "d1.zarr" / "0").mkdir(parents=True)
    (train / "d1.zarr" / "0" / "chunk").write_bytes(b"volume-bytes")
    (train / "d1.geff" / "nodes").mkdir(parents=True)
    (train / "d1.geff" / "nodes" / "ids").write_bytes(b"node-ids")
    return base


def test_re_registration_does_not_downgrade_a_recorded_clearance(tmp_path: Path) -> None:
    """The risk this closes.

    Someone reviews the terms and records a clearance. The data is later
    fingerprinted again, and the fingerprint always reports external_uncleared
    because hashing establishes no terms. If registration merged that in, a
    completed review would be silently erased by a routine re-run.
    """
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

    # A human records the review, by hand, as a separate act.
    document = yaml.safe_load(registry.read_text(encoding="utf-8"))
    document["artifacts"][0]["provenance"] = dict(COMPLETE_REVIEW)
    cleared_id = document["artifacts"][0]["id"]
    registry.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    assert load_artifact_registry(registry)  # the edit is valid

    # The same data is fingerprinted and registered again.
    assert runner.invoke(app, ["data", "fingerprint", "--root", str(root)]).exit_code == 0
    assert (
        runner.invoke(
            app, ["artifacts", "register", "--from", str(report), "--registry", str(registry)]
        ).exit_code
        == 0
    )

    after = {record.id: record for record in load_artifact_registry(registry).artifacts}
    assert after[cleared_id].provenance.status is ProvenanceStatus.EXTERNAL_CLEARED
    assert after[cleared_id].provenance.reviewed_by == "a-named-person"
    assert after[cleared_id].provenance.access_restrictions


def test_a_fingerprint_never_claims_a_clearance(tmp_path: Path) -> None:
    # Byte identity establishes no terms, so the record it produces must not
    # imply a review. This is asserted at the source rather than trusted.
    root = official_root(tmp_path / "data")
    assert runner.invoke(app, ["data", "fingerprint", "--root", str(root)]).exit_code == 0
    import json

    report = json.loads((repository_root() / "artifacts/data-fingerprint.json").read_text(encoding="utf-8"))
    for entry in report["entries"]:
        assert entry["provenance"]["status"] == "external_uncleared"
        assert "competition_eligible" not in entry["provenance"]
