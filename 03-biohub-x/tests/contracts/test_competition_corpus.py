"""What the registered competition corpus actually is.

These read the committed registry, not the data, so they run on a machine that
holds no competition bytes at all. They exist because three properties of this
corpus are easy to assume wrongly, and assuming any of them wrongly produces a
number that looks like a result and is not one.

If the corpus ever changes, for instance because a genuine held-out test set
arrives, these fail loudly. That is the point: the situation would have changed
and every split decision resting on it would need revisiting.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from biohubx.artifacts import ArtifactRecord, load_artifact_registry

REPO_ROOT = Path(__file__).resolve().parents[2]


def competition_artifacts() -> list[ArtifactRecord]:
    registry = load_artifact_registry(REPO_ROOT / "registry/artifacts.yaml")
    return [a for a in registry.artifacts if a.kind.startswith("competition_dataset")]


def decompose(record: ArtifactRecord) -> tuple[str, str, str]:
    """Split an id of the form ``competition.<split>.<dataset>.<suffix>``."""
    _, split, dataset, suffix = record.id.split(".", 3)
    return split, dataset, suffix


def test_the_corpus_is_registered_at_all() -> None:
    # A positive control. Every assertion below would pass vacuously on an
    # empty registry, which is exactly the state before onboarding.
    artifacts = competition_artifacts()
    assert len(artifacts) > 100, "the competition corpus is not registered"


def test_the_local_test_split_carries_no_ground_truth() -> None:
    ground_truth_splits = {decompose(a)[0] for a in competition_artifacts() if decompose(a)[2] == "geff"}
    assert ground_truth_splits == {"train"}, (
        "ground truth appeared outside the train split; the layout is not what the split policy assumes"
    )


def test_every_local_test_volume_duplicates_a_train_volume() -> None:
    """The local test split is not a held-out set.

    Every volume in it is byte-identical to a train volume carrying the same
    dataset id. It is a format example for the submission pipeline, not an
    evaluation set, and the genuinely held-out data is the hidden set swapped in
    at rerun. Any score computed against local test/ is a score on training data.
    """
    artifacts = competition_artifacts()
    by_digest: dict[str, list[str]] = defaultdict(list)
    for record in artifacts:
        if decompose(record)[2] == "zarr":
            by_digest[record.digests["tree"]].append(record.id)

    test_volumes = [a for a in artifacts if decompose(a)[:2][0] == "test" and decompose(a)[2] == "zarr"]
    assert test_volumes, "no test volumes registered"

    for record in test_volumes:
        _, dataset, _ = decompose(record)
        twins = [other for other in by_digest[record.digests["tree"]] if other != record.id]
        assert twins, f"{record.id} is NOT a duplicate of a train volume; the corpus has changed"
        assert any(f"competition.train.{dataset}.zarr" == twin for twin in twins), (
            f"{record.id} duplicates {twins} but not the train volume of the same dataset id"
        )


def test_the_supervised_corpus_spans_only_two_embryos() -> None:
    """The hard limit on every generalisation claim this project can make.

    Dataset ids are ``{embryo}_{field_of_view}``, and there are two embryos.
    Leave-one-embryo-out therefore yields exactly two folds, each training on a
    single embryo. Any statement about holding up across embryos rests on n=2,
    and 199 movies is a count of fields of view, not of independent subjects.
    """
    embryos = {
        decompose(a)[1].split("_")[0]
        for a in competition_artifacts()
        if decompose(a)[0] == "train" and decompose(a)[2] == "geff"
    }
    assert embryos == {"44b6", "6bba"}, (
        f"the embryo structure changed: found {sorted(embryos)}. Every split policy and every "
        "cross-embryo claim needs revisiting"
    )


def test_ground_truth_is_a_vanishing_fraction_of_the_volumes() -> None:
    """Annotation is sparse by orders of magnitude, not by a little.

    Recorded so that no component is written on the assumption that most cells
    carry a label, and so that the node-count behaviour measured in F-0004 is
    read against the real ratio rather than a guessed one.
    """
    artifacts = competition_artifacts()
    volume_bytes = sum(a.shape.total_bytes for a in artifacts if decompose(a)[2] == "zarr" and a.shape)
    truth_bytes = sum(a.shape.total_bytes for a in artifacts if decompose(a)[2] == "geff" and a.shape)
    assert volume_bytes > 0 and truth_bytes > 0
    assert truth_bytes / volume_bytes < 0.001, (
        "ground truth is a larger share of the corpus than recorded; the sparsity "
        "assumption behind the split and calibration policy has changed"
    )


def test_no_competition_artifact_is_cleared_for_use() -> None:
    """The gate, restored after a clearance was withdrawn.

    A review was recorded on 2026-09-02 and withdrawn the same day: the
    eligibility confirmation had not come from a person able to join the
    competition, and the access restrictions had been paraphrased rather than
    read from the rules page. Uncleared is the honest state, because it asserts
    only that no review stands rather than asserting a negative conclusion
    nobody reached.
    """
    uncleared = {"external_uncleared"}
    statuses = {a.provenance.status.value for a in competition_artifacts()}
    assert statuses == uncleared, (
        f"a competition artifact claims a status other than external_uncleared: {sorted(statuses)}"
    )


def test_no_artifact_claims_eligibility_without_a_named_reviewer() -> None:
    """The invariant that holds in both states, cleared or not.

    Whatever the gate says today, an eligibility decision may never exist
    without a person attached to it. This is the check that would have caught
    the withdrawn clearance had the attestation been absent rather than
    misattributed.
    """
    offenders = [
        a.id
        for a in competition_artifacts()
        if a.provenance.competition_eligible is not None and not a.provenance.reviewed_by
    ]
    assert not offenders, f"eligibility recorded with no accountable reviewer: {offenders}"


def test_the_withdrawal_is_visible_on_the_record() -> None:
    # The registry must not look as though no clearance was ever attempted. A
    # silent revert would lose the fact that a wrong claim was once recorded.
    withdrawn = [a for a in competition_artifacts() if "withdrawn" in (a.provenance.note or "")]
    assert len(withdrawn) == len(competition_artifacts()), (
        "the withdrawn review is not recorded on every affected artifact"
    )


def test_identity_survived_the_withdrawal() -> None:
    # Withdrawing a review changes what was said about the terms, never what the
    # artifact is. Every record still carries its tree digest and shape.
    for record in competition_artifacts():
        assert record.is_tree
        assert record.digests["tree"].startswith("tree_sha256:sha256/v1:")
        assert record.shape is not None and record.shape.file_count > 0
