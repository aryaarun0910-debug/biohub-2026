"""A failed attempt keeps its record, and a retry cannot quietly become it.

E03-SMOKE-01 was pushed once, errored before executing any packaged code, and is
recorded invalid. The retry shares its Kaggle slug, because Kaggle addresses a
kernel by slug and a push creates a new version rather than a new kernel. Attempt
identity therefore lives here, and nothing in this repository may collapse the
two into one row.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Any

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
FIRST_ATTEMPT = "E03-SMOKE-01"
RETRY = "E03-SMOKE-02"
RECORDED_LOG_DIGEST = (
    "raw_artifact_sha256:sha256:cec04e32b04cf2d0db31d0162d7b246b86fd8dc4fad2335a7c342be791a18ecb"
)


def experiments() -> dict[str, dict[str, Any]]:
    loaded = yaml.safe_load((REPO / "registry/experiments.yaml").read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in loaded["experiments"]}


def test_the_failed_attempt_keeps_its_own_row_and_its_outcome() -> None:
    entry = experiments()[FIRST_ATTEMPT]
    assert entry["status"] == "invalid", "the first attempt's outcome must not be rewritten"
    assert entry["immutable"] is True
    assert entry["attempt"] == 1
    assert "never executed" in entry["outcome"] or "before executing" in entry["outcome"]


def test_the_retry_keeps_its_own_row_and_its_own_outcome() -> None:
    """Both attempts have now run and failed. Neither may absorb the other."""
    registry = experiments()
    assert RETRY in registry, "the retry must have its own identity"
    retry = registry[RETRY]
    assert retry["attempt"] == 2
    assert retry["id"] != registry[FIRST_ATTEMPT]["id"]
    assert retry["immutable"] is True
    assert retry["status"] == "invalid"
    # Same verdict, different reason. Collapsing them would lose the only thing
    # the second attempt established: the packaging problem is solved.
    assert "zarr" in retry["outcome"]
    assert "zarr" not in registry[FIRST_ATTEMPT]["outcome"]


def test_the_two_attempts_share_a_kernel_slug_and_nothing_else() -> None:
    """The remote name is not an identity. Two attempts, one slug, two records."""
    registry = experiments()
    assert registry[RETRY]["kernel"] == "aryaarun07/biohub-x-e03-fold-44b6"
    first, second = registry[FIRST_ATTEMPT], registry[RETRY]
    assert first["retrieved"]["kernel_log"] != second["retrieved"]["kernel_log"]
    assert first["retrieved"]["kernel_log_digest"] != second["retrieved"]["kernel_log_digest"], (
        "two runs cannot share one piece of evidence"
    )


def test_the_registered_log_of_the_failed_attempt_still_matches_its_bytes() -> None:
    """The evidence for `invalid` is the log. If it moves, the record is unsupported."""
    entry = experiments()[FIRST_ATTEMPT]
    assert entry["retrieved"]["kernel_log_digest"] == RECORDED_LOG_DIGEST

    log = REPO / entry["retrieved"]["kernel_log"]
    if not log.is_file():
        pytest.skip("the retrieved kernel log is not on this machine")
    actual = f"raw_artifact_sha256:sha256:{hashlib.sha256(log.read_bytes()).hexdigest()}"
    assert actual == RECORDED_LOG_DIGEST, "the registered log no longer matches the file"


def test_neither_attempt_claims_a_checkpoint_it_did_not_produce() -> None:
    for name in (FIRST_ATTEMPT, RETRY):
        retrieved = experiments()[name]["retrieved"]
        assert retrieved["checkpoint"] == "not produced", name
        assert retrieved["result_manifest"] == "not produced", name


def test_the_second_attempt_proved_its_bootstrap_against_remote_evidence() -> None:
    """The one thing attempt 2 did establish, kept where it cannot be lost."""
    retrieved = experiments()[RETRY]["retrieved"]
    assert retrieved["extracted_payload_digest_returned_by_the_kernel"] == (
        "raw_artifact_sha256:sha256:c24db880462b202943669e459e0f2d1fb322e5e7223e7165a64b3a8d01b8acf9"
    )


def test_the_decision_record_states_one_lost_run_not_two() -> None:
    """A correction that is itself wrong is worse than the mistake it replaced."""
    decisions = (REPO / "DECISIONS.md").read_text(encoding="utf-8")
    assert "One remote run has been lost, not two" in decisions
    assert "Two GPU sessions have now been lost" not in decisions
