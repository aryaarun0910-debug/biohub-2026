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


def test_the_retry_is_a_separate_row_with_no_status_until_it_runs() -> None:
    registry = experiments()
    assert RETRY in registry, "the retry must have its own identity"
    retry = registry[RETRY]
    assert retry["attempt"] == 2
    assert "status" not in retry, "a status is a result; the retry has not run"
    assert retry["id"] != registry[FIRST_ATTEMPT]["id"]


def test_the_two_attempts_share_a_kernel_slug_and_nothing_else() -> None:
    """The remote name is not an identity. Two attempts, one slug, two records."""
    registry = experiments()
    assert registry[RETRY]["kernel"] == "aryaarun07/biohub-x-e03-fold-44b6"
    assert registry[FIRST_ATTEMPT].get("status") != registry[RETRY].get("status")


def test_the_registered_log_of_the_failed_attempt_still_matches_its_bytes() -> None:
    """The evidence for `invalid` is the log. If it moves, the record is unsupported."""
    entry = experiments()[FIRST_ATTEMPT]
    assert entry["retrieved"]["kernel_log_digest"] == RECORDED_LOG_DIGEST

    log = REPO / entry["retrieved"]["kernel_log"]
    if not log.is_file():
        pytest.skip("the retrieved kernel log is not on this machine")
    actual = f"raw_artifact_sha256:sha256:{hashlib.sha256(log.read_bytes()).hexdigest()}"
    assert actual == RECORDED_LOG_DIGEST, "the registered log no longer matches the file"


def test_the_failed_attempt_produced_no_checkpoint_to_claim() -> None:
    entry = experiments()[FIRST_ATTEMPT]
    assert entry["retrieved"]["checkpoint"] == "not produced"
    assert entry["retrieved"]["result_manifest"] == "not produced"


def test_the_decision_record_states_one_lost_run_not_two() -> None:
    """A correction that is itself wrong is worse than the mistake it replaced."""
    decisions = (REPO / "DECISIONS.md").read_text(encoding="utf-8")
    assert "One remote run has been lost, not two" in decisions
    assert "Two GPU sessions have now been lost" not in decisions
