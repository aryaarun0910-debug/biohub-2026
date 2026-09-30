from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

from reach_oracle import (  # noqa: E402
    ReachRefusal,
    partition_reached_cat3,
    reconcile_reach_partition,
)


def _edge(n: int, recovered: bool) -> dict:
    return {
        "peak_source": n,
        "peak_target": n + 1,
        "final_source": n + 10,
        "final_target": n + 11,
        "already_in_final_graph": recovered,
    }


def test_reach_partition_splits_genuinely_lost_from_already_recovered() -> None:
    lost = _edge(1, False)
    recovered = _edge(20, True)

    genuinely_lost, already_recovered = partition_reached_cat3([lost, recovered])

    assert genuinely_lost == [lost]
    assert already_recovered == [recovered]
    assert len(genuinely_lost) + len(already_recovered) == 2


def test_reach_partition_empty_input_is_an_exhaustive_empty_partition() -> None:
    assert partition_reached_cat3([]) == ([], [])


@pytest.mark.parametrize("bad_state", [None, 0, 1, "false"])
def test_reach_partition_refuses_non_boolean_final_state(bad_state: object) -> None:
    edge = _edge(1, False)
    edge["already_in_final_graph"] = bad_state

    with pytest.raises(ReachRefusal, match="non-boolean"):
        partition_reached_cat3([edge])


def test_reach_partition_refuses_missing_final_state() -> None:
    edge = _edge(1, False)
    del edge["already_in_final_graph"]

    with pytest.raises(ReachRefusal, match="no deployed final-graph state"):
        partition_reached_cat3([edge])


def test_reach_partition_refuses_duplicate_identity_even_if_labels_disagree() -> None:
    lost = _edge(1, False)
    duplicate_recovered = {**lost, "already_in_final_graph": True}

    with pytest.raises(ReachRefusal, match="duplicate reached category-3 edge identity"):
        partition_reached_cat3([lost, duplicate_recovered])


def test_reach_partition_refuses_incomplete_identity() -> None:
    edge = _edge(1, False)
    del edge["final_target"]

    with pytest.raises(ReachRefusal, match="missing identity fields"):
        partition_reached_cat3([edge])


def test_reach_aggregation_accepts_only_exhaustive_split() -> None:
    reconcile_reach_partition(9, 6, 3, label="fold")


@pytest.mark.parametrize(
    ("total", "lost", "recovered", "message"),
    [
        (9, 6, 2, "does not reconcile"),
        (9, -1, 10, "non-negative integers"),
        (9, True, 8, "non-negative integers"),
    ],
)
def test_reach_aggregation_refuses_invalid_split(
        total: int, lost: int, recovered: int, message: str) -> None:
    with pytest.raises(ReachRefusal, match=message):
        reconcile_reach_partition(total, lost, recovered, label="fold")
