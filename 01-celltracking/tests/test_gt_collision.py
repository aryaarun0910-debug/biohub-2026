"""The gt_to_sub overwrite guard shared by the D0P/H0c/H0b/H1a/D0' lineage.

The five sites used to do a bare ``gt_to_sub[int(mid)] = s``. Under a one-to-one
``DistanceMatching`` that cannot collide, so the guard must be a strict no-op on any
injective match; the collision case must be loud rather than silently last-writer-wins.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))
from gt_collision import gt_maps_from_matches  # noqa: E402


def _legacy(sub_to_int, int_to_gt):
    """The exact unguarded loop the five sites used to run (last writer wins)."""
    gt_to_sub, sub_to_gt = {}, {}
    for s, iid in sub_to_int.items():
        mid = int_to_gt.get(iid)
        if mid not in (None, -1):
            gt_to_sub[int(mid)] = s
            sub_to_gt[s] = int(mid)
    return gt_to_sub, sub_to_gt


def test_injective_match_is_identical_to_the_unguarded_loop():
    sub_to_int = {10: 0, 11: 1, 12: 2, 13: 3}
    int_to_gt = {0: 100, 1: 101, 2: 102, 3: 103}
    got_gt, got_sub, n = gt_maps_from_matches(sub_to_int, int_to_gt)
    assert (got_gt, got_sub) == _legacy(sub_to_int, int_to_gt)
    assert n == 0


def test_unmatched_sentinels_are_dropped_exactly_as_before():
    sub_to_int = {10: 0, 11: 1, 12: 2, 13: 3}
    int_to_gt = {0: 100, 1: -1, 2: None, 3: 103}
    got_gt, got_sub, n = gt_maps_from_matches(sub_to_int, int_to_gt)
    assert (got_gt, got_sub) == _legacy(sub_to_int, int_to_gt)
    assert got_gt == {100: 10, 103: 13}
    assert n == 0


def test_missing_internal_id_is_treated_as_unmatched():
    sub_to_int = {10: 0, 11: 99}
    int_to_gt = {0: 100}  # internal id 99 never matched at all
    got_gt, got_sub, n = gt_maps_from_matches(sub_to_int, int_to_gt)
    assert (got_gt, got_sub) == _legacy(sub_to_int, int_to_gt)
    assert n == 0


def test_collision_raises_instead_of_clobbering():
    sub_to_int = {10: 0, 11: 1}
    int_to_gt = {0: 100, 1: 100}  # two predicted nodes claim GT 100
    with pytest.raises(AssertionError, match="mother-collision"):
        gt_maps_from_matches(sub_to_int, int_to_gt, context="unit")
    # the legacy loop silently kept the LAST writer; that is the behaviour being killed
    assert _legacy(sub_to_int, int_to_gt)[0] == {100: 11}


def test_allow_collisions_quantifies_and_keeps_the_first_writer():
    sub_to_int = {10: 0, 11: 1, 12: 2}
    int_to_gt = {0: 100, 1: 100, 2: 101}
    gt_to_sub, sub_to_gt, n = gt_maps_from_matches(sub_to_int, int_to_gt, allow_collisions=True)
    assert n == 1
    assert gt_to_sub == {100: 10, 101: 12}
    assert sub_to_gt == {10: 100, 12: 101}  # the loser is not half-inserted


def test_context_appears_in_the_message():
    with pytest.raises(AssertionError, match="h0c_replay 44b6_dead"):
        gt_maps_from_matches({1: 0, 2: 1}, {0: 7, 1: 7}, context="h0c_replay 44b6_dead")
