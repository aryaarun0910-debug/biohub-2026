"""Synthetic tests for the M/T/C/L/D partition, including the cube-corner defect.

Every case here is one the v4 smoke either got wrong or could not express.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biotrack.d1_partition import (  # noqa: E402
    MATCH_UM, SEARCH_UM, LocalMax, census, check_invariants, classify_gt,
    sphere_filter, strongest_within,
)

GRID_UM = 1.625  # isotropic output-grid step


def lm(d, logit=5.0, prob=0.99, accepted=False):
    return LocalMax(dist_um=d, logit=logit, prob=prob, accepted=accepted)


def test_accepted_peak_one_grid_voxel_away_is_M():
    """The v4 failure mode: exact GT voxel suppressed, accepted peak one voxel away.

    Under exact-voxel A/B/D this was called 'B' (suppressed). It is a MATCH.
    """
    assert classify_gt(matched=True, maxima=[lm(GRID_UM, accepted=True)]) == "M"


def test_under_threshold_local_max_within_7um_is_T():
    assert classify_gt(matched=False, maxima=[lm(3.0, prob=0.10, accepted=False)]) == "T"


def test_accepted_peak_consumed_by_another_gt_is_C():
    """An accepted peak is right there, but bipartite matching gave it to a different GT."""
    assert classify_gt(matched=False, maxima=[lm(2.0, accepted=True)]) == "C"


def test_C_takes_precedence_over_T_when_both_present():
    maxima = [lm(2.0, accepted=True), lm(3.0, prob=0.1, accepted=False)]
    assert classify_gt(matched=False, maxima=maxima) == "C"


@pytest.mark.parametrize("d", [8.0, 10.0, 14.9])
def test_maximum_between_8_and_15um_is_L(d):
    assert classify_gt(matched=False, maxima=[lm(d, accepted=True)]) == "L"


def test_no_maximum_within_15um_is_D():
    assert classify_gt(matched=False, maxima=[]) == "D"
    assert classify_gt(matched=False, maxima=[lm(15.001)]) == "D"


def test_cube_corner_beyond_15um_is_excluded():
    """A 15 um axis-aligned BOX admits 15*sqrt(3) = 25.98 um at the corner.

    v4 selected such corners: it reported near_dist p50 15.4 um and max 24.4 um for a search
    that was supposed to be 15 um. Anything beyond the SPHERE must be dropped.
    """
    corner = 15.0 * (3 ** 0.5)
    assert corner == pytest.approx(25.98, abs=0.01)
    assert sphere_filter([lm(corner)]) == []
    assert classify_gt(matched=False, maxima=[lm(corner, accepted=True)]) == "D"
    # and a corner candidate must never be chosen over a genuine in-sphere one
    inside, outside = lm(9.0, logit=1.0), lm(corner, logit=99.0)
    assert strongest_within([inside, outside], SEARCH_UM) is inside


def test_strongest_within_uses_sphere_not_cube():
    near_weak, far_strong = lm(2.0, logit=1.0), lm(20.0, logit=50.0)
    assert strongest_within([near_weak, far_strong], SEARCH_UM) is near_weak
    assert strongest_within([far_strong], MATCH_UM) is None


def test_plateau_tied_maxima_do_not_break_classification():
    """Tied logits on a plateau: classification must stay deterministic and single-valued."""
    tied = [lm(2.0, logit=7.0, accepted=True), lm(2.0, logit=7.0, accepted=True)]
    assert classify_gt(matched=False, maxima=tied) == "C"
    best = strongest_within(tied, SEARCH_UM)
    assert best is not None and best.logit == 7.0


def test_boundary_exactly_at_7um_counts_as_within_match_radius():
    assert classify_gt(matched=False, maxima=[lm(MATCH_UM, accepted=True)]) == "C"
    assert classify_gt(matched=False, maxima=[lm(MATCH_UM + 1e-9, accepted=True)]) == "L"


def test_partition_sums_to_all_gt_and_to_unmatched():
    rows = [
        (True, [lm(1.0, accepted=True)]),          # M
        (True, []),                                 # M
        (False, [lm(2.0, accepted=True)]),          # C
        (False, [lm(3.0, prob=0.1)]),               # T
        (False, [lm(9.0)]),                         # L
        (False, []),                                # D
        (False, [lm(30.0, accepted=True)]),         # D (cube corner excluded)
    ]
    c = census(rows)
    assert c == {"M": 2, "T": 1, "C": 1, "L": 1, "D": 2}
    check_invariants(c, n_gt=len(rows), n_unmatched=sum(1 for m, _ in rows if not m))


def test_check_invariants_rejects_a_miscount():
    with pytest.raises(ValueError, match="!= 99 GT"):
        check_invariants({"M": 1, "T": 0, "C": 0, "L": 0, "D": 0}, n_gt=99, n_unmatched=0)
    with pytest.raises(ValueError, match="scorer-unmatched"):
        check_invariants({"M": 1, "T": 1, "C": 0, "L": 0, "D": 0}, n_gt=2, n_unmatched=0)
