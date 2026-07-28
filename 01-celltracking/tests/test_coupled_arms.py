"""Lock the coupled-experiment arm definitions.

These tests exist because the clean-903 port silently drifted from the retained v122
source on three wrapper constants -- one of them structural (a transductive per-embryo
density prior that v122 does not have at all). Both promotion candidates (D, C0) must
use the faithful v122 wrapper, so that drift is pinned here rather than in a comment.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import coupled_arms as CA  # noqa: E402
from biotrack import wrapper as W  # noqa: E402


def test_retained_v122_source_matches_recorded_hash():
    """The promotion wrapper's provenance must be immutable."""
    assert CA.V122_SOURCE.exists(), f"retained v122 source missing: {CA.V122_SOURCE}"
    digest = hashlib.sha256(CA.V122_SOURCE.read_bytes()).hexdigest().upper()
    assert digest == CA.V122_SOURCE_SHA256, (
        "retained v122 notebook changed; re-verify the wrapper constants before "
        "updating V122_SOURCE_SHA256"
    )


@pytest.mark.parametrize("const,expected", sorted(CA.V122_LOCKED_CONSTANTS.items()))
def test_v122_wrapper_locks_drifted_constants(const, expected):
    CA.set_v122_wrapper()
    assert getattr(W, const) == expected


def test_v122_has_no_prefix_density_prior():
    """v122's frame_local_spacing uses pure local kNN spacing. A non-zero blend here
    would inject a transductive prior the public notebook does not have."""
    CA.set_v122_wrapper()
    assert W.PREFIX_DENSITY_BLEND == 0.0
    assert W.PREFIX_DENSITY_PRIOR_UM == {}


@pytest.mark.parametrize("arm", CA.PROMOTION_ARMS)
def test_promotion_arms_use_faithful_v122_wrapper(arm):
    """D and C0 are the only promotion candidates and must both be faithful."""
    assert CA.ARMS[arm]["wrapper"] == "v122"
    CA.arm_config(arm)
    for const, expected in CA.V122_LOCKED_CONSTANTS.items():
        assert getattr(W, const) == expected, f"{arm} drifted on {const}"


def test_dport_is_diagnostic_only_and_differs_only_in_three_constants():
    CA.set_v122_wrapper()
    faithful = CA.wrapper_state()
    CA.set_v122_port_wrapper()
    ported = CA.wrapper_state()

    differing = {k for k in faithful if faithful[k] != ported[k]}
    assert differing == set(CA.V122_LOCKED_CONSTANTS), (
        f"D-port must differ from D in exactly the three drifted constants; got {differing}"
    )
    assert CA.ARMS["Dport"]["role"] == "diagnostic"
    assert "Dport" not in CA.PROMOTION_ARMS


def test_dport_shares_detector_and_ilp_with_D():
    """D-port must isolate wrapper drift only -- same detections, same ILP."""
    d, dp = CA.ARMS["D"], CA.ARMS["Dport"]
    assert d["det"] == dp["det"]
    assert d["ilp"] == dp["ilp"]
    assert d["source"] == dp["source"]


def test_arm_A_is_greedy_not_ilp():
    """oof_clean was generated with USE_ILP=False; E0c has no ILP stage."""
    assert CA.ARMS["A"]["ilp"] is None
    assert CA.ARMS["A"]["det"] == 0.990


def test_all_arm_config_hashes_are_distinct():
    hashes = {arm: CA.config_hash(arm) for arm in CA.ARMS}
    assert len(set(hashes.values())) == len(hashes), f"config hash collision: {hashes}"
