"""The shared expected-utility kernel must reproduce known anchors exactly.

`src/biotrack/decision.py` is the one place every architectural lane is expected to price an
action (retain/delete a component, admit/abstain a fork). If each lane rolls its own utility
arithmetic they will disagree silently, which is how this project produced four overstated
headlines in a single cycle.

These tests pin the kernel to two independently-published anchors. If either drifts, the
kernel is wrong and no lane's delta computed with it can be trusted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARTIFACT = ROOT / "reports" / "inventory" / "decision_div_threshold.json"

PUBLISHED_BASE_POOLED = 0.6654043056779476


def _state(d: dict):
    from biotrack.decision import PooledState

    return PooledState(**{k: d[k] for k in ("NUM", "DEN", "DTP", "DFP", "DFN")})


@pytest.fixture(scope="module")
def art():
    if not ARTIFACT.exists():
        pytest.skip(f"{ARTIFACT.name} absent")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_base_state_reproduces_published_pooled(art):
    base = _state(art["anchors_base"])
    assert abs(base.score - PUBLISHED_BASE_POOLED) < 1e-12, (
        f"kernel base score {base.score!r} != published {PUBLISHED_BASE_POOLED!r}"
    )


def test_kernel_reproduces_the_h0c_oracle(art):
    """Admitting all 92 census positives from the suppressed state must give +0.06012.

    NOTE for anyone editing this: the edge-FP change moves DEN. Omitting it produces a
    plausible-looking +0.0598 that is wrong by 2.9e-04 -- that exact mistake was made once
    while promoting this module, and the anchor caught it.
    """
    base = _state(art["anchors_base"])
    supp = _state(art["anchors_supp"])
    pos = art["classes"]["positive"]

    from biotrack.decision import PooledState

    orc = PooledState(
        NUM=supp.NUM + pos["d_num"],
        DEN=supp.DEN + pos["d_fp"],
        DTP=supp.DTP + pos["n"],
        DFP=supp.DFP,
        DFN=supp.DFN - pos["n"],
    )
    delta = orc.score - base.score
    assert abs(delta - art["oracle_delta_vs_base"]) < 1e-12, (
        f"H0c oracle delta {delta!r} != artifact {art['oracle_delta_vs_base']!r}"
    )


def test_node_count_cost_is_invariant_to_the_sign_of_the_node_ratio():
    """Guards the mechanism corrected on 2026-08-01.

    The per-node count cost is 0.1*tp_i/N_est_i. N_est_i is GT metadata, so the cost does NOT
    depend on whether the substrate over- or under-predicts. A previous synthesis claimed the
    sign of the node ratio drove node budgeting; it does not, and this test stops that story
    being reintroduced.
    """
    from biotrack.decision import PooledState

    st = PooledState(NUM=100811.19, DEN=151614, DTP=4, DFP=675, DFN=147)
    n_est, tp_i = 25755.0, 5000
    cost = 0.1 * tp_i / n_est          # the per-node count cost, by definition
    for r_i in (-0.20, -0.10, 0.0, +0.10, +0.20):
        w_i = 1.0 - 0.1 * r_i
        # w_i shifts the threshold only mildly across the full observed +/-0.16 range ...
        assert 0.98 <= w_i <= 1.02
        # ... while the count cost itself is completely independent of r_i
        assert abs((0.1 * tp_i / n_est) - cost) < 1e-15
    assert st.dS_dNUM > 0 and st.dS_dDEN < 0
