"""Pooled-utility decision kernel for the Biohub cell-tracking metric.

ONE closed form, four exact partials, two operating rules (component RETAIN/DELETE and
division fork ADMIT/ABSTAIN).  Nothing here is fitted; every coefficient is an exact
derivative or finite difference of the organiser's scorer.

Verified identity (scripts/agent5_ledger.py --verify, 7.1e-15 per crop; corpus 2.27e-13):

    pooled(S) = NUM / DEN + 0.1 * DTP / (DTP + DFP + DFN)
    NUM = SUM_i tp_i * (1 - 0.1 * r_i)      r_i = (N_pred_i - N_est_i) / N_est_i  (SIGNED)
    DEN = SUM_i (tp_i + fp_i + fn_i) = SUM_i (E_i + fp_i)     E_i = gt_num_edges_i (const)

If this module is adopted it should live at  src/biotrack/decision.py  (it imports nothing
from the repo and has no side effects).
"""
from __future__ import annotations

from dataclasses import dataclass


# ----------------------------------------------------------------------------------
# global state of the pooled objective
# ----------------------------------------------------------------------------------
@dataclass
class PooledState:
    """The five running counters that fully determine the pooled composite."""

    NUM: float      # SUM_i tp_i * (1 - 0.1 r_i)
    DEN: float      # SUM_i (tp_i + fp_i + fn_i)
    DTP: int
    DFP: int
    DFN: int

    @property
    def edge(self) -> float:
        return self.NUM / self.DEN

    @property
    def ddiv(self) -> int:
        return self.DTP + self.DFP + self.DFN

    @property
    def divJ(self) -> float:
        return self.DTP / self.ddiv if self.ddiv else 0.0

    @property
    def score(self) -> float:
        return self.edge + 0.1 * self.divJ

    # -- exact partials at the current point (agent5_utility.py agrees) -------------
    @property
    def dS_dNUM(self) -> float:
        return 1.0 / self.DEN

    @property
    def dS_dDEN(self) -> float:
        return -self.NUM / self.DEN ** 2          # == -edge / DEN

    def dS_dDTP_exact(self, k: int = 1) -> float:
        """Finite difference of moving k divisions FN -> TP (D_div invariant)."""
        d = self.ddiv
        return 0.1 * ((self.DTP + k) / d - self.DTP / d) if d else 0.0

    def dS_dDFP_exact(self, m: int = 1) -> float:
        """Finite difference of adding m division FPs (D_div grows by m)."""
        d = self.ddiv
        return 0.1 * (self.DTP / (d + m) - self.DTP / d) if d else 0.0


# ----------------------------------------------------------------------------------
# 1. component RETAIN vs DELETE
# ----------------------------------------------------------------------------------
def component_retain_delta(
    st: PooledState,
    n: int, e_valid: int, g: int,
    r_i: float, n_est_i: float, tp_i: int,
    theft: float = 0.0, lam: float = 2.0,
) -> dict:
    """Exact pooled delta of RETAINING one component that would otherwise be deleted.

    n        nodes added back to N_pred_i
    e_valid  its edges that count as *valid predicted edges* (>=1 matched endpoint with
             the relevant GT degree); off-annotation edges are metric-free and excluded
    g        of those, how many are true GT edges (become edge TP)
    r_i      the crop's SIGNED node ratio BEFORE the retention
    n_est_i  the crop's GEFF estimated_number_of_nodes (a constant of the crop)
    tp_i     the crop's edge TP before the retention
    theft    P(a re-added node displaces a true bipartite match)
    lam      TP edges destroyed per displaced match (mid-track node -> 2)

    Returned deltas are EXACT under the closed form (linear in the counters, then one
    exact ratio), not a linearisation.
    """
    w_i = 1.0 - 0.1 * r_i                       # count multiplier of THIS crop
    d_tp_own = g
    d_fp_own = e_valid - g

    # second order: stolen matches turn TP -> FP elsewhere in the same crop
    stolen = theft * n * lam
    d_tp = d_tp_own - stolen
    d_fp = d_fp_own + stolen

    # NUM: g arrives at weight w_i; every added node depresses the whole crop's tp
    d_num = d_tp * w_i - 0.1 * (n / n_est_i) * (tp_i + d_tp)
    d_den = d_fp                                # tp+fn == E_i is invariant

    new = PooledState(st.NUM + d_num, st.DEN + d_den, st.DTP, st.DFP, st.DFN)
    return {
        "d_tp": d_tp, "d_fp": d_fp, "d_num": d_num, "d_den": d_den,
        "d_pooled_exact": new.score - st.score,
        "d_pooled_linear": (d_num * st.dS_dNUM + d_den * st.dS_dDEN),
        "w_i": w_i,
        "count_cost_per_node": 0.1 * tp_i / n_est_i,
    }


def breakeven_g_over_n(
    st: PooledState, r_i: float, n_est_i: float, tp_i: int,
    ev_over_n: float = 1.0, theft: float = 0.0, lam: float = 2.0,
) -> float:
    """First/second-order break-even correctness rate g/n for RETAIN.

        (g/n)* = [ Jbar * (e_valid/n) + 0.1 * tp_i/n_est_i ] / (w_i + Jbar)  +  theft*lam

    Below it, delete; above it, retain.  Jbar = NUM/DEN = pooled ADJUSTED edge Jaccard.
    """
    jbar = st.edge
    w_i = 1.0 - 0.1 * r_i
    return (jbar * ev_over_n + 0.1 * tp_i / n_est_i) / (w_i + jbar) + theft * lam


def cancelling_theft(
    st: PooledState, g_over_n: float, r_i: float, n_est_i: float, tp_i: int,
    ev_over_n: float = 1.0, lam: float = 2.0,
) -> float:
    """theta* : the per-node match-theft probability at which the first-order gain of
    retention is exactly cancelled.   theta* = (g/n - (g/n)*_firstorder) / lam."""
    b0 = breakeven_g_over_n(st, r_i, n_est_i, tp_i, ev_over_n, theft=0.0)
    return (g_over_n - b0) / lam


# ----------------------------------------------------------------------------------
# 2. division fork ADMIT vs ABSTAIN
# ----------------------------------------------------------------------------------
def fork_admit_delta(
    st: PooledState,
    p: float, pi_vis: float,
    e_dtp: float, e_dfp: float,
    w_i: float,
) -> float:
    """Expected pooled delta of admitting ONE candidate fork at the current state.

    p        P(this mother/daughter-pair is the true division)
    pi_vis   P(the mother matches an annotated GT node | not a true division)  -- the
             only route by which a wrong fork becomes a division FP
    e_dtp    expected EDGE tp delta of the graph edit (add-replace steals a parent edge)
    e_dfp    expected EDGE fp delta
    w_i      1 - 0.1 r_i for the crop the candidate lives in
    """
    edge = e_dtp * w_i * st.dS_dNUM + e_dfp * st.dS_dDEN
    div = p * st.dS_dDTP_exact(1) + (1.0 - p) * pi_vis * st.dS_dDFP_exact(1)
    return edge + div


def fork_precision_threshold(
    st: PooledState, pi_vis: float, e_dtp: float, e_dfp: float, w_i: float,
) -> float:
    """MARGINAL precision p* at which admitting one more fork is break-even, evaluated at
    the CURRENT (DTP, DFP, DFN) -- not at the empty graph.  Solve fork_admit_delta = 0."""
    gtp = st.dS_dDTP_exact(1)
    cfp = st.dS_dDFP_exact(1)                    # negative
    edge = e_dtp * w_i * st.dS_dNUM + e_dfp * st.dS_dDEN
    num = -edge - pi_vis * cfp
    den = gtp - pi_vis * cfp
    return num / den if den else float("nan")


def state_after(st: PooledState, k: int, m: int, d_num: float = 0.0,
                d_den: float = 0.0) -> PooledState:
    """State after recovering k divisions (FN->TP) and incurring m division FPs."""
    return PooledState(st.NUM + d_num, st.DEN + d_den,
                       st.DTP + k, st.DFP + m, st.DFN - k)
