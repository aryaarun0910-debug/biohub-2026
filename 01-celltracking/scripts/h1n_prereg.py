"""H1-N preregistration — WRITTEN AND HASHED BEFORE ANY COMPOSITION OUTCOME WAS COMPUTED.

This module is the single source of truth for every rule used by `h1n_forensics.py`
and `h1n_compose.py`. It contains NO results. Its SHA-256 is printed by both scripts
and recorded in their JSON output, so any post-hoc edit to a rule is detectable.

Run standalone to emit the preregistration and its hash:
  .venv\\Scripts\\python.exe scripts\\h1n_prereg.py
"""
from __future__ import annotations

import hashlib
import json

# ---------------------------------------------------------------------------
# 0. FROZEN INHERITED CONSTANTS (not chosen here; imported from measured assets)
# ---------------------------------------------------------------------------
H0C_CFG_HASH = "04eeac97500d"          # frozen proposer, phaseb_h0c_replay.CFG
H1M_FEATURE_HASH = "8697e2a779e2"      # frozen 77-feature representation
H1M_L2 = 30.0                          # inherited from laneC run, NOT retuned
H1M_FOLDS = 5
H1M_SEED = 17
H1M_MODEL = "logistic"                 # primary arm; GBDT already lost at the tail

GT_DIVISIONS = {"44b6": 26, "6bba": 125}
POOLED_BASE = 0.6654043056779474

# ---------------------------------------------------------------------------
# 1. ERROR TAXONOMY -- mutually exclusive, applied in this PRIORITY ORDER
# ---------------------------------------------------------------------------
TAXONOMY = [
    ("SUBSTRATE",
     "admitted node IS a true GT divider (label==1) but its true daughter pair is NOT on "
     "the frozen H0c top-3 shortlist (realisable==False). No gate on this substrate can "
     "convert it; the candidate surface cannot express the answer."),
    ("TEMPORAL_PHASE",
     "admitted node is not itself a divider, but the SAME predicted track "
     "(crop, mother_track_id) carries a GT division within +/-2 frames. Right cell, "
     "wrong frame."),
    ("RANKING",
     "the admitted FP's crop contains at least one realisable true mother that was NOT "
     "admitted and that scored BELOW this FP. The model ordered the wrong cell above the "
     "right cell inside the same crop."),
    ("CALIBRATION",
     "everything else: the FP sits in a crop with no un-admitted realisable mother below "
     "it, i.e. there was no local competitor to get wrong. The admission is a pure "
     "score-scale / threshold failure across crops."),
]
TEMPORAL_PHASE_WINDOW = 2              # frames, symmetric

# ---------------------------------------------------------------------------
# 2. SIGNALS -- all oriented so HIGHER == MORE DIVIDER-LIKE. Frozen, no fitting.
# ---------------------------------------------------------------------------
SIGNALS = {
    "H1M":   "cross-fitted OOF gate score (77-feature L2 logistic, crop-grouped 5-fold)",
    "H1I":   "-R_massn_p1 (mother core mass t+1 / t). RANK-IDENTICAL to the best frozen "
             "H1-I biological signal m_massratio_p1: Spearman 1.000000 over the 48,457 "
             "mother-events where both exist. Bilateral AUC 0.1758/0.1978.",
    "RTD_A": "-(log R_massn_p1 - log R_massn_m1): forward-minus-backward core-mass "
             "asymmetry. A true mitosis loses mass ONLY forward; detector/linking noise "
             "is time-symmetric. Reverse-time DISAGREEMENT in the appearance channel.",
    "RTD_G": "rtd_margin from h1n_reverse_contest.py -- min over the mother's two rank-0 "
             "daughters of (best residual any OTHER mother on the frozen census offers "
             "that daughter) minus (this mother's own residual). Reverse-DIRECTION "
             "association disagreement: the daughters' own view of who their parent is. "
             "HIGHER == this mother wins its daughters outright.",
    "GEOM":  "-best_resid_um, the rank-0 flow-midpoint residual (frozen geometry rank). "
             "IDENTICAL to G_flow_midpoint_residual at rank 0 (max abs diff 0.0 over the "
             "48,457 joint rows). Bilateral AUC 0.1904/0.2577.",
}
# overlap matrix rule: drop the bottom HALF of the admitted set by each signal
OVERLAP_DROP_FRACTION = 0.50

# AMENDMENT 1 -- 2026-07-31, recorded BEFORE any composition A/B/C outcome existed.
# Reason: COVERAGE DEFECT, not an outcome. agent3/eval_v3/candidates.parquet (the pair-level
# H1-I + H1-G table) covers 97 of 199 crops, so 122 of the 218 recorded H1-M decisions had
# NO value for H1I / RTD_G / GEOM and the overlap matrix degenerated to Jaccard 1.0 among
# them purely through shared missingness. The three signals were re-sourced to full-coverage
# columns and the substitution was VERIFIED numerically, not assumed:
#   H1I   R_massn_p1  vs  m_massratio_p1        Spearman = 1.000000  (rank-identical)
#   GEOM  best_resid_um vs G_flow_midpoint_residual  max abs diff = 0.0 (identical)
#   RTD_G G_competing_parents>0 == steal_rank0  agreement 1.0000; the binary form is too
#         coarse to rank a bottom-half drop, so the GRADED reverse-direction margin above
#         replaces it. It is computed from the same frozen census, GT-free.
# Compositions A/B/C are UNCHANGED in substance: B's gate is the same signal up to a strictly
# monotone transform, and C's abstention rule was already written on steal_rank0.
AMENDMENTS = [{
    "id": 1, "date": "2026-07-31", "kind": "coverage defect, pre-outcome",
    "prior_hash": "01ff6ee5cfd757d3",
    "detail": "H1I/GEOM/RTD_G re-sourced from 97-crop pair table to 199-crop full-coverage "
              "columns; equivalence verified (Spearman 1.000000 / max abs diff 0.0 / "
              "agreement 1.0). RTD_G upgraded from binary competing_parents to the graded "
              "reverse-direction residual margin. No composition outcome had been computed.",
}]

# ---------------------------------------------------------------------------
# 3. THE THREE COMPOSITIONS -- preregistered, no arbitrary combination search
# ---------------------------------------------------------------------------
COMPOSITIONS = {
    "A": {
        "name": "H1-M alone",
        "rule": "Rank every complete mother-event by the cross-fitted H1-M OOF score. "
                "Admit the top K. K = round(rate * n_family) where `rate` is the "
                "argmax-divJ admission rate measured on the OTHER family. Nothing else.",
    },
    "B": {
        "name": "H1-M GATED BY the best frozen H1-I biological signal",
        "rule": "Step 1 (gate): keep only mothers whose H1I signal is above the "
                "q-quantile of H1I over the OTHER family's complete mother-events. "
                "Step 2 (rank): among survivors, admit the top K by H1-M OOF score, "
                "K = round(rate * n_survivors) with `rate` fitted on the other family "
                "exactly as in A. NESTED: both q and rate come from the other family. "
                "q is preregistered as the single value 0.50 (median split) -- no sweep.",
    },
    "C": {
        "name": "H1-M PLUS association-safety abstention",
        "rule": "Step 1 (abstain): drop any mother whose rank-0 candidate pair requires "
                "stealing a daughter from another parent (steal_rank0 == True, "
                "equivalently G_competing_parents > 0). This is GT-free and uses only "
                "the frozen E0c graph. Step 2 (rank): admit top K by H1-M OOF score "
                "among survivors, K = round(rate * n_survivors), rate from the other "
                "family. No threshold is chosen on the family being scored.",
    },
}
COMPOSITION_B_QUANTILE = 0.50
COMPOSITION_C_ABSTAIN = "steal_rank0 == True"

# ---------------------------------------------------------------------------
# 4. MEASUREMENT
# ---------------------------------------------------------------------------
MEASUREMENT = {
    "primary": "pooled composite over ALL crops (never a family average, never min-fold).",
    "surrogate": "Because a 199-crop exact replay is compute-gated, the pooled composite "
                 "is reported through a 2-parameter surrogate CALIBRATED AND VALIDATED on "
                 "the 11 exactly-measured arms in reports/inventory/pooled_breakeven.json. "
                 "The surrogate's max abs residual over those arms is reported alongside "
                 "every composition number. It is labelled SURROGATE everywhere.",
    "exact": "scripts/h1n_exact_replay.py reproduces the frozen H0c cascade but drives "
             "reconstruction from an ADMISSION LIST instead of the GT oracle. It is "
             "smoke-tested on <=3 crops here; the 199-crop command is printed and NOT run.",
    "family_role": "family numbers are DIAGNOSTIC ONLY.",
    "threshold_rule": "no threshold, quantile or budget is ever selected on the family it "
                      "is evaluated on.",
}

# ---------------------------------------------------------------------------
# 5. DECISION RULE, fixed in advance
# ---------------------------------------------------------------------------
DECISION = {
    "win": "the composition with the highest pooled surrogate composite delta.",
    "precision_target": {"+0.002": 0.07967181859385201,
                         "+0.005": 0.10153251598309981,
                         "+0.010": 0.14753626707844655},
    "reach_005_claim": "a composition may only be claimed to 'plausibly reach +0.005' if "
                       "its pooled precision-among-visible >= 0.10153 AND it retains "
                       ">= 8 of the 92 realisable positives AND the same rule ordering "
                       "holds in BOTH families under nested cross-fitting.",
}


def payload() -> dict:
    return {
        "inherited": {"h0c_cfg_hash": H0C_CFG_HASH, "h1m_feature_hash": H1M_FEATURE_HASH,
                      "l2": H1M_L2, "folds": H1M_FOLDS, "seed": H1M_SEED,
                      "model": H1M_MODEL, "gt_divisions": GT_DIVISIONS,
                      "pooled_base": POOLED_BASE},
        "taxonomy": TAXONOMY, "temporal_phase_window": TEMPORAL_PHASE_WINDOW,
        "signals": SIGNALS, "overlap_drop_fraction": OVERLAP_DROP_FRACTION,
        "compositions": COMPOSITIONS, "composition_B_quantile": COMPOSITION_B_QUANTILE,
        "composition_C_abstain": COMPOSITION_C_ABSTAIN,
        "measurement": MEASUREMENT, "decision": DECISION,
        "amendments": AMENDMENTS,
    }


def prereg_hash() -> str:
    return hashlib.sha256(
        json.dumps(payload(), sort_keys=True).encode()).hexdigest()[:16]


if __name__ == "__main__":
    print(json.dumps(payload(), indent=2))
    print(f"\nPREREG_HASH = {prereg_hash()}")
