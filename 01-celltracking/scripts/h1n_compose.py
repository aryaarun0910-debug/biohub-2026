"""H1-N Task 2 -- the THREE PREREGISTERED COMPOSITIONS, nested cross-fitting, pooled
composite.

  A  H1-M alone
  B  H1-M GATED BY the best frozen H1-I biological signal (median split, quantile from
     the OTHER family)
  C  H1-M PLUS association-safety abstention (drop rank-0 pairs that require a steal)

The rules were written in `h1n_prereg.py` and hashed BEFORE any outcome below was
computed. No combination search was performed; only these three arms exist.

NESTING. Three levels, none of which ever sees the family it scores:
  1. the H1-M score is crop-grouped 5-fold OOF inside its own family;
  2. the composition's gate/abstention threshold is a quantile of the OTHER family;
  3. the admission budget K is a RATE argmax-ed on the OTHER family and applied here.

POOLED COMPOSITE. The primary currency is the pooled composite over all 199 crops. A
199-crop exact replay is compute-gated, so this script reports a SURROGATE that is
calibrated on, and validated against, the 11 exactly-measured arms of
`reports/inventory/pooled_breakeven.json`:

  d_fam    = c_fam * k/(G_fam + m) + e_fam        c and e derived from the FP=0 arms
  d_pooled = w44 * d_44b6 + w6bba * d_6bba        w from a delta-space least squares

The calibration is self-checking: inverting the four k-arms through c_fam recovers their
integer per-family true-positive counts to within 0.04 of an integer, and the pooled
weights reproduce all 11 measured pooled deltas to <= 1.1e-4 (<= 5.3e-5 in the low-FP
regime these compositions occupy). Both numbers are printed and stored.

Usage:
  .venv\\Scripts\\python.exe scripts\\h1n_compose.py ^
      --comp <...>\\comp_mother_events.parquet --rtd <...>\\rtd_full.parquet ^
      --breakeven reports\\inventory\\pooled_breakeven.json ^
      --out <...>\\compositions.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h1m_features as F                      # noqa: E402
import h1m_gate as G                          # noqa: E402
import h1n_prereg as P                        # noqa: E402
from h1n_forensics import attach_signals, h1m_oof   # noqa: E402

FAMS = ("44b6", "6bba")
OTHER = {"44b6": "6bba", "6bba": "44b6"}


# --------------------------------------------------------------------------- #
# pooled surrogate                                                            #
# --------------------------------------------------------------------------- #
class Pooled:
    def __init__(self, breakeven: dict):
        A = breakeven["arms"]
        b = A["base"]
        self.base = breakeven["base_pooled"]
        self.e = {"44b6": A["supp"]["f44b6"] - b["f44b6"],
                  "6bba": A["supp"]["f6bba"] - b["f6bba"]}
        # c from the FP=0 full-recovery arm: all realisable positives, zero false forks
        self.c = {"44b6": (A["k1.0"]["f44b6"] - b["f44b6"] - self.e["44b6"]) / (16 / 26),
                  "6bba": (A["k1.0"]["f6bba"] - b["f6bba"] - self.e["6bba"]) / (76 / 125)}
        ks = list(A)
        X = np.stack([[A[k]["f44b6"] - b["f44b6"] for k in ks],
                      [A[k]["f6bba"] - b["f6bba"] for k in ks]], 1)
        y = np.array([A[k]["delta"] for k in ks])
        self.w, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.res = X @ self.w - y
        self._A, self._ks, self._b = A, ks, b
        self._fit_edge_cost()

    def _fit_edge_cost(self) -> None:
        """The J-only model IGNORES the edge damage each false fork does (a wrong edge
        added, a right one displaced). That is invisible at m=0 -- which is why the k-arms
        invert to exact integers -- but it grows with m. Measure it directly on the five
        fpN arms, whose per-family (k, m) split is fixed by their construction (N false
        forks per true one, 16 : 76), and fit log(residual) = alpha + beta*log(m)."""
        A = self._A
        ms, rs = [], []
        for name, n in (("fp1", 1), ("fp2", 2), ("fp5", 5), ("fp10", 10), ("fp25", 25)):
            tp = A[name]["div_tp"]
            k44 = min(16, tp); k6 = tp - k44
            m44, m6 = 16 * n, 76 * n
            pred = float(self.w[0] * (self.c["44b6"] * k44 / (26 + m44) + self.e["44b6"])
                         + self.w[1] * (self.c["6bba"] * k6 / (125 + m6) + self.e["6bba"]))
            ms.append(m44 + m6); rs.append(pred - A[name]["delta"])
        ms, rs = np.array(ms, float), np.array(rs, float)
        b, a = np.polyfit(np.log(ms), np.log(rs), 1)
        self.edge_beta, self.edge_alpha = float(b), float(a)
        self.edge_fit = {"m": ms.tolist(), "residual": rs.tolist(),
                         "alpha": float(a), "beta": float(b),
                         "refit_residual": (np.exp(a) * ms ** b - rs).tolist()}

    def edge_correction(self, m_total: int) -> float:
        return float(np.exp(self.edge_alpha) * max(m_total, 1) ** self.edge_beta)

    def family_delta(self, fam: str, k: int, m: int) -> float:
        return self.c[fam] * k / (P.GT_DIVISIONS[fam] + m) + self.e[fam]

    def pooled_delta(self, k44: int, m44: int, k6: int, m6: int) -> float:
        """J-only surrogate: OPTIMISTIC by the false-fork edge cost (see corrected())."""
        return float(self.w[0] * self.family_delta("44b6", k44, m44)
                     + self.w[1] * self.family_delta("6bba", k6, m6))

    def corrected(self, k44: int, m44: int, k6: int, m6: int) -> float:
        return self.pooled_delta(k44, m44, k6, m6) - self.edge_correction(m44 + m6)

    def frontier(self, p: float, r: float) -> float:
        """Corrected pooled delta if BOTH families ran at precision p and recall r of
        their realisable positives (16 / 76)."""
        k44, k6 = 16 * r, 76 * r
        m44, m6 = k44 * (1 - p) / max(p, 1e-9), k6 * (1 - p) / max(p, 1e-9)
        d = float(self.w[0] * (self.c["44b6"] * k44 / (26 + m44) + self.e["44b6"])
                  + self.w[1] * (self.c["6bba"] * k6 / (125 + m6) + self.e["6bba"]))
        return d - self.edge_correction(int(round(m44 + m6)))

    def calibration_report(self) -> dict:
        """Self-check 1: invert the FP=0 arms and recover integer TP counts.
        Self-check 2: pooled weights vs every measured arm."""
        inv = {}
        for k in ("k0.25", "k0.5", "k0.75", "k1.0"):
            t44 = (self._A[k]["f44b6"] - self._b["f44b6"] - self.e["44b6"]) / self.c["44b6"] * 26
            t6 = (self._A[k]["f6bba"] - self._b["f6bba"] - self.e["6bba"]) / self.c["6bba"] * 125
            inv[k] = {"implied_TP_44b6": round(t44, 3), "implied_TP_6bba": round(t6, 3),
                      "implied_TP_total": round(t44 + t6, 3),
                      "measured_TP_total": self._A[k]["div_tp"],
                      "max_distance_to_integer": round(
                          max(abs(t44 - round(t44)), abs(t6 - round(t6))), 4)}
        low = [i for i, k in enumerate(self._ks) if self._A[k]["div_fp"] <= 200]
        return {
            "c": self.c, "e": self.e, "w": self.w.tolist(),
            "inverted_k_arms": inv,
            "pooled_weight_max_abs_residual_all_arms": float(np.max(np.abs(self.res))),
            "pooled_weight_max_abs_residual_low_FP_arms": float(
                np.max(np.abs(self.res[low]))),
            "low_FP_arms": [self._ks[i] for i in low],
        }


# --------------------------------------------------------------------------- #
# compositions                                                                 #
# --------------------------------------------------------------------------- #
ARM_NAMES = {
    "D_posthoc": "POST-HOC (NOT PREREGISTERED): H1-M gated by the GEOMETRY rank at the "
                 "other family's median. Generated BY the preregistered overlap matrix "
                 "(GEOM is the least-overlapping signal with H1-M, Jaccard 0.2711). "
                 "Requires independent confirmation before any claim.",
    "E_posthoc": "POST-HOC (NOT PREREGISTERED): C then D -- association-safety abstention "
                 "AND geometry median gate. Same caveat.",
    "G_posthoc": "POST-HOC (NOT PREREGISTERED), falsification test of the overlap matrix: "
                 "H1-M gated by the REVERSE-DIRECTION association margin at the other "
                 "family's median (Jaccard with H1-M 0.3043).",
    "H_posthoc": "POST-HOC (NOT PREREGISTERED), falsification test of the overlap matrix: "
                 "H1-M gated by the REVERSE-TIME appearance asymmetry at the other "
                 "family's median (Jaccard with H1-M 0.3269).",
    "F_posthoc": "POST-HOC (NOT PREREGISTERED), derived from the measured error taxonomy: "
                 "H1-M with a PER-CROP ADMISSION CAP. 131/202 false positives were "
                 "CALIBRATION errors and 118 of those sat in crops containing no "
                 "realisable division at all, with a single crop absorbing 51 admissions "
                 "and zero true mothers. The cap and the global rate are BOTH cross-fitted "
                 "from the other family. Requires preregistered confirmation.",
}
CAP_GRID = (1, 2, 3, 4, 5, 8, 12, 20)

# FALSIFICATION TEST of the overlap matrix as a PREDICTIVE instrument. The matrix says
# the composition value of gating H1-M by signal S should fall as Jaccard(H1M, S) rises:
#   GEOM 0.2711  <  RTD_G 0.3043  <  RTD_A 0.3269  <  H1I 0.3800
# Arms D (GEOM) and B (H1I) are the endpoints. G_ and H_ fill the middle. If the measured
# composite ordering does NOT follow the inverse-Jaccard ordering, the overlap matrix is
# descriptive only and must not be used to choose future compositions.
GATE_ARMS = {"D_posthoc": "S_GEOM", "G_posthoc": "S_RTD_G", "H_posthoc": "S_RTD_A"}


def survivors(d: pd.DataFrame, comp: str, thr: float | None) -> np.ndarray:
    """The composition's ELIGIBILITY mask. thr is always fitted on the other family."""
    if comp == "A":
        return np.ones(len(d), bool)
    if comp == "B":
        v = d.S_H1I.to_numpy(float)
        return np.where(np.isfinite(v), v >= thr, False)
    if comp == "C":
        return ~d.steal_rank0.to_numpy(bool)
    if comp in GATE_ARMS:
        v = d[GATE_ARMS[comp]].to_numpy(float)
        return np.where(np.isfinite(v), v >= thr, False)
    if comp == "E_posthoc":
        v = d.S_GEOM.to_numpy(float)
        return np.where(np.isfinite(v), v >= thr, False) & ~d.steal_rank0.to_numpy(bool)
    if comp == "F_posthoc":
        return np.ones(len(d), bool)
    raise ValueError(comp)


def apply_cap(sub: pd.DataFrame, s: np.ndarray, cap: int) -> np.ndarray:
    """Per-crop admission cap: keep at most `cap` highest-scoring mothers per crop.
    Returns positional indices into `sub`, ordered by score descending."""
    order = np.argsort(-s)
    seen: dict[str, int] = {}
    keep = []
    crops = sub.crop.to_numpy()
    for i in order:
        c = crops[i]
        if seen.get(c, 0) >= cap:
            continue
        seen[c] = seen.get(c, 0) + 1
        keep.append(i)
    return np.array(keep, int)


def rate_and_thr(d_other: pd.DataFrame, fam_other: str, comp: str) -> tuple[float, float | None]:
    """Everything the rule needs, measured ONLY on the other family."""
    thr = None
    if comp == "B":
        thr = float(np.nanquantile(d_other.S_H1I.to_numpy(float), P.COMPOSITION_B_QUANTILE))
    elif comp in GATE_ARMS:
        thr = float(np.nanquantile(d_other[GATE_ARMS[comp]].to_numpy(float),
                                   P.COMPOSITION_B_QUANTILE))
    elif comp == "E_posthoc":
        thr = float(np.nanquantile(d_other.S_GEOM.to_numpy(float), P.COMPOSITION_B_QUANTILE))
    mask = survivors(d_other, comp, thr)
    sub = d_other[mask].reset_index(drop=True)
    if len(sub) == 0:
        return 0.0, thr
    if comp == "F_posthoc":
        # BOTH scalars cross-fitted: the cap and the rate are argmax-ed together on the
        # other family, then applied here.
        s = sub.S_H1M.to_numpy(float)
        best = (-1.0, 1, 0.0)
        for cap in CAP_GRID:
            idx = apply_cap(sub, s, cap)
            capped = sub.iloc[idx].reset_index(drop=True)
            bp = G.best_operating_point(capped, capped.S_H1M.to_numpy(float), fam_other)
            if bp["divJ"] > best[0]:
                best = (bp["divJ"], cap, bp["K"] / max(len(capped), 1))
        return best[2], float(best[1])
    bp = G.best_operating_point(sub, sub.S_H1M.to_numpy(float), fam_other)
    return bp["K"] / len(sub), thr


def operating_curve(D: dict[str, pd.DataFrame], comp: str, PLc: "Pooled") -> dict:
    """DIAGNOSTIC, OPTIMISTIC. Sweep the admission budget jointly over both families
    (allocating the next admission to whichever family's next-ranked survivor scores
    higher after a per-family z-transform of the H1-M score) and report the pooled
    surrogate composite along the way. The threshold is chosen ON the data, so this is
    an UPPER BOUND on what the composition could deliver -- exactly the role
    `best_point_optimistic` plays in h1m_gate. It answers one question only: does the
    mechanism have +0.005 of headroom at all, or is it out of reach at every budget?"""
    parts = []
    for fam in FAMS:
        rate, thr = rate_and_thr(D[OTHER[fam]], OTHER[fam], comp)
        sub = D[fam][survivors(D[fam], comp, thr)].reset_index(drop=True)
        s = sub.S_H1M.to_numpy(float)
        if comp == "F_posthoc":
            sub = sub.iloc[apply_cap(sub, s, int(thr))].reset_index(drop=True)
            s = sub.S_H1M.to_numpy(float)
        z = (s - np.nanmedian(s)) / max(float(np.nanstd(s)), 1e-9)
        parts.append(pd.DataFrame({"family": fam, "z": z,
                                   "realisable": sub.realisable.to_numpy()}))
    allf = pd.concat(parts, ignore_index=True).sort_values("z", ascending=False)
    is44 = (allf.family == "44b6").to_numpy()
    real = allf.realisable.to_numpy().astype(int)
    k44 = np.cumsum(real * is44); n44 = np.cumsum(is44)
    k6 = np.cumsum(real * ~is44); n6 = np.cumsum(~is44)
    m44, m6 = n44 - k44, n6 - k6
    delta = np.array([PLc.corrected(int(a), int(b), int(c), int(d))
                      for a, b, c, d in zip(k44, m44, k6, m6)])
    K = np.arange(1, len(allf) + 1)
    prec = (k44 + k6) / K
    i = int(np.argmax(delta))
    pts = []
    for kk in (10, 25, 50, 100, 200, 400, 800, 1600):
        if kk <= len(K):
            pts.append({"K": kk, "TP": int(k44[kk - 1] + k6[kk - 1]),
                        "precision": float(prec[kk - 1]),
                        "composite_delta_SURROGATE": float(delta[kk - 1])})
    return {
        "note": "OPTIMISTIC upper bound: budget chosen on the evaluated data.",
        "argmax": {"K": int(K[i]), "TP": int(k44[i] + k6[i]), "FP": int(K[i] - k44[i] - k6[i]),
                   "TP_44b6": int(k44[i]), "TP_6bba": int(k6[i]),
                   "precision": float(prec[i]),
                   "composite_delta_SURROGATE": float(delta[i])},
        "max_precision_at_K_ge_25": float(np.max(prec[24:])) if len(prec) > 25 else float("nan"),
        "points": pts,
        "n_survivors_total": int(len(allf)),
        "n_realisable_survivors": int(real.sum()),
    }


def run_composition(D: dict[str, pd.DataFrame], comp: str) -> dict:
    spec = P.COMPOSITIONS.get(comp)
    out = {"composition": comp,
           "name": spec["name"] if spec else ARM_NAMES[comp],
           "rule": spec["rule"] if spec else ARM_NAMES[comp],
           "preregistered": spec is not None, "families": {}}
    for fam in FAMS:
        rate, thr = rate_and_thr(D[OTHER[fam]], OTHER[fam], comp)
        d = D[fam]
        mask = survivors(d, comp, thr)
        sub = d[mask].reset_index(drop=True)
        s = sub.S_H1M.to_numpy(float)
        if comp == "F_posthoc":
            sub = sub.iloc[apply_cap(sub, s, int(thr))].reset_index(drop=True)
            s = sub.S_H1M.to_numpy(float)
        K = int(round(rate * len(sub)))
        K = max(0, min(K, len(sub)))
        order = np.argsort(-s)[:K]
        sel = sub.iloc[order]
        k = int(sel.realisable.sum())
        m = K - k
        out["families"][fam] = {
            "n_population": int(len(d)),
            "n_realisable_population": int(d.realisable.sum()),
            "n_survivors": int(len(sub)),
            "n_realisable_survivors": int(sub.realisable.sum()),
            "gate_threshold_from_other_family": thr,
            "admission_rate_from_other_family": rate,
            "K": K, "TP_realisable": k, "FP": m,
            "TP_divider": int((sel.label.to_numpy() == 1).sum()),
            "precision_among_visible": (k / K) if K else float("nan"),
            "divJ": k / (P.GT_DIVISIONS[fam] + m),
            "recall_of_realisable": k / max(int(d.realisable.sum()), 1),
            "admitted_keys": sel.key.tolist(),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True)
    ap.add_argument("--rtd", required=True)
    ap.add_argument("--breakeven", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--posthoc", action="store_true",
                    help="also run the arms the overlap matrix suggested; NOT preregistered")
    a = ap.parse_args()

    print(f"PREREG_HASH  {P.prereg_hash()}")
    assert F.config_hash() == P.H1M_FEATURE_HASH

    be = json.loads(Path(a.breakeven).read_text())
    assert be["cfg_hash"] == P.H0C_CFG_HASH
    PL = Pooled(be)
    cal = PL.calibration_report()
    print("surrogate calibration: c=%.6f/%.6f  w=%.5f/%.5f  maxres(all)=%.2e "
          "maxres(lowFP)=%.2e" % (cal["c"]["44b6"], cal["c"]["6bba"], cal["w"][0],
                                  cal["w"][1],
                                  cal["pooled_weight_max_abs_residual_all_arms"],
                                  cal["pooled_weight_max_abs_residual_low_FP_arms"]))
    for k, v in cal["inverted_k_arms"].items():
        print(f"  invert {k:6s} -> TP 44b6={v['implied_TP_44b6']:.3f} "
              f"6bba={v['implied_TP_6bba']:.3f} total={v['implied_TP_total']:.3f} "
              f"(measured {v['measured_TP_total']}), max dist to integer "
              f"{v['max_distance_to_integer']}")

    cache = Path(a.out).with_suffix(".scored.parquet")
    if cache.exists() and not a.refit:
        DD = pd.read_parquet(cache)
        print(f"  reusing cached OOF scores {cache}")
    else:
        cm = pd.read_parquet(a.comp)
        rtd = pd.read_parquet(a.rtd)[["crop", "t", "mother", "rtd_margin", "rtd_nclaim",
                                      "rtd_loses", "r_alt_min"]]
        raw = h1m_oof(cm, F.H1M_FEATURES)
        ps = []
        for fam in FAMS:
            d, s = raw[fam]
            d = attach_signals(d, s, rtd)
            d["key"] = d.crop + ":" + d.mother.astype(str) + ":" + d.t.astype(str)
            ps.append(d[["key", "crop", "family", "t", "mother", "label", "realisable",
                         "steal_rank0", "S_H1M", "S_H1I", "S_RTD_A", "S_RTD_G", "S_GEOM"]])
        DD = pd.concat(ps, ignore_index=True)
        DD.to_parquet(cache)
    D = {fam: DD[DD.family == fam].reset_index(drop=True) for fam in FAMS}

    res = {"prereg_hash": P.prereg_hash(), "feature_hash": F.config_hash(),
           "h0c_cfg_hash": P.H0C_CFG_HASH, "surrogate_calibration": cal,
           "pooled_base": PL.base, "arms": {}}
    arms = ["A", "B", "C"] + (["D_posthoc", "G_posthoc", "H_posthoc",
                                  "E_posthoc", "F_posthoc"] if a.posthoc else [])
    for comp in arms:
        r = run_composition(D, comp)
        r["operating_curve_OPTIMISTIC"] = operating_curve(D, comp, PL)
        f44, f6 = r["families"]["44b6"], r["families"]["6bba"]
        k44, m44, k6, m6 = (f44["TP_realisable"], f44["FP"],
                            f6["TP_realisable"], f6["FP"])
        r["pooled"] = {
            "TP": k44 + k6, "FP": m44 + m6, "K": k44 + m44 + k6 + m6,
            "precision_among_visible": (k44 + k6) / max(k44 + m44 + k6 + m6, 1),
            "recall_of_92": (k44 + k6) / 92,
            "family_delta_44b6_SURROGATE": PL.family_delta("44b6", k44, m44),
            "family_delta_6bba_SURROGATE": PL.family_delta("6bba", k6, m6),
            "composite_delta_SURROGATE_Jonly": PL.pooled_delta(k44, m44, k6, m6),
            "false_fork_edge_correction": PL.edge_correction(m44 + m6),
            "composite_delta_SURROGATE": PL.corrected(k44, m44, k6, m6),
            "surrogate_uncertainty_low_FP": cal[
                "pooled_weight_max_abs_residual_low_FP_arms"],
        }
        d = r["pooled"]
        tgt = P.DECISION["precision_target"]
        r["pooled"]["reaches_10_15pct"] = bool(
            d["precision_among_visible"] >= tgt["+0.005"] and d["TP"] >= 8
            and f44["precision_among_visible"] >= tgt["+0.005"]
            and f6["precision_among_visible"] >= tgt["+0.005"])
        res["arms"][comp] = r
        print(f"\n===== COMPOSITION {comp}: {r['name']} =====")
        for fam in FAMS:
            f = r["families"][fam]
            print(f"  {fam}: survivors {f['n_survivors']:,}/{f['n_population']:,} "
                  f"(realisable {f['n_realisable_survivors']}/{f['n_realisable_population']}) "
                  f"K={f['K']} TP={f['TP_realisable']} FP={f['FP']} "
                  f"prec={f['precision_among_visible']:.4f} divJ={f['divJ']:.4f}")
        print(f"  POOLED: TP={d['TP']} FP={d['FP']} K={d['K']} "
              f"precision={d['precision_among_visible']:.4f} "
              f"recall={d['recall_of_92']:.3f}  "
              f"composite delta (SURROGATE) = {d['composite_delta_SURROGATE']:+.5f} "
              f"+/- {d['surrogate_uncertainty_low_FP']:.1e}")
        oc = r["operating_curve_OPTIMISTIC"]["argmax"]
        print(f"  [OPTIMISTIC ceiling, budget chosen on the data] K={oc['K']} "
              f"TP={oc['TP']} ({oc['TP_44b6']}+{oc['TP_6bba']}) FP={oc['FP']} "
              f"prec={oc['precision']:.4f} delta={oc['composite_delta_SURROGATE']:+.5f}")

    # what (precision, recall) pair does +0.005 actually require?
    fr = {}
    for r_ in (0.2, 0.4, 0.6, 0.8, 1.0):
        need = None
        for p_ in np.arange(0.02, 0.999, 0.0005):
            if PL.frontier(float(p_), r_) >= 0.005:
                need = round(float(p_), 4); break
        fr[f"recall_{r_:.1f}"] = {
            "precision_needed_for_+0.005": need,
            "delta_at_10.15pct_precision": PL.frontier(0.10153, r_)}
    res["frontier_for_plus_0.005"] = fr
    res["edge_cost_fit"] = PL.edge_fit
    print("\n--- what +0.005 really costs (corrected surrogate, both families at p, r) ---")
    for k, v in fr.items():
        print(f"  {k}: precision needed = {v['precision_needed_for_+0.005']}   "
              f"delta at the quoted 10.15% = {v['delta_at_10.15pct_precision']:+.5f}")

    best = max([c for c in res["arms"] if res["arms"][c]["preregistered"]],
               key=lambda c: res["arms"][c]["pooled"]["composite_delta_SURROGATE"])
    res["winner"] = best
    res["winner_delta_over_A"] = (
        res["arms"][best]["pooled"]["composite_delta_SURROGATE"]
        - res["arms"]["A"]["pooled"]["composite_delta_SURROGATE"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    print(f"\nWINNER = {best}  ({res['winner_delta_over_A']:+.5f} vs A)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
