"""Composition-uncertain two-submission portfolio value + division-term
false-pass analysis.  CPU only, no Kaggle, no GPU.

Companion to ``scripts/private_split_simulator.py`` (which supplies the exact
per-crop loader and the exact ``summarise`` re-implementation).

Stage ``portfolio``
    The uniform 29/71 simulator holds the *composition* of the private set at
    the observed OOF composition, so it answers only "how noisy is the score".
    It cannot value a second submission, because under a fixed composition the
    ranking never moves.  Here the composition itself is a random variable:
    Dirichlet draws over family x density cells, plus explicit family-mass
    sweeps.  Kaggle scores both selected submissions on private and keeps the
    better one, so the portfolio value of a pair is E[max(s_a, s_b)].

Stage ``spurious``
    Independent verification of the claimed +/-0.0086 SD on the 44b6 composite
    delta and the 12.7% P(spurious bilateral pass at 5 trials), for a
    division-term-driven candidate (the H0c operating point: division FP driven
    to 0, k of n dividing mothers recovered).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_split_simulator import (  # noqa: E402
    DIV_W, SYSTEMS, load, score, score_batch, scores_all, terciles,
)

OUTDIR = Path(r"C:\Users\aryaa\Documents\Biohub-CellTracking-2026_RESEARCH"
              r"\agent_runs\agent5")

# H0c replay operating point (reports/inventory/phaseb_h0c_replay.json)
H0C = {
    "44b6": {"n_mothers": 26, "recovered": 16, "adj_delta": +0.000990974024206559},
    "6bba": {"n_mothers": 125, "recovered": 76, "adj_delta": -0.0005501828227123529},
}


# ------------------------------------------------------------------ portfolio
def cell_labels(fam, cov, mode):
    if mode == "family":
        return (fam == "6bba").astype(int), 2
    if mode == "density_global":
        q = np.quantile(cov["density"], [1 / 3, 2 / 3])
        return np.digitize(cov["density"], q), 3
    if mode == "family_x_density":
        return (fam == "6bba").astype(int) * 3 + terciles(cov["density"], fam), 6
    raise ValueError(mode)


def composition_draws(rng, fam, arr, cov, mode, alpha_scale, draws, frac=0.71):
    """Draw private-set weight vectors: crop-blocked 71% subsample inside each
    cell, then reweight cells so their EDGE-MASS shares follow a Dirichlet.

    ``alpha_scale=None`` -> cell mass shares fixed at the observed values
    (composition certain).  Small alpha_scale -> wide composition uncertainty.
    """
    n = len(fam)
    cell, ncell = cell_labels(fam, cov, mode)
    w = arr["A"]["w"]
    obs = np.array([w[cell == c].sum() for c in range(ncell)], float)
    obs /= obs.sum()
    idx = [np.where(cell == c)[0] for c in range(ncell)]
    ks = [max(1, int(round(frac * len(i)))) for i in idx]

    M = np.zeros((draws, n))
    if alpha_scale is None:
        tgt = np.tile(obs, (draws, 1))
    else:
        tgt = rng.dirichlet(np.maximum(obs * alpha_scale, 0.05), size=draws)
    for d in range(draws):
        sel = np.zeros(n)
        for i, k in zip(idx, ks):
            sel[rng.choice(i, k, replace=False)] = 1.0
        cm = np.array([(sel * w)[cell == c].sum() for c in range(ncell)])
        scale = np.where(cm > 0, tgt[d] / np.maximum(cm, 1e-12), 0.0)
        M[d] = sel * scale[cell]
    return M


def portfolio_report(rng, D, draws):
    fam, arr, cov = D["fam"], D["arr"], D["cov"]
    keys, names = list(SYSTEMS), [SYSTEMS[k] for k in SYSTEMS]
    out = {}
    scenarios = [
        ("composition_certain(OOF-like)", "family_x_density", None),
        ("mild_uncertainty(alpha=32)", "family_x_density", 32.0),
        ("moderate_uncertainty(alpha=8)", "family_x_density", 8.0),
        ("wide_uncertainty(alpha=2)", "family_x_density", 2.0),
        ("family_only_wide(alpha=2)", "family", 2.0),
        ("density_only_wide(alpha=2)", "density_global", 2.0),
    ]
    for label, mode, a in scenarios:
        M = composition_draws(rng, fam, arr, cov, mode, a, draws)
        s = scores_all(arr, M)
        rec = {"per_system": {names[i]: {"mean": round(float(s[i].mean()), 5),
                                        "p5": round(float(np.percentile(s[i], 5)), 5),
                                        "p25": round(float(np.percentile(s[i], 25)), 5)}
                              for i in range(len(keys))},
               "winner_share": {names[i]: round(float((s.argmax(0) == i).mean()), 4)
                                for i in range(len(keys))},
               "pairs": {}}
        best_single = int(np.argmax(s.mean(1)))
        for i, j in itertools.combinations(range(len(keys)), 2):
            mx = np.maximum(s[i], s[j])
            rec["pairs"][f"{names[i]}+{names[j]}"] = {
                "E_max": round(float(mx.mean()), 5),
                "p5_max": round(float(np.percentile(mx, 5)), 5),
                "p25_max": round(float(np.percentile(mx, 25)), 5),
                "gain_vs_best_of_the_two": round(float(mx.mean() - max(s[i].mean(), s[j].mean())), 5),
                "p5_gain_vs_best_of_the_two": round(
                    float(np.percentile(mx, 5) - max(np.percentile(s[i], 5),
                                                     np.percentile(s[j], 5))), 5),
                "gain_vs_global_best_single": round(float(mx.mean() - s[best_single].mean()), 5),
                "P_partner_rescues": round(float((np.minimum(s[i], s[j]) < mx - 1e-12).mean()), 4),
                "P_i_beats_j": round(float((s[i] > s[j]).mean()), 4),
                "corr_scores": round(float(np.corrcoef(s[i], s[j])[0, 1]), 4),
            }
        rec["global_best_single"] = names[best_single]
        out[label] = rec
    return out


# ------------------------------------------------------------------- spurious
def spurious_report(rng, D, reps=200000):
    """Sampling SD of a division-driven composite delta, and false-pass rates."""
    fam, arr, cov = D["fam"], D["arr"], D["cov"]
    ndiv = cov["n_div"]
    res = {"bar": 0.005, "div_weight": DIV_W, "note": (
        "Candidate class modelled = H0c-like: division FP driven to 0, k of n "
        "GT dividing mothers converted FN->TP, edge term essentially unchanged. "
        "Then division_jaccard = k/n exactly and delta_composite = 0.1*k/n "
        "(E0c baseline division_jaccard is 0.0000 on 44b6 and 0.0057 on 6bba).")}

    for f, cfg in H0C.items():
        m = fam == f
        crops = np.where(m)[0]
        nd = ndiv[crops].astype(int)
        n_tot = int(nd.sum())
        assert n_tot == cfg["n_mothers"], (f, n_tot, cfg["n_mothers"])
        p = cfg["recovered"] / n_tot
        quantum = DIV_W / n_tot

        # (i) analytic binomial / ratio-estimator SD, mothers i.i.d.
        sd_binom = DIV_W * np.sqrt(p * (1 - p) / n_tot)

        # (ii) crop-clustered: bootstrap crops AND redraw which mothers are hit
        nb = reps // 20
        boot = rng.multinomial(len(crops), np.full(len(crops), 1 / len(crops)), size=nb)
        # per-replicate mother total
        n_rep = boot @ nd
        # recovered count: sum over crops of Binom(mult_c * nd_c, p)
        k_rep = rng.binomial(np.maximum(n_rep, 0), p)
        d_clust = DIV_W * np.where(n_rep > 0, k_rep / np.maximum(n_rep, 1), 0.0)
        sd_clust = float(d_clust.std(ddof=1))

        # (iii) fixed crop set (the real OOF measurement), only mother-level noise
        k_fix = rng.binomial(n_tot, p, size=reps)
        d_fix = DIV_W * k_fix / n_tot
        sd_fix = float(d_fix.std(ddof=1))

        res[f] = {
            "n_mothers": n_tot, "recovered_at_H0c": cfg["recovered"],
            "p_recovery": round(p, 4),
            "quantum_per_mother": round(quantum, 6),
            "mothers_needed_for_+0.005": int(np.ceil(0.005 / quantum)),
            "sd_delta_analytic_binomial": round(float(sd_binom), 6),
            "sd_delta_fixed_crops": round(sd_fix, 6),
            "sd_delta_crop_bootstrap_clustered": round(sd_clust, 6),
            "design_effect_from_crop_clustering": round((sd_clust / sd_binom) ** 2, 3),
            "bar_in_SD_units(fixed)": round(0.005 / sd_fix, 3),
            "SD_as_multiple_of_bar(fixed)": round(sd_fix / 0.005, 3),
        }

    # --------- false-pass rates under a zero-true-effect null -----------------
    s44 = res["44b6"]["sd_delta_fixed_crops"]
    s6 = res["6bba"]["sd_delta_fixed_crops"]
    p44 = float(stats.norm.sf(0.005 / s44))
    p6 = float(stats.norm.sf(0.005 / s6))
    res["null_zero_true_effect_gaussian"] = {
        "sd_44b6": s44, "sd_6bba": s6,
        "P_pass_44b6": round(p44, 4), "P_pass_6bba": round(p6, 4),
        "P_bilateral_pass_independent": round(p44 * p6, 5),
        "P_at_least_one_bilateral_pass": {
            str(t): round(1 - (1 - p44 * p6) ** t, 4) for t in (1, 3, 5, 10, 20)},
    }
    # correlated-noise version: a single algorithm's luck is shared across families
    corr_tab = {}
    n = 400000
    for rho in (0.0, 0.25, 0.5, 0.75):
        z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
        pb = float(((z[:, 0] * s44 >= 0.005) & (z[:, 1] * s6 >= 0.005)).mean())
        corr_tab[f"rho={rho}"] = {
            "P_bilateral_pass": round(pb, 5),
            "P_at_least_one_in_5_trials": round(1 - (1 - pb) ** 5, 4)}
    res["null_zero_true_effect_correlated"] = corr_tab

    # --------- integer-lattice (exact) version --------------------------------
    lat = {}
    for f in ("44b6", "6bba"):
        nt = res[f]["n_mothers"]
        need = res[f]["mothers_needed_for_+0.005"]
        # null "no skill beyond the base rate": expected TPs among a retained
        # shortlist of size S at the H1a census base rate 92 / 14,371,002
        lat[f] = {"mothers_needed": need,
                  "P_pass_if_true_p_is_half_the_bar": round(
                      float(stats.binom.sf(need - 1, nt, (need / 2) / nt)), 4)}
    res["integer_lattice"] = lat

    # --------- decision-relevant posterior ------------------------------------
    res["posterior_flat_prior_given_measured_+0.005_bilateral"] = {
        "P_true_delta_44b6_below_0": round(float(stats.norm.cdf(-0.005 / s44)), 4),
        "P_true_delta_6bba_below_0": round(float(stats.norm.cdf(-0.005 / s6)), 4),
        "P_either_family_truly_negative": round(
            1 - (1 - float(stats.norm.cdf(-0.005 / s44)))
            * (1 - float(stats.norm.cdf(-0.005 / s6))), 4),
    }

    # --------- where H0c actually sits ----------------------------------------
    res["H0c_measured_in_SD_units"] = {
        "44b6": round(0.0625294355626681 / res["44b6"]["sd_delta_fixed_crops"], 2),
        "6bba": round(0.059684046314487094 / res["6bba"]["sd_delta_fixed_crops"], 2),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("portfolio", "spurious", "all"), default="all")
    ap.add_argument("--draws", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=20260731)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    D = load()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rep = {}
    if a.stage in ("portfolio", "all"):
        rep["portfolio"] = portfolio_report(rng, D, a.draws)
    if a.stage in ("spurious", "all"):
        rep["spurious"] = spurious_report(rng, D)
    (OUTDIR / "portfolio_risk.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
