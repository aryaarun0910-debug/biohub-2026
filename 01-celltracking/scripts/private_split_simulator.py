"""Private-composition robustness simulator (CPU only, no Kaggle, no GPU).

Purpose
-------
The public leaderboard is 29% of the test set. Selection of the final two
submissions must be robust to plausible *private* compositions. This script
draws large numbers of pseudo-public / pseudo-private splits from the 199
per-crop OOF rows and reports, per candidate system:

  * mean and lower-tail (5th percentile) pseudo-private score
  * P(public-selected winner == private winner)
  * pairwise rank-reversal probability
  * worst-family and worst-regime loss
  * per-crop error covariance / correlation between systems
  * two-submission portfolio value E[max(s_a, s_b)] and its 5th percentile

Exactness
---------
The Kaggle run-level score is NOT a naive mean. From
``tracking_cellmot.metrics.summarise``:

    adj_edge_jaccard = sum_i w_i * adjJ_i / sum_i w_i ,  w_i = tp_i+fp_i+fn_i
    division_jaccard = (sum_i dtp_i) / (sum_i dtp_i + dfp_i + dfn_i)     [micro]
    score            = adj_edge_jaccard + 0.1 * division_jaccard

Every resampled score below reproduces that weighting exactly, including the
system-specific edge mass w_i and the micro-pooled (not averaged) division term.
Re-aggregation of the cached per-crop rows reproduces the published per-family
numbers to 4 d.p. (asserted at load time).

Usage
-----
  .venv\\Scripts\\python.exe scripts\\private_split_simulator.py --draws 20000
  .venv\\Scripts\\python.exe scripts\\private_split_simulator.py --stage spurious
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "artifacts" / "kaggle" / "coupled_cache" / "scores"
STATS = ROOT / "reports" / "inventory" / "embryo_stats.csv"
DIV_W = 0.1
PUBLIC_FRAC = 0.29

# arm -> display name (public LB where known)
SYSTEMS = {
    "A": "E0c",        # public 0.889
    "B": "B_detpop",
    "Bp": "Bp_ilp",
    "C": "C_survival",
    "D": "v122",       # public 0.908
}
PUBLIC_LB = {"A": 0.889, "D": 0.908}

# published per-family aggregates used as a load-time parity assertion
PARITY = {
    ("A", "44b6"): 0.7595, ("A", "6bba"): 0.6490,
    ("B", "44b6"): 0.7483, ("B", "6bba"): 0.6526,
    ("Bp", "44b6"): 0.7273, ("Bp", "6bba"): 0.6854,
    ("C", "44b6"): 0.6914, ("C", "6bba"): 0.6979,
    ("D", "44b6"): 0.6962, ("D", "6bba"): 0.6997,
}


# --------------------------------------------------------------------- loading
def load() -> dict:
    per = {}
    for p in sorted(SCORES.glob("*.json")):
        d = json.loads(p.read_text())
        per.setdefault(d["arm"], {})[d["crop"]] = d
    crops = sorted(per["A"])
    assert len(crops) == 199, len(crops)
    for arm in SYSTEMS:
        assert sorted(per[arm]) == crops, arm

    fam = np.array([c.split("_")[0] for c in crops])
    arr = {}
    for arm in SYSTEMS:
        rows = [per[arm][c] for c in crops]
        arr[arm] = {
            "adj": np.array([r["adj_edge_jaccard"] for r in rows], float),
            "etp": np.array([r["edge_tp"] for r in rows], float),
            "efp": np.array([r["edge_fp"] for r in rows], float),
            "efn": np.array([r["edge_fn"] for r in rows], float),
            "dtp": np.array([r["division_tp"] for r in rows], float),
            "dfp": np.array([r["division_fp"] for r in rows], float),
            "dfn": np.array([r["division_fn"] for r in rows], float),
        }
        arr[arm]["w"] = arr[arm]["etp"] + arr[arm]["efp"] + arr[arm]["efn"]

    # deployment-observable covariates (from arm A rows = E0c, the reference)
    cov = {k: np.array([per["A"][c][k] for c in crops], float)
           for k in ("density", "ratio", "displacement")}
    cov["mass"] = arr["A"]["w"].copy()          # annotation mass

    # ground-truth dividing-mother counts per crop
    ndiv = {}
    for line in STATS.read_text().splitlines()[1:]:
        f = line.split(",")
        ndiv[f[0]] = int(f[3])
    cov["n_div"] = np.array([ndiv[c] for c in crops], float)

    # parity assertion against the published per-family table
    for arm in SYSTEMS:
        for f in ("44b6", "6bba"):
            m = (fam == f).astype(float)
            got = score(arr[arm], m)
            assert abs(got - PARITY[(arm, f)]) < 5e-5, (arm, f, got)

    return {"crops": crops, "fam": fam, "arr": arr, "cov": cov}


# ----------------------------------------------------------------- exact score
def score(a: dict, m: np.ndarray) -> float:
    """Exact run-level score for crop weights ``m`` (>=0, may be fractional)."""
    wm = m * a["w"]
    tot = wm.sum()
    adj = float((wm * a["adj"]).sum() / tot) if tot > 0 else float("nan")
    dtp, dfp, dfn = (float((m * a[k]).sum()) for k in ("dtp", "dfp", "dfn"))
    den = dtp + dfp + dfn
    return adj + DIV_W * (dtp / den) if den > 0 else adj


def score_batch(a: dict, M: np.ndarray) -> np.ndarray:
    """Vectorised ``score`` over a (n_draws, n_crops) weight matrix."""
    wm = M * a["w"][None, :]
    tot = wm.sum(1)
    adj = np.where(tot > 0, (wm * a["adj"][None, :]).sum(1) / np.maximum(tot, 1e-12), np.nan)
    dtp = M @ a["dtp"]
    den = dtp + M @ a["dfp"] + M @ a["dfn"]
    return adj + DIV_W * np.where(den > 0, dtp / np.maximum(den, 1e-12), 0.0)


def scores_all(arr: dict, M: np.ndarray) -> np.ndarray:
    """(n_systems, n_draws) score matrix."""
    return np.stack([score_batch(arr[s], M) for s in SYSTEMS])


# ------------------------------------------------------------------- terciles
def terciles(v: np.ndarray, fam: np.ndarray) -> np.ndarray:
    """Within-family tercile label 0/1/2 (matches the coupled_score report)."""
    out = np.zeros(len(v), int)
    for f in np.unique(fam):
        idx = np.where(fam == f)[0]
        q = np.quantile(v[idx], [1 / 3, 2 / 3])
        out[idx] = np.digitize(v[idx], q)
    return out


# ------------------------------------------------------------------ scenarios
def draw_uniform(rng, n, n_pub, draws):
    """Crop-blocked uniform 29/71 splits. Returns (public_mask, private_mask)."""
    P = np.zeros((draws, n))
    for d in range(draws):
        P[d, rng.choice(n, n_pub, replace=False)] = 1.0
    return P, 1.0 - P


def draw_stratified(rng, fam, frac, draws):
    n = len(fam)
    P = np.zeros((draws, n))
    idxs = [np.where(fam == f)[0] for f in np.unique(fam)]
    ks = [max(1, int(round(frac * len(i)))) for i in idxs]
    for d in range(draws):
        for i, k in zip(idxs, ks):
            P[d, rng.choice(i, k, replace=False)] = 1.0
    return P, 1.0 - P


def draw_family_shifted(rng, fam, arr, target_44b6_mass, draws, frac_private=0.71):
    """Stratified private draws re-weighted so the 44b6 share of *edge mass*
    (the quantity that actually weights adj_edge_jaccard) equals the target."""
    n = len(fam)
    is44 = fam == "44b6"
    w = arr["A"]["w"]
    Q = np.zeros((draws, n))
    i44, i6 = np.where(is44)[0], np.where(~is44)[0]
    k44, k6 = int(round(frac_private * len(i44))), int(round(frac_private * len(i6)))
    for d in range(draws):
        sel = np.zeros(n)
        sel[rng.choice(i44, k44, replace=False)] = 1.0
        sel[rng.choice(i6, k6, replace=False)] = 1.0
        m44 = (sel * w * is44).sum()
        m6 = (sel * w * ~is44).sum()
        # alpha * m44 / (alpha*m44 + m6) = target
        t = target_44b6_mass
        alpha = (t * m6) / ((1 - t) * m44) if m44 > 0 and t < 1 else 1.0
        Q[d] = sel * np.where(is44, alpha, 1.0)
    return Q


def draw_bootstrap(rng, n, draws):
    """Per-crop bootstrap: multiplicity weights from a multinomial(n, 1/n)."""
    return rng.multinomial(n, np.full(n, 1.0 / n), size=draws).astype(float)


# -------------------------------------------------------------------- reporting
def pct(x, q):
    return float(np.percentile(x, q))


def main(draws: int, seed: int, out: Path) -> None:
    rng = np.random.default_rng(seed)
    D = load()
    fam, arr, cov = D["fam"], D["arr"], D["cov"]
    n = len(fam)
    keys = list(SYSTEMS)
    names = [SYSTEMS[k] for k in keys]
    rep: dict = {"n_crops": n, "draws": draws, "systems": names}

    # ---- reference full-OOF numbers (pooled and per family) -----------------
    ones = np.ones(n)
    rep["full_oof"] = {
        SYSTEMS[k]: {
            "pooled": round(score(arr[k], ones), 6),
            "44b6": round(score(arr[k], (fam == "44b6").astype(float)), 6),
            "6bba": round(score(arr[k], (fam == "6bba").astype(float)), 6),
            "public_lb": PUBLIC_LB.get(k),
        } for k in keys
    }

    # ---- 1. uniform crop-blocked 29/71 -------------------------------------
    n_pub = int(round(PUBLIC_FRAC * n))
    P, Q = draw_uniform(rng, n, n_pub, draws)
    sp, sq = scores_all(arr, P), scores_all(arr, Q)
    rep["scenario_uniform_29_71"] = summarise_scenario(sq, sp, keys)

    # ---- 2a. family-stratified ---------------------------------------------
    P2, Q2 = draw_stratified(rng, fam, PUBLIC_FRAC, draws)
    rep["scenario_family_stratified"] = summarise_scenario(
        scores_all(arr, Q2), scores_all(arr, P2), keys)

    # ---- 2b. family-SHIFTED mixtures ---------------------------------------
    obs44 = (arr["A"]["w"] * (fam == "44b6")).sum() / arr["A"]["w"].sum()
    rep["observed_44b6_mass_share"] = round(float(obs44), 4)
    shifted = {}
    for t in (0.90, 0.75, 0.50, 0.25, 0.10):
        Qs = draw_family_shifted(rng, fam, arr, t, max(2000, draws // 4))
        ss = scores_all(arr, Qs)
        shifted[f"44b6_mass={t:.2f}"] = {
            "per_system": {SYSTEMS[k]: {"mean": round(float(ss[i].mean()), 5),
                                        "p5": round(pct(ss[i], 5), 5)}
                           for i, k in enumerate(keys)},
            "winner_share": winner_share(ss, keys),
            "pairwise_A_beats": {SYSTEMS[k]: round(float((ss[0] > ss[i]).mean()), 4)
                                 for i, k in enumerate(keys) if k != "A"},
        }
    rep["scenario_family_shifted"] = shifted

    # ---- 3. regime reweighting ---------------------------------------------
    regimes = {}
    for cname in ("density", "displacement", "ratio", "mass"):
        tc = terciles(cov[cname], fam)
        for lvl, lab in enumerate(("low", "mid", "high")):
            m = (tc == lvl).astype(float)
            regimes[f"{cname}:{lab}"] = {
                SYSTEMS[k]: round(score(arr[k], m), 5) for k in keys}
        # tilted mixtures: 70% of mass on one tercile, 15% each on the others
        for lvl, lab in enumerate(("low", "mid", "high")):
            base = np.where(tc == lvl, 0.70, 0.15)
            cnt = np.array([(tc == j).sum() for j in range(3)], float)
            m = base / cnt[tc] * n
            regimes[f"{cname}:tilt_{lab}70"] = {
                SYSTEMS[k]: round(score(arr[k], m), 5) for k in keys}
    rep["scenario_regime_reweight"] = regimes

    # ---- 4. adversarial mixture over a plausible simplex --------------------
    rep["scenario_adversarial"] = adversarial(rng, fam, arr, cov, keys)

    # ---- 5. per-crop bootstrap ---------------------------------------------
    B = draw_bootstrap(rng, n, draws)
    sb = scores_all(arr, B)
    rep["scenario_bootstrap"] = {
        SYSTEMS[k]: {"mean": round(float(sb[i].mean()), 5),
                     "sd": round(float(sb[i].std(ddof=1)), 5),
                     "p5": round(pct(sb[i], 5), 5),
                     "p50": round(pct(sb[i], 50), 5),
                     "p95": round(pct(sb[i], 95), 5)}
        for i, k in enumerate(keys)}
    # paired bootstrap SD of the delta vs E0c
    rep["bootstrap_delta_vs_E0c"] = {}
    for f in ("44b6", "6bba", "pooled"):
        mask = np.ones(n) if f == "pooled" else (fam == f).astype(float)
        Bf = B * mask[None, :]
        sA = score_batch(arr["A"], Bf)
        rep["bootstrap_delta_vs_E0c"][f] = {
            SYSTEMS[k]: {"mean_delta": round(float((score_batch(arr[k], Bf) - sA).mean()), 5),
                         "sd_delta": round(float((score_batch(arr[k], Bf) - sA).std(ddof=1)), 5)}
            for k in keys if k != "A"}

    # ---- per-crop error covariance ------------------------------------------
    E = np.stack([1.0 - arr[k]["adj"] for k in keys])           # per-crop edge error
    rep["per_crop_error"] = {
        "definition": "e_i = 1 - adj_edge_jaccard_i (per crop, unweighted)",
        "mean": {SYSTEMS[k]: round(float(E[i].mean()), 5) for i, k in enumerate(keys)},
        "cov": {SYSTEMS[keys[i]]: {SYSTEMS[keys[j]]: round(float(np.cov(E)[i, j]), 6)
                                   for j in range(len(keys))} for i in range(len(keys))},
        "corr": {SYSTEMS[keys[i]]: {SYSTEMS[keys[j]]: round(float(np.corrcoef(E)[i, j]), 4)
                                    for j in range(len(keys))} for i in range(len(keys))},
    }
    # mass-weighted error (what actually drives the run-level score)
    wA = arr["A"]["w"]
    EW = E * (wA / wA.mean())[None, :]
    rep["per_crop_error_massweighted_corr"] = {
        SYSTEMS[keys[i]]: {SYSTEMS[keys[j]]: round(float(np.corrcoef(EW)[i, j]), 4)
                           for j in range(len(keys))} for i in range(len(keys))}

    # ---- two-submission portfolio value E[max] ------------------------------
    port = {}
    for i, j in itertools.combinations(range(len(keys)), 2):
        mx = np.maximum(sq[i], sq[j])
        port[f"{names[i]}+{names[j]}"] = {
            "E_max": round(float(mx.mean()), 5),
            "p5_max": round(pct(mx, 5), 5),
            "p25_max": round(pct(mx, 25), 5),
            "gain_over_best_single": round(float(mx.mean() - max(sq[i].mean(), sq[j].mean())), 5),
            "corr_private_score": round(float(np.corrcoef(sq[i], sq[j])[0, 1]), 4),
            "P_second_rescues": round(float((sq[j] > sq[i]).mean() if sq[i].mean() > sq[j].mean()
                                            else (sq[i] > sq[j]).mean()), 4),
        }
    for i in range(len(keys)):
        port[f"{names[i]}+{names[i]}(single)"] = {
            "E_max": round(float(sq[i].mean()), 5), "p5_max": round(pct(sq[i], 5), 5),
            "p25_max": round(pct(sq[i], 25), 5), "gain_over_best_single": 0.0,
            "corr_private_score": 1.0, "P_second_rescues": 0.0}
    rep["portfolio_two_submission"] = port

    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def winner_share(s: np.ndarray, keys) -> dict:
    w = s.argmax(0)
    return {SYSTEMS[k]: round(float((w == i).mean()), 4) for i, k in enumerate(keys)}


def summarise_scenario(sq: np.ndarray, sp: np.ndarray, keys) -> dict:
    """sq = pseudo-private scores, sp = pseudo-public scores. (n_sys, draws)."""
    ns = len(keys)
    names = [SYSTEMS[k] for k in keys]
    per = {names[i]: {"mean": round(float(sq[i].mean()), 5),
                      "sd": round(float(sq[i].std(ddof=1)), 5),
                      "p5": round(pct(sq[i], 5), 5),
                      "p50": round(pct(sq[i], 50), 5),
                      "p95": round(pct(sq[i], 95), 5),
                      "public_mean": round(float(sp[i].mean()), 5),
                      "public_sd": round(float(sp[i].std(ddof=1)), 5)}
           for i in range(ns)}
    pub_win, prv_win = sp.argmax(0), sq.argmax(0)
    rev = {}
    for i, j in itertools.combinations(range(ns), 2):
        dpub, dprv = sp[i] - sp[j], sq[i] - sq[j]
        rev[f"{names[i]} vs {names[j]}"] = {
            "P_i_beats_j_private": round(float((dprv > 0).mean()), 4),
            "P_rank_reversal_pub_vs_priv": round(float(((dpub > 0) != (dprv > 0)).mean()), 4),
            "mean_private_gap": round(float(dprv.mean()), 5),
            "p5_private_gap": round(pct(dprv, 5), 5),
            "p95_private_gap": round(pct(dprv, 95), 5),
        }
    return {
        "per_system": per,
        "private_winner_share": {names[i]: round(float((prv_win == i).mean()), 4) for i in range(ns)},
        "public_winner_share": {names[i]: round(float((pub_win == i).mean()), 4) for i in range(ns)},
        "P_public_winner_is_private_winner": round(float((pub_win == prv_win).mean()), 4),
        "pairwise": rev,
    }


def adversarial(rng, fam, arr, cov, keys, n_search=40000):
    """Worst/best case over family x tercile cell reweighting inside a box:
    each cell's weight may move to between 1/4x and 4x its observed share."""
    out = {}
    for cname in ("density", "displacement", "ratio", "mass"):
        tc = terciles(cov[cname], fam)
        cell = (fam == "6bba").astype(int) * 3 + tc      # 6 cells
        ncell = 6
        counts = np.array([(cell == c).sum() for c in range(ncell)], float)
        base = counts / counts.sum()
        lo, hi = base / 4.0, base * 4.0
        U = rng.uniform(lo, hi, size=(n_search, ncell))
        U /= U.sum(1, keepdims=True)
        M = (U / counts[None, :])[:, cell] * len(fam)
        s = scores_all(arr, M)
        out[cname] = {
            SYSTEMS[k]: {"adv_min": round(float(s[i].min()), 5),
                         "adv_p1": round(pct(s[i], 1), 5),
                         "adv_max": round(float(s[i].max()), 5)}
            for i, k in enumerate(keys)}
        out[cname]["winner_share_over_simplex"] = winner_share(s, keys)
        # worst-case gap of every system vs E0c
        out[cname]["worst_gap_vs_E0c"] = {
            SYSTEMS[k]: round(float((s[i] - s[0]).min()), 5)
            for i, k in enumerate(keys) if k != "A"}
        out[cname]["best_gap_vs_E0c"] = {
            SYSTEMS[k]: round(float((s[i] - s[0]).max()), 5)
            for i, k in enumerate(keys) if k != "A"}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--out", type=Path,
                    default=Path(r"C:\Users\aryaa\Documents\Biohub-CellTracking-2026_RESEARCH"
                                 r"\agent_runs\agent5\private_split_sim.json"))
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    main(a.draws, a.seed, a.out)
