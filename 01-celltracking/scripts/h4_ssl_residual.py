"""H4-S: SELF-SUPERVISED normal-continuation residual as a division anomaly score.

Motivation
----------
Every supervised division attempt in this campaign is starved: only 92 realisable
positive dividing mothers exist (16 in 44b6, 76 in 6bba).  At 10 events-per-variable
the 44b6 direction supports ~1.6 parameters, and H1-M measured the consequence
directly (fit on 44b6 transfers at AUC 0.630; fit on 6bba transfers at 0.842).

This module stops depending on positive labels.  The ORDINARY CONTINUATION
population is enormous and unlabelled-but-known: on the frozen H0c metric-visible
mother universe there are 97,592 events of which 137 are dividers (0.14%).  We fit
what NORMAL one-to-one continuation looks like on that population -- with no labels
at all -- and score DEVIATION from it.

    y            = T(descriptor at t+1)                      (17 dims)
    yhat         = Ridge( T(d_t), T(d_t-1), T(d_t-2), context )
    r            = y - yhat
    z            = r / robust_scale(r on the TRAINING family)
    ssl_split    = <z, u>   with u an A-PRIORI physical mitotic direction
    ssl_struct   = <z, u_struct>  (one-to-two STRUCTURE only, no mass-collapse term)
    ssl_mag      = ||z||_2  (pure novelty, direction free)

Nothing above reads a division label.  Labels enter in exactly two places, both
cross-fitted: (a) the admission-rate scalar, fitted on the OTHER family; (b) the
diagnostic `ssl_orient` arm, which is reported but never led with.

Why this is not H1-M with extra steps
-------------------------------------
H1-M's `R_*_p1` / `D_*_p1` block IS the residual of the trivial predictor
yhat = T(d_t).  This module keeps that as the `diff0` baseline, so the incremental
value of *learning the conditional mean of normal continuation* is measured, not
assumed.  The learned predictor removes context (local density, track age, speed,
imaging depth, developmental time, and the node's own 3-frame history) from the
residual; the hand-built ratio cannot.

Measurement rule
----------------
PRIMARY IS EXACT POOLED.  division_jaccard is micro-pooled by the official scorer
(`vendor/.../metrics.py:519`), and adj_edge_jaccard is edge-volume weighted, so the
pooled currency is

    divJ_pooled          = (k_44b6 + k_6bba) / (151 + m_44b6 + m_6bba)
    dcomposite_pooled    = 0.1 * (divJ_pooled - 0.00484262) + POOLED_EDGE_COST

with the family edge-mass shares 0.148633 / 0.851367 measured exactly from
artifacts/kaggle/coupled_cache/scores/A__*.json (151,615 total edge volume).
Family numbers are diagnostics only.

Usage
-----
  .venv\\Scripts\\python.exe scripts\\h4_ssl_residual.py build-events \\
      --feat-dir <node_feats_mv> --comp <comp_mother_events.parquet> \\
      --out <ssl_events.parquet>

  .venv\\Scripts\\python.exe scripts\\h4_ssl_residual.py evaluate \\
      --events <ssl_events.parquet> --comp <comp_mother_events.parquet> \\
      --out <ssl_eval.json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h1m_features as F  # noqa: E402
import h1m_gate as G  # noqa: E402
import h1i_node_appearance as h1i  # noqa: E402

# ----------------------------------------------------------------- constants

# Exact pooled weights, measured from the authoritative E0c per-crop scores.
POOLED_EDGE_SHARE = {"44b6": 0.148633, "6bba": 0.851367}
POOLED_G = 151                       # 26 + 125 annotated GT divisions
POOLED_J_BASE = 0.00484262           # 4 / (4 + 675 + 147)
POOLED_EDGE_COST = (POOLED_EDGE_SHARE["44b6"] * G.EDGE_COST["44b6"]
                    + POOLED_EDGE_SHARE["6bba"] * G.EDGE_COST["6bba"])
DIV_W = 0.1                          # SCORE_DIVISION_WEIGHT
# H0c oracle pooled ceiling, for the "30% of upside" gate.
POOLED_ORACLE = DIV_W * (92 / POOLED_G - POOLED_J_BASE) + POOLED_EDGE_COST

# 17 self-normalising descriptors, in a frozen order.
DESC = ["massn", "conc", "contrast", "saddle_self", "peakn",          # log-transformed
        "rg", "spher", "aniso", "linearity", "planarity", "detn",
        "kurt", "skew", "coff", "p10", "p20", "p30"]
LOG_DESC = set(F.RATIO_FEATS)        # massn, conc, contrast, saddle_self, peakn
EPS = 1e-6

# Label-free, deployment-observable context. NO family id, NO crop id, NO raw
# intensity level (H1-I established raw intensity flips sign between families).
CONTEXT = ["local_density", "log_track_age", "speed_um", "z_um", "t_norm"]

# A-PRIORI mitotic direction. Signs are fixed by the measured competition
# signature (forward one-step core-mass collapse, radius-of-gyration growth) and
# by the physics of one-to-two splitting. No label was consulted to choose them.
U_APRIORI = {
    "massn": -1.0, "conc": -1.0, "contrast": -1.0, "saddle_self": -1.0,
    "peakn": -1.0, "rg": +1.0, "coff": +1.0, "kurt": -1.0,
    "p10": +1.0, "p20": +1.0, "p30": +1.0,
}
# STRUCTURE-ONLY direction: deliberately excludes every core-mass term that H1-M
# already exploits, so a gain here cannot be a relabelling of m_massratio_p1.
U_STRUCT = {"rg": +1.0, "coff": +1.0, "kurt": -1.0, "p20": +1.0, "p30": +1.0}


def cfg_hash() -> str:
    payload = {"desc": DESC, "log": sorted(LOG_DESC), "context": CONTEXT,
               "u_apriori": U_APRIORI, "u_struct": U_STRUCT,
               "radii": [h1i.R_INNER, h1i.R_CORE, h1i.R_MID, h1i.R_OUTER],
               "n_core_vox": int(h1i.BND["core"]), "unit": "mother_time_event",
               "target_dt": 1, "input_dts": [0, -1, -2], "version": 1}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


# ------------------------------------------------------------- event building

def build_events(feat_dir: Path, comp_path: Path, out: Path) -> dict:
    """Join the per-node/per-dt appearance cache onto the frozen mother-event
    universe, producing one row per event with the 17 descriptors at each of the
    five frames. No ground truth is read here beyond the labels already carried
    by the mother-event table (used for EVALUATION only)."""
    t0 = time.time()
    comp = pd.read_parquet(comp_path, columns=[
        "crop", "family", "fold", "mother", "t", "label", "realisable",
        "hard_negative", "mother_gt_outdeg", "n_frames_seen", "local_density",
        "track_age", "speed_um", "z_um", "n_cand", "best_resid_um", "inside_volume"])
    n_core = float(h1i.BND["core"])

    frames = []
    files = sorted(Path(feat_dir).rglob("*.parquet"))
    for p in files:
        nf = pq.read_table(p).to_pandas()
        d = F.node_derived(nf, n_core)
        d["node_id"] = nf.node_id.to_numpy()
        d["dt"] = nf.dt.to_numpy()
        d["crop"] = nf.crop.to_numpy()
        frames.append(d)
    feats = pd.concat(frames, ignore_index=True)

    key = comp[["crop", "mother"]].copy()
    out_df = comp.copy()
    for dt in (-2, -1, 0, 1, 2):
        sub = feats[feats.dt == dt][["crop", "node_id"] + DESC]
        sub = sub.rename(columns={c: f"{c}@{dt}" for c in DESC})
        out_df = out_df.merge(sub, left_on=["crop", "mother"],
                              right_on=["crop", "node_id"], how="left")
        out_df = out_df.drop(columns=["node_id"])

    out_df["t_norm"] = out_df.t.to_numpy(float) / 100.0
    out_df["log_track_age"] = np.log1p(out_df.track_age.to_numpy(float))
    out_df["cfg_hash"] = cfg_hash()
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, compression="zstd", index=False)

    have = out_df[[f"massn@{dt}" for dt in (-2, -1, 0, 1, 2)]].notna().all(axis=1)
    return {"n_events": int(len(out_df)), "n_crops": len(files),
            "n_complete_desc": int(have.sum()),
            "n_dividers": int((out_df.label == 1).sum()),
            "n_realisable": int(out_df.realisable.sum()),
            "unmatched_key_rows": int(len(key) - len(out_df)),
            "cfg_hash": cfg_hash(), "seconds": round(time.time() - t0, 1)}


# --------------------------------------------------------------- transforms

def T(df: pd.DataFrame, dt: int) -> np.ndarray:
    """Frozen descriptor transform: log for ratio-like quantities, identity for
    bounded shape quantities. Chosen so that (T(d_{t+1}) - T(d_t)) is EXACTLY the
    H1-M `R_*_p1` / `D_*_p1` block -- the learned model is therefore a strict
    generalisation of the hand-built difference, and the delta is attributable."""
    cols = []
    for c in DESC:
        v = df[f"{c}@{dt}"].to_numpy(float)
        cols.append(np.log(np.maximum(v, EPS)) if c in LOG_DESC else v)
    return np.column_stack(cols)


def robust_scale(R: np.ndarray) -> np.ndarray:
    q1, q3 = np.nanpercentile(R, [25, 75], axis=0)
    return np.maximum((q3 - q1) / 1.349, 1e-6)


def clean(X: np.ndarray, lo=-12.0, hi=12.0) -> np.ndarray:
    return np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), lo, hi)


def ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    n, d = X.shape
    X1 = np.hstack([X, np.ones((n, 1))])
    A = X1.T @ X1
    A[np.arange(d), np.arange(d)] += lam        # intercept unpenalised
    return np.linalg.solve(A, X1.T @ Y)


def design(df: pd.DataFrame) -> np.ndarray:
    """Inputs to the normal-continuation predictor: the node's own 3-frame
    descriptor history plus label-free context."""
    blocks = [T(df, 0), T(df, -1), T(df, -2)]
    ctx = np.column_stack([df[c].to_numpy(float) for c in CONTEXT])
    return clean(np.hstack(blocks + [ctx]))


def direction(u: dict) -> np.ndarray:
    v = np.array([u.get(c, 0.0) for c in DESC], dtype=float)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------- SSL scores

def ssl_scores(fit: pd.DataFrame, ev: pd.DataFrame, lam: float) -> dict:
    """Fit the normal-continuation predictor on `fit` (NO labels used) and score
    `ev`. `fit` and `ev` must be from disjoint embryo families."""
    Xf, Yf = design(fit), clean(T(fit, 1))
    mu, sd = Xf.mean(0), np.maximum(Xf.std(0), 1e-6)
    W = ridge((Xf - mu) / sd, Yf, lam)

    def predict(df):
        X = (design(df) - mu) / sd
        return np.hstack([X, np.ones((len(X), 1))]) @ W

    Rf = Yf - predict(fit)
    s_ssl = robust_scale(Rf)

    # trivial-predictor control: yhat = T(d_t). Its residual IS the H1-M p1 block.
    R0f = Yf - clean(T(fit, 0))
    s_dif = robust_scale(R0f)

    Ye = clean(T(ev, 1))
    Z_ssl = clean((Ye - predict(ev)) / s_ssl, -20, 20)
    Z_dif = clean((Ye - clean(T(ev, 0))) / s_dif, -20, 20)

    ua, us = direction(U_APRIORI), direction(U_STRUCT)
    return {
        "ssl_split": Z_ssl @ ua,
        "ssl_struct": Z_ssl @ us,
        "ssl_mag": np.linalg.norm(Z_ssl, axis=1),
        "diff0_split": Z_dif @ ua,
        "diff0_struct": Z_dif @ us,
        "_Z_ssl": Z_ssl, "_Z_fit": clean((Rf) / s_ssl, -20, 20),
        "_r2": float(1.0 - (Rf ** 2).sum() / max(((Yf - Yf.mean(0)) ** 2).sum(), 1e-9)),
        "_r2_diff0": float(1.0 - (R0f ** 2).sum() / max(((Yf - Yf.mean(0)) ** 2).sum(), 1e-9)),
    }


# -------------------------------------------------------------- pooled scoring

def pooled_point(rows: dict) -> dict:
    """rows: {family: (k, m)} -> micro-pooled division Jaccard and pooled composite."""
    k = sum(v[0] for v in rows.values())
    m = sum(v[1] for v in rows.values())
    J = k / (POOLED_G + m)
    return {"k": int(k), "m": int(m), "divJ_pooled": J,
            "dcomposite_pooled": DIV_W * (J - POOLED_J_BASE) + POOLED_EDGE_COST,
            "frac_of_h0c_oracle": (DIV_W * (J - POOLED_J_BASE) + POOLED_EDGE_COST)
            / POOLED_ORACLE}


def pooled_best(per_fam: dict) -> dict:
    """Optimistic pooled ceiling: sweep a common admission RATE across both
    families simultaneously (one global knob, not one per family)."""
    best = None
    rates = np.unique(np.concatenate([
        np.linspace(1e-5, 0.02, 400), np.linspace(0.02, 0.35, 200)]))
    for rate in rates:
        rows = {}
        for fam, (df, s) in per_fam.items():
            K = int(max(1, round(rate * len(s))))
            order = np.argsort(-s)[:K]
            kk = int(df.realisable.to_numpy()[order].sum())
            rows[fam] = (kk, K - kk)
        r = pooled_point(rows)
        r["rate"] = float(rate)
        if best is None or r["divJ_pooled"] > best["divJ_pooled"]:
            best = r
    return best


def pooled_crossfit(per_fam: dict) -> dict:
    """HONEST pooled point: the admission rate applied to each family is the rate
    that was optimal on the OTHER family. Never chosen on the family it scores."""
    opt = {}
    for fam, (df, s) in per_fam.items():
        order = np.argsort(-s)
        real = df.realisable.to_numpy()[order].astype(int)
        kk = np.cumsum(real)
        K = np.arange(1, len(order) + 1)
        J = kk / (G.GT_DIVISIONS[fam] + (K - kk))
        opt[fam] = float(K[int(np.argmax(J))]) / len(order)
    rows, detail = {}, {}
    for fam, other in (("44b6", "6bba"), ("6bba", "44b6")):
        if fam not in per_fam or other not in per_fam:
            continue
        df, s = per_fam[fam]
        K = int(max(1, round(opt[other] * len(s))))
        order = np.argsort(-s)[:K]
        kk = int(df.realisable.to_numpy()[order].sum())
        rows[fam] = (kk, K - kk)
        detail[fam] = {"K": K, "rate_from": other, "rate": opt[other],
                       "TP_realisable": kk, "FP": K - kk,
                       "divJ_family": kk / (G.GT_DIVISIONS[fam] + K - kk),
                       "dcomposite_family": DIV_W * (kk / (G.GT_DIVISIONS[fam] + K - kk)
                                                     - G.J_BASE[fam]) + G.EDGE_COST[fam]}
    out = pooled_point(rows)
    out["per_family"] = detail
    return out


# ------------------------------------------------------------ FP-set analysis

def fp_overlap(df: pd.DataFrame, sa: np.ndarray, sb: np.ndarray, K: int) -> dict:
    """Do two scorers make the SAME mistakes? Compare the admitted FALSE-POSITIVE
    sets at a matched admission budget K, and test whether requiring BOTH to admit
    (the monotone AND rule) removes FPs faster than it removes TPs."""
    real = df.realisable.to_numpy()
    A = np.argsort(-sa)[:K]
    B = np.argsort(-sb)[:K]
    fa, fb = set(A[~real[A]].tolist()), set(B[~real[B]].tolist())
    ta, tb = set(A[real[A]].tolist()), set(B[real[B]].tolist())
    inter, union = fa & fb, fa | fb
    ra = pd.Series(-sa).rank().to_numpy()
    rb = pd.Series(-sb).rank().to_numpy()
    neg = ~real
    return {
        "K": int(K),
        "FP_a": len(fa), "FP_b": len(fb),
        "FP_shared": len(inter), "FP_union": len(union),
        "FP_jaccard": len(inter) / max(len(union), 1),
        "FP_unique_to_a": len(fa - fb), "FP_unique_to_b": len(fb - fa),
        "TP_a": len(ta), "TP_b": len(tb), "TP_shared": len(ta & tb),
        "AND_TP": len(ta & tb), "AND_FP": len(inter),
        "spearman_on_negatives": float(np.corrcoef(ra[neg], rb[neg])[0, 1]),
        "spearman_all": float(np.corrcoef(ra, rb)[0, 1]),
    }


def rank_avg(*scores: np.ndarray) -> np.ndarray:
    """Monotone combination: mean of per-scorer percentile ranks. Uses no labels
    and no fitted weight, so it cannot overfit 92 positives."""
    return np.mean([pd.Series(s).rank(pct=True).to_numpy() for s in scores], axis=0)


# ------------------------------------------------------------------ evaluate

def evaluate(events: Path, comp_path: Path, lam: float, out: Path) -> dict:
    t0 = time.time()
    ev = pd.read_parquet(events)
    ev = ev[ev.label.isin([0, 1]) & (ev.n_frames_seen == 5)].reset_index(drop=True)
    ok = ev[[f"{c}@{dt}" for c in DESC for dt in (-2, -1, 0, 1, 2)]].notna().all(axis=1)
    ev = ev[ok].reset_index(drop=True)

    comp = pd.read_parquet(comp_path)
    comp = G.labelled(comp)
    comp_key = comp.set_index(["crop", "mother"])

    res = {"cfg_hash": cfg_hash(), "lam": lam,
           "pooled_constants": {"G": POOLED_G, "J_base": POOLED_J_BASE,
                                "edge_cost": POOLED_EDGE_COST,
                                "edge_share": POOLED_EDGE_SHARE,
                                "h0c_oracle_pooled": POOLED_ORACLE},
           "universe": {"n": int(len(ev)),
                        "n_divider": int((ev.label == 1).sum()),
                        "n_realisable": int(ev.realisable.sum())},
           "families": {}, "scorers": {}}

    # ---- H1-M cross-family gate on the SAME rows (labels used; cross-fitted) ----
    h1m_score = {}
    for test_fam, fit_fam in (("44b6", "6bba"), ("6bba", "44b6")):
        tr = comp[comp.family == fit_fam]
        st = G.Standardiser(tr[F.H1M_FEATURES].to_numpy(float))
        w = G.fit_logistic(st(tr[F.H1M_FEATURES].to_numpy(float)),
                           tr.label.to_numpy(float), l2=30.0)
        te_idx = ev.index[ev.family == test_fam]
        te = comp_key.loc[list(zip(ev.crop[te_idx], ev.mother[te_idx]))]
        h1m_score[test_fam] = G.score(w, st(te[F.H1M_FEATURES].to_numpy(float)))

    per_fam_scores: dict[str, dict[str, np.ndarray]] = {}
    for test_fam, fit_fam in (("44b6", "6bba"), ("6bba", "44b6")):
        te = ev[ev.family == test_fam].reset_index(drop=True)
        fit = ev[ev.family == fit_fam].reset_index(drop=True)
        s = ssl_scores(fit, te, lam)
        # H1-I hand-built single feature: mother core-mass ratio at t+1, sign a priori.
        h1i_single = -(np.log(np.maximum(te["massn@1"].to_numpy(float), EPS))
                       - np.log(np.maximum(te["massn@0"].to_numpy(float), EPS)))
        sc = {
            "ssl_split": s["ssl_split"],
            "ssl_struct": s["ssl_struct"],
            "ssl_mag": s["ssl_mag"],
            "diff0_split": s["diff0_split"],
            "diff0_struct": s["diff0_struct"],
            "h1i_massratio": h1i_single,
            "h1m_cross": h1m_score[test_fam],
        }
        sc["combo_ssl_h1m"] = rank_avg(sc["ssl_split"], sc["h1m_cross"])
        sc["combo_struct_h1m"] = rank_avg(sc["ssl_struct"], sc["h1m_cross"])
        sc["combo_ssl_h1i"] = rank_avg(sc["ssl_split"], sc["h1i_massratio"])
        sc["combo_all"] = rank_avg(sc["ssl_split"], sc["ssl_struct"],
                                   sc["h1m_cross"], sc["h1i_massratio"])
        per_fam_scores[test_fam] = sc
        res["families"][test_fam] = {
            "n": int(len(te)), "n_divider": int((te.label == 1).sum()),
            "n_realisable": int(te.realisable.sum()),
            "fit_on": fit_fam, "n_fit": int(len(fit)),
            "n_fit_contaminating_dividers": int((fit.label == 1).sum()),
            "contamination_rate": float((fit.label == 1).sum() / max(len(fit), 1)),
            "ncp_train_r2": s["_r2"], "diff0_train_r2": s["_r2_diff0"],
        }
        res["families"][test_fam]["_frame"] = te

    # ---------------------------------------------------------- per scorer
    names = list(per_fam_scores["44b6"].keys())
    for nm in names:
        entry = {"family": {}}
        per_fam = {}
        for fam in ("44b6", "6bba"):
            te = res["families"][fam]["_frame"]
            s = np.asarray(per_fam_scores[fam][nm], dtype=float)
            y = te.label.to_numpy()
            entry["family"][fam] = {
                "auc_divider": G.auc(s[y == 1], s[y == 0]),
                "auc_realisable": G.auc(s[te.realisable.to_numpy()], s[y == 0]),
                "best_point_optimistic": G.best_operating_point(te, s, fam),
            }
            per_fam[fam] = (te, s)
        entry["pooled_optimistic"] = pooled_best(per_fam)
        entry["pooled_crossfitted"] = pooled_crossfit(per_fam)
        entry["family_crossfitted"] = G.cross_fit_budget(
            per_fam, {f: entry["family"][f] for f in per_fam})
        res["scorers"][nm] = entry

    # ------------------------------------------------- FP complementarity
    res["fp_overlap"] = {}
    pairs = [("ssl_split", "h1m_cross"), ("ssl_struct", "h1m_cross"),
             ("ssl_split", "h1i_massratio"), ("ssl_struct", "h1i_massratio"),
             ("ssl_split", "diff0_split"), ("ssl_struct", "ssl_split"),
             ("h1m_cross", "h1i_massratio")]
    for a, b in pairs:
        res["fp_overlap"][f"{a}|{b}"] = {}
        for fam in ("44b6", "6bba"):
            te = res["families"][fam]["_frame"]
            K = res["scorers"]["h1m_cross"]["pooled_crossfitted"]["per_family"][fam]["K"]
            res["fp_overlap"][f"{a}|{b}"][fam] = fp_overlap(
                te, np.asarray(per_fam_scores[fam][a], float),
                np.asarray(per_fam_scores[fam][b], float), K)

    for fam in ("44b6", "6bba"):
        res["families"][fam].pop("_frame")
    res["seconds"] = round(time.time() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-events")
    b.add_argument("--feat-dir", required=True)
    b.add_argument("--comp", required=True)
    b.add_argument("--out", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--events", required=True)
    e.add_argument("--comp", required=True)
    e.add_argument("--lam", type=float, default=30.0)
    e.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.cmd == "build-events":
        r = build_events(Path(a.feat_dir), Path(a.comp), Path(a.out))
    else:
        r = evaluate(Path(a.events), Path(a.comp), a.lam, Path(a.out))
        r = {k: v for k, v in r.items() if k != "fp_overlap"}
    print(json.dumps(r, indent=2, default=float))


if __name__ == "__main__":
    main()
