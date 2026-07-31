"""H1-M2: does REALIGNING the external division time restore the mitotic signature,
and does the realigned corpus then transfer to the competition?

Consumes the temporal profiles written by `h1m2_profile.py` (one row per event x dt,
dt in [-4, +10] around the tracker's fork frame) and answers, in order:

  ab      the single measurement that decides the lane -- forward core-mass-collapse
          amplitude BEFORE realignment (anchor 0, stride 1, exactly lane C's setting)
          versus AFTER (anchor 0, per-event stride s = k*, the track-geometry
          realignment), with negatives evaluated at the SAME stride so the comparison
          is not a change of units.  Plus the full anchor x stride sweep, and an
          image-chosen (oracle) upper bound that says how much signature exists at
          ANY anchor -- if the oracle is also flat, no realignment rule can work.

  smd     feature-distribution distance to the competition BEFORE and AFTER temporal
          resampling: the resampling is supposed to move Zebrahub towards the
          competition, and this says whether it does.

  train   MOTHER-LEVEL HAZARD model -- P(this mother divides during the next
          competition-scale transition | its temporal patch).  Not a candidate-pair
          classifier: the pair ranker is already solved by frozen geometry.
          Evaluated external-embryo-held-out, then transferred to the competition and
          scored by exact-composite budget arithmetic under a cross-fitted rule.

Everything is stride-aware: a Zebrahub feature block built at stride s is the same
PHYSICAL transition as a competition feature block at stride 1, which is what makes the
transfer well posed.

Usage:
  .venv\\Scripts\\python.exe scripts\\h1m2_gate.py --mode ab \
      --profiles prof_*.parquet --comp comp_mother_events.parquet --out ab.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h1m_features as F  # noqa: E402

EPS = 1e-6
DT_LO, DT_HI = -4, 10
MAX_STRIDE = 5                       # 2*s must stay inside DT_HI


# ------------------------------------------------------------------ utilities
def load_profiles(patterns: list[str]) -> pd.DataFrame:
    files: list[str] = []
    for p in patterns:
        files += sorted(glob.glob(p))
    if not files:
        raise SystemExit(f"no profile parquet matched {patterns}")
    out = []
    for f in files:
        d = pd.read_parquet(f)
        d["src"] = Path(f).stem
        d["uid"] = d.src + "|" + d.ev.astype(str)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def pivot(prof: pd.DataFrame, col: str) -> pd.DataFrame:
    """uid x dt matrix of one derived statistic."""
    return prof.pivot_table(index="uid", columns="dt", values=col, aggfunc="first")


def effect(p: np.ndarray, n: np.ndarray) -> float:
    p = np.asarray(p, float)[np.isfinite(p)]
    n = np.asarray(n, float)[np.isfinite(n)]
    if p.size < 5 or n.size < 5:
        return float("nan")
    sd = np.sqrt(((p.size - 1) * p.var(ddof=1) + (n.size - 1) * n.var(ddof=1))
                 / (p.size + n.size - 2))
    return float((p.mean() - n.mean()) / max(sd, 1e-9))


def auc(score: np.ndarray, y: np.ndarray) -> float:
    s = np.asarray(score, float)
    ok = np.isfinite(s)
    s, y = s[ok], np.asarray(y)[ok]
    if y.sum() == 0 or (1 - y).sum() == 0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------------ A/B
def mode_ab(prof: pd.DataFrame, comp: pd.DataFrame) -> dict:
    meta = prof.drop_duplicates("uid").set_index("uid")[
        ["embryo", "label", "stride", "realigned", "sep_p1_um",
         "sep_at_stride_um", "max_sep_um", "t"]]
    M = np.log(np.maximum(pivot(prof, "massn"), EPS))          # log core mass
    RG = pivot(prof, "rg")
    SA = np.log(np.maximum(pivot(prof, "saddle_self"), EPS))
    M, RG, SA = (X.reindex(meta.index) for X in (M, RG, SA))
    y = meta.label.to_numpy()
    s_star = meta.stride.to_numpy()

    def fwd(X, a, s):
        if (a + s) not in X.columns or a not in X.columns:
            return np.full(len(X), np.nan)
        return X[a + s].to_numpy() - X[a].to_numpy()

    rep: dict = {"n_events": int(len(meta)),
                 "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}

    # -------- competition reference, same statistic, at stride 1 and 2 ----------
    ref = {}
    for fam, g in comp.groupby("family"):
        p, n = g[g.mother_gt_outdeg >= 2], g[g.mother_gt_outdeg == 1]
        ref[fam] = {
            "n_pos": int(len(p)),
            "R_massn_p1": effect(p.R_massn_p1, n.R_massn_p1),
            "R_massn_p2": effect(p.R_massn_p2, n.R_massn_p2),
            "D_rg_p1": effect(p.D_rg_p1, n.D_rg_p1),
            "R_saddle_self_p1": effect(p.R_saddle_self_p1, n.R_saddle_self_p1),
        }
    rep["competition_reference"] = ref

    # -------- BEFORE: lane C's setting exactly (anchor 0, stride 1) -------------
    before = {}
    for emb, idx in meta.groupby("embryo").groups.items():
        sel = meta.index.get_indexer(idx)
        yy = y[sel]
        before[emb] = {
            "n_pos": int((yy == 1).sum()),
            "R_massn_p1": effect(fwd(M, 0, 1)[sel][yy == 1], fwd(M, 0, 1)[sel][yy == 0]),
            "D_rg_p1": effect(fwd(RG, 0, 1)[sel][yy == 1], fwd(RG, 0, 1)[sel][yy == 0]),
            "R_saddle_p1": effect(fwd(SA, 0, 1)[sel][yy == 1], fwd(SA, 0, 1)[sel][yy == 0]),
            "auc_R_massn_p1": auc(-fwd(M, 0, 1)[sel], (yy == 1).astype(int)),
        }
    rep["before"] = before

    # -------- AFTER: anchor 0, per-event stride s = k*, stride-matched negatives --
    def realigned_stat(X, sign=-1.0):
        """Standardise each positive against the negatives at ITS OWN stride, so
        mixing strides cannot manufacture (or destroy) an effect."""
        zs, es, keep_s = [], [], []
        for s in range(1, MAX_STRIDE + 1):
            v = fwd(X, 0, s)
            neg = v[(y == 0) & np.isfinite(v)]
            if neg.size < 20:
                continue
            mu, sd = neg.mean(), neg.std(ddof=1)
            m = (y == 1) & (s_star == s) & np.isfinite(v)
            if m.sum() == 0:
                continue
            zs.append((v[m] - mu) / max(sd, 1e-9))
            es.append(np.full(int(m.sum()), s))
            keep_s.append(s)
        if not zs:
            return {}
        z = np.concatenate(zs)
        return {"effect_pooled": float(z.mean()), "n": int(z.size),
                "se": float(z.std(ddof=1) / np.sqrt(z.size)),
                "by_stride": {int(s): float(np.concatenate(zs)[np.concatenate(es) == s].mean())
                              for s in keep_s}}

    after = {}
    for emb in sorted(meta.embryo.unique()):
        e = meta.embryo.to_numpy() == emb
        yy, ss = y.copy(), s_star.copy()
        yy[~e] = -9
        sub = {}
        for name, X in (("R_massn", M), ("D_rg", RG), ("R_saddle", SA)):
            zs, ns = [], []
            for s in range(1, MAX_STRIDE + 1):
                v = fwd(X, 0, s)
                neg = v[e & (y == 0) & np.isfinite(v)]
                pos = v[e & (y == 1) & (s_star == s) & np.isfinite(v)]
                if neg.size < 20 or pos.size == 0:
                    continue
                zs.append((pos - neg.mean()) / max(neg.std(ddof=1), 1e-9))
                ns.append(pos.size)
            if zs:
                z = np.concatenate(zs)
                sub[name] = {"effect": float(z.mean()), "n": int(z.size),
                             "se": float(z.std(ddof=1) / np.sqrt(z.size))}
        # stride-matched AUC on the realigned subset
        aucs = []
        for s in range(1, MAX_STRIDE + 1):
            v = fwd(M, 0, s)
            m = e & np.isfinite(v) & (((y == 1) & (s_star == s)) | (y == 0))
            if (m & (y == 1)).sum() < 10:
                continue
            aucs.append((s, int((m & (y == 1)).sum()),
                         auc(-v[m], (y[m] == 1).astype(int))))
        sub["auc_by_stride"] = aucs
        after[emb] = sub
    rep["after_realigned"] = after

    # -------- anchor x stride sweep, all positives (no realignment rule) --------
    sweep = {}
    for emb in sorted(meta.embryo.unique()):
        e = meta.embryo.to_numpy() == emb
        tab = {}
        for a in range(-2, 5):
            for s in range(1, MAX_STRIDE + 1):
                if not (DT_LO <= a and a + s <= DT_HI):
                    continue
                v = fwd(M, a, s)
                tab[f"a{a}_s{s}"] = effect(v[e & (y == 1)], v[e & (y == 0)])
        sweep[emb] = tab
    rep["anchor_stride_sweep_R_massn"] = sweep

    # -------- ORACLE: per-event best forward drop over all (a, s) --------------
    # NOT deployable. It bounds how much mitotic signature the imagery contains at
    # ANY label time; if this is flat there is no anchor worth finding.
    cand = [(a, s) for a in range(-2, 5) for s in range(1, MAX_STRIDE + 1)
            if DT_LO <= a and a + s <= DT_HI]
    V = np.stack([fwd(M, a, s) for a, s in cand], axis=1)
    with np.errstate(invalid="ignore"):
        best = np.nanmin(V, axis=1)
    rep["oracle_best_drop"] = {
        emb: {"effect": effect(best[(meta.embryo.to_numpy() == emb) & (y == 1)],
                               best[(meta.embryo.to_numpy() == emb) & (y == 0)]),
              "auc": auc(-best[meta.embryo.to_numpy() == emb],
                         (y[meta.embryo.to_numpy() == emb] == 1).astype(int))}
        for emb in sorted(meta.embryo.unique())}

    # -------- separation-stratified (the "approach 10 um" instruction) ---------
    strat = {}
    sep = meta.sep_at_stride_um.to_numpy()
    for emb in sorted(meta.embryo.unique()):
        e = meta.embryo.to_numpy() == emb
        rows = {}
        for lo, hi in ((0, 8), (8, 12), (12, 99)):
            zs = []
            for s in range(1, MAX_STRIDE + 1):
                v = fwd(M, 0, s)
                neg = v[e & (y == 0) & np.isfinite(v)]
                m = e & (y == 1) & (s_star == s) & np.isfinite(v) & (sep >= lo) & (sep < hi)
                if neg.size < 20 or m.sum() == 0:
                    continue
                zs.append((v[m] - neg.mean()) / max(neg.std(ddof=1), 1e-9))
            if zs:
                z = np.concatenate(zs)
                rows[f"sep_{lo}_{hi}"] = {"effect": float(z.mean()), "n": int(z.size)}
        strat[emb] = rows
    rep["sep_stratified"] = strat
    return rep


# ------------------------------------------------------- stride-aware features
def build_block(prof: pd.DataFrame, anchor: int, stride_col: str | int,
                clamp_back: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the frozen 77-dim H1M block from a profile at (anchor, stride).

    `stride_col` is either a fixed int stride or the name of a per-event column.
    Backward offsets are clamped into the fetched window when `clamp_back`; the
    offsets actually used are returned so the clamp is never silent.
    """
    keep = ["embryo", "label", "stride", "realigned", "sep_p1_um", "sep_at_stride_um",
            "max_sep_um", "t", "mother_track_id"]
    if isinstance(stride_col, str) and stride_col not in keep:
        keep.append(stride_col)
    meta = prof.drop_duplicates("uid").set_index("uid")[keep]
    stats = {c: pivot(prof, c).reindex(meta.index) for c in F.NODE_FEATS}
    n = len(meta)
    s = (np.full(n, int(stride_col)) if isinstance(stride_col, int)
         else meta[stride_col].to_numpy())
    s = np.where(np.isfinite(s) & (s >= 1), s, 1).astype(int)
    s = np.clip(s, 1, MAX_STRIDE)

    offs = {0: np.full(n, anchor), 1: anchor + s, 2: anchor + 2 * s,
            -1: anchor - s, -2: anchor - 2 * s}
    if clamp_back:
        offs[-1] = np.maximum(offs[-1], DT_LO)
        offs[-2] = np.maximum(offs[-2], DT_LO)
    for k in offs:
        offs[k] = np.clip(offs[k], DT_LO, DT_HI)

    by_dt = {}
    for k, off in offs.items():
        d = pd.DataFrame(index=meta.index)
        for c, X in stats.items():
            cols = np.asarray(X.columns)
            pos = np.searchsorted(cols, off)
            pos = np.clip(pos, 0, len(cols) - 1)
            A = X.to_numpy()
            d[c] = A[np.arange(n), pos]
        by_dt[k] = d.reset_index(drop=True)
    feats = F.mother_event_features(by_dt, pd.RangeIndex(n))
    feats.index = meta.index
    meta = meta.copy()
    meta["stride_used"] = s
    return feats, meta


# ------------------------------------------------------------------------ SMD
def mode_smd(prof: pd.DataFrame, comp: pd.DataFrame) -> dict:
    """Distance to the competition BEFORE and AFTER temporal resampling.

    Measured on the DIVIDERS (the class whose label time is in question) and, as a
    control, on the non-dividers.  Negatives carry no k*, so for the resampled arm they
    are given strides drawn from the positive k* distribution -- otherwise the two arms
    would be numerically identical on that class and the comparison would be vacuous.
    """
    out = {}
    prof = prof.copy()
    pos_k = prof.loc[(prof.label == 1) & prof.realigned
                     & (prof.stride >= 1) & (prof.stride <= MAX_STRIDE), "stride"]
    pos_k = pos_k.drop_duplicates().to_numpy() if pos_k.empty else pos_k.to_numpy()
    rng = np.random.default_rng(20260731)
    uids = prof.uid.unique()
    draw = pd.Series(rng.choice(pos_k, size=len(uids)), index=uids)
    prof["stride_resampled"] = np.where(
        (prof.label == 1) & prof.realigned & (prof.stride >= 1),
        prof.stride, prof.uid.map(draw))

    for cls, cmask, emask in (("dividers", comp.mother_gt_outdeg >= 2, 1),
                              ("non_dividers", comp.mother_gt_outdeg == 1, 0)):
        cref = comp[cmask]
        for name, sc in (("before_stride1", 1), ("after_realigned", "stride_resampled")):
            feats, meta = build_block(prof, 0, sc)
            ext = feats[meta.label.to_numpy() == emask]
            smd = {}
            for c in F.H1M_FEATURES:
                a, b = ext[c].to_numpy(float), cref[c].to_numpy(float)
                a, b = a[np.isfinite(a)], b[np.isfinite(b)]
                if a.size < 10 or b.size < 10:
                    continue
                sd = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
                smd[c] = float(abs(a.mean() - b.mean()) / max(sd, 1e-9))
            v = np.array(list(smd.values()))
            out[f"{cls}__{name}"] = {
                "n_external": int(len(ext)), "n_competition": int(len(cref)),
                "mean_abs_smd": float(v.mean()), "median": float(np.median(v)),
                "n_over_1sd": int((v > 1).sum()), "n_features": int(v.size),
                "worst": sorted(smd.items(), key=lambda kv: -kv[1])[:6]}
    return out


# ---------------------------------------------------------------------- train
def _fit_logistic(X, y, l2=1.0, iters=200):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    Xs = sc.fit_transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))
    m = LogisticRegression(C=1.0 / l2, max_iter=iters, class_weight="balanced")
    m.fit(Xs, y)
    return sc, m


def _apply(sc, m, X):
    return m.decision_function(sc.transform(
        np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)))


def mode_train(prof: pd.DataFrame, comp: pd.DataFrame, arm: str) -> dict:
    sc_col: int | str = 1 if arm == "unaligned" else "stride"
    feats, meta = build_block(prof, 0, sc_col)
    cols = [c for c in F.H1M_FEATURES if c in feats.columns]
    X = feats[cols].to_numpy(float)
    y = (meta.label.to_numpy() == 1).astype(int)
    emb = meta.embryo.to_numpy()
    if arm == "realigned":                    # only forks with a realigned anchor
        keep = (y == 0) | (meta.realigned.to_numpy() & (meta.stride_used <= MAX_STRIDE))
        X, y, emb = X[keep], y[keep], emb[keep]

    rep = {"arm": arm, "n": int(len(y)), "n_pos": int(y.sum()),
           "features": len(cols)}
    loeo = {}
    for e in sorted(set(emb)):
        tr, te = emb != e, emb == e
        if y[tr].sum() < 20 or y[te].sum() < 5:
            continue
        s, m = _fit_logistic(X[tr], y[tr])
        loeo[e] = {"n_fit": int(tr.sum()), "n_pos_fit": int(y[tr].sum()),
                   "n_test": int(te.sum()), "n_pos_test": int(y[te].sum()),
                   "auc": auc(_apply(s, m, X[te]), y[te])}
    rep["external_loeo"] = loeo
    rep["external_loeo_mean_auc"] = float(np.nanmean(
        [v["auc"] for v in loeo.values()])) if loeo else float("nan")

    # ---- frozen model on ALL external, transferred to the competition ---------
    # Composite arithmetic is H0c's own measured conversion, unchanged:
    #   divJ = TP / (G + K - TP),  d(composite) ~= 0.10 * (divJ - divJ_base) + edge_cost
    G = {"44b6": 26, "6bba": 125}
    J_BASE = {"44b6": 0.0000, "6bba": 0.0057}
    EDGE_COST = {"44b6": -0.00004, "6bba": -0.00204}
    s, m = _fit_logistic(X, y)
    tr_out, curves = {}, {}
    for fam, g in comp.groupby("family"):
        Xc = g[cols].to_numpy(float)
        yd = (g.mother_gt_outdeg >= 2).astype(int).to_numpy()
        yr = g.realisable.astype(int).to_numpy()
        sc_ = _apply(s, m, Xc)
        order = np.argsort(-sc_)
        cum = np.cumsum(yr[order])                 # only realisable can become a TP
        Ks = np.unique(np.clip(np.round(np.geomspace(10, 20000, 60)).astype(int),
                               1, len(order)))
        j = np.array([cum[K - 1] / (G[fam] + K - cum[K - 1]) for K in Ks])
        best = int(np.argmax(j))
        curves[fam] = {"K": Ks.tolist(), "divJ": j.tolist(),
                       "tp": cum[Ks - 1].tolist(),
                       "thresh": sc_[order][Ks - 1].tolist()}
        tr_out[fam] = {
            "n": int(len(g)), "n_div": int(yd.sum()), "n_realisable": int(yr.sum()),
            "auc_divider": auc(sc_, yd), "auc_realisable": auc(sc_, yr),
            "top100_tp": int(yd[order[:100]].sum()),
            "top500_tp": int(yd[order[:500]].sum()),
            "best_K": int(Ks[best]), "best_tp": int(cum[Ks[best] - 1]),
            "best_divJ": float(j[best]),
            "best_dcomposite": float(0.10 * (j[best] - J_BASE[fam]) + EDGE_COST[fam]),
            "divJ_base": J_BASE[fam], "edge_cost": EDGE_COST[fam]}
    # cross-fitted: the admission RATE is chosen on one family, applied to the other
    fams = list(tr_out)
    if len(fams) == 2:
        for a_, b_ in ((0, 1), (1, 0)):
            fa, fb = fams[a_], fams[b_]
            rate = tr_out[fb]["best_K"] / tr_out[fb]["n"]
            K = max(1, int(round(rate * tr_out[fa]["n"])))
            Ks = np.asarray(curves[fa]["K"])
            i = int(np.argmin(np.abs(Ks - K)))
            j = curves[fa]["divJ"][i]
            tr_out[fa]["crossfit_K"] = int(Ks[i])
            tr_out[fa]["crossfit_tp"] = int(curves[fa]["tp"][i])
            tr_out[fa]["crossfit_divJ"] = float(j)
            tr_out[fa]["crossfit_dcomposite"] = float(
                0.10 * (j - J_BASE[fa]) + EDGE_COST[fa])
    rep["competition_transfer"] = tr_out
    rep["coef_l2"] = float(np.linalg.norm(m.coef_))
    rep["top_features"] = sorted(
        zip(cols, m.coef_[0].tolist()), key=lambda kv: -abs(kv[1]))[:12]
    return rep


# ------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ab", "smd", "train"], required=True)
    ap.add_argument("--profiles", nargs="+", required=True)
    ap.add_argument("--comp", required=True)
    ap.add_argument("--arm", choices=["unaligned", "realigned"], default="realigned")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    prof = load_profiles(a.profiles)
    comp = pd.read_parquet(a.comp)
    if a.mode == "ab":
        rep = mode_ab(prof, comp)
    elif a.mode == "smd":
        rep = mode_smd(prof, comp)
    else:
        rep = {arm: mode_train(prof, comp, arm) for arm in ("unaligned", "realigned")}
    rep["feature_hash"] = F.config_hash()
    Path(a.out).write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps(rep, indent=2, default=float))


if __name__ == "__main__":
    main()
