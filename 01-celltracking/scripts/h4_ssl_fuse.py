"""H4-S stage 2: fuse the self-supervised residual with H0c candidate geometry and
with the existing gates, and measure everything on the EXACT POOLED objective.

Three questions, in order:

  1. COST CURVE.  The pooled currency is  divJ = k / (151 + m), so what matters is
     not AUC but the pooled Pareto frontier -- for every attainable number of
     realisable true mothers k, the minimum number of admitted false mothers m.
     This computes it exactly with a 2-D per-family threshold sweep (the optimistic
     ceiling) and with leave-one-family-out rates (the honest point).

  2. COMPLEMENTARITY.  Does the SSL residual remove DIFFERENT false positives than
     the existing gates?  Measured as (a) FP-set Jaccard against a chance baseline,
     (b) whether a monotone AND / rank-average rule beats both parents on the
     pooled frontier.

  3. GEOMETRY FUSION.  H0c already ranks each mother's candidate pairs by the
     flow-midpoint residual.  `best_resid_um` is a deployment-observable geometric
     quality for the mother.  Fusing it with the appearance residual is the
     combination the H0c cascade actually admits.

Two edge-cost conventions are reported side by side and never mixed:

  conservative  the GT-FREE suppress-all control composite delta
                (-0.00004 / -0.00204), which is what scripts/h1m_gate.py uses.
  h0c_measured  the adjEdgeJ effect actually measured in the frozen H0c replay
                (+0.0010 / -0.0006), the arithmetic that reproduces +0.0625/+0.0597.

Usage:
  .venv\\Scripts\\python.exe scripts\\h4_ssl_fuse.py \\
      --events <ssl_events.parquet> --comp <comp_mother_events.parquet> \\
      --out <ssl_fuse.json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h1m_features as F  # noqa: E402
import h1m_gate as G  # noqa: E402
import h4_ssl_residual as S  # noqa: E402

EDGE_CONV = {
    "conservative": {"44b6": -0.00004, "6bba": -0.00204},
    "h0c_measured": {"44b6": +0.00100, "6bba": -0.00060},
}
FAMS = ("44b6", "6bba")


def pooled_edge(conv: str) -> float:
    e = EDGE_CONV[conv]
    return sum(S.POOLED_EDGE_SHARE[f] * e[f] for f in FAMS)


def dcomp(J: float, conv: str) -> float:
    return S.DIV_W * (J - S.POOLED_J_BASE) + pooled_edge(conv)


def km_curve(real: np.ndarray, s: np.ndarray, grid: np.ndarray) -> tuple:
    """(k, m) at each admission budget in `grid` for one family."""
    order = np.argsort(-s, kind="stable")
    r = real[order].astype(int)
    ck = np.concatenate([[0], np.cumsum(r)])
    K = np.clip(grid, 0, len(r))
    return ck[K], K - ck[K]


def pooled_frontier(per_fam: dict, n_grid: int = 320) -> dict:
    """Exact pooled optimum over INDEPENDENT per-family thresholds (optimistic:
    both thresholds are chosen on the data they are scored on)."""
    ks, ms = {}, {}
    for f, (real, s) in per_fam.items():
        g = np.unique(np.concatenate([
            np.arange(0, 201),
            np.unique(np.round(np.geomspace(200, max(len(s), 201), n_grid)).astype(int))]))
        ks[f], ms[f] = km_curve(real, s, g)
    K1 = ks["44b6"][:, None] + ks["6bba"][None, :]
    M1 = ms["44b6"][:, None] + ms["6bba"][None, :]
    J = K1 / (S.POOLED_G + M1)
    i, j = np.unravel_index(int(np.argmax(J)), J.shape)
    return {"k": int(K1[i, j]), "m": int(M1[i, j]), "divJ_pooled": float(J[i, j]),
            "k_44b6": int(ks["44b6"][i]), "m_44b6": int(ms["44b6"][i]),
            "k_6bba": int(ks["6bba"][j]), "m_6bba": int(ms["6bba"][j]),
            **{f"dcomposite_{c}": dcomp(float(J[i, j]), c) for c in EDGE_CONV}}


def pooled_lofo(per_fam: dict) -> dict:
    """Honest pooled point: each family's admission RATE is the rate that was
    optimal on the OTHER family. No threshold is ever chosen on its own family."""
    opt = {}
    for f, (real, s) in per_fam.items():
        order = np.argsort(-s, kind="stable")
        r = real[order].astype(int)
        ck = np.cumsum(r)
        K = np.arange(1, len(r) + 1)
        J = ck / (G.GT_DIVISIONS[f] + (K - ck))
        opt[f] = float(K[int(np.argmax(J))]) / len(r)
    k = m = 0
    detail = {}
    for f, other in (("44b6", "6bba"), ("6bba", "44b6")):
        real, s = per_fam[f]
        K = int(max(1, round(opt[other] * len(s))))
        order = np.argsort(-s, kind="stable")[:K]
        kk = int(real[order].sum())
        k += kk
        m += K - kk
        detail[f] = {"K": K, "rate_from": other, "rate": opt[other],
                     "k": kk, "m": K - kk}
    J = k / (S.POOLED_G + m)
    return {"k": k, "m": m, "divJ_pooled": J, "per_family": detail,
            **{f"dcomposite_{c}": dcomp(J, c) for c in EDGE_CONV}}


def and_rule(per_fam_a: dict, per_fam_b: dict, n_grid: int = 120) -> dict:
    """Monotone AND: a mother is admitted only if BOTH scorers rank it inside their
    own budget. If the two scorers err independently this removes FPs quadratically
    while removing TPs only linearly."""
    grids = {}
    for f in FAMS:
        real, sa = per_fam_a[f]
        _, sb = per_fam_b[f]
        n = len(real)
        g = np.unique(np.round(np.geomspace(5, n, n_grid)).astype(int))
        ra = np.argsort(np.argsort(-sa, kind="stable"), kind="stable")
        rb = np.argsort(np.argsort(-sb, kind="stable"), kind="stable")
        rows = []
        for ka in g:
            inA = ra < ka
            for kb in g:
                sel = inA & (rb < kb)
                kk = int(real[sel].sum())
                rows.append((kk, int(sel.sum()) - kk))
        rows = np.array(rows)
        # Pareto: for each k keep min m
        best = {}
        for kk, mm in rows:
            if kk not in best or mm < best[kk]:
                best[kk] = mm
        grids[f] = np.array(sorted(best.items()))
    K1 = grids["44b6"][:, 0][:, None] + grids["6bba"][:, 0][None, :]
    M1 = grids["44b6"][:, 1][:, None] + grids["6bba"][:, 1][None, :]
    J = K1 / (S.POOLED_G + M1)
    i, j = np.unravel_index(int(np.argmax(J)), J.shape)
    return {"k": int(K1[i, j]), "m": int(M1[i, j]), "divJ_pooled": float(J[i, j]),
            **{f"dcomposite_{c}": dcomp(float(J[i, j]), c) for c in EDGE_CONV}}


def and_rule_lofo(per_fam_a: dict, per_fam_b: dict, n_grid: int = 60) -> dict:
    """HONEST AND rule. The pair of admission RATES is optimised on one family and
    applied unchanged to the other, then the two families are pooled. No threshold
    is ever selected on the family it is scored on."""
    def ranks(s):
        return np.argsort(np.argsort(-s, kind="stable"), kind="stable")

    R = {f: (per_fam_a[f][0], ranks(per_fam_a[f][1]), ranks(per_fam_b[f][1]))
         for f in FAMS}
    best_rate = {}
    for f in FAMS:
        real, ra, rb = R[f]
        n = len(real)
        g = np.unique(np.round(np.geomspace(5, n, n_grid)).astype(int))
        bj, br = -1.0, (g[-1], g[-1])
        for ka in g:
            inA = ra < ka
            for kb in g:
                sel = inA & (rb < kb)
                kk = int(real[sel].sum())
                mm = int(sel.sum()) - kk
                j = kk / (G.GT_DIVISIONS[f] + mm)
                if j > bj:
                    bj, br = j, (ka, kb)
        best_rate[f] = (br[0] / n, br[1] / n, bj)
    k = m = 0
    detail = {}
    for f, other in (("44b6", "6bba"), ("6bba", "44b6")):
        real, ra, rb = R[f]
        n = len(real)
        ka = int(max(1, round(best_rate[other][0] * n)))
        kb = int(max(1, round(best_rate[other][1] * n)))
        sel = (ra < ka) & (rb < kb)
        kk = int(real[sel].sum())
        mm = int(sel.sum()) - kk
        k += kk
        m += mm
        detail[f] = {"K_a": ka, "K_b": kb, "rate_from": other, "k": kk, "m": mm,
                     "admitted": int(sel.sum())}
    J = k / (S.POOLED_G + m)
    return {"k": k, "m": m, "divJ_pooled": J, "per_family": detail,
            **{f"dcomposite_{c}": dcomp(J, c) for c in EDGE_CONV}}


def chance_fp_jaccard(K: int, n_neg: int) -> float:
    """Expected FP-set Jaccard for two INDEPENDENT rankers admitting K each."""
    p = K / n_neg
    inter = K * p
    return inter / max(2 * K - inter, 1e-9)


def build_scores(ev: pd.DataFrame, lam: float) -> tuple[dict, dict]:
    """All scorers, per family, with the SSL predictor fitted leave-one-family-out
    (no labels) and the H1-M logistic fitted cross-family (labels, other family)."""
    per: dict[str, dict[str, np.ndarray]] = {}
    real: dict[str, np.ndarray] = {}
    meta = {}
    comp_cols = F.H1M_FEATURES

    for test_fam, fit_fam in (("44b6", "6bba"), ("6bba", "44b6")):
        te = ev[ev.family == test_fam].reset_index(drop=True)
        fit = ev[ev.family == fit_fam].reset_index(drop=True)
        s = S.ssl_scores(fit, te, lam)

        tr_c = te.attrs  # unused; H1-M features come from the comp table join below
        h1i = -(np.log(np.maximum(te["massn@1"].to_numpy(float), S.EPS))
                - np.log(np.maximum(te["massn@0"].to_numpy(float), S.EPS)))
        geom = -te.best_resid_um.to_numpy(float)          # H0c flow-midpoint quality
        ncand = te.n_cand.to_numpy(float)

        Xtr = fit[comp_cols].to_numpy(float)
        st = G.Standardiser(Xtr)
        w = G.fit_logistic(st(Xtr), fit.label.to_numpy(float), l2=30.0)
        h1m = G.score(w, st(te[comp_cols].to_numpy(float)))

        d = {"ssl_split": s["ssl_split"], "ssl_struct": s["ssl_struct"],
             "ssl_mag": s["ssl_mag"], "diff0_split": s["diff0_split"],
             "h1i_massratio": h1i, "h1m_cross": h1m,
             "geom_resid": geom, "n_cand": ncand}
        d["combo_ssl_geom"] = S.rank_avg(d["ssl_split"], geom)
        d["combo_mag_geom"] = S.rank_avg(d["ssl_mag"], geom)
        d["combo_ssl_h1m"] = S.rank_avg(d["ssl_split"], h1m)
        d["combo_ssl_h1i"] = S.rank_avg(d["ssl_split"], h1i)
        d["combo_ssl_h1m_geom"] = S.rank_avg(d["ssl_split"], h1m, geom)
        d["combo_ssl_h1i_geom"] = S.rank_avg(d["ssl_split"], h1i, geom)
        d["combo_h1m_geom"] = S.rank_avg(h1m, geom)
        d["combo_mag_h1m"] = S.rank_avg(d["ssl_mag"], h1m)
        d["combo_mag_h1i"] = S.rank_avg(d["ssl_mag"], h1i)
        d["combo_mag_h1m_h1i"] = S.rank_avg(d["ssl_mag"], h1m, h1i)

        per[test_fam] = d
        real[test_fam] = te.realisable.to_numpy()
        meta[test_fam] = {"n": int(len(te)), "n_neg": int((~te.realisable).sum()),
                          "n_realisable": int(te.realisable.sum()),
                          "n_divider": int((te.label == 1).sum()),
                          "ncp_r2": s["_r2"], "diff0_r2": s["_r2_diff0"],
                          "fit_on": fit_fam, "n_fit": int(len(fit))}
    return per, {"real": real, "meta": meta}


def bootstrap_point(crop: dict, real: dict, sel: dict, n_boot: int, seed: int) -> dict:
    """Crop-level bootstrap of an ALREADY-CHOSEN operating point. The rates are held
    fixed at their LOFO values; only the evaluation sample is resampled, so this
    measures how fragile k and m are given 92 positives -- not threshold variance."""
    rng = np.random.default_rng(seed)
    Js = np.empty(n_boot)
    for b in range(n_boot):
        k = m = 0
        for f in FAMS:
            cr = crop[f]
            uc = np.unique(cr)
            pick = rng.choice(uc, len(uc), replace=True)
            idx = {c: np.flatnonzero(cr == c) for c in uc}
            for c in pick:
                i = idx[c]
                s = sel[f][i]
                k += int((s & real[f][i]).sum())
                m += int((s & ~real[f][i]).sum())
        Js[b] = k / (S.POOLED_G + m)
    lo, hi = np.percentile(Js, [2.5, 97.5])
    return {"divJ_median": float(np.median(Js)), "divJ_ci95": [float(lo), float(hi)],
            "dcomposite_h0c_ci95": [dcomp(float(lo), "h0c_measured"),
                                    dcomp(float(hi), "h0c_measured")],
            "dcomposite_cons_ci95": [dcomp(float(lo), "conservative"),
                                     dcomp(float(hi), "conservative")],
            "frac_boot_above_breakeven_h0c": float(
                (Js > S.POOLED_J_BASE - pooled_edge("h0c_measured") / S.DIV_W).mean()),
            "frac_boot_above_breakeven_cons": float(
                (Js > S.POOLED_J_BASE - pooled_edge("conservative") / S.DIV_W).mean()),
            "n_boot": n_boot}


def and_selection(per_fam_a: dict, per_fam_b: dict, detail: dict) -> dict:
    """Reconstruct the boolean admission mask of the LOFO AND rule."""
    out = {}
    for f in FAMS:
        ra = np.argsort(np.argsort(-per_fam_a[f][1], kind="stable"), kind="stable")
        rb = np.argsort(np.argsort(-per_fam_b[f][1], kind="stable"), kind="stable")
        out[f] = (ra < detail[f]["K_a"]) & (rb < detail[f]["K_b"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--comp", required=True)
    ap.add_argument("--lam", type=float, default=30.0)
    ap.add_argument("--lam-sweep", nargs="*", type=float, default=None,
                    help="refit the SSL predictor at each ridge lambda and report "
                         "the LOFO ssl_split&geom_resid point")
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--export-admit", default=None,
                    help="write the (crop, mother) admission table of --admit-arm for "
                         "scripts/win_bet/h4_ssl_gate_replay.py")
    ap.add_argument("--admit-arm", default="ssl_split&geom_resid",
                    help="'A&B' for the LOFO AND rule, or a single scorer name")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    ev = pd.read_parquet(a.events)
    comp = pd.read_parquet(a.comp)[["crop", "mother"] + F.H1M_FEATURES]
    ev = ev.merge(comp, on=["crop", "mother"], how="left")
    ev = ev[ev.label.isin([0, 1]) & (ev.n_frames_seen == 5)].reset_index(drop=True)
    ok = ev[[f"{c}@{dt}" for c in S.DESC for dt in (-2, -1, 0, 1, 2)]].notna().all(axis=1)
    ev = ev[ok].reset_index(drop=True)

    per, aux = build_scores(ev, a.lam)
    real = aux["real"]

    res = {"cfg_hash": S.cfg_hash(), "lam": a.lam, "meta": aux["meta"],
           "edge_conventions": EDGE_CONV,
           "pooled_edge": {c: pooled_edge(c) for c in EDGE_CONV},
           "pooled_break_even_divJ": {
               c: S.POOLED_J_BASE - pooled_edge(c) / S.DIV_W for c in EDGE_CONV},
           "pooled_divJ_for_plus0.005": {
               c: S.POOLED_J_BASE + (0.005 - pooled_edge(c)) / S.DIV_W
               for c in EDGE_CONV},
           "h0c_oracle_pooled": {c: dcomp(92 / S.POOLED_G, c) for c in EDGE_CONV},
           "scorers": {}}

    for nm in per["44b6"]:
        pf = {f: (real[f], np.asarray(per[f][nm], float)) for f in FAMS}
        e = {"pooled_optimistic": pooled_frontier(pf),
             "pooled_lofo": pooled_lofo(pf), "auc": {}}
        for f in FAMS:
            s = np.asarray(per[f][nm], float)
            e["auc"][f] = G.auc(s[real[f]], s[~real[f]])
        res["scorers"][nm] = e

    # ------------------------------------------------ AND rule + complementarity
    res["and_rule"] = {}
    res["complementarity"] = {}
    pairs = [("ssl_split", "h1m_cross"), ("ssl_mag", "h1m_cross"),
             ("ssl_split", "h1i_massratio"), ("ssl_mag", "h1i_massratio"),
             ("ssl_split", "geom_resid"), ("ssl_mag", "geom_resid"),
             ("h1m_cross", "h1i_massratio"), ("h1m_cross", "geom_resid"),
             ("ssl_split", "diff0_split"),
             # appearance ENSEMBLE against geometry: still only two admission rates,
             # so the honest-transfer risk is the same as the two-scorer arms, but
             # the appearance side now spans all three decorrelated appearance axes.
             ("combo_mag_h1m_h1i", "geom_resid"), ("combo_ssl_h1i", "geom_resid"),
             ("combo_mag_h1m", "geom_resid")]
    for x, y in pairs:
        pa = {f: (real[f], np.asarray(per[f][x], float)) for f in FAMS}
        pb = {f: (real[f], np.asarray(per[f][y], float)) for f in FAMS}
        res["and_rule"][f"{x}&{y}"] = and_rule(pa, pb)
        res["and_rule"][f"{x}&{y}"]["lofo"] = and_rule_lofo(pa, pb)
        c = {}
        for f in FAMS:
            K = int(max(20, round(0.004 * len(real[f]))))
            sa, sb = np.asarray(per[f][x], float), np.asarray(per[f][y], float)
            A, B = np.argsort(-sa)[:K], np.argsort(-sb)[:K]
            fa = set(A[~real[f][A]].tolist())
            fb = set(B[~real[f][B]].tolist())
            n_neg = aux["meta"][f]["n_neg"]
            jac = len(fa & fb) / max(len(fa | fb), 1)
            c[f] = {"K": K, "FP_jaccard": jac,
                    "chance_FP_jaccard": chance_fp_jaccard(K, n_neg),
                    "ratio_to_chance": jac / max(chance_fp_jaccard(K, n_neg), 1e-12),
                    "spearman_neg": float(pd.Series(sa[~real[f]]).corr(
                        pd.Series(sb[~real[f]]), method="spearman")),
                    "TP_shared": len(set(A[real[f][A]].tolist())
                                     & set(B[real[f][B]].tolist())),
                    "TP_a": int(real[f][A].sum()), "TP_b": int(real[f][B].sum())}
        res["complementarity"][f"{x}|{y}"] = c

    # ---------------------------------------------------------- fragility
    if a.boot:
        crop = {f: ev[ev.family == f].crop.to_numpy() for f in FAMS}
        res["bootstrap"] = {}
        for nm in ("ssl_split", "h1m_cross", "h1i_massratio", "ssl_mag"):
            pf = {f: (real[f], np.asarray(per[f][nm], float)) for f in FAMS}
            lo = pooled_lofo(pf)
            sel = {}
            for f in FAMS:
                K = lo["per_family"][f]["K"]
                r = np.argsort(np.argsort(-pf[f][1], kind="stable"), kind="stable")
                sel[f] = r < K
            res["bootstrap"][nm] = bootstrap_point(crop, real, sel, a.boot, 11)
        for x, y in (("ssl_split", "geom_resid"), ("ssl_mag", "geom_resid"),
                     ("h1m_cross", "geom_resid"), ("ssl_split", "h1m_cross")):
            pa = {f: (real[f], np.asarray(per[f][x], float)) for f in FAMS}
            pb = {f: (real[f], np.asarray(per[f][y], float)) for f in FAMS}
            lo = and_rule_lofo(pa, pb)
            sel = and_selection(pa, pb, lo["per_family"])
            res["bootstrap"][f"{x}&{y}"] = bootstrap_point(crop, real, sel, a.boot, 11)

    if a.lam_sweep:
        res["lam_sweep"] = {}
        for lm in a.lam_sweep:
            p2, ax2 = build_scores(ev, lm)
            r2 = ax2["real"]
            pa = {f: (r2[f], np.asarray(p2[f]["ssl_split"], float)) for f in FAMS}
            pb = {f: (r2[f], np.asarray(p2[f]["geom_resid"], float)) for f in FAMS}
            res["lam_sweep"][str(lm)] = {
                "ssl_split_lofo": pooled_lofo(pa),
                "ssl_split&geom_lofo": and_rule_lofo(pa, pb),
                "auc": {f: G.auc(pa[f][1][r2[f]], pa[f][1][~r2[f]]) for f in FAMS}}

    # ------------------------------------------------- admission table export
    if a.export_admit:
        arm = a.admit_arm
        sel = {}
        if "&" in arm:
            x, y = arm.split("&", 1)
            pa = {f: (real[f], np.asarray(per[f][x], float)) for f in FAMS}
            pb = {f: (real[f], np.asarray(per[f][y], float)) for f in FAMS}
            lo = and_rule_lofo(pa, pb)
            sel = and_selection(pa, pb, lo["per_family"])
        else:
            pf = {f: (real[f], np.asarray(per[f][arm], float)) for f in FAMS}
            lo = pooled_lofo(pf)
            for f in FAMS:
                r = np.argsort(np.argsort(-pf[f][1], kind="stable"), kind="stable")
                sel[f] = r < lo["per_family"][f]["K"]
        keep = []
        for f in FAMS:
            sub = ev[ev.family == f].reset_index(drop=True)
            keep.append(sub.loc[sel[f], ["crop", "mother", "t", "family", "realisable"]])
        adm = pd.concat(keep, ignore_index=True)
        Path(a.export_admit).parent.mkdir(parents=True, exist_ok=True)
        adm.to_parquet(a.export_admit, index=False)
        res["admit_export"] = {"arm": arm, "path": a.export_admit,
                               "n_admitted": int(len(adm)),
                               "n_realisable": int(adm.realisable.sum()),
                               "lofo_point": lo}

    res["seconds"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("complementarity", "and_rule")},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
