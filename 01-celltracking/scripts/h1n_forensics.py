"""H1-N Task 1 -- FORENSIC DECOMPOSITION of the recorded H1-M cross-fitted decisions,
plus the OVERLAP MATRIX between H1-M / H1-I / reverse-time / geometry.

Nothing here is fitted. The H1-M OOF scores are REPRODUCED bit-for-bit from the frozen
laneC configuration (l2=30, 5 crop-grouped folds, seed 17, logistic, 77 features,
feature hash 8697e2a779e2) by importing `h1m_gate` itself -- not re-implemented -- and the
reproduction is asserted against the recorded cross-fitted budget before any forensics run.

All rules come from `h1n_prereg.py` (hash printed below).

Usage:
  .venv\\Scripts\\python.exe scripts\\h1n_forensics.py ^
      --comp <...>\\laneC\\comp_mother_events.parquet ^
      --cands <...>\\agent3\\eval_v3\\candidates.parquet ^
      --recorded <...>\\laneC\\h1m_comp_infamily_cv.json ^
      --out <...>\\h1m_forensics\\forensics.json
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

FAMS = ("44b6", "6bba")
OTHER = {"44b6": "6bba", "6bba": "44b6"}


# --------------------------------------------------------------------------- #
# 1. reproduce the recorded H1-M OOF scores                                    #
# --------------------------------------------------------------------------- #
def h1m_oof(comp: pd.DataFrame, feats: list[str]) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    """Identical to h1m_gate.comp_infamily_cv's inner loop -- same rng, same fold
    assignment, same standardiser, same solver. Returns {family: (rows, oof_score)}."""
    rng = np.random.default_rng(P.H1M_SEED)
    out = {}
    for fam, s in comp.groupby("family"):
        d = G.labelled(s)
        crops = np.array(sorted(d.crop.unique()))
        assign = {c: i for c, i in zip(crops, rng.permutation(len(crops)) % P.H1M_FOLDS)}
        f = d.crop.map(assign).to_numpy()
        oof = np.full(len(d), np.nan)
        for k in range(P.H1M_FOLDS):
            tr, te = d[f != k], np.flatnonzero(f == k)
            if (tr.label == 1).sum() == 0 or te.size == 0:
                continue
            st = G.Standardiser(tr[feats].to_numpy(float))
            w = G.fit_model(P.H1M_MODEL, st(tr[feats].to_numpy(float)),
                            tr.label.to_numpy(float), P.H1M_L2)
            oof[te] = G.score(w, st(d.iloc[te][feats].to_numpy(float)))
        ok = np.isfinite(oof)
        out[fam] = (d[ok].reset_index(drop=True), oof[ok])
    return out


def crossfit_K(raw: dict) -> dict[str, int]:
    """The recorded honest rule: admission RATE = argmax-divJ rate on the OTHER family."""
    K = {}
    for fam in FAMS:
        do, so = raw[OTHER[fam]]
        rate = G.best_operating_point(do, so, OTHER[fam])["K"] / len(so)
        K[fam] = int(round(rate * len(raw[fam][1])))
    return K


# --------------------------------------------------------------------------- #
# 2. mother-level signal table                                                 #
# --------------------------------------------------------------------------- #
def equivalence_check(comp: pd.DataFrame, cands: pd.DataFrame) -> dict:
    """AMENDMENT 1 evidence: prove the full-coverage columns are the SAME signals as the
    97-crop H1-I / H1-G pair-table columns before substituting them."""
    c0 = cands.sort_values(["crop", "mother", "t", "rank"]).groupby(
        ["crop", "mother", "t"], as_index=False).first()
    j = comp.merge(c0[["crop", "mother", "t", "m_massratio_p1",
                       "G_flow_midpoint_residual", "G_competing_parents"]],
                   on=["crop", "mother", "t"])
    out = {"n_joint_rows": int(len(j))}
    for a, b in (("R_massn_p1", "m_massratio_p1"),
                 ("best_resid_um", "G_flow_midpoint_residual")):
        x, y = j[a].to_numpy(float), j[b].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        out[f"{a}__vs__{b}"] = {
            "spearman": round(float(pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman")), 6),
            "max_abs_diff": float(np.max(np.abs(x[ok] - y[ok]))),
        }
    out["steal_rank0__vs__competing_parents_gt0"] = float(
        (j.steal_rank0.astype(bool) == (j.G_competing_parents > 0)).mean())
    return out


def attach_signals(d: pd.DataFrame, s: np.ndarray, rtd: pd.DataFrame) -> pd.DataFrame:
    """All signals oriented HIGHER == MORE DIVIDER-LIKE (prereg section 2, amendment 1).
    Every signal has FULL 199-crop coverage."""
    d = d.copy()
    d["h1m_score"] = s
    d = d.merge(rtd, on=["crop", "mother", "t"], how="left")
    eps = 1e-9
    d["S_H1M"] = d.h1m_score
    d["S_H1I"] = -d.R_massn_p1
    # reverse-time disagreement, appearance channel: forward drop minus backward drop.
    d["S_RTD_A"] = -(np.log(np.clip(d.R_massn_p1, eps, None))
                     - np.log(np.clip(d.R_massn_m1, eps, None)))
    # reverse-time disagreement, association channel: the daughters' own best claimant
    d["S_RTD_G"] = d.rtd_margin
    d["S_GEOM"] = -d.best_resid_um
    return d


# --------------------------------------------------------------------------- #
# 3. error taxonomy                                                            #
# --------------------------------------------------------------------------- #
def classify(d: pd.DataFrame, admitted: np.ndarray) -> pd.DataFrame:
    """Priority order SUBSTRATE > TEMPORAL_PHASE > RANKING > CALIBRATION (prereg 1)."""
    d = d.copy()
    d["admitted"] = False
    d.loc[admitted, "admitted"] = True

    # GT division frames per predicted track (from mother_gt_outdeg on the same table)
    div = d[d.mother_gt_outdeg >= 2][["crop", "mother_track_id", "t"]]
    dmap: dict[tuple, list[int]] = {}
    for cr, tid, tt in div.itertuples(index=False):
        dmap.setdefault((cr, int(tid)), []).append(int(tt))

    # per crop: realisable mothers and their scores
    real = d[d.realisable]
    real_by_crop: dict[str, list[tuple[float, bool]]] = {}
    for cr, sc, adm in real[["crop", "h1m_score", "admitted"]].itertuples(index=False):
        real_by_crop.setdefault(cr, []).append((float(sc), bool(adm)))

    W = P.TEMPORAL_PHASE_WINDOW
    cls, dtt = [], []
    for r in d.itertuples(index=False):
        if not r.admitted:
            cls.append(""); dtt.append(np.nan); continue
        if r.realisable:
            cls.append("TP"); dtt.append(0.0); continue
        ts = dmap.get((r.crop, int(r.mother_track_id)), [])
        dt = min((abs(int(r.t) - x) for x in ts), default=np.nan)
        dtt.append(float(dt) if ts else np.nan)
        if r.label == 1:
            cls.append("SUBSTRATE")
        elif ts and dt <= W:
            cls.append("TEMPORAL_PHASE")
        elif any((not adm) and sc < r.h1m_score for sc, adm in real_by_crop.get(r.crop, [])):
            cls.append("RANKING")
        else:
            cls.append("CALIBRATION")
    d["err_class"] = cls
    d["frames_to_true_division"] = dtt
    return d


# --------------------------------------------------------------------------- #
# 4. overlap matrix                                                            #
# --------------------------------------------------------------------------- #
def overlap(dadm: pd.DataFrame, sigs: list[str], q: float) -> dict:
    """Preregistered: each signal drops the bottom `q` fraction of the ADMITTED set
    (within family). Overlap is measured on the FALSE POSITIVES each rule removes."""
    rem: dict[str, set] = {s: set() for s in sigs}
    keptTP: dict[str, int] = {s: 0 for s in sigs}
    for fam, g in dadm.groupby("family"):
        n_drop = int(round(q * len(g)))
        for s in sigs:
            v = g[s].to_numpy(float)
            v = np.where(np.isfinite(v), v, -np.inf)
            drop_idx = np.argsort(v, kind="stable")[:n_drop]
            keys = set(g.iloc[drop_idx].key)
            rem[s] |= keys
    fp_keys = set(dadm[~dadm.realisable].key)
    tp_keys = set(dadm[dadm.realisable].key)
    fp_rem = {s: rem[s] & fp_keys for s in sigs}
    tp_rem = {s: rem[s] & tp_keys for s in sigs}

    mat, union, excl = {}, {}, {}
    for a in sigs:
        mat[a] = {}
        for b in sigs:
            A, B = fp_rem[a], fp_rem[b]
            mat[a][b] = round(len(A & B) / max(len(A | B), 1), 4)
        union[a] = {b: len(fp_rem[a] | fp_rem[b]) for b in sigs}
        excl[a] = {b: len(fp_rem[a] - fp_rem[b]) for b in sigs}
    return {
        "drop_fraction": q,
        "n_admitted": int(len(dadm)), "n_admitted_TP": len(tp_keys),
        "fp_removed": {s: len(fp_rem[s]) for s in sigs},
        "tp_lost": {s: len(tp_rem[s]) for s in sigs},
        "jaccard_of_removed_FPs": mat,
        "union_removed_FPs": union,
        "exclusive_removed_FPs": excl,
        "spearman_on_admitted": {
            a: {b: round(float(pd.Series(dadm[a]).corr(pd.Series(dadm[b]), method="spearman")), 4)
                for b in sigs} for a in sigs},
    }


def signal_auc(d: pd.DataFrame, sigs: list[str]) -> dict:
    out = {}
    for fam, g in d.groupby("family"):
        y = g.realisable.to_numpy()
        out[fam] = {s: round(G.auc(g[s].to_numpy(float)[y], g[s].to_numpy(float)[~y]), 4)
                    for s in sigs}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True)
    ap.add_argument("--cands", required=True)
    ap.add_argument("--rtd", required=True)
    ap.add_argument("--recorded", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    print(f"PREREG_HASH   {P.prereg_hash()}")
    print(f"FEATURE_HASH  {F.config_hash()} (expected {P.H1M_FEATURE_HASH})")
    assert F.config_hash() == P.H1M_FEATURE_HASH, "feature representation drifted"

    comp = pd.read_parquet(a.comp)
    cands = pd.read_parquet(a.cands)
    feats = F.H1M_FEATURES
    raw = h1m_oof(comp, feats)
    K = crossfit_K(raw)

    rec = json.loads(Path(a.recorded).read_text())["cross_fitted_budget"]
    repro = {}
    for fam in FAMS:
        d, s = raw[fam]
        r = G.admit(d, s, fam, K[fam])
        repro[fam] = r
        e = rec[fam]
        ok = (r["K"] == e["K"] and r["TP_realisable"] == e["TP_realisable"]
              and r["FP"] == e["FP"])
        print(f"  reproduce {fam}: K={r['K']} TP={r['TP_realisable']} FP={r['FP']} "
              f"dcomp={r['composite_delta']:+.5f}   recorded K={e['K']} TP="
              f"{e['TP_realisable']} FP={e['FP']}  -> {'EXACT' if ok else 'MISMATCH'}")
        assert ok, f"cannot reproduce recorded H1-M decisions for {fam}"

    eqv = equivalence_check(comp, cands)
    print("  amendment-1 equivalence: " + json.dumps(eqv))
    rtd = pd.read_parquet(a.rtd)[["crop", "t", "mother", "rtd_margin", "rtd_nclaim",
                                  "rtd_loses", "r_alt_min"]]
    parts = []
    for fam in FAMS:
        d, s = raw[fam]
        d = attach_signals(d, s, rtd)
        order = np.argsort(-s)[:K[fam]]
        d = classify(d, order)
        d["key"] = d.crop + ":" + d.mother.astype(str) + ":" + d.t.astype(str)
        parts.append(d)
    D = pd.concat(parts, ignore_index=True)
    ADM = D[D.admitted].reset_index(drop=True)

    sigs = ["S_H1M", "S_H1I", "S_RTD_A", "S_RTD_G", "S_GEOM"]
    cov = {s: int(ADM[s].notna().sum()) for s in sigs}

    tax = (ADM.groupby(["family", "err_class"]).size().unstack(fill_value=0)
           .to_dict(orient="index"))
    tax_pooled = ADM.err_class.value_counts().to_dict()

    # regime profile of the admitted decisions
    def prof(g):
        return {"n": int(len(g)),
                "t_median": float(g.t.median()),
                "local_density_median": float(g.local_density.median()),
                "track_age_median": float(g.track_age.median()),
                "speed_um_median": float(g.speed_um.median()),
                "best_resid_um_median": float(g.best_resid_um.median()),
                "steal_rank0_frac": float(g.steal_rank0.mean()),
                "n_cand_median": float(g.n_cand.median())}
    regime = {"TP": prof(ADM[ADM.realisable]), "FP": prof(ADM[~ADM.realisable])}
    for c in sorted(set(ADM.err_class)):
        regime[f"FP::{c}"] = prof(ADM[ADM.err_class == c])
    regime["ALL_realisable_population"] = prof(D[D.realisable])
    regime["ALL_negatives_population"] = prof(D[~D.realisable])

    # time / density quartile distribution of admitted FPs vs TPs
    D["t_q"] = pd.qcut(D.t, 4, labels=False, duplicates="drop")
    D["dens_q"] = pd.qcut(D.local_density, 4, labels=False, duplicates="drop")
    A2 = D[D.admitted]
    byq = {
        "t_quartile": {"TP": A2[A2.realisable].t_q.value_counts().sort_index().to_dict(),
                       "FP": A2[~A2.realisable].t_q.value_counts().sort_index().to_dict()},
        "density_quartile": {"TP": A2[A2.realisable].dens_q.value_counts().sort_index().to_dict(),
                             "FP": A2[~A2.realisable].dens_q.value_counts().sort_index().to_dict()},
    }

    ov = overlap(ADM, sigs, P.OVERLAP_DROP_FRACTION)
    auc = signal_auc(D, sigs)

    res = {
        "prereg_hash": P.prereg_hash(), "feature_hash": F.config_hash(),
        "h0c_cfg_hash": P.H0C_CFG_HASH,
        "amendment1_equivalence": eqv,
        "reproduction": repro,
        "admitted_total": int(len(ADM)),
        "admitted_TP": int(ADM.realisable.sum()),
        "admitted_FP": int((~ADM.realisable).sum()),
        "precision_among_visible": float(ADM.realisable.mean()),
        "signal_coverage_on_admitted": cov,
        "taxonomy_by_family": tax, "taxonomy_pooled": tax_pooled,
        "regime": regime, "by_quartile": byq,
        "signal_auc_realisable_vs_rest": auc,
        "overlap": ov,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    ADM.to_parquet(str(Path(a.out).with_suffix(".admitted.parquet")))
    D[["key", "crop", "family", "t", "mother", "mother_track_id", "label", "realisable",
       "admitted", "err_class", "frames_to_true_division", "h1m_score", "steal_rank0",
       "local_density", "track_age", "speed_um", "best_resid_um", "n_cand"]
      + sigs].to_parquet(str(Path(a.out).with_suffix(".all.parquet")))
    print(json.dumps({k: v for k, v in res.items()
                      if k in ("admitted_total", "admitted_TP", "precision_among_visible",
                               "taxonomy_pooled", "taxonomy_by_family",
                               "signal_coverage_on_admitted")}, indent=2, default=float))
    print("\n--- overlap: Jaccard of removed FP sets (lower == more complementary) ---")
    print(pd.DataFrame(ov["jaccard_of_removed_FPs"]).to_string())
    print("\nFPs removed / TPs lost per signal:")
    print(pd.DataFrame({"fp_removed": ov["fp_removed"], "tp_lost": ov["tp_lost"]}).to_string())
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
