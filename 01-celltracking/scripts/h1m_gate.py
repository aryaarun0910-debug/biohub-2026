"""H1-M: train / evaluate the MOTHER-DIVISION GATE and convert its operating point
into the only currency that matters -- exact-composite headroom.

The gate answers P(this mother divides at t -> t+1). It is NOT a candidate-pair
ranker, and it is never evaluated on pair rows.

Scoring arithmetic (from the frozen H0c replay, reports/NEXT_DECISION.md):
    division Jaccard  J = k / (G + m)
        G = annotated GT divisions in the family      26 (44b6) / 125 (6bba)
        k = admitted mothers that are TRUE dividers whose true daughter pair is on
            the frozen top-3 shortlist  ("realisable")
        m = every other admitted mother (non-divider, or divider reconstructed with
            the wrong pair) -- each one is a division false positive
    composite delta ~= 0.10 * (J - J_base) + edge_cost
        0.10 is H0c's own measured conversion (J 0.6154 -> +0.0625; 0.6080 -> +0.0597)
        edge_cost is suppression's UNCONDITIONAL cost, -0.00004 (44b6) / -0.00204 (6bba)
        J_base = 0.0000 / 0.0057 (E0c)
Promotion needs >= +0.005 on both families.

Modes:
  comp-crossfamily  in-domain upper bound: fit one embryo family, test the other.
                    Uses competition labels for FITTING and is therefore a DIAGNOSTIC
                    CEILING, not a deployable model.
  external          fit on Zebrahub with leave-one-embryo-out, freeze, then transfer
                    to the competition with at most one calibrated scalar (threshold).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h1m_features as F  # noqa: E402

GT_DIVISIONS = {"44b6": 26, "6bba": 125}
EDGE_COST = {"44b6": -0.00004, "6bba": -0.00204}
J_BASE = {"44b6": 0.0000, "6bba": 0.0057}
DJ_DCOMPOSITE = 0.10
KS = (25, 50, 100, 250, 500, 1000, 2500)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


class Standardiser:
    def __init__(self, X: np.ndarray):
        self.mu = np.nanmedian(X, axis=0)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        self.sd = np.maximum((q3 - q1) / 1.349, 1e-6)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        Z = np.clip(np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), -8, 8)
        return Z


def fit_logistic(X, y, l2=1.0, iters=400, w_pos=None):
    """Class-balanced L2 logistic regression, plain Newton/IRLS with damping."""
    n, d = X.shape
    X1 = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    sw = np.ones(n)
    if w_pos is None:
        w_pos = (y == 0).sum() / max((y == 1).sum(), 1)
    sw[y == 1] = w_pos
    reg = l2 * np.eye(d + 1)
    reg[-1, -1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X1 @ w, -30, 30)))
        g = X1.T @ (sw * (y - p)) - reg @ w
        s = np.maximum(sw * p * (1 - p), 1e-9)
        H = (X1 * s[:, None]).T @ X1 + reg
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def score(w, X):
    if callable(w):                      # gbdt arm
        return w(X)
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ w


def fit_model(kind: str, Z, y, l2: float):
    """Two parameterizations only: the primary cross-fitted L2 logistic, and one
    confirmatory shallow gradient-boosted arm."""
    if kind == "logistic":
        return fit_logistic(Z, y, l2=l2)
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(
        max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0,
        min_samples_leaf=40, class_weight="balanced", random_state=0)
    m.fit(Z, y)
    return lambda X: m.predict_proba(X)[:, 1]


def operating_table(df: pd.DataFrame, s: np.ndarray, family: str) -> list[dict]:
    """Rank mothers by gate score and convert each admission budget K into divJ."""
    order = np.argsort(-s)
    real = df.realisable.to_numpy()[order]
    lab = (df.label.to_numpy() == 1)[order]
    G = GT_DIVISIONS[family]
    rows = []
    for K in KS:
        if K > len(order):
            break
        k = int(real[:K].sum())
        m = K - k
        J = k / (G + m)
        rows.append({
            "K": K, "TP_realisable": k, "TP_divider": int(lab[:K].sum()), "FP": m,
            "precision_realisable": k / K, "divJ": J,
            "composite_delta": DJ_DCOMPOSITE * (J - J_BASE[family]) + EDGE_COST[family],
        })
    return rows


def best_operating_point(df: pd.DataFrame, s: np.ndarray, family: str) -> dict:
    """Sweep every admission budget and return the one maximising divJ (an
    optimistic upper bound -- the threshold is chosen on the test family itself)."""
    order = np.argsort(-s)
    real = df.realisable.to_numpy()[order].astype(int)
    G = GT_DIVISIONS[family]
    k = np.cumsum(real)
    K = np.arange(1, len(order) + 1)
    J = k / (G + (K - k))
    i = int(np.argmax(J))
    return {"K": int(K[i]), "TP_realisable": int(k[i]), "FP": int(K[i] - k[i]),
            "divJ": float(J[i]),
            "composite_delta": DJ_DCOMPOSITE * (float(J[i]) - J_BASE[family]) + EDGE_COST[family]}


def admit(df: pd.DataFrame, s: np.ndarray, family: str, K: int) -> dict:
    """Score a FIXED admission budget K -- the deployable form of the rule."""
    K = int(max(1, min(K, len(s))))
    order = np.argsort(-s)[:K]
    k = int(df.realisable.to_numpy()[order].sum())
    m = K - k
    J = k / (GT_DIVISIONS[family] + m)
    return {"K": K, "TP_realisable": k, "TP_divider": int((df.label.to_numpy()[order] == 1).sum()),
            "FP": m, "divJ": J,
            "composite_delta": DJ_DCOMPOSITE * (J - J_BASE[family]) + EDGE_COST[family]}


def labelled(df: pd.DataFrame, complete_only: bool = True) -> pd.DataFrame:
    d = df[df.label.isin([0, 1])]
    if complete_only:
        d = d[d.n_frames_seen == 5]
    return d.reset_index(drop=True)


def comp_crossfamily(comp: pd.DataFrame, feats: list[str], l2: float,
                     model: str = "logistic") -> dict:
    out = {"mode": "comp-crossfamily", "features": len(feats), "l2": l2,
           "model": model, "folds": {}}
    raw = {}
    for test_fam, fit_fam in (("44b6", "6bba"), ("6bba", "44b6")):
        tr = labelled(comp[comp.family == fit_fam])
        te = labelled(comp[comp.family == test_fam])
        Xtr, Xte = tr[feats].to_numpy(float), te[feats].to_numpy(float)
        st = Standardiser(Xtr)
        w = fit_model(model, st(Xtr), tr.label.to_numpy(float), l2)
        s = score(w, st(Xte))
        y = te.label.to_numpy()
        r = {
            "fit_on": fit_fam, "n_fit": int(len(tr)), "n_fit_pos": int((tr.label == 1).sum()),
            "n_test": int(len(te)), "n_test_divider": int((y == 1).sum()),
            "n_test_realisable": int(te.realisable.sum()),
            "auc_divider": auc(s[y == 1], s[y == 0]),
            "auc_realisable": auc(s[te.realisable.to_numpy()], s[y == 0]),
            "operating": operating_table(te, s, test_fam),
            "best_point_optimistic": best_operating_point(te, s, test_fam),
        }
        out["folds"][test_fam] = r
        raw[test_fam] = (te, s)
    out["cross_fitted_budget"] = cross_fit_budget(raw, out["folds"])
    return out


def comp_infamily_cv(comp: pd.DataFrame, feats: list[str], l2: float,
                     n_folds: int = 5, seed: int = 17, model: str = "logistic") -> dict:
    """CEILING PROBE, not a deployable model. Fits and tests INSIDE one embryo family
    with crop-grouped cross-validation, so the model gets abundant same-domain
    positives and only has to generalise across crops. If this cannot reach the
    promotion bar, the limit is the FEATURE SPACE, not the amount of training data,
    and no external corpus can rescue it."""
    rng = np.random.default_rng(seed)
    out = {"mode": "comp-infamily-cv", "folds": n_folds, "l2": l2,
           "model": model, "families": {}}
    raw = {}
    for fam, s in comp.groupby("family"):
        d = labelled(s)
        crops = np.array(sorted(d.crop.unique()))
        assign = {c: i for c, i in zip(crops, rng.permutation(len(crops)) % n_folds)}
        f = d.crop.map(assign).to_numpy()
        oof = np.full(len(d), np.nan)
        for k in range(n_folds):
            tr, te = d[f != k], np.flatnonzero(f == k)
            if (tr.label == 1).sum() == 0 or te.size == 0:
                continue
            st = Standardiser(tr[feats].to_numpy(float))
            w = fit_model(model, st(tr[feats].to_numpy(float)),
                          tr.label.to_numpy(float), l2)
            oof[te] = score(w, st(d.iloc[te][feats].to_numpy(float)))
        ok = np.isfinite(oof)
        d2, s2 = d[ok].reset_index(drop=True), oof[ok]
        y = d2.label.to_numpy()
        out["families"][fam] = {
            "n": int(len(d2)), "n_divider": int((y == 1).sum()),
            "n_realisable": int(d2.realisable.sum()),
            "auc_divider": auc(s2[y == 1], s2[y == 0]),
            "auc_realisable": auc(s2[d2.realisable.to_numpy()], s2[y == 0]),
            "operating": operating_table(d2, s2, fam),
            "best_point_optimistic": best_operating_point(d2, s2, fam),
        }
        raw[fam] = (d2, s2)
    out["cross_fitted_budget"] = cross_fit_budget(raw, out["families"])
    return out


def cross_fit_budget(raw: dict, res: dict) -> dict:
    """The only honest operating rule: the admission budget is a RATE fitted on one
    family and applied to the other, never chosen on the family it is scored on."""
    out = {}
    for fam, other in (("44b6", "6bba"), ("6bba", "44b6")):
        if fam not in raw or other not in raw:
            continue
        dfo, so = raw[other]
        rate = res[other]["best_point_optimistic"]["K"] / max(len(so), 1)
        df, s = raw[fam]
        r = admit(df, s, fam, int(round(rate * len(s))))
        r["rate_from"], r["rate"] = other, rate
        out[fam] = r
    return out


def single_feature_scan(comp: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    rows = []
    for c in feats:
        r = {"feature": c}
        for fam in ("44b6", "6bba"):
            d = labelled(comp[comp.family == fam])
            v = d[c].to_numpy(float)
            y = d.label.to_numpy()
            r[f"auc_{fam}"] = auc(v[y == 1], v[y == 0])
        a, b = r["auc_44b6"], r["auc_6bba"]
        r["sign_consistent"] = bool(np.isfinite(a) and np.isfinite(b)
                                    and np.sign(a - .5) == np.sign(b - .5))
        r["min_sep"] = min(abs(a - .5), abs(b - .5))
        rows.append(r)
    return pd.DataFrame(rows).sort_values("min_sep", ascending=False)


def external_mode(ext: pd.DataFrame, comp: pd.DataFrame, feats: list[str],
                  l2: float, model: str = "logistic") -> dict:
    out = {"mode": "external", "features": len(feats), "l2": l2, "model": model,
           "embryos": sorted(ext.embryo.unique().tolist()), "loeo": {}}
    E = sorted(ext.embryo.unique())
    for held in E:
        tr = labelled(ext[ext.embryo != held])
        te = labelled(ext[ext.embryo == held])
        if len(tr) == 0 or (te.label == 1).sum() == 0:
            continue
        assert set(tr.embryo) & set(te.embryo) == set(), "embryo leakage"
        Xtr = tr[feats].to_numpy(float)
        st = Standardiser(Xtr)
        w = fit_model(model, st(Xtr), tr.label.to_numpy(float), l2)
        s = score(w, st(te[feats].to_numpy(float)))
        y = te.label.to_numpy()
        out["loeo"][held] = {
            "n_fit": int(len(tr)), "n_fit_pos": int((tr.label == 1).sum()),
            "n_test": int(len(te)), "n_test_pos": int((y == 1).sum()),
            "auc": auc(s[y == 1], s[y == 0]),
            "fit_embryos": sorted(tr.embryo.unique().tolist()),
        }
    # frozen model: all external embryos
    tr = labelled(ext)
    Xtr = tr[feats].to_numpy(float)
    st = Standardiser(Xtr)
    w = fit_model(model, st(Xtr), tr.label.to_numpy(float), l2)
    out["frozen"] = {"n_fit": int(len(tr)), "n_fit_pos": int((tr.label == 1).sum())}
    out["transfer"] = {}
    for fam in ("44b6", "6bba"):
        te = labelled(comp[comp.family == fam])
        s = score(w, st(te[feats].to_numpy(float)))
        y = te.label.to_numpy()
        out["transfer"][fam] = {
            "n_test": int(len(te)), "n_divider": int((y == 1).sum()),
            "n_realisable": int(te.realisable.sum()),
            "auc_divider": auc(s[y == 1], s[y == 0]),
            "auc_realisable": auc(s[te.realisable.to_numpy()], s[y == 0]),
            "operating": operating_table(te, s, fam),
            "best_point_optimistic": best_operating_point(te, s, fam),
        }
    # ONE cross-fitted scalar -- the admission RATE fitted on the other family only
    raw = {fam: (labelled(comp[comp.family == fam]), None) for fam in ("44b6", "6bba")}
    for fam in raw:
        te = raw[fam][0]
        raw[fam] = (te, score(w, st(te[feats].to_numpy(float))))
    out["cross_fitted_budget"] = cross_fit_budget(
        raw, {f: out["transfer"][f] for f in raw})
    out["weights"] = {"coef": (w.tolist() if not callable(w) else "gbdt"),
                      "mu": st.mu.tolist(), "sd": st.sd.tolist(),
                      "feature_order": feats}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True)
    ap.add_argument("--external", nargs="*", default=[])
    ap.add_argument("--mode", default="comp-crossfamily",
                    choices=["comp-crossfamily", "comp-infamily-cv", "external", "scan"])
    ap.add_argument("--l2", type=float, default=30.0)
    ap.add_argument("--no-intensity", action="store_true")
    ap.add_argument("--model", default="logistic", choices=["logistic", "gbdt"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    comp = pd.read_parquet(a.comp)
    feats = F.H1M_NO_INTENSITY if a.no_intensity else F.H1M_FEATURES

    if a.mode == "scan":
        res = single_feature_scan(comp, feats)
        res.to_csv(a.out, index=False)
        print(res.head(30).to_string(index=False))
        return
    if a.mode == "comp-crossfamily":
        res = comp_crossfamily(comp, feats, a.l2, a.model)
    elif a.mode == "comp-infamily-cv":
        res = comp_infamily_cv(comp, feats, a.l2, model=a.model)
    else:
        ext = pd.concat([pd.read_parquet(p) for p in a.external], ignore_index=True)
        res = external_mode(ext, comp, feats, a.l2, a.model)
    res["feature_hash"] = F.config_hash()
    res["no_intensity"] = bool(a.no_intensity)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "weights"},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
