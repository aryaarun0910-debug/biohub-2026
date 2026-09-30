r"""Error atlas part 2: does ANY feature clear the break-even deletion-precision bar?

Takes ``feats_<tag>.parquet`` (from ``ea_features.py``) + ``crops_<tag>.parquet``
(from ``ea_atlas.py``) and, for every feature and every deletion budget:

  * ranks ALL emitted edges by the feature (a deployable global threshold),
  * deletes the worst k,
  * counts how many deleted edges were scored-FP (gain), scored-TP (loss), free (neutral),
  * recomputes the EXACT pooled adj_edge_jaccard after the deletion.

deletion_precision = FP_deleted / (FP_deleted + TP_deleted)   -- free deletions are
metric-neutral and are excluded from the ratio, which is exactly the quantity the
59.0 % (6bba) / 53.1 % (44b6) break-even bar refers to.

The pooled recompute is the authority; the precision bar is reported alongside it so
the two can be checked against each other.

Usage
-----
  .\.venv\Scripts\python.exe scripts\win_bet\ea_abstain.py --tag f1 --dir c:\temp\error_atlas
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# feature -> +1 if LOW values are suspicious, -1 if HIGH values are suspicious
DIRECTION = {
    "edge_prob": +1,
    "src_margin": -1,
    "rank_in_src": -1,
    "nn_margin_um": -1,
    "rev_margin_um": -1,
    "disp_um": -1,
    "disp_ratio": -1,
    "accel_um": -1,
    "cos_prev": +1,
    "dens15_src": -1,
    "src_outdeg": -1,
    "tgt_indeg": -1,
    "s_z": -1,
}
BUDGETS = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05,
           0.075, 0.10, 0.15, 0.20, 0.30]


def pooled_adj(tp, fp, fn, r):
    w = tp + fp + fn
    J = np.where(w > 0, tp / np.maximum(w, 1), 0.0)
    adj = np.maximum(0.0, J * (1 - 0.1 * r))
    return float((w * adj).sum() / w.sum())


def sweep(F: pd.DataFrame, crops: pd.DataFrame, score: np.ndarray, name: str) -> list[dict]:
    """score: lower == more suspicious. NaN -> never deleted (pushed to +inf)."""
    s = np.where(np.isnan(score), np.inf, score)
    order = np.argsort(s, kind="stable")
    lab = F.label.values[order]
    ds = F.dataset.values[order]
    n_valid = int(np.isfinite(s).sum())

    base = crops.set_index("dataset")
    tp0 = base.edge_tp.astype(float)
    fp0 = base.edge_fp.astype(float)
    fn0 = base.edge_fn.astype(float)
    r0 = base.total_node_ratio.astype(float)
    base_score = pooled_adj(tp0.values, fp0.values, fn0.values, r0.values)

    rows = []
    N = len(F)
    for b in BUDGETS:
        k = int(round(b * N))
        if k == 0 or k > n_valid:
            continue
        sl = slice(0, k)
        d = pd.DataFrame({"dataset": ds[sl], "label": lab[sl]})
        ct = d.pivot_table(index="dataset", columns="label", aggfunc=len, fill_value=0)
        ct = ct.reindex(base.index, fill_value=0)
        dtp = ct.get("TP", pd.Series(0, index=base.index)).astype(float)
        dfp = ct.get("FP", pd.Series(0, index=base.index)).astype(float)
        dfree = ct.get("free", pd.Series(0, index=base.index)).astype(float)
        new = pooled_adj((tp0 - dtp).values, (fp0 - dfp).values,
                         (fn0 + dtp).values, r0.values)
        nt, nf = float(dtp.sum()), float(dfp.sum())
        prec = nf / (nf + nt) if (nf + nt) else float("nan")
        rows.append({
            "feature": name, "budget_frac_all_edges": b, "n_deleted": k,
            "deleted_TP": nt, "deleted_FP": nf, "deleted_free": float(dfree.sum()),
            "deletion_precision": prec,
            "frac_scored_edges_touched": (nt + nf) / float(tp0.sum() + fp0.sum()),
            "pooled_adjJ_after": new, "delta": new - base_score,
        })
    return rows


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """P(score of an FP < score of a TP); 1.0 == perfect FP-first ranking.

    *s* is oriented so that LOW == suspicious (same orientation the sweep deletes in),
    and *y* is True for TP. Mann-Whitney U for "FP ranks below TP".
    """
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    r = pd.Series(s).rank().values
    n_fp, n_tp = int((~y).sum()), int(y.sum())
    u_fp_above = r[~y].sum() - n_fp * (n_fp + 1) / 2      # #(FP ranked above TP)
    return float(1.0 - u_fp_above / (n_fp * n_tp))


def _fit_logreg(X: np.ndarray, y: np.ndarray, iters: int = 60, lam: float = 1.0):
    """Plain L2-penalised logistic regression via Newton-IRLS (numpy only)."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    I = np.eye(d + 1)
    I[-1, -1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) + lam * (I @ w)
        s = np.clip(p * (1 - p), 1e-6, None)
        H = (Xb * s[:, None]).T @ Xb + lam * I
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return w


def _predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.hstack([X, np.ones((len(X), 1))])
    return 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--json-out")
    args = ap.parse_args()
    D = Path(args.dir)
    F = pd.read_parquet(D / f"feats_{args.tag}.parquet")
    crops = pd.read_parquet(D / f"crops_{args.tag}.parquet")

    res: dict = {"tag": args.tag, "n_emitted_edges": int(len(F)),
                 "label_counts": F.label.value_counts().to_dict()}
    base = pooled_adj(crops.edge_tp.values.astype(float), crops.edge_fp.values.astype(float),
                      crops.edge_fn.values.astype(float), crops.total_node_ratio.values)
    res["baseline_pooled_adjJ"] = base
    # analytic break-even (pooled, ignoring the per-crop weighting subtlety)
    W = float((crops.edge_tp + crops.edge_fp + crops.edge_fn).sum())
    TP = float(crops.edge_tp.sum())
    res["analytic_breakeven_precision"] = W / (W + TP)

    sc = F[F.scored]
    y = (sc.label == "TP").values
    feats = [c for c in DIRECTION if c in F.columns]
    res["auc_fp_vs_tp_on_scored"] = {}
    allrows = []
    for c in feats:
        v = pd.to_numeric(sc[c], errors="coerce").values.astype(float)
        res["auc_fp_vs_tp_on_scored"][c] = auc(y, DIRECTION[c] * v)
        col = pd.to_numeric(F[c], errors="coerce").values.astype(float)
        allrows += sweep(F, crops, DIRECTION[c] * col, c)

    # ---- learned combination, CROP-HELD-OUT (5 folds over crops) -------------
    # Trained ONLY on scorer-evaluable edges (the only ones carrying a label), then
    # applied to every emitted edge of the held-out crops. Squared terms give the
    # linear model some curvature without a tree library.
    use = [c for c in feats if F[c].notna().mean() > 0.5]
    Xr = F[use].apply(pd.to_numeric, errors="coerce")
    Xr = Xr.replace([np.inf, -np.inf], np.nan)
    Xr = Xr.fillna(Xr.median())
    Xr = np.hstack([Xr.values, Xr.values ** 2])
    mu, sd = Xr.mean(0), Xr.std(0)
    Xs = (Xr - mu) / np.where(sd > 0, sd, 1.0)
    ysc = F.scored.values
    ylab = (F.label == "TP").values.astype(float)
    ds_all = F.dataset.values
    uniq = np.array(sorted(pd.unique(ds_all)))
    rng = np.random.default_rng(0)
    fold_of = dict(zip(uniq, rng.permutation(len(uniq)) % 5))
    fid = np.array([fold_of[d] for d in ds_all])
    oof = np.full(len(F), np.nan)
    for k in range(5):
        tr = (fid != k) & ysc
        te = fid == k
        if tr.sum() < 50:
            continue
        w = _fit_logreg(Xs[tr], ylab[tr])
        oof[te] = _predict(w, Xs[te])
    res["auc_fp_vs_tp_on_scored"]["LOGREG_oof"] = auc(y, oof[ysc])
    allrows += sweep(F, crops, oof, "LOGREG_oof")
    res["logreg_features"] = use

    sw = pd.DataFrame(allrows)
    sw.to_parquet(D / f"sweep_{args.tag}.parquet")
    res["best_delta_per_feature"] = (
        sw.sort_values("delta", ascending=False).groupby("feature").head(1)
        .sort_values("delta", ascending=False).to_dict("records"))
    # precision at the smallest budget that touches >=1% of scored edges
    tgt = sw[sw.frac_scored_edges_touched >= 0.01]
    res["precision_at_1pct_scored"] = (
        tgt.sort_values("frac_scored_edges_touched").groupby("feature").head(1)
        .sort_values("deletion_precision", ascending=False).to_dict("records"))
    out = json.dumps(res, indent=2, default=float)
    print(out)
    if args.json_out:
        Path(args.json_out).write_text(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
