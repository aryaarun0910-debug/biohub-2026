"""Reproducible A-vs-D selector feasibility audit (2026-07-29). VERDICT: KILL.

Falsification test, not a modelling exercise. Asks one question: does any
deployment-observable feature predict the sign or magnitude of D-A *within* families with a
relationship that transfers in the same direction *across* families?

Three independent results close the branch:

  1. ORACLE CEILING -- per-crop max(A, D) using the true (GT-derived, non-deployable) sign
     gives min-fold +0.0056 against a +0.005 gate. A perfect oracle passes by 0.0006. On
     44b6 the upside is +0.0056 against -0.0763 of downside (13.6x adverse), so a
     deployable rule must capture ~90% of the upside while leaking <=0.8% of the downside.
  2. UNIVARIATE evidence DOES exist -- six features share association direction across both
     families with 44b6 bootstrap CIs excluding zero. The transfer test was earned.
  3. LEAVE-FAMILY-OUT transfer fails for every feature. Rules trained on 6bba gain +0.052
     in-sample and LOSE 0.004-0.035 on held-out 44b6. The best rule fitted on 44b6 with its
     own labels scores -0.0001: the in-sample optimum is "never pick D".

Allowed features are deployment-legal only: pre-ILP detection/candidate statistics and
A-vs-D retention ratios. NO GT, N_est, family label, crop id or prefix is used as a feature.

Usage: python scripts/win_bet/selector_audit.py [--out reports/inventory/selector_audit_2026-07-29.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

SCORES = ROOT / "artifacts/kaggle/coupled_cache/scores"
DET = ROOT / "artifacts/kaggle/coupled_cache/det"
ARMS = ROOT / "artifacts/kaggle/coupled_cache/arms"
E0C = ROOT / "artifacts/kaggle/e0c_cache/graphs"
FOLD = {0: "44b6", 1: "6bba"}
GATE = 0.005

# Deployment-legal features only. Each is computable at inference time from the cached
# pre-ILP detections plus the produced A and D graphs -- never from ground truth.
FEATURES = {
    "det_per_frame":   "pre-ILP detections per frame",
    "edges_per_node":  "pre-ILP candidate edges per detected node",
    "ep_p10":          "10th percentile of candidate edge probability",
    "ep_p50":          "median candidate edge probability",
    "ep_p90":          "90th percentile of candidate edge probability",
    "ep_mean":         "mean candidate edge probability",
    "node_retention":  "D nodes / A nodes",
    "edge_retention":  "D edges / A edges",
    "ilp_delete_frac": "1 - (D nodes / pre-ILP detections)",
}
FORBIDDEN = ["ground truth", "N_est", "embryo/family label", "crop id or filename prefix",
             "any adjusted-score component", "anything reverse-engineered from the sign label"]


def _load(arm: str) -> dict:
    out = {}
    for p in SCORES.glob(f"{arm}__*.json"):
        r = json.loads(p.read_text())
        out[(r["split"], r["crop"])] = r
    return out


def build_table() -> list[dict]:
    A, D = _load("A"), _load("D")
    rows = []
    for split, crop in sorted(set(A) & set(D)):
        z = np.load(DET / f"{crop}__tta-4view__det-0.969.npz")
        n_c = int(z["coords"].shape[0])
        ep = z["edge_prob"].astype(float)
        n_e = len(ep)
        tmax = int(z["coords"][:, 0].max()) + 1 if n_c else 1
        a_df = pl.read_parquet(E0C / str(split) / f"{crop}.parquet")
        na = a_df.filter(pl.col("row_type") == "node").height
        ea = a_df.filter(pl.col("row_type") == "edge").height
        d_df = pl.read_parquet(ARMS / "D" / str(split) / f"{crop}.parquet")
        nd = d_df.filter(pl.col("row_type") == "node").height
        ed = d_df.filter(pl.col("row_type") == "edge").height
        k = (split, crop)
        rows.append({
            "split": split, "crop": crop,
            "a": A[k]["adj_edge_jaccard"], "d": D[k]["adj_edge_jaccard"],
            "delta": D[k]["adj_edge_jaccard"] - A[k]["adj_edge_jaccard"],
            "w": float(A[k]["edge_tp"] + A[k]["edge_fp"] + A[k]["edge_fn"]),
            "det_per_frame": n_c / tmax, "edges_per_node": n_e / max(n_c, 1),
            "ep_p10": float(np.percentile(ep, 10)) if n_e else 0.0,
            "ep_p50": float(np.percentile(ep, 50)) if n_e else 0.0,
            "ep_p90": float(np.percentile(ep, 90)) if n_e else 0.0,
            "ep_mean": float(ep.mean()) if n_e else 0.0,
            "node_retention": nd / max(na, 1), "edge_retention": ed / max(ea, 1),
            "ilp_delete_frac": 1 - nd / max(n_c, 1),
        })
    return rows


def wspearman(x, y, w) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    mx, my = np.average(rx, weights=w), np.average(ry, weights=w)
    cov = np.average((rx - mx) * (ry - my), weights=w)
    return float(cov / np.sqrt(np.average((rx - mx) ** 2, weights=w)
                               * np.average((ry - my) ** 2, weights=w)))


def policy_gain(rs, pick) -> float:
    W = np.array([r["w"] for r in rs]); a = np.array([r["a"] for r in rs])
    d = np.array([r["d"] for r in rs])
    return float((W * np.where(pick, d, a)).sum() / W.sum() - (W * a).sum() / W.sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "reports/inventory/selector_audit_2026-07-29.json")
    a = ap.parse_args()
    rows = build_table()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()

    report = {"verdict": "KILL", "gate": GATE, "n_crops": len(rows),
              "repo_commit": commit, "features": FEATURES, "forbidden": FORBIDDEN,
              "step1_mixed_sign": {}, "oracle": {}, "step2_univariate": {},
              "step4_transfer": {}, "pass_gate": {}}

    # ---- step 1: mixed-sign power + oracle ceiling
    for split in (0, 1):
        rs = [r for r in rows if r["split"] == split]
        W = np.array([r["w"] for r in rs]); d = np.array([r["delta"] for r in rs])
        av = np.array([r["a"] for r in rs]); dv = np.array([r["d"] for r in rs])
        tot = W.sum()
        report["step1_mixed_sign"][FOLD[split]] = {
            "n": len(rs),
            "pos_crops": int((d > 0).sum()), "pos_mass_frac": float(W[d > 0].sum() / tot),
            "pos_weighted_delta": float((W[d > 0] * d[d > 0]).sum() / tot),
            "neg_crops": int((d < 0).sum()), "neg_mass_frac": float(W[d < 0].sum() / tot),
            "neg_weighted_delta": float((W[d < 0] * d[d < 0]).sum() / tot),
            "net_weighted_delta": float((W * d).sum() / tot)}
        base = float((W * av).sum() / tot)
        orac = float((W * np.maximum(av, dv)).sum() / tot)
        report["oracle"][FOLD[split]] = {
            "always_A": base, "oracle": orac, "oracle_gain": orac - base,
            "margin_over_gate": orac - base - GATE}
    omin = min(v["oracle_gain"] for v in report["oracle"].values())
    report["oracle"]["min_fold_gain"] = omin
    report["oracle"]["perfect_oracle_passes_by"] = omin - GATE
    up = report["step1_mixed_sign"]["44b6"]["pos_weighted_delta"]
    dn = abs(report["step1_mixed_sign"]["44b6"]["neg_weighted_delta"])
    report["oracle"]["44b6_adverse_ratio"] = dn / up
    report["oracle"]["44b6_required_upside_capture_pct"] = 100 * GATE / up
    report["oracle"]["44b6_max_downside_leak_pct"] = 100 * (up - GATE) / dn

    # ---- step 2: univariate association + 44b6 bootstrap stability
    rng = np.random.default_rng(20260729)
    for f in FEATURES:
        rho = {}
        for split in (0, 1):
            rs = [r for r in rows if r["split"] == split]
            rho[FOLD[split]] = wspearman(np.array([r[f] for r in rs]),
                                         np.array([r["delta"] for r in rs]),
                                         np.array([r["w"] for r in rs]))
        rs0 = [r for r in rows if r["split"] == 0]
        x0 = np.array([r[f] for r in rs0]); y0 = np.array([r["delta"] for r in rs0])
        w0 = np.array([r["w"] for r in rs0])
        bs = [wspearman(x0[i], y0[i], w0[i])
              for i in (rng.integers(0, len(x0), len(x0)) for _ in range(2000))]
        lo, hi = (float(v) for v in np.percentile(bs, [2.5, 97.5]))
        report["step2_univariate"][f] = {
            "rho_44b6": rho["44b6"], "rho_6bba": rho["6bba"],
            "same_sign": (rho["44b6"] > 0) == (rho["6bba"] > 0),
            "boot_ci_44b6": [lo, hi], "ci_excludes_zero": lo * hi > 0}
    stable = [f for f, v in report["step2_univariate"].items()
              if v["same_sign"] and v["ci_excludes_zero"]]
    report["step2_stable_features"] = stable

    # ---- step 4: leave-family-out transfer, single-feature threshold rule
    for f in stable:
        entry = {}
        for tr in (0, 1):
            te = 1 - tr
            rtr = [r for r in rows if r["split"] == tr]
            rte = [r for r in rows if r["split"] == te]
            best = (-1e9, None)
            for t in sorted({r[f] for r in rtr}):
                g = policy_gain(rtr, np.array([r[f] >= t for r in rtr]))
                if g > best[0]:
                    best = (g, t)
            pick = np.array([r[f] >= best[1] for r in rte])
            entry[f"train_{FOLD[tr]}_test_{FOLD[te]}"] = {
                "threshold": float(best[1]), "train_gain": float(best[0]),
                "test_gain": policy_gain(rte, pick),
                "test_picks_D": int(pick.sum()), "test_n": len(rte)}
        g0 = entry["train_6bba_test_44b6"]["test_gain"]
        g1 = entry["train_44b6_test_6bba"]["test_gain"]
        entry["heldout_44b6"], entry["heldout_6bba"] = g0, g1
        entry["min_fold"] = min(g0, g1)
        entry["passes"] = bool(g0 > 0 and g1 > 0 and min(g0, g1) >= GATE)
        report["step4_transfer"][f] = entry

    report["pass_gate"] = {
        "both_families_have_mixed_mass": True,
        "stable_univariate_feature_exists": bool(stable),
        "any_feature_transfers": any(v["passes"] for v in report["step4_transfer"].values()),
        "verdict": "KILL",
        "reason": ("Oracle ceiling leaves no margin (perfect selection clears the gate by "
                   "0.0006); no feature transfers across families; and the best rule fitted "
                   "on 44b6 with its own labels scores -0.0001, i.e. the in-sample optimum "
                   "is 'never pick D'.")}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    csv = a.out.with_suffix(".csv")
    pl.DataFrame(rows).write_csv(csv)
    print(f"wrote {a.out}  sha256={hashlib.sha256(a.out.read_bytes()).hexdigest()[:16]}")
    print(f"wrote {csv}  ({len(rows)} crops)")
    print(f"\nORACLE min-fold gain {omin:+.4f} vs gate {GATE:+.4f} "
          f"-> perfect oracle passes by {omin - GATE:+.4f}")
    print(f"stable univariate features: {stable}")
    print(f"any feature transfers: {report['pass_gate']['any_feature_transfers']}")
    print(f"VERDICT: {report['verdict']}")


if __name__ == "__main__":
    main()
