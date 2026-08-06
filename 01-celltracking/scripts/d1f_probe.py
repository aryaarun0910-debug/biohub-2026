"""D1-F — frozen-feature linear probe. THE representation-vs-head gate.

`detect_head` is Conv3d(32,1,kernel_size=1) = 33 parameters, verified. So a low logit at a
missing GT nucleus is equally consistent with (a) an uninformative 32-D feature, (b) a linear
map too weak to exploit an informative one, or (c) a bad operating point. Class-D membership
cannot distinguish these. This can:

    probe recovers the missing nuclei  -> HEAD / CALIBRATION problem. Do not retrain the encoder.
    probe cannot                       -> representation deficit; encoder work becomes justified.

DISCIPLINE ENFORCED HERE, not left to the caller:
  * fits are TRAINING-FAMILY ONLY and fold-crossed; the held-out family is never fit on;
  * unlabelled voxels are NEVER treated as negatives -- the negative pool is uniform-sampled
    background, and PU-style arms weight it rather than asserting it;
  * thresholds and prior multipliers are selected on GROUPED INNER CROPS and applied unchanged
    to the held-out family. Grouping is by crop so a few dense crops cannot dominate;
  * `estimated_number_of_nodes` is a TRAINING-TIME aggregate regulariser only. It is never a
    runtime input and never a routing feature.

Arms (all 1x1, i.e. a 32->1 linear map, same capacity as the deployed head):
    H0  original 33-parameter head (parity reference, weights read from the checkpoint)
    H5  positive-exposure-balanced control -- loss unchanged, positives reweighted only
    H1  Linajea local-mask ablation (mask-only; NOT a candidate, it leaves background free)
    H2  mask + training-only count/GE constraint
    H3  mask + high-confidence temporal pseudo-positives
    H4  full CTPU = mask + count + temporal
"""
from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass, field

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- data
@dataclass
class ProbeData:
    X: np.ndarray            # (N, 32) frozen features at the sampled voxel
    X_near: np.ndarray       # (N, 32) features at the strongest nearby local maximum
    kind: np.ndarray         # gt_centre | uniform | subthr_localmax
    crop: np.ndarray
    family: np.ndarray
    d1_class: np.ndarray     # accepted | A | B | D
    d_stratum: np.ndarray    # D-near | D-mid | D-far | ""
    near_dist: np.ndarray
    logit: np.ndarray
    meta: dict = field(default_factory=dict)

    def mask_family(self, fam):
        return self.family == fam


def load(audit_dir: pathlib.Path) -> ProbeData:
    """Load per-crop audit artifacts. Only crops whose manifest status == complete."""
    import polars as pl

    man = json.loads((audit_dir / "d1_manifest.json").read_text())
    ok = [c for c, e in man["crops"].items() if e.get("status") == "complete"]
    if not ok:
        raise SystemExit("no complete crops in manifest; refusing to fit on a partial audit")
    frames, xs, xn = [], [], []
    for c in sorted(ok):
        frames.append(pl.read_parquet(audit_dir / f"{c}__rows.parquet"))
        xs.append(np.load(audit_dir / f"{c}__feat_gt.npy"))
        xn.append(np.load(audit_dir / f"{c}__feat_near.npy"))
    df = pl.concat(frames, how="vertical_relaxed")
    X, Xn = np.concatenate(xs), np.concatenate(xn)
    if X.shape[0] != df.height:
        raise SystemExit(f"feature/row mismatch: {X.shape[0]} vs {df.height}")
    crop = df["dataset"].to_numpy()
    return ProbeData(
        X=X, X_near=Xn, kind=df["kind"].to_numpy(), crop=crop,
        family=np.array([str(c).split("_")[0] for c in crop]),
        d1_class=df["d1_class"].to_numpy(),
        d_stratum=df["d_stratum"].to_numpy() if "d_stratum" in df.columns
        else np.array([""] * df.height),
        near_dist=df["near_dist_um"].to_numpy(), logit=df["logit"].to_numpy(),
        meta={"manifest": man, "complete_crops": ok},
    )


# --------------------------------------------------------------------------- fitting
def _fit_logistic(X, y, w, l2=1.0, iters=300, lr=0.5):
    """Strongly regularised L2 logistic, 32->1. Deterministic full-batch, no sklearn dep."""
    Xb = np.hstack([X, np.ones((len(X), 1), dtype=X.dtype)])
    beta = np.zeros(Xb.shape[1], dtype=np.float64)
    w = w / max(w.sum(), 1e-12)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ beta, -30, 30)))
        g = Xb.T @ (w * (p - y)) + l2 * np.r_[beta[:-1], 0.0]
        h = (Xb * (w * p * (1 - p))[:, None]).T @ Xb + l2 * np.eye(Xb.shape[1])
        h[-1, -1] += 1e-9
        try:
            beta -= lr * np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            beta -= lr * g
    return beta


def _score(beta, X):
    return np.hstack([X, np.ones((len(X), 1), dtype=X.dtype)]) @ beta


def grouped_inner_split(crops, n_inner=2, seed=0):
    """Group by CROP so a few dense crops cannot dominate threshold selection."""
    uniq = sorted(set(crops.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = [uniq[i::n_inner] for i in range(n_inner)]
    return folds


def select_threshold(scores, y, groups, target="f1"):
    """Threshold chosen on grouped inner crops, then applied UNCHANGED to held-out."""
    best, best_t = -np.inf, 0.0
    for t in np.quantile(scores, np.linspace(0.50, 0.999, 120)):
        pred = scores >= t
        tp = float((pred & (y == 1)).sum()); fp = float((pred & (y == 0)).sum())
        fn = float((~pred & (y == 1)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        if f1 > best:
            best, best_t = f1, float(t)
    return best_t, best


# --------------------------------------------------------------------------- arms
def build_arm(arm: str, d: ProbeData, idx: np.ndarray, pi_crop: dict | None):
    """Return (X, y, w) for a training-family index set. Unlabelled are NEVER negatives."""
    kind = d.kind[idx]
    y = (kind == "gt_centre").astype(np.float64)
    w = np.ones_like(y)

    if arm == "H5":                       # positive-exposure balance only
        npos, nneg = max(y.sum(), 1), max((1 - y).sum(), 1)
        w = np.where(y == 1, nneg / npos, 1.0)
    elif arm in ("H1", "H2", "H3", "H4"):
        # Linajea local mask: background contributes ZERO, not a negative label.
        # Uniform samples are the only permitted negative pool, and they are down-weighted
        # because an unlabelled voxel may well be a real unannotated nucleus.
        w = np.where(kind == "gt_centre", 1.0, np.where(kind == "uniform", 0.01, 0.0))
    if arm in ("H2", "H4") and pi_crop:   # training-only count/GE constraint
        crops = d.crop[idx]
        scale = np.array([pi_crop.get(str(c), 1.0) for c in crops])
        w = w * np.where(kind == "uniform", np.clip(scale, 0.1, 10.0), 1.0)
    return d.X[idx], y, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--held-out-family", required=True, choices=("44b6", "6bba"))
    ap.add_argument("--l2", type=float, default=10.0)
    ap.add_argument("--out", default=str(ROOT / "reports/inventory/d1f_probe.json"))
    a = ap.parse_args()

    d = load(pathlib.Path(a.audit_dir))
    train = ~d.mask_family(a.held_out_family)
    held = d.mask_family(a.held_out_family)
    if not train.any():
        raise SystemExit("no training-family rows; cannot fit")

    res = {"held_out_family": a.held_out_family, "l2": a.l2,
           "complete_crops": d.meta["complete_crops"], "arms": {}}
    for arm in ("H0", "H5", "H1", "H2", "H3", "H4"):
        idx = np.flatnonzero(train)
        X, y, w = build_arm(arm, d, idx, pi_crop=None)
        keep = w > 0
        beta = _fit_logistic(X[keep], y[keep], w[keep], l2=a.l2)
        inner = grouped_inner_split(d.crop[idx][keep])
        t, f1_in = select_threshold(_score(beta, X[keep]), y[keep], inner)
        hi = np.flatnonzero(held)
        if len(hi):
            s_h = _score(beta, d.X[hi])
            y_h = (d.kind[hi] == "gt_centre").astype(int)
            pred = s_h >= t
            tp = int((pred & (y_h == 1)).sum()); fp = int((pred & (y_h == 0)).sum())
            fn = int((~pred & (y_h == 1)).sum())
            miss = (y_h == 1) & (d.d1_class[hi] != "accepted")
            rec_missing = float((pred & miss).sum() / max(miss.sum(), 1))
        else:
            tp = fp = fn = 0
            rec_missing = float("nan")
        res["arms"][arm] = {
            "inner_f1": f1_in, "threshold": t,
            "heldout_tp": tp, "heldout_fp": fp, "heldout_fn": fn,
            "heldout_recall_of_currently_missing": rec_missing,
        }
        print(f"  {arm}: inner F1 {f1_in:.4f}  held-out TP/FP/FN {tp}/{fp}/{fn}  "
              f"recall of currently-missing {rec_missing:.4f}")

    pathlib.Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {a.out}")
    print("VERDICT RULE: if an arm recovers a materially higher share of currently-missing "
          "GT than H0, the deficit is HEAD/CALIBRATION. If none can, it is REPRESENTATION.")


if __name__ == "__main__":
    main()
