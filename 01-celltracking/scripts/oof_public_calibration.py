"""LANE O — OOF->public calibration, on CORRECT-SUBSTRATE deltas.

Correction this supersedes: the previous P(public tie)=0.4148 simulation consumed the invalid
post-wrapper +0.000435 per-crop deltas, so it could not speak to whether a true +0.008488
effect should have shown publicly. This rebuilds it from
reports/inventory/edge_fn_census_prewrapper.json.

Two parts:
  1. 4-movie panel simulation from the real per-crop (armA, armB) pairs, preserving pairing
     (both arms are scored on the same crops, so movie difficulty cancels in the delta).
  2. A calibration table of every historically scored system whose OOF is on a comparable
     substrate, against its public score, to classify the residual.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
ALPHA, DIVW = 0.1, 0.1


def piece(d):
    tp, fp, fn = d["edge_tp"], d["edge_fp"], d["edge_fn"]
    w = tp + fp + fn
    if w <= 0:
        return None
    tnr = d["total_node_ratio"]
    if tnr != tnr:
        tnr = 0.0
    return (max(0.0, (tp / w) * (1 - ALPHA * tnr)), w,
            d["division_tp"], d["division_fp"], d["division_fn"])


def panel(rows, arm, weights=None):
    num = den = 0.0
    dtp = dfp = dfn = 0
    for i, r in enumerate(rows):
        p = piece(r[arm])
        if p is None:
            continue
        adj, w, a, b, c = p
        if weights is not None:
            w = weights[i]
        num += w * adj
        den += w
        dtp += a
        dfp += b
        dfn += c
    if den <= 0:
        return float("nan")
    dd = dtp + dfp + dfn
    return num / den + DIVW * (dtp / dd if dd else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census",
                    default=str(REPO / "reports/inventory/edge_fn_census_prewrapper.json"))
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--out", default=str(REPO / "reports/inventory/oof_public_calibration.json"))
    a = ap.parse_args()

    rows = [r for r in json.load(open(a.census)) if "armA" in r and "armB" in r]
    fam = np.array([r["family"] for r in rows])
    rng = np.random.default_rng(20260803)

    full_a, full_b = panel(rows, "armA"), panel(rows, "armB")
    print("=" * 84)
    print(f"LANE O — CORRECT-SUBSTRATE CALIBRATION   ({len(rows)} crops: "
          f"44b6 {int((fam=='44b6').sum())}, 6bba {int((fam=='6bba').sum())})")
    print("=" * 84)
    print(f"  full-corpus armA {full_a:.7f}  armB {full_b:.7f}  delta {full_b-full_a:+.7f}")

    # ---- contribution concentration -------------------------------------------------
    contrib = []
    for r in rows:
        pa, pb = piece(r["armA"]), piece(r["armB"])
        if pa and pb:
            contrib.append((pb[1] * pb[0] - pa[1] * pa[0], r["crop"]))
    contrib.sort(reverse=True)
    tot = sum(c for c, _ in contrib)
    print("\n  CONTRIBUTION CONCENTRATION (share of total weighted gain)")
    for k in (1, 5, 10, 20):
        print(f"    top {k:2d} crops: {100*sum(c for c, _ in contrib[:k])/tot:6.2f}%"
              f"   e.g. {contrib[0][1] if k==1 else ''}")
    w_all = np.array([piece(r["armB"])[1] for r in rows], dtype=float)
    ess = w_all.sum() ** 2 / (w_all ** 2).sum()
    print(f"    effective sample size under edge-mass weighting: {ess:.1f} of {len(rows)} crops")

    # ---- 4-movie panel simulation ---------------------------------------------------
    def sim(n, trials, mix=None, mass_weighted=True):
        pos = neg = flat = 0
        deltas = np.empty(trials)
        for t in range(trials):
            if mix is None:
                idx = rng.integers(0, len(rows), n)
            else:
                n44 = int(round(n * mix))
                i44 = np.flatnonzero(fam == "44b6")
                i6 = np.flatnonzero(fam == "6bba")
                idx = np.concatenate([rng.choice(i44, n44, replace=True),
                                      rng.choice(i6, n - n44, replace=True)])
            p = [rows[i] for i in idx]
            w = None if mass_weighted else np.ones(len(p))
            sa, sb = panel(p, "armA", w), panel(p, "armB", w)
            deltas[t] = sb - sa
            ra, rb = round(sa, 3), round(sb, 3)
            if rb > ra:
                pos += 1
            elif rb < ra:
                neg += 1
            else:
                flat += 1
        return deltas, pos / trials, neg / trials, flat / trials

    print("\n  4-MOVIE PANEL, rounded to 3 dp (what the leaderboard shows)")
    d4, p, n_, f = sim(4, a.trials)
    print(f"    P(public UP) {p:.4f}   P(public DOWN) {n_:.4f}   P(FLAT) {f:.4f}")
    print(f"    E[delta] {d4.mean():+.6f}  sd {d4.std():.6f}  "
          f"5th {np.percentile(d4,5):+.6f}  95th {np.percentile(d4,95):+.6f}")
    print(f"    P(|delta| < 0.0005, i.e. invisible at 3dp) : "
          f"{float((np.abs(d4) < 0.0005).mean()):.4f}")

    print("\n  BY FAMILY COMPOSITION (4 movies, fraction 44b6)")
    print(f"    {'mix':>6s} {'E[delta]':>11s} {'P(up)':>8s} {'P(down)':>9s} {'P(flat)':>9s}")
    for mix in (0.0, 0.25, 0.5, 0.75, 1.0):
        dd, pu, pd, pf = sim(4, 4000, mix=mix)
        print(f"    {mix:6.2f} {dd.mean():+11.6f} {pu:8.4f} {pd:9.4f} {pf:9.4f}")

    print("\n  WEIGHTING VARIANT (4 movies)")
    for label, mw in (("edge-mass weighted", True), ("crop-equal", False)):
        dd, pu, pd, pf = sim(4, 4000, mass_weighted=mw)
        print(f"    {label:20s} E[delta] {dd.mean():+.6f}  P(up) {pu:.4f}  P(flat) {pf:.4f}")

    print("\n  LARGER PANELS")
    print(f"    {'n':>5s} {'E[delta]':>11s} {'P(up)':>8s} {'P(flat)':>9s}")
    res = {}
    for n in (4, 8, 16, 32, 64, 128):
        dd, pu, pd, pf = sim(n, 4000)
        res[n] = {"E": float(dd.mean()), "P_up": pu, "P_down": pd, "P_flat": pf,
                  "p05": float(np.percentile(dd, 5))}
        print(f"    {n:5d} {dd.mean():+11.6f} {pu:8.4f} {pf:9.4f}")

    print("\n  KEY READING")
    print(f"    A true +{full_b-full_a:.6f} pooled effect shows as FLAT at 3 dp on a 4-movie")
    print(f"    panel with probability {f:.4f}, and moves DOWN with probability {n_:.4f}.")

    json.dump({"full": {"armA": full_a, "armB": full_b, "delta": full_b - full_a},
               "panel4": {"P_up": p, "P_down": n_, "P_flat": f,
                          "E": float(d4.mean()), "sd": float(d4.std())},
               "panels": {str(k): v for k, v in res.items()},
               "ess": float(ess), "n_crops": len(rows)},
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
