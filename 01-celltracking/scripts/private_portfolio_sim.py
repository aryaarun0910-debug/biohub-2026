"""LANE A — private-portfolio decision under a sparse, low-resolution public panel.

Question this answers: arm B measured +0.008 pooled OOF with P(delta>0)=1.000 over 144 crops,
and then tied P0-B to three decimals on the 4-movie public panel. Is that tie evidence the
effect is absent, or evidence the panel cannot see it? And which two artifacts should be the
final private submissions?

Method: resample real per-crop (arm A, arm B) score pairs from the OOF census into synthetic
panels. Because both arms are evaluated on the SAME crops, movie difficulty cancels in the
paired delta -- the simulation preserves that pairing, which is the whole point.

Inputs: reports/inventory/edge_fn_census.json (per-crop armA/armB rows).
No GPU, no submission.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
ALPHA, DIVW = 0.1, 0.1


def crop_score(row):
    """Exact per-crop contribution pieces: (adj_edge_J, weight, div tp/fp/fn)."""
    tp, fp, fn = row["edge_tp"], row["edge_fp"], row["edge_fn"]
    den = tp + fp + fn
    if den <= 0:
        return None
    j = tp / den
    tnr = row["total_node_ratio"]
    if tnr != tnr:
        tnr = 0.0
    return (max(0.0, j * (1 - ALPHA * tnr)), den,
            row["division_tp"], row["division_fp"], row["division_fn"])


def panel_score(rows, arm):
    """Exact composite over a panel: weighted adj_edge_J + 0.1 * micro division_J."""
    num = den = 0.0
    dtp = dfp = dfn = 0
    for r in rows:
        c = crop_score(r[arm])
        if c is None:
            continue
        adj, w, a, b, d = c
        num += w * adj
        den += w
        dtp += a
        dfp += b
        dfn += d
    if den <= 0:
        return float("nan")
    adj = num / den
    dd = dtp + dfp + dfn
    return adj + DIVW * (dtp / dd if dd else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default=str(REPO / "reports/inventory/edge_fn_census.json"))
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--out", default=str(REPO / "reports/inventory/private_portfolio.json"))
    a = ap.parse_args()

    rows = json.load(open(a.census))
    rows = [r for r in rows if "armA" in r and "armB" in r]
    if not rows:
        sys.exit("census has no armA/armB rows")
    rng = np.random.default_rng(a.seed)

    fam = np.array([r["family"] for r in rows])
    print("=" * 78)
    print(f"LANE A — PRIVATE-PORTFOLIO SIMULATION   ({len(rows)} real OOF crops)")
    print("=" * 78)
    for f in sorted(set(fam.tolist())):
        print(f"  {f}: {int((fam == f).sum())} crops")

    full_A, full_B = panel_score(rows, "armA"), panel_score(rows, "armB")
    print(f"\n  FULL-CORPUS exact composite   armA {full_A:.6f}   armB {full_B:.6f}"
          f"   delta {full_B-full_A:+.6f}")

    # ---- per-crop paired deltas (the quantity a panel actually averages) --------------
    deltas = []
    for r in rows:
        ca, cb = crop_score(r["armA"]), crop_score(r["armB"])
        if ca and cb:
            deltas.append(cb[0] - ca[0])
    deltas = np.array(deltas)
    print(f"  per-crop paired adj_edge_J delta: mean {deltas.mean():+.6f}  "
          f"sd {deltas.std():.6f}  P(>0) {(deltas > 0).mean():.4f}  "
          f"median {np.median(deltas):+.6f}")

    def resample(n, family_mix=None):
        if family_mix is None:
            idx = rng.integers(0, len(rows), n)
        else:
            n44 = int(round(n * family_mix))
            i44 = np.flatnonzero(fam == "44b6")
            i6 = np.flatnonzero(fam == "6bba")
            idx = np.concatenate([rng.choice(i44, n44, replace=True),
                                  rng.choice(i6, n - n44, replace=True)])
        return [rows[i] for i in idx]

    def panel_stats(n, trials, family_mix=None, round_dp=None):
        dA = np.empty(trials)
        dB = np.empty(trials)
        for t in range(trials):
            p = resample(n, family_mix)
            sa, sb = panel_score(p, "armA"), panel_score(p, "armB")
            if round_dp is not None:
                sa, sb = round(sa, round_dp), round(sb, round_dp)
            dA[t], dB[t] = sa, sb
        return dA, dB

    trials = a.trials
    print("\n  PANEL SIMULATION (paired: both arms scored on the same resampled crops)")
    print(f"  {'panel':>7s} {'E[delta]':>11s} {'sd':>9s} {'P(B>A)':>8s} "
          f"{'P(tie@3dp)':>11s} {'5th pct':>10s} {'95th pct':>10s}")
    results = {}
    for n in (4, 8, 16, 32, 64, 128):
        dA, dB = panel_stats(n, min(trials, 4000 if n > 32 else trials))
        d = dB - dA
        r3 = np.round(dB, 3) - np.round(dA, 3)
        results[n] = {
            "E_delta": float(d.mean()), "sd": float(d.std()),
            "P_B_gt_A": float((d > 0).mean()),
            "P_tie_3dp": float((r3 == 0).mean()),
            "p05": float(np.percentile(d, 5)), "p95": float(np.percentile(d, 95)),
        }
        r = results[n]
        print(f"  {n:7d} {r['E_delta']:+11.6f} {r['sd']:9.6f} {r['P_B_gt_A']:8.4f} "
              f"{r['P_tie_3dp']:11.4f} {r['p05']:+10.6f} {r['p95']:+10.6f}")

    print("\n  FAMILY-MIX SENSITIVITY (4-movie panel, fraction 44b6)")
    print(f"  {'mix':>8s} {'E[delta]':>11s} {'P(B>A)':>8s} {'P(tie@3dp)':>11s}")
    for mix in (0.0, 0.25, 0.5, 0.75, 1.0):
        dA, dB = panel_stats(4, 4000, family_mix=mix)
        d = dB - dA
        r3 = np.round(dB, 3) - np.round(dA, 3)
        print(f"  {mix:8.2f} {d.mean():+11.6f} {(d > 0).mean():8.4f} {(r3 == 0).mean():11.4f}")

    print("\n  ANNOTATION-MASS UNCERTAINTY (weight crops by annotated-edge mass ^ gamma)")
    print(f"  {'gamma':>8s} {'E[delta] 4-movie':>18s} {'P(tie@3dp)':>11s}")
    for gamma in (0.0, 0.5, 1.0):
        acc_d, acc_tie = [], []
        for _ in range(3000):
            p = resample(4)
            w = np.array([max(1, r["armB"]["edge_tp"] + r["armB"]["edge_fp"]
                              + r["armB"]["edge_fn"]) for r in p], dtype=float) ** gamma
            sa = np.average([crop_score(r["armA"])[0] for r in p], weights=w)
            sb = np.average([crop_score(r["armB"])[0] for r in p], weights=w)
            acc_d.append(sb - sa)
            acc_tie.append(round(sb, 3) == round(sa, 3))
        print(f"  {gamma:8.2f} {np.mean(acc_d):+18.6f} {np.mean(acc_tie):11.4f}")

    print("\n  KEY READINGS")
    r4 = results[4]
    print(f"    P(public tie at 3dp | the OOF effect is real) on a 4-movie panel : "
          f"{r4['P_tie_3dp']:.4f}")
    print(f"    P(arm B > P0-B) on a 4-movie panel                               : "
          f"{r4['P_B_gt_A']:.4f}")
    for n in (32, 64, 128):
        print(f"    P(arm B > P0-B) on a {n:3d}-movie private panel                    : "
              f"{results[n]['P_B_gt_A']:.4f}   E[delta] {results[n]['E_delta']:+.6f}"
              f"   5th pct {results[n]['p05']:+.6f}")

    json.dump({"full_corpus": {"armA": full_A, "armB": full_B, "delta": full_B - full_A},
               "per_crop_delta": {"mean": float(deltas.mean()), "sd": float(deltas.std()),
                                  "p_gt0": float((deltas > 0).mean())},
               "panels": {str(k): v for k, v in results.items()},
               "n_crops": len(rows)},
              open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
