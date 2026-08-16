"""Crop-block bootstrap intervals for the POOLED OOF objective and its pairwise deltas.

The re-ranked arm table (v122 0.69909, C_survival 0.69692, Bp_ilp 0.69162, B_detpop
0.66677, E0c 0.66539) is five point estimates on ONE draw of 199 crops. The gaps between
the top three are 0.002-0.007, i.e. the same order as the sampling noise nobody had
measured. This supplies that measurement.

Resampling unit
---------------
The crop, not the edge. Edges inside a crop are massively dependent (one tracking failure
costs a whole track), so an edge-level bootstrap would understate the interval by a large
factor. Crops are the exchangeable unit the private split also draws on.

Two schemes are reported because they answer different questions:

  iid          crops resampled with replacement from all 199. Family composition is a
               random variable, so this is the interval on "the score on a fresh draw of
               crops from this population".
  stratified   resampled with replacement WITHIN each family, family crop counts held at
               the observed 63/136. Composition is fixed, so this isolates within-family
               noise from composition noise. Compare the two to see how much of the width
               is composition.

Deltas are PAIRED: both arms are scored on the same resampled crop multiset, so the
common crop-difficulty term cancels and the interval on the delta is far tighter than the
difference of the two marginal intervals. Reading a delta off the overlap of two marginal
CIs is the classic error and is explicitly avoided here.

Objective
---------
Exactly ``private_split_simulator.score``: mass-weighted ``adj_edge_jaccard`` plus
``0.1 * micro-pooled division_jaccard``, with the arm's own edge mass. The loader asserts
per-family parity against the published table before anything is resampled.

Usage:
  .venv\\Scripts\\python.exe scripts\\pooled_bootstrap.py --draws 20000
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from private_split_simulator import SYSTEMS, load, score, score_batch  # noqa: E402

DEFAULT_OUT = Path(r"C:\Users\aryaa\Documents\Biohub-CellTracking-2026\_evidence"
                   r"\agent_runs\laneE\pooled_bootstrap.json")


def draw_iid(rng, n, draws):
    """Crop-block bootstrap: multinomial counts over crops (equivalently, sample with
    replacement). ``score`` accepts non-integer weights, so counts are used directly."""
    return rng.multinomial(n, np.full(n, 1.0 / n), size=draws).astype(float)


def draw_stratified(rng, fam, draws):
    """Same, but resampled within family so family crop counts are held fixed."""
    n = len(fam)
    M = np.zeros((draws, n))
    for f in np.unique(fam):
        idx = np.where(fam == f)[0]
        counts = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=draws)
        M[:, idx] = counts
    return M


def ci(v, lo=2.5, hi=97.5):
    return [round(float(np.percentile(v, lo)), 5), round(float(np.percentile(v, hi)), 5)]


def summarise_scheme(rng, D, scheme, draws, keys, names):
    fam, arr = D["fam"], D["arr"]
    M = draw_iid(rng, len(fam), draws) if scheme == "iid" else draw_stratified(rng, fam, draws)
    s = np.stack([score_batch(arr[k], M) for k in keys])          # (n_arms, draws)
    point = {k: float(score(arr[k], np.ones(len(fam)))) for k in keys}

    per_arm = {}
    for i, k in enumerate(keys):
        per_arm[names[i]] = {
            "point": round(point[k], 5),
            "boot_mean": round(float(s[i].mean()), 5),
            "bias": round(float(s[i].mean() - point[k]), 5),
            "se": round(float(s[i].std(ddof=1)), 5),
            "ci95": ci(s[i]),
            "ci90": ci(s[i], 5, 95),
        }

    pairs = {}
    order = np.argsort([-point[k] for k in keys])
    for a, b in itertools.combinations(range(len(keys)), 2):
        d = s[a] - s[b]
        obs = point[keys[a]] - point[keys[b]]
        pairs[f"{names[a]} - {names[b]}"] = {
            "observed": round(obs, 5),
            "boot_mean": round(float(d.mean()), 5),
            "se": round(float(d.std(ddof=1)), 5),
            "ci95": ci(d),
            "P_first_better": round(float((d > 0).mean()), 4),
            "rank_reversal_P": round(float((d < 0).mean() if obs > 0 else (d > 0).mean()), 4),
            "corr_of_bootstrap_scores": round(float(np.corrcoef(s[a], s[b])[0, 1]), 4),
        }

    ranks = (-s).argsort(0).argsort(0)          # 0 == best
    top1 = {names[i]: round(float((ranks[i] == 0).mean()), 4) for i in range(len(keys))}
    return {
        "draws": draws,
        "resampling_unit": "crop",
        "per_arm": per_arm,
        "pairwise_delta_paired": pairs,
        "P_rank1": top1,
        "point_order": [names[i] for i in order],
        "mean_rank": {names[i]: round(float(ranks[i].mean()) + 1, 3) for i in range(len(keys))},
    }


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)

    D = load()
    keys = list(SYSTEMS)
    names = [SYSTEMS[k] for k in keys]
    rng = np.random.default_rng(a.seed)

    report = {"seed": a.seed, "n_crops": len(D["crops"]),
              "family_counts": {f: int((D["fam"] == f).sum()) for f in np.unique(D["fam"])},
              "schemes": {}}
    for scheme in ("iid", "stratified"):
        report["schemes"][scheme] = summarise_scheme(rng, D, scheme, a.draws, keys, names)

    for scheme, rep in report["schemes"].items():
        print(f"\n=== POOLED OOF, crop-block bootstrap [{scheme}], {rep['draws']} draws ===")
        print(f"{'arm':<12}{'point':>9}{'se':>9}   95% CI")
        for nm, r in sorted(rep["per_arm"].items(), key=lambda kv: -kv[1]["point"]):
            print(f"{nm:<12}{r['point']:>9.5f}{r['se']:>9.5f}   [{r['ci95'][0]:.5f}, {r['ci95'][1]:.5f}]")
        print(f"\n{'paired delta':<26}{'obs':>9}{'se':>9}   95% CI                 P(>0)  reversal")
        for nm, r in sorted(rep["pairwise_delta_paired"].items(),
                            key=lambda kv: -abs(kv[1]["observed"])):
            print(f"{nm:<26}{r['observed']:>+9.5f}{r['se']:>9.5f}   "
                  f"[{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}]   {r['P_first_better']:.4f}  "
                  f"{r['rank_reversal_P']:.4f}")
        print("  P(rank 1):", rep["P_rank1"])

    a.json_out.parent.mkdir(parents=True, exist_ok=True)
    a.json_out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
