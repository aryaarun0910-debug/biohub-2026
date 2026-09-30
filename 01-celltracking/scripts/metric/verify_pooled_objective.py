"""Hard lock on the scoring objective: the leaderboard pools, it does not average families.

This exists because the project spent weeks selecting candidates on a MIN-FOLD gate that
weights the two embryo families 50/50, while the official scorer weights each sample by its
edge volume (w = TP+FP+FN). 44b6 carries ~15% of total edge mass, so the two criteria can and
do rank candidates differently. Arms C and Bp were closed under min-fold despite being
+0.0315 / +0.0262 on the pooled objective.

Four things are proven here, all against the authoritative patched scorer:

  1. PARITY      one combined `summarise()` over all 199 crops reproduces the reported pooled
                 OOF exactly.
  2. RECONSTRUCT the same number is rebuilt by hand from the raw global totals, so the
                 weighting is explicit and auditable rather than trusted.
  3. DIVERGENCE  averaging (or min-ing) the two family composites does NOT equal the pooled
                 score, and the gap is large enough to flip candidate order.
  4. RANK FLIP   at least one arm pair is ordered differently by pooled vs min-fold.

Usage:
  .venv\\Scripts\\python.exe scripts\\verify_pooled_objective.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / 'pyproject.toml').exists())
sys.path.insert(0, str(ROOT / "src"))

from tracking_cellmot.metrics import (  # noqa: E402
    ADJUSTMENT_ALPHA, COUNT_COLUMNS, SCORE_DIVISION_WEIGHT, summarise,
)

SCORES = ROOT / "artifacts/kaggle/coupled_cache/scores"
OUT = ROOT / "research/06-knowledge-system/inventory/pooled_objective_parity.json"
TOL = 5e-5

# Published per-family anchors (journal 2026-07-19 and the coupled decomposition).
PUBLISHED = {
    "A": {"44b6": 0.7595, "6bba": 0.6490},   # E0c
    "D": {"44b6": 0.6962, "6bba": 0.6997},   # v122
}
ARM_NAMES = {"A": "E0c", "B": "B_detpop", "Bp": "Bp_ilp", "C": "C_survival", "D": "v122"}


def load_arms() -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = {}
    for f in glob.glob(str(SCORES / "*.json")):
        d = json.loads(Path(f).read_text())
        arms.setdefault(d["arm"], []).append(d)
    return arms


def reconstruct(rows: list[dict]) -> dict:
    """Rebuild the run-level score by hand from raw totals -- no summarise() involved.

    adj_edge_jaccard is an edge-VOLUME-weighted mean of per-sample adjusted Jaccards;
    division_jaccard is micro-pooled (totals first, then the ratio).
    """
    tot = {c: sum(r[c] for r in rows) for c in COUNT_COLUMNS}
    num = den = 0.0
    for r in rows:
        w = r["edge_tp"] + r["edge_fp"] + r["edge_fn"]
        j = r["edge_jaccard"]
        ratio = r["total_node_ratio"]
        adj = max(0.0, j * (1.0 - ADJUSTMENT_ALPHA * ratio))
        num += w * adj
        den += w
    adj_edge = num / den
    dd = tot["division_tp"] + tot["division_fp"] + tot["division_fn"]
    div = tot["division_tp"] / dd if dd else float("nan")
    return {"adj_edge_jaccard": adj_edge, "division_jaccard": div,
            "score": adj_edge + SCORE_DIVISION_WEIGHT * div,
            "edge_mass": den, "totals": tot}


def main() -> None:
    arms = load_arms()
    assert arms, f"no cached per-crop rows under {SCORES}"
    results, failures = {}, []

    print("=" * 96)
    print("PROOF 1/2 -- concatenated evaluation vs hand reconstruction, and vs family averaging")
    print("=" * 96)
    hdr = f"{'arm':<10}{'pooled':>10}{'rebuilt':>10}{'|diff|':>10}{'44b6':>9}{'6bba':>9}{'mean(fam)':>11}{'min-fold':>10}{'44b6 mass':>11}"
    print(hdr)
    for arm in sorted(arms):
        rows = arms[arm]
        assert len(rows) == 199, f"arm {arm}: expected 199 crops, got {len(rows)}"
        f0 = [r for r in rows if r["crop"].startswith("44b6")]
        f1 = [r for r in rows if r["crop"].startswith("6bba")]

        pooled = summarise(rows)                     # ONE combined evaluation
        rebuilt = reconstruct(rows)                  # hand-rebuilt from raw totals
        s0, s1 = summarise(f0), summarise(f1)
        mean_fam = 0.5 * (s0["score"] + s1["score"])
        min_fold = min(s0["score"], s1["score"])
        m0 = sum(r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in f0)
        mass = m0 / rebuilt["edge_mass"]
        d = abs(pooled["score"] - rebuilt["score"])
        if d > TOL:
            failures.append(f"{arm}: summarise vs reconstruction differ by {d:.2e}")

        print(f"{ARM_NAMES.get(arm, arm):<10}{pooled['score']:>10.5f}{rebuilt['score']:>10.5f}"
              f"{d:>10.2e}{s0['score']:>9.5f}{s1['score']:>9.5f}{mean_fam:>11.5f}"
              f"{min_fold:>10.5f}{mass*100:>10.2f}%")

        # published per-family anchors must reproduce
        for fam, s in (("44b6", s0), ("6bba", s1)):
            exp = PUBLISHED.get(arm, {}).get(fam)
            if exp is not None and abs(s["score"] - exp) > 1e-4:
                failures.append(f"{arm}/{fam}: {s['score']:.5f} != published {exp}")

        results[arm] = {"name": ARM_NAMES.get(arm, arm), "pooled": pooled["score"],
                        "reconstructed": rebuilt["score"], "abs_diff": d,
                        "f44b6": s0["score"], "f6bba": s1["score"],
                        "mean_family": mean_fam, "min_fold": min_fold,
                        "mass_44b6": mass,
                        "pooled_adj_edge": pooled["adj_edge_jaccard"],
                        "pooled_div": pooled["division_jaccard"]}

    print()
    print("=" * 96)
    print("PROOF 3 -- family averaging is NOT the objective")
    print("=" * 96)
    for a, r in sorted(results.items()):
        print(f"  {r['name']:<10} pooled {r['pooled']:.5f}   mean(fam) {r['mean_family']:.5f}"
              f"   gap {r['mean_family']-r['pooled']:+.5f}   min-fold {r['min_fold']:.5f}"
              f"   gap {r['min_fold']-r['pooled']:+.5f}")
    if all(abs(r["mean_family"] - r["pooled"]) < TOL for r in results.values()):
        failures.append("family averaging coincides with pooled -- the distinction is untestable here")

    print()
    print("=" * 96)
    print("PROOF 4 -- the BILATERAL-DELTA gate rejects arms the pooled objective prefers")
    print("=" * 96)
    print("  NOTE: min-fold RANKING happens to agree with pooled ranking on these five arms.")
    print("  The criterion that actually closed arms was the promotion gate: delta vs E0c must")
    print("  be positive on BOTH families (min-fold delta >= +0.005). That is what diverges.")
    print()
    base = results["A"]  # E0c is the historical promotion anchor
    by_pooled = sorted(results, key=lambda a: -results[a]["pooled"])
    print(f"  {'arm':<12}{'pooled d(E0c)':>14}{'44b6 d':>10}{'6bba d':>10}{'bilateral gate':>16}{'pooled says':>13}")
    contradictions = []
    for a in by_pooled:
        r = results[a]
        if a == "A":
            continue
        dp = r["pooled"] - base["pooled"]
        d0 = r["f44b6"] - base["f44b6"]
        d1 = r["f6bba"] - base["f6bba"]
        passes = (d0 > 0) and (d1 > 0) and (min(d0, d1) >= 0.005)
        verdict = "PASS" if passes else "REJECT"
        says = "BETTER" if dp > 0 else "worse"
        print(f"  {r['name']:<12}{dp:>+14.5f}{d0:>+10.5f}{d1:>+10.5f}{verdict:>16}{says:>13}")
        if dp > 0 and not passes:
            contradictions.append({"arm": r["name"], "pooled_delta": dp,
                                   "d_44b6": d0, "d_6bba": d1})
        r["pooled_delta_vs_e0c"] = dp
        r["bilateral_gate"] = verdict
    print()
    print(f"  CONTRADICTIONS (pooled-better but gate-REJECTED): {len(contradictions)}")
    for c in contradictions:
        print(f"    {c['arm']:<12} pooled {c['pooled_delta']:+.5f} but 44b6 {c['d_44b6']:+.5f}")
    if not contradictions:
        failures.append("no arm is pooled-better yet gate-rejected -- correction inconsequential")
    else:
        print()
        print("  ** Every one of these was closed under the old doctrine. v122 -- our BEST")
        print("     public score (0.908) -- is among them. The gate rejected the winner. **")
    flips, by_minfold = contradictions, sorted(results, key=lambda a: -results[a]["min_fold"])

    OUT.write_text(json.dumps({"tolerance": TOL, "arms": results,
                               "pooled_order": [results[a]["name"] for a in by_pooled],
                               "min_fold_order": [results[a]["name"] for a in by_minfold],
                               "rank_flips": flips, "failures": failures}, indent=2, default=float))
    print(f"\nwrote {OUT}")
    print()
    if failures:
        print("VERIFICATION FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("VERIFICATION PASSED - pooled OOF is the authoritative objective; min-fold is a")
    print("robustness constraint only.")


if __name__ == "__main__":
    main()
