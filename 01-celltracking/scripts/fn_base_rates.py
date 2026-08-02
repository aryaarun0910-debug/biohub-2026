"""LANE B — per-stage base-rate table and required selector accuracy.

Reads reports/inventory/edge_fn_census.json and answers, per first-failure stage:

  * 44b6 / 6bba counts
  * candidate-set size a selector would have to rank
  * true-candidate base rate
  * exact oracle composite ceiling (that stage's FNs all converted to TP)
  * minimum selector accuracy needed for +0.005 / +0.011 / +0.015
  * whether the repair can hold node count and edge cardinality fixed

Arithmetic. Two regimes, because they have different denominators:

  ADD (cardinality grows). Adding k candidates at precision q:
      TP' = TP + qk ; FP' = FP + (1-q)k ; FN' = FN - qk
      D = TP+FP+FN  ->  D' = D + (1-q)k
      J' = (TP + qk) / (D + (1-q)k)

  SWAP (cardinality fixed -- the re-aim regime). Net TP gain d:
      J' = (TP + d) / (D - d)
  Strictly better than ADD at equal d: the denominator falls instead of rising.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter, defaultdict

REPO = pathlib.Path(__file__).resolve().parents[1]
ALPHA, DIVW = 0.1, 0.1

# which stages an association repair can even reach, and whether it holds cardinality
STAGE_META = {
    "det_never_detected":           ("detector",     "no  (needs a new node)"),
    "wrapper_node_loss":            ("wrapper",      "no  (node was deleted)"),
    "assoc_synthetic_endpoint":     ("wrapper",      "no  (endpoint is synthetic)"),
    "assoc_nonconsecutive":         ("out of scope", "n/a (not a 1-frame edge)"),
    "assoc_enum_beyond_cap":        ("association",  "yes (swap) but needs radius > 10um"),
    "assoc_enum_relaxed_starved":   ("association",  "YES -- pure re-aim, fixed cardinality"),
    "assoc_bipartite_source_taken": ("association",  "YES -- swap"),
    "assoc_bipartite_target_taken": ("association",  "YES -- swap"),
    "assoc_bipartite_both_taken":   ("association",  "YES -- 2-for-2 swap"),
    "assoc_post_relink_deleted":    ("wrapper",      "yes (retention, no new edges)"),
    "final_endpoint_only":          ("unattributed", "unknown"),
}
ADDRESSABLE = {"assoc_enum_relaxed_starved", "assoc_bipartite_source_taken",
               "assoc_bipartite_target_taken", "assoc_bipartite_both_taken"}


def totals(rows, arm):
    t = Counter()
    for r in rows:
        for k in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn"):
            t[k] += r[arm][k]
    return t


def composite(rows, arm, extra_tp=0.0, extra_fp=0.0):
    """Weighted adj_edge_J + 0.1*div_J, distributing extra TP/FP proportionally by crop weight."""
    tw = sum(r[arm]["edge_tp"] + r[arm]["edge_fp"] + r[arm]["edge_fn"] for r in rows) or 1
    num = den = 0.0
    dtp = dfp = dfn = 0
    for r in rows:
        d = r[arm]
        w0 = d["edge_tp"] + d["edge_fp"] + d["edge_fn"]
        share = w0 / tw
        tp = d["edge_tp"] + extra_tp * share
        fp = d["edge_fp"] + extra_fp * share
        fn = max(0.0, d["edge_fn"] - extra_tp * share)
        w = tp + fp + fn
        if w <= 0:
            continue
        tnr = d["total_node_ratio"]
        if tnr != tnr:
            tnr = 0.0
        num += w0 * max(0.0, (tp / w) * (1 - ALPHA * tnr))
        den += w0
        dtp += d["division_tp"]
        dfp += d["division_fp"]
        dfn += d["division_fn"]
    adj = num / den if den else float("nan")
    dd = dtp + dfp + dfn
    return adj + DIVW * (dtp / dd if dd else 0.0)


def required_d_for(rows, arm, target):
    """Net TP gain d (SWAP regime) needed to raise the composite by `target`."""
    base = composite(rows, arm)
    lo, hi = 0.0, float(totals(rows, arm)["edge_fn"])
    if composite(rows, arm, extra_tp=hi) - base < target:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if composite(rows, arm, extra_tp=mid) - base < target:
            lo = mid
        else:
            hi = mid
    return hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default=str(REPO / "reports/inventory/edge_fn_census.json"))
    ap.add_argument("--arm", default="armB")
    a = ap.parse_args()
    rows = json.load(open(a.census))
    rows = [r for r in rows if a.arm in r and "fn_stages" in r]
    arm = a.arm

    fams = sorted({r["family"] for r in rows})
    stage = defaultdict(Counter)
    for r in rows:
        for k, v in r["fn_stages"].items():
            stage[k][r["family"]] += v
            stage[k]["ALL"] += v

    tot = totals(rows, arm)
    base = composite(rows, arm)
    D = tot["edge_tp"] + tot["edge_fp"] + tot["edge_fn"]
    surf = sum(r.get("candidate_surface", 0) for r in rows)
    surf_t = sum(r.get("candidate_surface_tight", 0) for r in rows)
    surf_r = sum(r.get("candidate_surface_relaxed", 0) for r in rows)

    print("=" * 100)
    print(f"LANE B — EDGE-FN BASE RATES, {arm}, {len(rows)} crops "
          f"({', '.join(f'{f}:{sum(1 for r in rows if r[chr(102)+chr(97)+chr(109)+chr(105)+chr(108)+chr(121)]==f)}' for f in fams)})")
    print("=" * 100)
    print(f"  composite {base:.6f}   edge TP/FP/FN {tot['edge_tp']}/{tot['edge_fp']}/{tot['edge_fn']}"
          f"   D={D}")
    print(f"  relink candidate surface: {surf} pairs (tight {surf_t}, relaxed {surf_r})")

    print(f"\n  {'stage':30s}{'44b6':>8s}{'6bba':>8s}{'ALL':>8s}{'%':>7s}"
          f"{'oracleD':>10s}  {'owner':<13s}{'cardinality'}")
    for st in sorted(stage, key=lambda s: -stage[s]["ALL"]):
        c = stage[st]
        ceil = composite(rows, arm, extra_tp=c["ALL"]) - base
        owner, card = STAGE_META.get(st, ("?", "?"))
        print(f"  {st:30s}{c[fams[0]]:8d}{c[fams[1]] if len(fams)>1 else 0:8d}"
              f"{c['ALL']:8d}{100*c['ALL']/max(sum(x['ALL'] for x in stage.values()),1):6.1f}%"
              f"{ceil:+10.5f}  {owner:<13s}{card}")

    addr = sum(stage[s]["ALL"] for s in ADDRESSABLE if s in stage)
    addr6 = sum(stage[s].get("6bba", 0) for s in ADDRESSABLE if s in stage)
    fn6 = sum(r[arm]["edge_fn"] for r in rows if r["family"] == "6bba")
    print(f"\n  ASSOCIATION-ADDRESSABLE (starved + bipartite): {addr} FNs "
          f"({100*addr/max(tot['edge_fn'],1):.1f}% of all FN)")
    print(f"    oracle composite ceiling if ALL are recovered : "
          f"{composite(rows, arm, extra_tp=addr)-base:+.6f}")
    print(f"    6bba share {addr6} = {100*addr6/max(fn6,1):.2f}% of 6bba FN "
          f"(Gate C0 wants >= 12%)")

    print("\n  REQUIRED NET TP GAIN d (SWAP regime, cardinality fixed)")
    print(f"  {'target':>10s}{'d needed':>12s}{'as % of addressable':>22s}")
    for target in (0.005, 0.011, 0.015, 0.020):
        d = required_d_for(rows, arm, target)
        s = "unreachable" if d is None else f"{d:.1f}"
        pct = "" if d is None else f"{100*d/max(addr,1):.1f}%"
        print(f"  {target:+10.3f}{s:>12s}{pct:>22s}")

    print("\n  MINIMUM SELECTOR ACCURACY, ADD regime  (k candidates selected at precision q)")
    print(f"  {'k':>8s}" + "".join(f"{f'q for {t:+.3f}':>16s}" for t in (0.005, 0.011, 0.015)))
    for k in (500, 1000, 2000, 4000, 8000):
        cells = []
        for target in (0.005, 0.011, 0.015):
            got = None
            for qi in range(1, 1001):
                q = qi / 1000.0
                if composite(rows, arm, extra_tp=q * k, extra_fp=(1 - q) * k) - base >= target:
                    got = q
                    break
            cells.append(f"{'impossible' if got is None else f'{got:.3f}':>16s}")
        print(f"  {k:8d}" + "".join(cells))

    print("\n  BASE RATE OF THE SURFACE A SELECTOR MUST RANK")
    if surf:
        print(f"    addressable FNs / candidate surface = {addr}/{surf} = "
              f"{addr/surf:.6f}  ({surf/max(addr,1):.0f} candidates per useful one)")
        print(f"    relaxed-only surface {surf_r}; starved FNs "
              f"{stage.get('assoc_enum_relaxed_starved', Counter())['ALL']} -> base rate "
              f"{stage.get('assoc_enum_relaxed_starved', Counter())['ALL']/max(surf_r,1):.6f}")


if __name__ == "__main__":
    main()
