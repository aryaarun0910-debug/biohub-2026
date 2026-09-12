#!/usr/bin/env python3
"""EXP-17 -- are the ground-truth edges we lose RECOVERABLE, or does the truth look wrong?

A 42nd-place competitor reported (Focus3D thread, 2026-09-12) that on 381 lost GT edges across
15 films, velocity extrapolation picks the true child in 0 of them, and swapping is cheaper under
velocity continuity in 0 of 271 -- median margin 16.75um AGAINST the truth. Both parents sat
~1.5um from what they linked, matching a 1.16um/frame step, while the truth asked for ~9um jumps.

If our losses have the same shape then velocity/Kalman linking is dead for us too, and the losses
are bounded by ground-truth quality rather than by our model. Independently, our own DB records
that Kaggle GT contains jumps to -37 voxels (-60um), which the forum reads as dropped or repeated
frames. This tests whether the two observations are the same phenomenon.
"""
import sys
from collections import Counter
from pathlib import Path
import numpy as np, tracksdata as td

sys.path.insert(0, "src"); sys.path.insert(0, "reference/royerlab-baseline/src")
from biohub.contracts import Config, SCALE
from biohub.detect import detect_oracle
from biohub.edges import score_edges
from biohub.refine import refine
from biohub.resolve import resolve
from biohub.repair import repair

GT = Path("data/train_geff")
cfg = Config()
lost_d, kept_d, step_d = [], [], []
verdict = Counter()

for p in sorted(GT.glob("*.geff")):
    g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g, tuple) else g
    n = g.node_attrs(); e = g.edge_attrs()
    idx = {r["node_id"]: i for i, r in enumerate(n.iter_rows(named=True))}
    t = np.array([r["t"] for r in n.iter_rows(named=True)])
    zyx = np.array([[r["z"], r["y"], r["x"]] for r in n.iter_rows(named=True)], float)
    P = zyx * SCALE
    ge = [(idx[r["source_id"]], idx[r["target_id"]]) for r in e.iter_rows(named=True)
          if r["source_id"] in idx and r["target_id"] in idx]
    if not ge:
        continue
    gg = repair(resolve(score_edges(refine(detect_oracle(t.copy(), zyx.copy(), p.stem), cfg), cfg),
                        cfg), cfg)
    pred = {(int(a), int(b)) for a, b in gg.edges}
    cand = {(int(a), int(b)) for a, b in score_edges(detect_oracle(t.copy(), zyx.copy(), p.stem), cfg).edges}
    chose_from = {}
    for a, b in pred:
        chose_from.setdefault(a, []).append(b)

    # typical per-frame step, this dataset, from GT edges we DID reproduce
    for a, b in ge:
        if (a, b) in pred:
            step_d.append(float(np.linalg.norm(P[b] - P[a])))

    for a, b in ge:
        if (a, b) in pred:
            continue
        d_true = float(np.linalg.norm(P[b] - P[a]))
        lost_d.append(d_true)
        alt = chose_from.get(a, [])
        if alt:
            kept_d.append(min(float(np.linalg.norm(P[c] - P[a])) for c in alt))
        if (a, b) not in cand:
            verdict["truth OUTSIDE our candidate radius"] += 1
        elif alt and min(float(np.linalg.norm(P[c] - P[a])) for c in alt) < d_true:
            verdict["truth was a candidate but LONGER than what we chose"] += 1
        else:
            verdict["truth was a candidate and not obviously worse"] += 1

lost_d, kept_d, step_d = map(np.array, (lost_d, kept_d, step_d))
print(f"\n  GT edges reproduced: {len(step_d):,}   lost: {len(lost_d):,} "
      f"({len(lost_d)/(len(lost_d)+len(step_d)):.2%})\n")


def q(name, v):
    if not len(v): return
    pc = np.percentile(v, (10, 50, 90, 99))
    print(f"    {name:<34} n={len(v):>6}  p10={pc[0]:>6.2f} p50={pc[1]:>6.2f} "
          f"p90={pc[2]:>6.2f} p99={pc[3]:>6.2f}")


print("  displacement in um:")
q("typical step (edges we got right)", step_d)
q("what the LOST truth demanded", lost_d)
q("what we chose for that parent", kept_d)
print(f"\n  verdicts:")
tot = sum(verdict.values())
for k, v in verdict.most_common():
    print(f"    {k:<46} {v:>5}  {v/max(tot,1):>6.1%}")
if len(kept_d):
    m = min(len(kept_d), len(lost_d))
    worse = (lost_d[:m] > kept_d[:m]).mean()
    print(f"\n  the truth is a LONGER jump than our choice in {worse:.1%} of contested cases")
    print(f"  median margin against the truth: {np.median(lost_d[:m]-kept_d[:m]):+.2f} um")
