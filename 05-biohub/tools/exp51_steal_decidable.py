#!/usr/bin/env python3
"""EXP-51 -- can a contended division steal be decided on GEOMETRY alone?

EXP-50: 86.8% of missed divisions have a daughter already linked to another parent, so they are
excluded from the candidate set before any gate runs. To recover them the daughter must be STOLEN
from its incumbent parent. This asks whether geometry can decide that, which is the cheap test --
no image reads -- and the same shape as EXP-43's head-to-head.

Two questions, and the second is the one that matters:

1. HEAD-TO-HEAD. For a true contended division P->{C,Q} where Q currently has parent X, is P
   preferred to X by distance? If the incumbent is usually closer, raw distance cannot decide.

2. PRECISION AT THE BREAK-EVEN. Head-to-head is not enough -- a rule has to fire on candidates
   without knowing the answer. So enumerate ALL steal candidates the same way a production rule
   would (any single-child parent P, any contended Q in the next frame within radius), label them
   from GT, and measure precision at various rule thresholds against the base rate.

Division break-even precision is roughly 12-24% at our operating point (Jaccard moves +0.609 per
true division against -0.086 per false one). A steal is worth MORE than a plain division because
the incumbent edge X->Q is already a false positive when Q matched GT, so a correct steal wins
edge TP+1, FP-1, FN-1 as well. But a WRONG steal breaks a correct edge, so precision still rules.

Features are all geometric and all available at inference:
    d_PQ / d_XQ      incumbent head-to-head
    sym              |d(P,C) - d(P,Q)| / mean   -- a division should be symmetric about the parent
    cos              angle between P->C and P->Q -- daughters separate, so cos should be low
    diverge          d(grandchild_C, grandchild_Q) - d(C,Q) -- daughters keep separating
"""
import sys, warnings
from pathlib import Path
import numpy as np, tracksdata as td
from scipy.spatial import cKDTree

sys.path.insert(0, "src"); sys.path.insert(0, "reference/royerlab-baseline/src")
warnings.filterwarnings("ignore")
import div_sweep as D

GT = Path("data/train_geff"); TOL = 7.0; RADIUS = 12.0
N_PER_EMBRYO = int(sys.argv[1]) if len(sys.argv) > 1 else 60


def load(p):
    g = td.graph.IndexedRXGraph.from_geff(p); return g[0] if isinstance(g, tuple) else g


h2h_win = h2h_lose = 0
rows = []                       # (features..., is_true_division)
files = sorted(Path("work/train_graphs").rglob("*.geff"))
sel = [p for p in files if p.stem.startswith("44b6")][:N_PER_EMBRYO] + \
      [p for p in files if p.stem.startswith("6bba")][:N_PER_EMBRYO]
for p in sel:
    gp = GT / f"{p.stem}.geff"
    if not gp.exists(): continue
    g = load(p)
    pos, tt = {}, {}
    for r in g.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); pos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        tt[i] = int(r["t"])
    succ, parent = {}, {}
    for r in g.edge_attrs().iter_rows(named=True):
        s, d = int(r["source_id"]), int(r["target_id"])
        succ.setdefault(s, []).append(d); parent[d] = s
    by_t = {}
    for i, t in tt.items(): by_t.setdefault(t, []).append(i)
    trees = {t: (cKDTree(np.stack([pos[i] for i in ids])), ids) for t, ids in by_t.items()}
    def match(q, t):
        ent = trees.get(t)
        if ent is None: return None
        tr, ids = ent; dd, j = tr.query(q)
        return ids[int(j)] if dd <= TOL else None

    gt = load(gp)
    gpos, gtt = {}, {}
    for r in gt.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); gpos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        gtt[i] = int(r["t"])
    gout = {}
    for r in gt.edge_attrs().iter_rows(named=True):
        gout.setdefault(int(r["source_id"]), []).append(int(r["target_id"]))
    # true divisions expressed in predicted node ids
    true_div = {}
    for s, kids in gout.items():
        if len(kids) < 2 or s not in gpos: continue
        ms = match(gpos[s], gtt[s])
        md = [match(gpos[k], gtt[k]) for k in kids if k in gpos]
        if ms is None or any(x is None for x in md) or len(md) < 2: continue
        true_div[ms] = set(md)

    def feats(P, C, Q):
        dPC = float(np.linalg.norm(pos[C] - pos[P]))
        dPQ = float(np.linalg.norm(pos[Q] - pos[P]))
        sym = abs(dPC - dPQ) / max((dPC + dPQ) / 2, 1e-6)
        a, b = pos[C] - pos[P], pos[Q] - pos[P]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        cos = float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else 1.0
        gc, gq = succ.get(C, []), succ.get(Q, [])
        div = 0.0
        if gc and gq:
            div = float(np.linalg.norm(pos[gq[0]] - pos[gc[0]]) - np.linalg.norm(pos[Q] - pos[C]))
        X = parent.get(Q)
        dXQ = float(np.linalg.norm(pos[Q] - pos[X])) if X is not None else 99.0
        return dPQ, dXQ, sym, cos, div

    # 1. head-to-head on the TRUE contended divisions
    for P, kids in true_div.items():
        if len(succ.get(P, [])) != 1: continue
        C = succ[P][0]
        for Q in kids:
            if Q == C or parent.get(Q) in (None, P): continue
            dPQ, dXQ, *_ = feats(P, C, Q)
            if dPQ < dXQ: h2h_win += 1
            else: h2h_lose += 1

    # 2. enumerate every steal candidate a production rule would see
    for P, ch in succ.items():
        if len(ch) != 1: continue
        C = ch[0]
        t = tt[P]
        ent = trees.get(t + 1)
        if ent is None: continue
        tr, ids = ent
        for k in tr.query_ball_point(pos[P], RADIUS):
            Q = ids[k]
            if Q == C or parent.get(Q) in (None, P): continue   # contended only
            f = feats(P, C, Q)
            rows.append((*f, 1 if (P in true_div and Q in true_div[P]) else 0))

print(f"\n  === 1. HEAD-TO-HEAD on true contended divisions ===")
n = h2h_win + h2h_lose
print(f"  true parent closer than the incumbent: {h2h_win}/{n} = {h2h_win/max(n,1):.1%}")

a = np.array(rows, float)
if len(a) == 0:
    print("  no candidates"); sys.exit()
y = a[:, 5].astype(int)
print(f"\n  === 2. ALL steal candidates: {len(y):,}   true {y.sum()}   BASE RATE {y.mean():.3%} ===")
print(f"  break-even division precision is ~12-24%\n")
names = ["d_PQ", "d_XQ", "sym", "cos", "diverge"]
print(f"  {'rule':<34}{'fires':>8}{'true':>7}{'PRECISION':>11}")
print("  " + "-" * 60)
def rule(name, m):
    k = int(m.sum())
    if k == 0: print(f"  {name:<34}{0:>8}{0:>7}{'-':>11}"); return
    pr = y[m].sum() / k
    flag = "  <-- clears break-even" if pr >= 0.12 else ""
    print(f"  {name:<34}{k:>8,}{int(y[m].sum()):>7}{pr:>10.2%}{flag}")
rule("all candidates (base rate)", np.ones(len(y), bool))
rule("d_PQ < d_XQ", a[:, 0] < a[:, 1])
rule("sym < 0.6", a[:, 2] < 0.6)
rule("sym < 0.3", a[:, 2] < 0.3)
rule("cos < 0.0", a[:, 3] < 0.0)
rule("diverge > 0", a[:, 4] > 0)
rule("d_PQ<d_XQ & sym<0.6", (a[:, 0] < a[:, 1]) & (a[:, 2] < 0.6))
rule("d_PQ<d_XQ & sym<0.3 & cos<0", (a[:, 0] < a[:, 1]) & (a[:, 2] < 0.3) & (a[:, 3] < 0.0))
rule("ALL FOUR", (a[:, 0] < a[:, 1]) & (a[:, 2] < 0.3) & (a[:, 3] < 0.0) & (a[:, 4] > 0))
