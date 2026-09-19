"""Why are 7 of 12 divisions never proposed on the CLEAN raw ILP graph?

The earlier autopsy ran on my own weak rebuild (9.3% orphan pool) and blamed
the assignment holding the daughter. That answer was about my graph. This one
runs on deployed-quality ILP graphs, where safe_div's gates are already at a
local optimum -- so whatever blocks these 7 is NOT a threshold.

Walks each GT division through the same order safe_div uses and records the
first condition that fails.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from biohub import metric2 as M2
exec(open("scripts/24_keep_ilp_edges.py").read().split("stems = sorted")[0].split('"""', 2)[2])

PRED = Path("artifacts/s01_output/tracking_repo/predictions/unknown/unet_transformer_val/split_0")
SCALE = M2.SCALE
rows = []
for p in sorted(PRED.glob("*.geff")):
    P, G = load_pred(p), load_gt(p.stem)
    pos, gpos = P["zyx"] * SCALE, G["zyx"] * SCALE
    p2g, g2p = M2.match(P["t"], P["zyx"], G["t"], G["zyx"])
    gt_out = {}
    for a, b in G["edges"]:
        gt_out.setdefault(int(a), []).append(int(b))
    succ, indeg = {}, {}
    for s, t in P["edges"]:
        succ.setdefault(s, []).append(t); indeg[t] = indeg.get(t, 0) + 1
    by_t = {}
    for i, t in enumerate(P["t"]):
        by_t.setdefault(int(t), []).append(i)
    d = lambda a, b: float(np.linalg.norm(pos[a] - pos[b]))

    for g, ch in gt_out.items():
        if len(ch) < 2:
            continue
        r = dict(film=p.stem, fold=p.stem.split("_")[0], gt=g)
        if g not in g2p:
            r["blocked"] = "parent not detected"; rows.append(r); continue
        miss = [c for c in ch[:2] if c not in g2p]
        if miss:
            r["blocked"] = "daughter not detected"; rows.append(r); continue
        Pn, D1, D2 = g2p[g], g2p[ch[0]], g2p[ch[1]]
        kids = succ.get(Pn, [])
        if len(kids) == 0:
            r["blocked"] = "parent has no child (track ends)"; rows.append(r); continue
        if len(kids) >= 2:
            r["blocked"] = "already a fork"; rows.append(r); continue
        C = kids[0]
        Q = D2 if C == D1 else (D1 if C == D2 else None)
        if Q is None:
            r["blocked"] = "linked child is NEITHER daughter"
            r["d_to_D1"], r["d_to_D2"] = d(C, D1), d(C, D2); rows.append(r); continue
        if indeg.get(Q, 0) != 0:
            other = next(s for s, ts in succ.items() if Q in ts)
            og = p2g.get(other)
            wrong = og is None or (og, p2g.get(Q)) not in {(int(a), int(b)) for a, b in G["edges"]}
            r["blocked"] = f"daughter held by {'a WRONG' if wrong else 'a correct'} parent"
            r["d_thief"], r["d_true"] = d(other, Q), d(Pn, Q); rows.append(r); continue
        # reaches the gates
        orph = [j for j in by_t.get(int(P["t"][C]), []) if indeg.get(j, 0) == 0]
        nn = orph[int(np.argmin(np.linalg.norm(pos[orph] - pos[C], axis=1)))] if orph else None
        dpc, dpq, dcq = d(Pn, C), d(Pn, Q), d(C, Q)
        sc, sq = succ.get(C, []), succ.get(Q, [])
        fails = []
        if nn != Q: fails.append("mutual-NN")
        if dpq > 9.0: fails.append(f"parent {dpq:.1f}")
        if dcq > 14.0: fails.append(f"sister {dcq:.1f}")
        if abs(dpc - dpq) / max((dpc + dpq) / 2, 1e-9) > 0.6:
            fails.append(f"symmetry {abs(dpc-dpq)/((dpc+dpq)/2):.2f}")
        if len(sc) != 1 or len(sq) != 1:
            fails.append("no unique grandchild")
        elif d(sc[0], sq[0]) - dcq < 2.25:
            fails.append(f"diverge {d(sc[0],sq[0])-dcq:.1f}")
        r["blocked"] = "GATE: " + ", ".join(fails) if fails else "PROPOSED"
        r.update(d_pc=dpc, d_pq=dpq, d_cq=dcq, n_orphans=len(orph))
        rows.append(r)

df = pd.DataFrame(rows); df.to_csv("artifacts/autopsy_raw.csv", index=False)
print(f"GT divisions on the 8 deployed graphs: {len(df)}\n")
print(df.blocked.value_counts().to_string())
print("\nper fold:")
print(pd.crosstab(df.blocked, df.fold).to_string())
held = df[df.blocked.str.contains("held by", na=False)]
if len(held):
    print(f"\nheld-daughter cases: thief distance vs true-parent distance")
    for _, r in held.iterrows():
        print(f"  {r.film:<18} thief {r.d_thief:>5.2f} um   true parent {r.d_true:>5.2f} um")
