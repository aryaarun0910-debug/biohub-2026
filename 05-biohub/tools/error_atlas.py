#!/usr/bin/env python3
"""The error atlas -- WHICH ground-truth items we lose, and WHICH decision lost them.

Aggregate scores tell you that 25 divisions are missing. They never tell you why, so every
threshold gets tuned by grid search against a scalar and the failure modes stay invisible. This
attributes each ground-truth division to the exact stage that lost it:

FALSE divisions get the same treatment, and they need it more: 81 FP against 25 FN is the
larger error and was the cruder analysis. Two questions are asked of every emitted fork.

STRUCTURALLY, what did the parent really do in the ground truth -- continue as one track, end,
divide (and we picked the wrong daughter), or is it a node whose successors simply are not
annotated? Ground truth is 2.82% dense, so "no GT successor" usually means UNANNOTATED, not
"the cell vanished", and a fork there is unfalsifiable rather than wrong.

GEOMETRICALLY, does it look like a real division? We now hold 76,217 Zebrahub divisions (EXP-13),
so we can ask whether a false fork sits inside the real division distribution in cos, sister
separation and parent arc. A fork that looks exactly like every real division probably IS one
that nobody annotated; a fork with implausible geometry is a genuine linking error. Only the
second kind is fixable, and knowing the split tells us whether FP is even worth attacking.

    RECOVERED          both daughter edges survive to the final graph
    NOT_PROPOSED       score_edges never offered one of the daughter edges as a candidate
    LOST_IN_ASSIGNMENT the candidate existed but the Hungarian pass gave the target to someone else
                       and the fork pass never retried it
    FORK_REJECTED/<r>  the fork pass tried and a named rule refused it (cos/sister/parent/diverge)
    REPAIR_DESTROYED   resolve produced it and repair removed it

Under oracle detection node index i IS ground-truth node i, so the attribution is exact rather
than distance-matched.
"""
import sys, argparse
from collections import Counter
from pathlib import Path
import numpy as np, tracksdata as td

sys.path.insert(0, "src"); sys.path.insert(0, "reference/royerlab-baseline/src")
from biohub.contracts import Config
from biohub.detect import detect_oracle
from biohub.edges import score_edges
from biohub.refine import refine
from biohub.resolve import resolve
from biohub.repair import repair
from biohub.trace import Trace

GT = Path("data/train_geff")


def gt_tables(p):
    g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g, tuple) else g
    n = g.node_attrs(); e = g.edge_attrs()
    ids = [r["node_id"] for r in n.iter_rows(named=True)]
    idx = {nid: i for i, nid in enumerate(ids)}
    t = np.array([r["t"] for r in n.iter_rows(named=True)])
    zyx = np.array([[r["z"], r["y"], r["x"]] for r in n.iter_rows(named=True)], float)
    edges = [(idx[r["source_id"]], idx[r["target_id"]])
             for r in e.iter_rows(named=True)
             if r["source_id"] in idx and r["target_id"] in idx]
    return t, zyx, edges


def divisions(edges):
    out = {}
    for s, d in edges:
        out.setdefault(s, []).append(d)
    return {s: kids for s, kids in out.items() if len(kids) == 2}


def geometry(P, parent, a, b):
    """cos between the two parent->daughter arcs, sister separation, longest arc -- all um."""
    va, vb = P[a] - P[parent], P[b] - P[parent]
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-9 or nb < 1e-9:
        return None
    return (float(va @ vb / (na * nb)), float(np.linalg.norm(P[a] - P[b])), float(max(na, nb)))


# Zebrahub reference envelope for a REAL division, from 76,217 of them (EXP-13, ZSNS003+ZSNS005).
# p1..p99 of each quantity; a fork inside all three is geometrically indistinguishable from real.
ZEB = {"cos": (-0.999, 0.060), "sister": (2.942, 14.496), "arc": (1.315, 8.964)}


def violations(g):
    """Which envelope dimensions a fork falls outside. Empty tuple = looks like a real division."""
    if g is None:
        return ("degenerate",)
    cos, sis, arc = g
    v = []
    if not ZEB["cos"][0] <= cos <= ZEB["cos"][1]:       v.append("cos")
    if not ZEB["sister"][0] <= sis <= ZEB["sister"][1]: v.append("sister")
    if not ZEB["arc"][0] <= arc <= ZEB["arc"][1]:       v.append("arc")
    return tuple(v)


def plausible(g):
    return violations(g) == ()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    files = sorted(GT.glob("*.geff"))
    if a.limit: files = files[:a.limit]
    cfg = Config()
    verdicts, per_embryo = Counter(), {}
    fp_reasons, fp_viol, fp_geo, tp_geo = Counter(), Counter(), [], []

    for p in files:
        t, zyx, gedges = gt_tables(p)
        gdiv = divisions(gedges)
        if not gdiv:
            continue
        tr = Trace()
        g = detect_oracle(t.copy(), zyx.copy(), p.stem)
        g = refine(g, cfg)
        g = score_edges(g, cfg)
        cand = {(int(s), int(d)) for s, d in g.edges}
        after_resolve = resolve(g, cfg, trace=tr)
        chosen = {(int(s), int(d)) for s, d in after_resolve.edges}
        final = {(int(s), int(d)) for s, d in repair(after_resolve, cfg).edges}
        rej = {}
        for e in tr.by_kind("fork_reject"):
            rej[(e["parent"], e["second"])] = e["reason"]
            rej.setdefault((e["parent"], e["first"]), e["reason"])

        emb = p.stem[:4]
        per_embryo.setdefault(emb, Counter())
        for par, kids in gdiv.items():
            e1, e2 = (par, kids[0]), (par, kids[1])
            if e1 in final and e2 in final:
                v = "RECOVERED"
            elif not (e1 in cand and e2 in cand):
                v = "NOT_PROPOSED"
            elif e1 in chosen and e2 in chosen:
                v = "REPAIR_DESTROYED"
            elif e1 in rej or e2 in rej:
                v = f"FORK_REJECTED/{rej.get(e1) or rej.get(e2)}"
            else:
                v = "LOST_IN_ASSIGNMENT"
            verdicts[v] += 1; per_embryo[emb][v] += 1

        # ---- false divisions, attributed structurally AND geometrically ----
        P = after_resolve.um()
        gt_out, gt_in = Counter(s for s, _ in gedges), Counter(d for _, d in gedges)
        kids_of = {}
        for s_, d_ in gedges:
            kids_of.setdefault(s_, []).append(d_)
        pred_kids = {}
        for s_, d_ in final:
            pred_kids.setdefault(s_, []).append(d_)
        for par, kids in pred_kids.items():
            if len(kids) != 2:
                continue
            a, b = kids
            if par in gdiv and set(kids) == set(gdiv[par]):
                tp_geo.append(geometry(P, par, a, b))      # a true positive, counted above
                continue
            g = geometry(P, par, a, b)
            if par in gdiv:
                why = "GT_DIVIDES_wrong_daughter"
            elif gt_out.get(par, 0) == 1:
                why = "GT_CONTINUES_second_child_invented"
            elif gt_out.get(par, 0) == 0:
                # sparse GT: the parent's successors are simply not annotated
                why = "GT_SUCCESSOR_UNANNOTATED"
            else:
                why = "other"
            fp_geo.append(g)
            fp_viol["+".join(violations(g)) or "NONE (looks real)"] += 1
            stolen = sum(1 for k in kids if gt_in.get(k, 0) == 1 and k not in kids_of.get(par, []))
            fp_reasons[why] += 1
            fp_reasons[f"  {why} :: geometry {'PLAUSIBLE' if plausible(g) else 'implausible'}"] += 1
            if stolen:
                fp_reasons[f"  {why} :: daughter stolen from another GT track"] += stolen

    tot = sum(verdicts.values())
    print(f"\n  GROUND-TRUTH DIVISIONS: {tot}  over {len(files)} datasets\n")
    for v, n in verdicts.most_common():
        print(f"    {v:<26} {n:>5}  {n/tot:>6.1%}")
    top = {k: v for k, v in fp_reasons.items() if not k.startswith("  ")}
    tf = sum(top.values())
    print(f"\n  FALSE divisions emitted: {tf}\n")
    for k, v in sorted(top.items(), key=lambda x: -x[1]):
        print(f"    {k:<38} {v:>5}  {v/max(tf,1):>6.1%}")
        for k2, v2 in sorted(fp_reasons.items()):
            if k2.startswith(f"  {k} ::"):
                print(f"      {k2.split('::')[1].strip():<36} {v2:>5}")
    pl = sum(v for k, v in fp_reasons.items() if k.endswith("geometry PLAUSIBLE"))
    print(f"\n    geometrically indistinguishable from a real division: {pl}/{tf} = {pl/max(tf,1):.1%}")
    print(f"    (envelope = p1..p99 of 76,217 Zebrahub divisions, EXP-13)")
    print(f"\n    which dimension puts a false fork outside the envelope:")
    for k, v in fp_viol.most_common():
        print(f"      {k:<24} {v:>5}")
    def dist(name, rows, i):
        vals = np.array([r[i] for r in rows if r])
        if not len(vals): return
        q = np.percentile(vals, (5, 25, 50, 75, 95))
        print(f"      {name:<16} n={len(vals):>4}  p5={q[0]:>7.2f} p25={q[1]:>7.2f} "
              f"p50={q[2]:>7.2f} p75={q[3]:>7.2f} p95={q[4]:>7.2f}")
    print(f"\n    geometry of our OWN forks (um; cos is dimensionless):")
    for i, nm in enumerate(("cos", "sister", "arc")):
        print(f"      --- {nm}")
        dist("TRUE  divisions", tp_geo, i)
        dist("FALSE divisions", fp_geo, i)
    print("\n  per embryo:")
    for emb, c in sorted(per_embryo.items()):
        s = sum(c.values())
        print(f"    {emb} (n={s}): " + "  ".join(f"{k}={v}" for k, v in c.most_common()))


if __name__ == "__main__":
    main()
