#!/usr/bin/env python3
"""The error atlas -- WHICH ground-truth items we lose, and WHICH decision lost them.

Aggregate scores tell you that 25 divisions are missing. They never tell you why, so every
threshold gets tuned by grid search against a scalar and the failure modes stay invisible. This
attributes each ground-truth division to the exact stage that lost it:

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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    files = sorted(GT.glob("*.geff"))
    if a.limit: files = files[:a.limit]
    cfg = Config()
    verdicts, per_embryo = Counter(), {}
    fp_reasons = Counter()

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

        # false divisions: predicted forks with no ground-truth division at that parent
        outdeg = Counter(s for s, _ in final)
        for s, n in outdeg.items():
            if n == 2 and s not in gdiv:
                fp_reasons["parent_has_no_GT_division" if s < len(t) else "?"] += 1

    tot = sum(verdicts.values())
    print(f"\n  GROUND-TRUTH DIVISIONS: {tot}  over {len(files)} datasets\n")
    for v, n in verdicts.most_common():
        print(f"    {v:<26} {n:>5}  {n/tot:>6.1%}")
    print(f"\n  FALSE divisions emitted: {sum(fp_reasons.values())}")
    print("\n  per embryo:")
    for emb, c in sorted(per_embryo.items()):
        s = sum(c.values())
        print(f"    {emb} (n={s}): " + "  ".join(f"{k}={v}" for k, v in c.most_common()))


if __name__ == "__main__":
    main()
