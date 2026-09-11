#!/usr/bin/env python3
"""EXP-4 — gate a submission.csv BEFORE it is submitted.

Stdlib only: no polars, no torch, no tracksdata. It must run inside the internet-disabled
Kaggle notebook that produces the file, which is the only place it can prevent the failure.

Exit 0 = safe, 1 = FATAL (would score wrong or zero), 2 = could not read.
"""
import csv, sys, math
from collections import defaultdict, Counter

REQUIRED = ["id","dataset","row_type","node_id","t","z","y","x","source_id","target_id"]
LIMITS = {"t":(0,99), "z":(0,63), "y":(0,255), "x":(0,255)}   # verified from the GT geffs

def main(path, expect_datasets=None):
    fatal, warn, info = [], [], []
    nodes=defaultdict(dict); edges=defaultdict(list); n_rows=0
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        missing=[c for c in REQUIRED if c not in (r.fieldnames or [])]
        if missing: return fail(f"missing columns: {missing}")
        for row in r:
            n_rows+=1; ds=row["dataset"]
            if row["row_type"]=="node":
                try: t,z,y,x=(int(float(row[k])) for k in ("t","z","y","x"))
                except ValueError: fatal.append(f"{ds}: non-numeric node coordinate"); continue
                nid=int(float(row["node_id"]))
                if nid in nodes[ds]: fatal.append(f"{ds}: duplicate node_id {nid}")
                nodes[ds][nid]=(t,z,y,x)
            elif row["row_type"]=="edge":
                edges[ds].append((int(float(row["source_id"])), int(float(row["target_id"]))))
            else:
                fatal.append(f"{ds}: unknown row_type {row['row_type']!r}")

    for ds in sorted(set(nodes)|set(edges)):
        N, E = nodes[ds], edges[ds]
        if not E: fatal.append(f"{ds}: EDGELESS — evaluate() returns early and raises; unscorable")
        # 1. the silent zero
        bad_dt=[(s,t) for s,t in E if s in N and t in N and N[t][0]-N[s][0]!=1]
        if bad_dt:
            fatal.append(f"{ds}: {len(bad_dt)} NON-CONSECUTIVE edges (dt != 1). The scorer DROPS these "
                         f"before counting, so they contribute nothing and can silently score 0.0")
        # 2. dangling endpoints
        dang=[(s,t) for s,t in E if s not in N or t not in N]
        if dang: fatal.append(f"{ds}: {len(dang)} edges reference a node_id that does not exist")
        # 3. exploit / implausible-position signatures
        neg=[n for n,(t,*_) in N.items() if t<0]
        if neg: fatal.append(f"{ds}: {len(neg)} nodes at NEGATIVE t — hub-exploit signature, patched and now scores as FP")
        oob=[n for n,(t,z,y,x) in N.items()
             if not all(LIMITS[k][0]<=v<=LIMITS[k][1] for k,v in zip(("t","z","y","x"),(t,z,y,x)))]
        if oob: fatal.append(f"{ds}: {len(oob)} nodes outside the volume bounds {LIMITS}")
        # 4. out-degree: the scorer keeps at most 2 children
        od=Counter(s for s,_ in E)
        over=[n for n,c in od.items() if c>2]
        if over: warn.append(f"{ds}: {len(over)} nodes with out-degree >2 — a fork must have exactly 2")
        # 5. merges
        idg=Counter(t for _,t in E); merged=[n for n,c in idg.items() if c>1]
        if merged: warn.append(f"{ds}: {len(merged)} nodes with in-degree >1 (merges); the scorer collapses duplicates")
        forks=sum(1 for c in od.values() if c==2)
        info.append(f"{ds}: {len(N):>7} nodes, {len(E):>7} edges, {forks:>4} forks")

    if expect_datasets:
        miss=set(expect_datasets)-set(nodes)
        if miss: fatal.append(f"MISSING datasets: {sorted(miss)} — every test dataset must appear")

    print(f"  rows {n_rows:,} | datasets {len(nodes)}")
    for i in info: print(f"    {i}")
    for w in warn: print(f"  WARN  {w}")
    for e in fatal: print(f"  FATAL {e}")
    print(f"\n  {'FATAL — DO NOT SUBMIT' if fatal else ('PASS with warnings' if warn else 'PASS')}")
    return 1 if fatal else 0

def fail(m): print(f"  FATAL {m}"); return 1
if __name__=="__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:] or None))
