#!/usr/bin/env python3
"""EXP-49 -- do ANNOTATED cells live in long predicted components? The Ultrack protocol says yes.

The published Ultrack ground-truth protocol (Nature Methods 2025) selects "long,
green-overlapping, high-quality lineages" -- 152 tracklets spanning 85-521 frames. If the
competition annotations follow it, annotation membership correlates with LINEAGE LENGTH, and
comp_len was already the strongest single feature in EXP-36 (AUC 0.726) without us knowing why.

The competition provenance is UNCONFIRMED, so this measures the claim directly rather than
assuming it: on the four real test movies, compare the predicted-component length of nodes that
MATCH a released annotation against those that do not.

If matched nodes sit in components far longer than unmatched ones, aggressive OUTPUT_MIN_TRACK_LEN
pruning is nearly free and the multiplier -- currently 1.0138 of a possible 1.1000 -- has a lot
more to give. We have only swept to minlen 14.

Caveat kept in view: predicted components are FRAGMENTED, so a real 300-frame lineage may appear
as several shorter pieces. This measures predicted length, which is what the pruning acts on.
"""
import sys, warnings
from pathlib import Path
import numpy as np, polars as pl, tracksdata as td
from scipy.spatial import cKDTree

sys.path.insert(0, "src"); sys.path.insert(0, "tools"); sys.path.insert(0, "reference/royerlab-baseline/src")
warnings.filterwarnings("ignore")
import div_sweep as D

GT = Path("data/train_geff"); TOL = 7.0
STEMS = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]
SUB = sys.argv[1] if len(sys.argv) > 1 else "work/repro_out/submission.csv"

df = pl.read_csv(SUB)
allm, allu = [], []
print(f"  submission: {SUB}\n")
print(f"  {'stem':<16}{'matched':>9}{'med len':>9}{'p10':>7}{'unmatched':>11}{'med len':>9}{'p90':>7}")
for s in STEMS:
    d = df.filter(pl.col("dataset") == s)
    nodes = d.filter(pl.col("row_type") == "node")
    edges = d.filter(pl.col("row_type") == "edge")
    nid = nodes["node_id"].to_list()
    pos = {i: np.array([z, y, x]) * D.VOXEL_SCALE_UM
           for i, z, y, x in zip(nid, nodes["z"], nodes["y"], nodes["x"])}
    tt = dict(zip(nid, [int(t) for t in nodes["t"]]))
    par = {i: i for i in nid}
    def find(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    for a, b in zip(edges["source_id"], edges["target_id"]):
        if a in par and b in par:
            ra, rb = find(a), find(b)
            if ra != rb: par[ra] = rb
    members = {}
    for i in nid: members.setdefault(find(i), []).append(i)
    clen = {i: len(members[find(i)]) for i in nid}

    gt = td.graph.IndexedRXGraph.from_geff(GT / f"{s}.geff")
    gt = gt[0] if isinstance(gt, tuple) else gt
    by_t = {}
    for i, t in tt.items(): by_t.setdefault(t, []).append(i)
    trees = {t: (cKDTree(np.stack([pos[i] for i in ids])), ids) for t, ids in by_t.items()}
    matched = set()
    for r in gt.node_attrs().iter_rows(named=True):
        q = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        ent = trees.get(int(r["t"]))
        if ent is None: continue
        tr, ids = ent; dd, j = tr.query(q)
        if dd <= TOL: matched.add(ids[int(j)])
    m = np.array([clen[i] for i in matched]) if matched else np.array([0])
    u = np.array([clen[i] for i in nid if i not in matched])
    allm += list(m); allu += list(u)
    print(f"  {s:<16}{len(m):>9,}{np.median(m):>9.0f}{np.percentile(m,10):>7.0f}"
          f"{len(u):>11,}{np.median(u):>9.0f}{np.percentile(u,90):>7.0f}")

m, u = np.array(allm), np.array(allu)
print(f"\n  ALL   matched {len(m):,} (median component {np.median(m):.0f} nodes)"
      f"   unmatched {len(u):,} (median {np.median(u):.0f})")
print(f"\n  {'minlen':>7}{'matched KEPT':>14}{'unmatched KEPT':>16}{'nodes pruned':>14}")
for k in (9, 14, 20, 30, 50, 80, 120, 200):
    mk = (m >= k).mean(); uk = (u >= k).mean()
    pruned = 1 - (mk*len(m) + uk*len(u)) / (len(m) + len(u))
    print(f"  {k:>7}{mk:>13.1%}{uk:>16.1%}{pruned:>14.1%}")
print("\n  'matched KEPT' must stay at ~100% or edge Jaccard collapses; 'nodes pruned' is")
print("  what buys multiplier. The gap between the two columns is the whole opportunity.")
