#!/usr/bin/env python3
"""EXP-22 -- is the sparse annotation PREDICTABLE? The node-count pool depends on it.

score = edgeJ * (1.1 - 0.1 * N_pred/n_total) + 0.1 * divJ

The 0.947 family predicts ~98.5% of the host's estimated cell count and so collects a multiplier
of 1.0015 out of 1.1000, forfeiting 0.091 of score -- a LARGER pool than divisions (0.077).

Removing nodes at random is catastrophic (our own measurement: scattered 10% costs 0.163) because
a removed node takes its edges with it. But only 2.82% of real cells are annotated, and a
predicted node that matches no GT node contributes NOTHING to edge Jaccard while still counting
in N_pred. If those were identifiable, they could be dropped for free.

So: is the annotated subset STRUCTURED, or is it a random sample of cells? If annotators labelled
a contiguous region, or a time window, or a depth band, then restricting predictions to that
region is legitimate and worth up to 0.09. If it is spatially uniform, the pool is unreachable
and we should stop thinking about it.
"""
import sys
from pathlib import Path
import numpy as np, tracksdata as td

sys.path.insert(0, "reference/royerlab-baseline/src")
from geff import GeffMetadata

GT = Path("data/train_geff")
SHAPE = np.array([64, 256, 256])          # z, y, x in voxels

rows = []
for p in sorted(GT.glob("*.geff")):
    g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g, tuple) else g
    n = g.node_attrs()
    t = np.array([r["t"] for r in n.iter_rows(named=True)])
    zyx = np.array([[r["z"], r["y"], r["x"]] for r in n.iter_rows(named=True)], float)
    est = (GeffMetadata.read(p).extra or {}).get("estimated_number_of_nodes")
    if not est or not len(t):
        continue
    rows.append((p.stem, t, zyx, float(est)))

print(f"  {len(rows)} datasets\n")

# 1. TIME: do annotations cover all frames, or a window?
cov, span = [], []
for stem, t, zyx, est in rows:
    u = np.unique(t)
    cov.append(len(u) / 100.0)
    span.append((u.max() - u.min() + 1) / 100.0)
cov, span = np.array(cov), np.array(span)
print(f"  TIME   frames touched: median {np.median(cov):.1%}  p10 {np.percentile(cov,10):.1%}  p90 {np.percentile(cov,90):.1%}")
print(f"         frame SPAN:     median {np.median(span):.1%}  p10 {np.percentile(span,10):.1%}  p90 {np.percentile(span,90):.1%}")

# 2. SPACE: how much of the volume do annotations occupy?
#    compare the annotated bounding box to the full volume, and to a uniform-random null.
frac_bbox, null_bbox = [], []
rng = np.random.default_rng(0)
for stem, t, zyx, est in rows:
    lo, hi = zyx.min(0), zyx.max(0)
    frac_bbox.append(float(np.prod(np.maximum(hi - lo, 1)) / np.prod(SHAPE)))
    r = rng.random((len(zyx), 3)) * SHAPE          # same count, uniform over the volume
    lo2, hi2 = r.min(0), r.max(0)
    null_bbox.append(float(np.prod(np.maximum(hi2 - lo2, 1)) / np.prod(SHAPE)))
fb, nb = np.array(frac_bbox), np.array(null_bbox)
print(f"\n  SPACE  annotated bbox / volume: median {np.median(fb):.1%}")
print(f"         uniform-random null:      median {np.median(nb):.1%}")
print(f"         ratio (lower = more clustered): {np.median(fb)/np.median(nb):.2f}x")

# 3. Nearest-neighbour clustering vs the null, per dataset (scale-free)
from scipy.spatial import cKDTree
SCALE = np.array([1.625, 0.40625, 0.40625])
obs, nul = [], []
for stem, t, zyx, est in rows[:60]:
    for tt in np.unique(t)[:20]:
        P = zyx[t == tt] * SCALE
        if len(P) < 4:
            continue
        d, _ = cKDTree(P).query(P, k=2)
        obs.append(np.median(d[:, 1]))
        R = rng.random((len(P), 3)) * (SHAPE * SCALE)
        d2, _ = cKDTree(R).query(R, k=2)
        nul.append(np.median(d2[:, 1]))
obs, nul = np.array(obs), np.array(nul)
print(f"\n  CLUSTERING  median NN distance, annotated: {np.median(obs):.2f} um")
print(f"              same count, uniform random:    {np.median(nul):.2f} um")
print(f"              ratio: {np.median(obs)/np.median(nul):.2f}x  (<1 = clustered, ~1 = uniform)")
