#!/usr/bin/env python3
"""Confidence interval on a local-proxy delta, by paired bootstrap over GT EDGES.

Every decision today rests on differences measured by tools/score_submission_local.py: minlen9
+0.0032, ppgrid+minlen9 +0.0082. None has had an error bar. The proxy scores against the released
annotations on the four scored films, which are only ~18-25% of typical embryo density, so the
sample is small: about 2,100 GT edges in total.

Resampling DATASETS is useless here -- there are four. So this resamples GT EDGES within each
dataset, paired (the same resampled edge set scores both submissions), and recomputes the full
weighted score each time. Pairing matters: the two submissions share most of their edges, so an
unpaired interval would be dominated by variance that cancels.

    python tools/proxy_ci.py work/repro_out/submission.csv work/ppgridml9_out/submission.csv
"""
import sys, warnings
from pathlib import Path
import numpy as np, polars as pl, tracksdata as td
from scipy.spatial import cKDTree

sys.path.insert(0, "src"); sys.path.insert(0, "tools"); sys.path.insert(0, "reference/royerlab-baseline/src")
warnings.filterwarnings("ignore")
from geff import GeffMetadata
import div_sweep as D

GT = Path("data/train_geff"); TOL = 7.0
STEMS = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]
B = 2000


def per_edge(path):
    """For each GT edge: was it predicted? Plus the per-dataset multiplier and chargeable FP count."""
    df = pl.read_csv(path)
    out = {}
    for s in STEMS:
        d = df.filter(pl.col("dataset") == s)
        nodes = d.filter(pl.col("row_type") == "node")
        edges = d.filter(pl.col("row_type") == "edge")
        nid = nodes["node_id"].to_list()
        pos = {i: np.array([z, y, x]) * D.VOXEL_SCALE_UM
               for i, z, y, x in zip(nid, nodes["z"], nodes["y"], nodes["x"])}
        tt = dict(zip(nid, [int(t) for t in nodes["t"]]))
        eset = {(int(a), int(b)) for a, b in zip(edges["source_id"], edges["target_id"])}
        by_t = {}
        for i, t in tt.items(): by_t.setdefault(t, []).append(i)
        trees = {t: (cKDTree(np.stack([pos[i] for i in ids])), ids) for t, ids in by_t.items()}
        def match(q, t):
            ent = trees.get(t)
            if ent is None: return None
            tr, ids = ent; dd, j = tr.query(q)
            return ids[int(j)] if dd <= TOL else None
        gt = td.graph.IndexedRXGraph.from_geff(GT / f"{s}.geff")
        gt = gt[0] if isinstance(gt, tuple) else gt
        gpos, gtt = {}, {}
        for r in gt.node_attrs().iter_rows(named=True):
            i = int(r["node_id"]); gpos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
            gtt[i] = int(r["t"])
        hit, gt_nodes = [], set()
        for r in gt.edge_attrs().iter_rows(named=True):
            a, b = int(r["source_id"]), int(r["target_id"])
            if a not in gpos or b not in gpos: continue
            ma, mb = match(gpos[a], gtt[a]), match(gpos[b], gtt[b])
            for x in (ma, mb):
                if x is not None: gt_nodes.add(x)
            hit.append(1 if (ma is not None and mb is not None and (ma, mb) in eset) else 0)
        charge = sum(1 for a, b in eset if a in gt_nodes or b in gt_nodes)
        est = float((GeffMetadata.read(GT / f"{s}.geff").extra or {})["estimated_number_of_nodes"])
        mult = 1 - 0.1 * ((len(nid) - est) / est)
        out[s] = (np.array(hit), charge, mult)
    return out


def score(sample, per, idx=None):
    num = den = 0.0
    for s in STEMS:
        hit, charge, mult = per[s]
        h = hit if idx is None else hit[idx[s]]
        tp = int(h.sum()); n = len(h)
        fn = n - tp
        fp = max(charge - tp, 0)
        j = tp / max(tp + fp + fn, 1)
        w = tp + fp + fn
        num += j * mult * w; den += w
    return num / max(den, 1)


A, Bp = sys.argv[1], sys.argv[2]
pa, pb = per_edge(A), per_edge(Bp)
sa, sb = score(None, pa), score(None, pb)
print(f"\n  A {A}\n    score {sa:.5f}")
print(f"  B {Bp}\n    score {sb:.5f}")
print(f"  delta {sb - sa:+.5f}\n")

rng = np.random.default_rng(0)
deltas = []
for _ in range(B):
    idx = {s: rng.integers(0, len(pa[s][0]), len(pa[s][0])) for s in STEMS}
    deltas.append(score(None, pb, idx) - score(None, pa, idx))
d = np.array(deltas)
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"  paired bootstrap over GT edges, B={B}")
print(f"    mean delta {d.mean():+.5f}")
print(f"    95% CI     [{lo:+.5f}, {hi:+.5f}]")
print(f"    P(delta <= 0) = {(d <= 0).mean():.3f}")
print(f"\n  total GT edges in the proxy: {sum(len(pa[s][0]) for s in STEMS):,}")
print("  NB the leaderboard resolves 0.001.")
