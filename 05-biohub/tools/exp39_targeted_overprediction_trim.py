#!/usr/bin/env python3
"""EXP-39 -- trim only where N_pred EXCEEDS n_total, which is the one provably-spurious pool.

Every previous N_pred attempt failed on the same wall (EXP-22, EXP-34, v-outgrid, EXP-36):
97.3% of predicted nodes match no GT node, but almost all of those are REAL cells that a human
simply did not annotate, so removing them destroys edges. Chance is 97.3% "safe" and the
break-even precision is 99.9%. Nothing clears that.

This is different. The score applies the multiplier PER DATASET:

    adj_i = J_i * (1 - 0.1 * (N_pred_i - n_total_i)/n_total_i)

and adj is then weight-averaged by each sample's annotated-edge count (metrics.summarise).
n_total is `estimated_number_of_nodes` and ships in the GEFF metadata, so it is KNOWN AT
INFERENCE with no ground truth at all. Their base run spreads from -0.288 to +0.358:

    44b6_267148e4  ratio +0.163  mult 0.9837  edgeJ 0.8182
    6bba_07e24132  ratio +0.358  mult 0.9642  edgeJ 0.8384   <- worst of eight
    6bba_085bf656  ratio -0.019  mult 1.0019  edgeJ 0.9932   <- best of eight

Two things follow. The over-predicting datasets are exactly the ones with the worst edge
Jaccard, so over-detection looks causally upstream of the fragmentation. And where N_pred >
n_total the EXCESS cannot be real-but-unannotated cells -- there are not that many cells to
be unannotated. That excess is provably spurious, which is the property EXP-36 could not find.

So: trim only datasets above a ratio threshold, only down to a target ratio, removing nodes in
order of increasing connected-component size (comp_len was the strongest single annotation
signal in EXP-36 at AUC 0.726). Scored with the competition's own evaluate().
"""
import sys, warnings
from pathlib import Path
import numpy as np, polars as pl, tracksdata as td

sys.path.insert(0, "src"); sys.path.insert(0, "tools")
sys.path.insert(0, "reference/royerlab-baseline/src")
warnings.filterwarnings("ignore")
from geff import GeffMetadata
from tracking_cellmot.metrics import evaluate, per_sample_metrics, node_recall
from biohub.contracts import SCALE

GT = Path("data/train_geff"); GRAPHS = Path("work/train_graphs")
N_PER_EMBRYO = int(sys.argv[1]) if len(sys.argv) > 1 else 25


def comp_size(nodes, edges):
    parent = {n: n for n in nodes}
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for s, d in edges:
        ra, rb = find(s), find(d)
        if ra != rb: parent[ra] = rb
    sz = {}
    for n in nodes: sz[find(n)] = sz.get(find(n), 0) + 1
    return {n: sz[find(n)] for n in nodes}


def build(nodes_by_id, keep, edges):
    G = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"): G.add_node_attr_key(k, pl.Float64, -999999.0)
    ks = [k for k in nodes_by_id if k in keep]
    ids = G.bulk_add_nodes([{"t": int(nodes_by_id[k]["t"]), "z": float(nodes_by_id[k]["z"]),
                             "y": float(nodes_by_id[k]["y"]), "x": float(nodes_by_id[k]["x"])}
                            for k in ks])
    m = dict(zip(ks, ids))
    ee = [{"source_id": m[s], "target_id": m[d]} for s, d in edges if s in m and d in m]
    if ee: G.bulk_add_edges(ee)
    return G


cache = []
files = sorted(GRAPHS.rglob("*.geff"))
sel = [p for p in files if p.stem.startswith("44b6")][:N_PER_EMBRYO] + \
      [p for p in files if p.stem.startswith("6bba")][:N_PER_EMBRYO]
for p in sel:
    gp = GT / f"{p.stem}.geff"
    if not gp.exists(): continue
    g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g, tuple) else g
    nbi = {int(r["node_id"]): {"t": int(r["t"]), "z": r["z"], "y": r["y"], "x": r["x"]}
           for r in g.node_attrs().iter_rows(named=True)}
    edges = [(int(r["source_id"]), int(r["target_id"])) for r in g.edge_attrs().iter_rows(named=True)]
    gt = td.graph.IndexedRXGraph.from_geff(gp); gt = gt[0] if isinstance(gt, tuple) else gt
    est = (GeffMetadata.read(gp).extra or {}).get("estimated_number_of_nodes")
    if est is None or len(gt.edge_attrs()) == 0: continue
    cache.append((nbi, edges, comp_size(set(nbi), edges), gt, float(est), p.stem[:4], p.stem))

ratios = [(len(c[0]) - c[4]) / c[4] for c in cache]
over = sum(1 for r in ratios if r > 0)
print(f"  {len(cache)} datasets   node ratio {min(ratios):+.3f} .. {max(ratios):+.3f}   "
      f"median {np.median(ratios):+.3f}   OVER-PREDICTING {over}/{len(cache)}\n", flush=True)


def run(trim_above, trim_to):
    """Trim datasets whose ratio exceeds trim_above down to ratio trim_to, smallest comps first."""
    rows = []
    for nbi, edges, sz, gt, est, emb, stem in cache:
        ratio = (len(nbi) - est) / est
        keep = set(nbi)
        if trim_above is not None and ratio > trim_above:
            target = int(est * (1.0 + trim_to))
            n_drop = max(0, len(nbi) - target)
            if n_drop:
                order = sorted(nbi, key=lambda n: (sz.get(n, 1), n))
                keep = set(nbi) - set(order[:n_drop])
        pred = build(nbi, keep, edges)
        e_kept = [(s, d) for s, d in edges if s in keep and d in keep]
        er = evaluate(pred, gt, scale=tuple(SCALE), max_distance=7.0)
        nr = node_recall(pred, gt) if e_kept else 0.0
        m = per_sample_metrics(er, est, nr); m["emb"] = emb
        rows.append(m)
    return pl.DataFrame(rows)


def agg(df):
    w = (df["edge_tp"] + df["edge_fp"] + df["edge_fn"]).to_numpy().astype(float)
    a = df["adj_edge_jaccard"].to_numpy()
    ok = ~np.isnan(a) & (w > 0)
    adj = float(np.average(a[ok], weights=w[ok])) if ok.any() else float("nan")
    dt, dfp, dfn = df["division_tp"].sum(), df["division_fp"].sum(), df["division_fn"].sum()
    dJ = dt / max(dt + dfp + dfn, 1)
    et, ef, en = df["edge_tp"].sum(), df["edge_fp"].sum(), df["edge_fn"].sum()
    return adj + 0.1 * dJ, et / max(et + ef + en, 1), dJ, adj


CFGS = [("no trim", None, None),
        ("ratio>0.00 -> 0.00", 0.00, 0.00),
        ("ratio>0.05 -> 0.00", 0.05, 0.00),
        ("ratio>0.10 -> 0.00", 0.10, 0.00),
        ("ratio>0.00 -> -0.05", 0.00, -0.05),
        ("ratio>0.00 -> -0.10", 0.00, -0.10),
        ("ratio>-0.10 -> -0.15", -0.10, -0.15),
        ("ALL -> -0.20", -1.0, -0.20)]

hdr = f"  {'config':<22} {'edgeJ':>8} {'divJ':>7} {'adj':>8} {'SCORE':>8} {'delta':>8}"
print(hdr); print("  " + "-" * (len(hdr) - 2), flush=True)
store, b = {}, None
for name, ta, tt in CFGS:
    df = run(ta, tt); store[name] = df
    sc, eJ, dJ, adj = agg(df)
    if b is None: b = sc
    print(f"  {name:<22} {eJ:>8.4f} {dJ:>7.4f} {adj:>8.4f} {sc:>8.4f} {sc-b:>+8.4f}", flush=True)

print("\n  === leave-one-embryo-out ===", flush=True)
for emb in ("44b6", "6bba"):
    print(f"    held out {emb}:")
    b2 = None
    for name, df in store.items():
        sc, eJ, dJ, adj = agg(df.filter(pl.col("emb") == emb))
        if b2 is None: b2 = sc
        print(f"      {name:<22} edgeJ {eJ:>7.4f}  adj {adj:>7.4f}  score {sc:>7.4f}  {sc-b2:>+7.4f}")
