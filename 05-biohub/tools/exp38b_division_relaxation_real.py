#!/usr/bin/env python3
"""EXP-38b -- division gate relaxation, scored with the REAL scorer.

EXP-38's hand-rolled division metric failed its own validation: divJ 0.0172 where
work/repro_out/validator_results.csv reads 0.3125, a 100x discrepancy on div_fp. The scorer's
FP rule (division_metrics._pred_division_fork_sets) involves evaluable / cross-component /
malformed fork sets and a per-division bipartite pairing that is not worth reimplementing --
and reimplementing it is exactly how you get a number that agrees with your hypothesis.

So: call evaluate() and per_sample_metrics() directly, the same functions the competition uses.

The hypothesis under test is unchanged. Their operating point is

    div_tp 0.375  div_fn 1.125  div_fp 0.141  ->  recall 25.0%, PRECISION 72.7%

and at that point Jaccard moves +0.609 per true division and -0.086 per false one, so break-even
precision is 12.4%. If that is right we are ~6x too conservative and relaxing the gates should
buy divJ faster than it loses edgeJ. If it is wrong, this says so.
"""
import sys, warnings
from pathlib import Path
import numpy as np, polars as pl, tracksdata as td

sys.path.insert(0, "src"); sys.path.insert(0, "tools")
sys.path.insert(0, "reference/royerlab-baseline/src")
warnings.filterwarnings("ignore")
from geff import GeffMetadata
from tracking_cellmot.metrics import evaluate, per_sample_metrics, node_recall
import div_sweep as D
from div_sweep import DivCfg, add_safe_divisions
from biohub.contracts import SCALE

GT = Path("data/train_geff"); GRAPHS = Path("work/train_graphs")
N_PER_EMBRYO = int(sys.argv[1]) if len(sys.argv) > 1 else 20

def build(nodes_by_id, edges):
    G = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        G.add_node_attr_key(k, pl.Float64, -999999.0)
    keys = list(nodes_by_id)
    ids = G.bulk_add_nodes([{"t": int(nodes_by_id[k]["t"]), "z": float(nodes_by_id[k]["z"]),
                             "y": float(nodes_by_id[k]["y"]), "x": float(nodes_by_id[k]["x"])}
                            for k in keys])
    m = dict(zip(keys, ids))
    ee = [{"source_id": m[int(e["source_id"])], "target_id": m[int(e["target_id"])]}
          for e in edges if int(e["source_id"]) in m and int(e["target_id"]) in m]
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
    nodes_by_id = {int(r["node_id"]): {"t": int(r["t"]), "z": r["z"], "y": r["y"], "x": r["x"]}
                   for r in g.node_attrs().iter_rows(named=True)}
    edges = [{"source_id": int(r["source_id"]), "target_id": int(r["target_id"])}
             for r in g.edge_attrs().iter_rows(named=True)]
    gt = td.graph.IndexedRXGraph.from_geff(gp); gt = gt[0] if isinstance(gt, tuple) else gt
    est = (GeffMetadata.read(gp).extra or {}).get("estimated_number_of_nodes")
    if est is None or len(gt.edge_attrs()) == 0: continue
    cache.append((nodes_by_id, edges, gt, float(est), p.stem[:4]))
print(f"  {len(cache)} datasets cached\n", flush=True)

base = DivCfg()
CFGS = [("baseline (as-is)", None), ("theirs (9,14,.6,2.25)", {})]
for name, kw in [
    ("max_um 13",       {"max_um": 13.0}),
    ("tau .25",         {"symmetry_tau": 0.25}),
    ("no divergence",   {"diverge_um": 0.0, "require_divergence": False}),
    ("no mutual_nn",    {"require_mutual_nn": False}),
    ("caps x4",         {"frame_frac_cap": 0.0304, "global_frac_cap": 0.015}),
    ("RELAX ALL",       {"max_um": 13.0, "sister_max_um": 18.0, "symmetry_tau": 0.25,
                         "diverge_um": 0.0, "require_divergence": False,
                         "require_mutual_nn": False,
                         "frame_frac_cap": 0.0304, "global_frac_cap": 0.015}),
]:
    CFGS.append((name, kw))

def run(cfgkw):
    rows = []
    for nodes_by_id, edges, gt, est, emb in cache:
        e = edges if cfgkw is None else add_safe_divisions(
            nodes_by_id, edges, DivCfg(**{**base.__dict__, **cfgkw}))
        pred = build(nodes_by_id, e)
        er = evaluate(pred, gt, scale=tuple(SCALE), max_distance=7.0)
        nr = node_recall(pred, gt) if len(e) else 0.0
        m = per_sample_metrics(er, est, nr); m["emb"] = emb
        rows.append(m)
    return pl.DataFrame(rows)

def agg(df):
    et = df["edge_tp"].sum(); ef = df["edge_fp"].sum(); en = df["edge_fn"].sum()
    dt = df["division_tp"].sum(); dfp = df["division_fp"].sum(); dfn = df["division_fn"].sum()
    eJ = et / max(et + ef + en, 1); dJ = dt / max(dt + dfp + dfn, 1)
    adj = df["adj_edge_jaccard"].mean()
    return adj + 0.1 * dJ, eJ, dJ, adj, dt, dfp, dfn

hdr = f"  {'config':<24} {'divTP':>6} {'divFP':>6} {'divFN':>6} {'divJ':>7} {'edgeJ':>8} {'adj':>8} {'SCORE':>8} {'delta':>8}"
print(hdr); print("  " + "-" * (len(hdr) - 2), flush=True)
store, b = {}, None
for name, kw in CFGS:
    df = run(kw); store[name] = df
    sc, eJ, dJ, adj, dt, dfp, dfn = agg(df)
    if b is None: b = sc
    print(f"  {name:<24} {dt:>6} {dfp:>6} {dfn:>6} {dJ:>7.4f} {eJ:>8.4f} {adj:>8.4f} {sc:>8.4f} {sc-b:>+8.4f}", flush=True)

print("\n  === leave-one-embryo-out ===", flush=True)
for emb in ("44b6", "6bba"):
    print(f"    held out {emb}:")
    b2 = None
    for name, df in store.items():
        sc, eJ, dJ, adj, dt, dfp, dfn = agg(df.filter(pl.col("emb") == emb))
        if b2 is None: b2 = sc
        print(f"      {name:<24} divJ {dJ:>7.4f}  edgeJ {eJ:>7.4f}  score {sc:>7.4f}  {sc-b2:>+7.4f}")
