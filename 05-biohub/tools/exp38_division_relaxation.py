#!/usr/bin/env python3
"""EXP-38 -- we are running division detection 6x more conservatively than the metric rewards.

The decomposition of their 0.947 (work/repro_out/validator_results.csv, 64 rows):

    edge_jaccard 0.9116   adj 0.9114   div_jaccard 0.3125
    div_tp 0.375  div_fn 1.125  div_fp 0.141   -> recall 25.0%, PRECISION 72.7%

Jaccard is not symmetric in TP and FP. At that operating point (TP .375, denom 1.641) one true
division moves divJ by +0.609 and one false by -0.086 -- a ratio of 7.1, so break-even precision
is 12.4%. We are at 72.7%. Every proposal with better than a 1-in-8 chance of being real is +EV
and we are refusing them.

Their caps are not what binds: global_frac_cap 0.00375 * ~25,000 edges = ~94 allowed, and only
~14 are added. The GATES bind. So relax the gates and measure.

Why we can see this and their sweep cannot: their in-kernel proxy holds 8 stems carrying 12 GT
divisions. Twelve. It cannot resolve a division change at all. We hold 151 across 199 datasets.

Scored the way the real scorer does -- an edge or a division is only CHARGEABLE if an endpoint
matched a GT node, because ground truth annotates 2.82% of cells (the sparse-annotation trap,
which has now bitten four times).
"""
import sys, itertools
from pathlib import Path
import numpy as np, tracksdata as td
from scipy.spatial import cKDTree

sys.path.insert(0, "src"); sys.path.insert(0, "tools")
sys.path.insert(0, "reference/royerlab-baseline/src")
from geff import GeffMetadata
import div_sweep as D
from div_sweep import DivCfg, add_safe_divisions

GT = Path("data/train_geff"); GRAPHS = Path("work/train_graphs"); TOL = 7.0
N_PER_EMBRYO = 40

cache = []
files = sorted(GRAPHS.rglob("*.geff"))
sel = [p for p in files if p.stem.startswith("44b6")][:N_PER_EMBRYO] + \
      [p for p in files if p.stem.startswith("6bba")][:N_PER_EMBRYO]
for p in sel:
    gp = GT / f"{p.stem}.geff"
    if not gp.exists(): continue
    g = td.graph.IndexedRXGraph.from_geff(p); g = g[0] if isinstance(g, tuple) else g
    nodes_by_id, pos, tt = {}, {}, {}
    for r in g.node_attrs().iter_rows(named=True):
        i = int(r["node_id"])
        nodes_by_id[i] = {"t": int(r["t"]), "z": r["z"], "y": r["y"], "x": r["x"]}
        pos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM; tt[i] = int(r["t"])
    edges = [{"source_id": int(r["source_id"]), "target_id": int(r["target_id"])}
             for r in g.edge_attrs().iter_rows(named=True)]

    gg = td.graph.IndexedRXGraph.from_geff(gp); gg = gg[0] if isinstance(gg, tuple) else gg
    gpos, gtt = {}, {}
    for r in gg.node_attrs().iter_rows(named=True):
        i = int(r["node_id"]); gpos[i] = np.array([r["z"], r["y"], r["x"]]) * D.VOXEL_SCALE_UM
        gtt[i] = int(r["t"])
    ge = [(int(r["source_id"]), int(r["target_id"])) for r in gg.edge_attrs().iter_rows(named=True)]
    est = (GeffMetadata.read(gp).extra or {}).get("estimated_number_of_nodes")
    if not ge or not est: continue

    by_t = {}
    for i, t in tt.items(): by_t.setdefault(t, []).append(i)
    trees = {t: (cKDTree(np.stack([pos[i] for i in ids])), ids) for t, ids in by_t.items()}
    def match(pt, t):
        ent = trees.get(t)
        if ent is None: return None
        tr_, ids = ent; d, j = tr_.query(pt)
        return ids[int(j)] if d <= TOL else None

    gt_out = {}
    for s, d in ge: gt_out.setdefault(s, []).append(d)
    gt_pairs, gt_nodes, gt_divs = set(), set(), set()
    for s, d in ge:
        if s not in gpos or d not in gpos: continue
        ms, md = match(gpos[s], gtt[s]), match(gpos[d], gtt[d])
        gt_nodes.update(x for x in (ms, md) if x is not None)
        if ms is not None and md is not None: gt_pairs.add((ms, md))
    gt_succ = set()
    for s, kids in gt_out.items():
        if s not in gpos: continue
        ms = match(gpos[s], gtt[s])
        if ms is None: continue
        # a predicted fork is only EVALUABLE (chargeable as FP) when the GT node it matched
        # itself has a successor -- see division_metrics._pred_division_fork_sets
        if len(kids) >= 1: gt_succ.add(ms)
        if len(kids) >= 2: gt_divs.add(ms)
    cache.append((nodes_by_id, pos, edges, gt_pairs, gt_nodes, gt_succ, gt_divs, float(est), p.stem[:4]))

print(f"  {len(cache)} datasets   GT divisions (matched) {sum(len(c[6]) for c in cache)}"
      f"   evaluable GT nodes {sum(len(c[5]) for c in cache):,}\n")
print("  NB divJ here is an approximation: TP requires only that the fork sits on a GT dividing")
print("  node, where the real scorer also checks parent/child/grandchild structure, and we do not")
print("  model cross-component or malformed forks. Validate on the baseline against 0.3125.\n")

def score(cache, cfg, emb_filter=None):
    TP = FP = FN = 0; dTP = dFP = dFN = 0; np_tot = nt_tot = 0; added = 0
    for nodes_by_id, pos, edges, gtp, gtn, gtsucc, gtd, est, emb in cache:
        if emb_filter and emb != emb_filter: continue
        e = edges
        if cfg is not None:
            e = add_safe_divisions(nodes_by_id, edges, cfg)   # returns [*edges, *added]
            added += len(e) - len(edges)
        s = {(int(x["source_id"]), int(x["target_id"])) for x in e}
        tp = len(gtp & s)
        chargeable = sum(1 for a, b in s if a in gtn or b in gtn)
        TP += tp; FP += chargeable - tp; FN += len(gtp) - tp
        outdeg = {}
        for a, b in s: outdeg[a] = outdeg.get(a, 0) + 1
        pred_div = {a for a, k in outdeg.items() if k >= 2}
        charge_div = pred_div & gtsucc
        dTP += len(charge_div & gtd); dFP += len(charge_div - gtd); dFN += len(gtd - charge_div)
        np_tot += len(nodes_by_id); nt_tot += est
    eJ = TP / max(TP + FP + FN, 1)
    dJ = dTP / max(dTP + dFP + dFN, 1)
    mult = max(0.0, 1.1 - 0.1 * np_tot / nt_tot)
    return eJ * mult + 0.1 * dJ, eJ, dJ, mult, dTP, dFP, dFN, added

CFGS = [("baseline (no extra)", None)]
base = DivCfg()
for name, kw in [
    ("theirs    (9,14,.6,2.25)", {}),
    ("max_um 11",                {"max_um": 11.0}),
    ("max_um 13",                {"max_um": 13.0}),
    ("sister 18",                {"sister_max_um": 18.0}),
    ("tau .40",                  {"symmetry_tau": 0.40}),
    ("tau .25",                  {"symmetry_tau": 0.25}),
    ("diverge 1.5",              {"diverge_um": 1.5}),
    ("diverge 0.0",              {"diverge_um": 0.0, "require_divergence": False}),
    ("no mutual_nn",             {"require_mutual_nn": False}),
    ("caps x4",                  {"frame_frac_cap": 0.0304, "global_frac_cap": 0.015}),
    ("RELAX ALL",                {"max_um": 13.0, "sister_max_um": 18.0, "symmetry_tau": 0.25,
                                  "diverge_um": 0.0, "require_divergence": False,
                                  "require_mutual_nn": False,
                                  "frame_frac_cap": 0.0304, "global_frac_cap": 0.015}),
    ("RELAX gates, their caps",  {"max_um": 13.0, "sister_max_um": 18.0, "symmetry_tau": 0.25,
                                  "diverge_um": 0.0, "require_divergence": False,
                                  "require_mutual_nn": False}),
]:
    CFGS.append((name, DivCfg(**{**base.__dict__, **kw})))

hdr = f"  {'config':<26} {'added':>7} {'divTP':>6} {'divFP':>6} {'divJ':>7} {'edgeJ':>8} {'SCORE':>8} {'delta':>8}"
print(hdr); print("  " + "-" * (len(hdr) - 2))
b = None
for name, cfg in CFGS:
    sc, eJ, dJ, mult, dtp, dfp, dfn, added = score(cache, cfg)
    if b is None: b = sc
    print(f"  {name:<26} {added:>7} {dtp:>6} {dfp:>6} {dJ:>7.4f} {eJ:>8.4f} {sc:>8.4f} {sc-b:>+8.4f}")

print("\n  === leave-one-embryo-out ===")
for emb in ("44b6", "6bba"):
    print(f"    held out {emb}:")
    b2 = None
    for name, cfg in CFGS:
        sc, eJ, dJ, mult, dtp, dfp, dfn, added = score(cache, cfg, emb_filter=emb)
        if b2 is None: b2 = sc
        print(f"      {name:<26} divJ {dJ:>7.4f}  edgeJ {eJ:>7.4f}  score {sc:>7.4f}  {sc-b2:>+7.4f}")
