"""Agent 5 / Subtrack A — DETECTION DIVERSITY PREFLIGHT.

The portfolio problem: every candidate we own shares ONE detector and ONE transformer
(per-crop error correlation with v122: C_survival 0.9995, Bp_ilp 0.987, E0c 0.959).  Before
spending GPU on a genuinely different detector we must answer three questions with
measurements, not opinion:

  Q1  How much of the edge FN is DETECTION-attributable at all?
      (a GT edge whose endpoint was never detected cannot be recovered by any associator)
  Q2  Is the diversity we already have real?  Measure node-level disagreement between the
      cached TTA / threshold variants under the SCORER's own 7 um matching, and whether
      their UNION recovers annotated GT nodes that no single variant does.
  Q3  What must a new detector achieve to be worth deploying, in exact pooled terms?

Q3 is answered in closed form, not fitted.  From the exact identity proved by
`scripts/agent5_ledger.py --verify`,

    pooled = NUM/DEN + 0.1*DTP/(DTP+DFP+DFN),
    NUM = SUM_i tp_i*(1-0.1*r_i),   DEN = SUM_i (tp_i+fp_i+fn_i),   tp_i+fn_i = gt_edges_i

a new node that yields one extra TRUE edge moves FN->TP: NUM += w_i, DEN unchanged.
A new node that yields one extra annotation-VISIBLE false edge: DEN += 1.  Hence

    dPooled ~ [ n_tp*w  -  n_fp_visible*adj ] / DEN            (adj = NUM/DEN)

so a new detector breaks even at

    n_fp_visible / n_tp  =  w/adj  ->  precision among visible new edges  =  adj/(w+adj)

which for the measured E0c anchor (adj ~ 0.665, w ~ 1) is ~40%.  Compare the DIVISION
layer, where +0.005 pooled needs only 10.15% precision among visible.  Detection actions
are ~4x less forgiving than division actions.  That asymmetry is the whole reason Subtrack
B is the bigger idea.

The node-count penalty is the SMALL term, not the big one: adding a node raises r_i and
costs only 0.1*tp_i/E_i in NUM (order 1e-3 of one edge).  What actually kills detector
widening is displacement -- extra nodes create extra valid FP edges inside a one-to-one
assignment (branch A measured -0.1596 / -0.1496).  Therefore a diverse detector must be
deployed as CONFIDENCE-ADAPTIVE CONSENSUS -- add a node only where the primary detector
has NO node within the matching radius and the alternate is confident -- never a global
average.  A public notebook's global harmonic detection fusion cost -0.025 and removed
3,181 nodes; that failure mode is a hard constraint here, not a warning.

Usage:
  .venv\\Scripts\\python.exe scripts\\agent5_detector_preflight.py
  .venv\\Scripts\\python.exe scripts\\agent5_detector_preflight.py --ledger <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DET = ROOT / "artifacts/kaggle/coupled_cache/det"
SCALE = (1.625, 0.40625, 0.40625)
MATCH_UM = 7.0
OUT = Path(r"C:\Users\aryaa\Documents\Biohub-CellTracking-2026"
           r"_RESEARCH\agent_runs\diversity")


def load_det(path: Path):
    z = np.load(path)
    c = z["coords"].astype(np.float64)          # (N, 4) = t, z, y, x in voxels
    t = c[:, 0].astype(np.int64)
    xyz = c[:, 1:4] * np.asarray(SCALE)
    return t, xyz


def match_rate(tA, pA, tB, pB, radius=MATCH_UM):
    """Fraction of A nodes with a B node within `radius` um in the SAME frame, and the
    symmetric rate. One-to-one is not enforced (this is agreement, not the scorer)."""
    from scipy.spatial import cKDTree
    byB = defaultdict(list)
    for i, f in enumerate(tB):
        byB[int(f)].append(i)
    trees = {f: cKDTree(pB[np.asarray(ii)]) for f, ii in byB.items()}
    hit = 0
    for i, f in enumerate(tA):
        tr = trees.get(int(f))
        if tr is not None and len(tr.query_ball_point(pA[i], radius)) > 0:
            hit += 1
    return hit / max(len(tA), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path,
                    default=OUT / "ledger")
    ap.add_argument("--radius", type=float, default=MATCH_UM)
    a = ap.parse_args()

    # ---------------- Q2: is the diversity we already have real? ----------------------
    groups = defaultdict(dict)
    for p in sorted(DET.glob("*.npz")):
        crop, var = p.name.split("__", 1)
        groups[crop][var[:-4]] = p
    multi = {c: v for c, v in groups.items() if len(v) > 1}
    print(f"=== Q2  detector-variant disagreement (scorer radius {a.radius} um) ===")
    print(f"crops carrying >1 cached variant: {sorted(multi)}")
    q2 = []
    for crop, vs in sorted(multi.items()):
        names = sorted(vs)
        loaded = {n: load_det(vs[n]) for n in names}
        print(f"\n  {crop}   node counts: "
              + "  ".join(f"{n}={len(loaded[n][0]):,}" for n in names))
        ref = "tta-4view__det-0.969"      # the variant used for all 199 crops
        if ref not in loaded:
            ref = names[0]
        tR, pR = loaded[ref]
        for n in names:
            if n == ref:
                continue
            tX, pX = loaded[n]
            f_x_in_r = match_rate(tX, pX, tR, pR, a.radius)
            f_r_in_x = match_rate(tR, pR, tX, pX, a.radius)
            uniq_x = int(round((1 - f_x_in_r) * len(tX)))
            uniq_r = int(round((1 - f_r_in_x) * len(tR)))
            jac = (len(tX) - uniq_x) / max(len(tX) + uniq_r, 1)
            print(f"    {n:<26} vs {ref}:  {f_x_in_r:6.2%} of its nodes are already in the "
                  f"reference; {uniq_x:,} genuinely new, {uniq_r:,} reference-only, "
                  f"node Jaccard ~{jac:.4f}")
            q2.append({"crop": crop, "variant": n, "ref": ref,
                       "n_variant": len(tX), "n_ref": len(tR),
                       "frac_variant_in_ref": f_x_in_r, "frac_ref_in_variant": f_r_in_x,
                       "unique_to_variant": uniq_x, "unique_to_ref": uniq_r,
                       "node_jaccard": jac})

    # ---------------- Q2b: does that disagreement recover ANNOTATED GT nodes? ----------
    # The gate. Node-set disagreement is worthless unless the disagreeing nodes land on GT
    # the reference missed. Greedy nearest-first assignment inside the scorer radius,
    # one GT node consumed at most once (an optimistic but one-to-one recall).
    print(f"\n=== Q2b  does the disagreement recover ANNOTATED GT nodes? ===")
    from scipy.spatial import cKDTree
    from biotrack.metric import load_graph
    import tracksdata as td
    q2b = []
    for crop, vs in sorted(multi.items()):
        gt_geff = ROOT / "data" / "train" / f"{crop}.geff"
        if not gt_geff.exists():
            print(f"  {crop}: no GT geff, skipped")
            continue
        gt = load_graph(str(gt_geff))
        ga = gt.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.T, "z", "y", "x"])
        gt_t = np.asarray(ga[td.DEFAULT_ATTR_KEYS.T].to_list(), dtype=np.int64)
        gt_p = np.stack([np.asarray(ga[k].to_list(), float) * s
                         for k, s in zip(("z", "y", "x"), SCALE)], axis=1)

        def recall(t_, p_):
            byf = defaultdict(list)
            for i, f in enumerate(t_):
                byf[int(f)].append(i)
            trees = {f: cKDTree(p_[np.asarray(ii)]) for f, ii in byf.items()}
            hit = 0
            for i in range(len(gt_t)):
                tr = trees.get(int(gt_t[i]))
                if tr is not None and len(tr.query_ball_point(gt_p[i], a.radius)) > 0:
                    hit += 1
            return hit, hit / max(len(gt_t), 1)

        names = sorted(vs)
        loaded = {n: load_det(vs[n]) for n in names}
        ref = "tta-4view__det-0.969" if "tta-4view__det-0.969" in loaded else names[0]
        hR, rR = recall(*loaded[ref])
        print(f"  {crop}: GT nodes {len(gt_t):,}   reference {ref} recall "
              f"{hR:,} ({rR:.4%})")
        best_union = (hR, rR, ref)
        for n in names:
            if n == ref:
                continue
            tU = np.concatenate([loaded[ref][0], loaded[n][0]])
            pU = np.concatenate([loaded[ref][1], loaded[n][1]])
            hU, rU = recall(tU, pU)
            print(f"    union with {n:<26} recall {hU:,} ({rU:.4%})  "
                  f"delta {hU-hR:+,} GT node(s)  cost {len(loaded[n][0]):,} extra raw nodes")
            q2b.append({"crop": crop, "variant": n, "ref_recall": hR,
                        "union_recall": hU, "delta_gt_nodes": hU - hR,
                        "gt_nodes": int(len(gt_t))})
            if hU > best_union[0]:
                best_union = (hU, rU, n)
        print(f"    best union: {best_union[2]} -> {best_union[0]:,} "
              f"({best_union[1]:.4%}), i.e. {best_union[0]-hR:+,} over reference")

    # ---------------- Q1 + Q3: FN attribution and the exact break-even ----------------
    print(f"\n=== Q1  detection-attributable FN, and Q3 the exact pooled break-even ===")
    crops = [json.loads(p.read_text()) for p in sorted((a.ledger / "crops").glob("*/*.json"))]
    if not crops:
        print("  (no ledger yet -- run scripts/agent5_ledger.py first)")
        return

    # Q1b: is that "detection" FN the DETECTOR's fault, or the pipeline's? Compare the RAW
    # detector's GT-node recall against the recall of the E0c graph the scorer actually
    # sees. Any gap is created downstream (one-to-one matching, wrapper filters, linefit
    # smoothing) and CANNOT be recovered by a better or more diverse detector.
    print(f"\n  Q1b  raw detector recall vs the recall the scorer actually sees")
    q1b = []
    for c in crops:
        dp = DET / f"{c['crop']}__tta-4view__det-0.969.npz"
        gt_geff = ROOT / "data" / "train" / f"{c['crop']}.geff"
        if not dp.exists() or not gt_geff.exists():
            continue
        gt = load_graph(str(gt_geff))
        ga = gt.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.T, "z", "y", "x"])
        gt_t = np.asarray(ga[td.DEFAULT_ATTR_KEYS.T].to_list(), dtype=np.int64)
        gt_p = np.stack([np.asarray(ga[k].to_list(), float) * s
                         for k, s in zip(("z", "y", "x"), SCALE)], axis=1)
        t_, p_ = load_det(dp)
        byf = defaultdict(list)
        for i, f in enumerate(t_):
            byf[int(f)].append(i)
        trees = {f: cKDTree(p_[np.asarray(ii)]) for f, ii in byf.items()}
        hit = sum(1 for i in range(len(gt_t))
                  if trees.get(int(gt_t[i])) is not None
                  and len(trees[int(gt_t[i])].query_ball_point(gt_p[i], a.radius)) > 0)
        raw = hit / max(len(gt_t), 1)
        seen = c["base"]["node_recall"]
        print(f"    {c['crop']:<16} raw detector {raw:8.4%}   E0c graph {seen:8.4%}   "
              f"pipeline loses {raw-seen:+8.4%}  ({int(round((raw-seen)*len(gt_t))):+,} GT nodes)")
        q1b.append({"crop": c["crop"], "gt_nodes": int(len(gt_t)), "raw_detector_recall": raw,
                    "e0c_graph_node_recall": seen, "pipeline_recall_loss": raw - seen})
    if q1b:
        wr = sum(x["raw_detector_recall"] * x["gt_nodes"] for x in q1b)
        ws = sum(x["e0c_graph_node_recall"] * x["gt_nodes"] for x in q1b)
        wn = sum(x["gt_nodes"] for x in q1b)
        print(f"    {'POOLED':<16} raw detector {wr/wn:8.4%}   E0c graph {ws/wn:8.4%}   "
              f"pipeline loses {(wr-ws)/wn:+8.4%}  ({int(round(wr-ws)):+,} GT nodes)")
        print(f"    -> share of the detection-attributable FN that a BETTER DETECTOR could "
              f"even address: {1-(wr-ws)/max(wn-ws,1e-9):.2%}")
    gte = sum(c["decomp"]["gt_edges"] for c in crops)
    det = sum(c["decomp"]["fn_detection"] for c in crops)
    asc = sum(c["decomp"]["fn_association_base"] for c in crops)
    num = sum(c["supp"]["edge_tp"] * (1 - 0.1 * c["supp"]["total_node_ratio"]) for c in crops)
    den = sum(c["supp"]["edge_tp"] + c["supp"]["edge_fp"] + c["supp"]["edge_fn"]
              for c in crops)
    adj = num / den
    w = float(np.mean([c["w_node"] for c in crops]))
    breakeven_ratio = w / adj
    breakeven_prec = 1.0 / (1.0 + breakeven_ratio)
    print(f"  crops in ledger: {len(crops)}   GT edges {gte:,}   NUM {num:.1f}   DEN {den:,}"
          f"   adj {adj:.5f}   mean w_node {w:.5f}")
    print(f"  FN_detection   {det:,} ({det/max(gte,1):.3%} of GT edges) "
          f"-> ORACLE detector ceiling {det*w/den:+.5f} pooled")
    print(f"  FN_association {asc:,} ({asc/max(gte,1):.3%} of GT edges) "
          f"-> ORACLE associator ceiling {asc*w/den:+.5f} pooled")
    print(f"  detection share of all FN = {det/max(det+asc,1):.2%}")
    print(f"  value of ONE recovered true edge = {w/den:+.3e} pooled")
    print(f"  value of ONE visible false edge  = {-adj/den:+.3e} pooled")
    print(f"  BREAK-EVEN for any new detector: max {breakeven_ratio:.3f} visible-FP edges "
          f"per new TP edge  ->  precision among visible new edges >= {breakeven_prec:.2%}")
    print(f"  node-count side-term per added node = {0.1*num/ (den):.3e} / E_i "
          f"(negligible vs the edge terms -- displacement, not node ratio, is the risk)")

    res = {"radius_um": a.radius, "q2_variant_disagreement": q2, "q2b_gt_recovery": q2b,
           "q1b_raw_vs_pipeline_recall": q1b,
           "ledger_crops": [c["crop"] for c in crops],
           "gt_edges": gte, "fn_detection": det, "fn_association": asc,
           "NUM": num, "DEN": den, "adj": adj, "w_node_mean": w,
           "value_per_true_edge": w / den, "value_per_visible_false_edge": -adj / den,
           "oracle_detector_ceiling_pooled": det * w / den,
           "oracle_associator_ceiling_pooled": asc * w / den,
           "breakeven_fp_per_tp": breakeven_ratio,
           "breakeven_precision_among_visible": breakeven_prec}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "detector_preflight.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {OUT}\\detector_preflight.json")


if __name__ == "__main__":
    main()
