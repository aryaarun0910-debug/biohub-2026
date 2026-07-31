"""Agent 5 — exact per-candidate pooled-utility ledger + detection/association decomposition.

Two deliverables from ONE ground-truth matching pass per crop.

--------------------------------------------------------------------------------------
WHY THIS EXISTS
--------------------------------------------------------------------------------------
Subtrack B says: stop optimising AUC, estimate the EXPECTED POOLED SCORE DELTA of every
candidate action.  To do that you need the exact counter deltas of each action, not a
proxy.  This script produces them.

The enabling observation is an exact algebraic identity in the organiser's scorer
(`tracking_cellmot/metrics.py::summarise` + `per_sample_metrics`).  Per crop the
adjusted-edge weight is `w_i = tp_i + fp_i + fn_i`, and the adjusted Jaccard is
`adjJ_i = (tp_i / w_i) * (1 - 0.1 * r_i)` (before the max(0, .) clip), so

        w_i * adjJ_i  ==  tp_i * (1 - 0.1 * r_i)

and therefore the whole pooled edge term collapses to

        adj_edge_jaccard  =  SUM_i tp_i * (1 - 0.1 * r_i)  /  SUM_i (tp_i + fp_i + fn_i)

with `r_i = (N_pred_i - N_est_i) / N_est_i` INVARIANT under edges-only edits.  Also
`tp_i + fn_i == gt_num_edges_i` is a constant, so the denominator only moves through
`fp_i`.  The division term is a single global micro Jaccard.  Hence the pooled composite
of ANY edges-only edit set is a closed form in four running scalars:

        NUM = SUM_i tp_i * (1 - 0.1 * r_i)
        DEN = SUM_i (gt_edges_i + fp_i)
        (DTP, DFP, DFN)  global division counters
        pooled = NUM / DEN + 0.1 * DTP / (DTP + DFP + DFN)

That turns "cumulative pooled gain as candidates are added in utility order" from 14.4M
scorer invocations into vector arithmetic.  `--verify` proves the identity against the
authoritative `summarise()` on the crops actually processed; the run refuses to write a
ledger if it does not hold to 1e-12.

--------------------------------------------------------------------------------------
WHAT IS EMITTED
--------------------------------------------------------------------------------------
crops/<fold>/<crop>.json      per-crop anchors, both for the unmodified E0c graph and for
                              the suppress-all graph the reconstruction acts on:
                                edge_tp/fp/fn, num_pred_nodes, n_est, node_recall, r_i
                              plus the SUBTRACK-A detection/association decomposition:
                                gt_nodes, gt_nodes_matched
                                gt_edges, gt_edges_both_endpoints_matched
                                fn_detection   (GT edge with an unmatched endpoint —
                                                unreachable without a better detector)
                                fn_association (both endpoints detected, not linked —
                                                reachable by association alone)
cand/<fold>/<crop>.parquet    one row per frozen-H0c candidate (m, d1, d2) with the EXACT
                              counter deltas of taking that action in isolation on the
                              suppress-all graph:
                                e_add_tp, e_add_valid, e_rem_tp, e_rem_valid
                                d_tp = e_add_tp - e_rem_tp
                                d_fp = (e_add_valid - e_rem_valid) - d_tp
                                d_num = d_tp * (1 - 0.1 * r_i)     -> pooled numerator
                                d_den = d_fp                       -> pooled denominator
                                div_dtp, div_dfp, div_dfn

The candidate surface is IMMUTABLE: it is READ from the frozen H0c census
(`fork_candidates/h0c_top3`, cfg hash 04eeac97500d), never regenerated, never retuned.

NO-REFILTER, exactly like `phaseb_pooled_breakeven.py`: the wrapper filter is not re-run,
so pooled deltas here are a LOWER BOUND (H0d showed the live filter is strictly more
favourable).  Any promoted operating point must be confirmed by an exact replay with the
real filter (`phaseb_h0d_livefilter.py`).

--------------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------------
smoke (3 crops, with the identity proof):
  .venv\\Scripts\\python.exe scripts\\agent5_ledger.py --smoke --verify --workers 3

full corpus (199 crops) — REQUIRES HUMAN GO-AHEAD:
  .venv\\Scripts\\python.exe scripts\\agent5_ledger.py --workers 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import polars as pl  # noqa: E402

from phaseb_d0p_proposer import CACHE, cached_crops, load_e0c_tables  # noqa: E402
from phaseb_h0c_replay import CFG_HASH  # noqa: E402

CENSUS = CACHE / "fork_candidates" / "h0c_top3"
OUT = Path(r"C:\Users\aryaa\Documents\Biohub-CellTracking-2026_RESEARCH\agent_runs\diversity\ledger")

# 3 smoke crops, chosen to contain positives in BOTH folds so the smoke exercises the
# positive path; sizes deliberately small so the smoke stays inside the CPU policy.
SMOKE = [(0, "44b6_d754aa59"), (1, "6bba_afb141ff"), (1, "6bba_4f99ce20")]


# ---------------------------------------------------------------------------------
# closed-form pooled composite from running counters (see module docstring)
# ---------------------------------------------------------------------------------
def pooled_from_counters(num: float, den: float, dtp: int, dfp: int, dfn: int) -> float:
    ddiv = dtp + dfp + dfn
    edge = num / den if den > 0 else float("nan")
    return edge + 0.1 * (dtp / ddiv if ddiv > 0 else 0.0)


def suppress_all(edges):
    """GT-free deterministic lowest-id child retention (identical to the frozen cascade)."""
    es = {(int(a), int(b)) for a, b in edges}
    par: dict[int, set[int]] = {}
    ch: dict[int, set[int]] = {}
    for a, b in es:
        ch.setdefault(a, set()).add(b)
        par.setdefault(b, set()).add(a)
    for m in [n for n, k in ch.items() if len(k) >= 2]:
        keep = min(ch[m])
        for k in list(ch[m]):
            if k != keep:
                es.discard((m, k))
                ch[m].discard(k)
                par.get(k, set()).discard(m)
    return es, par, ch


def ledger_one(args) -> dict:
    split, crop, verify = args
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    _sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    from tracksdata.options import set_options
    set_options(show_progress=False)
    from biotrack.metric import (DEFAULT_SCALE, MAX_DISTANCE, estimated_nodes, load_graph,
                                 score_pred_graph)
    from biotrack.submission import submission_to_graphs
    from gt_collision import gt_maps_from_matches

    t0 = time.time()
    sub, t, pos, edges, nd = load_e0c_tables(split, crop)
    gt_geff = str(ROOT / "data" / "train" / f"{crop}.geff")
    node_rows = nd.select("row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id")

    def score(edge_set):
        er = pl.DataFrame([{"row_type": "edge", "node_id": -1, "t": -1, "z": -1.0, "y": -1.0,
                            "x": -1.0, "source_id": a, "target_id": b} for a, b in edge_set])
        o = pl.concat([node_rows, er]) if er.height else node_rows
        g = submission_to_graphs(o.with_columns(pl.lit(crop).alias("dataset"))
                                 .with_row_index("id"))[crop]
        return score_pred_graph(g, gt_geff)

    base_edges = {(int(a), int(b)) for a, b in edges}
    base_row = score(base_edges)
    supp_edges, supp_par, supp_ch = suppress_all(edges)
    supp_row = score(set(supp_edges))
    n_est = estimated_nodes(gt_geff)

    # ---- GT match (ONE pass; labels/validity only, generation already frozen) --------
    gt = load_graph(gt_geff)
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    internal = g.bulk_add_nodes([{"t": int(a), "z": float(z), "y": float(y), "x": float(x)}
                                 for a, z, y, x in zip(nd["t"], nd["z"], nd["y"], nd["x"])])
    s2i = {int(s): internal[i] for i, s in enumerate(nd["node_id"].to_list())}
    if edges:
        g.bulk_add_edges([{"source_id": s2i[a], "target_id": s2i[b]} for a, b in edges])
    g.match(gt, matching=DistanceMatching(max_distance=MAX_DISTANCE, scale=DEFAULT_SCALE))
    na = g.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID,
                                 td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    i2g = {int(r[td.DEFAULT_ATTR_KEYS.NODE_ID]): r[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
           for r in na.iter_rows(named=True)}
    gt_to_sub, sub_to_gt, _ = gt_maps_from_matches(s2i, i2g, context=f"agent5_ledger {crop}")

    gt_ids = [int(n) for n in gt.node_ids()]
    outdeg = dict(zip(gt_ids, gt.out_degree(gt_ids)))
    indeg = dict(zip(gt_ids, gt.in_degree(gt_ids)))
    gt_edges = [(int(a), int(b)) for a, b in gt.edge_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET]
    ).select(td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,
             td.DEFAULT_ATTR_KEYS.EDGE_TARGET).iter_rows()]

    # GT edges lifted into submission id space; the pair is a TP iff BOTH endpoints matched
    gte_sub = set()
    n_edges_both_matched = 0
    for u, v in gt_edges:
        su, sv = gt_to_sub.get(u), gt_to_sub.get(v)
        if su is not None and sv is not None:
            n_edges_both_matched += 1
            gte_sub.add((su, sv))

    # "valid predicted edge" per the scorer: source has GT out-degree>0 OR target in-degree>0
    out_valid = {s: outdeg.get(gv, 0) > 0 for s, gv in sub_to_gt.items()}
    in_valid = {s: indeg.get(gv, 0) > 0 for s, gv in sub_to_gt.items()}

    def is_valid(e):
        return out_valid.get(e[0], False) or in_valid.get(e[1], False)

    # ---- SUBTRACK A: detection vs association decomposition -------------------------
    linked_supp = len(gte_sub & set(supp_edges))
    linked_base = len(gte_sub & base_edges)
    decomp = {
        "gt_nodes": len(gt_ids),
        "gt_nodes_matched": len(gt_to_sub),
        "gt_edges": len(gt_edges),
        "gt_edges_both_endpoints_matched": n_edges_both_matched,
        "fn_detection": len(gt_edges) - n_edges_both_matched,
        "fn_association_base": n_edges_both_matched - linked_base,
        "fn_association_supp": n_edges_both_matched - linked_supp,
        "recomputed_edge_tp_base": linked_base,
        "recomputed_edge_tp_supp": linked_supp,
    }

    # ---- SUBTRACK B: exact per-candidate counter deltas on the suppressed graph ------
    # Fully vectorised. After suppress-all the graph has out-degree <= 1 and (E0c already)
    # in-degree <= 1, so "the mother's other child" and "the daughter's competing parent"
    # are each a single node -- a dense array lookup, not a set walk.
    cen = pl.read_parquet(CENSUS / str(split) / f"{crop}.parquet")
    r_i = (float(supp_row["num_pred_nodes"]) - n_est) / n_est if n_est > 0 else float("nan")
    wnode = 1.0 - 0.1 * r_i

    nn = len(sub)
    def ix(v):                                   # submission id -> dense index
        return np.searchsorted(sub, v)
    par_ix = np.full(nn, -1, dtype=np.int64)     # suppressed in-neighbour
    ch_ix = np.full(nn, -1, dtype=np.int64)      # suppressed out-neighbour
    if supp_edges:
        ea = np.fromiter((e[0] for e in supp_edges), dtype=np.int64, count=len(supp_edges))
        eb = np.fromiter((e[1] for e in supp_edges), dtype=np.int64, count=len(supp_edges))
        ea_i, eb_i = ix(ea), ix(eb)
        ch_ix[ea_i] = eb_i
        par_ix[eb_i] = ea_i
    ov = np.zeros(nn, dtype=bool)
    iv = np.zeros(nn, dtype=bool)
    if sub_to_gt:
        ss = np.fromiter(sub_to_gt.keys(), dtype=np.int64, count=len(sub_to_gt))
        ss_i = ix(ss)
        ov[ss_i] = [out_valid[int(s)] for s in ss]
        iv[ss_i] = [in_valid[int(s)] for s in ss]
    gt_keys = np.sort(np.fromiter(((ix(np.int64(u)) * nn + ix(np.int64(v)))
                                   for u, v in gte_sub), dtype=np.int64, count=len(gte_sub))) \
        if gte_sub else np.empty(0, dtype=np.int64)

    def is_gt(a_i, b_i):
        """Boolean -> int64. NEVER return numpy bool here: `bool_arr + bool_arr` is a
        logical OR in numpy, which silently caps two-true-edge candidates at 1."""
        k = a_i.astype(np.int64) * nn + b_i.astype(np.int64)
        if len(gt_keys) == 0:
            return np.zeros(len(k), dtype=np.int64)
        j = np.clip(np.searchsorted(gt_keys, k), 0, len(gt_keys) - 1)
        return (gt_keys[j] == k).astype(np.int64)

    M = ix(cen["mother"].to_numpy())
    A = ix(cen["d1"].to_numpy())
    B = ix(cen["d2"].to_numpy())
    cm = ch_ix[M]
    add_a = (cm != A).astype(np.int64)           # (m,d1) not already present
    add_b = (cm != B).astype(np.int64)
    e_add_tp = add_a * is_gt(M, A) + add_b * is_gt(M, B)
    e_add_valid = (add_a * (ov[M] | iv[A]).astype(np.int64)
                   + add_b * (ov[M] | iv[B]).astype(np.int64))

    # mother's retained child, displaced
    rm_m = ((cm >= 0) & (cm != A) & (cm != B)).astype(np.int64)
    pa, pb = par_ix[A], par_ix[B]
    rm_a = ((pa >= 0) & (pa != M)).astype(np.int64)   # parent stealing on d1
    rm_b = ((pb >= 0) & (pb != M)).astype(np.int64)   # parent stealing on d2
    cms, pas, pbs = np.maximum(cm, 0), np.maximum(pa, 0), np.maximum(pb, 0)
    e_rem_tp = (rm_m * is_gt(M, cms) + rm_a * is_gt(pas, A)
                + rm_b * is_gt(pbs, B))
    e_rem_valid = (rm_m * (ov[M] | iv[cms]).astype(np.int64)
                   + rm_a * (ov[pas] | iv[A]).astype(np.int64)
                   + rm_b * (ov[pbs] | iv[B]).astype(np.int64))

    d_tp = e_add_tp - e_rem_tp
    d_fp = (e_add_valid - e_rem_valid) - d_tp

    # ---- vectorised-vs-reference parity on a subsample (guards the bool-OR trap) -----
    vec_parity = None
    if verify and cen.height:
        rng = np.random.default_rng(20260731)
        pick = rng.permutation(cen.height)[:2000]
        bad = 0
        mo_a, d1_a, d2_a = (cen["mother"].to_numpy(), cen["d1"].to_numpy(),
                            cen["d2"].to_numpy())
        for i in pick:
            m, e1, e2 = int(mo_a[i]), int(d1_a[i]), int(d2_a[i])
            add = [(m, d) for d in (e1, e2) if (m, d) not in supp_edges]
            rem = [(m, c) for c in supp_ch.get(m, ()) if c not in (e1, e2)]
            rem += [(p, d) for d in (e1, e2) for p in supp_par.get(d, ()) if p != m]
            rem = [e for e in set(rem) if e in supp_edges]
            ref = (sum(e in gte_sub for e in add), sum(is_valid(e) for e in add),
                   sum(e in gte_sub for e in rem), sum(is_valid(e) for e in rem))
            got = (int(e_add_tp[i]), int(e_add_valid[i]), int(e_rem_tp[i]),
                   int(e_rem_valid[i]))
            bad += int(ref != got)
        vec_parity = {"checked": int(len(pick)), "mismatches": bad}
        if bad:
            raise AssertionError(f"{crop}: vectorised counter deltas disagree with the "
                                 f"reference loop on {bad}/{len(pick)} candidates")
    is_pos = (cen["label"] == "positive").to_numpy()
    vis = cen["metric_visible"].to_numpy()
    cdf = cen.select("cand_id", "crop", "fold", "family", "mother", "d1", "d2", "rank",
                     "label", "metric_visible").with_columns([
        pl.Series("e_add_tp", e_add_tp), pl.Series("e_add_valid", e_add_valid),
        pl.Series("e_rem_tp", e_rem_tp), pl.Series("e_rem_valid", e_rem_valid),
        pl.Series("d_tp", d_tp), pl.Series("d_fp", d_fp),
        pl.Series("d_num", d_tp.astype(float) * wnode),
        pl.Series("d_den", d_fp.astype(float)),
        pl.Series("div_dtp", is_pos.astype(np.int64)),
        pl.Series("div_dfp", (vis & ~is_pos).astype(np.int64)),
        pl.Series("div_dfn", -is_pos.astype(np.int64)),
    ])
    (OUT / "cand" / str(split)).mkdir(parents=True, exist_ok=True)
    cdf.write_parquet(OUT / "cand" / str(split) / f"{crop}.parquet")

    rec = {
        "crop": crop, "fold": split, "family": crop.split("_")[0], "cfg_hash": CFG_HASH,
        "n_est": n_est, "gt_num_edges": len(gt_edges),
        "base": {k: base_row[k] for k in ("edge_tp", "edge_fp", "edge_fn", "division_tp",
                                          "division_fp", "division_fn", "num_pred_nodes",
                                          "node_recall", "total_node_ratio",
                                          "adj_edge_jaccard")},
        "supp": {k: supp_row[k] for k in ("edge_tp", "edge_fp", "edge_fn", "division_tp",
                                          "division_fp", "division_fn", "num_pred_nodes",
                                          "node_recall", "total_node_ratio",
                                          "adj_edge_jaccard")},
        "r_supp": r_i, "w_node": wnode, "vec_parity": vec_parity,
        "decomp": decomp, "n_cand": cdf.height,
        "n_positive": int((cen["label"] == "positive").sum()),
        "n_visible": int(cen["metric_visible"].sum()),
        "runtime_s": time.time() - t0,
    }

    # ---- identity proof: closed form must equal the authoritative per-sample row -----
    if verify:
        for arm, row in (("base", base_row), ("supp", supp_row)):
            w = row["edge_tp"] + row["edge_fp"] + row["edge_fn"]
            lhs = w * row["adj_edge_jaccard"]
            rhs = row["edge_tp"] * (1 - 0.1 * row["total_node_ratio"])
            rec.setdefault("identity", {})[arm] = {
                "w_times_adjJ": lhs, "tp_times_wnode": rhs, "abs_err": abs(lhs - rhs),
                "tp_plus_fn_eq_gt_edges": (row["edge_tp"] + row["edge_fn"]) == len(gt_edges),
                "recomputed_tp_matches": (linked_base if arm == "base" else linked_supp)
                                         == row["edge_tp"],
            }
    (OUT / "crops" / str(split)).mkdir(parents=True, exist_ok=True)
    (OUT / "crops" / str(split) / f"{crop}.json").write_text(json.dumps(rec, indent=2,
                                                                       default=float))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="only the 3 frozen smoke crops")
    ap.add_argument("--verify", action="store_true", help="prove the closed-form identity")
    a = ap.parse_args()

    tasks = ([(s, c, a.verify) for s, c in SMOKE] if a.smoke
             else [(s, c, a.verify) for s in (0, 1) for c in cached_crops(s)])
    print(f"agent5 ledger: {len(tasks)} crop(s), cfg {CFG_HASH}, "
          f"{'SMOKE' if a.smoke else 'FULL CORPUS'}, workers={a.workers}")
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    res = []
    if a.workers <= 1:
        for tk in tasks:
            res.append(ledger_one(tk))
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(ledger_one, tasks):
                res.append(r)

    print(f"\n{'crop':<16}{'cand':>9}{'pos':>5}{'vis':>7}{'gtE':>7}{'FNdet':>7}"
          f"{'FNassoc':>8}{'nodeRec':>9}{'r_supp':>9}{'sec':>7}")
    for r in sorted(res, key=lambda z: (z["fold"], z["crop"])):
        d = r["decomp"]
        print(f"{r['crop']:<16}{r['n_cand']:>9,}{r['n_positive']:>5}{r['n_visible']:>7,}"
              f"{d['gt_edges']:>7,}{d['fn_detection']:>7,}{d['fn_association_base']:>8,}"
              f"{r['base']['node_recall']:>9.4f}{r['r_supp']:>9.4f}{r['runtime_s']:>7.1f}")

    tot_e = sum(r["decomp"]["gt_edges"] for r in res)
    tot_det = sum(r["decomp"]["fn_detection"] for r in res)
    tot_as = sum(r["decomp"]["fn_association_base"] for r in res)
    tot_tp = sum(r["base"]["edge_tp"] for r in res)
    print(f"\nSUBTRACK A -- FN attribution over these crops:")
    print(f"  GT edges {tot_e:,}   edge TP {tot_tp:,}   "
          f"FN_detection {tot_det:,} ({tot_det/max(tot_e,1):.3%} of GT edges)   "
          f"FN_association {tot_as:,} ({tot_as/max(tot_e,1):.3%})")
    print(f"  detection share of all FN = {tot_det/max(tot_det+tot_as,1):.3%}"
          f"   <-- ceiling on what ANY better detector can add")

    if a.verify:
        errs = [v["abs_err"] for r in res for v in r.get("identity", {}).values()]
        ok_gt = all(v["tp_plus_fn_eq_gt_edges"] for r in res
                    for v in r.get("identity", {}).values())
        ok_tp = all(v["recomputed_tp_matches"] for r in res
                    for v in r.get("identity", {}).values())
        vp = [r["vec_parity"] for r in res if r.get("vec_parity")]
        print(f"\nIDENTITY PROOF  max|w*adjJ - tp*(1-0.1r)| = {max(errs):.3e}   "
              f"tp+fn==gt_edges {ok_gt}   recomputed TP == scorer TP {ok_tp}")
        print(f"VECTOR PARITY   {sum(v['checked'] for v in vp):,} candidates re-derived by "
              f"the reference loop, {sum(v['mismatches'] for v in vp)} mismatches")
        if max(errs) > 1e-9 or not ok_gt or not ok_tp:
            raise SystemExit("IDENTITY FAILED -- closed-form pooled evaluator is not valid here")

    (OUT / "summary.json").write_text(json.dumps(
        {"cfg_hash": CFG_HASH, "smoke": a.smoke, "n_crops": len(res),
         "wall_s": time.time() - t0, "crops": res}, indent=2, default=float))
    print(f"\nwall {time.time()-t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
