"""D0P — deployment-realistic GT-free fork-proposer audit.

D0' showed that correctly rebuilding divisions is worth +0.0783 / +0.0737, but every arm
selected forks with ground truth. D0P asks the prerequisite question: can a proposer that
never consults GT generate those forks at all, and at what candidate cost?

CANDIDATE GENERATION NEVER CONSULTS GT.
  - every E0c node at frame t may be a mother;
  - daughters are distinct E0c nodes at t+1;
  - daughters that already have a parent stay eligible (parent stealing is required);
  - both incumbent-child+alternative and two-alternative pairs are included;
  - daughter order is canonicalised and triplets deduplicated.

Frozen surfaces (never re-tuned per family, never tuned after seeing results):
  native          pairs formed from the full pre-assignment E0c candidate-edge cache,
                  grouped by mother.
  geometric_core  parent->daughter <= 10.5 um, sister separation <= 8.5 um.
  outer_diag      fixed 15.0 / 15.0 um diagnostic, to show whether the core cap rather
                  than the approach is limiting recall.

GT is used ONLY after generation, for auditing and labels. Mother categories:
  true_divider       matched to a GT node with out-degree >= 2
  reliable_negative  matched to an annotated GT node that continues (out-degree == 1)
  annotated_end      matched to a GT node with out-degree == 0 (annotation ends; not a
                     reliable negative)
  unlabeled          unmatched -> excluded from supervised loss downstream

Usage:
  .venv\\Scripts\\python.exe scripts\\win_bet\\phaseb_d0p_proposer.py --workers 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl  # noqa: E402

CACHE = ROOT / "artifacts/kaggle/e0c_cache"
OUT_JSON = ROOT / "reports/inventory/phaseb_d0p_proposer.json"
TABLE_DIR = CACHE / "fork_candidates"  # canonical bridge table (D0R persists here)
E0C = {0: 0.7595, 1: 0.6490}
SCALE = (1.625, 0.40625, 0.40625)

SURFACES = {
    "native": {"parent_um": None, "sister_um": None},        # from cached candidate edges
    "geometric_core": {"parent_um": 10.5, "sister_um": 8.5},
    "outer_diag": {"parent_um": 15.0, "sister_um": 15.0},
    # --- division_flow_pair: frozen 2026-07-30 BEFORE execution, no cap sweep -------------
    # Ordinary migration constrains the daughters' CENTRE OF MASS, not either daughter
    # independently. A raw mother->daughter gate conflates tissue motion with daughter
    # separation; this surface allows large individual displacement while keeping the pair
    # biologically coherent. Every constant is inherited, none selected from D0P outcomes:
    #   parent_um  15.0 : already-frozen outer raw cap (outer_diag)
    #   sister_um   8.5 : already-frozen core cap (geometric_core)
    #   midpoint_um 6.0 : E0c's existing tight motion gate MOTION_RELINK_TIGHT_UM
    # knn_k / knn_min are conventional defaults for a robust local median, frozen here and
    # included in the surface hash; they are not tuned.
    "division_flow_pair": {"parent_um": 15.0, "sister_um": 8.5, "midpoint_um": 6.0,
                           "flow": "knn", "knn_k": 16, "knn_min": 4},
    "division_flow_pair_framemedian": {"parent_um": 15.0, "sister_um": 8.5, "midpoint_um": 6.0,
                                       "flow": "frame_median", "knn_k": 0, "knn_min": 0},
    # --- H0b flow-gated wide-sister: frozen 2026-07-30 BEFORE execution -------------------
    # Preregistration amendment. H0 falsified the 8.5 um SISTER prior, not deployable
    # reconstruction: parent 10.5->15.0 and knn->frame_median were exact nulls, while
    # sister 8.5->15.0 recovered 5/20 -> 20/20 and 20/93 -> 82/93. This surface keeps every
    # other constant identical to division_flow_pair and relaxes only the falsified prior.
    #   parent_um  15.0 : unchanged from division_flow_pair
    #   sister_um  15.0 : already-frozen outer_diag cap (the only change)
    #   midpoint_um 6.0 : unchanged E0c MOTION_RELINK_TIGHT_UM
    #   flow/knn        : identical estimator, identical constants
    "division_flow_pair_wide": {"parent_um": 15.0, "sister_um": 15.0, "midpoint_um": 6.0,
                                "flow": "knn", "knn_k": 16, "knn_min": 4},
}


def surface_hash(name: str) -> str:
    return hashlib.sha256(json.dumps({name: SURFACES[name]}, sort_keys=True).encode()).hexdigest()[:12]


def cached_crops(split: int):
    return sorted(json.loads(p.read_text())["crop"]
                  for p in (CACHE / "status").glob(f"{split}__*.json")
                  if json.loads(p.read_text()).get("status") == "ok")


def load_e0c_tables(split: int, crop: str):
    df = pl.read_parquet(CACHE / "graphs" / str(split) / f"{crop}.parquet")
    nd = df.filter(pl.col("row_type") == "node").sort("node_id")
    sub = np.asarray(nd["node_id"].to_list(), dtype=np.int64)
    t = np.asarray(nd["t"].to_list(), dtype=np.int64)
    pos = np.stack([np.asarray(nd["z"].to_list(), float) * SCALE[0],
                    np.asarray(nd["y"].to_list(), float) * SCALE[1],
                    np.asarray(nd["x"].to_list(), float) * SCALE[2]], axis=1)
    edges = [(int(r["source_id"]), int(r["target_id"]))
             for r in df.filter(pl.col("row_type") == "edge").iter_rows(named=True)]
    return sub, t, pos, edges, nd


def native_pairs_by_mother(split: int, crop: str, nd: pl.DataFrame) -> dict[int, set[int]]:
    """Mother -> allowed daughters, taken from the pre-assignment candidate-edge cache.

    The cache is keyed by RAW geff ids while the E0c graph is renumbered, so endpoints are
    mapped through quantised (t, z, y, x) coordinates.
    """
    from scipy.spatial import cKDTree
    cand_p = CACHE / "candidates" / str(split) / f"{crop}.parquet"
    if not cand_p.exists():
        return {}, 0.0
    c = pl.read_parquet(cand_p)
    if c.height == 0 or "sz" not in c.columns:
        return {}, 0.0

    # E0c applies OUTPUT_LINEFIT_SMOOTH (weight 0.8), so cached graph coordinates are
    # displaced from the raw candidate coordinates. Map by nearest neighbour within a
    # tolerance, per frame, and report the fidelity so the surface stays auditable.
    TOL_UM = 2.0
    by_t: dict[int, tuple] = {}
    for nid, a, z, y, x in zip(nd["node_id"].to_list(), nd["t"].to_list(),
                               nd["z"].to_list(), nd["y"].to_list(), nd["x"].to_list()):
        by_t.setdefault(int(a), ([], []))[0].append(int(nid))
        by_t[int(a)][1].append([float(z) * SCALE[0], float(y) * SCALE[1], float(x) * SCALE[2]])
    trees = {a: (np.asarray(ids), cKDTree(np.asarray(ps))) for a, (ids, ps) in by_t.items()}

    def to_sub(a: int, z: float, y: float, x: float):
        e = trees.get(a)
        if e is None:
            return None
        ids, tree = e
        d, j = tree.query([z * SCALE[0], y * SCALE[1], x * SCALE[2]], k=1)
        return int(ids[j]) if d <= TOL_UM else None

    out: dict[int, set[int]] = {}
    hit = tot = 0
    for r in c.iter_rows(named=True):
        tot += 1
        ms = to_sub(int(r["t"]), r["sz"], r["sy"], r["sx"])
        ds = to_sub(int(r["t"]) + 1, r["tz"], r["ty"], r["tx"])
        if ms is not None and ds is not None:
            out.setdefault(ms, set()).add(ds)
            hit += 1
    return out, (hit / tot if tot else 0.0)


def _continuations(edges, idx_of_sub, t, pos):
    """High-confidence one-parent/one-child continuations -> (frame, src_idx, displacement).

    Deployment-observable: uses only E0c graph topology, never GT.
    """
    outdeg, indeg = Counter(), Counter()
    for a, b in edges:
        outdeg[a] += 1
        indeg[b] += 1
    by_frame: dict[int, tuple[list[int], list[np.ndarray]]] = {}
    for a, b in edges:
        if outdeg[a] != 1 or indeg[b] != 1:
            continue
        ia, ib = idx_of_sub.get(int(a)), idx_of_sub.get(int(b))
        if ia is None or ib is None:
            continue
        f = int(t[ia])
        e = by_frame.setdefault(f, ([], []))
        e[0].append(ia)
        e[1].append(pos[ib] - pos[ia])
    return by_frame


def propose(sub, t, pos, surface, native_map, edges=()):
    """GT-FREE candidate generation. Returns (mother -> canonical (d1,d2) pairs, flow_stats)."""
    from scipy.spatial import cKDTree
    cfg = SURFACES[surface]
    by_t: dict[int, list[int]] = {}
    for i, tt in enumerate(t):
        by_t.setdefault(int(tt), []).append(i)
    idx_of_sub = {int(s): i for i, s in enumerate(sub)}
    use_flow = "midpoint_um" in cfg

    fstat = Counter()
    cont = _continuations(edges, idx_of_sub, t, pos) if use_flow else {}
    if use_flow:
        all_d = [d for f in cont.values() for d in f[1]]
        global_med = np.median(np.stack(all_d), axis=0) if all_d else np.zeros(3)

    out: dict[int, list[tuple[int, int]]] = {}
    for tt in sorted(by_t):
        mothers = by_t.get(tt, [])
        kids = by_t.get(tt + 1, [])
        if not mothers or not kids:
            continue
        if surface == "native":
            get_d = lambda mi: [idx_of_sub[d] for d in native_map.get(int(sub[mi]), ())
                                if d in idx_of_sub]
        else:
            tree = cKDTree(pos[kids])
            get_d = lambda mi: [kids[j] for j in tree.query_ball_point(pos[mi], cfg["parent_um"])]

        if use_flow:
            src, disp = cont.get(tt, ([], []))
            frame_med = np.median(np.stack(disp), axis=0) if disp else None
            ktree = cKDTree(pos[src]) if (cfg["flow"] == "knn" and len(src) >= cfg["knn_min"]) else None
            disp_arr = np.stack(disp) if disp else None

            def flow_at(mi):
                if ktree is not None:
                    k = min(cfg["knn_k"], len(src))
                    _, jj = ktree.query(pos[mi], k=k)
                    jj = np.atleast_1d(jj)
                    if len(jj) >= cfg["knn_min"]:
                        fstat["knn"] += 1
                        return np.median(disp_arr[jj], axis=0)
                if frame_med is not None:
                    fstat["frame_median"] += 1
                    return frame_med
                fstat["global_median"] += 1
                return global_med

        for mi in mothers:
            D = get_d(mi)
            if len(D) < 2:
                continue
            pc = pos[mi] + flow_at(mi) if use_flow else None
            pairs = []
            for a in range(len(D)):
                for b in range(a + 1, len(D)):
                    i1, i2 = D[a], D[b]
                    if cfg["sister_um"] is not None:
                        if float(np.linalg.norm(pos[i1] - pos[i2])) > cfg["sister_um"]:
                            continue
                    if use_flow:
                        mid = 0.5 * (pos[i1] + pos[i2])
                        if float(np.linalg.norm(mid - pc)) > cfg["midpoint_um"]:
                            continue
                    s1, s2 = int(sub[i1]), int(sub[i2])
                    pairs.append((min(s1, s2), max(s1, s2)))
            if pairs:
                out[int(sub[mi])] = sorted(set(pairs))
    return out, dict(fstat)


def audit_one(args) -> dict:
    split, crop, surface, persist = args
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, load_graph, score_pred_graph
    from biotrack.submission import submission_to_graphs

    t0 = time.time()
    sub, tt, pos, edges, nd = load_e0c_tables(split, crop)
    native_map, native_fid = native_pairs_by_mother(split, crop, nd) if surface == "native" else ({}, 1.0)

    # ---- GT-FREE generation ------------------------------------------------
    prop, flow_stats = propose(sub, tt, pos, surface, native_map, edges)

    # ---- GT used ONLY from here, for auditing and labels -------------------
    gt_geff = str(ROOT / "data" / "train" / f"{crop}.geff")
    gt = load_graph(gt_geff)
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    internal = g.bulk_add_nodes([{"t": int(a), "z": float(z), "y": float(y), "x": float(x)}
                                 for a, z, y, x in zip(nd["t"], nd["z"], nd["y"], nd["x"])])
    sub_to_int = {int(s): internal[i] for i, s in enumerate(nd["node_id"].to_list())}
    if edges:
        g.bulk_add_edges([{"source_id": sub_to_int[s], "target_id": sub_to_int[t_]} for s, t_ in edges])
    g.match(gt, matching=DistanceMatching(max_distance=MAX_DISTANCE, scale=DEFAULT_SCALE))
    na = g.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    int_to_gt = {int(r[td.DEFAULT_ATTR_KEYS.NODE_ID]): r[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
                 for r in na.iter_rows(named=True)}
    from gt_collision import gt_maps_from_matches
    gt_to_sub, sub_to_gt, _ = gt_maps_from_matches(
        sub_to_int, int_to_gt, context=f"d0p_proposer {crop}")

    ids = [int(n) for n in gt.node_ids()]
    outdeg = dict(zip(ids, gt.out_degree(ids)))
    gt_div = {}
    for n in ids:
        if outdeg[n] >= 2:
            ch = [int(c) for c in gt.successors(int(n))][:2]
            if len(ch) == 2:
                gt_div[int(n)] = (ch[0], ch[1])

    parents_of = {}
    for s, t_ in edges:
        parents_of.setdefault(t_, set()).add(s)

    # coverage
    n_gt_div = len(gt_div)
    reachable = covered_reach = covered_any = 0
    proposable: dict[int, tuple[int, int]] = {}
    for M, (C1, C2) in gt_div.items():
        pM, p1, p2 = gt_to_sub.get(M), gt_to_sub.get(C1), gt_to_sub.get(C2)
        is_reach = pM is not None and p1 is not None and p2 is not None and p1 != p2
        if is_reach:
            reachable += 1
        if not is_reach:
            continue
        key = (min(p1, p2), max(p1, p2))
        if key in set(prop.get(pM, ())):
            covered_reach += 1
            covered_any += 1
            proposable[M] = (C1, C2)

    # candidate accounting by mother category
    cat = Counter()
    per_mother, steals = [], 0
    total = 0
    for m, pairs in prop.items():
        gm = sub_to_gt.get(m)
        if gm is None:
            c = "unlabeled"
        elif outdeg.get(gm, 0) >= 2:
            c = "true_divider"
        elif outdeg.get(gm, 0) == 1:
            c = "reliable_negative"
        else:
            c = "annotated_end"
        cat[c] += len(pairs)
        total += len(pairs)
        per_mother.append(len(pairs))
        for d1, d2 in pairs:
            if any(p != m for d in (d1, d2) for p in parents_of.get(d, ())):
                steals += 1

    if persist and prop:
        TABLE_DIR.joinpath(surface, str(split)).mkdir(parents=True, exist_ok=True)
        t_of_sub = {int(s): int(a) for s, a in zip(sub, tt)}
        rows = [{"crop": crop, "split": split, "surface": surface,
                 "surface_hash": surface_hash(surface), "t": t_of_sub[m],
                 "mother": m, "d1": d1, "d2": d2,
                 "steal_required": int(any(p != m for d in (d1, d2)
                                           for p in parents_of.get(d, ()))),
                 "mother_category": ("unlabeled" if sub_to_gt.get(m) is None else
                                     "true_divider" if outdeg.get(sub_to_gt[m], 0) >= 2 else
                                     "reliable_negative" if outdeg.get(sub_to_gt[m], 0) == 1 else
                                     "annotated_end")}
                for m, prs in prop.items() for d1, d2 in prs]
        pl.DataFrame(rows).write_parquet(TABLE_DIR / surface / str(split) / f"{crop}.parquet")

    # ---- restricted oracle: suppress_all then add_replace, PROPOSABLE forks only
    edge_set = {(int(a), int(b)) for a, b in edges}
    par, ch = {}, {}
    for a, b in edge_set:
        ch.setdefault(a, set()).add(b)
        par.setdefault(b, set()).add(a)
    for m in [n for n, k in ch.items() if len(k) >= 2]:           # suppress_all (GT-free rule)
        keep = min(ch[m])
        for k in list(ch[m]):
            if k != keep:
                edge_set.discard((m, k)); ch[m].discard(k); par.get(k, set()).discard(m)
    used = set()
    for M, (C1, C2) in proposable.items():
        pM, p1, p2 = gt_to_sub[M], gt_to_sub[C1], gt_to_sub[C2]
        if p1 in used or p2 in used:
            continue
        for k in list(ch.get(pM, set())):
            if k not in (p1, p2):
                edge_set.discard((pM, k)); ch[pM].discard(k); par.get(k, set()).discard(pM)
        for pc in (p1, p2):
            for s in list(par.get(pc, set())):
                if s != pM:
                    edge_set.discard((s, pc)); par[pc].discard(s); ch.get(s, set()).discard(pc)
            edge_set.add((pM, pc)); ch.setdefault(pM, set()).add(pc); par.setdefault(pc, set()).add(pM)
        used |= {p1, p2}

    node_rows = nd.select("row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id")
    er = pl.DataFrame([{"row_type": "edge", "node_id": -1, "t": -1, "z": -1.0, "y": -1.0, "x": -1.0,
                        "source_id": a, "target_id": b} for a, b in edge_set])
    out = pl.concat([node_rows, er]) if er.height else node_rows
    graph = submission_to_graphs(out.with_columns(pl.lit(crop).alias("dataset")).with_row_index("id"))[crop]
    row = score_pred_graph(graph, gt_geff)

    pm = np.asarray(per_mother) if per_mother else np.array([0])
    return {"split": split, "crop": crop, "surface": surface,
            "n_gt_div": n_gt_div, "reachable": reachable, "covered_reachable": covered_reach,
            "cand_total": total, "cand_steals": steals,
            **{f"cat_{k}": v for k, v in cat.items()},
            "mothers_with_cands": len(prop),
            "pm_median": float(np.median(pm)), "pm_p90": float(np.percentile(pm, 90)),
            "pm_p99": float(np.percentile(pm, 99)), "pm_max": int(pm.max()),
            "runtime_s": time.time() - t0, "native_map_fidelity": native_fid,
            **{f"flow_{k}": v for k, v in flow_stats.items()},
            **{k: row[k] for k in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp",
                                   "division_fn", "node_recall", "num_pred_nodes")}}


def aggregate(res, surface) -> list[dict]:
    from biotrack.metric import estimated_nodes, per_sample_metrics
    from tracking_cellmot.metrics import EvaluationResult, summarise
    print(f"\n########## D0P  surface={surface} (hash {surface_hash(surface)}) ##########")
    out = []
    for fold, fam in ((0, "44b6"), (1, "6bba")):
        rows, agg = [], Counter()
        pm_all, npred = [], 0
        for c in cached_crops(fold):
            r = res[(fold, c)]
            rows.append(per_sample_metrics(
                EvaluationResult(r["edge_tp"], r["edge_fp"], r["edge_fn"], r["division_tp"],
                                 r["division_fp"], r["division_fn"], r["num_pred_nodes"]),
                estimated_nodes(str(ROOT / "data" / "train" / f"{c}.geff")), r["node_recall"]))
            npred += r["num_pred_nodes"]
            for k, v in r.items():
                if k.startswith(("cat_", "cand_", "flow_")) or k in ("n_gt_div", "reachable",
                                                            "covered_reachable", "mothers_with_cands"):
                    agg[k] += v
            agg["runtime_s"] += r["runtime_s"]
            pm_all.append((r["pm_median"], r["pm_p90"], r["pm_p99"], r["pm_max"]))
        s = summarise(rows)
        delta = s["score"] - E0C[fold]
        pm = np.asarray(pm_all)
        print(f"  {fam}: proposable-only oracle composite={s['score']:.4f} (dvs E0c {delta:+.4f}) "
              f"divJ={s['division_jaccard']:.4f} (TP{s['division_tp']}/FP{s['division_fp']}/"
              f"FN{s['division_fn']}) recall={s['node_recall']:.4f}")
        print(f"     coverage: reachable divisions {agg['covered_reachable']}/{agg['reachable']} "
              f"| all GT divisions {agg['covered_reachable']}/{agg['n_gt_div']}")
        print(f"     candidates: total={agg['cand_total']:,} steals={agg['cand_steals']:,} "
              f"mothers={agg['mothers_with_cands']:,}")
        print(f"       by mother: true_divider={agg.get('cat_true_divider',0):,} "
              f"reliable_negative={agg.get('cat_reliable_negative',0):,} "
              f"annotated_end={agg.get('cat_annotated_end',0):,} "
              f"unlabeled={agg.get('cat_unlabeled',0):,}")
        print(f"       per-mother pairs: median={np.median(pm[:,0]):.1f} p90={np.median(pm[:,1]):.1f} "
              f"p99={np.median(pm[:,2]):.1f} max={int(pm[:,3].max())}")
        fl = {k[5:]: v for k, v in agg.items() if k.startswith("flow_")}
        if fl:
            tot_f = sum(fl.values()) or 1
            print("     flow source: " + " ".join(f"{k}={v:,}({v/tot_f:.3f})" for k, v in sorted(fl.items())))
        print(f"     runtime={agg['runtime_s']:.0f}s over {len(rows)} crops")
        out.append({"surface": surface, "surface_hash": surface_hash(surface), "fold": fold,
                    "family": fam, "composite": s["score"], "delta_vs_e0c": delta,
                    "division_jaccard": s["division_jaccard"], "division_tp": s["division_tp"],
                    "division_fp": s["division_fp"], "division_fn": s["division_fn"],
                    "node_recall": s["node_recall"], **dict(agg),
                    "pm_median": float(np.median(pm[:, 0])), "pm_p90": float(np.median(pm[:, 1])),
                    "pm_p99": float(np.median(pm[:, 2])), "pm_max": int(pm[:, 3].max())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--surfaces", default="geometric_core,native,outer_diag")
    ap.add_argument("--persist", action="store_true", help="write the canonical bridge table")
    a = ap.parse_args()
    all_rows = []
    for surface in a.surfaces.split(","):
        tasks = [(s, c, surface, a.persist) for s in (0, 1) for c in cached_crops(s)]
        res = {}
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(audit_one, tasks):
                res[(r["split"], r["crop"])] = r
        all_rows.extend(aggregate(res, surface))
    OUT_JSON.write_text(json.dumps({"surfaces": SURFACES, "e0c_anchor": E0C,
                                    "rows": all_rows}, indent=2, default=float))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
