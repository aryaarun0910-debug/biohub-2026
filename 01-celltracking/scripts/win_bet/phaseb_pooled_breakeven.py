"""Lane B1 — pooled-objective break-even for the division layer.

All previous H1 break-even numbers (4.07% / 6.38% required precision) were derived from a
per-family +0.005 requirement. That is no longer the primary objective: the leaderboard pools
all samples with edge-volume weighting, and 44b6 carries only ~15% of edge mass.

This recomputes break-even against the REAL objective. It does not re-weight already-computed
composites -- it applies the actual edits to the actual graphs and calls the authoritative
combined `summarise()` over all 199 crops in one evaluation.

Measured:
  * pooled response to k recovered true forks and m admitted false forks;
  * pooled effect of suppress-only;
  * required visible precision for pooled gains of +0.002 / +0.005 / +0.010 / +0.015;
  * marginal pooled contribution of ONE recovered division in 44b6 vs 6bba;
  * NOTE: this variant does NOT re-run the wrapper filter after the edits; it scores the
    edited edge set against the cached node rows. H0d showed the live filter is strictly more
    favourable, so these are a LOWER BOUND.

Usage:
  .venv\\Scripts\\python.exe scripts\\win_bet\\phaseb_pooled_breakeven.py --workers 6
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

from phaseb_d0p_proposer import CACHE, cached_crops, load_e0c_tables  # noqa: E402
from phaseb_h0c_replay import CFG_HASH, shortlist  # noqa: E402

OUT = ROOT / "reports/inventory/pooled_breakeven.json"
CENSUS = CACHE / "fork_candidates" / "h0c_top3"


def _score_rows(rows):
    from tracking_cellmot.metrics import summarise
    return summarise(rows)


def replay_one(args) -> dict:
    """Apply a parameterised edit and return per-crop metric rows for several arms.

    arms:
      base         unmodified cached graph
      supp         suppress-all only
      supp_k{f}    suppress-all + admit a fraction f of TRUE forks (oracle order), no FP
      supp_kfp{f}  suppress-all + all true forks + a number of FALSE forks
    """
    split, crop, fp_multipliers, true_fracs = args
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, estimated_nodes, load_graph, score_pred_graph
    from biotrack.submission import submission_to_graphs

    sub, t, pos, edges, nd = load_e0c_tables(split, crop)
    prop, _ = shortlist(sub, t, pos, edges)
    gt_geff = str(ROOT / "data" / "train" / f"{crop}.geff")
    node_rows = nd.select("row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id")

    def score(edge_set):
        er = pl.DataFrame([{"row_type": "edge", "node_id": -1, "t": -1, "z": -1.0, "y": -1.0,
                            "x": -1.0, "source_id": a, "target_id": b} for a, b in edge_set])
        o = pl.concat([node_rows, er]) if er.height else node_rows
        g = submission_to_graphs(o.with_columns(pl.lit(crop).alias("dataset")).with_row_index("id"))[crop]
        r = score_pred_graph(g, gt_geff)
        return {k: r[k] for k in ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp",
                                  "division_fn", "node_recall", "num_pred_nodes")}

    # --- GT (labels only) ---------------------------------------------------
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
    na = g.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    i2g = {int(r[td.DEFAULT_ATTR_KEYS.NODE_ID]): r[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
           for r in na.iter_rows(named=True)}
    gt_to_sub = {}
    for s, iid in s2i.items():
        m = i2g.get(iid)
        if m not in (None, -1):
            gt_to_sub[int(m)] = int(s)
    ids = [int(n) for n in gt.node_ids()]
    outdeg = dict(zip(ids, gt.out_degree(ids)))
    true_pair = {}
    for n in ids:
        if outdeg.get(n, 0) < 2:
            continue
        ch = [int(c) for c in gt.successors(int(n))][:2]
        if len(ch) != 2:
            continue
        pM, p1, p2 = gt_to_sub.get(n), gt_to_sub.get(ch[0]), gt_to_sub.get(ch[1])
        if pM is None or p1 is None or p2 is None or p1 == p2:
            continue
        if (min(p1, p2), max(p1, p2)) in {k for _, k in prop.get(pM, ())}:
            true_pair[pM] = (min(p1, p2), max(p1, p2))

    def suppressed():
        es = {(int(a), int(b)) for a, b in edges}
        par, ch_ = {}, {}
        for a, b in es:
            ch_.setdefault(a, set()).add(b)
            par.setdefault(b, set()).add(a)
        for m in [n for n, k in ch_.items() if len(k) >= 2]:
            keep = min(ch_[m])
            for k in list(ch_[m]):
                if k != keep:
                    es.discard((m, k)); ch_[m].discard(k); par.get(k, set()).discard(m)
        return es, par, ch_

    def admit(es, par, ch_, forks):
        used = set()
        for pM, (p1, p2) in forks:
            if p1 in used or p2 in used:
                continue
            for k in list(ch_.get(pM, set())):
                if k not in (p1, p2):
                    es.discard((pM, k)); ch_[pM].discard(k); par.get(k, set()).discard(pM)
            for pc in (p1, p2):
                for s in list(par.get(pc, set())):
                    if s != pM:
                        es.discard((s, pc)); par[pc].discard(s); ch_.get(s, set()).discard(pc)
                es.add((pM, pc)); ch_.setdefault(pM, set()).add(pc); par.setdefault(pc, set()).add(pM)
            used |= {p1, p2}
        return es

    out = {"split": split, "crop": crop, "n_true": len(true_pair),
           "n_est": estimated_nodes(gt_geff)}
    out["base"] = score({(int(a), int(b)) for a, b in edges})
    es, par, ch_ = suppressed()
    out["supp"] = score(set(es))

    tp_list = sorted(true_pair.items())
    rng = np.random.default_rng(20260731 + hash(crop) % 10000)

    # true-fork fraction sweep (no false forks)
    for f in true_fracs:
        k = int(round(f * len(tp_list)))
        es2, par2, ch2 = suppressed()
        out[f"k{f}"] = score(admit(es2, par2, ch2, tp_list[:k]))

    # false-fork sweep: all true forks + m false forks drawn from METRIC-VISIBLE non-true
    cen_p = CENSUS / str(split) / f"{crop}.parquet"
    fps = []
    if cen_p.exists():
        c = pl.read_parquet(cen_p).filter(pl.col("metric_visible") & (pl.col("label") != "positive"))
        if c.height:
            idx = rng.permutation(c.height)[:2000]
            fps = [(int(c["mother"][int(i)]), (int(c["d1"][int(i)]), int(c["d2"][int(i)])))
                   for i in idx]
    for mult in fp_multipliers:
        m = int(round(mult * len(tp_list)))
        es3, par3, ch3 = suppressed()
        es3 = admit(es3, par3, ch3, tp_list)
        es3 = admit(es3, par3, ch3, fps[:m])
        out[f"fp{mult}"] = score(es3)
    out["n_fp_pool"] = len(fps)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    true_fracs = [0.25, 0.50, 0.75, 1.0]
    fp_mults = [1, 2, 5, 10, 25]
    tasks = [(s, c, fp_mults, true_fracs) for s in (0, 1) for c in cached_crops(s)]
    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(replay_one, tasks):
            res.append(r)

    from biotrack.metric import per_sample_metrics
    from tracking_cellmot.metrics import EvaluationResult

    def rows_for(arm, subset=None):
        rows = []
        for r in res:
            if subset and not r["crop"].startswith(subset):
                continue
            d = r[arm]
            rows.append(per_sample_metrics(
                EvaluationResult(d["edge_tp"], d["edge_fp"], d["edge_fn"], d["division_tp"],
                                 d["division_fp"], d["division_fn"], d["num_pred_nodes"]),
                r["n_est"], d["node_recall"]))
        return rows

    arms = ["base", "supp"] + [f"k{f}" for f in true_fracs] + [f"fp{m}" for m in fp_mults]
    base_pooled = _score_rows(rows_for("base"))["score"]
    print(f"pooled BASE = {base_pooled:.5f}   (cfg {CFG_HASH})")
    print(f"\n{'arm':<10}{'pooled':>10}{'delta':>10}{'44b6':>10}{'6bba':>10}{'divTP':>8}{'divFP':>8}")
    table = {}
    for arm in arms:
        p = _score_rows(rows_for(arm))
        f0 = _score_rows(rows_for(arm, "44b6"))
        f1 = _score_rows(rows_for(arm, "6bba"))
        d = p["score"] - base_pooled
        print(f"{arm:<10}{p['score']:>10.5f}{d:>+10.5f}{f0['score']:>10.5f}{f1['score']:>10.5f}"
              f"{p['division_tp']:>8}{p['division_fp']:>8}")
        table[arm] = {"pooled": p["score"], "delta": d, "f44b6": f0["score"], "f6bba": f1["score"],
                      "div_tp": p["division_tp"], "div_fp": p["division_fp"],
                      "div_fn": p["division_fn"]}

    n_true = sum(r["n_true"] for r in res)
    n0 = sum(r["n_true"] for r in res if r["crop"].startswith("44b6"))
    print(f"\nretained true forks: {n_true} total ({n0} in 44b6, {n_true-n0} in 6bba)")

    # marginal contribution of one recovered division, by family
    for fam, tag in (("44b6", "44b6"), ("6bba", "6bba")):
        rows_b = rows_for("base", tag)
        rows_f = rows_for("k1.0", tag)
        nb = n0 if fam == "44b6" else n_true - n0
        if nb:
            dd = _score_rows(rows_f)["score"] - _score_rows(rows_b)["score"]
            print(f"  {fam}: {nb} recovered -> family delta {dd:+.5f}  ({dd/nb:+.6f} per division)")

    # required precision for pooled targets, interpolated on the FP sweep
    print(f"\n{'target':>8}{'max FP ratio':>14}{'implied precision':>20}")
    xs = [0] + fp_mults
    ys = [table["k1.0"]["delta"]] + [table[f"fp{m}"]["delta"] for m in fp_mults]
    req = {}
    for tgt in (0.002, 0.005, 0.010, 0.015):
        r = None
        for i in range(len(xs) - 1):
            if (ys[i] - tgt) * (ys[i + 1] - tgt) <= 0 and ys[i] != ys[i + 1]:
                r = xs[i] + (xs[i + 1] - xs[i]) * (ys[i] - tgt) / (ys[i] - ys[i + 1])
                break
        if r is None:
            r = float("inf") if ys[0] < tgt else float(xs[-1])
        prec = 1.0 / (1.0 + r) if np.isfinite(r) else float("nan")
        req[tgt] = {"max_fp_ratio": r, "implied_precision": prec}
        print(f"{tgt:>8.3f}{r:>14.2f}{prec*100:>19.2f}%")

    OUT.write_text(json.dumps({"cfg_hash": CFG_HASH, "base_pooled": base_pooled,
                               "arms": table, "n_true_retained": n_true,
                               "n_true_44b6": n0, "required": req}, indent=2, default=float))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
