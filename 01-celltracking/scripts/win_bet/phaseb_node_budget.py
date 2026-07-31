"""Node-budget pruning sweep — the unattacked channel.

The metric is `adj_J = J * (1 - 0.1*(N_pred - N_est)/N_est)` and it is NOT clipped above 1, so
UNDER-prediction is rewarded. Measured on our own 199-crop OOF cache, pooled score is
near-monotonic in the realised multiplier:

    arm    N_pred      ratio     mult     pooled
    E0c    5,118,041   +0.0832   0.9963   0.66539
    B      5,555,512   +0.1757   0.9895   0.66677
    Bp     4,695,278   -0.0063   1.0010   0.69162
    C      3,902,230   -0.1742   1.0129   0.69692
    v122   3,969,944   -0.1598   1.0114   0.69909

That spread is ~0.023 of composite against a 0.034 total arm spread. The channel has never been
deliberately optimised.

CRITICAL CONSTRAINT: random deletion is strictly negative -- d(adjJ)/df ~ J*(-2 + 0.1(1+r)) < 0 --
so only SELECTIVE deletion can pay, and the realisable gain IS detector/component precision. This
script therefore prunes WEAKEST-COMPONENT-FIRST and reports the honest curve rather than assuming
a gain exists.

Ranking is deployment-observable only (no GT): components are ordered by (length, mean edge
probability where available), weakest first, and division-containing components are protected.

KILL RULE (preregistered): if the pooled composite optimum sits at keep_frac >= 0.99 on BOTH
families, selective pruning cannot beat the status quo and the channel closes.

Usage:
  .venv\\Scripts\\python.exe scripts\\win_bet\\phaseb_node_budget.py --arm A --workers 4 --crops 20
  .venv\\Scripts\\python.exe scripts\\win_bet\\phaseb_node_budget.py --arm D --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

from phaseb_d0p_proposer import CACHE, cached_crops  # noqa: E402

OUT = ROOT / "reports/inventory/node_budget_sweep.json"
KEEP_FRACS = (1.00, 0.975, 0.95, 0.90, 0.85, 0.80)


def _components(nodes, edges):
    """Weakly connected components over the predicted graph."""
    adj: dict[int, set[int]] = {n: set() for n in nodes}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen, comps = set(), []
    for n in nodes:
        if n in seen:
            continue
        stack, cur = [n], []
        seen.add(n)
        while stack:
            u = stack.pop()
            cur.append(u)
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append(cur)
    return comps


def prune_one(args) -> dict:
    split, crop, arm, keep_fracs = args
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from biotrack.metric import estimated_nodes, score_pred_graph
    from biotrack.submission import submission_to_graphs

    # arm A == the canonical E0c cache; other arms live in the coupled cache if materialised
    gp = CACHE / "graphs" / str(split) / f"{crop}.parquet"
    if not gp.exists():
        return {"split": split, "crop": crop, "skipped": True}
    df = pl.read_parquet(gp)
    nd = df.filter(pl.col("row_type") == "node").sort("node_id")
    edges = [(int(r["source_id"]), int(r["target_id"]))
             for r in df.filter(pl.col("row_type") == "edge").iter_rows(named=True)]
    nodes = [int(x) for x in nd["node_id"].to_list()]
    gt_geff = str(ROOT / "data" / "train" / f"{crop}.geff")
    n_est = estimated_nodes(gt_geff)

    outdeg: dict[int, int] = {}
    for a, _b in edges:
        outdeg[a] = outdeg.get(a, 0) + 1
    comps = _components(nodes, edges)
    # weakest first: short components first; protect any component containing a fork
    def key(c):
        has_div = any(outdeg.get(n, 0) >= 2 for n in c)
        return (1 if has_div else 0, len(c))
    comps.sort(key=key)

    node_rows = nd.select("row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id")
    out = {"split": split, "crop": crop, "n_nodes": len(nodes), "n_est": n_est,
           "n_comps": len(comps), "arms": {}}

    for kf in keep_fracs:
        target = int(round(kf * len(nodes)))
        drop: set[int] = set()
        i = 0
        while len(nodes) - len(drop) > target and i < len(comps):
            drop.update(comps[i])
            i += 1
        keep = [n for n in nodes if n not in drop]
        if not keep:
            continue
        keep_set = set(keep)
        nr = node_rows.filter(pl.col("node_id").is_in(keep))
        ee = [(a, b) for a, b in edges if a in keep_set and b in keep_set]
        er = pl.DataFrame([{"row_type": "edge", "node_id": -1, "t": -1, "z": -1.0, "y": -1.0,
                            "x": -1.0, "source_id": a, "target_id": b} for a, b in ee])
        o = pl.concat([nr, er]) if er.height else nr
        g = submission_to_graphs(o.with_columns(pl.lit(crop).alias("dataset")).with_row_index("id"))[crop]
        r = score_pred_graph(g, gt_geff)
        out["arms"][str(kf)] = {k: r[k] for k in
                                ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp",
                                 "division_fn", "node_recall", "num_pred_nodes")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crops", type=int, default=0, help="0 = all")
    a = ap.parse_args()
    from biotrack.metric import per_sample_metrics
    from tracking_cellmot.metrics import EvaluationResult, summarise

    tasks = []
    for s in (0, 1):
        cs = cached_crops(s)
        if a.crops:
            cs = cs[: a.crops]
        tasks += [(s, c, a.arm, KEEP_FRACS) for c in cs]
    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(prune_one, tasks):
            if not r.get("skipped"):
                res.append(r)

    def rows(kf, subset=None):
        out = []
        for r in res:
            if subset and not r["crop"].startswith(subset):
                continue
            d = r["arms"].get(str(kf))
            if d is None:
                continue
            out.append(per_sample_metrics(
                EvaluationResult(d["edge_tp"], d["edge_fp"], d["edge_fn"], d["division_tp"],
                                 d["division_fp"], d["division_fn"], d["num_pred_nodes"]),
                r["n_est"], d["node_recall"]))
        return out

    print(f"arm={a.arm} crops={len(res)}")
    print(f"{'keep':>7}{'pooled':>10}{'delta':>10}{'44b6':>10}{'6bba':>10}{'ratio':>10}{'recall':>9}")
    base = None
    table = {}
    for kf in KEEP_FRACS:
        rr = rows(kf)
        if not rr:
            continue
        p = summarise(rr)
        npred = sum(r["arms"][str(kf)]["num_pred_nodes"] for r in res if str(kf) in r["arms"])
        nest = sum(r["n_est"] for r in res)
        if base is None:
            base = p["score"]
        f0, f1 = summarise(rows(kf, "44b6")), summarise(rows(kf, "6bba"))
        print(f"{kf:>7.3f}{p['score']:>10.5f}{p['score']-base:>+10.5f}{f0['score']:>10.5f}"
              f"{f1['score']:>10.5f}{(npred-nest)/nest:>+10.4f}{p['node_recall']:>9.4f}")
        table[str(kf)] = {"pooled": p["score"], "delta": p["score"] - base,
                          "f44b6": f0["score"], "f6bba": f1["score"],
                          "ratio": (npred - nest) / nest, "node_recall": p["node_recall"]}

    best = max(table, key=lambda k: table[k]["pooled"])
    b0 = max(table, key=lambda k: table[k]["f44b6"])
    b1 = max(table, key=lambda k: table[k]["f6bba"])
    print(f"\n  pooled optimum keep_frac = {best}  (+{table[best]['delta']:.5f})")
    print(f"  per-family optima: 44b6 {b0}, 6bba {b1}")
    killed = float(b0) >= 0.99 and float(b1) >= 0.99
    print(f"  KILL RULE (optimum >= 0.99 on BOTH families): {'TRIGGERED - channel closes' if killed else 'NOT triggered - selective pruning pays'}")
    OUT.write_text(json.dumps({"arm": a.arm, "n_crops": len(res), "table": table,
                               "pooled_optimum": best, "f44b6_optimum": b0,
                               "f6bba_optimum": b1, "kill_triggered": killed},
                              indent=2, default=float))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
