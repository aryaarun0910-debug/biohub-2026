"""H1-T — exact H0c replay with the ORACLE fork selector replaced by the external critic.

H0c measured the ceiling of the frozen top-3 shortlist by admitting a pair iff ground truth
said it was the true one (+0.0625 / +0.0597, retention 80.0% / 81.7%). That number is not
submittable. This script keeps the cascade byte-identical and swaps in the only thing that
was oracle: instead of `retained = GT true pair`, a mother's highest-scoring shortlist pair
is admitted when the critic score clears tau.

Everything else is the frozen H0c code path: 15/15 generation, kNN flow, flow-midpoint top-3,
suppress-all with GT-free lowest-id child retention, add-replace conflict resolution, exact
patched scorer, per-crop baseline scored alongside for node invariance.

The critic weights come from `h1t_external_critic.py --deploy`, trained only on Zebrahub.
tau is either fully external (no competition calibration at all) or cross-fitted between
families (one scalar, never chosen and tested on the same embryos). Family identity is not
an input.

Usage:
  .venv\\Scripts\\python.exe scripts\\win_bet\\h1t_h0c_learned.py --tau 0.0 --workers 6
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

from phaseb_d0p_proposer import cached_crops, load_e0c_tables  # noqa: E402
from phaseb_h0c_replay import CFG, CFG_HASH, shortlist  # noqa: E402
from h1t_external_critic import FEATS  # noqa: E402
from h1t_zebrahub_events import h1g_features  # noqa: E402

SCRATCH = Path(r"C:\Users\aryaa\Documents\Biohub-CellTracking-2026_RESEARCH\agent_runs\agent4")
WEIGHTS = SCRATCH / "h1t_critic_weights.npz"
OUT_JSON = SCRATCH / "h1t_h0c_learned.json"
E0C = {0: 0.7595, 1: 0.6490}


def load_mlp():
    z = np.load(WEIGHTS)
    mu, sd = z["mu"], z["sd"]
    W = [(z[f"{i}.weight"], z[f"{i}.bias"]) for i in (0, 2, 4)]
    def fwd(X):
        h = (X - mu) / sd
        for k, (w, b) in enumerate(W):
            h = h @ w.T + b
            if k < len(W) - 1:
                h = np.maximum(h, 0.0)
        return h[:, 0]
    return fwd


def replay_one(args) -> dict:
    split, crop, tau = args
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, load_graph, score_pred_graph
    from biotrack.submission import submission_to_graphs

    fwd = load_mlp()
    sub, t, pos, edges, nd = load_e0c_tables(split, crop)
    prop, _ = shortlist(sub, t, pos, edges)                      # GT-free, frozen
    gt_geff = str(ROOT / "data" / "train" / f"{crop}.geff")
    node_rows = nd.select("row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id")

    def score(edge_set):
        er = pl.DataFrame([{"row_type": "edge", "node_id": -1, "t": -1, "z": -1.0, "y": -1.0,
                            "x": -1.0, "source_id": a, "target_id": b} for a, b in edge_set])
        o = pl.concat([node_rows, er]) if er.height else node_rows
        g = submission_to_graphs(o.with_columns(pl.lit(crop).alias("dataset")).with_row_index("id"))[crop]
        return score_pred_graph(g, gt_geff)

    base = score({(int(a), int(b)) for a, b in edges})

    # ---- GT-FREE selection: features -> critic -> argmax pair per mother above tau ----
    parent_of = {int(b): int(a) for a, b in edges}
    rows = []
    for m, ranked in prop.items():
        for rank, (resid, (d1, d2)) in enumerate(ranked):
            p1, p2 = parent_of.get(d1, -1), parent_of.get(d2, -1)
            rows.append({"cand_id": f"{crop}:{m}:{d1}:{d2}", "mother": int(m),
                         "d1": int(d1), "d2": int(d2), "rank": rank,
                         "flow_midpoint_residual": float(resid),
                         "steal_required": int((p1 not in (-1, m)) or (p2 not in (-1, m)))})
    retained: dict[int, tuple[int, int]] = {}
    n_admit = 0
    if rows:
        f = h1g_features(sub, t, pos, edges, rows)
        s = fwd(f.select(FEATS).to_numpy().astype(np.float32))
        f = f.with_columns(pl.Series("critic", s))
        top = (f.sort("critic", descending=True)
                 .group_by("mother", maintain_order=True).first()
                 .filter(pl.col("critic") >= tau))
        for r in top.iter_rows(named=True):
            d1, d2 = int(r["d1"]), int(r["d2"])
            retained[int(r["mother"])] = (min(d1, d2), max(d1, d2))
        n_admit = len(retained)

    # ---- identical suppress-all then add-replace --------------------------------------
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
    steals = 0
    used = set()
    for pM, (p1, p2) in retained.items():
        if p1 in used or p2 in used:
            continue
        for k in list(ch_.get(pM, set())):
            if k not in (p1, p2):
                es.discard((pM, k)); ch_[pM].discard(k); par.get(k, set()).discard(pM)
        for pc in (p1, p2):
            for s_ in list(par.get(pc, set())):
                if s_ != pM:
                    es.discard((s_, pc)); par[pc].discard(s_); ch_.get(s_, set()).discard(pc)
                    steals += 1
            es.add((pM, pc)); ch_.setdefault(pM, set()).add(pc); par.setdefault(pc, set()).add(pM)
        used |= {p1, p2}
    treat = score(es)

    return {"split": split, "crop": crop, "admitted": n_admit, "steals": steals,
            "mothers": len(prop),
            **{f"b_{k}": base[k] for k in ("edge_tp", "edge_fp", "edge_fn", "division_tp",
                                           "division_fp", "division_fn", "node_recall",
                                           "num_pred_nodes")},
            **{f"t_{k}": treat[k] for k in ("edge_tp", "edge_fp", "edge_fn", "division_tp",
                                            "division_fp", "division_fn", "node_recall",
                                            "num_pred_nodes")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--tau-44b6", type=float, default=None)
    ap.add_argument("--tau-6bba", type=float, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--label", type=str, default="external_tau")
    a = ap.parse_args()
    from biotrack.metric import estimated_nodes, per_sample_metrics
    from tracking_cellmot.metrics import EvaluationResult, summarise

    taus = {0: a.tau_44b6 if a.tau_44b6 is not None else a.tau,
            1: a.tau_6bba if a.tau_6bba is not None else a.tau}
    print(f"H1-T learned selection on frozen H0c cascade (cfg {CFG_HASH}); taus={taus}")
    rows_out = []
    for fold, fam in ((0, "44b6"), (1, "6bba")):
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(replay_one, [(fold, c, taus[fold]) for c in cached_crops(fold)]))
        b_rows, t_rows, regress = [], [], []
        for r in res:
            ne = estimated_nodes(str(ROOT / "data" / "train" / f"{r['crop']}.geff"))
            b = per_sample_metrics(EvaluationResult(*[r[f"b_{k}"] for k in
                ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn",
                 "num_pred_nodes")]), ne, r["b_node_recall"])
            t_ = per_sample_metrics(EvaluationResult(*[r[f"t_{k}"] for k in
                ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn",
                 "num_pred_nodes")]), ne, r["t_node_recall"])
            b_rows.append(b); t_rows.append(t_)
            if t_["adj_edge_jaccard"] < b["adj_edge_jaccard"] - 1e-9:
                regress.append((r["crop"], round(t_["adj_edge_jaccard"] - b["adj_edge_jaccard"], 5)))
        sb, st = summarise(b_rows), summarise(t_rows)
        node_ok = all(r["b_num_pred_nodes"] == r["t_num_pred_nodes"] for r in res)
        print(f"\n########## H1-T learned  {fam}  tau={taus[fold]} ##########")
        print(f"  baseline composite={sb['score']:.4f} divJ={sb['division_jaccard']:.4f} "
              f"(TP{sb['division_tp']}/FP{sb['division_fp']}/FN{sb['division_fn']})")
        print(f"  learned  composite={st['score']:.4f} divJ={st['division_jaccard']:.4f} "
              f"(TP{st['division_tp']}/FP{st['division_fp']}/FN{st['division_fn']})")
        print(f"  DELTA composite={st['score']-sb['score']:+.4f}  vs E0c anchor "
              f"{st['score']-E0C[fold]:+.4f}  adjEdgeJ={st['adj_edge_jaccard']-sb['adj_edge_jaccard']:+.4f}")
        print(f"  admitted forks={sum(r['admitted'] for r in res):,} "
              f"steals={sum(r['steals'] for r in res):,} node_invariant={node_ok} "
              f"regressed crops={len(regress)}")
        rows_out.append({"family": fam, "fold": fold, "tau": taus[fold], "baseline": sb,
                         "learned": st, "delta_composite": st["score"] - sb["score"],
                         "delta_vs_e0c_anchor": st["score"] - E0C[fold],
                         "admitted": sum(r["admitted"] for r in res),
                         "steals": sum(r["steals"] for r in res),
                         "node_count_invariant": node_ok, "n_regressed_crops": len(regress)})
    prev = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    prev[a.label] = {"config": CFG, "config_hash": CFG_HASH, "taus": taus, "rows": rows_out}
    OUT_JSON.write_text(json.dumps(prev, indent=2, default=float))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
