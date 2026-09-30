"""H4-S: exact-scorer replay of the frozen H0c cascade with a GT-FREE GATE in place of
oracle fork selection.

`phaseb_h0c_replay.py` measures the H0c ceiling by retaining, for every mother, the true
daughter pair whenever it survived the top-3 shortlist -- oracle selection.  This script
replaces that single step, and only that step, with a deployable admission decision:

    retained = { m -> rank-0 shortlist pair : m in GATE_ADMITTED }

Everything else is byte-identical to the frozen cascade: the same 15/15 generation, the
same flow-midpoint ranking, the same K=3, the same suppress-all with deterministic
lowest-id child retention, the same add-replace conflict resolver, the same exact patched
scorer, and the same per-crop baseline scored alongside.  `shortlist` and `score` are
imported from the frozen module rather than re-implemented, so drift is impossible.

WHY THIS RUN EXISTS.  Every number in the H4-S analysis is converted to composite through
H0c's own measured arithmetic,  dcomposite ~= 0.1*(divJ - divJ_base) + edge_cost,  with a
CONSTANT edge cost.  reports/NEXT_DECISION.md warns precisely against trusting that:
"the edge cost is unconditional while the division gain is conditional on correct fork
selection -- a weak classifier keeps the cost and loses the gain."  A gate admitting ~200
mothers suppresses far fewer forks and steals far fewer parents than the oracle arm, so
its true edge cost is NOT the oracle arm's edge cost.  Only the exact scorer settles it.

This is heavy local CPU (199 crops, exact scorer, ~6 workers) and MUST NOT be run without
explicit human go-ahead.

Inputs
------
--admit  parquet with columns (crop, mother) listing the gate's admitted mothers, written
         by `scripts/h4_ssl_fuse.py --export-admit`.  Nothing else about the gate enters.

Usage (AWAITING GO)
-------------------
  .venv\\Scripts\\python.exe scripts\\win_bet\\h4_ssl_gate_replay.py ^
      --admit <admit.parquet> --workers 6 --out research/06-knowledge-system/inventory/h4_ssl_gate_replay.json
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

ADMIT: dict[str, set[int]] = {}


def replay_one(args) -> dict:
    split, crop, admitted = args
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from biotrack.metric import estimated_nodes, load_graph, score_pred_graph  # noqa: F401
    from biotrack.submission import submission_to_graphs

    sub, t, pos, edges, nd = load_e0c_tables(split, crop)
    prop, n_pre = shortlist(sub, t, pos, edges)          # GT-free, frozen
    gt_geff = str(ROOT / "data" / "train" / f"{crop}.geff")
    node_rows = nd.select("row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id")

    def score(edge_set):
        er = pl.DataFrame([{"row_type": "edge", "node_id": -1, "t": -1, "z": -1.0, "y": -1.0,
                            "x": -1.0, "source_id": a, "target_id": b} for a, b in edge_set])
        o = pl.concat([node_rows, er]) if er.height else node_rows
        g = submission_to_graphs(
            o.with_columns(pl.lit(crop).alias("dataset")).with_row_index("id"))[crop]
        return score_pred_graph(g, gt_geff)

    base = score({(int(a), int(b)) for a, b in edges})

    # ---- GATE selection: rank-0 shortlist pair of every ADMITTED mother -----
    # No ground truth is consulted anywhere in this block.
    retained = {}
    for m in admitted:
        cand = prop.get(int(m))
        if cand:
            retained[int(m)] = cand[0][1]                # rank-0 pair

    # ---- suppress-all then add-replace (identical to phaseb_h0c_replay) -----
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
    used: set[int] = set()
    for pM, (p1, p2) in sorted(retained.items()):
        if p1 in used or p2 in used:
            continue
        for k in list(ch_.get(pM, set())):
            if k not in (p1, p2):
                es.discard((pM, k)); ch_[pM].discard(k); par.get(k, set()).discard(pM)
        for pc in (p1, p2):
            for s in list(par.get(pc, set())):
                if s != pM:
                    es.discard((s, pc)); par[pc].discard(s); ch_.get(s, set()).discard(pc)
                    steals += 1
            es.add((pM, pc)); ch_.setdefault(pM, set()).add(pc); par.setdefault(pc, set()).add(pM)
        used |= {p1, p2}
    treat = score(es)

    keys = ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn",
            "node_recall", "num_pred_nodes")
    return {"split": split, "crop": crop, "n_admitted": len(admitted),
            "n_forked": len(retained), "steals": steals,
            # 2026-07-31: n_est is REQUIRED by agg().  The count multiplier is defined against
            # the GEFF estimated_number_of_nodes, not against our own baseline node count.
            "n_est": int(estimated_nodes(gt_geff)),
            "cand_shortlist": sum(len(v) for v in prop.values()), "cand_pre_topk": n_pre,
            **{f"b_{k}": base[k] for k in keys},
            **{f"t_{k}": treat[k] for k in keys}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--admit", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "research/06-knowledge-system/inventory/h4_ssl_gate_replay.json"))
    a = ap.parse_args()

    adm = pl.read_parquet(a.admit)
    by_crop: dict[str, set[int]] = {}
    for r in adm.iter_rows(named=True):
        by_crop.setdefault(r["crop"], set()).add(int(r["mother"]))

    # 2026-07-31 DEFECT FIX: this called cached_crops() with no argument, but it is defined
    # cached_crops(split: int) and yields crop STEMS, not (split, crop) pairs -- so this line
    # raised TypeError before any work happened.  The script had therefore never run at all,
    # despite being recorded as "built, smoked, not run".
    jobs = [(s, c, by_crop.get(c, set())) for s in (0, 1) for c in cached_crops(s)]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(replay_one, jobs):
            rows.append(r)
            print(json.dumps(r), flush=True)

    from tracking_cellmot.metrics import ADJUSTMENT_ALPHA, SCORE_DIVISION_WEIGHT  # noqa

    def agg(pfx: str, sel) -> dict:
        rs = [r for r in rows if sel(r)]
        num = den = 0.0
        dtp = dfp = dfn = 0
        for r in rs:
            tp, fp, fn = r[f"{pfx}edge_tp"], r[f"{pfx}edge_fp"], r[f"{pfx}edge_fn"]
            w = tp + fp + fn
            j = tp / w if w else 0.0
            # 2026-07-31 DEFECT FIX.  This was
            #     abs(N_pred_arm - N_pred_baseline) / N_pred_baseline
            # which is identically ZERO for an edges-only replay (the node set never moves),
            # so every arm was scored with an UNPENALISED multiplier and the resulting
            # adj_edge_jaccard was not comparable to the published 0.7595 / 0.6490 anchors.
            # The canonical multiplier is signed and defined against N_est (metrics.py:440),
            # NOT absolute and NOT against our own baseline.  With this fix the baseline
            # arm reproduces the published anchors exactly (44b6 0.759549, 6bba 0.648965).
            ratio = (r[f"{pfx}num_pred_nodes"] - r["n_est"]) / max(r["n_est"], 1)
            num += w * max(0.0, j * (1.0 - ADJUSTMENT_ALPHA * ratio))
            den += w
            dtp += r[f"{pfx}division_tp"]; dfp += r[f"{pfx}division_fp"]
            dfn += r[f"{pfx}division_fn"]
        adj = num / den if den else 0.0
        dj = dtp / (dtp + dfp + dfn) if (dtp + dfp + dfn) else float("nan")
        return {"adj_edge_jaccard": adj, "division_jaccard": dj,
                "composite": adj + SCORE_DIVISION_WEIGHT * dj,
                "division_tp": dtp, "division_fp": dfp, "division_fn": dfn,
                "n_crops": len(rs)}

    res = {"cfg": CFG, "cfg_hash": CFG_HASH, "admit_file": a.admit,
           "n_admitted_total": int(adm.height),
           "selection": "gate_rank0 (GT-free) -- oracle selection REPLACED",
           "pooled": {"baseline": agg("b_", lambda r: True),
                      "gated": agg("t_", lambda r: True)},
           "per_family": {}}
    for fam in ("44b6", "6bba"):
        res["per_family"][fam] = {
            "baseline": agg("b_", lambda r, f=fam: r["crop"].startswith(f)),
            "gated": agg("t_", lambda r, f=fam: r["crop"].startswith(f))}
    for k in ("pooled",):
        res[k]["delta_composite"] = (res[k]["gated"]["composite"]
                                     - res[k]["baseline"]["composite"])
        res[k]["delta_adj_edge"] = (res[k]["gated"]["adj_edge_jaccard"]
                                    - res[k]["baseline"]["adj_edge_jaccard"])
    for fam, v in res["per_family"].items():
        v["delta_composite"] = v["gated"]["composite"] - v["baseline"]["composite"]
        v["delta_adj_edge"] = (v["gated"]["adj_edge_jaccard"]
                               - v["baseline"]["adj_edge_jaccard"])
    res["rows"] = rows
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2, default=float))


if __name__ == "__main__":
    main()
