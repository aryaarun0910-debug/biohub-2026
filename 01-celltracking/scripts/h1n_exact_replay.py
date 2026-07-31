"""H1-N -- EXACT pooled replay of a composition's admission list.

This is the deployable form of the H0c cascade with the GT ORACLE REMOVED. `shortlist`
and `CFG` are imported VERBATIM from `phaseb_h0c_replay` (config hash 04eeac97500d);
nothing about generation or ranking is re-implemented or retuned. The only change is the
source of `retained`:

    phaseb_h0c_replay : retained[m] = the mother's TRUE GT daughter pair, if it survived
                        into the top-3 shortlist                       (ORACLE, not submittable)
    this script       : retained[m] = the mother's RANK-0 shortlist pair, for every m on
                        the admission list                             (GT-FREE)

GT is still loaded, but ONLY to score. It never touches selection.

PRIMARY OUTPUT is the POOLED composite: one combined `summarise()` over all 199 crops,
exactly as `tracking_cellmot.metrics.summarise` weights them. Per-family scores are
printed as diagnostics only.

COMPUTE POLICY. A 199-crop run is heavy and is NOT started here. Use `--crops N` to smoke
a handful of crops; the full command is printed at the end and must be authorised.

Usage (smoke):
  .venv\\Scripts\\python.exe scripts\\h1n_exact_replay.py --admissions <compositions.json> ^
      --arm C --crops 3 --workers 3 --out <smoke.json>
Usage (full, REQUIRES GO-AHEAD):
  .venv\\Scripts\\python.exe scripts\\h1n_exact_replay.py --admissions <compositions.json> ^
      --arm C --workers 6 --out <exact.json>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import polars as pl  # noqa: E402

from phaseb_d0p_proposer import cached_crops, load_e0c_tables  # noqa: E402
from phaseb_h0c_replay import CFG, CFG_HASH, shortlist  # noqa: E402

_ADMIT: dict[str, set[int]] = {}


def _init(admit: dict[str, list[int]]) -> None:
    global _ADMIT
    _ADMIT = {k: set(v) for k, v in admit.items()}


def replay_one(args) -> dict:
    split, crop = args
    from biotrack.metric import score_pred_graph
    from biotrack.submission import submission_to_graphs

    sub, t, pos, edges, nd = load_e0c_tables(split, crop)
    prop, _ = shortlist(sub, t, pos, edges)              # GT-FREE, frozen
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

    # ---- GT-FREE selection: admission list -> the mother's own rank-0 pair -----
    admitted = _ADMIT.get(crop, set())
    retained = {}
    for m in admitted:
        cand = prop.get(int(m))
        if cand:
            retained[int(m)] = cand[0][1]                # rank-0 (d1, d2)

    es = {(int(a), int(b)) for a, b in edges}
    par, ch_ = {}, {}
    for a, b in es:
        ch_.setdefault(a, set()).add(b)
        par.setdefault(b, set()).add(a)
    for m in [n for n, k in ch_.items() if len(k) >= 2]:      # suppress-all
        keep = min(ch_[m])
        for k in list(ch_[m]):
            if k != keep:
                es.discard((m, k)); ch_[m].discard(k); par.get(k, set()).discard(m)
    steals, used = 0, set()
    for pM, (p1, p2) in retained.items():                     # add-replace
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

    ks = ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn",
          "node_recall", "num_pred_nodes")
    return {"split": split, "crop": crop, "admitted": len(admitted),
            "reconstructed": len(retained), "steals": steals,
            **{f"b_{k}": base[k] for k in ks}, **{f"t_{k}": treat[k] for k in ks}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--admissions", required=True, help="h1n_compose.py output json")
    ap.add_argument("--arm", default="C", choices=["A", "B", "C"])
    ap.add_argument("--crops", type=int, default=0, help="0 = ALL 199 (compute-gated)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    res = json.loads(Path(a.admissions).read_text())
    arm = res["arms"][a.arm]
    admit: dict[str, list[int]] = {}
    for fam in ("44b6", "6bba"):
        for key in arm["families"][fam]["admitted_keys"]:
            crop, mother, _t = key.split(":")
            admit.setdefault(crop, []).append(int(mother))
    print(f"H1-N exact replay  arm={a.arm} ({arm['name']})")
    print(f"  frozen proposer cfg hash {CFG_HASH}  prereg {res['prereg_hash']}")
    print(f"  admissions: {sum(len(v) for v in admit.values())} over {len(admit)} crops")

    from biotrack.metric import estimated_nodes, per_sample_metrics
    from tracking_cellmot.metrics import EvaluationResult, summarise

    jobs = []
    for fold in (0, 1):
        cs = [c for c in cached_crops(fold) if c in admit] or cached_crops(fold)
        if a.crops:
            cs = cs[:a.crops]
        jobs += [(fold, c) for c in cs]
    print(f"  crops to run: {len(jobs)}"
          + ("  [SMOKE -- not the pooled number]" if a.crops else "  [FULL POOLED]"))

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(admit,)) as ex:
        rows = list(ex.map(replay_one, jobs))
    dt = time.time() - t0

    ks = ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn",
          "num_pred_nodes")
    b_all, t_all, by_fam = [], [], {0: ([], []), 1: ([], [])}
    for r in rows:
        ne = estimated_nodes(str(ROOT / "data" / "train" / f"{r['crop']}.geff"))
        b = per_sample_metrics(EvaluationResult(*[r[f"b_{k}"] for k in ks]), ne, r["b_node_recall"])
        t_ = per_sample_metrics(EvaluationResult(*[r[f"t_{k}"] for k in ks]), ne, r["t_node_recall"])
        b_all.append(b); t_all.append(t_)
        by_fam[r["split"]][0].append(b); by_fam[r["split"]][1].append(t_)

    sb, st = summarise(b_all), summarise(t_all)          # ONE combined pooled summarise
    out = {"arm": a.arm, "cfg_hash": CFG_HASH, "prereg_hash": res["prereg_hash"],
           "smoke": bool(a.crops), "n_crops": len(jobs), "seconds": dt,
           "pooled_baseline": sb, "pooled_treatment": st,
           "pooled_delta": st["score"] - sb["score"],
           "families": {}, "per_crop": rows}
    print(f"\n  POOLED baseline   composite={sb['score']:.5f} divJ={sb['division_jaccard']:.5f} "
          f"(TP{sb['division_tp']}/FP{sb['division_fp']}/FN{sb['division_fn']})")
    print(f"  POOLED treatment  composite={st['score']:.5f} divJ={st['division_jaccard']:.5f} "
          f"(TP{st['division_tp']}/FP{st['division_fp']}/FN{st['division_fn']})")
    print(f"  POOLED DELTA      {st['score']-sb['score']:+.5f}")
    for fold, fam in ((0, "44b6"), (1, "6bba")):
        if not by_fam[fold][0]:
            continue
        fb, ft = summarise(by_fam[fold][0]), summarise(by_fam[fold][1])
        out["families"][fam] = {"baseline": fb, "treatment": ft,
                                "delta": ft["score"] - fb["score"]}
        print(f"    [diag] {fam}: {fb['score']:.5f} -> {ft['score']:.5f} "
              f"({ft['score']-fb['score']:+.5f})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\n  {dt:.1f}s for {len(jobs)} crops = {dt/max(len(jobs),1):.1f}s/crop; "
          f"199 crops on {a.workers} workers ~ {199*dt/max(len(jobs),1)/60:.0f} min")
    print("\nFULL POOLED COMMAND (compute-gated, NOT run):")
    print(f"  .venv\\Scripts\\python.exe scripts\\h1n_exact_replay.py "
          f"--admissions {a.admissions} --arm {a.arm} --workers 6 --out <exact_{a.arm}.json>")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
