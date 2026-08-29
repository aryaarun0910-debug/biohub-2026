r"""THE DIVISION METRIC ADAPTER - and the proof it reproduces the deployed counts.

LEVER-0040 / PKT-0032 stage 4. This connects a VERIFIER DECISION ("reject this fork") to the
OFFICIAL division metric, and it is not shipped until it reproduces a number we already know.

WHAT THE ADAPTER IS FOR
-----------------------
A verifier does not emit a score - it emits a decision to un-fork a predicted node. The official
metric charges divisions through ``tracking_cellmot.division_metrics.evaluate_divisions``, whose
FP tally is literally ``len(score_divisions(...).fp_forks)``. So a rejection changes the metric
only through the graph. The adapter therefore has TWO paths and they must agree:

  ACCOUNTING PATH   arithmetic on the scorer's own fork sets:
                      tp' = tp - |rejected & tp_forks|
                      fp' = fp - |rejected & fp_forks|
                      fn' = fn + |rejected & tp_forks|
                    Rejecting a fork the metric IGNORES changes nothing - and that is the point
                    the adapter exists to make visible.

  EXACT PATH        delete the chosen child edge, re-run the OFFICIAL scorer on the edited graph,
                    read TP/FP/FN and the complete per-crop metric row back out.

The accounting path is a prediction. The exact path is the truth. ``compare`` runs both and a
divergence is REPORTED, never smoothed - a silent no-op here would let a verifier be scored by
arithmetic that the scorer does not honour.

WHY AN EMPTY REJECTION SET IS THE FIRST RUNG
--------------------------------------------
PKT-0032 falsifier (c): "the metric adapter does not reproduce the official division TP/FP/FN on
a known control". So rung 1 is the identity case - zero rejections must reproduce the DEPLOYED
fold-0 division counts recorded in FACT-0375's scope, on
``C:/temp/p28_f0/loeo_split0_champion.csv.gz``, EXACTLY. Anything else and the adapter is not
shipped and the lane stops here.

Rung 2 rejects every FP fork the scorer names, and re-scores exactly. It bounds the whole lever:
it is what a PERFECT verifier buys, including the edge-term cost of the deleted child edges,
which the division-Jaccard arithmetic alone never shows. Rung 2 is an ORACLE - it uses the
scorer's own FP labels to choose - and is reported as a ceiling, never as an achievable result.

WHICH CHILD EDGE A REJECTION DELETES
------------------------------------
Un-forking needs a choice, and the choice must be GT-free or the ceiling is not interpretable.
The ``weakest_child`` rule: drop the child that does not persist (has no successor) before one
that does; break ties by the LARGER mother-daughter displacement; break remaining ties by node
id. Deterministic, computable at inference, and stated here rather than tuned later.

Usage
-----
  .venv\Scripts\python.exe scripts\win_bet\divverify_adapter.py prove ^
      --csv C:/temp/p28_f0/loeo_split0_champion.csv.gz --fold 0 --out-dir C:/temp/divverify
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")
ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SCALE = (1.625, 0.40625, 0.40625)
MAX_DISTANCE = 7.0

# The deployed fold-0 division counts this adapter must reproduce. Recorded in FACT-0375's scope
# block (control arm) and independently in FACT-0371's f0.deployed block. Stated here as a TEST
# EXPECTATION, which is the one place a number may be written down outside the registry: if the
# registry ever moves, this test fails loudly instead of passing on a stale constant.
DEPLOYED_F0_DIVISION = {"division_tp": 5, "division_fp": 67, "division_fn": 21}


def _ea_atlas():
    spec = importlib.util.spec_from_file_location("ea_atlas", ROOT / "scripts" / "win_bet" / "ea_atlas.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================ the adapter
class DivisionMetricAdapter:
    """Maps a set of rejected forks (in SUBMISSION node ids) to official division counts.

    Constructed per crop from the export frame and the GT. Holds the scorer's own fork sets so a
    caller never has to guess which forks the metric charges.
    """

    def __init__(self, name: str, sub: pl.DataFrame, gt_geff: Path, ea, dm, mods):
        load_graph, evaluate, estimated_nodes, node_recall, per_sample_metrics = mods
        self.name = name
        self.sub = sub
        self.gt_geff = gt_geff
        self.ea, self.dm = ea, dm
        self._mods = mods
        self.gt = load_graph(gt_geff)

        g, i2s = ea.build_graph(sub)
        self.i2s = i2s
        ds = dm.score_divisions(g, self.gt, SCALE, MAX_DISTANCE)
        er = evaluate(g, self.gt, scale=SCALE, max_distance=MAX_DISTANCE)
        self.base = {"division_tp": int(sum(ds.scores.values())),
                     "division_fp": int(len(ds.fp_forks)),
                     "division_fn": int(len(ds.scores) - sum(ds.scores.values()))}
        official = {"division_tp": er.division_tp, "division_fp": er.division_fp,
                    "division_fn": er.division_fn}
        if self.base != official:
            raise RuntimeError(f"{name}: adapter baseline {self.base} != official scorer {official}")
        self.tp_forks = {int(i2s[int(f)]) for f in ds.tp_forks}
        self.fp_forks = {int(i2s[int(f)]) for f in ds.fp_forks}
        self.all_forks = {int(i2s[int(n)]) for n in g.node_ids() if g.out_degree(n) >= 2}
        self.ignored_forks = self.all_forks - self.tp_forks - self.fp_forks

        nodes = sub.filter(pl.col("row_type") == "node").sort("node_id")
        ids = nodes["node_id"].to_numpy().astype(np.int64)
        zyx = nodes.select(["z", "y", "x"]).to_numpy().astype(np.float64) * np.asarray(SCALE)
        self.pos = {int(v): zyx[i] for i, v in enumerate(ids)}
        e = sub.filter(pl.col("row_type") == "edge")
        self.edges = set(zip(e["source_id"].to_list(), e["target_id"].to_list()))
        self.children_of: dict[int, list[int]] = defaultdict(list)
        for a, b in self.edges:
            self.children_of[int(a)].append(int(b))

    # ---------------------------------------------------------------- accounting path
    def predict_counts(self, rejected: set[int]) -> dict:
        """What the metric SHOULD read after these rejections, by arithmetic on the fork sets."""
        lost_tp = len(rejected & self.tp_forks)
        cut_fp = len(rejected & self.fp_forks)
        inert = len(rejected & self.ignored_forks)
        return {
            "division_tp": self.base["division_tp"] - lost_tp,
            "division_fp": self.base["division_fp"] - cut_fp,
            "division_fn": self.base["division_fn"] + lost_tp,
            "rejections_that_cut_a_charged_fp": cut_fp,
            "rejections_that_cost_a_true_positive": lost_tp,
            "rejections_the_metric_ignores": inert,
        }

    # ---------------------------------------------------------------- the edit
    def weakest_child(self, fork: int) -> int:
        """GT-free choice of which child edge a rejection deletes. Deterministic."""
        kids = self.children_of.get(int(fork), [])
        if len(kids) < 2:
            raise RuntimeError(f"{self.name}: fork {fork} has {len(kids)} children - not a fork")
        pm = self.pos[int(fork)]

        def key(b):
            persists = 1 if self.children_of.get(int(b)) else 0
            return (persists, -float(np.linalg.norm(self.pos[int(b)] - pm)), int(b))

        return int(sorted(kids, key=key)[0])

    def edited_frame(self, rejected: set[int]) -> pl.DataFrame:
        drop = {(int(f), self.weakest_child(f)) for f in rejected}
        missing = drop - self.edges
        if missing:
            raise RuntimeError(f"{self.name}: {len(missing)} edges to delete are not in the export")
        return _rewrite_edges(self.sub, self.edges - drop)

    # ---------------------------------------------------------------- exact path
    def exact_counts(self, rejected: set[int]) -> dict:
        """Re-run the OFFICIAL scorer on the edited graph. The truth the accounting predicts."""
        load_graph, evaluate, estimated_nodes, node_recall, per_sample_metrics = self._mods
        frame = self.sub if not rejected else self.edited_frame(rejected)
        g, _ = self.ea.build_graph(frame)
        er = evaluate(g, self.gt, scale=SCALE, max_distance=MAX_DISTANCE)
        rec = node_recall(g, self.gt) if g.num_edges() and g.num_nodes() else 0.0
        row = per_sample_metrics(er, estimated_nodes(self.gt_geff), rec)
        return {"dataset": self.name, **row}

    def compare(self, rejected: set[int]) -> dict:
        pred = self.predict_counts(rejected)
        exact = self.exact_counts(rejected)
        agree = all(pred[k] == exact[k] for k in ("division_tp", "division_fp", "division_fn"))
        return {"predicted": pred, "exact": exact, "accounting_agrees": bool(agree)}


def _rewrite_edges(sub: pl.DataFrame, edges: set[tuple[int, int]]) -> pl.DataFrame:
    """Nodes unchanged, edge rows rebuilt. Same layout contract as
    div_reach_steal.rewrite_crop_edges, including its degree-invariant guard (audit F4)."""
    nodes = sub.filter(pl.col("row_type") == "node")
    dataset = nodes["dataset"][0]
    out_deg: dict[int, int] = defaultdict(int)
    in_deg: dict[int, int] = defaultdict(int)
    for s, t in edges:
        out_deg[s] += 1
        in_deg[t] += 1
    bad_out = [n for n, d in out_deg.items() if d > 2]
    bad_in = [n for n, d in in_deg.items() if d > 1]
    if bad_out or bad_in:
        raise RuntimeError(f"{dataset}: degree invariant violated - out>2 {bad_out[:5]}, in>1 {bad_in[:5]}")
    src, tgt = zip(*sorted(edges)) if edges else ((), ())
    edge_rows = pl.DataFrame({
        "id": pl.Series([0] * len(src), dtype=pl.Int64),
        "dataset": [dataset] * len(src),
        "row_type": ["edge"] * len(src),
        "node_id": pl.Series([-1] * len(src), dtype=pl.Int64),
        "t": pl.Series([-1] * len(src), dtype=pl.Int64),
        "z": pl.Series([-1] * len(src), dtype=nodes["z"].dtype),
        "y": pl.Series([-1] * len(src), dtype=nodes["y"].dtype),
        "x": pl.Series([-1] * len(src), dtype=nodes["x"].dtype),
        "source_id": pl.Series(list(src), dtype=pl.Int64),
        "target_id": pl.Series(list(tgt), dtype=pl.Int64),
    })
    return pl.concat([nodes.select(edge_rows.columns), edge_rows], how="vertical")


# ============================================================ the proof
def cmd_prove(args) -> int:
    ea = _ea_atlas()
    from biotrack.metric import estimated_nodes, load_graph
    from biotrack.submission import read_submission
    from tracking_cellmot import division_metrics as dm
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics, summarise
    mods = (load_graph, evaluate, estimated_nodes, node_recall, per_sample_metrics)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = read_submission(ea.open_csv(Path(args.csv)))
    names = sorted(df["dataset"].unique().to_list())
    if args.stride:
        names = names[::args.stride]
    if args.max_crops:
        names = names[: args.max_crops]

    rung1_rows, rung2_rows = [], []
    totals = defaultdict(int)
    disagreements: list[dict] = []
    fork_pop = defaultdict(int)
    t0 = time.time()
    for i, name in enumerate(names, 1):
        gt_geff = Path(args.gt_dir) / f"{name}.geff"
        if not gt_geff.exists():
            raise RuntimeError(f"{name}: no GT at {gt_geff}")
        ad = DivisionMetricAdapter(name, df.filter(pl.col("dataset") == name), gt_geff, ea, dm, mods)
        fork_pop["all"] += len(ad.all_forks)
        fork_pop["tp"] += len(ad.tp_forks)
        fork_pop["fp"] += len(ad.fp_forks)
        fork_pop["ignored"] += len(ad.ignored_forks)

        # rung 1 - the identity case. Zero rejections must reproduce the deployed counts.
        r1 = ad.compare(set())
        rung1_rows.append(r1["exact"])
        if not r1["accounting_agrees"]:
            disagreements.append({"crop": name, "rung": 1, **r1})

        # rung 2 - reject every FP fork the scorer names (an ORACLE ceiling).
        r2 = ad.compare(set(ad.fp_forks))
        rung2_rows.append(r2["exact"])
        if not r2["accounting_agrees"]:
            disagreements.append({"crop": name, "rung": 2,
                                  "predicted": r2["predicted"],
                                  "exact": {k: r2["exact"][k] for k in
                                            ("division_tp", "division_fp", "division_fn")}})
        for k in ("division_tp", "division_fp", "division_fn"):
            totals[f"r1_{k}"] += int(r1["exact"][k])
            totals[f"r2_{k}"] += int(r2["exact"][k])
        print(f"  [{i}/{len(names)}] {name} base {r1['exact']['division_tp']}/"
              f"{r1['exact']['division_fp']}/{r1['exact']['division_fn']} -> ceiling "
              f"{r2['exact']['division_tp']}/{r2['exact']['division_fp']}/{r2['exact']['division_fn']} "
              f"forks {len(ad.all_forks)} (tp {len(ad.tp_forks)}, fp {len(ad.fp_forks)}, "
              f"ignored {len(ad.ignored_forks)}) ({time.time() - t0:.0f}s)", flush=True)

    control = dict(summarise(rung1_rows))
    ceiling = dict(summarise(rung2_rows))
    for k in ("edge_tp", "edge_fp", "edge_fn"):
        control[k] = int(sum(r[k] for r in rung1_rows))
        ceiling[k] = int(sum(r[k] for r in rung2_rows))

    measured = {k: int(control[k]) for k in ("division_tp", "division_fp", "division_fn")}
    rung1_pass = (args.fold != 0) or (measured == DEPLOYED_F0_DIVISION)

    report = {
        "schema_version": 1,
        "heartbeat": "DIVVERIFY_ADAPTER_PROOF_COMPLETE",
        "fold": args.fold, "csv": str(args.csv), "n_crops": len(names),
        "rung1_identity": {
            "expected_from_registry": DEPLOYED_F0_DIVISION if args.fold == 0 else None,
            "measured": measured,
            "exact_match": bool(rung1_pass),
            "note": ("PKT-0032 falsifier (c) fires if this is false" if args.fold == 0
                     else "no registry expectation for this fold; identity checked structurally only"),
        },
        "accounting_vs_exact": {
            "crops_checked": len(names) * 2,
            "disagreements": len(disagreements),
            "detail": disagreements[:20],
        },
        "fork_population": {k: int(v) for k, v in fork_pop.items()},
        "rung2_perfect_verifier_ceiling": {
            "division": {k: int(ceiling[k]) for k in ("division_tp", "division_fp", "division_fn")},
            "division_jaccard": {"control": float(control["division_jaccard"]),
                                 "ceiling": float(ceiling["division_jaccard"])},
            "edge_totals": {"control": {k: control[k] for k in ("edge_tp", "edge_fp", "edge_fn")},
                            "ceiling": {k: ceiling[k] for k in ("edge_tp", "edge_fp", "edge_fn")}},
            "edge_jaccard": {"control": float(control["edge_jaccard"]),
                             "ceiling": float(ceiling["edge_jaccard"])},
            "adj_edge_jaccard": {"control": float(control["adj_edge_jaccard"]),
                                 "ceiling": float(ceiling["adj_edge_jaccard"])},
            "score": {"control": float(control["score"]), "ceiling": float(ceiling["score"]),
                      "delta": float(ceiling["score"] - control["score"])},
            "is_an_oracle": True,
            "note": ("rejects every fork the OFFICIAL scorer calls a false positive, deleting the "
                     "weakest child edge of each. Division counts and the edge cost are both real; "
                     "the SELECTION is oracle, so this is a ceiling, not a result."),
        },
    }
    (out / f"adapter_proof_f{args.fold}.json").write_text(json.dumps(report, indent=2, default=float),
                                                          encoding="utf-8")
    pl.DataFrame(rung1_rows).write_parquet(out / f"adapter_control_f{args.fold}.parquet")
    pl.DataFrame(rung2_rows).write_parquet(out / f"adapter_ceiling_f{args.fold}.parquet")
    print(json.dumps(report, indent=2, default=float))
    if not rung1_pass:
        print("\nADAPTER NOT SHIPPED: rung 1 did not reproduce the deployed fold-0 division "
              "counts. PKT-0032 falsifier (c) FIRES.", flush=True)
        return 2
    print("\nADAPTER PROOF PASSED rung 1"
          + ("" if not disagreements else
             f" BUT the accounting path disagreed with the exact path on {len(disagreements)} "
             f"crop-rungs - use the exact path only"), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prove")
    p.add_argument("--csv", required=True)
    p.add_argument("--fold", type=int, required=True, choices=(0, 1))
    p.add_argument("--gt-dir", default=str(ROOT / "data" / "train"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-crops", type=int)
    p.add_argument("--stride", type=int)
    p.set_defaults(func=cmd_prove)
    args = ap.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
