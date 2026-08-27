r"""Division reach and the targeted daughter steal - LEVER-0025 instrument (PKT-0021).

WHY THIS EXISTS
---------------
FACT-0295 (2026-08-26) found two things the deployed proposer never uses, and measured them with
an UNCOMMITTED analysis:

  (1) the scorer accepts a fork at the node matched to the GT divider OR at its predecessor, and
      daughter evidence may come from the GT child OR a grandchild
      (vendor tracking_cellmot/division_metrics.py::_is_strongly_connected_division);
  (2) when the second daughter already has a parent, that parent edge CANNOT be a GT edge - the
      daughter's only GT parent is the mother - so stealing it is free on the edge term.

This file commits the measurement (stage A, `census`) and adds the scorer round-trip the fact
lacked (stage B, `oracle`): apply the GT-guided edits to an exported champion-control CSV, score
control and oracle with the official per-crop metrics, and report the paired delta with a crop
bootstrap. The oracle is a CEILING - it uses GT to choose the edits. A GT-free rule is stage C
and lives behind the same scoring path (`score_csv`).

Every count reconciles with the scorer: `scored` uses score_divisions() itself, and reach uses
the scorer's own _matched_division_nodes() on the scorer's own per-division matching.

Usage
-----
  .\.venv\Scripts\python.exe scripts\win_bet\div_reach_steal.py census ^
      --csv C:/temp/p20_relink_sweep_f0/sweep_pen_off.csv.gz --tag f0 --out-dir C:/temp/div_reach/f0
  .\.venv\Scripts\python.exe scripts\win_bet\div_reach_steal.py oracle ^
      --csv C:/temp/p19_relink_sweep_f1/sweep_pen_off.csv.gz --tag f1 --out-dir C:/temp/div_reach/f1
  (add --max-crops N or --stride K for a pilot; a pilot is not scientific evidence)
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
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")
ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))

SCALE = (1.625, 0.40625, 0.40625)   # z, y, x um per level-0 voxel (biotrack.metric.DEFAULT_SCALE)


# ============================================================================ pure planner
def lineage_of(b, daughter_ids, children_of):
    """Which GT daughter lineages a t+1 node ``b`` gives evidence for.

    Direct match first (``b`` itself is matched to a GT child); otherwise via its successors
    (a successor matched to a grandchild). Mirrors the scorer's pred-lineage
    ``{child, *successors(child)}`` intersected with each GT lineage's matched set.
    """
    direct = {i for i, ids in enumerate(daughter_ids) if b in ids}
    if direct:
        return direct, "direct"
    via = {i for i, ids in enumerate(daughter_ids) for c in children_of.get(b, ()) if c in ids}
    return via, ("grand" if via else "none")


def plan_division(*, t_div, divider, parent_ids, daughter_ids, node_to_gt,
                  parent_of, children_of, t_of, pos_um):
    """The cheapest GT-guided edit set that makes one GT division scorable, or None.

    Candidate forks, in order: nodes at ``t_div`` matched to the divider itself (route
    ``direct``), then nodes at ``t_div`` whose predecessor is on the parent side - i.e. matched to
    the GT grandparent (route ``pred``, the scorer's predecessor allowance). A fork must currently
    have out-degree <= 1, and an existing child must already give evidence for one lineage
    (a fork with a wrong child is left alone). Each missing lineage is filled by a t+1 node that
    gives evidence for it - a direct match preferred over grandchild evidence, then nearest to
    the fork; its current parent edge, if any, is removed (the steal) and fork -> daughter added.

    Returns {"route", "fork", "keep", "add": [(fork, b)], "remove": [(parent, b)],
             "daughter_kinds": ["direct"|"grand", ...]} or None.
    """
    t1 = t_div + 1
    cands: dict[int, list[tuple[str, int]]] = {}
    for i, ids in enumerate(daughter_ids):
        found: list[tuple[str, int]] = []
        for p in ids:
            if t_of.get(p) == t1:
                found.append(("direct", p))
            elif t_of.get(p) == t1 + 1:
                par = parent_of.get(p)
                if par is not None and t_of.get(par) == t1:
                    found.append(("grand", par))
        seen: set[int] = set()
        uniq: list[tuple[str, int]] = []
        for kind, b in sorted(found, key=lambda kb: 0 if kb[0] == "direct" else 1):
            if b not in seen:
                seen.add(b)
                uniq.append((kind, b))
        cands[i] = uniq

    forks: list[tuple[str, int]] = []
    for p in sorted(parent_ids):
        if t_of.get(p) == t_div and node_to_gt.get(p) == divider:
            forks.append(("direct", p))
    for p in sorted(parent_ids):
        if t_of.get(p) == t_div - 1:
            for x in children_of.get(p, ()):
                if t_of.get(x) == t_div and ("direct", x) not in forks and ("pred", x) not in forks:
                    forks.append(("pred", x))

    def dist(a, b):
        pa, pb = pos_um.get(a), pos_um.get(b)
        if pa is None or pb is None:
            return float("inf")
        return float(np.linalg.norm(np.asarray(pa, dtype=float) - np.asarray(pb, dtype=float)))

    for route, fork in forks:
        existing = list(children_of.get(fork, ()))
        if len(existing) >= 2:
            continue
        covered: set[int] = set()
        keep: list[int] = []
        if existing:
            li, _kind = lineage_of(existing[0], daughter_ids, children_of)
            if not li:
                continue
            covered |= li
            keep = [existing[0]]
        adds: list[tuple[int, int]] = []
        removes: list[tuple[int, int]] = []
        kinds: list[str] = []
        used: set[int] = set(keep)
        for i in sorted(cands):
            if i in covered:
                continue
            if len(covered) + len(adds) >= 2:
                break
            best = None
            for kind, b in cands[i]:
                if b in used or b == fork:
                    continue
                key = (0 if kind == "direct" else 1, dist(fork, b))
                if best is None or key < best[0]:
                    best = (key, kind, b)
            if best is None:
                continue
            _key, kind, b = best
            used.add(b)
            adds.append((fork, b))
            kinds.append(kind)
            parent = parent_of.get(b)
            if parent is not None:
                removes.append((parent, b))
        if len(covered) + len(adds) < 2:
            continue
        return {"route": route, "fork": fork, "keep": keep, "add": adds,
                "remove": removes, "daughter_kinds": kinds}
    return None


def apply_plans(edges: set[tuple[int, int]], plans: list[dict]) -> tuple[set[tuple[int, int]], list[dict]]:
    """Apply plans sequentially, refusing any that would break in-degree <= 1 / out-degree <= 2
    on the CURRENT edge set (plans were computed on the original graph and may collide)."""
    edges = set(edges)
    out_deg: dict[int, int] = defaultdict(int)
    parent: dict[int, int] = {}
    for s, t in edges:
        out_deg[s] += 1
        parent[t] = s
    applied: list[dict] = []
    for plan in plans:
        removes = [(p, b) for p, b in plan["remove"]]
        adds = [(f, b) for f, b in plan["add"]]
        if any((p, b) not in edges for p, b in removes):
            continue
        removed_targets = {b for _p, b in removes}
        if any(b in parent and b not in removed_targets for _f, b in adds):
            continue
        by_fork: dict[int, int] = defaultdict(int)
        for f, _b in adds:
            by_fork[f] += 1
        if any(out_deg[f] + n > 2 for f, n in by_fork.items()):
            continue
        for p, b in removes:
            edges.discard((p, b))
            out_deg[p] -= 1
            parent.pop(b, None)
        for f, b in adds:
            edges.add((f, b))
            out_deg[f] += 1
            parent[b] = f
        applied.append(plan)
    return edges, applied


# ============================================================================ scorer glue
def _ea_atlas():
    spec = importlib.util.spec_from_file_location("ea_atlas", ROOT / "scripts" / "win_bet" / "ea_atlas.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scorer():
    import tracksdata as td
    from biotrack.metric import MAX_DISTANCE, estimated_nodes, load_graph
    from tracking_cellmot import division_metrics as dm
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics, summarise
    return td, MAX_DISTANCE, estimated_nodes, load_graph, dm, evaluate, node_recall, per_sample_metrics, summarise


def analyse_crop(name: str, sub: pl.DataFrame, gt_geff: Path, ea, sc) -> tuple[list[dict], list[dict]]:
    """Census rows and GT-guided plans (in SUBMISSION node ids) for one crop."""
    td, MAX_DISTANCE, _est, load_graph, dm, *_ = sc
    K = td.DEFAULT_ATTR_KEYS
    g, i2s = ea.build_graph(sub)
    gt = load_graph(gt_geff)
    gt_divs = dm.extract_divisions(gt)
    if not gt_divs:
        return [], []
    matched = dm.match_divisions(g, gt, SCALE, MAX_DISTANCE)
    scores = dm.score_divisions(g, gt, SCALE, MAX_DISTANCE).scores

    na = g.node_attrs(attr_keys=[K.NODE_ID, "t", "z", "y", "x"]).to_pandas()
    t_of = dict(zip(na[K.NODE_ID].astype(int), na["t"].astype(int)))
    pos_um = {int(n): (z * SCALE[0], y * SCALE[1], x * SCALE[2])
              for n, z, y, x in zip(na[K.NODE_ID], na["z"], na["y"], na["x"])}
    parent_of: dict[int, int] = {}
    children_of: dict[int, list[int]] = defaultdict(list)
    if g.num_edges() > 0:
        ea_edges = g.edge_attrs(attr_keys=[K.EDGE_SOURCE, K.EDGE_TARGET]).to_pandas()
        for s, t in zip(ea_edges[K.EDGE_SOURCE].astype(int), ea_edges[K.EDGE_TARGET].astype(int)):
            children_of[s].append(t)
            if t in parent_of:
                raise RuntimeError(f"{name}: node {t} has two parents in the export")
            parent_of[t] = s

    full = dm._match_full(g, gt, SCALE, MAX_DISTANCE)
    fa = dm._matched_node_attrs(full)
    full_gt = dict(zip(fa[K.NODE_ID].to_list(), fa[K.MATCHED_NODE_ID].to_list()))
    gt_ea = gt.edge_attrs(attr_keys=[K.EDGE_SOURCE, K.EDGE_TARGET]).to_pandas()
    gt_edges = set(zip(gt_ea[K.EDGE_SOURCE].astype(int), gt_ea[K.EDGE_TARGET].astype(int)))
    gt_na = gt.node_attrs(attr_keys=[K.NODE_ID, "t"]).to_pandas()
    gt_t = dict(zip(gt_na[K.NODE_ID].astype(int), gt_na["t"].astype(int)))

    rows: list[dict] = []
    plans: list[dict] = []
    for divider, gt_div in gt_divs.items():
        mp = matched[divider]
        attrs = dm._matched_node_attrs(mp)
        node_to_gt = dict(zip(attrs[K.NODE_ID].to_list(), attrs[K.MATCHED_NODE_ID].to_list()))
        children = [int(c) for c in gt_div.successors(divider)]
        grandparents = [int(p) for p in gt_div.predecessors(divider)]
        mn = dm._matched_division_nodes(attrs, gt_div, divider)
        gt_vals = set(node_to_gt.values())
        naive = (divider in gt_vals) and all(c in gt_vals for c in children)
        row = {
            "dataset": name, "divider": int(divider), "t_div": int(gt_t[divider]),
            "n_children": len(children), "has_grandparent": bool(grandparents),
            "divider_matched": divider in gt_vals,
            "grandparent_matched": any(gp in gt_vals for gp in grandparents),
            "children_matched": sum(c in gt_vals for c in children),
            "naive_reach": bool(naive), "legal_reach": mn is not None,
            "scored": int(scores.get(divider, 0)) == 1,
            "status": None, "route": None, "n_add": 0, "n_remove": 0, "daughter_kinds": "",
            "removed_edge_is_full_tp": False, "removed_parent_outdeg_before": -1,
            "second_daughter_state": None,
        }
        if mn is None:
            row["status"] = "illegal"
        elif row["scored"]:
            row["status"] = "scored"
        else:
            parent_ids, daughter_ids = mn
            plan = plan_division(t_div=row["t_div"], divider=divider, parent_ids=set(parent_ids),
                                 daughter_ids=[set(d) for d in daughter_ids], node_to_gt=node_to_gt,
                                 parent_of=parent_of, children_of=children_of, t_of=t_of, pos_um=pos_um)
            if plan is None:
                row["status"] = "legal_no_plan"
            else:
                row["status"] = "planned"
                row["route"] = plan["route"]
                row["n_add"] = len(plan["add"])
                row["n_remove"] = len(plan["remove"])
                row["daughter_kinds"] = "+".join(plan["daughter_kinds"])
                if plan["remove"]:
                    tps = []
                    outs = []
                    for p, b in plan["remove"]:
                        tps.append((full_gt.get(p), full_gt.get(b)) in gt_edges)
                        outs.append(len(children_of.get(p, ())))
                    row["removed_edge_is_full_tp"] = any(tps)
                    row["removed_parent_outdeg_before"] = max(outs)
                    row["second_daughter_state"] = "wrong_parent"
                else:
                    row["second_daughter_state"] = "orphan"
                plans.append({
                    "dataset": name, "divider": int(divider), "route": plan["route"],
                    "fork": int(i2s[plan["fork"]]),
                    "add": [(int(i2s[f]), int(i2s[b])) for f, b in plan["add"]],
                    "remove": [(int(i2s[p]), int(i2s[b])) for p, b in plan["remove"]],
                    "daughter_kinds": plan["daughter_kinds"],
                    "removed_edge_is_full_tp": row["removed_edge_is_full_tp"],
                })
        rows.append(row)
    return rows, plans


def score_crop(sub: pl.DataFrame, gt_geff: Path, ea, sc) -> dict:
    """Official per-crop metric row for one submission frame (identical path to ea_atlas.run_crop)."""
    _td, MAX_DISTANCE, estimated_nodes, load_graph, _dm, evaluate, node_recall, per_sample_metrics, _s = sc
    g, _ = ea.build_graph(sub)
    gt = load_graph(gt_geff)
    er = evaluate(g, gt, scale=SCALE, max_distance=MAX_DISTANCE)
    rec = node_recall(g, gt) if g.num_edges() and g.num_nodes() else 0.0
    return per_sample_metrics(er, estimated_nodes(gt_geff), rec)


def rewrite_crop_edges(sub: pl.DataFrame, edges: set[tuple[int, int]]) -> pl.DataFrame:
    """Nodes unchanged; edge rows rebuilt from ``edges`` in the export's layout (node_id=t=z=y=x=-1)."""
    nodes = sub.filter(pl.col("row_type") == "node")
    dataset = nodes["dataset"][0]
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


def paired_bootstrap(rows_c: list[dict], rows_o: list[dict], summarise, n: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    m = len(rows_c)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        sc = summarise([rows_c[i] for i in idx])["score"]
        so = summarise([rows_o[i] for i in idx])["score"]
        deltas.append(so - sc)
    d = np.asarray(deltas)
    return {"ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "mean": float(d.mean()), "resamples": n}


def select_names(names: list[str], max_crops: int | None, stride: int | None) -> list[str]:
    if stride:
        names = names[::stride]
    if max_crops:
        names = names[:max_crops]
    return names


def _summarise_census(census: pd.DataFrame) -> dict:
    n = int(len(census))
    out = {
        "gt_divisions": n,
        "naive_reach": int(census.naive_reach.sum()),
        "metric_legal_reach": int(census.legal_reach.sum()),
        "scored_now": int(census.scored.sum()),
        "status": census.status.value_counts().to_dict(),
        "route": census.loc[census.status == "planned", "route"].value_counts().to_dict(),
        "daughter_kinds": census.loc[census.status == "planned", "daughter_kinds"].value_counts().to_dict(),
        "second_daughter_state": census.loc[census.status == "planned", "second_daughter_state"].value_counts().to_dict(),
        "steals": int((census.n_remove > 0).sum()),
        "steal_breaks_full_tp": int(census.removed_edge_is_full_tp.sum()),
        "reach_if_oracle_applied": int(census.scored.sum() + (census.status == "planned").sum()),
    }
    return out


def cmd_census(args, *, return_plans: bool = False):
    ea = _ea_atlas()
    sc = _scorer()
    from biotrack.submission import read_submission
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = read_submission(ea.open_csv(Path(args.csv)))
    names = select_names(sorted(df["dataset"].unique().to_list()), args.max_crops, args.stride)
    print(f"{len(names)} crops", flush=True)
    rows: list[dict] = []
    plans: list[dict] = []
    t0 = time.time()
    for i, name in enumerate(names, 1):
        gt_geff = Path(args.gt_dir) / f"{name}.geff"
        if not gt_geff.exists():
            print(f"  skip {name}: no GT")
            continue
        r, p = analyse_crop(name, df.filter(pl.col("dataset") == name), gt_geff, ea, sc)
        rows += r
        plans += p
        print(f"  [{i}/{len(names)}] {name} divisions={len(r)} planned={len(p)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    census = pd.DataFrame(rows)
    census.to_parquet(out / f"census_{args.tag}.parquet")
    pd.DataFrame([{**pl_, "add": json.dumps(pl_["add"]), "remove": json.dumps(pl_["remove"]),
                   "daughter_kinds": "+".join(pl_["daughter_kinds"])} for pl_ in plans]
                 ).to_parquet(out / f"plans_{args.tag}.parquet")
    summary = _summarise_census(census) if len(census) else {"gt_divisions": 0}
    (out / f"census_summary_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if return_plans:
        return df, names, plans, census
    return 0


def cmd_oracle(args) -> int:
    ea = _ea_atlas()
    sc = _scorer()
    summarise = sc[-1]
    df, names, plans, census = cmd_census(args, return_plans=True)
    out = Path(args.out_dir)
    plans_by_crop: dict[str, list[dict]] = defaultdict(list)
    for p in plans:
        plans_by_crop[p["dataset"]].append(p)
    rows_c: list[dict] = []
    rows_o: list[dict] = []
    applied_all: list[dict] = []
    oracle_frames: list[pl.DataFrame] = []
    t0 = time.time()
    for i, name in enumerate(names, 1):
        gt_geff = Path(args.gt_dir) / f"{name}.geff"
        if not gt_geff.exists():
            continue
        sub = df.filter(pl.col("dataset") == name)
        edges_df = sub.filter(pl.col("row_type") == "edge")
        edges = set(zip(edges_df["source_id"].to_list(), edges_df["target_id"].to_list()))
        new_edges, applied = apply_plans(edges, plans_by_crop.get(name, []))
        applied_all += applied
        sub_o = rewrite_crop_edges(sub, new_edges)
        oracle_frames.append(sub_o)
        rc = {"dataset": name, **score_crop(sub, gt_geff, ea, sc)}
        ro = {"dataset": name, **score_crop(sub_o, gt_geff, ea, sc)}
        rows_c.append(rc)
        rows_o.append(ro)
        print(f"  [{i}/{len(names)}] {name} applied={len(applied)}/{len(plans_by_crop.get(name, []))} "
              f"div c={rc['division_tp']}/{rc['division_fp']}/{rc['division_fn']} "
              f"o={ro['division_tp']}/{ro['division_fp']}/{ro['division_fn']} "
              f"edge c={rc['edge_tp']}/{rc['edge_fp']}/{rc['edge_fn']} o={ro['edge_tp']}/{ro['edge_fp']}/{ro['edge_fn']} "
              f"({time.time() - t0:.0f}s)", flush=True)
    oracle = pl.concat(oracle_frames, how="vertical").with_columns(pl.Series("id", np.arange(sum(f.height for f in oracle_frames), dtype=np.int64)))
    oracle_csv = out / f"oracle_{args.tag}.csv.gz"
    oracle.write_csv(out / f"oracle_{args.tag}.csv")
    import gzip
    import shutil
    with open(out / f"oracle_{args.tag}.csv", "rb") as fin, gzip.open(oracle_csv, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    (out / f"oracle_{args.tag}.csv").unlink()
    pd.DataFrame(rows_c).to_parquet(out / f"crops_control_{args.tag}.parquet")
    pd.DataFrame(rows_o).to_parquet(out / f"crops_oracle_{args.tag}.parquet")
    write_oracle_summary(out, args.tag, str(args.csv), rows_c, rows_o, plans, applied_all, census,
                         str(oracle_csv), summarise)
    return 0


def _arm_summary(rows: list[dict], summarise) -> dict:
    """summarise() plus the raw edge totals it does not return."""
    s = dict(summarise(rows))
    for k in ("edge_tp", "edge_fp", "edge_fn"):
        s[k] = int(sum(r[k] for r in rows))
    return s


def write_oracle_summary(out: Path, tag: str, csv: str, rows_c: list[dict], rows_o: list[dict],
                         plans: list[dict], applied_all: list[dict], census: pd.DataFrame,
                         oracle_csv: str, summarise) -> dict:
    sc_c = _arm_summary(rows_c, summarise)
    sc_o = _arm_summary(rows_o, summarise)
    boot = paired_bootstrap(rows_c, rows_o, summarise)
    keys = ("score", "adj_edge_jaccard", "edge_jaccard", "division_jaccard",
            "division_tp", "division_fp", "division_fn", "edge_tp", "edge_fp", "edge_fn")
    delta = {}
    for k in keys:
        a, b = sc_o.get(k), sc_c.get(k)
        ok = isinstance(a, (int, float)) and isinstance(b, (int, float)) and a == a and b == b
        delta[k] = (a - b) if ok else None
    summary = {
        "tag": tag, "csv": csv, "n_crops": len(rows_c),
        "control": sc_c, "oracle": sc_o, "delta": delta,
        "paired_bootstrap": boot,
        "plans": len(plans), "applied": len(applied_all),
        "applied_by_route": pd.Series([p["route"] for p in applied_all]).value_counts().to_dict() if applied_all else {},
        "applied_steals": int(sum(1 for p in applied_all if p["remove"])),
        "applied_steal_breaks_full_tp": int(sum(1 for p in applied_all if p.get("removed_edge_is_full_tp"))),
        "census": _summarise_census(census) if len(census) else {},
        "oracle_csv": oracle_csv,
    }
    (out / f"oracle_summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "census"}, indent=2))
    return summary


def cmd_resummarise(args) -> int:
    """Rebuild oracle_summary_<tag>.json from the per-crop parquets an earlier `oracle` run saved
    (no re-scoring). `applied` is re-derived by replaying apply_plans on the control export."""
    ea = _ea_atlas()
    sc = _scorer()
    summarise = sc[-1]
    from biotrack.submission import read_submission
    out = Path(args.out_dir)
    rows_c = pd.read_parquet(out / f"crops_control_{args.tag}.parquet").to_dict("records")
    rows_o = pd.read_parquet(out / f"crops_oracle_{args.tag}.parquet").to_dict("records")
    census = pd.read_parquet(out / f"census_{args.tag}.parquet")
    pdf = pd.read_parquet(out / f"plans_{args.tag}.parquet")
    plans = []
    for r in pdf.to_dict("records"):
        plans.append({**r, "add": [tuple(x) for x in json.loads(r["add"])],
                      "remove": [tuple(x) for x in json.loads(r["remove"])]})
    df = read_submission(ea.open_csv(Path(args.csv)))
    plans_by_crop: dict[str, list[dict]] = defaultdict(list)
    for p in plans:
        plans_by_crop[p["dataset"]].append(p)
    applied_all: list[dict] = []
    for name, crop_plans in plans_by_crop.items():
        edges_df = df.filter((pl.col("dataset") == name) & (pl.col("row_type") == "edge"))
        edges = set(zip(edges_df["source_id"].to_list(), edges_df["target_id"].to_list()))
        _new, applied = apply_plans(edges, crop_plans)
        applied_all += applied
    write_oracle_summary(out, args.tag, str(args.csv), rows_c, rows_o, plans, applied_all, census,
                         str(out / f"oracle_{args.tag}.csv.gz"), summarise)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd, fn in (("census", cmd_census), ("oracle", cmd_oracle), ("resummarise", cmd_resummarise)):
        p = sub.add_parser(cmd)
        p.add_argument("--csv", required=True)
        p.add_argument("--tag", required=True)
        p.add_argument("--gt-dir", default=str(ROOT / "data" / "train"))
        p.add_argument("--out-dir", required=True)
        p.add_argument("--max-crops", type=int)
        p.add_argument("--stride", type=int)
        p.set_defaults(func=fn)
    args = ap.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
