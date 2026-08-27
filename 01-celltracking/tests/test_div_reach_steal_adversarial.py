"""Adversarial checks for scripts/win_bet/div_reach_steal.py (LEVER-0025 instrument, PKT-0021).

Two groups.

* Toy graphs pushed through the OFFICIAL scorer (always run). They pin what the scorer admits
  against what the planner considers, and the hazards of the census's window-matching /
  full-matching split. A failing test here documents a defect; its docstring names the fix.
* Artifact-gated reconciliations (skipped when C:/temp/div_reach and the p19/p20 atlases are
  absent). The packet's numbers must be re-derivable from the per-crop parquets and explained
  edge by edge against the atlas; the control arm must equal the atlas rows and FACT-0323
  (read from the registry, never restated here).
"""
from __future__ import annotations

import functools
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "div_reach_steal", ROOT / "scripts" / "win_bet" / "div_reach_steal.py")
drs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(drs)

td = pytest.importorskip("tracksdata")
pl = pytest.importorskip("polars")
from tracking_cellmot import division_metrics as dm  # noqa: E402
from tracking_cellmot.metrics import evaluate  # noqa: E402

K = td.DEFAULT_ATTR_KEYS
SCALE, MD = drs.SCALE, 7.0

DIV_REACH = Path("C:/temp/div_reach")
ATLAS = {"f0": Path("C:/temp/p20_relink_sweep_f0"), "f1": Path("C:/temp/p19_relink_sweep_f1")}
HAVE_ARTIFACTS = all((DIV_REACH / t / f"oracle_summary_{t}.json").exists() for t in ("f0", "f1")) and all(
    (ATLAS[t] / "atlas" / "edges_pen_off.parquet").exists() for t in ("f0", "f1"))
artifacts = pytest.mark.skipif(not HAVE_ARTIFACTS, reason="C:/temp/div_reach + p19/p20 atlases not on disk")


# ----------------------------------------------------------------------------- toy scaffolding
def mk(nodes: dict, edges: list):
    """nodes: name -> (t, z, y, x) in VOXELS; edges in ROW ORDER (edge ids follow row order)."""
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    names = list(nodes)
    ids = g.bulk_add_nodes([{"t": int(nodes[n][0]), "z": float(nodes[n][1]), "y": float(nodes[n][2]),
                             "x": float(nodes[n][3])} for n in names])
    m = dict(zip(names, ids))
    if edges:
        g.bulk_add_edges([{"source_id": m[a], "target_id": m[b]} for a, b in edges])
    return g, m


# GT: G(4) -> D(5) -> C1(6) -> GC1(7); D -> C2(6) -> GC2(7). x is 0.40625 um/voxel: 40 voxels = 16 um.
GT = {"G": (4, 10, 100, 100), "D": (5, 10, 100, 100), "C1": (6, 10, 100, 80), "C2": (6, 10, 100, 120),
      "GC1": (7, 10, 100, 70), "GC2": (7, 10, 100, 130)}
GT_EDGES = [("G", "D"), ("D", "C1"), ("D", "C2"), ("C1", "GC1"), ("C2", "GC2")]


def gt_graph(extra_nodes=None, extra_edges=()):
    nodes = dict(GT)
    nodes.update(extra_nodes or {})
    return mk(nodes, GT_EDGES + list(extra_edges))


def planner_inputs(g, m, gt_m, roles):
    """Build plan_division's inputs from a toy pred graph. roles: pred name -> GT name (the window match)."""
    na = g.node_attrs(attr_keys=[K.NODE_ID, "t", "z", "y", "x"]).to_pandas()
    t_of = dict(zip(na[K.NODE_ID].astype(int), na["t"].astype(int)))
    pos = {int(n): (z * SCALE[0], y * SCALE[1], x * SCALE[2])
           for n, z, y, x in zip(na[K.NODE_ID], na["z"], na["y"], na["x"])}
    parent_of, children_of = {}, {}
    if g.num_edges():
        ea = g.edge_attrs(attr_keys=[K.EDGE_SOURCE, K.EDGE_TARGET]).to_pandas()
        for s, t in zip(ea[K.EDGE_SOURCE].astype(int), ea[K.EDGE_TARGET].astype(int)):
            children_of.setdefault(s, []).append(t)
            parent_of[t] = s
    node_to_gt = {m[p]: gt_m[gname] for p, gname in roles.items()}
    parent_ids = {m[p] for p, gname in roles.items() if gname in ("G", "D")}
    daughter_ids = [{m[p] for p, gname in roles.items() if gname in ("C1", "GC1")},
                    {m[p] for p, gname in roles.items() if gname in ("C2", "GC2")}]
    return dict(t_div=5, divider=gt_m["D"], parent_ids=parent_ids, daughter_ids=daughter_ids,
                node_to_gt=node_to_gt, parent_of=parent_of, children_of=children_of, t_of=t_of, pos_um=pos)


# ----------------------------------------------------------------------------- toy: scorer vs planner
def test_scorer_admits_early_fork_at_the_grandparent_node_but_the_planner_never_considers_it():
    """division_metrics._is_strongly_connected_division accepts pred_div itself in parent_ids, so a fork
    AT the grandparent-matched node (t_div-1) whose two branches reach C1 and C2 scores the division.
    plan_division only tries forks at t_div (routes direct/pred), so it returns None here: the
    D-matched node already has a wrong second child, and the planner declines it."""
    gt, gm = gt_graph()
    pred = dict(GT)
    pred["W"] = (6, 10, 100, 100)     # wrong second child of the D-matched node (unmatched)
    pred["X"] = (5, 10, 100, 140)     # unmatched t_div detection that the grandparent could fork to
    control = [("G", "D"), ("D", "C1"), ("D", "W"), ("C1", "GC1"), ("C2", "GC2")]
    g, m = mk(pred, control)
    assert dm.score_divisions(g, gt, SCALE, MD).scores[gm["D"]] == 0
    plan = drs.plan_division(**planner_inputs(
        g, m, gm, {"G": "G", "D": "D", "C1": "C1", "C2": "C2", "GC1": "GC1", "GC2": "GC2"}))
    assert plan is None
    g2, m2 = mk(pred, control + [("G", "X"), ("X", "C2")])
    s = dm.score_divisions(g2, gt, SCALE, MD)
    assert s.scores[gm["D"]] == 1 and m2["G"] in s.tp_forks


def test_scorer_admits_late_fork_one_frame_after_the_division_but_the_planner_never_considers_it():
    """Candidate forks are parent_ids | successors(parent_ids): the C1-matched node at t_div+1 is a
    candidate, and its branches {GC1}, {GC2} hit both GT lineages (grandchild evidence). The planner
    only looks for daughters at t_div+1 from a fork at t_div, so with C2 undetected it plans nothing."""
    gt, gm = gt_graph()
    pred = {k: v for k, v in GT.items() if k != "C2"}
    control = [("G", "D"), ("D", "C1"), ("C1", "GC1")]
    g, m = mk(pred, control)
    assert dm.score_divisions(g, gt, SCALE, MD).scores[gm["D"]] == 0
    plan = drs.plan_division(**planner_inputs(
        g, m, gm, {"G": "G", "D": "D", "C1": "C1", "GC1": "GC1", "GC2": "GC2"}))
    assert plan is None
    g2, m2 = mk(pred, control + [("C1", "GC2")])
    s = dm.score_divisions(g2, gt, SCALE, MD)
    assert s.scores[gm["D"]] == 1 and m2["C1"] in s.tp_forks


def test_window_admissible_daughter_from_another_gt_component_is_planned_but_rejected_by_the_scorer():
    """The census chooses daughters from the per-division WINDOW matching. A detection W that the window
    assigns to C2 can be assigned to a nearer node of a different GT track (Q6) in the FULL matching;
    _pred_division_fork_sets then flags the fork cross-component and score_divisions returns 0 with the
    fork in fp_forks. plan_division has no guard for this (analyse_crop computes full_gt but only uses
    it for removed_edge_is_full_tp), so such a plan is counted as planned/applied and would ADD a
    division FP. Zero such cases on f0/f1 (per-crop applied == delta division_tp), so the rescored
    ceilings are unaffected - but the census field reach_if_oracle_applied assumes planned => scored."""
    gt, gm = gt_graph({"Q5": (5, 10, 100, 124), "Q6": (6, 10, 100, 121)}, [("Q5", "Q6")])
    pred = {"G": GT["G"], "D": GT["D"], "C1": GT["C1"], "W": (6, 10, 100, 121),
            "GC1": GT["GC1"], "GC2": GT["GC2"]}
    control = [("G", "D"), ("D", "C1"), ("C1", "GC1")]
    g, m = mk(pred, control)
    win = dm.match_divisions(g, gt, SCALE, MD)[gm["D"]]
    wa = dm._matched_node_attrs(win)
    w2g = dict(zip(wa[K.NODE_ID].to_list(), wa[K.MATCHED_NODE_ID].to_list()))
    fa = dm._matched_node_attrs(dm._match_full(g, gt, SCALE, MD))
    f2g = dict(zip(fa[K.NODE_ID].to_list(), fa[K.MATCHED_NODE_ID].to_list()))
    assert w2g[m["W"]] == gm["C2"] and f2g[m["W"]] == gm["Q6"]
    inputs = planner_inputs(g, m, gm, {"G": "G", "D": "D", "C1": "C1", "W": "C2", "GC1": "GC1", "GC2": "GC2"})
    plan = drs.plan_division(**inputs)
    assert plan is not None and plan["add"] == [(m["D"], m["W"])]
    g2, m2 = mk(pred, control + [("D", "W")])
    s = dm.score_divisions(g2, gt, SCALE, MD)
    assert s.scores[gm["D"]] == 0 and m2["D"] in s.fp_forks


def test_planner_never_emits_an_edit_free_plan_and_apply_plans_never_counts_one_as_applied():
    """DEFECT (latent; zero impact on f0/f1, where every plan has exactly one add). When the fork's single
    existing child is unmatched and its successors cover BOTH GT lineages, lineage_of() returns {0, 1},
    covered has size 2, the add loop is skipped, and plan_division returns a plan with add == [] and
    remove == []. apply_plans accepts it (nothing to collide) and counts it as applied, so applied and
    reach_if_oracle_applied would overstate without any division TP. Fix: in plan_division, after the
    add loop, `if not adds and not removes: continue` (an edit-free plan is not a plan); defensively,
    apply_plans should skip plans with no adds."""
    D, GC1, GC2 = 900, 904, 905
    M, a, gc1, gc2 = 1, 2, 3, 4
    t_of = {M: 5, a: 6, gc1: 7, gc2: 7}
    parent_of = {a: M, gc1: a, gc2: a}
    children_of = {M: [a], a: [gc1, gc2]}
    pos = {n: (float(t_of[n]), float(n), 0.0) for n in t_of}
    plan = drs.plan_division(t_div=5, divider=D, parent_ids={M}, daughter_ids=[{gc1}, {gc2}],
                             node_to_gt={M: D, gc1: GC1, gc2: GC2}, parent_of=parent_of,
                             children_of=children_of, t_of=t_of, pos_um=pos)
    assert plan is None or plan["add"], f"edit-free plan emitted: {plan}"
    empty = {"route": "direct", "fork": M, "keep": [a], "add": [], "remove": [], "daughter_kinds": [],
             "removed_edge_is_full_tp": False}
    _edges, applied = drs.apply_plans({(M, a), (a, gc1), (a, gc2)}, [empty])
    assert applied == [], "apply_plans counted an edit-free plan as applied"


def test_a_fork_with_no_children_gets_both_daughters_and_two_steals():
    """Refutes the suspected cap: the planner DOES fill both lineages (two adds, two removals) when the
    divider-matched node has out-degree 0."""
    D, C1, C2 = 900, 902, 903
    M, P1, P2, b1, b2 = 1, 2, 3, 4, 5
    t_of = {M: 5, P1: 5, P2: 5, b1: 6, b2: 6}
    parent_of = {b1: P1, b2: P2}
    children_of = {P1: [b1], P2: [b2]}
    pos = {n: (float(t_of[n]), float(n), 0.0) for n in t_of}
    plan = drs.plan_division(t_div=5, divider=D, parent_ids={M}, daughter_ids=[{b1}, {b2}],
                             node_to_gt={M: D, b1: C1, b2: C2}, parent_of=parent_of,
                             children_of=children_of, t_of=t_of, pos_um=pos)
    assert plan["add"] == [(M, b1), (M, b2)] and plan["remove"] == [(P1, b1), (P2, b2)]


def test_out_degree_cap_is_row_order_dependent_but_merge_dedup_is_not():
    """metrics._evaluate_matched_graph keeps the two lowest EDGE_IDs per source, and edge ids follow row
    order (ea_atlas.build_graph). rewrite_crop_edges re-emits ALL edges of a crop in sorted (source,
    target) order, so the oracle is only order-safe because no node has out-degree > 2 in the control
    (checked on the artifacts below). Merge-dedup drops the duplicate from BOTH the TP and the FP
    count, so it is order-free."""
    gt, _ = gt_graph()
    pred = dict(GT)
    pred["Z"] = (6, 10, 100, 100)
    a = evaluate(mk(pred, [("G", "D"), ("D", "C1"), ("D", "C2"), ("D", "Z")])[0], gt, SCALE, MD)
    b = evaluate(mk(pred, [("G", "D"), ("D", "Z"), ("D", "C1"), ("D", "C2")])[0], gt, SCALE, MD)
    assert (a.edge_tp, a.edge_fp) != (b.edge_tp, b.edge_fp)
    pred = dict(GT)
    pred["D2"] = (5, 10, 100, 101)
    a = evaluate(mk(pred, [("G", "D"), ("D", "C1"), ("D", "C2"), ("D2", "C1")])[0], gt, SCALE, MD)
    b = evaluate(mk(pred, [("D2", "C1"), ("G", "D"), ("D", "C1"), ("D", "C2")])[0], gt, SCALE, MD)
    assert (a.edge_tp, a.edge_fp, a.edge_fn) == (b.edge_tp, b.edge_fp, b.edge_fn)


# ----------------------------------------------------------------------------- artifact reconciliation
@functools.lru_cache(maxsize=None)
def pd_():
    import pandas as pd
    return pd


@functools.lru_cache(maxsize=None)
def crops(tag):
    pd = pd_()
    d = DIV_REACH / tag
    c = pd.read_parquet(d / f"crops_control_{tag}.parquet").set_index("dataset")
    o = pd.read_parquet(d / f"crops_oracle_{tag}.parquet").set_index("dataset")
    plans = pd.read_parquet(d / f"plans_{tag}.parquet")
    census = pd.read_parquet(d / f"census_{tag}.parquet")
    atlas = pd.read_parquet(ATLAS[tag] / "atlas" / "crops_pen_off.parquet").set_index("dataset")
    return c, o, plans, census, atlas


@functools.lru_cache(maxsize=None)
def submission(tag, arm):
    from biotrack.submission import read_submission
    ea = drs._ea_atlas()
    path = ATLAS[tag] / "sweep_pen_off.csv.gz" if arm == "control" else DIV_REACH / tag / f"oracle_{tag}.csv.gz"
    return read_submission(ea.open_csv(path))


def fact_control_row(fact_id="FACT-0323"):
    import yaml
    data = yaml.safe_load((ROOT / "research" / "00-system" / "registry" / "facts.yaml").read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else next(v for v in data.values() if isinstance(v, list))
    fact = next(f for f in items if f.get("id") == fact_id)
    return fact["scope"]["control"]


@artifacts
def test_control_arm_equals_the_atlas_rows_and_the_registry_control_row():
    """Q1. score_crop is the same path as ea_atlas.run_crop; the control parquet must equal the atlas
    crops_pen_off rows per crop on every column, and its summarise() must round to FACT-0323's control."""
    from tracking_cellmot.metrics import summarise
    for tag in ("f0", "f1"):
        c, _o, _p, _cen, atlas = crops(tag)
        assert set(c.index) == set(atlas.index)
        cols = ["edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn",
                "num_pred_nodes", "node_recall", "adj_edge_jaccard"]
        diff = (c.loc[atlas.index, cols] - atlas[cols]).abs().max()
        assert float(diff.max()) < 1e-12, diff
    c, *_ = crops("f0")
    rows = c.reset_index().to_dict("records")
    s = summarise(rows)
    ref = fact_control_row()
    assert round(s["score"], 4) == ref["score"] and round(s["adj_edge_jaccard"], 4) == ref["adj_edge"]
    assert [s["division_tp"], s["division_fp"], s["division_fn"]] == ref["division"]
    assert [int(sum(r[k] for r in rows)) for k in ("edge_tp", "edge_fp", "edge_fn")] == ref["edge"]


@artifacts
def test_five_fold0_crops_rescore_through_score_crop_to_the_saved_rows_for_both_arms():
    """Q1 (live). Re-run score_crop on five fold-0 crops from the control CSV and from the saved oracle CSV;
    both must reproduce the saved per-crop rows, proving the oracle CSV on disk is what was scored."""
    ea, sc = drs._ea_atlas(), drs._scorer()
    c, o, *_ = crops("f0")
    names = ["44b6_12dfb391", "44b6_d5e7d891", "44b6_c50204e0", "44b6_0113de3b", "44b6_f28707c6"]
    keys = ["edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn",
            "num_pred_nodes", "node_recall", "adj_edge_jaccard"]
    for name in names:
        gt_geff = ROOT / "data" / "train" / f"{name}.geff"
        for arm, saved in (("control", c), ("oracle", o)):
            sub = submission("f0", arm).filter(pl.col("dataset") == name)
            row = drs.score_crop(sub, gt_geff, ea, sc)
            for k in keys:
                assert abs(row[k] - saved.loc[name, k]) < 1e-12, (name, arm, k, row[k], saved.loc[name, k])


@artifacts
def test_per_crop_applied_equals_delta_division_tp_and_division_fp_never_moves():
    """Q3 (empirical). No planned fork was rejected (cross-component, malformed, pairing) on either fold,
    and no steal turned a charged FP fork into a non-fork: division_fp is unchanged on every crop."""
    for tag in ("f0", "f1"):
        c, o, plans, *_ = crops(tag)
        napp = plans.groupby("dataset").size()
        for name in c.index:
            n = int(napp.get(name, 0))
            assert int(o.loc[name, "division_tp"] - c.loc[name, "division_tp"]) == n, (tag, name)
            assert int(o.loc[name, "division_fn"] - c.loc[name, "division_fn"]) == -n, (tag, name)
            assert int(o.loc[name, "division_fp"] - c.loc[name, "division_fp"]) == 0, (tag, name)


@artifacts
def test_matching_is_edit_invariant_node_recall_and_node_count_identical_per_crop():
    """Q6. DistanceMatching is centroid-only (tracksdata/metrics/_matching.py compute_weights uses cdist on
    scaled z,y,x), so edge edits cannot move the matching: node_recall and num_pred_nodes must be
    identical per crop between arms."""
    for tag in ("f0", "f1"):
        c, o, *_ = crops(tag)
        assert (c["node_recall"] == o.loc[c.index, "node_recall"]).all()
        assert (c["num_pred_nodes"] == o.loc[c.index, "num_pred_nodes"]).all()


@artifacts
def test_every_edge_delta_is_explained_by_the_plans_against_the_atlas():
    """Q5. Per fold: every removed edge is unmatched in the scored frame (never a TP); delta edge_tp equals
    the number of added edges that are GT edges under the FULL matching; delta edge_fp equals the
    pred_valid non-TP adds minus the pred_valid removals; the non-TP adds are exactly the grand
    daughters (b unmatched in the full matching) and the pred forks (fork unmatched); every
    direct/direct add has both endpoints full-matched to (divider, child)."""
    pd = pd_()
    for tag in ("f0", "f1"):
        c, o, plans, *_ = crops(tag)
        A = ATLAS[tag] / "atlas"
        E = pd.read_parquet(A / "edges_pen_off.parquet",
                            columns=["dataset", "source_id", "target_id", "matched", "pred_valid"]
                            ).set_index(["dataset", "source_id", "target_id"])
        N = pd.read_parquet(A / "nodes_pen_off.parquet", columns=["dataset", "node_id", "gt_id"]
                            ).set_index(["dataset", "node_id"])["gt_id"]
        GE = pd.read_parquet(A / "gtedges_pen_off.parquet", columns=["dataset", "gt_source", "gt_target"])
        GN = pd.read_parquet(A / "gtnodes_pen_off.parquet", columns=["dataset", "gt_id", "out_degree", "in_degree"]
                             ).set_index(["dataset", "gt_id"])
        gt_edges = set(zip(GE.dataset, GE.gt_source.astype(int), GE.gt_target.astype(int)))
        tp_adds = fp_adds = fp_rems = 0
        for r in plans.itertuples():
            for f, b in json.loads(r.add):
                gf, gb = int(N[(r.dataset, f)]), int(N[(r.dataset, b)])
                is_tp = (r.dataset, gf, gb) in gt_edges
                out_valid = gf != -1 and int(GN.loc[(r.dataset, gf), "out_degree"]) > 0
                in_valid = gb != -1 and int(GN.loc[(r.dataset, gb), "in_degree"]) > 0
                if is_tp:
                    tp_adds += 1
                    assert r.route == "direct" and r.daughter_kinds == "direct"
                    assert gf == r.divider and (r.dataset, r.divider, gb) in gt_edges
                else:
                    assert (r.route, r.daughter_kinds) in {("direct", "grand"), ("pred", "direct")}, (tag, r.dataset, r.divider)
                    assert (gb == -1) if r.daughter_kinds == "grand" else (gf == -1)
                    fp_adds += int(out_valid or in_valid)
            for p, b in json.loads(r.remove):
                e = E.loc[(r.dataset, p, b)]
                assert not bool(e.matched), (tag, r.dataset, p, b)
                fp_rems += int(bool(e.pred_valid))
        d_tp = int(o["edge_tp"].sum() - c["edge_tp"].sum())
        d_fp = int(o["edge_fp"].sum() - c["edge_fp"].sum())
        d_fn = int(o["edge_fn"].sum() - c["edge_fn"].sum())
        assert (d_tp, d_fp, d_fn) == (tp_adds, fp_adds - fp_rems, -tp_adds), (tag, d_tp, d_fp, d_fn, tp_adds, fp_adds, fp_rems)


@artifacts
def test_oracle_csv_is_the_control_plus_exactly_the_plans_with_degree_invariants():
    """Q3. Out-degree cap and merge-dedup are moot for the rewrite: no source has out-degree > 2 and no
    duplicate edge exists in either arm; in-degree <= 1 everywhere; node rows are untouched; the edge
    set differs from the control by exactly the plans' adds and removes."""
    for tag in ("f0", "f1"):
        _c, _o, plans, *_ = crops(tag)
        dc, do = submission(tag, "control"), submission(tag, "oracle")
        ec, eo = dc.filter(pl.col("row_type") == "edge"), do.filter(pl.col("row_type") == "edge")
        for e in (ec, eo):
            assert e.group_by(["dataset", "source_id"]).len()["len"].max() <= 2
            assert e.group_by(["dataset", "target_id"]).len()["len"].max() == 1
            assert e.group_by(["dataset", "source_id", "target_id"]).len().filter(pl.col("len") > 1).height == 0
        cols = ["dataset", "node_id", "t", "z", "y", "x"]
        nc = dc.filter(pl.col("row_type") == "node").select(cols).sort(["dataset", "node_id"])
        no = do.filter(pl.col("row_type") == "node").select(cols).sort(["dataset", "node_id"])
        assert nc.equals(no)
        sc = set(zip(ec["dataset"], ec["source_id"], ec["target_id"]))
        so = set(zip(eo["dataset"], eo["source_id"], eo["target_id"]))
        assert so - sc == {(r.dataset, f, b) for r in plans.itertuples() for f, b in json.loads(r.add)}
        assert sc - so == {(r.dataset, p, b) for r in plans.itertuples() for p, b in json.loads(r.remove)}


@artifacts
def test_summary_json_reproduces_from_the_parquets_and_census_scored_equals_control_division_tp():
    """Q2. `scored` is score_divisions().scores: its per-crop sum must equal the control division_tp
    (evaluate() calls the same function); legal_reach == (status != illegal); scored => legal.
    The summary JSON (rebuilt by `resummarise` after the first oracle run crashed at summary time -
    see the oracle_f0.log tail) must reproduce from the per-crop parquets."""
    from tracking_cellmot.metrics import summarise
    for tag in ("f0", "f1"):
        c, o, _plans, census, _a = crops(tag)
        J = json.loads((DIV_REACH / tag / f"oracle_summary_{tag}.json").read_text(encoding="utf-8"))
        sc = summarise(c.reset_index().to_dict("records"))
        so = summarise(o.reset_index().to_dict("records"))
        assert abs(sc["score"] - J["control"]["score"]) < 1e-12 and abs(so["score"] - J["oracle"]["score"]) < 1e-12
        assert abs((so["score"] - sc["score"]) - J["delta"]["score"]) < 1e-12
        per_crop = census.groupby("dataset")["scored"].sum()
        for name, n in per_crop.items():
            assert int(n) == int(c.loc[name, "division_tp"]), (tag, name)
        assert (census["legal_reach"] == (census["status"] != "illegal")).all()
        assert (~census["scored"] | census["legal_reach"]).all()
        assert (census.loc[census.status == "planned", "n_add"] >= 1).all()


def window_match(g, gt_div):
    from tracksdata.metrics import DistanceMatching
    pc = g.copy()
    dm._reset_matching_attrs(pc)
    pc.match(gt_div, matching=DistanceMatching(max_distance=MD, scale=SCALE))
    return pc


@artifacts
def test_the_two_pred_route_plans_are_admitted_through_the_scorers_predecessor_allowance():
    """Q4. Both fold-1 pred plans fork at a node NOT in parent_ids whose predecessor IS (matched to the
    grandparent): pred_parent_ids = {fork, *predecessors(fork)} intersects parent_ids. The fork is
    strongly connected only after the edit, is neither cross-component nor malformed under the full
    matching, and the divider scores 1 in the oracle arm and 0 in the control."""
    ea, sc = drs._ea_atlas(), drs._scorer()
    load_graph = sc[3]
    _c, _o, plans, *_ = crops("f1")
    pred_plans = plans[plans.route == "pred"]
    assert len(pred_plans) == 2
    for r in pred_plans.itertuples():
        gt = load_graph(ROOT / "data" / "train" / f"{r.dataset}.geff")
        gt_div = dm.extract_divisions(gt)[r.divider]
        gp = [int(x) for x in gt_div.predecessors(r.divider)]
        checks = {}
        for arm in ("control", "oracle"):
            sub = submission("f1", arm).filter(pl.col("dataset") == r.dataset)
            g, i2s = ea.build_graph(sub)
            s2i = {v: k for k, v in i2s.items()}
            fork = s2i[r.fork]
            mp = window_match(g, gt_div)
            attrs = dm._matched_node_attrs(mp)
            parent_ids, daughter_ids = dm._matched_division_nodes(attrs, gt_div, r.divider)
            n2g = dict(zip(attrs[K.NODE_ID].to_list(), attrs[K.MATCHED_NODE_ID].to_list()))
            assert fork not in parent_ids and n2g.get(fork) is None
            preds = [int(p) for p in mp.predecessors(fork)]
            assert len(preds) == 1 and preds[0] in parent_ids and n2g[preds[0]] in gp
            checks[arm] = dm._is_strongly_connected_division(mp, fork, parent_ids, daughter_ids)
            if arm == "oracle":
                _ev, cross, malformed = dm._pred_division_fork_sets(g, gt, SCALE, MD)
                assert fork not in cross and fork not in malformed
                s = dm.score_divisions(g, gt, SCALE, MD)
                assert s.scores[r.divider] == 1 and fork in s.tp_forks
        assert checks == {"control": False, "oracle": True}


@artifacts
def test_illegal_divisions_are_edge_invariant_so_no_edge_edit_can_reach_them():
    """Q7. legal_reach is _matched_division_nodes(...) is not None on the window matching, which depends only
    on centroids. For fold 0's illegal divisions, the same call on a graph with EVERY edge removed gives
    the same (None) answer: fewer than two daughter lineages own a matched node. They are detection
    failures, not planner blind spots."""
    ea, sc = drs._ea_atlas(), drs._scorer()
    load_graph = sc[3]
    _c, _o, _p, census, _a = crops("f0")
    illegal = census[census.status == "illegal"]
    assert len(illegal) == 3
    for r in illegal.itertuples():
        gt = load_graph(ROOT / "data" / "train" / f"{r.dataset}.geff")
        gt_div = dm.extract_divisions(gt)[r.divider]
        sub = submission("f0", "control").filter(pl.col("dataset") == r.dataset)
        g, _ = ea.build_graph(sub)
        g_noedge, _ = ea.build_graph(sub.filter(pl.col("row_type") == "node"))
        for gg in (g, g_noedge):
            attrs = dm._matched_node_attrs(window_match(gg, gt_div))
            assert dm._matched_division_nodes(attrs, gt_div, r.divider) is None
            n2g = dict(zip(attrs[K.NODE_ID].to_list(), attrs[K.MATCHED_NODE_ID].to_list()))
            children = [int(c) for c in gt_div.successors(r.divider)]
            lineages = [{c, *[int(x) for x in gt_div.successors(c)]} for c in children]
            assert sum(bool(set(n2g.values()) & lin) for lin in lineages) < 2


@artifacts
def test_legal_no_plan_divisions_can_be_scored_by_edits_the_planner_never_tries():
    """Q6/Q7. The oracle is NOT an upper bound on this edit family. Verified through score_divisions on the
    real crops: (a) a LATE fork at t_div+1 (fold 0, 44b6_c50204e0 / 208000000033) scores with ONE added
    edge and no removal; (b) replacing the divider-matched fork's wrong child (fold 1, 6bba_fc5f39dc /
    55000245: remove the non-TP edges (8408,8638),(8438,8669), add (8408,8669),(8408,8626)) scores too.
    Both are census status legal_no_plan. The exhaustive search over the scorer's own candidate set
    found 7 of the 9 legal_no_plan divisions fixable this way (3 late-fork, 4 wrong-child)."""
    ea, sc = drs._ea_atlas(), drs._scorer()
    load_graph = sc[3]
    cases = [("f0", "44b6_c50204e0", 208000000033, [(33286, 33834)], []),
             ("f1", "6bba_fc5f39dc", 55000245, [(8408, 8669), (8408, 8626)], [(8408, 8638), (8438, 8669)])]
    pd = pd_()
    for tag, name, divider, adds, removes in cases:
        _c, _o, _p, census, _a = crops(tag)
        row = census[(census.dataset == name) & (census.divider == divider)]
        assert len(row) == 1 and row.status.iloc[0] == "legal_no_plan"
        E = pd.read_parquet(ATLAS[tag] / "atlas" / "edges_pen_off.parquet",
                            columns=["dataset", "source_id", "target_id", "matched"])
        E = E[E.dataset == name]
        tp = set(zip(E.source_id[E.matched].astype(int), E.target_id[E.matched].astype(int)))
        assert not any(e in tp for e in removes)
        sub = submission(tag, "control").filter(pl.col("dataset") == name)
        gt = load_graph(ROOT / "data" / "train" / f"{name}.geff")
        ed = sub.filter(pl.col("row_type") == "edge")
        edges = set(zip(ed["source_id"].to_list(), ed["target_id"].to_list()))
        assert all(e in edges for e in removes) and not any(e in edges for e in adds)
        g0, _ = ea.build_graph(sub)
        assert dm.score_divisions(g0, gt, SCALE, MD).scores[divider] == 0
        g1, _ = ea.build_graph(drs.rewrite_crop_edges(sub, (edges - set(removes)) | set(adds)))
        assert dm.score_divisions(g1, gt, SCALE, MD).scores[divider] == 1
