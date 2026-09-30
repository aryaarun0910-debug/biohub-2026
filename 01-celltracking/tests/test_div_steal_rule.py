"""Software contracts for scripts/win_bet/div_steal_rule.py (LEVER-0025 stage C, PKT-0021).

These pin the pure graph arrays, the candidate enumeration, the GT-free features and the rule on
hand-built toy graphs. They say nothing about whether the rule scores - that is the packet's job.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "div_steal_rule", ROOT / "scripts" / "win_bet" / "div_steal_rule.py")
dsr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dsr)

Z, Y, X = dsr.SCALE   # 1.625, 0.40625, 0.40625


def um(z_um, y_um, x_um):
    """Voxel coordinates that land at the requested um position."""
    return (z_um / Z, y_um / Y, x_um / X)


def build(nodes, edges):
    """nodes: {id: (t, (z_um, y_um, x_um))}; edges: [(s, t)]."""
    ids = list(nodes)
    t = [nodes[i][0] for i in ids]
    zyx = [um(*nodes[i][1]) for i in ids]
    src = [s for s, _ in edges]
    tgt = [d for _, d in edges]
    return dsr.build_crop(ids, t, zyx, src, tgt)


# A textbook division seen by the linker as two tracks:
#   G(4) -> M(5) -> a(6) -> a2(7)         M's existing daughter a
#   Q(4) -> P(5) -> b(6) -> b2(7)         b is the second daughter, wrongly owned by P
#   o(6) is an orphan near P (P's real child), far from M
def textbook():
    nodes = {
        10: (4, (0.0, 0.0, 0.0)),    # G
        1: (5, (0.0, 0.0, 1.0)),     # M
        3: (6, (0.0, 3.0, 1.0)),     # a  (3 um from M)
        4: (6, (0.0, -2.0, 1.0)),    # b  (2 um from M, 5 um from a)
        5: (7, (0.0, 5.0, 1.0)),     # a2 (daughters diverge to 10 um)
        6: (7, (0.0, -5.0, 1.0)),    # b2
        20: (4, (0.0, -10.0, 1.0)),  # Q
        2: (5, (0.0, -9.0, 1.0)),    # P  (7 um from b)
        7: (6, (0.0, -8.5, 1.0)),    # o  orphan 0.5 um from P, 6.5 um from b (a is nearer to b)
    }
    edges = [(10, 1), (1, 3), (3, 5), (20, 2), (2, 4), (4, 6)]
    return nodes, edges


def test_build_crop_degrees_tracks_and_um_scale():
    nodes, edges = textbook()
    c = build(nodes, edges)
    i = {n: int(c.index_of([n])[0]) for n in nodes}
    assert c.outdeg[i[1]] == 1 and c.parent[i[3]] == i[1] and c.parent[i[7]] == -1
    assert c.len_before[i[3]] == 2 and c.len_after[i[1]] == 2 and c.len_after[i[5]] == 0
    d = np.linalg.norm(c.pos[i[3]] - c.pos[i[1]])
    assert abs(d - 3.0) < 1e-6


def test_enumerate_candidates_finds_the_steal_and_the_orphan_but_not_the_own_child():
    nodes, edges = textbook()
    c = build(nodes, edges)
    M, A, B = dsr.enumerate_candidates(c, parent_max=8.0, sister_max=11.0)
    trip = {(int(c.ids[m]), int(c.ids[a]), int(c.ids[b])) for m, a, b in zip(M, A, B)}
    assert (1, 3, 4) in trip                 # M with a, stealing b
    assert (2, 4, 7) in trip                 # P with b, orphan o within 8 um
    assert all(a != b for _m, a, b in trip)  # never its own child


def test_features_of_the_textbook_steal():
    nodes, edges = textbook()
    c = build(nodes, edges)
    i = {n: int(c.index_of([n])[0]) for n in nodes}
    f = dsr.candidate_features(c, [i[1]], [i[3]], [i[4]])
    r = f.iloc[0]
    assert r.b_has_parent and r.P_id == 2
    assert abs(r.d_mb - 2.0) < 1e-6 and abs(r.d_ab - 5.0) < 1e-6 and abs(r.d_pb - 7.0) < 1e-6
    assert abs(r.steal_gain - 5.0) < 1e-6
    assert r.c1_has_pred and r.c3_ok and abs(r.divergence - 5.0) < 1e-6
    assert r.mutual_nn_all and r.nearest_t_is_m and not r.nearest_t_is_p
    assert abs(r.cos_daughters + 1.0) < 1e-6          # opposite directions
    assert abs(r.d_p_orphan - 0.5) < 1e-6 and r.p_alt_tight
    assert r.p_outdeg == 1 and r.p_len_before == 1 and r.b_len_after == 1


def test_rule_accepts_the_textbook_steal_and_rejects_the_orphan_by_default():
    nodes, edges = textbook()
    c = build(nodes, edges)
    M, A, B = dsr.enumerate_candidates(c)
    f = dsr.candidate_features(c, M, A, B)
    mask = dsr.rule_mask(f, {})
    fired = {(int(r.M_id), int(r.b_id)) for r in f.loc[mask].itertuples()}
    assert fired == {(1, 4)}
    plans = dsr.select_edits(f, mask)
    assert plans[0]["add"] == [(1, 4)] and plans[0]["remove"] == [(2, 4)]


def test_rule_refuses_when_p_is_nearer_to_b_than_m():
    nodes, edges = textbook()
    nodes[2] = (5, (0.0, -3.0, 1.0))     # move P to 1 um from b: M (2 um) is no longer nearest
    c = build(nodes, edges)
    M, A, B = dsr.enumerate_candidates(c)
    f = dsr.candidate_features(c, M, A, B)
    assert not dsr.rule_mask(f, {}).any()
    # relaxing the nearest-t requirement alone is not enough: steal_gain is negative
    assert not dsr.rule_mask(f, {"require_m_nearest_t": False}).any()


def test_c3_divergence_gate_and_c1_track_start_gate():
    nodes, edges = textbook()
    nodes[5] = (7, (0.0, 2.0, 1.0))      # a2 falls back: divergence 7 - 5 = 2 < 2.25
    c = build(nodes, edges)
    M, A, B = dsr.enumerate_candidates(c)
    f = dsr.candidate_features(c, M, A, B)
    assert not dsr.rule_mask(f, {}).any()
    assert dsr.rule_mask(f, {"require_c3": False}).any()
    nodes, edges = textbook()
    edges = [e for e in edges if e != (10, 1)]   # M becomes a track start
    c = build(nodes, edges)
    M, A, B = dsr.enumerate_candidates(c)
    f = dsr.candidate_features(c, M, A, B)
    assert not dsr.rule_mask(f, {}).any()
    assert dsr.rule_mask(f, {"require_c1": False}).any()


def test_select_edits_takes_one_daughter_per_mother_and_one_mother_per_daughter():
    import pandas as pd
    f = pd.DataFrame({
        "M_id": [1, 1, 9], "a_id": [3, 3, 8], "b_id": [4, 5, 4], "P_id": [2, 2, 2],
        "b_has_parent": [True, True, True], "rank_score": [1.0, 2.0, 0.5],
    })
    plans = dsr.select_edits(f, np.ones(3, dtype=bool))
    adds = [p["add"][0] for p in plans]
    assert adds == [(9, 4), (1, 5)]       # b=4 goes to the better-ranked mother 9; mother 1 takes b=5


def test_build_crop_rejects_two_parents():
    nodes, edges = textbook()
    edges.append((10, 3))
    try:
        build(nodes, edges)
    except RuntimeError as e:
        assert "two parents" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_planned_in_population_uses_enumeration_gates_not_deployed_constants():
    import pandas as pd
    pf = pd.DataFrame({"d_mb": [7.0, 10.0, 13.0], "d_ab": [9.0, 12.0, 12.0]})
    assert dsr.planned_in_population(pf, 8.0, 11.0).tolist() == [True, False, False]     # deployed 8/11
    assert dsr.planned_in_population(pf, 12.0, 14.0).tolist() == [True, True, False]     # the recorded enumeration


def test_proxy_values_price_each_label_exactly():
    import pandas as pd
    ctrl = {"De": 1000, "Je": 0.8, "Dd": 100, "Jd": 0.05}
    rows = {
        # oracle positive orphan: a new TP division plus the TP edge M->b
        "pos": dict(oracle_pos=True, gt_M=1, gt_b=2, gt_P=-1, b_has_parent=False, mb_is_gt_edge=True, pb_matched=False),
        # annotated mother, wrong daughter, orphan: a charged FP fork and an FP edge
        "fork": dict(oracle_pos=False, gt_M=1, gt_b=3, gt_P=-1, b_has_parent=False, mb_is_gt_edge=False, pb_matched=False),
        # steal breaking a matched P->b with an unannotated mother: the lost TP edge, and M->b is an
        # FP edge because b is annotated (pred_valid = out_valid | in_valid)
        "break": dict(oracle_pos=False, gt_M=-1, gt_b=2, gt_P=5, b_has_parent=True, mb_is_gt_edge=False, pb_matched=True),
        # everything unannotated: metric-free
        "free": dict(oracle_pos=False, gt_M=-1, gt_b=-1, gt_P=-1, b_has_parent=True, mb_is_gt_edge=False, pb_matched=False),
        # oracle positive steal from an unannotated P: the division, the TP edge, and the FP edge P->b removed
        "steal_pos": dict(oracle_pos=True, gt_M=1, gt_b=2, gt_P=-1, b_has_parent=True, mb_is_gt_edge=True, pb_matched=False),
    }
    t = pd.DataFrame(list(rows.values()))
    v = dsr.proxy_values(t, ctrl)
    assert abs(v[0] - (0.1 / 100 + 1 / 1000)) < 1e-12
    assert abs(v[1] - (-0.1 * 0.05 / 100 - 0.8 / 1000)) < 1e-12
    assert abs(v[2] - (-1 / 1000 - 0.8 / 1000)) < 1e-12
    assert v[3] == 0.0
    assert abs(v[4] - (0.1 / 100 + 1 / 1000 + 0.8 / 1000)) < 1e-12


def _toy_search_table(n_neg=60):
    import pandas as pd
    rng = np.random.default_rng(0)
    n = n_neg + 1
    t = pd.DataFrame({
        "dataset": ["c"] * n, "M_id": np.arange(n), "b_id": np.arange(n) + 1000,
        "d_mb": rng.uniform(6, 12, n), "d_ab": rng.uniform(6, 14, n), "d_ma": rng.uniform(0.5, 3, n),
        "divergence": rng.uniform(-1, 1, n), "d_pb": np.nan, "ratio_mb_pb": np.nan, "asym": rng.uniform(4, 10, n),
        "d_m_mid": rng.uniform(3, 6, n), "cos_daughters": rng.uniform(-0.4, 0.8, n), "d_ppred_b": np.nan,
        "cos_mb_next": rng.uniform(-1, 1, n), "cos_ab_next": rng.uniform(-1, 1, n),
        "n_t1_closer_than_b": rng.integers(1, 5, n), "n_t_closer_than_m": rng.integers(0, 4, n), "d_p_orphan": np.nan,
        "n_t1_within_parent_max": rng.integers(3, 8, n), "rank_score": rng.uniform(8, 13, n), "p_outdeg": 0,
        "p_len_before": -1, "b_len_after": rng.integers(2, 40, n), "a_len_after": rng.integers(2, 40, n),
        "m_len_before": rng.integers(2, 40, n), "c1_has_pred": True, "a_continues": True, "b_continues": True,
        "b_has_parent": False, "p_has_pred": False, "nn_a_is_b": False, "nn_b_is_a": False, "mutual_nn_all": False,
        "nearest_t_is_m": False, "oracle_pos": False, "gt_M": -1, "gt_b": -1, "gt_P": -1, "mb_is_gt_edge": False,
        "pb_matched": False,
    })
    # the single positive: strongly diverging, back-to-back daughters; a few annotated negatives share only one trait
    t.loc[0, ["divergence", "cos_daughters", "oracle_pos", "gt_M", "gt_b", "mb_is_gt_edge"]] = [3.0, -0.9, True, 7, 8, True]
    t.loc[1:5, "gt_M"] = 9          # annotated mothers = charged FP forks if fired
    t.loc[1:5, "divergence"] = 2.5  # they pass the divergence gate but not the cosine gate
    return t


def test_condition_menu_and_greedy_isolate_the_positive_on_the_train_fold():
    t = _toy_search_table()
    ctrl = {"De": 1000, "Je": 0.8, "Dd": 100, "Jd": 0.05}
    v = dsr.proxy_values(t, ctrl)
    menu = dsr.condition_menu(t)
    assert menu["divergence>=2.25"].sum() == 6 and menu["cos_daughters<=-0.7"].sum() == 1
    hist = dsr.greedy_conjunction(t, v, t, v, "orphan", max_steps=3, min_fired=1)
    final = hist[-1]
    assert final["train"]["tp_oracle"] == 1 and final["train"]["M_ann"] == 1      # only the positive's own mother
    assert final["train"]["proxy_dscore"] > hist[0]["train"]["proxy_dscore"]
    # the positive is separable on the daughter cosine (any threshold <= -0.5 isolates it) or on divergence
    assert any(c.startswith("cos_daughters<=") or c.startswith("divergence>=") for c in final["rule"][1:])
