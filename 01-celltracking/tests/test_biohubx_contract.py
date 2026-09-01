"""The Biohub-X I/O contract, proved by MUTATION rather than asserted.

Every test below takes a CLEAN artifact that passes, breaks exactly one thing, and proves the
contract refuses it. The clean cases come first: a suite in which everything refuses is not
evidence that the refusals are aimed at anything.

The defects chosen are not hypothetical. Each names the fact that recorded it happening:
  FACT-0432  an uncovered pair silently took 0.0 - the WORST score, not a neutral one
  FACT-0425  strict=False loaded a 411M-parameter model onto random weights, prints commented out
  FACT-0454  14 of 19 features zero-filled behind a correctly-shaped vector
  FACT-0457  an output slot that is torch.new_zeros, read as a learned score
  FACT-0459  a post-standardisation fill arriving as a finite constant, invisible to shape, NaN
             and no-zeros checks
  FACT-0428  a score handed to the incumbent relink survives on 15.4% / 30.3% of forced changes
  FACT-0344  the float export scored 0.914 against 0.928 - a -0.014 leaderboard reversal
  FACT-0040  the anisotropic (1.625, 0.40625, 0.40625) convention, and the isotropic decoy

Software contracts only (CLAUDE.md rule 4). Nothing here promotes or demotes a scientific result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

from biohubx import contract as C  # noqa: E402

RNG = np.random.default_rng(20260831)


# ------------------------------------------------------------------------------------------
# fixtures - small, and differing from production in COST, never in KIND
# ------------------------------------------------------------------------------------------
def clean_nodes(n=40, crops=("44b6_aaaa",), frames=4):
    rows = {"crop": [], "node_uid": [], "t": [], "instance_label": [],
            "z_vox": [], "y_vox": [], "x_vox": [],
            "center_confidence": [], "volume_vox": [], "division_prob": []}
    for c in crops:
        for t in range(frames):
            for i in range(n // frames):
                rows["crop"].append(c)
                rows["node_uid"].append(t * (n // frames) + i)
                rows["t"].append(t)
                rows["instance_label"].append(i + 1)
                rows["z_vox"].append(float(RNG.integers(0, 32)))
                rows["y_vox"].append(float(RNG.integers(0, 256)))
                rows["x_vox"].append(float(RNG.integers(0, 256)))
                rows["center_confidence"].append(float(RNG.uniform(0.5, 1.0)))
                rows["volume_vox"].append(int(RNG.integers(50, 400)))
                rows["division_prob"].append(float(RNG.uniform(0.0, 0.3)))
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["node_uid"] = out["node_uid"].astype(np.int64)
    out["t"] = out["t"].astype(np.int64)
    out["instance_label"] = out["instance_label"].astype(np.int64)
    out["volume_vox"] = out["volume_vox"].astype(np.int64)
    vox = np.stack([out["z_vox"], out["y_vox"], out["x_vox"]], 1)
    um = C.voxel_to_um(vox)
    out["z_um"], out["y_um"], out["x_um"] = um[:, 0], um[:, 1], um[:, 2]
    return out


def clean_candidates(n_targets=25, k=3):
    crop, t_t, tgt, src, pc, pd, pn = [], [], [], [], [], [], []
    for u in range(n_targets):
        for s in list(range(k)) + [C.NO_PARENT]:
            crop.append("44b6_aaaa"); t_t.append(1); tgt.append(1000 + u); src.append(s)
            v = RNG.dirichlet([4.0, 1.0, 2.0])
            pc.append(v[0]); pd.append(v[1]); pn.append(v[2])
    return {"crop": np.asarray(crop), "t_target": np.asarray(t_t),
            "target_uid": np.asarray(tgt), "source_uid": np.asarray(src),
            "p_continuation": np.asarray(pc), "p_division": np.asarray(pd),
            "p_neither": np.asarray(pn)}


def clean_graph(n=30, frames=3):
    crop, uid, par, t, z, y, x = [], [], [], [], [], [], []
    prev: list[int] = []
    nxt = 0
    for f in range(frames):
        cur = []
        for i in range(n // frames):
            crop.append("44b6_aaaa"); uid.append(nxt); t.append(f)
            par.append(prev[i] if (f > 0 and i < len(prev)) else C.NO_PARENT)
            z.append(int(RNG.integers(0, 32))); y.append(int(RNG.integers(0, 256)))
            x.append(int(RNG.integers(0, 256)))
            cur.append(nxt); nxt += 1
        prev = cur
    return {"crop": np.asarray(crop),
            "node_uid": np.asarray(uid, dtype=np.int64),
            "parent_uid": np.asarray(par, dtype=np.int64),
            "t": np.asarray(t, dtype=np.int64),
            "z_vox": np.asarray(z, dtype=np.int64),
            "y_vox": np.asarray(y, dtype=np.int64),
            "x_vox": np.asarray(x, dtype=np.int64)}


CLEAN_CHAIN = ["biohubx.detector.v0", "biohubx.association.v0", "biohubx.graph_builder.v0"]


def clean_complete_artifact(tmp_path):
    nodes = clean_nodes(n=30, frames=3)
    graph = {
        "crop": nodes["crop"].copy(),
        "node_uid": nodes["node_uid"].copy(),
        "parent_uid": np.full(30, C.NO_PARENT, dtype=np.int64),
        "t": nodes["t"].copy(),
        "z_vox": np.rint(nodes["z_vox"]).astype(np.int64),
        "y_vox": np.rint(nodes["y_vox"]).astype(np.int64),
        "x_vox": np.rint(nodes["x_vox"]).astype(np.int64),
    }
    per_frame = 10
    graph["parent_uid"][per_frame:] = graph["node_uid"][:-per_frame]

    rows = {k: [] for k in C.CANDIDATE_COLUMNS}
    for crop, tt, tgt, src in zip(graph["crop"], graph["t"], graph["node_uid"],
                                  graph["parent_uid"]):
        sources = [C.NO_PARENT] if src == C.NO_PARENT else [int(src), C.NO_PARENT]
        for source in sources:
            rows["crop"].append(crop); rows["t_target"].append(tt)
            rows["target_uid"].append(tgt); rows["source_uid"].append(source)
            v = RNG.dirichlet([4.0, 1.0, 2.0])
            rows["p_continuation"].append(v[0]); rows["p_division"].append(v[1])
            rows["p_neither"].append(v[2])
    candidates = {k: np.asarray(v) for k, v in rows.items()}
    for k in ("t_target", "target_uid", "source_uid"):
        candidates[k] = candidates[k].astype(np.int64)

    embeddings = RNG.normal(size=(30, 16))
    producer = tmp_path / "graph_builder.py"
    producer.write_text("def build_graph(candidates):\n    return candidates\n", encoding="utf-8")
    receipt = C.make_consumer_receipt(
        producer_path=producer, producer_symbol="build_graph", candidates=candidates,
        graph=graph, consumer_chain=CLEAN_CHAIN)
    offered = [(c, int(t), int(u), int(s)) for c, t, u, s in
               zip(candidates["crop"], candidates["t_target"],
                   candidates["target_uid"], candidates["source_uid"])]
    return nodes, embeddings, candidates, graph, receipt, offered


# ------------------------------------------------------------------------------------------
# clean cases
# ------------------------------------------------------------------------------------------
def test_clean_artifacts_all_pass():
    rn = C.verify_nodes(clean_nodes())
    rc = C.verify_candidates(clean_candidates())
    rg = C.verify_graph(clean_graph(), consumer_chain=CLEAN_CHAIN)
    re = C.verify_embeddings(RNG.normal(size=(40, 16)))
    assert rn["coordinate_round_trip_max_um"] == 0.0
    assert rc["n_targets"] == 25 and rc["abstain"]["constant"] is False
    assert rg["max_out_degree"] <= 2 and rg["relink_applied"] is False
    assert re["dead_dims"] == 0


# ------------------------------------------------------------------------------------------
# STAGE A - nodes and the coordinate convention
# ------------------------------------------------------------------------------------------
def test_isotropic_coordinates_are_refused():
    """The single mistake AGENTS.md section 4 records twice in one day."""
    nodes = clean_nodes()
    vox = np.stack([nodes["z_vox"], nodes["y_vox"], nodes["x_vox"]], 1)
    iso = vox * C.SCALE_UM[0]                      # isotropic 1.625 - shape-identical, wrong
    nodes["z_um"], nodes["y_um"], nodes["x_um"] = iso[:, 0], iso[:, 1], iso[:, 2]
    with pytest.raises(C.ContractRefusal, match="do not round-trip"):
        C.verify_nodes(nodes)


def test_transposed_axis_order_is_refused():
    nodes = clean_nodes()
    nodes["z_um"], nodes["x_um"] = nodes["x_um"].copy(), nodes["z_um"].copy()
    with pytest.raises(C.ContractRefusal, match="do not round-trip"):
        C.verify_nodes(nodes)


def test_float_instance_label_is_refused():
    nodes = clean_nodes()
    nodes["instance_label"] = nodes["instance_label"].astype(np.float64)
    with pytest.raises(C.ContractRefusal, match="INTEGER dtype"):
        C.verify_nodes(nodes)


def test_fractional_node_time_is_refused_before_casting():
    nodes = clean_nodes()
    nodes["t"] = nodes["t"].astype(np.float64) + 0.25
    with pytest.raises(C.ContractRefusal, match="INTEGER dtype"):
        C.verify_nodes(nodes)


def test_node_columns_with_different_lengths_are_refused_before_zip():
    nodes = clean_nodes()
    nodes["t"] = np.append(nodes["t"], 99)
    with pytest.raises(C.ContractRefusal, match="equal lengths"):
        C.verify_nodes(nodes)


def test_zero_volume_is_refused_as_a_points_only_producer():
    """FACT-0454: a points-only producer pretending to have masks emits exactly this."""
    nodes = clean_nodes()
    nodes["volume_vox"] = np.zeros_like(nodes["volume_vox"])
    with pytest.raises(C.ContractRefusal, match="not an instance"):
        C.verify_nodes(nodes)


def test_constant_division_prob_is_refused():
    """FACT-0457: output 3 is torch.new_zeros. A constant head passes every structural check."""
    nodes = clean_nodes()
    nodes["division_prob"] = np.zeros_like(nodes["division_prob"])
    with pytest.raises(C.ContractRefusal, match="CONSTANT"):
        C.verify_nodes(nodes)


def test_post_standardisation_fill_is_refused_although_it_is_not_zero():
    """FACT-0459 exactly: the fill arrives as the finite constant -mean/std, never as 0."""
    nodes = clean_nodes()
    nodes["center_confidence"] = np.full_like(nodes["center_confidence"], 0.4052)
    assert not np.any(nodes["center_confidence"] == 0)      # no-zeros check would PASS
    assert np.isfinite(nodes["center_confidence"]).all()    # NaN check would PASS
    with pytest.raises(C.ContractRefusal, match="FACT-0459"):
        C.verify_nodes(nodes)


def test_duplicate_instance_labels_within_a_frame_are_refused():
    nodes = clean_nodes()
    nodes["instance_label"] = np.ones_like(nodes["instance_label"])
    with pytest.raises(C.ContractRefusal, match="duplicate"):
        C.verify_nodes(nodes)


# ------------------------------------------------------------------------------------------
# the calibration gate must be able to FAIL
# ------------------------------------------------------------------------------------------
def _synthetic_gt(n=4000, target_um=C.CALIBRATION_ANCHOR_UM):
    a = RNG.uniform(0, 200, size=(n, 3))
    step = RNG.normal(size=(n, 3))
    step /= np.linalg.norm(step, axis=1, keepdims=True)
    b = a + (step * target_um) / np.asarray(C.SCALE_UM)
    return a, b


def test_calibration_gate_passes_and_discriminates():
    rep = C.calibration_gate(*_synthetic_gt())
    assert rep["correct_convention_passes"] and rep["wrong_convention_fails"]
    assert rep["gate_discriminates"] is True


def test_calibration_gate_refuses_a_wrong_convention():
    a, b = _synthetic_gt(target_um=4.0)
    with pytest.raises(C.ContractRefusal, match="FACT-0040 anchor"):
        C.calibration_gate(a, b)


def test_calibration_gate_refuses_a_sample_too_small_to_be_a_gate():
    a, b = _synthetic_gt(n=50)
    with pytest.raises(C.ContractRefusal, match="too few"):
        C.calibration_gate(a, b)


# ------------------------------------------------------------------------------------------
# STAGE B - embeddings
# ------------------------------------------------------------------------------------------
def test_partially_dead_embedding_is_refused_although_the_whole_array_varies():
    """The FACT-0454 shape: 14 of 19 slots constant behind a correctly-shaped vector."""
    e = RNG.normal(size=(60, 19))
    e[:, 5:] = 0.7331                                       # 14 of 19 constant
    assert np.std(e) > 0.1                                  # a whole-array check would PASS
    with pytest.raises(C.ContractRefusal, match="14 of 19 embedding dimension"):
        C.verify_embeddings(e)


def test_all_zero_embedding_row_is_refused():
    e = RNG.normal(size=(60, 8))
    e[7] = 0.0
    with pytest.raises(C.ContractRefusal, match="all-zero embedding row"):
        C.verify_embeddings(e)


def test_embedding_dim_mismatch_is_refused():
    with pytest.raises(C.ContractRefusal, match="declared dim"):
        C.verify_embeddings(RNG.normal(size=(20, 8)), expect_dim=16)


# ------------------------------------------------------------------------------------------
# STAGES C and D - abstention as a CLASS, and coverage that refuses
# ------------------------------------------------------------------------------------------
def test_missing_no_parent_row_is_refused():
    """Without it, abstention silently reverts to a threshold - the FACT-0457 mechanism."""
    cand = clean_candidates()
    keep = ~((cand["source_uid"] == C.NO_PARENT) & (cand["target_uid"] == 1000))
    cand = {k: v[keep] for k, v in cand.items()}
    with pytest.raises(C.ContractRefusal, match="NO explicit no-parent row"):
        C.verify_candidates(cand)


def test_duplicate_no_parent_row_is_refused():
    """A second abstain row on one target counts the no-parent mass twice."""
    cand = clean_candidates()
    i = int(np.flatnonzero(cand["source_uid"] == C.NO_PARENT)[0])
    dup = {k: np.concatenate([v, v[i:i + 1]]) for k, v in cand.items()}
    dup["source_uid"][-1] = C.NO_PARENT
    dup["target_uid"][-1] = dup["target_uid"][i]
    # distinguish it from the pair-uniqueness rule so the right refusal is under test
    dup["p_neither"][-1] = float(dup["p_neither"][i]) * 0.5
    dup["p_continuation"][-1] = 1.0 - dup["p_neither"][-1] - dup["p_division"][-1]
    with pytest.raises(C.ContractRefusal, match="duplicate"):
        C.verify_candidates(dup)


def test_a_pair_scored_twice_is_refused():
    cand = clean_candidates()
    dup = {k: np.concatenate([v, v[:1]]) for k, v in cand.items()}
    with pytest.raises(C.ContractRefusal, match="duplicate"):
        C.verify_candidates(dup)


def test_constant_abstain_mass_is_refused():
    """A no-parent head that was never trained emits a constant and looks entirely plausible."""
    cand = clean_candidates()
    is_np = cand["source_uid"] == C.NO_PARENT
    cand["p_neither"] = cand["p_neither"].copy()
    cand["p_continuation"] = cand["p_continuation"].copy()
    cand["p_neither"][is_np] = 0.25
    cand["p_continuation"][is_np] = 1.0 - 0.25 - cand["p_division"][is_np]
    with pytest.raises(C.ContractRefusal, match="CONSTANT"):
        C.verify_candidates(cand)


def test_three_independent_sigmoids_are_refused():
    cand = clean_candidates()
    cand["p_continuation"] = np.full_like(cand["p_continuation"], 0.9)
    cand["p_division"] = np.full_like(cand["p_division"], 0.8)
    cand["p_neither"] = np.full_like(cand["p_neither"], 0.7)
    with pytest.raises(C.ContractRefusal, match="must sum to 1"):
        C.verify_candidates(cand)


def test_uncovered_offered_pair_refuses_rather_than_filling():
    """FACT-0432: the uncovered pair took 0.0 - the WORST score - and nothing could tell."""
    cand = clean_candidates()
    offered = [("44b6_aaaa", 1, 1000, 0), ("44b6_aaaa", 1, 1000, 99)]   # 99 was never scored
    with pytest.raises(C.ContractRefusal, match="REFUSING rather than filling"):
        C.verify_candidates(cand, offered_pairs=offered)


def test_coverage_on_a_wrong_key_shape_is_refused():
    cand = clean_candidates()
    with pytest.raises(C.ContractRefusal, match="SAME key"):
        C.verify_candidates(cand, offered_pairs=[("44b6_aaaa", 1000, 0)])


def test_fractional_candidate_uid_is_refused_before_casting():
    cand = clean_candidates()
    cand["source_uid"] = cand["source_uid"].astype(np.float64)
    cand["source_uid"][0] += 0.25
    with pytest.raises(C.ContractRefusal, match="INTEGER dtype"):
        C.verify_candidates(cand)


def test_candidate_columns_with_different_lengths_are_refused_before_zip():
    cand = clean_candidates()
    cand["crop"] = np.append(cand["crop"], "44b6_aaaa")
    with pytest.raises(C.ContractRefusal, match="equal lengths"):
        C.verify_candidates(cand)


def test_full_coverage_passes():
    cand = clean_candidates()
    offered = [(c, int(t), int(u), int(s)) for c, t, u, s in
               zip(cand["crop"], cand["t_target"], cand["target_uid"], cand["source_uid"])]
    rep = C.verify_candidates(cand, offered_pairs=offered)
    assert rep["coverage"]["n_uncovered"] == 0


# ------------------------------------------------------------------------------------------
# STAGE E - the graph, and the consumer that must not appear
# ------------------------------------------------------------------------------------------
def test_incumbent_relink_in_the_consumer_chain_is_refused():
    """The whole reason Biohub-X exists (FACT-0428)."""
    with pytest.raises(C.ContractRefusal, match="INCUMBENT relink"):
        C.verify_graph(clean_graph(),
                       consumer_chain=["biohubx.association.v0", "motion_relink_edges"])


def test_relink_hidden_behind_a_case_change_is_still_refused():
    with pytest.raises(C.ContractRefusal, match="INCUMBENT relink"):
        C.verify_graph(clean_graph(), consumer_chain=["Notebook.Motion_Relink_Edges(v3)"])


def test_empty_consumer_chain_is_refused():
    with pytest.raises(C.ContractRefusal, match="does not say what produced it"):
        C.verify_graph(clean_graph(), consumer_chain=[])


def test_float_export_is_refused():
    """FACT-0344: -0.014 on the public leaderboard. The field-validated form is integer."""
    g = clean_graph()
    g["z_vox"] = g["z_vox"].astype(np.float64) + 0.5
    with pytest.raises(C.ContractRefusal, match="INTEGER dtype"):
        C.verify_graph(g, consumer_chain=CLEAN_CHAIN)


def test_integral_looking_float_export_is_also_refused():
    """A float dtype is not made safe merely because this sample happens to have .0 values."""
    g = clean_graph()
    g["z_vox"] = g["z_vox"].astype(np.float64)
    with pytest.raises(C.ContractRefusal, match="INTEGER dtype"):
        C.verify_graph(g, consumer_chain=CLEAN_CHAIN)


def test_graph_columns_with_different_lengths_are_refused_before_zip():
    g = clean_graph()
    g["crop"] = np.append(g["crop"], "44b6_aaaa")
    with pytest.raises(C.ContractRefusal, match="equal lengths"):
        C.verify_graph(g, consumer_chain=CLEAN_CHAIN)


def test_out_degree_three_is_refused():
    g = clean_graph()
    par = g["parent_uid"].copy()
    kids = np.flatnonzero(g["t"] == 1)[:3]
    par[kids] = int(g["node_uid"][0])
    g["parent_uid"] = par
    with pytest.raises(C.ContractRefusal, match="out-degree 3"):
        C.verify_graph(g, consumer_chain=CLEAN_CHAIN)


def test_a_division_out_degree_two_is_allowed():
    g = clean_graph()
    par = g["parent_uid"].copy()
    kids = np.flatnonzero(g["t"] == 1)[:2]
    par[kids] = int(g["node_uid"][0])
    g["parent_uid"] = par
    rep = C.verify_graph(g, consumer_chain=CLEAN_CHAIN)
    assert rep["divisions"] >= 1 and rep["max_out_degree"] == 2


def test_frame_skipping_edge_is_refused():
    g = clean_graph(frames=3)
    par = g["parent_uid"].copy()
    kid = int(np.flatnonzero(g["t"] == 2)[0])
    par[kid] = int(g["node_uid"][np.flatnonzero(g["t"] == 0)[0]])
    g["parent_uid"] = par
    with pytest.raises(C.ContractRefusal, match="do not span exactly one frame"):
        C.verify_graph(g, consumer_chain=CLEAN_CHAIN)


def test_dangling_parent_is_refused():
    g = clean_graph()
    par = g["parent_uid"].copy()
    par[-1] = 999999
    g["parent_uid"] = par
    with pytest.raises(C.ContractRefusal, match="not in the table"):
        C.verify_graph(g, consumer_chain=CLEAN_CHAIN)


# ------------------------------------------------------------------------------------------
# THE FREEZE. These constants were fixed BEFORE any model existed, on 2026-08-31, so that no
# result can be obtained and then have the contract adjusted to fit it. Changing one is a NEW
# contract version (biohubx_io_v2) and a new preregistration - never an edit to this test.
# ------------------------------------------------------------------------------------------
def test_the_contract_is_frozen():
    assert C.CONTRACT_VERSION == "biohubx_io_v1"
    assert C.SCALE_UM == (1.625, 0.40625, 0.40625)          # FACT-0040, NOT isotropic
    assert C.CLASSES == ("continuation", "division", "neither")
    assert C.NO_PARENT == -1
    assert C.MATCH_RADIUS_UM == 7.0                         # the official matcher
    assert C.CALIBRATION_ANCHOR_UM == 1.81681               # FACT-0040 via FACT-0447
    assert C.CALIBRATION_ISOTROPIC_DECOY_UM == 5.13870      # the value that must FAIL
    assert "motion_relink_edges" in C.FORBIDDEN_CONSUMERS   # FACT-0428 - the fourth arrow


def test_partial_report_cannot_emit_the_success_heartbeat(tmp_path):
    rep = C.ContractReport().add(C.verify_graph(clean_graph(), consumer_chain=CLEAN_CHAIN))
    with pytest.raises(C.ContractRefusal, match="missing="):
        rep.to_json(tmp_path / "partial.json")
    assert not (tmp_path / "partial.json").exists()


def test_complete_cross_stage_artifact_round_trips_to_json(tmp_path):
    nodes, emb, cand, graph, receipt, offered = clean_complete_artifact(tmp_path)
    rep = C.verify_complete_artifact(
        nodes=nodes, embeddings=emb, candidates=cand, graph=graph,
        consumer_receipt=receipt, offered_pairs=offered)
    p = rep.to_json(tmp_path / "report.json")
    import json
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["heartbeat"] == "BIOHUBX_CONTRACT_OK"
    assert set(payload["stages"]) == C.ContractReport.REQUIRED_STAGES


def test_cross_stage_refuses_unknown_candidate_node(tmp_path):
    nodes, emb, cand, graph, _, _ = clean_complete_artifact(tmp_path)
    cand["target_uid"] = cand["target_uid"].copy()
    cand["target_uid"][0] = 999999
    with pytest.raises(C.ContractRefusal, match="unknown_targets"):
        C.verify_cross_stage(nodes, emb, cand, graph)


def test_cross_stage_refuses_an_emitted_edge_that_was_not_offered(tmp_path):
    nodes, emb, cand, graph, _, _ = clean_complete_artifact(tmp_path)
    child = int(np.flatnonzero(graph["parent_uid"] != C.NO_PARENT)[0])
    key = (graph["crop"][child], graph["t"][child], graph["node_uid"][child],
           graph["parent_uid"][child])
    keep = ~((cand["crop"] == key[0]) & (cand["t_target"] == key[1]) &
             (cand["target_uid"] == key[2]) & (cand["source_uid"] == key[3]))
    cand = {k: v[keep] for k, v in cand.items()}
    with pytest.raises(C.ContractRefusal, match="never offered"):
        C.verify_cross_stage(nodes, emb, cand, graph)


def test_cross_stage_refuses_embedding_row_mismatch(tmp_path):
    nodes, emb, cand, graph, _, _ = clean_complete_artifact(tmp_path)
    with pytest.raises(C.ContractRefusal, match="row-aligned"):
        C.verify_cross_stage(nodes, emb[:-1], cand, graph)


def test_consumer_receipt_refuses_a_forbidden_source_even_with_a_clean_declared_chain(tmp_path):
    _, _, cand, graph, _, _ = clean_complete_artifact(tmp_path)
    producer = tmp_path / "bad_builder.py"
    producer.write_text(
        "def build_graph(candidates):\n"
        "    return motion_relink_edges(candidates)\n",
        encoding="utf-8")
    with pytest.raises(C.ContractRefusal, match="forbidden incumbent consumer"):
        C.make_consumer_receipt(
            producer_path=producer, producer_symbol="build_graph", candidates=cand,
            graph=graph, consumer_chain=CLEAN_CHAIN)


def test_consumer_receipt_is_bound_to_the_exact_graph(tmp_path):
    _, _, cand, graph, receipt, _ = clean_complete_artifact(tmp_path)
    graph["x_vox"] = graph["x_vox"].copy()
    graph["x_vox"][0] += 1
    with pytest.raises(C.ContractRefusal, match="graph does not match"):
        C.verify_consumer_provenance(receipt, candidates=cand, graph=graph)
