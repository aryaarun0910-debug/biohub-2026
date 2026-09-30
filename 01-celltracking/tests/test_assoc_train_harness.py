"""Contract tests for the association training harness (PKT-0034 / PKT-0038) that RUN it.

WHY THIS SHAPE. This tree has already been burned by a test that grepped source instead of running
it, by a Gate-1 attempt that compared zero crops while looking well formed (FACT-0387), and - the
one this file is a direct response to - by a fixture that stubbed `model.detection_head`, a method
the real class has never had, so a green suite certified a harness against a model that did not
exist. So every test here builds a real cache on disk IN THE TAP'S PRODUCTION LAYOUT
(`tests/assoc_v2_world.py`, itself required to pass `audit_feature_cache`), a real pre-ILP parquet,
real ECB sidecars, a real Gate-1 receipt and a real bound manifest, and then executes the harness
against it.

SCHEMA V2 - THE MIGRATION THESE TESTS ENFORCE (`FACT-0402`). The retired layout keyed features BY
FRAME and the harness scattered them into one `(n_nodes, dim)` matrix. That object does not exist:
`TemporalUNet3D._TemporalAttention` mixes across the window's time axis, so at the deployed
`window_size` 2 an interior node is the TARGET of one pair and the SOURCE of the next and carries
two different vectors. Every rejection below is PROVEN BY MUTATION - the defect is manufactured on
a production-layout cache and the harness must refuse it under its own NAMED condition - and every
mutation has an ACCEPT CONTROL, because a harness that refuses everything checks nothing.

CONTRACT 2 - THE SECOND MIGRATION THESE TESTS ENFORCE (`FACT-0407`, `FACT-0408`). Schema 2 fixed
the KEY (pair-and-role, not frame). Contract 2 fixes the NAME: contract 1's one probability column
per band held the deployed POST-fusion value under a name that claimed no surface, so a
primary-side replay compared against it was measuring two different stages. The harness inherits
the key list from the auditor (`assoc_train_harness.py:211`), so this suite's fixture had to move
with it. THE ARCHIVED P36 CACHE IS CONTRACT 1 AND IS NOW REFUSED; see
`test_the_archived_p36_cache_is_refused_as_contract_1_and_says_which_contract` for what was
retired and why no legacy read mode exists.

These are SOFTWARE contract tests. They decide nothing scientific.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import assoc_train_harness as H  # noqa: E402
import assoc_v2_world as W  # noqa: E402
import audit_feature_cache as AFC  # noqa: E402
from assoc_parent_dataset import evaluate  # noqa: E402

CROPS = W.CROPS
DIM = W.DIM

# The archived P36 cache: the only REAL artifact this suite can reach on CPU, and it is a
# CONTRACT-1 file. `FACT-0407` proved it FAITHFUL - a primary-only replay reproduces the recorded
# GPU verdict to seven digits and the deployed bidirectional harmonic collapses the gap to inside
# the gate's own tolerance - but faithful is not readable: contract 1 stored ONE probability
# column per band, and the pre-fusion surface it never held cannot be recovered by rewriting these
# bytes. It keeps its EVIDENTIARY value (it is the artifact FACT-0403, FACT-0406 and FACT-0407
# were measured on) and has lost its SCHEMA-FIXTURE and TRAINING-LICENCE value. The one test that
# still reads it asserts that it is REFUSED, by name.
P36_CACHE = Path("C:/temp/assoc_tournament/schema_v2/aft_cache")
P36_RECEIPT = Path("C:/temp/p36/assoc_feature_tap_gate.json")
needs_p36 = pytest.mark.skipif(
    not (P36_CACHE.is_dir() and list(P36_CACHE.glob("*.npz"))),
    reason="the archived P36 cache is not extracted at C:/temp/assoc_tournament/schema_v2/aft_cache")

REPRODUCER_SRC = '''
"""A declared reproducer that re-derives the recorded probabilities FROM THE CACHE ALONE.

It reads the ROLE-SPECIFIC feature blocks of each recorded pair - `role_feat` sliced by
`pair_src_ptr` and `pair_tgt_ptr` - and applies the deployed softmax over the SOURCE axis. It
never touches a per-frame array, because there is no such thing in schema 2.
"""
import numpy as np


def reproduce(crop, cache):
    out = {}
    n_pairs = int(cache["pair_f_idx"].shape[0])
    gid = np.asarray(cache["role_gid"]).astype(np.int64)
    feat = np.asarray(cache["role_feat"]).astype(np.float64)
    for pair in range(n_pairs):
        sp = int(cache["pair_src_ptr"][pair]); sn = int(cache["pair_src_n"][pair])
        tp = int(cache["pair_tgt_ptr"][pair]); tn = int(cache["pair_tgt_n"][pair])
        logits = feat[sp:sp + sn] @ feat[tp:tp + tn].T
        e = np.exp(logits - logits.max(axis=0, keepdims=True))
        p = e / e.sum(axis=0, keepdims=True)
        for i in range(sn):
            for j in range(tn):
                out[(int(gid[sp + i]), int(gid[tp + j]))] = float(p[i, j])
    return out
'''


@pytest.fixture(scope="module")
def repro_module(tmp_path_factory):
    d = tmp_path_factory.mktemp("repro")
    (d / "synth_repro.py").write_text(REPRODUCER_SRC, encoding="utf-8")
    sys.path.insert(0, str(d))
    return "callable:synth_repro:reproduce"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return W.build_world(tmp_path_factory.mktemp("world"))


@pytest.fixture(scope="module")
def surface(world):
    return W.build_surface(world)


def _gate(world, **kw):
    r = H.gate_cache(cache_dir=world["cache_dir"], preilp=world["preilp"],
                     ecb_dir=world["ecb_dir"], crops=None, receipt=world["receipt"],
                     reproducer="receipt", manifest=world["manifest"], **kw)
    assert r["passed"], (r["refusals"], [c["reasons"] for c in r["crops"] if not c["passed"]])
    return r


def _role_index(world):
    return {c: H.role_features(H.load_cache(world["cache_dir"] / f"{c}.npz"), c) for c in CROPS}


def _rebuild_union(payload: dict) -> None:
    """Re-derive the auditor-facing union after a mutation edits the bands it is made of.

    Driven by ``AFC.UNION_BANDS`` rather than a literal ``("a", "b")``: band P is a gate
    instrument and is deliberately NOT in the deployed candidate surface, and a mutation that
    quietly folded it in would inflate that surface by rows the deployment never saw. A mutation
    that leaves the union stale fails under ``union_surface_disagrees_with_its_bands``, which is
    a different defect from the one each caller is manufacturing.
    """
    payload["source_id"] = np.concatenate(
        [payload[f"band_{b}_source_id"] for b in AFC.UNION_BANDS])
    payload["target_id"] = np.concatenate(
        [payload[f"band_{b}_target_id"] for b in AFC.UNION_BANDS])
    payload["deployed_edge_prob_postblend"] = np.concatenate(
        [payload[f"band_{b}_deployed_prob_postblend"] for b in AFC.UNION_BANDS]).astype(np.float32)
    payload["primary_edge_prob_preblend"] = np.concatenate(
        [payload[f"band_{b}_primary_prob_preblend"] for b in AFC.UNION_BANDS]).astype(np.float32)


def _synth(dir_: Path, name: str = "44b6_aaaaaaaa", **kw) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    AFC._synth_cache(dir_ / f"{name}.npz", crop=name, trunk_seed=7, **kw)
    return dir_ / f"{name}.npz"


def chain_rows(tp, fp, fn, n_pred, n_est, div=(1, 1, 1)):
    denom = tp + fp + fn
    ratio = (n_pred - n_est) / n_est
    jac = tp / denom
    return {"edge_tp": tp, "edge_fp": fp, "edge_fn": fn,
            "division_tp": div[0], "division_fp": div[1], "division_fn": div[2],
            "num_pred_nodes": n_pred, "node_recall": 0.99, "total_node_ratio": ratio,
            "edge_jaccard": jac, "adj_edge_jaccard": max(0.0, jac * (1 - 0.1 * ratio))}


def fake_summarise(rows):
    import assoc_report as ar

    tp = sum(r["edge_tp"] for r in rows)
    fp = sum(r["edge_fp"] for r in rows)
    fn = sum(r["edge_fn"] for r in rows)
    dtp = sum(r["division_tp"] for r in rows)
    dfp = sum(r["division_fp"] for r in rows)
    dfn = sum(r["division_fn"] for r in rows)
    w = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in rows]
    adj = sum(a * r["adj_edge_jaccard"] for a, r in zip(w, rows)) / sum(w)
    dj = dtp / (dtp + dfp + dfn) if (dtp + dfp + dfn) else 0.0
    return {"edge_jaccard": tp / (tp + fp + fn), "adj_edge_jaccard": adj, "division_jaccard": dj,
            "division_tp": dtp, "division_fp": dfp, "division_fn": dfn,
            "node_recall": sum(r["node_recall"] for r in rows) / len(rows),
            "score": adj + ar.SCORE_DIVISION_WEIGHT * dj}


# ======================================================================================
# 0. THE SCHEMA MIGRATION - defect 6, proven by mutation, with accept controls
# ======================================================================================

def test_the_fixture_is_the_production_layout_and_not_a_convenient_approximation(world):
    """If this fails, every other test in this file is certifying a layout nothing writes."""
    report = AFC.audit(world["cache_dir"], world["manifest"])
    assert report["passed"] and len(report["checks"]) > 50
    assert all(m["dual_role_nodes"] > 0 and m["dual_role_max_abs_delta"] > 0.0
               for m in report["measured"]), "no dual-role node: the window mixing is not modelled"
    assert set(H.CACHE_KEYS) <= set(np.load(world["cache_dir"] / f"{CROPS[0]}.npz").files)


def test_a_frame_keyed_cache_is_rejected_under_its_own_named_condition(tmp_path):
    """Defect 6, part one: the RETIRED ARTIFACT. Reject, and say which mechanism condemns it."""
    bad = _synth(tmp_path / "frame_keyed", frame_keyed=True)
    with pytest.raises(H.HarnessRefusal) as exc:
        H.load_cache(bad)
    assert H.FRAME_KEYED_CACHE in str(exc.value)
    assert "feat_frames" in str(exc.value) and "FACT-0402" in str(exc.value)
    # ACCEPT CONTROL: the same generator, without the mutation, must load.
    good = _synth(tmp_path / "clean")
    assert H.load_cache(good)["role_feat"].shape[0] > 0


def test_the_retired_frame_keyed_consumption_path_is_a_named_refusal():
    """Deleting node_features would give a caller an AttributeError and no diagnosis."""
    with pytest.raises(H.HarnessRefusal) as exc:
        H.node_features({"coords": np.zeros((1, 4))})
    assert H.FRAME_KEYED_CONSUMPTION in str(exc.value)
    assert "role_features" in str(exc.value)


def test_a_node_carries_a_different_vector_in_each_of_its_two_roles(world):
    """The positive contract. An interior node is a SOURCE in one pair and a TARGET in the next."""
    idx = _role_index(world)[CROPS[0]]
    frame, i = 2, 5                       # interior: source of pair 2, target of pair 1
    gid = frame * W.N_PER_FRAME + i
    as_source, _ = idx.rows_for_edges([gid], [gid + W.N_PER_FRAME])
    _, as_target = idx.rows_for_edges([gid - W.N_PER_FRAME], [gid])
    assert int(as_source[0]) != int(as_target[0]), "one node resolved to one row in both roles"
    src_vec, tgt_vec = idx.feat[as_source[0]], idx.feat[as_target[0]]
    assert not np.array_equal(src_vec, tgt_vec)
    assert np.array_equal(src_vec, W.source_role_features(frame)[i])
    assert np.array_equal(tgt_vec, W.target_role_features(frame)[i])


def test_a_cache_whose_dual_role_vectors_are_identical_is_rejected(tmp_path):
    """Defect 6, part three: pair-and-role SHAPE with frame-keyed CONTENT. A shape check misses it."""
    d = tmp_path / "frame_keyed_content"
    _synth(d, frame_keyed_features=True)
    r = H.schema_report(d)
    assert not r["schema_ok"]
    assert any("role_features_frame_keyed" in reason
               for c in r["crops"] for reason in c["reasons"])
    # ACCEPT CONTROL
    assert H.schema_report(Path(_synth(tmp_path / "clean").parent))["schema_ok"]


def test_a_cache_that_de_duplicates_a_nodes_two_roles_is_rejected(tmp_path):
    """Defect 6, part two: the old worker's `feats[t]` de-duplication, wearing the new schema."""
    d = tmp_path / "deduped"
    _synth(d, dedupe_roles=True)
    r = H.schema_report(d)
    assert not r["schema_ok"]
    assert any("role_records_missing" in reason for c in r["crops"] for reason in c["reasons"])


@pytest.mark.parametrize("mutation,condition", [
    ({"reordered_roles": True}, "role_node_order_is_not_the_deployed_frame_slice"),
    ({"bad_mask": True}, "role_mask_has_a_false_slot"),
    ({"torn_scaled": True}, "role_coord_scaled_disagrees_with_role_coord_rel"),
    ({"wrong_relative_time": True}, "role_relative_time_wrong"),
    ({"duplicate_role": True}, "duplicate_role_record"),
    ({"torn_union": True}, "union_surface_disagrees_with_its_bands"),
])
def test_node_ordering_masks_positions_and_bands_are_each_validated(tmp_path, mutation, condition):
    """Each defect manufactured on a production-layout cache; each must fail under ITS OWN name."""
    d = tmp_path / condition
    _synth(d, **mutation)
    r = H.schema_report(d)
    assert not r["schema_ok"], f"{condition} was accepted"
    assert any(condition in reason for c in r["crops"] for reason in c["reasons"]), \
        [c["reasons"] for c in r["crops"]]


def test_a_sub_threshold_band_over_its_rank_cap_is_rejected(tmp_path):
    """The ECB acquisition rule is top-k per TARGET; an uncapped band is not the deployed surface."""
    def over_cap(payload):
        i = 0                                    # duplicate one band-B row past the cap
        rep = 9
        for col in AFC.BAND_COLUMNS:
            arr = payload[f"band_b_{col}"]
            payload[f"band_b_{col}"] = np.concatenate([arr, np.repeat(arr[i:i + 1], rep)])
        _rebuild_union(payload)

    d = tmp_path / "overcap"
    W.write_v2_cache(d / f"{CROPS[0]}.npz", CROPS[0], mutate=over_cap)
    r = H.schema_report(d)
    assert not r["schema_ok"]
    assert any("band_b_exceeds_its_rank_cap" in reason
               for c in r["crops"] for reason in c["reasons"])


def test_a_cache_without_positional_features_is_rejected(tmp_path, world):
    """role_pos is a head INPUT; a cache without it forces a re-derivation in place of an artifact."""
    d = tmp_path / "nopos"
    d.mkdir()
    with np.load(world["cache_dir"] / f"{CROPS[0]}.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files if k != "role_pos"}
    np.savez_compressed(d / f"{CROPS[0]}.npz", **data)
    with pytest.raises(H.HarnessRefusal, match="positional_features_absent"):
        H.load_cache(d / f"{CROPS[0]}.npz")


def test_feature_and_positional_widths_must_match_what_the_cache_declares(tmp_path, world):
    d = tmp_path / "narrow"
    d.mkdir()
    with np.load(world["cache_dir"] / f"{CROPS[0]}.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["role_pos"] = data["role_pos"][:, :-1]
    np.savez_compressed(d / f"{CROPS[0]}.npz", **data)
    r = H.schema_report(d)
    assert not r["schema_ok"]
    assert any("pos_dim_disagrees" in reason for c in r["crops"] for reason in c["reasons"])


# ======================================================================================
# 1. ROLE-SPECIFIC CONSUMPTION - the thing the retired layout could not express
# ======================================================================================

def test_the_contextual_contract_takes_the_role_vector_of_the_candidates_own_pair(world, surface):
    """And the numbers DIFFER from what a frame-keyed scatter would have produced.

    This is the assertion that makes the migration more than a rename: build the pair features the
    migrated way, then build them the retired way (one vector per node) and require the two to
    disagree. If they agreed, the defect would have been harmless and FACT-0402 would be wrong.
    """
    contract = H.ContextContract.from_dict(
        {"name": "roles", "dim": DIM, "pair_builder": "concat_src_tgt"})
    idx = _role_index(world)
    got = contract.build(idx, surface)

    crops = surface["crop"].to_numpy()
    src = surface["source"].to_numpy().astype(np.int64)
    tgt = surface["target"].to_numpy().astype(np.int64)
    frame_keyed = np.empty_like(got)
    for crop in np.unique(crops):
        m = crops == crop
        index = idx[str(crop)]
        # THE RETIRED SCATTER: one vector per node, whichever role was seen first.
        seen: dict[int, int] = {}
        for row, g in zip(range(index.feat.shape[0]), index.role_gid.tolist()):
            seen.setdefault(int(g), row)
        rows_s = [seen[int(g)] for g in src[m]]
        rows_t = [seen[int(g)] for g in tgt[m]]
        frame_keyed[m] = np.concatenate([index.feat[rows_s], index.feat[rows_t]], axis=1)
    assert not np.array_equal(got, frame_keyed), \
        "the role-specific and frame-keyed assemblies agree; the fixture models no window mixing"
    # and the migrated half is the one that matches the role table
    for crop in np.unique(crops)[:1]:
        m = crops == crop
        index = idx[str(crop)]
        s_rows, t_rows = index.rows_for_edges(src[m], tgt[m])
        role = np.asarray(H.load_cache(world["cache_dir"] / f"{crop}.npz")["role_role"])
        assert np.all(role[s_rows] == H.ROLE_SRC) and np.all(role[t_rows] == H.ROLE_TGT)


def test_a_candidate_spanning_non_consecutive_frames_is_refused(world):
    """The deployed loop only scores consecutive frames, so such an edge has no (pair, role)."""
    idx = _role_index(world)[CROPS[0]]
    with pytest.raises(H.HarnessRefusal) as exc:
        idx.rows_for_edges([0], [2 * W.N_PER_FRAME])       # frame 0 -> frame 2
    assert H.CANDIDATE_NOT_A_DEPLOYED_PAIR in str(exc.value)


def test_a_candidate_in_a_pair_the_cache_never_recorded_is_refused(tmp_path, world):
    """A partially tapped cache covers only the pairs it captured - the BIOHUB_AFT_MAX_PAIRS case."""
    def keep_two_pairs(payload):
        keep_rows = 4 * W.N_PER_FRAME                      # pairs 0 and 1, src+tgt blocks
        for key in ("pair_f_idx", "pair_t_src", "pair_t_tgt", "pair_src_ptr", "pair_src_n",
                    "pair_tgt_ptr", "pair_tgt_n", "pair_window_shape"):
            payload[key] = payload[key][:2]
        for key in ("role_pair", "role_role", "role_gid", "role_feat", "role_coord_scaled",
                    "role_coord_rel", "role_mask", "role_pos"):
            payload[key] = payload[key][:keep_rows]
        # ALL THREE bands are truncated, band P included: it is the pre-fusion analogue of band
        # A over the same pairs, so leaving it whole would name pairs the cache no longer holds.
        for name in AFC.BANDS:
            m = payload[f"band_{name}_pair"] < 2
            for col in AFC.BAND_COLUMNS:
                payload[f"band_{name}_{col}"] = payload[f"band_{name}_{col}"][m]
        _rebuild_union(payload)

    d = tmp_path / "partial"
    W.write_v2_cache(d / f"{CROPS[0]}.npz", CROPS[0], mutate=keep_two_pairs)
    assert H.schema_report(d)["schema_ok"], "the truncated cache must still be well formed"
    idx = H.role_features(H.load_cache(d / f"{CROPS[0]}.npz"), CROPS[0])
    idx.rows_for_edges([0], [W.N_PER_FRAME])               # pair 0 is present
    with pytest.raises(H.HarnessRefusal) as exc:
        idx.rows_for_edges([3 * W.N_PER_FRAME], [4 * W.N_PER_FRAME])   # pair 3 was not tapped
    assert H.NO_CACHED_PAIR in str(exc.value)


def test_a_role_row_that_does_not_carry_its_node_id_is_refused(world):
    """The row arithmetic is verified against role_gid, never trusted."""
    idx = _role_index(world)[CROPS[0]]
    idx.role_gid = idx.role_gid.copy()
    idx.role_gid[0] += 1
    with pytest.raises(H.HarnessRefusal, match=H.ROLE_LOOKUP_DISAGREES):
        idx.rows_for_edges([0], [W.N_PER_FRAME])


# ======================================================================================
# 2. THE CACHE GATE - constraint 1, falsifier (a)
# ======================================================================================

def test_gate_passes_on_a_faithful_cache_with_a_declared_reproducer(world, repro_module):
    r = H.gate_cache(cache_dir=world["cache_dir"], preilp=world["preilp"],
                     ecb_dir=world["ecb_dir"], crops=None, receipt=None,
                     reproducer=repro_module, manifest=world["manifest"])
    assert r["passed"], [c["reasons"] for c in r["crops"] if not c["passed"]]
    assert all(c["band_b"]["checked"] > 0 for c in r["crops"]), "band B was never compared"
    assert all(c["band_a"]["missing"] == 0 and c["band_a"]["extra"] == 0 for c in r["crops"])
    assert all(c["dual_role_nodes"] > 0 for c in r["crops"])
    # the payload must say, in the artifact, that this is NOT the GPU-side numeric proof
    assert r["probability_parity_proof"].startswith("in_process_reproducer")
    assert r["schema_version"] == 2 and r["layout"] == "pair_and_role"


def test_a_perturbed_role_feature_breaks_the_reproduction_and_the_trunk_binding(tmp_path, world,
                                                                                repro_module):
    """Perturb ONE role-specific feature row and require BOTH independent guards to fire.

    The two are not the same check and neither subsumes the other. The MANIFEST binding sees that
    the features on disk no longer digest to the value bound against the trunk - which is
    `FACT-0392`'s confound, and is what a swapped trunk looks like. The REPRODUCER sees that the
    recorded probabilities are no longer what those features produce. Rebinding a manifest to the
    mutated cache silences the first and must NOT silence the second.
    """
    bad = tmp_path / "cache"
    bad.mkdir()
    for p in sorted(world["cache_dir"].glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        d["role_feat"] = d["role_feat"].copy()
        d["role_feat"][0] += 0.75
        np.savez_compressed(bad / p.name, **d)

    with_old_manifest = H.gate_cache(
        cache_dir=bad, preilp=world["preilp"], ecb_dir=world["ecb_dir"], crops=None,
        receipt=None, reproducer=repro_module, manifest=world["manifest"])
    assert not with_old_manifest["passed"]
    assert any("features_still_bound" in x or "features_unchanged" in x
               for x in with_old_manifest["refusals"]), with_old_manifest["refusals"]

    rebound = bad / "cache_manifest.json"
    rebound.write_text(json.dumps(AFC.build_manifest(AFC._bind_args(
        bad, world["trunk"], fold="0", role="official",
        weights_glob="loeo_official_f0_e3/split_0/*.pth", notebook=world["notebook"])), indent=2),
        encoding="utf-8")
    r = H.gate_cache(cache_dir=bad, preilp=world["preilp"], ecb_dir=world["ecb_dir"],
                     crops=None, receipt=None, reproducer=repro_module, manifest=rebound)
    assert not r["passed"], "rebinding the manifest silenced the reproducer too"
    assert any("probability delta" in reason for c in r["crops"] for reason in c["reasons"]), \
        [c["reasons"] for c in r["crops"]]


def test_gate_fails_when_the_cache_coordinates_are_not_the_recorded_nodes(tmp_path, world):
    """A cache that describes different nodes must not license a head. The CPU-side teeth."""
    bad_pre = tmp_path / "preilp.parquet"
    rows = pl.read_parquet(world["preilp"]).to_dicts()
    for row in rows:
        if row["row_type"] == "node" and row["node_id"] == 3:
            row["y"] = row["y"] + 7.0
    pl.DataFrame(rows, schema=W.PREILP_SCHEMA).write_parquet(bad_pre)
    r = H.gate_cache(cache_dir=world["cache_dir"], preilp=bad_pre, ecb_dir=world["ecb_dir"],
                     crops=None, receipt=world["receipt"], reproducer="receipt",
                     manifest=world["manifest"])
    assert not r["passed"]
    assert any("coordinates disagree" in reason for c in r["crops"] for reason in c["reasons"])


def test_coordinate_parity_applies_the_deployed_downsample_rescale(tmp_path, world):
    """THE SCHEMA-1 DEFECT IN THIS FILE'S OWN GATE, now a test.

    The tap flushes before the deployed rescale, so the cache holds the downsampled grid while the
    pre-ILP export holds `coords[:, 1:] * downsample` (predict_unet_transformer.py:798-802). The
    old check compared them RAW and passed only because its fixture used downsample (1,1,1). A
    pre-ILP table written in the cache's own grid must now be REFUSED.
    """
    raw_pre = tmp_path / "raw_preilp.parquet"
    rows = []
    for crop in CROPS:
        with np.load(world["cache_dir"] / f"{crop}.npz", allow_pickle=False) as z:
            coords = z["coords"]
        rows += [{"dataset": crop, "row_type": "node", "node_id": nid,
                  "t": int(coords[nid, 0]), "z": float(coords[nid, 1]),
                  "y": float(coords[nid, 2]), "x": float(coords[nid, 3]),
                  "source_id": None, "target_id": None, "edge_prob": None}
                 for nid in range(coords.shape[0])]
    pl.DataFrame(rows, schema=W.PREILP_SCHEMA).write_parquet(raw_pre)
    r = H.gate_cache(cache_dir=world["cache_dir"], preilp=raw_pre, ecb_dir=world["ecb_dir"],
                     crops=None, receipt=world["receipt"], reproducer="receipt",
                     manifest=world["manifest"])
    assert not r["passed"]
    assert any("downsample rescale" in reason for c in r["crops"] for reason in c["reasons"])
    # ACCEPT CONTROL: the correctly rescaled table passes.
    assert _gate(world)["passed"]


def test_gate_fails_when_a_frame_is_short(tmp_path, world):
    """A truncated detection silently shrinks the denominator; the gate must refuse, not pass."""
    short = tmp_path / "short.parquet"
    rows = [r for r in pl.read_parquet(world["preilp"]).to_dicts()
            if not (r["row_type"] == "node" and r["dataset"] == CROPS[0]
                    and r["node_id"] == W.NODES_PER_CROP - 1)]
    pl.DataFrame(rows, schema=W.PREILP_SCHEMA).write_parquet(short)
    r = H.gate_cache(cache_dir=world["cache_dir"], preilp=short, ecb_dir=world["ecb_dir"],
                     crops=[CROPS[0]], receipt=world["receipt"], reproducer="receipt",
                     manifest=world["manifest"])
    assert not r["passed"]
    assert any("node count" in reason for c in r["crops"] for reason in c["reasons"])


def test_gate_fails_when_the_sub_threshold_band_disagrees(tmp_path, world, repro_module):
    """FACT-0382: all the contested errors live below 0.5, so band B is the band that matters."""
    ecb = tmp_path / "ecb"
    ecb.mkdir()
    for p in sorted(world["ecb_dir"].glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        prob = d["edge_prob"].copy()
        prob[0] = float(prob[0]) + 0.01
        d["edge_prob"] = prob
        np.savez_compressed(ecb / p.name, **d)
    r = H.gate_cache(cache_dir=world["cache_dir"], preilp=world["preilp"], ecb_dir=ecb,
                     crops=None, receipt=None, reproducer=repro_module,
                     manifest=world["manifest"])
    assert not r["passed"]
    assert any("band B probability delta" in reason
               for c in r["crops"] for reason in c["reasons"])


def test_gate_refuses_without_a_gate1_receipt(world):
    r = H.gate_cache(cache_dir=world["cache_dir"], preilp=world["preilp"],
                     ecb_dir=world["ecb_dir"], crops=None, receipt=None, reproducer="receipt",
                     manifest=world["manifest"])
    assert not r["passed"]
    assert any("no Gate-1 receipt" in x for x in r["refusals"])


def test_gate_refuses_a_failed_receipt_and_a_zero_band_b_receipt(tmp_path, world):
    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps({"all_passed": False, "crops": []}), encoding="utf-8")
    r = H.gate_cache(cache_dir=world["cache_dir"], preilp=world["preilp"],
                     ecb_dir=world["ecb_dir"], crops=None, receipt=failed, reproducer="receipt",
                     manifest=world["manifest"])
    assert not r["passed"]

    hollow = tmp_path / "hollow.json"
    hollow.write_text(json.dumps({
        "all_passed": True,
        "crops": [{"crop": c, "passed": True, "node_count_mismatches": [],
                   "band_b": {"checked": 0}} for c in CROPS]}), encoding="utf-8")
    r2 = H.gate_cache(cache_dir=world["cache_dir"], preilp=world["preilp"],
                      ecb_dir=world["ecb_dir"], crops=None, receipt=hollow, reproducer="receipt",
                      manifest=world["manifest"])
    assert not r2["passed"]
    assert any("band-B" in reason for c in r2["crops"] for reason in c["reasons"])


# ======================================================================================
# 3. THE TRUNK BINDING - FACT-0392, and why size is not an identity
# ======================================================================================

def test_the_gate_refuses_a_cache_with_no_manifest(world):
    r = H.gate_cache(cache_dir=world["cache_dir"], preilp=world["preilp"],
                     ecb_dir=world["ecb_dir"], crops=None, receipt=world["receipt"],
                     reproducer="receipt", manifest=None)
    assert not r["passed"]
    assert any(H.TRUNK_NOT_BOUND in x and "FACT-0392" in x for x in r["refusals"])


def test_the_licence_pins_the_trunk_by_hash_and_a_same_size_impostor_is_refused(tmp_path, world):
    """The two trunks share a byte size EXACTLY, so only the hash can tell them apart."""
    lic = tmp_path / "licence.json"
    H.write_licence(_gate(world), lic)
    ok = H.verify_licence(lic, world["cache_dir"], CROPS)
    assert ok["verified"] == len(CROPS) and ok["trunk_verified_by"] == "sha256"
    assert ok["trunk_sha256"] == AFC.sha256_file(world["trunk"])

    assert world["impostor"].stat().st_size == world["trunk"].stat().st_size
    with pytest.raises(H.HarnessRefusal) as exc:
        H.verify_licence(lic, world["cache_dir"], CROPS, trunk=world["impostor"])
    assert H.TRUNK_SHA_CHANGED in str(exc.value)
    assert "byte size MATCHES" in str(exc.value) and "FACT-0392" in str(exc.value)


def test_licence_pins_cache_bytes_and_a_changed_cache_is_refused(tmp_path, world):
    lic = tmp_path / "licence.json"
    H.write_licence(_gate(world), lic)
    moved = tmp_path / "cache2"
    moved.mkdir()
    for p in sorted(world["cache_dir"].glob("*.npz")):
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        d["role_feat"] = d["role_feat"] + 0.5
        np.savez_compressed(moved / p.name, **d)
    with pytest.raises(H.HarnessRefusal, match="cache bytes changed"):
        H.verify_licence(lic, moved, CROPS)


def test_write_licence_refuses_a_failed_gate_and_a_gate_with_no_trunk():
    with pytest.raises(H.HarnessRefusal):
        H.write_licence({"passed": False}, Path("nowhere.json"))
    with pytest.raises(H.HarnessRefusal, match=H.TRUNK_NOT_BOUND):
        H.write_licence({"passed": True, "trunk": None, "cache_dir": "x", "reproducer": "receipt",
                         "cache_sha256": {}}, Path("nowhere.json"))


def test_a_schema_1_licence_cannot_license_a_pair_and_role_cache(tmp_path, world):
    lic = tmp_path / "old.json"
    lic.write_text(json.dumps({"schema_version": 1, "cache_sha256": {}}), encoding="utf-8")
    with pytest.raises(H.HarnessRefusal, match="FRAME-KEYED"):
        H.verify_licence(lic, world["cache_dir"], CROPS)


# ======================================================================================
# 4. SCHEMA VALIDATION IS NOT A LICENCE - and the archived P36 cache
# ======================================================================================

def test_a_schema_report_cannot_be_laundered_into_a_training_licence(tmp_path, world):
    r = H.schema_report(world["cache_dir"])
    assert r["schema_ok"] is True
    assert r["licenses_training"] is False
    assert "passed" not in r, "a schema report must not carry the key write_licence reads"
    with pytest.raises(H.HarnessRefusal, match="did not pass"):
        H.write_licence(r, tmp_path / "no.json")


def test_a_schema_check_that_reads_nothing_refuses(tmp_path):
    """FACT-0387: a gate that compares nothing looks exactly like a gate that passed."""
    empty = tmp_path / "empty"
    empty.mkdir()
    r = H.schema_report(empty)
    assert not r["schema_ok"] and r["refusals"]


@needs_p36
def test_the_archived_p36_cache_is_refused_as_contract_1_and_says_which_contract(tmp_path):
    """THE ARCHIVED P36 CACHE'S FATE ON ITS OWN BYTES - the replacement for three retired tests.

    THREE TESTS WERE RETIRED HERE, NOT ACCOMMODATED. They read this artifact as a CONTRACT-2
    schema fixture: that it validates under the pair-and-role schema, that its role index refuses
    an untapped pair, and that a frame-keyed down-conversion of it is rejected with the original
    bytes as accept control. Contract 2 makes every one of those claims unmakeable on this file,
    and correctly so. Contract 1 stored ONE probability column per band, taken from the deployed
    `probs` AFTER every fusion stage, under a name that claimed no surface; the pre-fusion surface
    it never held cannot be recovered by rewriting these bytes. `FACT-0407` recovered it by
    re-running the primary head against the real pack weights, NOT by reading the cache. A legacy
    read-only mode would therefore have to either fabricate `primary_prob_preblend` from
    `deployed_prob_postblend` - which is the P36 defect restated as a feature - or leave it absent
    and silently skip every contract-2 check, which is the quietest way to test nothing.

    RE-POINTING WAS NOT AVAILABLE. A contract-2 capture can only come from re-running the tap, and
    no such artifact exists on disk; this agent is CPU-only and launched no GPU session. When one
    exists, the retired coverage returns against it - the schema, role-index and down-conversion
    claims are all still worth making, just not on a contract-1 file.

    THE ARTIFACT KEEPS ITS EVIDENTIARY VALUE AND LOSES ONLY ITS TRAINING-LICENCE VALUE. It is
    still the file `FACT-0403`, `FACT-0406` and `FACT-0407` were measured on. What this test locks
    is the one claim that is still true of it and still worth a lock: it is refused BY NAME, as
    `cache_contract_1`, and NOT as a generic missing-key message - thirty keys are absent, and
    "missing thirty keys" sends the operator looking for a truncated write.

    `tests/test_audit_feature_cache.py::test_a_contract_1_cache_is_rejected_and_named_as_contract_1`
    proves the same rejection by DOWN-CONVERTING a synthetic contract-2 cache. This one proves it
    on the real archived bytes, through the harness's own refusal wrapper, which is a different
    claim: that the condition NAME survives `HarnessRefusal` verbatim.
    """
    for crop in ("44b6_0113de3b", "44b6_0b24845f"):
        with pytest.raises(H.HarnessRefusal) as exc:
            H.load_cache(P36_CACHE / f"{crop}.npz")
        text = str(exc.value)
        assert "cache_contract_1" in text, f"refused for the wrong reason: {text}"
        assert "cache_schema_incomplete" not in text
        assert "FACT-0403" in text and "FACT-0407" in text

    r = H.schema_report(P36_CACHE)
    assert not r["schema_ok"] and r["licenses_training"] is False
    assert {c["crop"] for c in r["crops"]} == {"44b6_0113de3b", "44b6_0b24845f"}
    assert all(not c["ok"] for c in r["crops"])
    assert all(any("cache_contract_1" in reason for reason in c["reasons"]) for c in r["crops"])

    # ACCEPT CONTROL, because a harness that refuses everything checks nothing: the SAME two
    # entry points accept a contract-2 cache in the production layout.
    control = W.build_world(tmp_path / "control", crops=[CROPS[0]])
    assert H.load_cache(control["cache_dir"] / f"{CROPS[0]}.npz")["role_feat"].shape[0] > 0
    assert H.schema_report(control["cache_dir"])["schema_ok"]


@pytest.mark.skipif(not P36_RECEIPT.is_file(), reason="the P36 gate receipt is not on disk")
def test_the_p36_receipt_cannot_license_training(tmp_path):
    """FACT-0403: the parity verdict is INVALID, and the receipt records all_passed false."""
    data = json.loads(P36_RECEIPT.read_text(encoding="utf-8"))
    assert data["all_passed"] is False
    assert all(c["passed"] is False for c in data["crops"])
    assert all(c["integrity_failure_count"] == 0 for c in data["crops"]), \
        "structure is sound; it is the COMPARISON TARGET that is wrong"
    assert all(c["pos_feature_max_abs_delta"] == 0.0 for c in data["crops"])


# ======================================================================================
# 5. GROUPING - constraint 2, falsifier (b)
# ======================================================================================

def test_no_crop_and_no_target_straddles_any_split(surface):
    from sklearn.model_selection import GroupKFold

    dec = surface.filter(pl.col("true_parent_is_candidate") == 1)
    crops = dec["crop"].to_numpy()
    targets = dec["target"].to_numpy().astype(np.int64)
    x = dec.select(["prob"]).to_numpy()
    y = dec["is_true_parent"].to_numpy()
    seen = 0
    for i, (tr, te) in enumerate(GroupKFold(n_splits=4).split(x, y, groups=crops)):
        H.assert_group_integrity(crops, targets, tr, te, f"t/fold{i}")
        seen += 1
    assert seen == 4


def test_group_integrity_catches_a_target_that_straddles():
    """Both halves of the guard must be reported, not whichever was checked first."""
    crops = np.array(["a", "a", "b", "b"])
    targets = np.array([1, 1, 2, 2])
    with pytest.raises(H.HarnessRefusal) as exc:
        H.assert_group_integrity(crops, targets, np.array([0, 2]), np.array([1, 3]), "t")
    assert "straddle" in str(exc.value) and "crop on both sides" in str(exc.value)


def test_group_integrity_refuses_a_degenerate_split():
    with pytest.raises(H.HarnessRefusal, match="degenerate split"):
        H.assert_group_integrity(np.array(["a"]), np.array([1]),
                                 np.array([0]), np.array([], dtype=int), "t")


def test_folds_report_that_they_are_not_embryo_held_out(surface):
    g = H.grouping_block(surface["crop"].to_numpy(), "GroupKFold", 2)
    assert g["embryos"] == ["44b6"]
    assert g["embryo_held_out"] is False
    assert "WITHIN-embryo" in g["note"]


# ======================================================================================
# 6. ABSTENTION AND SURFACE EQUIVALENCE - constraint 3
# ======================================================================================

def test_with_abstention_off_the_harness_is_the_frozen_surface(surface):
    eq = H.assert_surface_equivalence(surface, "prob")
    assert eq["heartbeat"] == "SURFACE_EQUIVALENT"
    frozen = evaluate(surface, "prob")
    mine, ledger, agg = H.decide(surface, "prob", None, preserve_single=False)
    assert mine == frozen["per_target_correct"]
    assert agg["contested"]["top1"] == frozen["contested"]["top1"]
    assert ledger.summary()["n_regressions"] == 0


def test_surface_equivalence_refuses_when_the_decision_diverges(surface):
    """A guard that cannot fire is decoration - plant a divergence and require the refusal."""
    original = H.decide

    def wrong(table, score_col, abstain_col, preserve_single):
        per_target, ledger, agg = original(table, score_col, abstain_col, preserve_single)
        first = next(iter(per_target))
        per_target[first] = 1 - per_target[first]
        return per_target, ledger, agg

    H.decide = wrong
    try:
        with pytest.raises(H.HarnessRefusal, match="differs from the frozen surface"):
            H.assert_surface_equivalence(surface, "prob")
    finally:
        H.decide = original


def test_abstention_ties_resolve_to_the_null_like_the_deployed_threshold(surface):
    """The deployed rule keeps pairs strictly above the threshold, so an exact tie abstains."""
    tbl = surface.with_columns(pl.col("prob").max().over(["crop", "target"]).alias("_null"))
    per_target, _ledger, agg = H.decide(tbl, "prob", "_null", preserve_single=False)
    assert sum(per_target.values()) == 0
    assert agg["abstentions"]["contested"] == agg["contested"]["n"]


# ======================================================================================
# 7. SINGLE-CANDIDATE ACCOUNTING - constraint 4, falsifier (c). THE DESIGN RISK.
# ======================================================================================

def _abstain_on_one_single_candidate(surface):
    """A score column plus a null column that abstains on exactly one single-candidate target."""
    counts = (surface.filter(pl.col("true_parent_is_candidate") == 1)
                     .group_by(["crop", "target"]).agg(pl.len().alias("n")))
    single = counts.filter(pl.col("n") == 1).sort(["crop", "target"]).row(0)
    tbl = surface.with_columns([
        pl.col("prob").alias("_score"),
        pl.when((pl.col("crop") == single[0]) & (pl.col("target") == single[1]))
          .then(pl.lit(1.0)).otherwise(pl.lit(H.NEVER)).alias("_null"),
    ])
    return tbl, (single[0], int(single[1]))


def test_a_single_candidate_regression_is_reported_by_identity_not_netted(surface):
    tbl, victim = _abstain_on_one_single_candidate(surface)
    per_target, ledger, _agg = H.decide(tbl, "_score", "_null", preserve_single=False)
    s = ledger.summary()
    assert s["n_regressions"] == 1
    assert (s["regressions"][0]["crop"], s["regressions"][0]["target"]) == victim
    assert s["regressions"][0]["reason"] == "abstained"
    assert len(s["regressions"]) == s["n_regressions"]
    assert s["side_b_of_the_two_sided_bar"].startswith("LIVE")
    assert per_target[victim] == 0


def test_preservation_by_construction_prevents_the_loss_and_still_reports_the_shadow(surface):
    """FACT-0386's blind spot closed: the constraint is priced, not hidden."""
    tbl, victim = _abstain_on_one_single_candidate(surface)
    per_target, ledger, _agg = H.decide(tbl, "_score", "_null", preserve_single=True)
    s = ledger.summary()
    assert s["n_regressions"] == 0
    assert s["top1"] == 1.0
    assert s["n_shadow_regressions"] == 1
    assert (s["shadow_regressions"][0]["crop"], s["shadow_regressions"][0]["target"]) == victim
    assert s["side_b_of_the_two_sided_bar"].startswith("INERT")
    assert per_target[victim] == 1


def test_a_count_without_its_identities_cannot_be_emitted():
    led = H.SingleCandidateLedger(preserved_by_construction=False, n_targets=10)
    led.regressions = [{"crop": "a", "target": 1, "source": 0, "reason": "abstained"}]
    assert led.summary()["n_regressions"] == 1
    # a ledger claiming preservation while holding losses is a contract violation
    bad = H.SingleCandidateLedger(preserved_by_construction=True, n_targets=3)
    bad.regressions = [{"crop": "a", "target": 1, "source": 0, "reason": "abstained"}]
    with pytest.raises(H.HarnessRefusal, match="constraint is not being applied"):
        bad.summary()


def test_verdict_blocks_on_single_candidate_regressions_even_when_contested_improves():
    model = {
        "surface": {"contested": {"top1": 1.0, "n": 10}},
        "single_candidate": {"n_regressions": 2, "n_shadow_regressions": 0,
                             "regressions": [{"crop": "a", "target": 1},
                                             {"crop": "a", "target": 2}],
                             "side_b_of_the_two_sided_bar": "LIVE"},
        "conversions": {"contested": {"crop_paired_bootstrap": {"favourable": True}}},
    }
    v = H.harness_verdict(model, 0.8338, {"verdict": {"promotable": True}}, degenerate=False)
    assert not v["promotable"]
    assert any("single-candidate target(s) regressed" in b for b in v["blockers"])


# ======================================================================================
# 8. THE THREE CLASSES, ONE REPORT - falsifier (d)
# ======================================================================================

def test_all_three_model_classes_run_and_emit_identical_channels(surface, world):
    models = [
        H.ModelSpec(tag="linear.geom", model_class="linear", features=["prob", "dist_um"]),
        H.ModelSpec(tag="tree.geom", model_class="tree", features=["prob", "dist_um"]),
        H.ModelSpec(tag="ctx.declared", model_class="contextual",
                    contract={"name": "declared_v1", "dim": DIM, "pair_builder": "diff",
                              "extra_features": ["prob", "dist_um"]}),
    ]
    payload = H.run_harness(table=surface, models=models, fold=0, n_splits=2,
                            cache_gate=_gate(world), role_index_by_crop=_role_index(world))
    assert payload["heartbeat"] == "ASSOC_TRAIN_HARNESS_COMPLETE"
    keys = [set(m.keys()) for m in payload["models"]]
    assert all(k == keys[0] for k in keys)
    assert {m["model_class"] for m in payload["models"]} == {"linear", "tree", "contextual"}
    for m in payload["models"]:
        assert m["surface"]["contested"]["n"] > 0
        assert m["single_candidate"]["preserved_by_construction"] is True
        assert set(m["conversions"]) == {"all_decidable", "contested", "single_candidate"}
    assert payload["grouping"]["embryo_held_out"] is False
    assert payload["cache_gate"]["trunk"]["sha256"] == AFC.sha256_file(world["trunk"])


def test_a_geometry_reading_model_converts_contested_targets_by_identity(surface, world):
    """The harness must show WHICH targets moved, not only that a number rose."""
    models = [H.ModelSpec(tag="linear.geom", model_class="linear",
                          features=["prob", "dist_um"])]
    payload = H.run_harness(table=surface, models=models, fold=0, n_splits=2,
                            cache_gate=_gate(world), role_index_by_crop=_role_index(world))
    m = payload["models"][0]
    conv = m["conversions"]["contested"]
    assert conv["gained"] > 0
    assert conv["net"] == conv["gained"] - conv["lost"]
    assert conv["churn"] == conv["gained"] + conv["lost"]
    assert m["surface"]["contested"]["top1"] > payload["deployed_baseline"]["contested"]["top1"]
    assert "ci95" in conv["crop_paired_bootstrap"]


def test_every_class_reports_through_assoc_report_build_report(surface, world):
    arms = {
        tag: {"control": [chain_rows(900, 60, 60, 1000, 1000), chain_rows(800, 90, 90, 990, 1000)],
              "candidate": [chain_rows(930, 50, 40, 1002, 1000),
                            chain_rows(830, 80, 70, 992, 1000)]}
        for tag in ("linear.geom", "ctx.declared")
    }
    models = [
        H.ModelSpec(tag="linear.geom", model_class="linear", features=["prob", "dist_um"]),
        H.ModelSpec(tag="ctx.declared", model_class="contextual",
                    contract={"name": "declared_v1", "dim": DIM,
                              "extra_features": ["prob", "dist_um"]}),
    ]
    payload = H.run_harness(table=surface, models=models, fold=0, n_splits=2,
                            cache_gate=_gate(world), role_index_by_crop=_role_index(world),
                            chain_arms=arms, summarise=fake_summarise)
    for m in payload["models"]:
        fc = m["full_chain"]
        assert fc is not None and fc["heartbeat"] == "ASSOC_REPORT_COMPLETE"
        assert set(fc["channels"]) >= {"edge_jaccard_raw", "count_adjustment", "node_recall",
                                       "division_counts", "final_graph_edges", "score",
                                       "parent_conversions"}
        assert abs(fc["channels"]["score"]["identity_check"]) < 1e-9


def _always_abstains():
    """A null scored at 1.0 dominates every calibrated probability, so abstention is certain."""
    return H.ModelSpec(tag="linear.abstain", model_class="linear",
                       features=["prob", "dist_um"],
                       abstain=H.AbstainPolicy(kind="fixed", tau=1.0))


def test_abstention_end_to_end_enumerates_every_single_candidate_regression(surface, world):
    """Abstention live on the whole surface: each loss must arrive with its identity."""
    payload = H.run_harness(table=surface, models=[_always_abstains()], fold=0, n_splits=2,
                            preserve_single=False,
                            cache_gate=_gate(world), role_index_by_crop=_role_index(world))
    m = payload["models"][0]
    s = m["single_candidate"]
    assert s["n_regressions"] > 0, "the abstention policy did not fire; the test proves nothing"
    assert s["n_regressions"] == s["n"], "every single-candidate target should have been lost"
    assert len(s["regressions"]) == s["n_regressions"]
    assert all(r["reason"] == "abstained" for r in s["regressions"])
    assert not m["verdict"]["promotable"]
    assert any("single-candidate" in b for b in m["verdict"]["blockers"])
    assert m["conversions"]["single_candidate"]["lost"] == s["n_regressions"]


def test_the_same_policy_under_preservation_moves_the_losses_into_the_shadow(surface, world):
    payload = H.run_harness(table=surface, models=[_always_abstains()], fold=0, n_splits=2,
                            preserve_single=True,
                            cache_gate=_gate(world), role_index_by_crop=_role_index(world))
    s = payload["models"][0]["single_candidate"]
    assert s["n_regressions"] == 0 and s["top1"] == 1.0
    assert s["n_shadow_regressions"] == s["n"]
    assert payload["models"][0]["verdict"]["single_candidate_shadow_regressions"] == s["n"]
    assert s["side_b_of_the_two_sided_bar"].startswith("INERT")


def test_train_quantile_tau_is_fitted_per_fold_and_never_on_validation(surface, world):
    models = [H.ModelSpec(tag="linear.q", model_class="linear", features=["prob", "dist_um"],
                          abstain=H.AbstainPolicy(kind="train_quantile", q=0.5))]
    payload = H.run_harness(table=surface, models=models, fold=0, n_splits=2,
                            preserve_single=True,
                            cache_gate=_gate(world), role_index_by_crop=_role_index(world))
    folds = payload["models"][0]["folds"]
    assert len(folds) == 2
    assert all(f["abstain_tau"] is not None and 0.0 <= f["abstain_tau"] <= 1.0 for f in folds)
    a, b = (set(f["val_crops"]) for f in folds)
    assert not (a & b) and (a | b) == set(CROPS)


def test_training_refuses_when_the_gate_did_not_pass(surface):
    models = [H.ModelSpec(tag="linear", model_class="linear", features=["prob", "dist_um"])]
    with pytest.raises(H.HarnessRefusal, match="cache gate did not pass"):
        H.run_harness(table=surface, models=models, fold=0, n_splits=2,
                      cache_gate={"passed": False, "crops": []})


def test_contextual_model_without_a_passing_gate_is_refused(surface):
    models = [H.ModelSpec(tag="ctx", model_class="contextual",
                          contract={"name": "declared", "dim": DIM})]
    with pytest.raises(H.HarnessRefusal, match="measures the cache"):
        H.run_harness(table=surface, models=models, fold=0, n_splits=2, cache_gate=None)


# ======================================================================================
# 9. THE DECLARED CONTRACT, AND THE DEGENERATE FOLD
# ======================================================================================

def test_an_undeclared_contextual_contract_is_refused():
    with pytest.raises(H.HarnessRefusal, match="no declared contract"):
        H.ContextContract.from_dict(None)


def test_a_contract_whose_dimension_does_not_match_the_cache_is_refused(world):
    c = H.ContextContract.from_dict({"name": "wrong", "dim": DIM + 1})
    with pytest.raises(H.HarnessRefusal, match="does not describe this cache"):
        c.validate_against_cache(_role_index(world)[CROPS[0]])


def test_a_from_cache_contract_adopts_the_role_feature_width(world):
    c = H.ContextContract.from_dict({"name": "ours", "dim": "from_cache"})
    c.validate_against_cache(_role_index(world)[CROPS[0]])
    assert c.dim == DIM and c.dim_from_cache is True


def test_a_contract_naming_features_outside_the_frozen_surface_is_refused():
    with pytest.raises(H.HarnessRefusal, match="outside the frozen surface"):
        H.ContextContract.from_dict({"name": "x", "dim": 4, "extra_features": ["invented"]})


def test_a_degenerate_fold_refuses_a_ranking_claim(surface):
    """FACT-0381 / FACT-0382: a metric that cannot fail is not evidence."""
    single_only = surface.filter(pl.col("n_candidates") == 1)
    with pytest.raises(H.HarnessRefusal, match="ZERO contested targets"):
        H.run_harness(table=single_only, models=[], fold=1, n_splits=2)


# ======================================================================================
# 10. THE CLI, END TO END
# ======================================================================================

def test_cli_schema_then_gate_then_train_runs_the_whole_path(tmp_path, world, surface):
    assert H.main(["schema", "--cache-dir", str(world["cache_dir"]),
                   "--out", str(tmp_path / "schema.json")]) == 0
    schema = json.loads((tmp_path / "schema.json").read_text(encoding="utf-8"))
    assert schema["licenses_training"] is False and "passed" not in schema

    table = tmp_path / "surface.parquet"
    surface.write_parquet(table)
    lic = tmp_path / "licence.json"
    rc = H.main(["gate", "--cache-dir", str(world["cache_dir"]), "--preilp", str(world["preilp"]),
                 "--ecb-dir", str(world["ecb_dir"]), "--receipt", str(world["receipt"]),
                 "--manifest", str(world["manifest"]),
                 "--licence", str(lic), "--out", str(tmp_path / "gate.json")])
    assert rc == 0 and lic.is_file()
    assert json.loads(lic.read_text(encoding="utf-8"))["trunk"]["sha256"]

    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "name": "synthetic", "fold": 0, "table": str(table),
        "cache": {"dir": str(world["cache_dir"]), "receipt": str(world["receipt"]),
                  "preilp": str(world["preilp"]), "ecb_dir": str(world["ecb_dir"]),
                  "licence": str(lic), "manifest": str(world["manifest"]),
                  "trunk": str(world["trunk"]), "reproducer": "receipt", "crops": CROPS},
        "cv": {"kind": "GroupKFold", "n_splits": 2},
        "preserve_single_candidate": True,
        "models": [{"tag": "linear.geom", "class": "linear", "features": ["prob", "dist_um"]},
                   {"tag": "ctx", "class": "contextual",
                    "contract": {"name": "declared_v1", "dim": DIM,
                                 "extra_features": ["prob", "dist_um"]}}],
        "out_dir": str(tmp_path / "out"),
        # FACT-0451 rule 2: omission is a REFUSAL at the schema boundary, so a fixture without
        # this block no longer reaches the harness at all. Declared true and re-derived at load
        # from the sha256 of every checkpoint the spec names - this synthetic trunk is not a
        # restricted artifact, and if it ever became one the derivation would override the
        # declaration rather than trusting it.
        "binding_restriction": {"fact": "FACT-0451", "offline_scoreable": True},
    }), encoding="utf-8")
    assert H.main(["train", "--spec", str(spec)]) == 0
    payload = json.loads((tmp_path / "out" / "harness_f0.json").read_text(encoding="utf-8"))
    assert payload["heartbeat"] == "ASSOC_TRAIN_HARNESS_COMPLETE"
    assert payload["cache_gate"]["passed"] is True
    assert payload["cache_gate"]["trunk"]["role"] == "official"
    assert len(payload["models"]) == 2


def test_cli_train_refuses_without_a_licence_and_without_a_manifest(tmp_path, world, surface):
    table = tmp_path / "surface.parquet"
    surface.write_parquet(table)

    def spec_with(cache: dict) -> Path:
        p = tmp_path / f"spec_{len(cache)}.json"
        p.write_text(json.dumps({
            "fold": 0, "table": str(table), "cache": cache,
            "cv": {"kind": "GroupKFold", "n_splits": 2},
            "models": [{"tag": "ctx", "class": "contextual",
                        "contract": {"name": "d", "dim": DIM}}],
            "out_dir": str(tmp_path / "out"),
            # FACT-0451 rule 2 - see the note on the fixture above. These specs are expected to
            # be refused for CACHE reasons, so the restriction block has to be present or they
            # would be refused for the wrong reason and the test would prove nothing.
            "binding_restriction": {"fact": "FACT-0451", "offline_scoreable": True},
        }), encoding="utf-8")
        return p

    base = {"dir": str(world["cache_dir"]), "receipt": str(world["receipt"]),
            "preilp": str(world["preilp"]), "ecb_dir": str(world["ecb_dir"]), "crops": CROPS}
    with pytest.raises(H.HarnessRefusal, match=H.TRUNK_NOT_BOUND):
        H.main(["train", "--spec", str(spec_with(dict(base, licence=str(tmp_path / "m.json"))))])
    with pytest.raises(H.HarnessRefusal, match="no cache licence"):
        H.main(["train", "--spec", str(spec_with(dict(base, licence=str(tmp_path / "m.json"),
                                                      manifest=str(world["manifest"]))))])


def test_prepared_fold_specs_are_valid_and_bind_a_trunk():
    d = ROOT / "scripts" / "win_bet" / "assoc_specs"
    f0 = json.loads((d / "harness_f0.json").read_text(encoding="utf-8"))
    f1 = json.loads((d / "harness_f1.json").read_text(encoding="utf-8"))
    for spec in (f0, f1):
        models = [H.ModelSpec.from_dict(m) for m in spec["models"]]
        assert models and spec["cv"]["kind"] in {"GroupKFold", "LeaveOneCropOut"}
        assert spec["preserve_single_candidate"] is True
        assert "licence" in spec["cache"]
        # FACT-0392: an unbound trunk makes a null uninterpretable, so the spec must name one.
        assert spec["cache"].get("manifest"), "the spec binds no trunk manifest"
        for m in models:
            if m.model_class == "contextual":
                H.ContextContract.from_dict(m.contract)
    assert f0["fold"] == 0 and f1["fold"] == 1
    # FACT-0381 / FACT-0382: fold 1 is degenerate for ranking today and must make no claim.
    assert f1["no_claim"] is True
    assert "FACT-0381" in f1["note"] and "FACT-0382" in f1["note"]


def test_module_docstring_carries_the_smoke_commands_and_the_schema_caveat():
    doc = H.__doc__
    assert "assoc_train_harness.py gate" in doc
    assert "--receipt" in doc and "--licence" in doc and "--manifest" in doc
    assert "assoc_train_harness.py schema" in doc
    assert "TRAINING_LICENCE=NOT_GRANTED" in doc and "FACT-0403" in doc
    assert textwrap.dedent(doc).count("ASSOC_CACHE_GATE_PASSED") >= 1
