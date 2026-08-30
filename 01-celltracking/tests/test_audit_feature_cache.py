"""The cache auditor must be shown to REJECT, not asserted to work (PKT-0037).

A checker that has never been demonstrated rejecting anything is indistinguishable from a
checker that returns True. This project has paid for that lesson three times: a test that grepped
source instead of running it, a Gate-1 harness that compared zero crops and would have licensed
two GPU sessions had it defaulted to a pass (`FACT-0387`), and - the one that matters most here -
a suite that stubbed ``model.detection_head``, A METHOD THE REAL CLASS HAS NEVER HAD, and so
validated a fiction while the real defect survived to burn a GPU session (`FACT-0400`, ledger
item 4).

So this file does two things and skips neither:

1. IT RUNS THE REAL PRODUCER. ``real_cache`` executes the REAL ``scripts/kaggle_edits/
   assoc_feature_tap.py`` patch against the REAL vendored ``predict_video``, over a real zarr,
   in a real subprocess with the production layout, by reusing the sandbox from
   ``tests/test_assoc_feature_tap.py``. Nothing about the cache under audit is imagined: it is
   what the deployed predictor actually emitted, including the 66-of-173 dual-role split that
   `FACT-0402` measured.

2. IT MUTATES THAT REAL CACHE. Every rejection below is proved by planting one defect in a
   cache that was accepted a moment earlier, and the assertion is on the CONDITION NAME, because
   "it rejected" and "it rejected for the right reason" are different claims and only the second
   one helps whoever has to fix the cache. Accept controls run first and are asserted, because
   an auditor that rejects everything checks nothing.

THE THREE DEFECT-6 FIXTURES are the reason this file was rewritten. `FACT-0402`: a trunk node
feature is WINDOW-DEPENDENT, so the retired frame-keyed layout is not merely wrong, it is
undefined. The auditor must refuse (a) the retired layout itself, (b) a pair-and-role file whose
role multiplicity was collapsed the way the old worker collapsed it, and (c) a pair-and-role file
whose SHAPE is correct but whose dual-role vectors are bit-identical - defect 6 reintroduced by
someone who read the schema and not the mechanism.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_feature_cache as A  # noqa: E402
import test_assoc_feature_tap as TAP  # noqa: E402

TAP_PATCH = ROOT / "scripts" / "kaggle_edits" / "assoc_feature_tap.py"
REPLAY = ROOT / "scripts" / "win_bet" / "assoc_tap_replay.py"


# =============================================================================================
# 1. THE ANTI-DRIFT LOCK
#
# The auditor mirrors the tap's key list rather than importing it, because the tap is a PATCH
# SCRIPT whose module body rewrites a file. A mirror drifts unless something fails when it does -
# the same reasoning that produced scripts/win_bet/sync_tap_worker.py for the worker literal.
# =============================================================================================
def _tap_schema_tuples() -> dict[str, tuple[str, ...]]:
    """Parse _AFT_SCHEMA_REQUIRED / _AFT_SCHEMA_OPTIONAL out of the tap's inserted source."""
    text = TAP_PATCH.read_text(encoding="utf-8")
    # The declarations live inside the _AFT_CONFIG string literal that the patch inserts.
    tree = ast.parse(text)
    config = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and node.targets
                and getattr(node.targets[0], "id", None) == "_AFT_CONFIG"
                and isinstance(node.value, ast.Constant)):
            config = node.value.value
    assert config, "could not find _AFT_CONFIG in the tap patch"
    out = {}
    for node in ast.walk(ast.parse(config)):
        if isinstance(node, ast.Assign) and node.targets:
            name = getattr(node.targets[0], "id", None)
            if name in ("_AFT_SCHEMA_REQUIRED", "_AFT_SCHEMA_OPTIONAL"):
                out[name] = tuple(ast.literal_eval(node.value))
    return out


def test_the_auditor_required_keys_are_exactly_the_tap_schema():
    """If the tap renames a key, this fails on CPU instead of after the GPU spend."""
    schema = _tap_schema_tuples()
    assert set(A.REQUIRED_KEYS) == set(schema["_AFT_SCHEMA_REQUIRED"]), (
        "audit_feature_cache.REQUIRED_KEYS has drifted from assoc_feature_tap's "
        "_AFT_SCHEMA_REQUIRED"
    )
    assert A.POSITIONAL_KEY in schema["_AFT_SCHEMA_OPTIONAL"]


def test_every_key_the_replay_worker_reads_is_a_key_the_auditor_requires():
    """The replay worker reads by name. A key it needs that the auditor does not police is a
    KeyError waiting at the end of a GPU session."""
    tree = ast.parse(REPLAY.read_text(encoding="utf-8"))
    read: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == "cache" and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            read.add(node.slice.value)
        # cache[f"band_{name}_pair"] and friends
        if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == "cache" and isinstance(node.slice, ast.JoinedStr)):
            for band in ("a", "b"):
                parts = []
                for v in node.slice.values:
                    parts.append(v.value if isinstance(v, ast.Constant) else band)
                read.add("".join(parts))
    assert read, "parsed no cache reads out of the replay worker - the parser has rotted"
    covered = set(A.REQUIRED_KEYS) | {A.POSITIONAL_KEY}
    assert read <= covered, f"replay reads keys the auditor does not require: {sorted(read - covered)}"


def test_the_auditor_names_the_deployed_window_and_it_is_the_config_value():
    """window_size 2 is read from the deployed config, not assumed; the whole dual-role
    phenomenon exists because of it."""
    assert A.DEPLOYED_WINDOW == 2


# =============================================================================================
# 2. A REAL CACHE, FROM THE REAL PRODUCER, IN PRODUCTION LAYOUT
# =============================================================================================
@pytest.fixture(scope="module")
def real_cache(tmp_path_factory):
    """Run the REAL tap over the REAL predict_video and return the crop cache it wrote."""
    base = tmp_path_factory.mktemp("aft_audit")
    box = TAP.Sandbox(base)
    box.working.mkdir(parents=True)
    import shutil
    shutil.copytree(TAP.VENDOR / "scripts", box.repo / "scripts",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(TAP.VENDOR / "src", box.repo / "src",
                    ignore=shutil.ignore_patterns("__pycache__"))
    TAP._build_zarr(box.repo, TAP.CROP)
    box.weights = TAP._build_weights(box.repo)
    box.pristine = box.predictor.read_text(encoding="utf-8")
    box.driver.write_text(TAP.PREDICT_DRIVER, encoding="utf-8")
    box.apply_tap()
    cache_dir = base / "cache"
    box.predict("tapped.json", {"BIOHUB_AFT_DIR": str(cache_dir),
                                "BIOHUB_AFT_BAND_B_FLOOR": str(TAP.BAND_B_FLOOR)})
    path = cache_dir / f"{TAP.CROP}.npz"
    assert path.is_file(), "the real tap wrote no cache; the fixture is not exercising it"
    return path


@pytest.fixture()
def bench(real_cache, tmp_path):
    """A fold-0 cache directory holding the REAL tap output under a fold-0 crop name."""
    import shutil
    nb = tmp_path / "notebook.ipynb"
    nb.write_text('{"cells": []}', encoding="utf-8")
    trunk = tmp_path / "trunk_official.pth"
    trunk.write_bytes(b"OFFICIAL-TRUNK-BYTES" * 64)
    cache = tmp_path / "cache"
    cache.mkdir()
    crop = "44b6_aaaaaaaa"
    shutil.copyfile(real_cache, cache / f"{crop}.npz")
    args = A._bind_args(cache, trunk, fold="0", role="official",
                        weights_glob="/kaggle/input/*/split_0/edge_predictor_best.pth",
                        notebook=nb)
    manifest = cache / "cache_manifest.json"
    manifest.write_text(json.dumps(A.build_manifest(args), indent=2), encoding="utf-8")
    return {"root": tmp_path, "cache": cache, "manifest": manifest, "trunk": trunk,
            "notebook": nb, "crop": crop, "path": cache / f"{crop}.npz"}


def _mutate(bench, fn):
    """Rewrite the REAL cache in place with one planted defect. Everything else is untouched."""
    path = bench["path"]
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    fn(data)
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **data)
    tmp.replace(path)
    return path


# --- ACCEPT CONTROLS -------------------------------------------------------------------------
def test_the_real_tap_output_is_accepted(bench):
    """Half the proof. If this ever fails, every rejection below is meaningless."""
    report = A.audit(bench["cache"], bench["manifest"])
    assert report["passed"] is True
    assert report["crops"] == 1
    names = {c["check"] for c in report["checks"]}
    assert "features_still_bound_to_the_recorded_trunk" in names
    assert f"{bench['crop']}:features_are_window_dependent" in names


def test_the_real_cache_exhibits_the_window_dependence_fact_0402_measured(bench):
    """The auditor re-derives FACT-0402 from the artifact rather than citing it.

    Two roles, two forward passes, two vectors - and the delta is orders of magnitude above the
    1e-4 gate tolerance, which is what made the frame-keyed layout undefined rather than lossy.
    """
    rec = A.bind_crop_file(bench["path"], 0.5)
    wd = rec["window_dependence"]
    assert wd["dual_role_nodes"] > 0
    assert wd["dual_role_nodes"] < wd["role_nodes"], (
        "every role node is dual - the fixture no longer has boundary frames")
    assert wd["max_abs_delta"] > 1e-4, (
        "the real trunk's two role vectors agree inside the gate tolerance; the temporal "
        "attention this whole schema exists for is not firing in the fixture")


def test_a_pair_capped_cache_is_accepted(bench):
    """P36 ships BIOHUB_AFT_MAX_PAIRS=8, so a real cache records a PREFIX of the frame pairs
    while `frames`/`starts`/`ends` still cover every detected frame. That must be an ACCEPT: a
    truncation check that fired here would refuse the very artifact the smoke produces."""
    keep = 2

    def truncate(data):
        cut = int(data["pair_src_ptr"][keep])
        for key in ("pair_f_idx", "pair_t_src", "pair_t_tgt", "pair_src_ptr", "pair_src_n",
                    "pair_tgt_ptr", "pair_tgt_n", "pair_window_shape"):
            data[key] = data[key][:keep]
        for key in ("role_pair", "role_role", "role_gid", "role_feat", "role_coord_scaled",
                    "role_coord_rel", "role_mask", "role_pos"):
            data[key] = data[key][:cut]
        for band in ("a", "b"):
            sel = data[f"band_{band}_pair"] < keep
            for col in ("pair", "source_id", "target_id", "i", "j", "prob"):
                data[f"band_{band}_{col}"] = data[f"band_{band}_{col}"][sel]
        data["source_id"] = np.concatenate([data["band_a_source_id"], data["band_b_source_id"]])
        data["target_id"] = np.concatenate([data["band_a_target_id"], data["band_b_target_id"]])
        data["edge_prob"] = np.concatenate(
            [data["band_a_prob"], data["band_b_prob"]]).astype(np.float32)

    _mutate(bench, truncate)
    rec = A.bind_crop_file(bench["path"], 0.5)
    assert rec["pairs"] == keep
    assert rec["window_dependence"]["dual_role_nodes"] > 0


def test_the_manifest_binds_every_field_the_packet_requires(bench):
    m = json.loads(bench["manifest"].read_text(encoding="utf-8"))
    assert m["schema_version"] == A.SCHEMA_VERSION and m["layout"] == "pair_and_role"
    assert m["trunk"]["sha256"] and m["trunk"]["provenance"] and m["trunk"]["role"]
    assert m["trunk"]["provenance_source"] == "manifest"
    assert m["trunk"]["checkpoint_carries_no_provenance"] is True
    assert m["fold"]["fold"] and m["fold"]["held_out_embryo"] and m["fold"]["crops"]
    assert m["window_contract"]["window"] == A.DEPLOYED_WINDOW
    assert m["window_contract"]["feature_key"] == "(pair, role)"
    assert m["feature_normalisation"]["transform"]
    crop = m["crops"][0]
    assert crop["node_order"]["coords_digest"] and crop["node_order"]["role_digest"]
    assert crop["node_order"]["features_digest"] and crop["node_order"]["positional_digest"]
    assert crop["storage"]["bytes_on_disk"] > 0 and crop["storage"]["uncompressed_bytes"] > 0
    assert m["candidate_graph"]["rule"] and m["provenance"]["source_commit"]


# =============================================================================================
# 3. DEFECT 6 - THE THREE FIXTURES THIS PACKET EXISTS FOR
# =============================================================================================
def test_a_frame_keyed_cache_is_rejected_and_names_defect_6(tmp_path):
    """The RETIRED layout: one feature vector per node, keyed by frame. FACT-0402 measured that
    a trunk node feature is window-dependent, so this file does not describe a defined object."""
    cache = tmp_path / "c"
    cache.mkdir()
    A._synth_cache(cache / "44b6_x.npz", crop="44b6_x", trunk_seed=1, frame_keyed=True)
    with pytest.raises(A.Reject) as err:
        A.load_cache(cache / "44b6_x.npz")
    text = str(err.value)
    assert "frame_keyed_cache" in text, f"rejected for the wrong reason: {text}"
    assert "feat_" in text and "WINDOW-DEPENDENT" in text
    assert "FACT-0402" in text


def test_a_frame_keyed_cache_is_not_rejected_merely_as_a_missing_key(tmp_path):
    """The condition must be defect 6, not 'cache_schema_incomplete'. A schema-1 file is missing
    most pair-and-role keys, so the cheap message would send the operator to look for a truncated
    write when the artifact's whole addressing scheme is undefined."""
    cache = tmp_path / "c"
    cache.mkdir()
    A._synth_cache(cache / "44b6_x.npz", crop="44b6_x", trunk_seed=1, frame_keyed=True)
    with pytest.raises(A.Reject) as err:
        A.load_cache(cache / "44b6_x.npz")
    assert "cache_schema_incomplete" not in str(err.value)


def test_a_real_cache_stripped_back_to_the_frame_keyed_layout_is_rejected(bench):
    """The same defect reached from the REAL artifact rather than from a synthetic one: keep the
    node partition, drop the role table, and re-key the features by frame the way the old worker
    did (assoc_feature_parity.py:138-147, 'cached feats[t] once at first sight')."""
    with np.load(bench["path"], allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    frames = data["frames"].tolist()
    starts, ends = data["starts"].tolist(), data["ends"].tolist()
    gid = data["role_gid"]
    legacy = {"coords": data["coords"], "frames": data["frames"],
              "starts": data["starts"], "ends": data["ends"],
              "feat_frames": data["frames"],
              "source_id": data["source_id"], "target_id": data["target_id"],
              "edge_prob": data["edge_prob"]}
    for k, (s, e) in zip(frames, zip(starts, ends)):
        rows = [int(np.argmax(gid == g)) for g in range(s, e)]
        legacy[f"feat_{k}"] = data["role_feat"][rows]      # first sight wins - the old defect
    out = bench["root"] / "legacy"
    out.mkdir()
    np.savez_compressed(out / f"{bench['crop']}.npz", **legacy)
    with pytest.raises(A.Reject, match="frame_keyed_cache"):
        A.load_cache(out / f"{bench['crop']}.npz")


def test_a_collapsed_role_multiplicity_is_rejected_as_a_missing_role_record(bench):
    """DEFECT 6, PART TWO. The pair-and-role SHAPE with the old worker's de-duplication inside:
    one block per FRAME, both roles pointed at it, so an interior node carries ONE vector where
    the window makes two."""
    def collapse(data):
        frames = data["frames"].tolist()
        starts, ends = data["starts"].tolist(), data["ends"].tolist()
        gid = data["role_gid"]
        rows = []
        for s, e in zip(starts, ends):
            rows.extend(int(np.argmax(gid == g)) for g in range(s, e))
        rows = np.asarray(rows, dtype=np.int64)
        for key in ("role_pair", "role_role", "role_gid", "role_feat", "role_coord_scaled",
                    "role_coord_rel", "role_mask", "role_pos"):
            data[key] = data[key][rows]
        index = {t: int(starts[frames.index(t)]) for t in frames}
        n = {t: int(ends[frames.index(t)]) - int(starts[frames.index(t)]) for t in frames}
        data["pair_src_ptr"] = np.asarray([index[int(t)] for t in data["pair_t_src"]],
                                          dtype=np.int64)
        data["pair_src_n"] = np.asarray([n[int(t)] for t in data["pair_t_src"]], dtype=np.int64)
        data["pair_tgt_ptr"] = np.asarray([index[int(t)] for t in data["pair_t_tgt"]],
                                          dtype=np.int64)
        data["pair_tgt_n"] = np.asarray([n[int(t)] for t in data["pair_t_tgt"]], dtype=np.int64)

    _mutate(bench, collapse)
    with pytest.raises(A.Reject) as err:
        A.bind_crop_file(bench["path"], 0.5)
    text = str(err.value)
    assert "role_records_missing" in text, f"rejected for the wrong reason: {text}"
    assert "FACT-0402" in text


def test_a_duplicated_role_block_is_rejected(bench):
    """The other direction: a node carrying MORE role records than the pair table allows, which
    is what a merged or double-flushed file looks like and what would double-count in training."""
    def duplicate(data):
        n = int(data["pair_src_n"][0])
        for key in ("role_pair", "role_role", "role_gid", "role_feat", "role_coord_scaled",
                    "role_coord_rel", "role_mask", "role_pos"):
            data[key] = np.concatenate([data[key], data[key][:n]])

    _mutate(bench, duplicate)
    with pytest.raises(A.Reject, match="duplicate_role_record"):
        A.bind_crop_file(bench["path"], 0.5)


def test_pair_and_role_shape_with_frame_keyed_content_is_rejected(bench):
    """DEFECT 6, PART THREE, AND THE ONE A SHAPE CHECK CANNOT SEE. Every array keeps its shape,
    its multiplicity and its ordering; the only change is that each dual-role node's target-role
    vector is overwritten with its source-role vector - exactly what a frame-keyed writer that
    had been 'fixed' to emit two rows would produce."""
    def copy_first_vector(data):
        gid = data["role_gid"]
        feat = data["role_feat"].copy()
        seen: dict[int, int] = {}
        for row, g in enumerate(gid.tolist()):
            if g in seen:
                feat[row] = feat[seen[g]]
            else:
                seen[g] = row
        data["role_feat"] = feat

    before = A.bind_crop_file(bench["path"], 0.5)
    assert before["window_dependence"]["dual_role_nodes"] > 0
    _mutate(bench, copy_first_vector)
    with pytest.raises(A.Reject) as err:
        A.bind_crop_file(bench["path"], 0.5)
    text = str(err.value)
    assert "role_features_frame_keyed" in text, f"rejected for the wrong reason: {text}"
    assert "_TemporalAttention" in text


def test_the_frame_keyed_content_check_does_not_fire_on_the_clean_cache(bench):
    """The accept side of the sharpest check. If it fired here it would be a tripwire, not a
    test, and it would block every legitimate cache."""
    rec = A.bind_crop_file(bench["path"], 0.5)
    assert rec["window_dependence"]["max_abs_delta"] > 0.0


# =============================================================================================
# 4. STRUCTURE, ORDERING AND GEOMETRY - each proved on the REAL cache
# =============================================================================================
def test_reordered_role_rows_are_rejected(bench):
    """Node ids are positional indices into coords_so_far. Reverse one block's ids and every
    downstream artifact silently renames its cells."""
    def flip(data):
        n = int(data["pair_src_n"][0])
        data["role_gid"] = data["role_gid"].copy()
        data["role_gid"][:n] = data["role_gid"][:n][::-1]

    _mutate(bench, flip)
    with pytest.raises(A.Reject,
                       match="role_node_order_is_not_the_deployed_frame_slice"):
        A.bind_crop_file(bench["path"], 0.5)


def test_a_torn_role_block_pointer_is_rejected(bench):
    def shift(data):
        data["pair_tgt_ptr"] = data["pair_tgt_ptr"].copy()
        data["pair_tgt_ptr"][0] += 1

    _mutate(bench, shift)
    with pytest.raises(A.Reject) as err:
        A.bind_crop_file(bench["path"], 0.5)
    assert "role_blocks_do_not_tile" in str(err.value) or \
           "role_node_order_is_not_the_deployed_frame_slice" in str(err.value)


def test_a_padded_mask_slot_is_rejected(bench):
    """The deployed site builds torch.ones and never pads, so a False slot is not a deployed
    input and the head would be handed a masked node it was never given."""
    def pad(data):
        data["role_mask"] = data["role_mask"].copy()
        data["role_mask"][0] = False

    _mutate(bench, pad)
    with pytest.raises(A.Reject, match="role_mask_has_a_false_slot"):
        A.bind_crop_file(bench["path"], 0.5)


def test_a_role_filed_under_the_wrong_relative_time_is_rejected(bench):
    """c_src_rel[:, 0] = f_idx and c_tgt_rel[:, 0] = f_idx + 1. The relative time column is what
    makes the ROLE part of the record, so a block whose column is wrong is filed as the other
    role and the positional features it produced belong to a different node."""
    def retime(data):
        ptr, n = int(data["pair_tgt_ptr"][0]), int(data["pair_tgt_n"][0])
        rel = data["role_coord_rel"].copy()
        rel[ptr:ptr + n, 0] = 0
        data["role_coord_rel"] = rel

    _mutate(bench, retime)
    with pytest.raises(A.Reject) as err:
        A.bind_crop_file(bench["path"], 0.5)
    # the scaled/rel cross-check is unaffected by the time column, so this must be the role check
    assert "role_relative_time_wrong" in str(err.value)


def test_torn_scaled_coordinates_are_rejected(bench):
    """role_coord_scaled is role_coord_rel[:, 1:] * downsample by construction. Both go to the
    head; if they disagree one of the two records is torn."""
    def tear(data):
        data["role_coord_scaled"] = data["role_coord_scaled"] + np.float32(1.0)

    _mutate(bench, tear)
    with pytest.raises(A.Reject,
                       match="role_coord_scaled_disagrees_with_role_coord_rel"):
        A.bind_crop_file(bench["path"], 0.5)


def test_a_missing_positional_feature_array_is_rejected(bench):
    """role_pos is an INPUT to predict_edges. A cache without it forces the gate to trust a
    re-derivation instead of an artifact."""
    def drop(data):
        data.pop("role_pos")

    _mutate(bench, drop)
    with pytest.raises(A.Reject, match="positional_features_absent"):
        A.load_cache(bench["path"])


def test_a_non_contiguous_frame_partition_is_rejected(bench):
    def gap(data):
        data["ends"] = data["ends"].copy()
        data["ends"][0] -= 1

    _mutate(bench, gap)
    with pytest.raises(A.Reject, match="frame_partition_not_contiguous"):
        A.bind_crop_file(bench["path"], 0.5)


def test_a_window_of_one_is_rejected(bench):
    """A window below 2 contains no consecutive pair, so no association was ever computed and
    the file cannot be what it claims to be."""
    def shrink(data):
        data["window"] = np.int64(1)

    _mutate(bench, shrink)
    with pytest.raises(A.Reject, match="window_too_small"):
        A.bind_crop_file(bench["path"], 0.5)


def test_a_cache_whose_window_disagrees_with_the_manifest_is_rejected(bench):
    def bump(data):
        data["window"] = np.int64(3)
        data["pair_window_shape"] = data["pair_window_shape"].copy()
        data["pair_window_shape"][:, 0] = 3

    _mutate(bench, bump)
    with pytest.raises(A.Reject, match="window_disagrees_with_the_manifest"):
        A.audit(bench["cache"], bench["manifest"])


# =============================================================================================
# 5. BOTH PROBABILITY BANDS
# =============================================================================================
def test_an_empty_deployed_band_is_rejected(bench):
    """A crop whose band A compared nothing satisfies every condition on that band trivially -
    FACT-0394 is the cost of a band that compared nothing and reported a pass."""
    def empty_a(data):
        for col in ("pair", "source_id", "target_id", "i", "j"):
            data[f"band_a_{col}"] = np.empty(0, dtype=np.int64)
        data["band_a_prob"] = np.empty(0, dtype=np.float64)
        data["source_id"] = data["band_b_source_id"]
        data["target_id"] = data["band_b_target_id"]
        data["edge_prob"] = data["band_b_prob"].astype(np.float32)

    _mutate(bench, empty_a)
    with pytest.raises(A.Reject, match="band_a_is_empty"):
        A.bind_crop_file(bench["path"], 0.5)


def test_an_empty_learnable_band_is_rejected(bench):
    """FACT-0382 measured that ALL 691 fold-0 contested errors have their true parent at or below
    the deployed threshold, so a cache with an empty band B holds none of the learnable
    population and a null on it would say nothing about any head."""
    def empty_b(data):
        for col in ("pair", "source_id", "target_id", "i", "j"):
            data[f"band_b_{col}"] = np.empty(0, dtype=np.int64)
        data["band_b_prob"] = np.empty(0, dtype=np.float64)
        data["source_id"] = data["band_a_source_id"]
        data["target_id"] = data["band_a_target_id"]
        data["edge_prob"] = data["band_a_prob"].astype(np.float32)

    _mutate(bench, empty_b)
    with pytest.raises(A.Reject, match="band_b_is_empty"):
        A.bind_crop_file(bench["path"], 0.5)


def test_a_band_a_row_below_the_deployed_threshold_is_rejected(bench):
    def lower(data):
        p = data["band_a_prob"].copy()
        p[0] = 0.4
        data["band_a_prob"] = p
        data["edge_prob"] = np.concatenate([p, data["band_b_prob"]]).astype(np.float32)

    _mutate(bench, lower)
    with pytest.raises(A.Reject, match="band_a_below_the_deployed_threshold"):
        A.bind_crop_file(bench["path"], 0.5)


def test_a_band_b_row_under_the_acquisition_floor_is_rejected(bench):
    def sink(data):
        p = data["band_b_prob"].copy()
        p[0] = 0.001
        data["band_b_prob"] = p
        data["edge_prob"] = np.concatenate([data["band_a_prob"], p]).astype(np.float32)

    _mutate(bench, sink)
    with pytest.raises(A.Reject, match="band_b_outside_its_acquisition_window"):
        A.bind_crop_file(bench["path"], 0.5)


def test_a_band_id_that_disagrees_with_the_role_table_is_rejected(bench):
    """The band artifacts address nodes by the same number the pre-ILP export and the ECB
    sidecars use. If the recorded global id is not what the role table says the local index is,
    an id in any downstream artifact names a different cell."""
    def relabel(data):
        s = data["band_a_source_id"].copy()
        s[0] = (int(s[0]) + 1) % int(data["node_count"])
        data["band_a_source_id"] = s
        data["source_id"] = np.concatenate([s, data["band_b_source_id"]])

    _mutate(bench, relabel)
    with pytest.raises(A.Reject, match="band_a_ids_disagree_with_the_role_table"):
        A.bind_crop_file(bench["path"], 0.5)


def test_a_union_surface_torn_from_its_bands_is_rejected(bench):
    def tear(data):
        data["edge_prob"] = data["edge_prob"][::-1].copy()

    _mutate(bench, tear)
    with pytest.raises(A.Reject, match="union_surface_disagrees_with_its_bands"):
        A.bind_crop_file(bench["path"], 0.5)


def test_an_uncapped_band_b_is_rejected(bench):
    """The tap's band B is top-k per TARGET. A cache holding more than that does not have the
    membership the ECB sidecars would record, so it is not the surface it claims to be."""
    def uncap(data):
        data["band_b_topk"] = np.int64(1)

    _mutate(bench, uncap)
    with pytest.raises(A.Reject, match="band_b_exceeds_its_rank_cap"):
        A.bind_crop_file(bench["path"], 0.5)


# =============================================================================================
# 6. TRUNK, FOLD AND PROVENANCE
# =============================================================================================
def test_a_swapped_trunk_is_rejected_by_the_feature_binding(bench):
    """FACT-0392's third risk in its realistic accidental form: two trunk caches produced in one
    session and the wrong directory consumed. Node set, shapes and counts identical; only the
    feature values differ."""
    def other_trunk(data):
        rng = np.random.default_rng(99)
        data["role_feat"] = rng.normal(
            size=data["role_feat"].shape).astype(data["role_feat"].dtype)

    _mutate(bench, other_trunk)
    with pytest.raises(A.Reject, match="features_unchanged"):
        A.audit(bench["cache"], bench["manifest"])


def test_a_swapped_trunk_checkpoint_is_rejected_by_its_hash(bench):
    other = bench["root"] / "trunk_stabledet.pth"
    other.write_bytes(b"STABLEDET-TRUNK-BYTE" * 64)
    with pytest.raises(A.Reject, match="trunk_checkpoint_sha256"):
        A.audit(bench["cache"], bench["manifest"], trunk=other)


def test_a_consumer_can_refuse_a_cache_from_the_wrong_trunk(bench):
    with pytest.raises(A.Reject, match="trunk_role_matches_expectation"):
        A.audit(bench["cache"], bench["manifest"], expect_role="stabledet")


@pytest.mark.parametrize("placeholder", ["", "   ", "unknown", "TBD", "n/a"])
def test_a_trunk_without_explicit_provenance_is_rejected(bench, placeholder):
    """Agent 4 established, and state_dict_shape re-measures, that every candidate trunk is a
    BARE state dict - 136 tensors, ZERO non-tensor keys - and that official_f0 and stabledet_f0
    share a byte size while differing in sha256. Nothing in the checkpoint can name it, so the
    manifest is the only place provenance can live and a placeholder there is a REJECT."""
    m = json.loads(bench["manifest"].read_text(encoding="utf-8"))
    m["trunk"]["provenance"] = placeholder
    bad = bench["root"] / "no_prov.json"
    bad.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(A.Reject, match="trunk_provenance_is_explicit_in_the_manifest"):
        A.audit(bench["cache"], bad)


def test_bind_refuses_a_placeholder_provenance_up_front(bench):
    """Rejecting at audit time is late; the operator has already spent the session. Refuse at
    bind."""
    args = A._bind_args(bench["cache"], bench["trunk"], fold="0", role="official",
                        weights_glob="/kaggle/input/*/split_0/edge_predictor_best.pth",
                        notebook=bench["notebook"], provenance="TODO")
    with pytest.raises(A.Reject, match="placeholder"):
        A.build_manifest(args)


def test_reordered_nodes_are_rejected(bench):
    """Positional node ids are only meaningful against a fixed order. This permutes coordinates
    inside one frame, changing nothing about shape, count or feature values."""
    def reorder(data):
        s, e = int(data["starts"][0]), int(data["ends"][0])
        coords = data["coords"].copy()
        coords[s:e] = coords[s:e][::-1]
        data["coords"] = coords

    _mutate(bench, reorder)
    with pytest.raises(A.Reject, match="node_order_unchanged"):
        A.audit(bench["cache"], bench["manifest"])


def test_wrong_fold_weights_are_rejected(bench):
    """The EXP-0019 defect reached through the cache: a fold-1 manifest carrying the pack's
    split_0 weights, which were trained on the very embryo fold 1 holds out."""
    m = json.loads(bench["manifest"].read_text(encoding="utf-8"))
    m["fold"].update({"fold": "1", "held_out_embryo": "6bba"})
    bad = bench["root"] / "f1.json"
    bad.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(A.Reject, match="fold1_does_not_use_the_leaky_pack_weights"):
        A.audit(bench["cache"], bad)


def test_a_fold_declared_over_the_other_embryo_is_rejected(bench):
    """Report both embryo directions separately (AGENTS.md). A cache whose crops belong to the
    other embryo than its declared fold would silently pool them."""
    m = json.loads(bench["manifest"].read_text(encoding="utf-8"))
    m["fold"].update({"fold": "1", "held_out_embryo": "6bba",
                      "weights_glob": "/kaggle/input/*/edge_predictor_best_split_1.pth"})
    bad = bench["root"] / "f1b.json"
    bad.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(A.Reject, match="embryo_matches_fold"):
        A.audit(bench["cache"], bad)


def test_an_unbound_cache_is_rejected(bench):
    """No manifest is not 'no opinion'. An unbound cache cannot distinguish 'the head does not
    work' from 'we fed it the wrong features', which is the whole failure this packet removes."""
    with pytest.raises(A.Reject, match="no manifest"):
        A.audit(bench["cache"], bench["root"] / "absent.json")


def test_a_schema_1_manifest_is_refused(bench):
    """Schema 1 keyed features BY FRAME. A schema-1 manifest cannot describe a pair-and-role
    cache, and silently accepting one would re-license the undefined layout."""
    m = json.loads(bench["manifest"].read_text(encoding="utf-8"))
    m["schema_version"] = 1
    bad = bench["root"] / "v1.json"
    bad.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(A.Reject, match="manifest schema"):
        A.audit(bench["cache"], bad)


# =============================================================================================
# 7. THE DUAL-TRUNK PAIR (FACT-0392 risk three, PKT-0029 item 3)
# =============================================================================================
def _sibling(bench, *, seed, role="stabledet", fold="0", scale=1.0):
    import shutil
    other = bench["root"] / f"sib_{seed}_{role}_{fold}_{scale}"
    other.mkdir()
    trunk = bench["root"] / f"trunk_{seed}_{role}_{fold}.pth"
    trunk.write_bytes(b"OTHER-TRUNK" * (64 + seed))
    crop = bench["crop"] if fold == "0" else "6bba_bbbbbbbb"
    shutil.copyfile(bench["path"], other / f"{crop}.npz")
    with np.load(other / f"{crop}.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if scale != 1.0:
        data["role_feat"] = (data["role_feat"] * np.float32(scale)).astype(np.float32)
    np.savez_compressed(other / f"{crop}.npz", **data)
    glob = ("/kaggle/input/*/split_0/edge_predictor_best.pth" if fold == "0"
            else "/kaggle/input/*/edge_predictor_best_split_1.pth")
    args = A._bind_args(other, trunk, fold=fold, role=role, weights_glob=glob,
                        notebook=bench["notebook"])
    (other / "cache_manifest.json").write_text(json.dumps(A.build_manifest(args)),
                                               encoding="utf-8")
    return other


def test_a_well_formed_dual_trunk_pair_is_accepted(bench):
    other = _sibling(bench, seed=7, scale=1.5)
    report = A.audit_dual_trunk(bench["cache"], other, require_roles=True)
    assert report["passed"] is True
    assert set(report["roles"]) == set(A.PKT0029_REQUIRED_TRUNK_ROLES)


def test_one_trunk_written_twice_is_not_a_pair(bench):
    """Identical features under two checkpoint names: the second trunk was never loaded."""
    other = _sibling(bench, seed=7, scale=1.0)
    with pytest.raises(A.Reject, match="dual_trunk_features_differ"):
        A.audit_dual_trunk(bench["cache"], other)


def test_a_pair_across_folds_is_rejected(bench):
    other = _sibling(bench, seed=7, fold="1", scale=1.5)
    with pytest.raises(A.Reject, match="dual_trunk_same_fold"):
        A.audit_dual_trunk(bench["cache"], other)


def test_a_pair_that_does_not_carry_both_pkt0029_roles_is_rejected(bench):
    """PKT-0029 item (3): the official and StableDet trunks must be cached and compared in the
    SAME session, or a null cannot separate 'the head does not work' from 'wrong trunk'."""
    other = _sibling(bench, seed=7, role="pack_split0", scale=1.5)
    with pytest.raises(A.Reject, match="dual_trunk_missing_a_required_role"):
        A.audit_dual_trunk(bench["cache"], other, require_roles=True)


# =============================================================================================
# 8. MEASURED BYTES AND THE HARD CAPACITY GUARD
# =============================================================================================
def test_bytes_per_crop_are_measured_not_estimated(bench):
    """The storage record must come from the file on disk, not from a rate."""
    rec = A.bind_crop_file(bench["path"], 0.5)
    st = rec["storage"]
    assert st["bytes_on_disk"] == bench["path"].stat().st_size
    assert st["uncompressed_bytes"] > st["bytes_on_disk"] > 0
    assert 0.0 < st["compression_ratio"] < 1.0
    assert st["bytes_per_role_node"] > 0 and st["bytes_per_band_row"] > 0


def test_the_capacity_guard_refuses_rather_than_warns(bench):
    """A guard that returns a warning is a guard the operator scrolls past. This one raises."""
    rep = A.audit(bench["cache"], bench["manifest"])
    counts = {f"44b6_{i:08x}": 25_000 for i in range(71)}
    with pytest.raises(A.Reject, match="CAPACITY"):
        A.project_full_fold(rep["measured"], counts, capacity_bytes=1 << 20, label="fold0")


def test_the_capacity_guard_accepts_a_projection_inside_budget(bench):
    rep = A.audit(bench["cache"], bench["manifest"])
    counts = {f"44b6_{i:08x}": 25_000 for i in range(71)}
    out = A.project_full_fold(rep["measured"], counts, capacity_bytes=64 * (1 << 30),
                              feat_dim=32, pos_dim=32, label="fold0")
    assert out["fits"] is True
    assert out["worst_case_bytes"] > out["expected_bytes_at_measured_compression"], (
        "the worst case must not be smaller than the compressed expectation")
    assert out["measured"]["bytes_on_disk"] == bench["path"].stat().st_size


def test_the_projection_refuses_without_a_stated_budget(bench):
    rep = A.audit(bench["cache"], bench["manifest"])
    with pytest.raises(A.Reject, match="projection_has_no_capacity"):
        A.project_full_fold(rep["measured"], {"44b6_a": 10}, capacity_bytes=0)


def test_the_projection_refuses_without_a_measurement(bench):
    """Every byte figure must trace to a real artifact; a projection from nothing is an estimate
    wearing a guard's clothes."""
    with pytest.raises(A.Reject, match="projection_has_no_measurement"):
        A.project_full_fold([], {"44b6_a": 10}, capacity_bytes=1 << 40)


def test_the_projection_refuses_when_its_band_a_bound_does_not_hold(bench):
    """The one-row-per-target bound on band A follows from the SOURCE-AXIS SOFTMAX summing to 1.
    Under any other activation it is unbounded, and a projection that quietly kept the bound
    would under-count by the size of the candidate surface."""
    rep = A.audit(bench["cache"], bench["manifest"])
    rep["measured"][0]["edge_activation"] = "sigmoid"
    with pytest.raises(A.Reject, match="projection_bound_does_not_hold"):
        A.project_full_fold(rep["measured"], {"44b6_a": 10}, capacity_bytes=1 << 40)


def test_the_worst_case_takes_no_compression_credit(bench):
    """The binding number must not depend on how well a fixture happened to compress."""
    rep = A.audit(bench["cache"], bench["manifest"])
    counts = {"44b6_a": 100_000}
    out = A.project_full_fold(rep["measured"], counts, capacity_bytes=1 << 40,
                              feat_dim=32, pos_dim=32, band_b_topk=8)
    per_node = A.schema_bytes_per_node(32, 32, 8)["total"]
    assert out["worst_case_bytes"] >= 100_000 * per_node


def test_read_node_counts_reads_a_real_run_stats_csv(tmp_path):
    csv_path = tmp_path / "run_stats.csv"
    csv_path.write_text("dataset,raw_nodes,nodes\n44b6_a,10,9\n44b6_b,20,18\n", encoding="utf-8")
    assert A.read_node_counts(csv_path) == {"44b6_a": 10, "44b6_b": 20}
    with pytest.raises(A.Reject, match="no 'dataset'"):
        A.read_node_counts(csv_path, column="not_a_column")


# =============================================================================================
# 9. THE CLI'S OWN MUTATION BATTERY, RUN SO IT CANNOT ROT
# =============================================================================================
def test_the_self_test_command_passes(tmp_path):
    out = tmp_path / "selftest.json"
    assert A.self_test(out) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["all_passed"] is True
    planted = {r["mutation"] for r in payload["results"] if r["expected"] == "reject"}
    accepted = {r["mutation"] for r in payload["results"] if r["expected"] == "accept"}
    assert {"defect6_frame_keyed_cache", "defect6_role_records_missing",
            "defect6_duplicate_role_record",
            "defect6_frame_keyed_features_in_a_pair_and_role_container",
            "projection_over_capacity_is_refused"} <= planted
    assert {"control_clean_cache", "control_valid_dual_trunk_pair",
            "control_projection_inside_budget"} <= accepted, (
        "an auditor with no accept control checks nothing")
    # every planted defect must have been rejected under the condition it was planted for
    for row in payload["results"]:
        if row["expected"] == "reject" and row["condition"]:
            assert row["condition"] in row["detail"], row
