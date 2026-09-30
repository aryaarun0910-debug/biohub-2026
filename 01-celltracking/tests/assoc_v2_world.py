"""The SCHEMA-V2 synthetic world: a pair-and-role feature cache in the tap's production layout.

WHY A SEPARATE MODULE. Both `test_assoc_train_harness.py` and its schema tests need this world,
and a fixture that two suites build differently is a fixture that proves two different things.

WHY THE LAYOUT IS COPIED FROM THE TAP RATHER THAN INVENTED.
`scripts/kaggle_edits/assoc_feature_tap.py` declares the schema in `_AFT_SCHEMA_REQUIRED` and
writes exactly it; `scripts/win_bet/audit_feature_cache.py` validates it. This writer is checked
against BOTH - every cache it emits is required to pass
`audit_feature_cache.crop_binding`, so a fixture that drifts from production fails its own suite
rather than quietly certifying a harness against a layout nothing produces. That is the direct
answer to the defect this migration exists to close: a prior fixture stubbed `model.detection_head`,
a method the real class has never had, and the suite stayed green.

THE FEATURES DETERMINE THE PROBABILITIES, WHICH IS WHAT GIVES THE GATE TEETH. `role_feat` is
constructed so that softmax over the SOURCE axis of `feat_src @ feat_tgt.T` reproduces the recorded
band probabilities exactly. A declared reproducer therefore re-derives them FROM THE CACHE ALONE,
and perturbing the cache breaks the reproduction rather than merely disagreeing with a hard-coded
number.

AND A NODE'S TWO ROLES CARRY DIFFERENT VECTORS, because that is the thing schema 1 could not
represent. A node at frame `f` is the SOURCE of pair `(f, f+1)` and the TARGET of pair `(f-1, f)`,
and this writer gives it a different vector in each - which is what `_TemporalAttention` does on the
real path (`FACT-0402`).

CONTRACT 2 - EVERY PROBABILITY NAMES ITS SURFACE. Contract 1 wrote one column per band,
`band_a_prob` / `band_b_prob`, plus an unqualified `edge_prob` union, all taken from the deployed
`probs` AFTER every fusion stage. A primary-only replay compared against that disagreed by up to
0.43 on P36 and the cache was blamed; `FACT-0407` re-derived the whole thing on CPU and found the
cache FAITHFUL to 1e-6 once the deployed bidirectional harmonic was applied. The defect was the
NAME. So this writer emits `primary_logit_preblend`, `primary_prob_preblend` and
`deployed_prob_postblend` per band, a BAND P (the same threshold rule on the PRE-fusion surface),
the contract header, and the fusion record - and it runs NO fusion stage, so band P is band A and
the two probability columns are bitwise equal, which is exactly the identity the auditor checks
when `fusion_stages` is "none".

TWO `edge_prob` COLUMNS SURVIVE THE RENAME AND MUST. The pre-ILP export (`preilp_rows`) and the
ECB sidecar (`build_world`) are DIFFERENT ARTIFACTS with UNCHANGED schemas - the harness
reconstructs its bands from them at `assoc_train_harness.py:450` and `:458`. Only the CACHE
payload's probability columns were renamed.

EVERY SCHEMA CONSTANT BELOW COMES FROM `audit_feature_cache` - `AFC.BANDS`, `AFC.BAND_COLUMNS`,
`AFC.BAND_SURFACE`, `AFC.UNION_BANDS`, `AFC.REQUIRED_KEYS`, `AFC.CACHE_CONTRACT_VERSION` - and the
writer asserts its emitted key set against `AFC.REQUIRED_KEYS` before any mutation. Restating a
schema by hand is how five of the eight defects in this lane happened.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import audit_feature_cache as AFC  # noqa: E402

# The deployed contract, mirrored: window 2, stride W-1 = 1, downsample (1, 4, 4).
N_FRAMES, N_PER_FRAME, DIM, POS_DIM = 6, 12, 12, 32
GRID = (8, 32, 32)
DOWNSAMPLE = (1, 4, 4)
WINDOW = 2
THRESHOLD, FLOOR_B, TOPK = 0.5, 0.02, 8
CROPS = [f"44b6_{i:08x}" for i in range(8)]
NODES_PER_CROP = N_FRAMES * N_PER_FRAME

# The logit scale, chosen so BOTH bands are non-empty and neither is vacuous: with 12 sources the
# aligned one lands at ~0.573 (band A, above the deployed 0.5), the runner-up at ~0.141 and the
# other ten at ~0.029 (band B, inside the (0.02, 0.5] acquisition window). A band that compares
# nothing is FACT-0387's failure mode and the fixture must not reintroduce it.
MATCH, RUNNER_UP = 3.0, 1.6

# The two surface descriptions, copied VERBATIM from the tap's `_AFT_SURFACE_PRIMARY` and
# `_AFT_SURFACE_DEPLOYED` (`scripts/kaggle_edits/assoc_feature_tap.py:103-112`). They are copied
# rather than imported because the tap is a PATCH SCRIPT: importing it executes a source rewrite
# against a `_ps` global. The auditor does not validate their text, so nothing here is load
# bearing - but a fixture that writes a DIFFERENT surface description is a fixture describing a
# different program, and that is defect 8's exact shape.
PRIMARY_SURFACE = (
    "primary: model.predict_edges(unet_feat_src, unet_feat_tgt, ...) forward output at "
    "predict_unet_transformer.py:595-600, BEFORE the bidirectional harmonic (:602-659) and "
    "BEFORE the secondary logit blend (:660-756)"
)
DEPLOYED_SURFACE = (
    "deployed: probs at predict_unet_transformer.py:758-762, AFTER every fusion stage that ran; "
    "this is the surface the deployed candidate rule reads"
)


def _onehot(k: int, scale: float = 1.0) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[k % DIM] = scale
    return v


def source_role_features(frame: int) -> np.ndarray:
    """The vector every node of `frame` carries when it is the SOURCE of pair (frame, frame+1)."""
    return np.stack([_onehot(i + frame) for i in range(N_PER_FRAME)])


def target_role_features(frame: int) -> np.ndarray:
    """The DIFFERENT vector the same nodes carry as the TARGET of pair (frame-1, frame).

    Different by construction, because on the deployed path the two come from different forward
    passes over different windows (`FACT-0402`). A writer that returned one array here would be
    defect 6 wearing the pair-and-role schema.
    """
    return np.stack([_onehot(j + frame, MATCH) + _onehot(j + frame + 1, RUNNER_UP)
                     for j in range(N_PER_FRAME)])


def _coords() -> np.ndarray:
    rows = []
    for f in range(N_FRAMES):
        for i in range(N_PER_FRAME):
            rows.append([f, i % GRID[0], (i * 2) % GRID[1], (i * 3 + f) % GRID[2]])
    return np.asarray(rows, dtype=np.int16)


def pair_logits(pair: int) -> np.ndarray:
    """The PRE-ACTIVATION logits of a pair - contract 2's `primary_logit_preblend`.

    The fixture stores the real dot-product logits rather than `log(p)`, because the auditor's
    rank check (`band_{x}_logit_and_probability_disagree`) exists to see a TORN logit column, and
    a column derived from the probability it is checked against could not be torn independently
    of it. A logit that is a function of the probability is a fixture differing from production
    in KIND, not in cost.
    """
    return source_role_features(pair) @ target_role_features(pair + 1).T


def pair_probabilities(pair: int) -> np.ndarray:
    """Softmax over the SOURCE axis of the role-specific features - the deployed activation."""
    logits = pair_logits(pair)
    e = np.exp(logits.astype(np.float64) - logits.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)


def _select_band_b(probs: np.ndarray) -> list[tuple[int, int, float]]:
    """The tap's `_aft_select_band_b`: top-k per TARGET above the floor, minus band A's members."""
    out = []
    for j in range(probs.shape[1]):
        col = [(float(probs[i, j]), i) for i in range(probs.shape[0]) if probs[i, j] > FLOOR_B]
        col.sort(key=lambda t: (-t[0], -t[1]))
        for rank, (p, i) in enumerate(col):
            if rank < TOPK and p <= THRESHOLD:
                out.append((i, j, p))
    return out


def write_v2_cache(path: Path, crop: str, *, mutate=None) -> dict:
    """Write ONE crop cache in the tap's production layout. Returns the recorded probabilities."""
    coords = _coords()
    starts = np.asarray([f * N_PER_FRAME for f in range(N_FRAMES)], dtype=np.int64)
    ends = starts + N_PER_FRAME
    rng = np.random.default_rng(abs(hash(crop)) % (2**32))

    r_pair, r_role, r_gid, r_feat, r_scaled, r_rel, r_mask, r_pos = ([] for _ in range(8))
    p_sp, p_sn, p_tp, p_tn = [], [], [], []
    cursor = 0
    for pair in range(N_FRAMES - 1):
        for role, frame in ((AFC.ROLE_SRC, pair), (AFC.ROLE_TGT, pair + 1)):
            gid = np.arange(starts[frame], ends[frame], dtype=np.int64)
            feat = source_role_features(frame) if role == AFC.ROLE_SRC \
                else target_role_features(frame)
            rel = coords[starts[frame]:ends[frame]].astype(np.int32).copy()
            rel[:, 0] = role                       # f_idx is 0 at window 2: src=0, tgt=1
            r_pair.append(np.full(N_PER_FRAME, pair, dtype=np.int64))
            r_role.append(np.full(N_PER_FRAME, role, dtype=np.int8))
            r_gid.append(gid)
            r_feat.append(feat.astype(np.float32))
            r_scaled.append(rel[:, 1:].astype(np.float32)
                            * np.asarray(DOWNSAMPLE, dtype=np.float32))
            r_rel.append(rel)
            r_mask.append(np.ones(N_PER_FRAME, dtype=bool))
            r_pos.append(rng.normal(size=(N_PER_FRAME, POS_DIM)).astype(np.float32))
            if role == AFC.ROLE_SRC:
                p_sp.append(cursor); p_sn.append(N_PER_FRAME)
            else:
                p_tp.append(cursor); p_tn.append(N_PER_FRAME)
            cursor += N_PER_FRAME

    gid_all = np.concatenate(r_gid)
    bands: dict[str, list] = {b: [] for b in AFC.BANDS}
    probs_by_pair = {}
    for pair in range(N_FRAMES - 1):
        logits = pair_logits(pair)
        probs = pair_probabilities(pair)
        probs_by_pair[pair] = probs
        for i, j in np.argwhere(probs > THRESHOLD).tolist():
            bands[AFC.BAND_A].append((pair, i, j, float(logits[i, j]), float(probs[i, j])))
        for i, j, p in _select_band_b(probs):
            bands[AFC.BAND_B].append((pair, i, j, float(logits[i, j]), p))
    # BAND P is the SAME threshold rule applied to the PRE-fusion surface. This writer runs NO
    # fusion stage, so the pre- and post-fusion surfaces are the same numbers and band P's
    # membership is band A's - the identity `_validate_bands` requires when `fusion_stages` is
    # "none", and the control arm FACT-0407 used to attribute P36's gap to the fusion rather than
    # to the cache. Emitting band P by copying band A is therefore not a shortcut: with no stage
    # it is the only membership the rule can select.
    bands[AFC.BAND_P] = list(bands[AFC.BAND_A])

    payload = {
        # THE CONTRACT-2 HEADER. Contract 1 wrote one probability column per band whose name
        # claimed no surface, which is what made P36's verdict invalid while its cache was
        # faithful (FACT-0403, corrected by FACT-0407). Both version stamps are the auditor's own
        # exported constants rather than literals; the two surface descriptions are the tap's,
        # copied verbatim above because the tap cannot be imported.
        "schema_version": np.int64(AFC.SCHEMA_VERSION),
        "contract_version": np.int64(AFC.CACHE_CONTRACT_VERSION),
        "primary_surface": np.str_(PRIMARY_SURFACE),
        "deployed_surface": np.str_(DEPLOYED_SURFACE),
        # NO FUSION STAGE RUNS IN THIS FIXTURE, and the record must say so rather than leave it
        # open: `fusion_stages` is what a reader consults to decide whether the deployed surface
        # is the primary surface, and it is the claim the two probability columns are checked
        # against. A fixture that claimed a stage it does not run would be defect 8's shape.
        "fusion_stages": np.str_("none"),
        "fusion_bidirectional_weight": np.float64(0.0),
        "fusion_secondary_enabled": np.bool_(False),
        "fusion_secondary_edge_weight": np.float64(0.0),
        "fusion_secondary_link_mode": np.str_(""),
        "fusion_secondary_mix_temperature": np.float64(1.0),
        "fusion_reproducible_from_primary_cache": np.bool_(True),
        "crop": np.str_(crop),
        "window": np.int64(WINDOW),
        "downsample": np.asarray(DOWNSAMPLE, dtype=np.int64),
        "voxel_size": np.asarray([1.625, 1.625, 1.625], dtype=np.float64),
        "pool_kernel": np.asarray([1, 3, 3], dtype=np.int64),
        "q_low": np.float64(50.0), "q_high": np.float64(950.0),
        "image_shape": np.asarray([N_FRAMES, *GRID], dtype=np.int64),
        "det_threshold": np.float64(0.96875), "det_tta": np.bool_(False),
        "edge_threshold": np.float64(THRESHOLD), "edge_activation": np.str_("softmax"),
        "band_b_floor": np.float64(FLOOR_B), "band_b_topk": np.int64(TOPK),
        "feat_dtype": np.str_("float32"), "feat_dim": np.int64(DIM),
        "pos_dim": np.int64(POS_DIM), "node_count": np.int64(len(coords)),
        "coords": coords,
        "frames": np.arange(N_FRAMES, dtype=np.int64),
        "starts": starts, "ends": ends,
        "pair_f_idx": np.zeros(N_FRAMES - 1, dtype=np.int64),
        "pair_t_src": np.arange(N_FRAMES - 1, dtype=np.int64),
        "pair_t_tgt": np.arange(1, N_FRAMES, dtype=np.int64),
        "pair_src_ptr": np.asarray(p_sp, dtype=np.int64),
        "pair_src_n": np.asarray(p_sn, dtype=np.int64),
        "pair_tgt_ptr": np.asarray(p_tp, dtype=np.int64),
        "pair_tgt_n": np.asarray(p_tn, dtype=np.int64),
        "pair_window_shape": np.asarray([[WINDOW, *GRID]] * (N_FRAMES - 1), dtype=np.int64),
        "role_pair": np.concatenate(r_pair), "role_role": np.concatenate(r_role),
        "role_gid": gid_all, "role_feat": np.concatenate(r_feat).astype(np.float32),
        "role_coord_scaled": np.concatenate(r_scaled).astype(np.float32),
        "role_coord_rel": np.concatenate(r_rel).astype(np.int32),
        "role_mask": np.concatenate(r_mask), "role_pos": np.concatenate(r_pos).astype(np.float32),
    }
    for name in AFC.BANDS:
        rows = bands[name]
        pair_c = np.asarray([r[0] for r in rows], dtype=np.int64)
        i_c = np.asarray([r[1] for r in rows], dtype=np.int64)
        j_c = np.asarray([r[2] for r in rows], dtype=np.int64)
        prob_c = np.asarray([r[4] for r in rows], dtype=np.float64)
        cols = {
            "pair": pair_c, "i": i_c, "j": j_c,
            "source_id": gid_all[payload["pair_src_ptr"][pair_c] + i_c],
            "target_id": gid_all[payload["pair_tgt_ptr"][pair_c] + j_c],
            "primary_logit_preblend": np.asarray([r[3] for r in rows], dtype=np.float64),
            "primary_prob_preblend": prob_c,
            # WITH NO STAGE THE DEPLOYED PROBABILITY *IS* THE PRIMARY ACTIVATION - the same
            # tensor through the same op - so the auditor requires the two columns to be BITWISE
            # equal. A separately recomputed array would be a different KIND of object.
            "deployed_prob_postblend": prob_c.copy(),
        }
        assert set(cols) == set(AFC.BAND_COLUMNS), (
            "the fixture no longer writes the auditor's band column contract: extra "
            f"{sorted(set(cols) - set(AFC.BAND_COLUMNS))}, missing "
            f"{sorted(set(AFC.BAND_COLUMNS) - set(cols))}")
        # WHICH SURFACE SELECTED THIS BAND, taken from the auditor's mirror of the tap's own
        # `_AFT_BAND_SURFACE` rather than typed here. Restating a schema by hand is how five of
        # the eight defects happened.
        payload[f"band_{name}_selected_on"] = np.str_(AFC.BAND_SURFACE[name])
        for col in AFC.BAND_COLUMNS:
            payload[f"band_{name}_{col}"] = cols[col]
    # The auditor-facing union is the DEPLOYED candidate surface, bands A and B only - band P is
    # a gate instrument and folding it in would inflate the surface by rows the deployment never
    # saw. There is no unqualified `edge_prob` in a contract-2 cache; both probability columns
    # name the surface they came from.
    payload["source_id"] = np.concatenate(
        [payload[f"band_{b}_source_id"] for b in AFC.UNION_BANDS])
    payload["target_id"] = np.concatenate(
        [payload[f"band_{b}_target_id"] for b in AFC.UNION_BANDS])
    payload["deployed_edge_prob_postblend"] = np.concatenate(
        [payload[f"band_{b}_deployed_prob_postblend"] for b in AFC.UNION_BANDS]).astype(np.float32)
    payload["primary_edge_prob_preblend"] = np.concatenate(
        [payload[f"band_{b}_primary_prob_preblend"] for b in AFC.UNION_BANDS]).astype(np.float32)
    # THE FIXTURE IS CHECKED AGAINST THE PRODUCTION KEY LIST BEFORE ANY MUTATION. A test that
    # manufactures a defect may drop or add keys on purpose; the UNMUTATED writer may not, and a
    # silent divergence here is exactly the class of failure that kept eight defects green.
    assert set(payload) == set(AFC.REQUIRED_KEYS) | {AFC.POSITIONAL_KEY}, (
        "the fixture no longer emits the schema production writes: extra "
        f"{sorted(set(payload) - set(AFC.REQUIRED_KEYS) - {AFC.POSITIONAL_KEY})}, missing "
        f"{sorted(set(AFC.REQUIRED_KEYS) - set(payload))}")
    if mutate is not None:
        mutate(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return probs_by_pair


def preilp_rows(crop: str, cache_path: Path) -> tuple[list[dict], list[dict]]:
    """Node and edge rows as the pre-ILP export records them - at ORIGINAL resolution.

    `predict_video` returns `coords[:, 1:] * downsample` cast to int16
    (`predict_unet_transformer.py:798-802`), and the export saves the graph built from THAT, while
    the tap flushes BEFORE the rescale. So the two artifacts are in different grids on purpose and
    the harness's parity check must apply the factor. Verified on the archived P36 cache: raw
    parity False, rescaled parity True on every node of both crops.

    THE PRE-ILP EXPORT'S `edge_prob` IS NOT THE CACHE'S. It is a DIFFERENT ARTIFACT with an
    UNCHANGED schema - the deployed graph export the harness reads at
    `assoc_train_harness.py:450` - and it holds the DEPLOYED post-fusion probability because that
    is what `predict_video` wrote. Contract 2 renamed the CACHE's column, not this one; renaming
    it here would break the harness's band-A reconstruction. What changes is only WHICH cache
    column the value is read from: `band_a_deployed_prob_postblend`, the post-fusion surface,
    which is what contract 1's `band_a_prob` held all along under a name that claimed no surface.
    """
    with np.load(cache_path, allow_pickle=False) as z:
        coords = z["coords"]
        ds = np.asarray(z["downsample"], dtype=np.float32)
        rescaled = (coords[:, 1:].astype(np.float32) * ds).astype(np.int16)
        nodes = [{"dataset": crop, "row_type": "node", "node_id": nid, "t": int(coords[nid, 0]),
                  "z": float(rescaled[nid, 0]), "y": float(rescaled[nid, 1]),
                  "x": float(rescaled[nid, 2]), "source_id": None, "target_id": None,
                  "edge_prob": None}
                 for nid in range(coords.shape[0])]
        edges = [{"dataset": crop, "row_type": "edge", "node_id": None, "t": None,
                  "z": None, "y": None, "x": None, "source_id": int(a), "target_id": int(b),
                  "edge_prob": float(p)}
                 for a, b, p in zip(z["band_a_source_id"], z["band_a_target_id"],
                                    z["band_a_deployed_prob_postblend"])]
    return nodes, edges


PREILP_SCHEMA = {"dataset": pl.String, "row_type": pl.String, "node_id": pl.Int64,
                 "t": pl.Int64, "z": pl.Float64, "y": pl.Float64, "x": pl.Float64,
                 "source_id": pl.Int64, "target_id": pl.Int64, "edge_prob": pl.Float64}


def build_world(root: Path, crops: list[str] | None = None) -> dict:
    """Cache, pre-ILP parquet, ECB sidecars, a Gate-1 receipt, a trunk and a bound manifest."""
    crops = list(crops or CROPS)
    cache_dir, ecb_dir = root / "cache", root / "ecb"
    node_rows, edge_rows, receipt_crops = [], [], []
    probs = {}
    for crop in crops:
        probs[crop] = write_v2_cache(cache_dir / f"{crop}.npz", crop)
        n, e = preilp_rows(crop, cache_dir / f"{crop}.npz")
        node_rows += n
        edge_rows += e
        with np.load(cache_dir / f"{crop}.npz", allow_pickle=False) as z:
            # THE ECB SIDECAR IS ALSO A DIFFERENT ARTIFACT with an UNCHANGED schema, read by the
            # harness at `assoc_train_harness.py:458`. Its `edge_prob` keeps its name; only the
            # cache column it is filled from is renamed, to the DEPLOYED post-fusion surface the
            # deployment's own acquisition rule selected band B on (`AFC.BAND_SURFACE`).
            ecb_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                ecb_dir / f"{crop}.npz",
                source_id=np.asarray(z["band_b_source_id"], dtype=np.int64),
                target_id=np.asarray(z["band_b_target_id"], dtype=np.int64),
                edge_prob=np.asarray(z["band_b_deployed_prob_postblend"], dtype=np.float32))
            n_b = int(z["band_b_deployed_prob_postblend"].shape[0])
        receipt_crops.append({"crop": crop, "passed": True, "node_count_mismatches": [],
                              "band_a": {"missing": 0, "extra": 0, "max_abs_prob_delta": 0.0},
                              "band_b": {"checked": n_b, "missing": 0,
                                         "max_abs_prob_delta": 0.0}})

    preilp = root / "preilp.parquet"
    pl.DataFrame(node_rows + edge_rows, schema=PREILP_SCHEMA).write_parquet(preilp)
    receipt = root / "assoc_feature_tap_gate.json"
    receipt.write_text(json.dumps({"gate": "feature_tap_parity", "attempt": 4,
                                   "all_passed": True, "crops": receipt_crops}), encoding="utf-8")

    # TWO TRUNKS OF THE SAME BYTE SIZE AND DIFFERENT sha256 - the FACT-0392 shape verbatim, so a
    # size-based identity check would confuse them and a hash-based one cannot.
    trunk = root / "trunk_official.pth"
    trunk.write_bytes(b"OFFICIAL-TRUNK-BYTES" * 64)
    impostor = root / "trunk_stabledet.pth"
    impostor.write_bytes(b"STABLEDET-TRUNK-BYTE" * 64)
    assert trunk.stat().st_size == impostor.stat().st_size
    assert AFC.sha256_file(trunk) != AFC.sha256_file(impostor)

    notebook = root / "notebook.ipynb"
    notebook.write_text('{"cells": []}', encoding="utf-8")
    manifest = cache_dir / "cache_manifest.json"
    manifest.write_text(json.dumps(AFC.build_manifest(AFC._bind_args(
        cache_dir, trunk, fold="0", role="official",
        weights_glob="loeo_official_f0_e3/split_0/*.pth", notebook=notebook)), indent=2),
        encoding="utf-8")
    return {"cache_dir": cache_dir, "ecb_dir": ecb_dir, "preilp": preilp, "receipt": receipt,
            "manifest": manifest, "trunk": trunk, "impostor": impostor, "crops": crops,
            "probs": probs, "notebook": notebook, "root": root}


# --------------------------------------------------------------------------------------
# the candidate surface, built FROM the cache so every id is a real node in a real pair
# --------------------------------------------------------------------------------------
# (n_candidates, baseline_is_correct). The true parent always has the smallest dist_um, so a model
# that reads geometry can convert the wrong ones; the deployed probability alone cannot.
PATTERN = [(3, True), (2, False), (1, True), (3, False), (1, True), (2, True)]


def build_surface(world: dict, crops: list[str] | None = None) -> pl.DataFrame:
    rows = []
    for ci, crop in enumerate(crops or world["crops"]):
        out_deg: dict[int, int] = {}
        g = 0
        for pair, probs in sorted(world["probs"][crop].items()):
            src_base, tgt_base = pair * N_PER_FRAME, (pair + 1) * N_PER_FRAME
            for j in range(N_PER_FRAME):
                n_cand, ok = PATTERN[g % len(PATTERN)]
                order = np.argsort(-probs[:, j])[:n_cand]
                target = tgt_base + j
                true_idx = 0 if ok else 1 % n_cand
                unreachable = (g % 12 == 11)
                best = float(probs[order[0], j])
                for k, i in enumerate(order.tolist()):
                    source = src_base + i
                    out_deg[source] = out_deg.get(source, 0) + 1
                    is_true = int((k == true_idx) and not unreachable)
                    dist = 2.0 + 0.01 * ci if is_true else 6.0 + 0.5 * k
                    rows.append({
                        "crop": crop, "target": target, "source": source,
                        "prob": float(probs[i, j]), "rank": k + 1,
                        "margin_to_best": best - float(probs[i, j]),
                        "n_candidates": n_cand, "dist_um": dist,
                        "dz_um": 0.0, "dy_um": dist, "dx_um": 0.0,
                        "src_out_degree": 0,
                        "is_true_parent": is_true,
                        "target_has_true_parent": 1,
                        "true_parent_is_candidate": int(not unreachable),
                        "target_matched_gt": 1,
                    })
                g += 1
        for r in rows:
            if r["crop"] == crop:
                r["src_out_degree"] = out_deg.get(r["source"], 1)
    return pl.DataFrame(rows)
