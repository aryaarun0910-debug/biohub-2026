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


def pair_probabilities(pair: int) -> np.ndarray:
    """Softmax over the SOURCE axis of the role-specific features - the deployed activation."""
    logits = source_role_features(pair) @ target_role_features(pair + 1).T
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
    bands: dict[str, list] = {"a": [], "b": []}
    probs_by_pair = {}
    for pair in range(N_FRAMES - 1):
        probs = pair_probabilities(pair)
        probs_by_pair[pair] = probs
        for i, j in np.argwhere(probs > THRESHOLD).tolist():
            bands["a"].append((pair, i, j, float(probs[i, j])))
        for i, j, p in _select_band_b(probs):
            bands["b"].append((pair, i, j, p))

    payload = {
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
    for name in ("a", "b"):
        rows = bands[name]
        pair_c = np.asarray([r[0] for r in rows], dtype=np.int64)
        i_c = np.asarray([r[1] for r in rows], dtype=np.int64)
        j_c = np.asarray([r[2] for r in rows], dtype=np.int64)
        payload[f"band_{name}_pair"] = pair_c
        payload[f"band_{name}_i"] = i_c
        payload[f"band_{name}_j"] = j_c
        payload[f"band_{name}_source_id"] = gid_all[payload["pair_src_ptr"][pair_c] + i_c]
        payload[f"band_{name}_target_id"] = gid_all[payload["pair_tgt_ptr"][pair_c] + j_c]
        payload[f"band_{name}_prob"] = np.asarray([r[3] for r in rows], dtype=np.float64)
    payload["source_id"] = np.concatenate([payload["band_a_source_id"],
                                           payload["band_b_source_id"]])
    payload["target_id"] = np.concatenate([payload["band_a_target_id"],
                                           payload["band_b_target_id"]])
    payload["edge_prob"] = np.concatenate([payload["band_a_prob"],
                                           payload["band_b_prob"]]).astype(np.float32)
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
                                    z["band_a_prob"])]
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
            ecb_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                ecb_dir / f"{crop}.npz",
                source_id=np.asarray(z["band_b_source_id"], dtype=np.int64),
                target_id=np.asarray(z["band_b_target_id"], dtype=np.int64),
                edge_prob=np.asarray(z["band_b_prob"], dtype=np.float32))
            n_b = int(z["band_b_prob"].shape[0])
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
