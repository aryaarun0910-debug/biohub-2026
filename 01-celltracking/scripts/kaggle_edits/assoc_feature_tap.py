# =====================================================================================
# PASSIVE FEATURE TAP  (PKT-0029 Gate 1, attempt 4 - REDESIGN ordered by the host 2026-08-30)
#
# WHY THIS EXISTS AND WHY IT IS NOT A REPAIR OF THE OLD WORKER.
# Three attempts at Gate 1 produced FIVE distinct defects (FACT-0387, FACT-0397, FACT-0399,
# FACT-0400) and every one shares a single cause: the gate's worker was written to APPROXIMATE
# the deployed path from a reading of it, instead of REUSING it. The decisive one is FACT-0400 -
# the worker called `open_dataset(path).image` bare, taking normalize=True and no downsample,
# while the deployed `predict_video` opens with normalize=False, load_image=False, reads the zarr
# group "0" directly, normalises from the dataset's own recorded 0.001/0.999 quantiles and carries
# downsample (1, 4, 4) into both the array and the voxel size. A parity gate that reimplements the
# pipeline it checks can only ever be as correct as its reimplementation.
#
# A SIXTH DEFECT OF THE SAME CLASS, FOUND WHILE DESIGNING THIS AND PROVEN ON CPU. The deployed
# node feature is NOT a per-frame quantity. `TemporalUNet3D` contains `_TemporalAttention`, which
# mixes across the time axis of the window (temporal_unet.py:30-48, 125-131), so `unet_out[:, i]`
# depends on every frame in the window. With the deployed window_size = 2 and stride W-1 = 1, an
# interior frame t appears in TWO windows: as the TARGET of pair (t-1, t) from window [t-1, t] and
# as the SOURCE of pair (t, t+1) from window [t, t+1]. Those are different UNet forward passes over
# different inputs, so frame t has TWO distinct feature vectors and the pair each belongs to is
# part of its identity. The old worker cached `feats[t]` once, on first sight, and reused it for
# both roles - so for every pair (t, t+1) with t >= 1 it fed the head SOURCE features taken from
# the wrong window. Measured on CPU with a random model: 4.1% relative difference, four orders of
# magnitude above the 1e-4 gate tolerance. That error is not uniform across the source axis, so the
# softmax blind spot below would NOT have hidden it: a fourth GPU session would have ended in a
# FAIL caused entirely by the gate's own reimplementation.
#
# WHAT THIS PATCH DOES INSTEAD. It injects a passive tap INSIDE the deployed predictor by rewriting
# its source text - the pattern every working patch in this tree uses (pre_ilp_export.py,
# edge_candidate_budget.py). The tap records the EXACT tensors `predict_video` hands to
# `model.predict_edges`, at the exact site, together with the deployed probabilities computed one
# line later. Nothing about the image pipeline, the normalisation, the downsample, the voxel scale,
# the detection peaks, the window partition, the masks or the node ordering is recomputed here.
# There is therefore no preprocessing replica left to be wrong. That removes the entire defect
# class rather than the five instances of it we have paid for.
#
# PASSIVE, AND PROVEN PASSIVE TWO WAYS.
#   (1) STATICALLY: every insertion is wrapped in AFT-INSERT-BEGIN/END markers and the patch is
#       pure insertion at line boundaries. `aft_strip_inserts()` removes the marked spans and the
#       result must be BYTE-IDENTICAL to the pre-patch source; the patch asserts this itself and
#       records both sha256s in its receipt, so the kernel can re-prove it after the fact.
#   (2) DYNAMICALLY: tests/test_assoc_feature_tap.py runs the REAL `predict_video` on CPU over a
#       real zarr, pristine versus tapped, and requires `coords` and `all_edges` to be identical.
#
# INERT UNLESS CONFIGURED. With BIOHUB_AFT_DIR unset the tap records nothing and writes nothing;
# the capture call is behind an `if _AFT_ON` guard so the deployed path pays nothing. That is also
# the in-kernel control arm.
#
# FAIL CLOSED. Every anchor must match exactly once or this raises. A misconfigured band floor or
# top-k raises at predictor import rather than exporting a quietly empty band - FACT-0394 is the
# cost of a band that compared nothing and reported a pass.
#
# ONE KNOWN BLIND SPOT, CARRIED FORWARD AS A STATED LIMIT. The comparison is over probabilities
# AFTER a softmax on the SOURCE axis, and softmax is invariant to a constant offset along the axis
# it normalises. A cache error that shifts EVERY SOURCE IN A FRAME PAIR BY THE SAME AMOUNT is
# mathematically invisible - not merely undetected in practice. Any non-uniform error, including a
# single node's features being wrong, does show up. Both halves are asserted by the test suite so
# neither can be forgotten. Closing it would need pre-softmax logits, which nothing in the current
# artifacts records.
# =====================================================================================
import hashlib as _aft_patch_hashlib
import json as _aft_patch_json
import os as _aft_patch_os
import re as _aft_patch_re

_AFT_BEGIN = "# --- AFT-INSERT-BEGIN {} ---"
_AFT_END = "# --- AFT-INSERT-END {} ---"
_AFT_SPAN_RE = _aft_patch_re.compile(
    r"^# --- AFT-INSERT-BEGIN (\w+) ---\n.*?^# --- AFT-INSERT-END \1 ---\n",
    _aft_patch_re.M | _aft_patch_re.S,
)


def aft_wrap(name, body):
    """Wrap an inserted span in removable markers, at line boundaries."""
    return _AFT_BEGIN.format(name) + "\n" + body + _AFT_END.format(name) + "\n"


def aft_strip_inserts(text):
    """Remove every AFT-marked span. Pure insertion means this restores the original bytes."""
    return _AFT_SPAN_RE.sub("", text)


# ---------------------------------------------------------------------------------------------
# SPAN 1: module-level config and the tap itself, inserted immediately before predict_video.
# ---------------------------------------------------------------------------------------------
_AFT_CONFIG = '''
# PASSIVE FEATURE TAP (PKT-0029 Gate 1). Records what the deployed head was actually fed.
# Inert unless BIOHUB_AFT_DIR is set. Never writes to coords, all_edges, probs or candidates.
import os as _aft_os
import numpy as _aft_np
import torch as _aft_torch


def _aft_flag(name, default):
    raw = _aft_os.environ.get(name, "").strip()
    return default if raw == "" else raw not in ("0", "false", "False", "no")


def _aft_probability(name, default):
    raw = _aft_os.environ.get(name, "").strip()
    value = default if raw == "" else float(raw)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value}")
    return value


def _aft_positive_int(name, default):
    raw = _aft_os.environ.get(name, "").strip()
    value = default if raw == "" else int(raw)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


_AFT_DIR = _aft_os.environ.get("BIOHUB_AFT_DIR", "").strip() or None
_AFT_ON = _AFT_DIR is not None
# Band B is where the learnable task lives: FACT-0382 measured that ALL 691 fold-0 contested
# errors have their true parent at or below the deployed 0.5, so a band-A-only gate validates
# exactly the band the task does not use. The floor and cap mirror the ECB acquisition rule.
_AFT_BAND_B_FLOOR = _aft_probability("BIOHUB_AFT_BAND_B_FLOOR", 0.02)
_AFT_BAND_B_TOPK = _aft_positive_int("BIOHUB_AFT_BAND_B_TOPK", 8)
_AFT_MAX_PAIRS = int(_aft_os.environ.get("BIOHUB_AFT_MAX_PAIRS", "0") or 0)   # 0 = uncapped
_AFT_STORE_POS = _aft_flag("BIOHUB_AFT_STORE_POS", True)
_AFT_CROPS = [c for c in _aft_os.environ.get("BIOHUB_AFT_CROPS", "").split(",") if c.strip()]

# THE CACHE SCHEMA, DECLARED RATHER THAN IMPLIED. The flush asserts it emits exactly this, and
# the preflight requires every key the replay worker reads to be one of these - so a rename on
# either side is caught on CPU instead of at the end of a GPU session.
_AFT_SCHEMA_REQUIRED = (
    "crop", "window", "downsample", "voxel_size", "pool_kernel", "q_low", "q_high",
    "image_shape", "det_threshold", "det_tta", "edge_threshold", "edge_activation",
    "band_b_floor", "band_b_topk", "feat_dtype", "feat_dim", "pos_dim", "node_count",
    "coords", "frames", "starts", "ends",
    "pair_f_idx", "pair_t_src", "pair_t_tgt", "pair_src_ptr", "pair_src_n",
    "pair_tgt_ptr", "pair_tgt_n", "pair_window_shape",
    "role_pair", "role_role", "role_gid", "role_feat", "role_coord_scaled",
    "role_coord_rel", "role_mask",
    "band_a_pair", "band_a_source_id", "band_a_target_id", "band_a_i", "band_a_j",
    "band_a_prob",
    "band_b_pair", "band_b_source_id", "band_b_target_id", "band_b_i", "band_b_j",
    "band_b_prob",
    "source_id", "target_id", "edge_prob",
)
_AFT_SCHEMA_OPTIONAL = ("role_pos",)

# Buffers are keyed by crop and reset when the crop changes, so a crop that raises part way
# through cannot leak its rows into the next one.
_AFT_STATE = {"crop": None}
_AFT_PAIRS = []
_AFT_ROLE = []
_AFT_BAND_A = []
_AFT_BAND_B = []
_AFT_TOTAL = {"crops": 0, "pairs": 0, "band_a": 0, "band_b": 0}


def _aft_reset(crop):
    _AFT_STATE["crop"] = crop
    _AFT_STATE["role_n"] = 0
    del _AFT_PAIRS[:], _AFT_ROLE[:], _AFT_BAND_A[:], _AFT_BAND_B[:]


def _aft_select_band_b(probs, threshold):
    """The ECB acquisition rule, then drop what band A already covers.

    Top-k per TARGET over everything above the floor - which is exactly what
    scripts/kaggle_edits/edge_candidate_budget.py's `_ecb_select` computes when it walks the
    global descending order capping each target, so band B here has the same membership the
    sidecars would have recorded. Pairs above the deployed threshold are then removed because
    band A owns them. Vectorised because the comprehension form is O(n_src x n_tgt) in Python
    and a full fold would spend hours in it.
    """
    hit = _aft_np.argwhere(probs > _AFT_BAND_B_FLOOR)
    if hit.shape[0] == 0:
        return _aft_np.empty((0, 2), dtype=_aft_np.int64), _aft_np.empty(0)
    i, j = hit[:, 0], hit[:, 1]
    vals = probs[i, j]
    # target asc, then probability desc, then i desc, then j desc - the ECB tie-break.
    order = _aft_np.lexsort((-j, -i, -vals, j))
    i, j, vals = i[order], j[order], vals[order]
    rank = _aft_np.arange(j.shape[0]) - _aft_np.searchsorted(j, j, side="left")
    keep = (rank < _AFT_BAND_B_TOPK) & (vals <= threshold)
    return _aft_np.stack([i[keep], j[keep]], axis=1), vals[keep]


def _aft_capture(
    crop, f_idx, t_src, t_tgt, idx_src, idx_tgt,
    unet_feat_src, unet_feat_tgt, p_coords_src, p_coords_tgt, ds_arr_t,
    p_pos_src, p_pos_tgt, p_mask_src, p_mask_tgt,
    c_src_rel, c_tgt_rel, window_shape, probs, threshold,
):
    """Record the exact inputs and outputs of ONE deployed predict_edges call.

    Everything here is a copy of a tensor the deployed code already built. Nothing is
    recomputed, re-normalised or re-derived, which is the whole point of the redesign.
    """
    if _AFT_CROPS and crop not in _AFT_CROPS:
        return
    if _AFT_STATE["crop"] != crop:
        _aft_reset(crop)
    if _AFT_MAX_PAIRS and len(_AFT_PAIRS) >= _AFT_MAX_PAIRS:
        return

    pair = len(_AFT_PAIRS)
    feat_dtype = str(unet_feat_src.dtype).replace("torch.", "")

    def _rows(role, feat, coords, ds_t, pos, mask, rel, gids):
        base = _AFT_STATE["role_n"]
        block = (
            _aft_np.full(feat.shape[1], pair, dtype=_aft_np.int64),
            _aft_np.full(feat.shape[1], role, dtype=_aft_np.int8),
            _aft_np.asarray(gids, dtype=_aft_np.int64),
            feat[0].detach().to("cpu", _aft_torch.float32).numpy(),
            pos[0].detach().to("cpu", _aft_torch.float32).numpy(),
            (coords * ds_t)[0].detach().to("cpu", _aft_torch.float32).numpy(),
            _aft_np.asarray(rel, dtype=_aft_np.int32),
            mask[0].detach().cpu().numpy().astype(bool),
        )
        _AFT_ROLE.append(block)
        _AFT_STATE["role_n"] = base + block[3].shape[0]
        return base, block[3].shape[0]

    s_ptr, s_n = _rows(0, unet_feat_src, p_coords_src, ds_arr_t, p_pos_src, p_mask_src,
                       c_src_rel, idx_src)
    t_ptr, t_n = _rows(1, unet_feat_tgt, p_coords_tgt, ds_arr_t, p_pos_tgt, p_mask_tgt,
                       c_tgt_rel, idx_tgt)
    _AFT_PAIRS.append((
        int(f_idx), int(t_src), int(t_tgt), s_ptr, s_n, t_ptr, t_n,
        tuple(int(v) for v in window_shape), feat_dtype,
    ))

    # BAND A: the deployed candidate rule, at the deployed site, on the deployed probabilities.
    a_hit = _aft_np.argwhere(probs > threshold)
    if a_hit.shape[0]:
        _AFT_BAND_A.append((pair, a_hit, probs[a_hit[:, 0], a_hit[:, 1]],
                            _aft_np.asarray(idx_src), _aft_np.asarray(idx_tgt)))
    # BAND B: the sub-threshold surface the learnable task actually uses.
    b_hit, b_val = _aft_select_band_b(probs, threshold)
    if b_hit.shape[0]:
        _AFT_BAND_B.append((pair, b_hit, b_val,
                            _aft_np.asarray(idx_src), _aft_np.asarray(idx_tgt)))


def _aft_flush(
    crop, coords_down, coord_offset, cfg, window_size, downsample, voxel_size,
    q_low, q_high, image_shape, pool_kernel, node_count,
):
    """Write one crop's cache. Called at predict_video's return, before the coord rescale."""
    if _AFT_CROPS and crop not in _AFT_CROPS:
        return
    if _AFT_STATE["crop"] != crop:
        # No pair was captured for this crop. Say so loudly rather than writing an empty file
        # that a downstream gate would read as "nothing to check, therefore fine".
        print(f"AFT_TAP_EMPTY crop={crop} captured_pairs=0", flush=True)
        return

    from pathlib import Path as _AftPath
    out = _AftPath(_AFT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    frames_sorted = sorted(coord_offset)

    def _cat(col, empty_shape, dtype):
        if not _AFT_ROLE:
            return _aft_np.empty(empty_shape, dtype=dtype)
        return _aft_np.concatenate([b[col] for b in _AFT_ROLE])

    role_feat = _cat(3, (0, 0), _aft_np.float32)
    role_pos = _cat(4, (0, 0), _aft_np.float32)
    n_role = int(role_feat.shape[0])
    feat_dim = int(role_feat.shape[1]) if n_role else 0
    pos_dim = int(role_pos.shape[1]) if n_role else 0
    payload = {
        "crop": _aft_np.str_(crop),
        "window": _aft_np.int64(window_size),
        "downsample": _aft_np.asarray(downsample, dtype=_aft_np.int64),
        "voxel_size": _aft_np.asarray(voxel_size, dtype=_aft_np.float64),
        "pool_kernel": _aft_np.asarray(pool_kernel, dtype=_aft_np.int64),
        "q_low": _aft_np.float64(q_low),
        "q_high": _aft_np.float64(q_high),
        "image_shape": _aft_np.asarray(image_shape, dtype=_aft_np.int64),
        "det_threshold": _aft_np.float64(cfg.det_threshold),
        "det_tta": _aft_np.bool_(cfg.det_tta),
        "edge_threshold": _aft_np.float64(cfg.threshold),
        "edge_activation": _aft_np.str_(cfg.edge_activation),
        "band_b_floor": _aft_np.float64(_AFT_BAND_B_FLOOR),
        "band_b_topk": _aft_np.int64(_AFT_BAND_B_TOPK),
        "feat_dtype": _aft_np.str_(_AFT_PAIRS[0][8] if _AFT_PAIRS else "float32"),
        "feat_dim": _aft_np.int64(feat_dim),
        "pos_dim": _aft_np.int64(pos_dim),
        "node_count": _aft_np.int64(node_count),
        # Deployed node partition and coordinates, in the DOWNSAMPLED grid the model indexed.
        "coords": _aft_np.asarray(coords_down, dtype=_aft_np.int16),
        "frames": _aft_np.asarray(frames_sorted, dtype=_aft_np.int64),
        "starts": _aft_np.asarray([coord_offset[t][0] for t in frames_sorted],
                                  dtype=_aft_np.int64),
        "ends": _aft_np.asarray([coord_offset[t][1] for t in frames_sorted],
                                dtype=_aft_np.int64),
        # Pair table.
        "pair_f_idx": _aft_np.asarray([p[0] for p in _AFT_PAIRS], dtype=_aft_np.int64),
        "pair_t_src": _aft_np.asarray([p[1] for p in _AFT_PAIRS], dtype=_aft_np.int64),
        "pair_t_tgt": _aft_np.asarray([p[2] for p in _AFT_PAIRS], dtype=_aft_np.int64),
        "pair_src_ptr": _aft_np.asarray([p[3] for p in _AFT_PAIRS], dtype=_aft_np.int64),
        "pair_src_n": _aft_np.asarray([p[4] for p in _AFT_PAIRS], dtype=_aft_np.int64),
        "pair_tgt_ptr": _aft_np.asarray([p[5] for p in _AFT_PAIRS], dtype=_aft_np.int64),
        "pair_tgt_n": _aft_np.asarray([p[6] for p in _AFT_PAIRS], dtype=_aft_np.int64),
        "pair_window_shape": _aft_np.asarray(
            [p[7] for p in _AFT_PAIRS] or _aft_np.empty((0, 4)), dtype=_aft_np.int64),
        # Role-node arrays. A node's feature belongs to (pair, role), NOT to a frame - see the
        # sixth-defect note at the head of this patch.
        "role_pair": _cat(0, (0,), _aft_np.int64),
        "role_role": _cat(1, (0,), _aft_np.int8),
        "role_gid": _cat(2, (0,), _aft_np.int64),
        "role_feat": role_feat,
        "role_coord_scaled": _cat(5, (0, 3), _aft_np.float32),
        "role_coord_rel": _cat(6, (0, 4), _aft_np.int32),
        "role_mask": _cat(7, (0,), bool),
    }
    if _AFT_STORE_POS:
        payload["role_pos"] = role_pos
    for band, blocks in (("a", _AFT_BAND_A), ("b", _AFT_BAND_B)):
        if blocks:
            pair_col = _aft_np.concatenate(
                [_aft_np.full(h.shape[0], p, dtype=_aft_np.int64) for p, h, _v, _s, _t in blocks])
            i_col = _aft_np.concatenate([h[:, 0] for _p, h, _v, _s, _t in blocks])
            j_col = _aft_np.concatenate([h[:, 1] for _p, h, _v, _s, _t in blocks])
            src_col = _aft_np.concatenate([s[h[:, 0]] for _p, h, _v, s, _t in blocks])
            tgt_col = _aft_np.concatenate([t[h[:, 1]] for _p, h, _v, _s, t in blocks])
            prob_col = _aft_np.concatenate([v for _p, _h, v, _s, _t in blocks])
        else:
            pair_col = i_col = j_col = src_col = tgt_col = _aft_np.empty(0, dtype=_aft_np.int64)
            prob_col = _aft_np.empty(0, dtype=_aft_np.float64)
        payload[f"band_{band}_pair"] = pair_col.astype(_aft_np.int64)
        payload[f"band_{band}_source_id"] = src_col.astype(_aft_np.int64)
        payload[f"band_{band}_target_id"] = tgt_col.astype(_aft_np.int64)
        payload[f"band_{band}_i"] = i_col.astype(_aft_np.int64)
        payload[f"band_{band}_j"] = j_col.astype(_aft_np.int64)
        payload[f"band_{band}_prob"] = prob_col.astype(_aft_np.float64)
    # Auditor-facing candidate surface: the union of both bands, under the names
    # scripts/win_bet/audit_feature_cache.py requires (REQUIRED_BAND_KEYS).
    payload["source_id"] = _aft_np.concatenate(
        [payload["band_a_source_id"], payload["band_b_source_id"]])
    payload["target_id"] = _aft_np.concatenate(
        [payload["band_a_target_id"], payload["band_b_target_id"]])
    payload["edge_prob"] = _aft_np.concatenate(
        [payload["band_a_prob"], payload["band_b_prob"]]).astype(_aft_np.float32)

    emitted = set(payload)
    if emitted - set(_AFT_SCHEMA_REQUIRED) - set(_AFT_SCHEMA_OPTIONAL) or (
        set(_AFT_SCHEMA_REQUIRED) - emitted
    ):
        raise RuntimeError(
            "AFT cache schema drift: emitted "
            f"{sorted(emitted - set(_AFT_SCHEMA_REQUIRED) - set(_AFT_SCHEMA_OPTIONAL))} "
            f"and omitted {sorted(set(_AFT_SCHEMA_REQUIRED) - emitted)}. The replay worker reads "
            "by name; a silent rename here becomes a KeyError after the GPU spend."
        )
    n_a = int(payload["band_a_prob"].shape[0])
    n_b = int(payload["band_b_prob"].shape[0])
    # NOTE the file HANDLE. np.savez_compressed appends '.npz' to a path that does not already
    # end in it, so passing the .tmp path would write '<crop>.npz.tmp.npz' and the rename would
    # then fail on a file that never existed - the trap edge_candidate_budget.py records.
    tmp = out / f"{crop}.npz.tmp"
    with open(tmp, "wb") as handle:
        _aft_np.savez_compressed(handle, **payload)
    tmp.replace(out / f"{crop}.npz")
    _AFT_TOTAL["crops"] += 1
    _AFT_TOTAL["pairs"] += len(_AFT_PAIRS)
    _AFT_TOTAL["band_a"] += n_a
    _AFT_TOTAL["band_b"] += n_b
    print(
        f"AFT_TAP crop={crop} pairs={len(_AFT_PAIRS)} role_nodes={n_role} "
        f"nodes={node_count} band_a={n_a} band_b={n_b} "
        f"feat_dim={feat_dim} pos_dim={pos_dim} q_low={q_low:.6g} q_high={q_high:.6g} "
        f"downsample={tuple(int(d) for d in downsample)} "
        f"crops_done={_AFT_TOTAL['crops']}",
        flush=True,
    )
    _aft_reset(None)


'''

# ---------------------------------------------------------------------------------------------
# SPAN 2: the capture call, immediately after the deployed probabilities are computed.
#
# Placed AFTER `probs` rather than after `_index_features` so that the recorded reference and the
# recorded inputs come from the same statement group - the gate then compares against the SAME
# deployed run rather than against a previous run's artifact, which FACT-0363 warns is not free.
# It sits BEFORE the `candidates` comprehension, so edge_candidate_budget.py's own anchor is
# untouched and the two patches compose in either order.
# ---------------------------------------------------------------------------------------------
_AFT_CAPTURE = '''            if _AFT_ON:
                _aft_capture(
                    ds_path.stem, f_idx, t_src, t_tgt, idx_src, idx_tgt,
                    unet_feat_src, unet_feat_tgt, p_coords_src, p_coords_tgt, ds_arr_t,
                    p_pos_src, p_pos_tgt, p_mask_src, p_mask_tgt,
                    c_src_rel, c_tgt_rel, window_shape, probs, cfg.threshold,
                )
'''

# ---------------------------------------------------------------------------------------------
# SPAN 3: the flush, at predict_video's return and BEFORE the coordinate rescale, so the cache
# carries the DOWNSAMPLED coordinates the model actually indexed rather than the original-
# resolution ones the caller receives.
# ---------------------------------------------------------------------------------------------
_AFT_FLUSH = '''    if _AFT_ON:
        _aft_flush(
            ds_path.stem, coords, coord_offset, cfg, W, downsample, voxel_size,
            q_low, q_high, image_shape, pool_k, global_node_count,
        )
'''

_AFT_ANCHOR_CFG = "@torch.no_grad()\ndef predict_video("
_AFT_ANCHOR_PROBS = '''            raw = edge_logits_pair[0]
            if cfg.edge_activation == "softmax":
                probs = torch.softmax(raw, dim=0).cpu().numpy()
            else:
                probs = torch.sigmoid(raw).cpu().numpy()
'''
_AFT_ANCHOR_COORDS = (
    "    coords = np.concatenate(coord_lists) if coord_lists else "
    "np.empty((0, 4), dtype=np.int16)\n"
)

_aft_src0 = _ps.read_text()
_AFT_PRE_SHA = _aft_patch_hashlib.sha256(_aft_src0.encode("utf-8")).hexdigest()
_aft_src = _aft_src0

for _aft_name, _aft_anchor, _aft_body, _aft_before in (
    # NOTE the decorator is part of the anchor. predict_video is wrapped in @torch.no_grad(), so
    # inserting between the decorator and the `def` would apply that decorator to the config
    # block's first statement and raise at import.
    ("config", _AFT_ANCHOR_CFG, _AFT_CONFIG, True),
    ("capture", _AFT_ANCHOR_PROBS, _AFT_CAPTURE, False),
    ("flush", _AFT_ANCHOR_COORDS, _AFT_FLUSH, False),
):
    _aft_count = _aft_src.count(_aft_anchor)
    if _aft_count != 1:
        raise RuntimeError(
            f"assoc_feature_tap: anchor '{_aft_name}' matched {_aft_count} times (expected 1). "
            "Refusing to patch - a silent miss here would look exactly like a passive tap that "
            "recorded nothing."
        )
    _aft_span = aft_wrap(_aft_name, _aft_body)
    _aft_src = _aft_src.replace(
        _aft_anchor,
        (_aft_span + _aft_anchor) if _aft_before else (_aft_anchor + _aft_span),
        1,
    )

# PURITY PROOF, EXECUTED HERE RATHER THAN ASSERTED. Removing the marked spans must restore the
# pre-patch bytes exactly. If it does not, the patch modified deployed code somewhere and the run
# is no longer a valid control for its own graph.
if aft_strip_inserts(_aft_src) != _aft_src0:
    raise RuntimeError(
        "assoc_feature_tap: stripping the inserted spans did NOT restore the original source. "
        "The patch is not pure insertion and cannot claim the graph is unchanged."
    )
compile(_aft_src, str(_ps), "exec")
_ps.write_text(_aft_src)

_AFT_POST_SHA = _aft_patch_hashlib.sha256(_aft_src.encode("utf-8")).hexdigest()
_AFT_PATCH_RECEIPT = {
    "patch": "assoc_feature_tap",
    "predictor": str(_ps),
    "pre_sha256": _AFT_PRE_SHA,
    "post_sha256": _AFT_POST_SHA,
    "spans": ["config", "capture", "flush"],
    "aft_dir": _aft_patch_os.environ.get("BIOHUB_AFT_DIR", "") or None,
    "band_b_floor": _aft_patch_os.environ.get("BIOHUB_AFT_BAND_B_FLOOR", "") or "0.02",
    "band_b_topk": _aft_patch_os.environ.get("BIOHUB_AFT_BAND_B_TOPK", "") or "8",
    "max_pairs": _aft_patch_os.environ.get("BIOHUB_AFT_MAX_PAIRS", "") or "0 (uncapped)",
    "crops": _aft_patch_os.environ.get("BIOHUB_AFT_CROPS", "") or "all",
}
try:
    from pathlib import Path as _AftPatchPath
    _AftPatchPath("/kaggle/working/aft_patch_receipt.json").write_text(
        _aft_patch_json.dumps(_AFT_PATCH_RECEIPT, indent=2), encoding="utf-8"
    )
except OSError:
    pass
print(
    "AFT_PATCH_APPLIED pure_insertion=True "
    f"pre_sha256={_AFT_PRE_SHA[:12]} post_sha256={_AFT_POST_SHA[:12]} "
    f"mode={'record' if _AFT_PATCH_RECEIPT['aft_dir'] else 'off (control arm)'} "
    f"dir={_AFT_PATCH_RECEIPT['aft_dir']} "
    f"band_b_floor={_AFT_PATCH_RECEIPT['band_b_floor']} "
    f"band_b_topk={_AFT_PATCH_RECEIPT['band_b_topk']} "
    f"max_pairs={_AFT_PATCH_RECEIPT['max_pairs']} crops={_AFT_PATCH_RECEIPT['crops']}",
    flush=True,
)
