# =====================================================================================
# PASSIVE FEATURE TAP - CONTRACT VERSION 2  (PKT-0029 Gate 1)
#
# CONTRACT v1 RAN AND MEASURED (P36 / EXP-0043, FACT-0403). Nothing about the cache was wrong:
# positional features round-tripped at EXACTLY 0.0, integrity failures were ZERO, both bands were
# non-vacuous on both crops. The VERDICT was invalid because the reference quantity was the wrong
# object - v1 recorded one field called `band_a_prob` / `band_b_prob` / `edge_prob`, took it from
# the deployed `probs`, and compared it against a replay of the PRIMARY HEAD ALONE. Those are two
# different quantities whenever any fusion stage runs, and they SHOULD differ.
#
# WHAT CONTRACT v2 CHANGES, AND WHY EACH CHANGE IS FORCED BY MEASUREMENT.
#
# 1. THREE EXPLICITLY NAMED QUANTITIES. No field named merely "prob" survives. Every probability
#    and every logit carries the surface it came from in its own name:
#      primary_logits_preblend    `model.predict_edges(...)` forward output, BEFORE ANY fusion
#                                 stage - predict_unet_transformer.py:595-600.
#      primary_probs_preblend     the source-axis activation of that, and nothing else.
#      deployed_probs_postblend   `probs` at predict_unet_transformer.py:758-762, AFTER every
#                                 fusion stage that actually ran.
#    An unqualified name is what made a two-stage quantity readable as a one-stage one.
#
# 2. FUSION PROVENANCE IS RECORDED BY THE TAP, NOT INFERRED FROM A LOG HEADER. This is the direct
#    fix for how FACT-0403 misattributed its own root cause. That fact blamed a SECONDARY-model
#    logit blend at weight 0.15, reading the setup cell's "Secondary edge-logit weight: 0.15"
#    header. But `scripts/kaggle_edits/loeo_retarget.py:128` sets BIOHUB_SECONDARY_WEIGHTS = ""
#    afterwards, the same log prints `"secondary_enabled": false`, and
#    predict_unet_transformer.py:862 gates the whole secondary block on that variable being
#    non-empty. The secondary was OFF. The stage that actually ran was the BIDIRECTIONAL HARMONIC
#    at weight 0.20 (predict_unet_transformer.py:602-659), applied by the notebook.
#    RE-DERIVED ON CPU FROM P36'S OWN ARCHIVED CACHE: primary-only replay reproduces the recorded
#    GPU gap to seven digits (0.22877353 vs 0.22877401; 0.42863509 vs 0.42863536) with band-A
#    missing/extra 12/0 and 109/39 - the recorded values exactly - and applying the bidirectional
#    harmonic at 0.20 collapses it to 6.6e-07 and 1.1e-06 with membership 0/0 on both crops.
#    A tap that records which stages ran cannot be misread this way again.
#
# 3. A THIRD BAND, SELECTED ON THE PARITY SURFACE. Bands A and B keep their v1 definitions and
#    both are recorded as SELECTED ON `deployed_probs_postblend` - because that is what selected
#    them, and re-deriving their membership on the pre-blend surface would silently change what
#    the artifact means. Their membership is therefore no longer a gate condition. Band P is the
#    pre-blend analogue, selected on `primary_probs_preblend`, and it carries the membership
#    property the gate would otherwise have lost.
#
# 4. THE SOFTMAX BLIND SPOT IS CLOSED. v1 stated it could not be closed because "nothing in the
#    current artifacts records pre-softmax logits". v2 records them per band row, so a cache error
#    whose net effect is a per-target constant logit offset - mathematically invisible after a
#    source-axis softmax - is now visible in the logit column.
#
# EVERYTHING THAT WAS ALREADY PROVEN IS UNCHANGED. The tap still reimplements no preprocessing:
# it copies tensors the deployed code already built, at the deployed site. Passivity is still
# proven two ways (pure insertion restoring the pre-patch bytes; and a real CPU `predict_video`
# run, pristine versus tapped, with identical coords and edges). The pair-and-role keying that
# FACT-0402 forced is unchanged: `TemporalUNet3D._TemporalAttention` mixes across the window's
# time axis, so a node's feature belongs to (pair, role) and never to a frame alone.
#
# INERT UNLESS CONFIGURED. With BIOHUB_AFT_DIR unset the tap records nothing and writes nothing.
# FAIL CLOSED. Every anchor must match exactly once or this raises.
# =====================================================================================
import hashlib as _aft_patch_hashlib
import json as _aft_patch_json
import os as _aft_patch_os
import re as _aft_patch_re

# The contract this patch implements. It is written into the cache, into the patch receipt and
# into the heartbeat line, so any artifact can be attributed to a contract version after the
# fact. P36 / EXP-0043 ran under contract 1, whose receipt carries no such field - so a rebuild
# of the p36 spec against this file produces a DIFFERENT experiment and must not be reported
# under P36's id.
AFT_CONTRACT_VERSION = 2

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
# PASSIVE FEATURE TAP, CONTRACT 2 (PKT-0029 Gate 1). Records what the deployed head was fed, what
# the PRIMARY head produced before any fusion, and what the deployed pipeline produced after it.
# Inert unless BIOHUB_AFT_DIR is set. Never writes to coords, all_edges, probs or candidates.
import os as _aft_os
import numpy as _aft_np
import torch as _aft_torch

_AFT_CONTRACT_VERSION = 2

# THE THREE NAMED SURFACES. Written into the cache so a reader never has to guess which one a
# column holds - the guess is what invalidated the v1 verdict (FACT-0403).
_AFT_SURFACE_PRIMARY = (
    "primary: model.predict_edges(unet_feat_src, unet_feat_tgt, ...) forward output at "
    "predict_unet_transformer.py:595-600, BEFORE the bidirectional harmonic (:602-659) and "
    "BEFORE the secondary logit blend (:660-756)"
)
_AFT_SURFACE_DEPLOYED = (
    "deployed: probs at predict_unet_transformer.py:758-762, AFTER every fusion stage that ran; "
    "this is the surface the deployed candidate rule reads"
)


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

# THE THREE BANDS AND THE SURFACE THAT SELECTS EACH. Declared as data so the flush writes the
# provenance instead of a comment claiming it.
#   a  the DEPLOYED candidate rule, on the DEPLOYED post-fusion surface
#   b  the ECB sidecar acquisition rule, on the DEPLOYED post-fusion surface
#   p  the same threshold rule on the PRIMARY pre-fusion surface - the parity surface, and the
#      only band whose membership the replay may re-derive without changing what it means
_AFT_BAND_SURFACE = {
    "a": "deployed_probs_postblend",
    "b": "deployed_probs_postblend",
    "p": "primary_probs_preblend",
}

# THE CACHE SCHEMA, DECLARED RATHER THAN IMPLIED. The flush asserts it emits exactly this, and
# the preflight requires every key the replay worker reads to be one of these - so a rename on
# either side is caught on CPU instead of at the end of a GPU session.
_AFT_SCHEMA_REQUIRED = (
    "schema_version", "contract_version", "primary_surface", "deployed_surface",
    "crop", "window", "downsample", "voxel_size", "pool_kernel", "q_low", "q_high",
    "image_shape", "det_threshold", "det_tta", "edge_threshold", "edge_activation",
    "band_b_floor", "band_b_topk", "feat_dtype", "feat_dim", "pos_dim", "node_count",
    "fusion_stages", "fusion_bidirectional_weight", "fusion_secondary_enabled",
    "fusion_secondary_edge_weight", "fusion_secondary_link_mode",
    "fusion_secondary_mix_temperature", "fusion_reproducible_from_primary_cache",
    "coords", "frames", "starts", "ends",
    "pair_f_idx", "pair_t_src", "pair_t_tgt", "pair_src_ptr", "pair_src_n",
    "pair_tgt_ptr", "pair_tgt_n", "pair_window_shape",
    "role_pair", "role_role", "role_gid", "role_feat", "role_coord_scaled",
    "role_coord_rel", "role_mask",
    "band_a_selected_on", "band_a_pair", "band_a_source_id", "band_a_target_id",
    "band_a_i", "band_a_j", "band_a_primary_logit_preblend",
    "band_a_primary_prob_preblend", "band_a_deployed_prob_postblend",
    "band_b_selected_on", "band_b_pair", "band_b_source_id", "band_b_target_id",
    "band_b_i", "band_b_j", "band_b_primary_logit_preblend",
    "band_b_primary_prob_preblend", "band_b_deployed_prob_postblend",
    "band_p_selected_on", "band_p_pair", "band_p_source_id", "band_p_target_id",
    "band_p_i", "band_p_j", "band_p_primary_logit_preblend",
    "band_p_primary_prob_preblend", "band_p_deployed_prob_postblend",
    "source_id", "target_id",
    "deployed_edge_prob_postblend", "primary_edge_prob_preblend",
)
_AFT_SCHEMA_OPTIONAL = ("role_pos",)

# Buffers are keyed by crop and reset when the crop changes, so a crop that raises part way
# through cannot leak its rows into the next one.
_AFT_STATE = {"crop": None}
_AFT_PAIRS = []
_AFT_ROLE = []
_AFT_BANDS = {"a": [], "b": [], "p": []}
_AFT_FUSION = {}
_AFT_TOTAL = {"crops": 0, "pairs": 0, "band_a": 0, "band_b": 0, "band_p": 0}


def _aft_reset(crop):
    _AFT_STATE["crop"] = crop
    _AFT_STATE["role_n"] = 0
    del _AFT_PAIRS[:], _AFT_ROLE[:]
    for _rows in _AFT_BANDS.values():
        del _rows[:]
    _AFT_FUSION.clear()


def _aft_activate(logits, activation):
    """The DEPLOYED activation, applied to a (n_src, n_tgt) logit block.

    Copied from predict_unet_transformer.py:758-762 rather than paraphrased, because the
    pre-blend probability must be the same FUNCTION of its logits that the deployed line is.
    """
    if activation == "softmax":
        return _aft_torch.softmax(logits, dim=0)
    return _aft_torch.sigmoid(logits)


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
        return _aft_np.empty((0, 2), dtype=_aft_np.int64)
    i, j = hit[:, 0], hit[:, 1]
    vals = probs[i, j]
    # target asc, then probability desc, then i desc, then j desc - the ECB tie-break.
    order = _aft_np.lexsort((-j, -i, -vals, j))
    i, j, vals = i[order], j[order], vals[order]
    rank = _aft_np.arange(j.shape[0]) - _aft_np.searchsorted(j, j, side="left")
    keep = (rank < _AFT_BAND_B_TOPK) & (vals <= threshold)
    return _aft_np.stack([i[keep], j[keep]], axis=1)


def _aft_record_fusion(bidirectional_weight, secondary_enabled, secondary_edge_weight,
                       secondary_link_mode, secondary_mix_temperature):
    """Record WHICH fusion stages ran, from the predictor's own locals.

    FACT-0403 attributed the v1 gap to a secondary blend because a setup-cell log header said
    the secondary weight was 0.15. It had been disabled afterwards
    (scripts/kaggle_edits/loeo_retarget.py:128) and the stage that actually ran was the
    bidirectional harmonic. A header is not provenance; this is.
    """
    stages = []
    if bidirectional_weight > 0.0:
        stages.append("bidirectional_harmonic")
    if secondary_enabled:
        stages.append("secondary_logit_blend")
    seen = {
        "stages": ",".join(stages) if stages else "none",
        "bidirectional_weight": float(bidirectional_weight),
        "secondary_enabled": bool(secondary_enabled),
        "secondary_edge_weight": float(secondary_edge_weight),
        "secondary_link_mode": str(secondary_link_mode),
        "secondary_mix_temperature": float(secondary_mix_temperature),
        # The bidirectional stage re-runs the SAME primary model on the SAME cached tensors with
        # the roles swapped, so a replay can reproduce it exactly from this cache. The secondary
        # stage needs a second model's features, which this cache does not and must not carry.
        "reproducible_from_primary_cache": not bool(secondary_enabled),
    }
    if not _AFT_FUSION:
        _AFT_FUSION.update(seen)
    elif _AFT_FUSION != seen:
        raise RuntimeError(
            "AFT: the fusion configuration changed part way through a crop "
            f"({_AFT_FUSION} -> {seen}). Every pair in one cache file must have been produced "
            "under one deployment configuration or the recorded surfaces are not comparable."
        )


def _aft_capture(
    crop, f_idx, t_src, t_tgt, idx_src, idx_tgt,
    unet_feat_src, unet_feat_tgt, p_coords_src, p_coords_tgt, ds_arr_t,
    p_pos_src, p_pos_tgt, p_mask_src, p_mask_tgt,
    c_src_rel, c_tgt_rel, window_shape,
    primary_logits_preblend, deployed_probs_postblend, threshold, activation,
    bidirectional_weight, secondary_enabled, secondary_edge_weight,
    secondary_link_mode, secondary_mix_temperature,
):
    """Record the exact inputs and BOTH output surfaces of ONE deployed predict_edges call.

    Everything here is a copy of a tensor the deployed code already built. Nothing is
    recomputed, re-normalised or re-derived, which is the whole point of the redesign. The one
    derived quantity is `primary_probs_preblend`, and it is derived by the DEPLOYED activation
    applied to the DEPLOYED pre-fusion logits - not by any reimplementation of the head.
    """
    if _AFT_CROPS and crop not in _AFT_CROPS:
        return
    if _AFT_STATE["crop"] != crop:
        _aft_reset(crop)
    if _AFT_MAX_PAIRS and len(_AFT_PAIRS) >= _AFT_MAX_PAIRS:
        return
    _aft_record_fusion(bidirectional_weight, secondary_enabled, secondary_edge_weight,
                       secondary_link_mode, secondary_mix_temperature)

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

    # THE THREE NAMED SURFACES, materialised once per pair.
    logit_pre = primary_logits_preblend[0].detach().to("cpu", _aft_torch.float64).numpy()
    prob_pre = _aft_activate(
        primary_logits_preblend[0].detach().to(_aft_torch.float32), activation,
    ).to("cpu", _aft_torch.float64).numpy()
    prob_post = _aft_np.asarray(deployed_probs_postblend, dtype=_aft_np.float64)

    # BAND A: the deployed candidate rule, on the DEPLOYED post-fusion surface.
    # BAND B: the ECB sub-threshold surface, also on the DEPLOYED post-fusion surface.
    # BAND P: the same threshold rule on the PRIMARY pre-fusion surface.
    hits = {
        "a": _aft_np.argwhere(prob_post > threshold),
        "b": _aft_select_band_b(prob_post, threshold),
        "p": _aft_np.argwhere(prob_pre > threshold),
    }
    src_ids = _aft_np.asarray(idx_src)
    tgt_ids = _aft_np.asarray(idx_tgt)
    for name, hit in hits.items():
        if hit.shape[0]:
            i, j = hit[:, 0], hit[:, 1]
            _AFT_BANDS[name].append((
                pair, i, j, src_ids[i], tgt_ids[j],
                logit_pre[i, j], prob_pre[i, j], prob_post[i, j],
            ))


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
        "schema_version": _aft_np.int64(2),
        "contract_version": _aft_np.int64(_AFT_CONTRACT_VERSION),
        "primary_surface": _aft_np.str_(_AFT_SURFACE_PRIMARY),
        "deployed_surface": _aft_np.str_(_AFT_SURFACE_DEPLOYED),
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
        # WHICH FUSION STAGES RAN, from the predictor's own state at the capture site.
        "fusion_stages": _aft_np.str_(_AFT_FUSION["stages"]),
        "fusion_bidirectional_weight": _aft_np.float64(_AFT_FUSION["bidirectional_weight"]),
        "fusion_secondary_enabled": _aft_np.bool_(_AFT_FUSION["secondary_enabled"]),
        "fusion_secondary_edge_weight": _aft_np.float64(_AFT_FUSION["secondary_edge_weight"]),
        "fusion_secondary_link_mode": _aft_np.str_(_AFT_FUSION["secondary_link_mode"]),
        "fusion_secondary_mix_temperature": _aft_np.float64(
            _AFT_FUSION["secondary_mix_temperature"]),
        "fusion_reproducible_from_primary_cache": _aft_np.bool_(
            _AFT_FUSION["reproducible_from_primary_cache"]),
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
        # FACT-0402 note at the head of this patch.
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
    counts = {}
    for band, blocks in _AFT_BANDS.items():
        cols = {}
        if blocks:
            cols["pair"] = _aft_np.concatenate(
                [_aft_np.full(b[1].shape[0], b[0], dtype=_aft_np.int64) for b in blocks])
            for k, idx in (("i", 1), ("j", 2), ("source_id", 3), ("target_id", 4)):
                cols[k] = _aft_np.concatenate([b[idx] for b in blocks]).astype(_aft_np.int64)
            for k, idx in (("primary_logit_preblend", 5), ("primary_prob_preblend", 6),
                           ("deployed_prob_postblend", 7)):
                cols[k] = _aft_np.concatenate([b[idx] for b in blocks]).astype(_aft_np.float64)
        else:
            for k in ("pair", "i", "j", "source_id", "target_id"):
                cols[k] = _aft_np.empty(0, dtype=_aft_np.int64)
            for k in ("primary_logit_preblend", "primary_prob_preblend",
                      "deployed_prob_postblend"):
                cols[k] = _aft_np.empty(0, dtype=_aft_np.float64)
        payload[f"band_{band}_selected_on"] = _aft_np.str_(_AFT_BAND_SURFACE[band])
        for k, v in cols.items():
            payload[f"band_{band}_{k}"] = v
        counts[band] = int(cols["pair"].shape[0])
    # Auditor-facing candidate surface: the union of bands A and B - the DEPLOYED candidate
    # surface, unchanged in membership from contract 1. Band P is deliberately NOT in the union:
    # it is a gate instrument, not a deployed candidate set. Both probability columns are named
    # for their surface; there is no unqualified `edge_prob` any more.
    payload["source_id"] = _aft_np.concatenate(
        [payload["band_a_source_id"], payload["band_b_source_id"]])
    payload["target_id"] = _aft_np.concatenate(
        [payload["band_a_target_id"], payload["band_b_target_id"]])
    payload["deployed_edge_prob_postblend"] = _aft_np.concatenate(
        [payload["band_a_deployed_prob_postblend"],
         payload["band_b_deployed_prob_postblend"]]).astype(_aft_np.float32)
    payload["primary_edge_prob_preblend"] = _aft_np.concatenate(
        [payload["band_a_primary_prob_preblend"],
         payload["band_b_primary_prob_preblend"]]).astype(_aft_np.float32)

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
    # NO UNQUALIFIED PROBABILITY MAY SURVIVE. This is the v1 defect expressed as a rule the tap
    # enforces on itself: any column holding a probability or a logit must name its surface.
    _aft_unqualified = sorted(
        k for k in emitted
        if ("prob" in k or "logit" in k)
        and not (k.endswith("_preblend") or k.endswith("_postblend"))
    )
    if _aft_unqualified:
        raise RuntimeError(
            f"AFT cache emits unqualified probability/logit columns {_aft_unqualified}. "
            "Contract 2 requires every such column to end in _preblend or _postblend - an "
            "unqualified name is what let a post-fusion quantity be read as a pre-fusion one "
            "(FACT-0403)."
        )
    # NOTE the file HANDLE. np.savez_compressed appends '.npz' to a path that does not already
    # end in it, so passing the .tmp path would write '<crop>.npz.tmp.npz' and the rename would
    # then fail on a file that never existed - the trap edge_candidate_budget.py records.
    tmp = out / f"{crop}.npz.tmp"
    with open(tmp, "wb") as handle:
        _aft_np.savez_compressed(handle, **payload)
    tmp.replace(out / f"{crop}.npz")
    _AFT_TOTAL["crops"] += 1
    _AFT_TOTAL["pairs"] += len(_AFT_PAIRS)
    for band in ("a", "b", "p"):
        _AFT_TOTAL[f"band_{band}"] += counts[band]
    print(
        f"AFT_TAP contract={_AFT_CONTRACT_VERSION} crop={crop} pairs={len(_AFT_PAIRS)} "
        f"role_nodes={n_role} nodes={node_count} "
        f"band_a={counts['a']} band_b={counts['b']} band_p={counts['p']} "
        f"feat_dim={feat_dim} pos_dim={pos_dim} q_low={q_low:.6g} q_high={q_high:.6g} "
        f"downsample={tuple(int(d) for d in downsample)} "
        f"fusion={_AFT_FUSION['stages']} "
        f"bidirectional_weight={_AFT_FUSION['bidirectional_weight']:.4g} "
        f"secondary_enabled={int(_AFT_FUSION['secondary_enabled'])} "
        f"secondary_edge_weight={_AFT_FUSION['secondary_edge_weight']:.4g} "
        f"reproducible_from_primary_cache="
        f"{int(_AFT_FUSION['reproducible_from_primary_cache'])} "
        f"crops_done={_AFT_TOTAL['crops']}",
        flush=True,
    )
    _aft_reset(None)


'''

# ---------------------------------------------------------------------------------------------
# SPAN 2: the PRE-FUSION primary logits, taken immediately after `model.predict_edges` returns
# and BEFORE the bidirectional harmonic (:602-659) or the secondary blend (:660-756) can rebind
# `edge_logits_pair`. This is the span contract 1 did not have, and its absence is the whole
# reason the v1 verdict was invalid: by the time v1 looked, `edge_logits_pair` had already been
# replaced by a fused quantity.
#
# Both fusion blocks REBIND `edge_logits_pair` rather than mutating it in place, so holding a
# reference would in fact be sound - but a clone costs one (n_src, n_tgt) tensor that is released
# on the next iteration, and it removes the need to re-verify that property every time the
# deployed source changes.
# ---------------------------------------------------------------------------------------------
_AFT_PRIMARY = '''            if _AFT_ON:
                _aft_primary_logits_preblend = edge_logits_pair.detach().clone()
'''

# ---------------------------------------------------------------------------------------------
# SPAN 3: the capture call, immediately after the DEPLOYED probabilities are computed.
#
# Placed AFTER `probs` so that both recorded surfaces and the recorded inputs come from the same
# statement group - the gate then compares against the SAME deployed run rather than a previous
# run's artifact, which FACT-0363 warns is not free. It sits BEFORE the `candidates`
# comprehension, so edge_candidate_budget.py's own anchor is untouched.
#
# THE FUSION PROVENANCE IS READ THROUGH `locals()` ON PURPOSE. `secondary_model`,
# `secondary_edge_weight`, `secondary_link_mode`, `secondary_mix_temperature` are parameters of
# `predict_video` in the deployed support pack, and `_bidirectional_weight` is a local the
# notebook's bidirectional patch introduces - but NONE of them exist in the organizer's public
# `vendor/kaggle-cell-tracking` copy, which has no fusion at all. A direct reference would raise
# NameError there and the tap would stop being applicable to both. Defaults are the "stage did
# not run" values, so a predictor without a stage records that stage as absent rather than
# guessing.
# ---------------------------------------------------------------------------------------------
_AFT_CAPTURE = '''            if _AFT_ON:
                _aft_locals = locals()
                _aft_capture(
                    ds_path.stem, f_idx, t_src, t_tgt, idx_src, idx_tgt,
                    unet_feat_src, unet_feat_tgt, p_coords_src, p_coords_tgt, ds_arr_t,
                    p_pos_src, p_pos_tgt, p_mask_src, p_mask_tgt,
                    c_src_rel, c_tgt_rel, window_shape,
                    _aft_primary_logits_preblend, probs, cfg.threshold, cfg.edge_activation,
                    float(_aft_locals.get("_bidirectional_weight", 0.0) or 0.0),
                    _aft_locals.get("secondary_model", None) is not None,
                    float(_aft_locals.get("secondary_edge_weight", 0.0) or 0.0),
                    str(_aft_locals.get("secondary_link_mode", "") or ""),
                    float(_aft_locals.get("secondary_mix_temperature", 1.0) or 1.0),
                )
'''

# ---------------------------------------------------------------------------------------------
# SPAN 4: the flush, at predict_video's return and BEFORE the coordinate rescale, so the cache
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
_AFT_ANCHOR_PRIMARY = '''            edge_logits_pair = model.predict_edges(
                unet_feat_src, unet_feat_tgt,
                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
                p_pos_src, p_pos_tgt,
                p_mask_src, p_mask_tgt,
            )  # (1, n_src, n_tgt)
'''
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
    ("primary", _AFT_ANCHOR_PRIMARY, _AFT_PRIMARY, False),
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
    "contract_version": AFT_CONTRACT_VERSION,
    "predictor": str(_ps),
    "pre_sha256": _AFT_PRE_SHA,
    "post_sha256": _AFT_POST_SHA,
    "spans": ["config", "primary", "capture", "flush"],
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
    f"AFT_PATCH_APPLIED contract={AFT_CONTRACT_VERSION} pure_insertion=True "
    f"pre_sha256={_AFT_PRE_SHA[:12]} post_sha256={_AFT_POST_SHA[:12]} "
    f"mode={'record' if _AFT_PATCH_RECEIPT['aft_dir'] else 'off (control arm)'} "
    f"dir={_AFT_PATCH_RECEIPT['aft_dir']} "
    f"band_b_floor={_AFT_PATCH_RECEIPT['band_b_floor']} "
    f"band_b_topk={_AFT_PATCH_RECEIPT['band_b_topk']} "
    f"max_pairs={_AFT_PATCH_RECEIPT['max_pairs']} crops={_AFT_PATCH_RECEIPT['crops']}",
    flush=True,
)
