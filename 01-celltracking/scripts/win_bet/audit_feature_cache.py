r"""BIND a tapped association feature cache to everything that could make a null uninterpretable,
INDEPENDENTLY re-check that binding afterwards, and REFUSE a full-fold launch that will not fit.

WHY THIS EXISTS (PKT-0037)
--------------------------
The expensive failure on this lane is not a wrong number. It is a NULL RESULT nobody can read.
If a head trained on a cached feature surface scores at chance, there are two explanations and
the campaign cannot currently tell them apart:

    (1) the head does not work; or
    (2) we fed it the wrong features.

`FACT-0392` names the sharpest ingredient of (2): WHICH TRUNK produced the 32-dim features is a
``--weights`` command-line argument in the publisher's own cache script and is recorded NOWHERE
in the checkpoint. Re-measured on CPU for this revision: every candidate trunk under
``C:/temp/hoct/trunks/`` is a BARE state dict of 136 tensors with ZERO non-tensor keys, and
``official_f0`` and ``stabledet_f0`` have the SAME byte size (8,357,783) with DIFFERENT sha256.
So neither the checkpoint's contents nor its size can identify it. **Provenance can only come
from the manifest**, and this auditor refuses a manifest that does not carry it.

WHY THE SCHEMA CHANGED - SCHEMA 1 WAS SCIENTIFICALLY UNDEFINED (`FACT-0402`)
---------------------------------------------------------------------------
Schema 1 required ``feat_frames`` and per-frame ``feat_{t}`` arrays: ONE feature vector per node.
That layout cannot exist. ``TemporalUNet3D`` contains ``_TemporalAttention``, which mixes across
the window's time axis, so ``unet_out[:, i]`` depends on EVERY frame in the window. At the
deployed ``window_size=2`` with stride ``W-1=1`` an interior frame ``t`` is the TARGET of the pair
``(t-1, t)`` and the SOURCE of the pair ``(t, t+1)`` - two forward passes, two different vectors
for one node. Measured on the real deployed path: 66 of 173 role-nodes carry two vectors, max abs
delta 0.0123, max relative 3.5%, which is 2.1 orders of magnitude above the gate's own 1e-4
tolerance and NOT uniform across the source axis.

A node's feature therefore belongs to a ``(pair, role)``, never to a frame. This auditor reads the
PAIR-AND-ROLE layout that ``scripts/kaggle_edits/assoc_feature_tap.py`` writes and
``scripts/win_bet/assoc_tap_replay.py`` reads, and it treats any surviving frame-keyed artifact as
a REJECT that names defect 6 by its own condition.

THE THREE DEFECT-6 CONDITIONS, WHICH ARE NOT THE SAME CHECK
-----------------------------------------------------------
``frame_keyed_cache``            the old layout itself - ``feat_frames`` or any ``feat_{t}``.
``role_records_missing``         the pair-and-role SHAPE, but a node that the pair table says
                                 must appear twice appears once. That is the old worker's
                                 ``feats[t]`` de-duplication wearing the new schema's clothes.
``role_features_frame_keyed``    the shape is right, the multiplicity is right, and yet EVERY
                                 dual-role node carries a BIT-IDENTICAL vector in both roles.
                                 A temporal-attention trunk cannot produce that; a frame-keyed
                                 writer producing two copies of one vector is exactly what does.

The third is the one a shape check cannot see, and it is the form the defect would take if the
tap were ever "repaired" by someone who had only read the schema.

CONTRACT 2 - AND WHY AN UNQUALIFIED PROBABILITY IS NOW A REJECT (`FACT-0403`, `FACT-0407`)
------------------------------------------------------------------------------------------
Contract 1 recorded ONE probability per band row, named `band_a_prob` / `band_b_prob` with an
`edge_prob` union, and took it from the deployed ``probs`` - AFTER every fusion stage. P36's gate
then compared a PRIMARY-ONLY replay against it and disagreed by up to 0.43. The cache was
faultless: re-derived on CPU from P36's own archived bytes, the primary-only arm reproduces the
recorded GPU gap to seven digits, and applying the deployed bidirectional harmonic at weight 0.20
collapses it to 1.1e-06 with band-A membership going to zero missing and zero extra on both crops
(`FACT-0407`). What failed was a NAME: a two-stage quantity that read as a one-stage one.

So this auditor now polices THREE NAMED SURFACES per band row - ``primary_logit_preblend``,
``primary_prob_preblend``, ``deployed_prob_postblend`` - a THIRD BAND (``p``) selected on the
pre-fusion surface, and the FUSION PROVENANCE the tap reads from the predictor's own locals. Four
new conditions follow directly from that record, and each one is a mechanism, not a preference:

``cache_contract_1``                     the retired naming itself, refused by name rather than
                                         as thirty missing keys.
``band_{x}_selected_on_is_wrong``        a band that mislabels which surface chose its rows. The
                                         mislabel IS the contract-1 defect, wearing a v2 header.
``fusion_record_is_internally_wrong``    stages, weights and the reproducibility claim must agree
                                         with each other. A cache claiming it can reproduce its
                                         own deployed surface while a SECOND MODEL contributed to
                                         it is claiming something arithmetically false.
``no_fusion_but_*``                      when the record says NO stage ran, pre- and post-fusion
                                         are the same tensor: the two columns must be equal and
                                         band P's membership must be band A's. Measured on the
                                         real tap: with no fusion the delta is EXACTLY 0.0 and
                                         the two key sets are identical; with a secondary at
                                         0.15 they part by 0.31 and 19 rows become 8.

WHAT THE MANIFEST BINDS
-----------------------
==========================  ==========================================================
trunk                       checkpoint sha256, byte size, role, EXPLICIT provenance, its
                            state-dict shape, and the digest of the FEATURES IT PRODUCED
fold and embryo             fold id, held-out embryo, every crop stem, cross-checked
                            against the fold<->embryo map (fold 0 = 44b6, fold 1 = 6bba)
window contract             the window size and the deployed pair stride
feature normalisation       the transform applied after ``_index_features`` and its stats
role table                  pair/role/gid ordering digests, so a reorder of coordinates
                            against features, or of roles against pairs, is detectable
candidate graph             rule, floor, cap, gate, det threshold, softmax axis, and the
                            per-band pair counts for BOTH bands
storage                     MEASURED bytes on disk, uncompressed bytes, compression ratio
notebook and commit         notebook path + sha256, spec, git commit, kernel + version
==========================  ==========================================================

FAIL CLOSED, EVERYWHERE
-----------------------
A missing manifest, an unreadable cache, a crop present in one and absent from the other, an
unknown schema version, an empty band, a torn role block, a checkpoint whose bytes no longer hash
to the recorded value, a projection that exceeds the stated capacity - every one is a REJECT,
never a warning and never a skip. `FACT-0387` is the standing lesson: the gate that compared zero
crops is the gate that would have licensed two wasted GPU sessions had it defaulted to a pass.

AND IT IS PROVEN BY MUTATION, NOT BY ASSERTION
----------------------------------------------
``self-test`` manufactures each defect on a production-layout cache and requires the auditor to
REJECT it, WITH ACCEPT CONTROLS - a clean cache, a valid dual-trunk pair, and a projection inside
budget - because an auditor that rejects everything checks nothing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 2

# The deployed window contract, read at source rather than assumed:
#   window_size 2                       secondary_seed_weights/unet_transformer/split_0/config.json
#   stride = max(W - 1, 1)              predict_unet_transformer.py:353
#   t_src, t_tgt = frame_indices[f_idx], frame_indices[f_idx + 1]      :557
#   c_src_rel[:, 0] = f_idx ; c_tgt_rel[:, 0] = f_idx + 1              :580-583
#   window_shape = (W,) + image_shape[1:]                              :579
#   p_mask_* = torch.ones(...)  - the deployed site never pads         :586-587
DEPLOYED_WINDOW = 2

# fold <-> held-out embryo. AGENTS.md and the upstream audit item U10 both fix this direction:
# fold 0 is the 71-crop 44b6 direction, fold 1 the 128-crop 6bba direction.
FOLD_EMBRYO = {"0": "44b6", "1": "6bba"}

# The support pack ships only ``split_0`` (trained on 6bba). It is LOEO-clean on fold 0 and
# LEAKY on fold 1 - the EXP-0019 defect. Mirrored from tests/test_loeo_weights_hygiene.py.
PACK_WEIGHTS_TOKEN = "split_0"

# THE ROLE TABLE AND EVERY FOLD-LEGITIMACY QUESTION LIVE IN ONE PLACE, WHICH IS NOT THIS FILE.
# `PKT0029_REQUIRED_TRUNK_ROLES = ('official', 'stabledet')` used to live here, and FACT-0418
# measured what it did: `audit-pair --require-roles` would have REFUSED the honest pair and
# ACCEPTED the pair with no fold-legitimate arm on either fold, while the newer
# gpu_protection_contract DATA-3 clause answered the SAME question the other way (FACT-0431). Two
# guards disagreeing is worse than one wrong guard, because whichever runs last looks
# authoritative. The constant is RETIRED rather than corrected: correcting it would have bought
# agreement by coincidence and left the next divergence invisible.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance_policy as PP  # noqa: E402

TRUNK_ROLES = set(PP.TRUNK_ROLES)

# --------------------------------------------------------------------------------------------
# THE CACHE SCHEMA - CONTRACT 2. Mirrored from assoc_feature_tap.py's own `_AFT_SCHEMA_REQUIRED`
# declaration. It is mirrored rather than imported because the tap is a PATCH SCRIPT: importing
# it executes a source rewrite against a `_ps` global. tests/test_audit_feature_cache.py parses
# the tap's literal with `ast` and fails on drift, the same anti-drift lock sync_tap_worker.py
# provides between the replay worker and the kernel patch. The lock is EXACT SET EQUALITY in both
# directions, so no key can be added on either side without this list moving with it.
#
# WHY THE KEYS CHANGED - CONTRACT 1'S PROBABILITY COLUMN NAMED NO SURFACE (`FACT-0403`, `FACT-0407`)
# --------------------------------------------------------------------------------------------
# Contract 1 wrote ONE column per band, `band_a_prob` / `band_b_prob`, plus an unqualified
# `edge_prob` union - taken from the deployed `probs`, i.e. AFTER every fusion stage. A replay of
# the PRIMARY head alone was then compared against it, and the two disagreed by up to 0.43 on
# P36. Nothing was wrong with the cache: re-derived on CPU from P36's own archived bytes, the
# primary-only arm reproduces the recorded GPU gap to seven digits and applying the deployed
# bidirectional harmonic collapses it to 1.1e-06, inside the gate's own 1e-4 tolerance by two
# orders of magnitude (`FACT-0407`). The defect was the NAME: a post-fusion quantity readable as
# a pre-fusion one.
#
# Contract 2 therefore admits NO unqualified probability. Every probability and every logit ends
# in `_preblend` or `_postblend`, the tap enforces that on itself at flush
# (assoc_feature_tap.py:493-506), and three named surfaces replace the one:
#     primary_logit_preblend      predict_edges output, BEFORE any fusion stage
#     primary_prob_preblend       the deployed activation of that, and nothing else
#     deployed_prob_postblend     `probs`, AFTER every fusion stage that ran
# It also adds BAND P - the same threshold rule applied to the PRE-fusion surface - and records
# WHICH FUSION STAGES RAN from the predictor's own locals rather than from a log header, which is
# how `FACT-0403` misattributed its own root cause in the first place.
# --------------------------------------------------------------------------------------------
CACHE_CONTRACT_VERSION = 2

REQUIRED_KEYS = (
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
# The tap calls this optional (BIOHUB_AFT_STORE_POS). This auditor does NOT: the positional
# features are a head INPUT, and a cache that omits them forces the gate to trust a
# re-derivation instead of an artifact. A full cache is launched with STORE_POS on.
POSITIONAL_KEY = "role_pos"

# The retired schema-1 markers. Their presence is defect 6 itself, not a missing-key problem, so
# they are checked FIRST and rejected under their own name.
FRAME_KEYED_MARKERS = ("feat_frames",)
FRAME_FEAT_RE = re.compile(r"^feat_-?\d+$")

# The CONTRACT-1 markers. A contract-1 cache is missing thirty keys, so the generic
# `cache_schema_incomplete` message would send the operator looking for a truncated write when
# the real answer is that its one probability column is a POST-FUSION quantity under a
# pre-fusion-sounding name. Named for the same reason the frame-keyed markers are, and refused
# in the same place the replay worker refuses it (assoc_tap_replay.py:130-150).
CONTRACT_1_MARKERS = ("band_a_prob", "band_b_prob", "edge_prob")

BAND_A, BAND_B, BAND_P = "a", "b", "p"
BANDS = (BAND_A, BAND_B, BAND_P)
ROLE_SRC, ROLE_TGT = 0, 1

# WHICH SURFACE SELECTED EACH BAND. Mirrored from the tap's own `_AFT_BAND_SURFACE`, and locked
# against it by the same ast drift test that locks the key list. Bands A and B keep their
# contract-1 membership - they are what the DEPLOYED candidate rule selected, on the deployed
# post-fusion surface - so their membership is a property of the deployment, not of the cache.
# Band P is the pre-fusion analogue and carries the parity property the gate needs.
BAND_SURFACE = {
    BAND_A: "deployed_probs_postblend",
    BAND_B: "deployed_probs_postblend",
    BAND_P: "primary_probs_preblend",
}
# surface name -> the per-row column holding that surface's value.
SURFACE_COLUMN = {
    "deployed_probs_postblend": "deployed_prob_postblend",
    "primary_probs_preblend": "primary_prob_preblend",
}
BAND_COLUMNS = ("pair", "source_id", "target_id", "i", "j",
                "primary_logit_preblend", "primary_prob_preblend", "deployed_prob_postblend")
# The auditor-facing union is bands A and B only - the DEPLOYED candidate surface. Band P is a
# gate instrument, not a deployed candidate set, and the tap deliberately leaves it out.
UNION_BANDS = (BAND_A, BAND_B)

# Per-band-row bytes at the tap's own dtypes: five int64 index columns plus THREE float64 surface
# columns (contract 1 had one).
BAND_ROW_BYTES = 5 * 8 + 3 * 8
# The union carries source_id/target_id int64 plus both named probability columns as float32.
UNION_ROW_BYTES = 8 + 8 + 4 + 4
# Per-role-node bytes that do NOT depend on feat_dim/pos_dim: role_pair int64, role_role int8,
# role_gid int64, role_coord_scaled 3x float32, role_coord_rel 4x int32, role_mask bool.
ROLE_FIXED_BYTES = 8 + 1 + 8 + 3 * 4 + 4 * 4 + 1
# Per-node bytes outside the role and band tables: one coords row (int16 x 4), and - bounded
# above by one per node because pairs <= frames <= nodes - one pair-table row (7 int64 + a
# 4-wide int64 window_shape) and one frames/starts/ends triple.
NODE_SIDE_BYTES = 4 * 2 + (7 * 8 + 4 * 8) + 3 * 8


class Reject(Exception):
    """Any condition that makes the cache uninterpretable. Never downgraded to a warning."""


# --------------------------------------------------------------------------------------------
# digests
# --------------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _reject(condition: str, crop: str, detail: str) -> None:
    raise Reject(f"{crop}:{condition}: {detail}")


def _require(ok: bool, condition: str, crop: str, detail: str) -> None:
    if not ok:
        _reject(condition, crop, detail)


# --------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------
def load_cache(path: Path) -> dict:
    """Read one crop cache, refusing anything whose node identity cannot be established.

    Frame-keying is tested BEFORE the required-key list. A schema-1 cache is missing most of the
    pair-and-role keys, so a plain "missing keys" message would send the operator to look for a
    truncated write when the real answer is that the artifact's whole addressing scheme is
    undefined (`FACT-0402`).
    """
    name = path.name
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as err:                                    # pragma: no cover - I/O shape
        raise Reject(f"{name}: unreadable cache ({type(err).__name__}: {err})") from err
    files = list(z.files)

    frame_keyed = [k for k in files if k in FRAME_KEYED_MARKERS or FRAME_FEAT_RE.match(k)]
    if frame_keyed:
        _reject(
            "frame_keyed_cache", name,
            f"the cache carries per-frame feature keys {sorted(frame_keyed)[:6]}. A trunk node "
            "feature is WINDOW-DEPENDENT: TemporalUNet3D's _TemporalAttention mixes across the "
            "window's time axis, so at the deployed window_size 2 an interior frame t is the "
            "TARGET of pair (t-1,t) and the SOURCE of pair (t,t+1) and carries TWO different "
            "vectors (FACT-0402: 66 of 173 role-nodes, max abs delta 0.0123, 2.1 orders of "
            "magnitude above the 1e-4 gate tolerance). A frame-keyed cache is therefore not a "
            "cache with a defect - it is undefined, and no head may be trained on it.",
        )

    # CONTRACT 1, refused by name and BEFORE the missing-key list, for the reason the frame-keyed
    # check comes first: a contract-1 cache is missing thirty keys, and "missing thirty keys"
    # sends the operator to look for a truncated write. The real answer is that its single
    # `band_a_prob` column holds the POST-FUSION deployed probability, so any primary-side
    # comparison against it reproduces exactly the defect that invalidated P36's verdict
    # (`FACT-0403`, corrected by `FACT-0407`). There is NO legacy read mode: a cache whose
    # probability column names no surface cannot be interpreted, only guessed at.
    if "contract_version" not in files and any(k in files for k in CONTRACT_1_MARKERS):
        _reject(
            "cache_contract_1", name,
            f"the cache carries the contract-1 columns "
            f"{sorted(k for k in CONTRACT_1_MARKERS if k in files)} and no contract_version. "
            "Contract 1 wrote ONE probability per band, taken from the deployed `probs` AFTER "
            "every fusion stage, under a name that claims no surface. Comparing a primary-side "
            "quantity against it is the defect FACT-0403 recorded and FACT-0407 re-derived: the "
            "P36 cache was FAITHFUL and its verdict was still invalid, because the reference "
            "column was post-fusion while the replay was pre-fusion. Contract 2 records "
            "primary_logit_preblend / primary_prob_preblend / deployed_prob_postblend "
            "separately, and there is no rewrite of a contract-1 file that recovers the "
            "pre-fusion surface it never stored. Re-run the tap at contract "
            f"{CACHE_CONTRACT_VERSION}.",
        )

    missing = [k for k in REQUIRED_KEYS if k not in files]
    if missing:
        _reject(
            "cache_schema_incomplete", name,
            f"missing {missing[:8]}{'...' if len(missing) > 8 else ''}. These are the keys "
            "assoc_feature_tap.py declares in _AFT_SCHEMA_REQUIRED and assoc_tap_replay.py "
            "reads by name; without them the pair, role and node identity of every feature row "
            "is unprovable.",
        )
    contract = int(z["contract_version"])
    if contract != CACHE_CONTRACT_VERSION:
        _reject(
            "cache_contract_version_unknown", name,
            f"contract_version {contract} != {CACHE_CONTRACT_VERSION}. Every column's MEANING is "
            "a property of the contract that wrote it - contract 1's `band_a_prob` and contract "
            "2's `band_a_deployed_prob_postblend` hold the same numbers under different claims - "
            "so a contract this auditor has not been written against cannot be read by guessing.",
        )
    schema = int(z["schema_version"])
    if schema != SCHEMA_VERSION:
        _reject(
            "cache_schema_version_unknown", name,
            f"schema_version {schema} != {SCHEMA_VERSION}. Schema 1 keyed features BY FRAME, "
            "which FACT-0402 measured to be undefined at the deployed window.",
        )
    if POSITIONAL_KEY not in files:
        _reject(
            "positional_features_absent", name,
            "role_pos is not in the cache. The positional features are an INPUT to the deployed "
            "predict_edges, so a cache without them forces the gate to trust a re-derivation "
            "rather than an artifact. Launch the tap with BIOHUB_AFT_STORE_POS=1.",
        )
    return {k: z[k] for k in files}


# --------------------------------------------------------------------------------------------
# the role table
# --------------------------------------------------------------------------------------------
def _role_blocks(data: dict) -> list[dict]:
    """Every (pair, role) block, in the order the tap appends them: src then tgt, pair by pair."""
    blocks = []
    n_pairs = int(data["pair_f_idx"].shape[0])
    for pair in range(n_pairs):
        for role, ptr_key, n_key, t_key in (
            (ROLE_SRC, "pair_src_ptr", "pair_src_n", "pair_t_src"),
            (ROLE_TGT, "pair_tgt_ptr", "pair_tgt_n", "pair_t_tgt"),
        ):
            blocks.append({
                "pair": pair, "role": role,
                "ptr": int(data[ptr_key][pair]), "n": int(data[n_key][pair]),
                "frame": int(data[t_key][pair]),
            })
    return blocks


def _validate_partition(crop: str, data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = np.asarray(data["frames"]).astype(np.int64)
    starts = np.asarray(data["starts"]).astype(np.int64)
    ends = np.asarray(data["ends"]).astype(np.int64)
    coords = np.asarray(data["coords"])
    node_count = int(data["node_count"])

    _require(len(frames) == len(starts) == len(ends), "frame_partition_torn", crop,
             f"frames/starts/ends lengths {len(frames)}/{len(starts)}/{len(ends)} disagree")
    _require(len(frames) > 0, "frame_partition_empty", crop, "no frames recorded")
    _require(coords.ndim == 2 and coords.shape[1] == 4, "coords_shape", crop,
             f"coords has shape {coords.shape}, expected (N, 4) of [t, z, y, x]")
    _require(int(coords.shape[0]) == node_count, "node_count_disagrees_with_coords", crop,
             f"coords has {coords.shape[0]} rows, node_count records {node_count}")
    _require(bool(np.all(np.diff(frames) > 0)), "frame_partition_not_increasing", crop,
             "frames are not strictly increasing, so a positional node id is ambiguous")
    _require(bool(np.all(starts[1:] == ends[:-1])) and int(starts[0]) == 0
             and int(ends[-1]) == node_count,
             "frame_partition_not_contiguous", crop,
             "the frame blocks do not tile [0, node_count) contiguously; node ids are "
             "positional indices into coords_so_far and mean nothing without that tiling")
    # coords column 0 is the ABSOLUTE frame index the detector stamped
    # (_detect_cells_pooled(det_logits[f][0], t, ...), predict_unet_transformer.py:543-546).
    for f, s, e in zip(frames.tolist(), starts.tolist(), ends.tolist()):
        _require(bool(np.all(coords[s:e, 0] == f)), "coords_frame_column_disagrees", crop,
                 f"coords rows [{s},{e}) are stamped with a frame other than {f}")
    return frames, starts, ends


def _validate_pairs(crop: str, data: dict) -> int:
    window = int(data["window"])
    n_pairs = int(data["pair_f_idx"].shape[0])
    _require(window >= 2, "window_too_small", crop,
             f"window {window}: a window below 2 contains no consecutive pair, so no association "
             "was ever computed")
    for key in ("pair_t_src", "pair_t_tgt", "pair_src_ptr", "pair_src_n",
                "pair_tgt_ptr", "pair_tgt_n"):
        _require(int(np.asarray(data[key]).shape[0]) == n_pairs, "pair_table_torn", crop,
                 f"{key} has {np.asarray(data[key]).shape[0]} rows, pair_f_idx has {n_pairs}")
    _require(n_pairs > 0, "no_pairs_recorded", crop,
             "the cache records zero frame pairs. The tap prints AFT_TAP_EMPTY rather than "
             "writing this, so an empty pair table means the file was rewritten")

    f_idx = np.asarray(data["pair_f_idx"]).astype(np.int64)
    t_src = np.asarray(data["pair_t_src"]).astype(np.int64)
    t_tgt = np.asarray(data["pair_t_tgt"]).astype(np.int64)
    ws = np.asarray(data["pair_window_shape"]).astype(np.int64)
    image_shape = np.asarray(data["image_shape"]).astype(np.int64)

    _require(bool(np.all((f_idx >= 0) & (f_idx <= window - 2))), "pair_f_idx_out_of_window", crop,
             f"pair_f_idx outside [0, {window - 2}]; f_idx is the WITHIN-WINDOW source index "
             "(predict_unet_transformer.py:556-557), not an absolute frame")
    _require(bool(np.all(t_tgt - t_src == 1)), "pair_stride_is_not_one_frame", crop,
             "t_tgt - t_src != 1. The deployed loop takes consecutive frame_indices, so a pair "
             "spanning more than one frame is not a deployed pair")
    keys = list(zip(t_src.tolist(), t_tgt.tolist()))
    _require(len(set(keys)) == len(keys), "duplicate_frame_pair", crop,
             "the same (t_src, t_tgt) is recorded twice; the deployed loop dedups on seen_pairs, "
             "so a repeat means two runs were merged into one file")
    _require(bool(np.all(np.diff(t_src) > 0)), "pairs_out_of_order", crop,
             "pair_t_src is not strictly increasing; windows slide forward")
    _require(ws.ndim == 2 and ws.shape == (n_pairs, image_shape.shape[0]),
             "window_shape_torn", crop,
             f"pair_window_shape {ws.shape} does not match {n_pairs} pairs x "
             f"{image_shape.shape[0]} dims")
    _require(bool(np.all(ws[:, 0] == window)), "window_shape_time_axis", crop,
             "pair_window_shape[:, 0] != window; window_shape = (W,) + image_shape[1:]")
    _require(bool(np.all(ws[:, 1:] == image_shape[1:])), "window_shape_spatial_axes", crop,
             "pair_window_shape spatial dims differ from image_shape; the positional features "
             "are normalised by this shape, so a torn record silently rescales every node")
    return n_pairs


def _validate_roles(crop: str, data: dict, frames, starts, ends) -> dict:
    """The role table: block tiling, per-block identity, node ordering, and MULTIPLICITY."""
    blocks = _role_blocks(data)
    role_pair = np.asarray(data["role_pair"]).astype(np.int64)
    role_role = np.asarray(data["role_role"]).astype(np.int64)
    role_gid = np.asarray(data["role_gid"]).astype(np.int64)
    role_feat = np.asarray(data["role_feat"])
    role_mask = np.asarray(data["role_mask"])
    role_rel = np.asarray(data["role_coord_rel"]).astype(np.int64)
    role_scaled = np.asarray(data["role_coord_scaled"])
    role_pos = np.asarray(data[POSITIONAL_KEY])
    n_role = int(role_feat.shape[0])
    feat_dim = int(data["feat_dim"])
    pos_dim = int(data["pos_dim"])
    downsample = np.asarray(data["downsample"]).astype(np.int64)
    f_idx = np.asarray(data["pair_f_idx"]).astype(np.int64)

    for key, arr in (("role_pair", role_pair), ("role_role", role_role), ("role_gid", role_gid),
                     ("role_mask", role_mask)):
        _require(int(arr.shape[0]) == n_role, "role_table_torn", crop,
                 f"{key} has {arr.shape[0]} rows, role_feat has {n_role}")
    _require(role_feat.ndim == 2 and int(role_feat.shape[1]) == feat_dim,
             "feat_dim_disagrees", crop,
             f"role_feat is {role_feat.shape}, feat_dim records {feat_dim}")
    _require(role_pos.shape == (n_role, pos_dim), "pos_dim_disagrees", crop,
             f"role_pos is {role_pos.shape}, expected ({n_role}, {pos_dim})")
    _require(pos_dim > 0 and feat_dim > 0, "zero_width_features", crop,
             f"feat_dim {feat_dim}, pos_dim {pos_dim}")
    _require(role_scaled.shape == (n_role, 3), "role_coord_scaled_shape", crop,
             f"role_coord_scaled is {role_scaled.shape}, expected ({n_role}, 3)")
    _require(role_rel.shape == (n_role, 4), "role_coord_rel_shape", crop,
             f"role_coord_rel is {role_rel.shape}, expected ({n_role}, 4)")
    _require(bool(np.isfinite(role_feat).all()), "non_finite_features", crop,
             "role_feat holds a NaN or an Inf")
    _require(bool(np.isfinite(role_pos).all()), "non_finite_positional_features", crop,
             "role_pos holds a NaN or an Inf")
    _require(str(role_feat.dtype) == str(data["feat_dtype"]), "feat_dtype_disagrees", crop,
             f"role_feat is {role_feat.dtype}, feat_dtype records {str(data['feat_dtype'])!r}")

    frame_list = frames.tolist()
    node_count = int(data["node_count"])
    _require(bool(np.all((role_gid >= 0) & (role_gid < node_count))), "role_gid_out_of_range",
             crop, f"a role row names a node outside [0, {node_count})")
    for blk in blocks:
        _require(blk["frame"] in frame_list, "role_frame_not_in_partition", crop,
                 f"pair {blk['pair']} role {blk['role']} names frame {blk['frame']}, absent from "
                 "the deployed coord_offset")

    # --- DEFECT 6, PART TWO: MULTIPLICITY ----------------------------------------------------
    # Checked BEFORE the block tiling, because a frame-keyed writer wearing this schema points
    # both roles at ONE shared per-frame block: the tiling check would fire first and send the
    # operator looking for a torn pointer when the real fault is that half the feature vectors
    # were never computed. The pair table says exactly how many (pair, role) records each global
    # node must have. FEWER is the old worker's feats[t] de-duplication; MORE is a merged or
    # double-flushed file. Neither is auditable, and both look clean by shape.
    expected_mult: dict[int, int] = {}
    for blk in blocks:
        k = frame_list.index(blk["frame"])
        for gid in range(int(starts[k]), int(ends[k])):
            expected_mult[gid] = expected_mult.get(gid, 0) + 1
    observed = np.bincount(role_gid, minlength=node_count)
    missing = [g for g, want in expected_mult.items() if int(observed[g]) < want]
    extra = [g for g, want in expected_mult.items() if int(observed[g]) > want]
    unexpected = [g for g in range(len(observed)) if observed[g] and g not in expected_mult]
    _require(not missing, "role_records_missing", crop,
             f"{len(missing)} nodes carry FEWER role records than the pair table requires "
             f"(first: {missing[:5]}). A node that is the target of (t-1,t) and the source of "
             "(t,t+1) must appear TWICE with two different vectors; appearing once is defect 6 "
             "recurring - the old worker cached feats[t] on first sight and reused it for both "
             "roles (FACT-0402)")
    _require(not extra and not unexpected, "duplicate_role_record", crop,
             f"{len(extra) + len(unexpected)} nodes carry MORE role records than the pair table "
             f"requires (first: {(extra + unexpected)[:5]}). Two flushes were merged, or a block "
             "was appended twice; either way a training set built from this file double-counts")

    # --- DEFECT 6, PART THREE: THE VECTORS MUST ACTUALLY DIFFER ------------------------------
    # The shape can be right and the content still be frame-keyed. `_TemporalAttention` mixes
    # across the window (temporal_unet.py:30-48, 125-131), so a node's source-role and
    # target-role vectors come from DIFFERENT forward passes and cannot be bit-identical for
    # every dual-role node at once. Measured on the real path: 66 of 173 differ, max abs 0.0123.
    first_row: dict[int, int] = {}
    dual: list[tuple[int, int]] = []
    for row, gid in enumerate(role_gid.tolist()):
        if gid in first_row:
            dual.append((first_row[gid], row))
        else:
            first_row[gid] = row
    dual_delta = 0.0
    dual_rel = 0.0
    if dual:
        a = role_feat[[i for i, _ in dual]].astype(np.float64)
        b = role_feat[[j for _, j in dual]].astype(np.float64)
        diff = np.abs(a - b)
        dual_delta = float(diff.max())
        scale = np.maximum(np.abs(a), np.abs(b))
        dual_rel = float((diff / np.where(scale > 0, scale, 1.0)).max())
        _require(dual_delta > 0.0, "role_features_frame_keyed", crop,
                 f"all {len(dual)} dual-role nodes carry BIT-IDENTICAL features in both roles. "
                 "TemporalUNet3D's _TemporalAttention mixes across the window's time axis, so "
                 "the source-role and target-role vectors come from different forward passes and "
                 "cannot all agree (FACT-0402 measured 66 of 173 differing, max abs 0.0123). "
                 "This file has the pair-and-role SHAPE with frame-keyed CONTENT - defect 6 "
                 "reintroduced by a writer that read the schema and not the mechanism")

    # --- the blocks must TILE [0, n_role) exactly, in emission order --------------------------
    cursor = 0
    for blk in blocks:
        _require(blk["n"] > 0, "empty_role_block", crop,
                 f"pair {blk['pair']} role {blk['role']} has zero nodes; the deployed loop skips "
                 "a pair whose source or target frame is empty rather than recording it")
        _require(blk["ptr"] == cursor, "role_blocks_do_not_tile", crop,
                 f"pair {blk['pair']} role {blk['role']} starts at {blk['ptr']}, expected "
                 f"{cursor}. Overlapping or gapped blocks mean a role row belongs to two pairs "
                 "or to none")
        sl = slice(blk["ptr"], blk["ptr"] + blk["n"])
        _require(bool(np.all(role_pair[sl] == blk["pair"])), "role_pair_column_wrong", crop,
                 f"role_pair disagrees with the pair table over rows {sl.start}:{sl.stop}")
        _require(bool(np.all(role_role[sl] == blk["role"])), "role_role_column_wrong", crop,
                 f"role_role disagrees with the pair table over rows {sl.start}:{sl.stop}")
        # NODE ORDERING: the block must be exactly the deployed frame slice, in the deployed
        # order (idx_src = np.arange(s_src, e_src), predict_unet_transformer.py:572-573).
        k = frame_list.index(blk["frame"])
        expected = np.arange(int(starts[k]), int(ends[k]), dtype=np.int64)
        _require(role_gid[sl].shape == expected.shape
                 and bool(np.array_equal(role_gid[sl], expected)),
                 "role_node_order_is_not_the_deployed_frame_slice", crop,
                 f"pair {blk['pair']} role {blk['role']} frame {blk['frame']}: cached ids are not "
                 f"arange({int(starts[k])}, {int(ends[k])}). Every downstream artifact addresses "
                 "nodes by that number, so a reorder silently renames every cell")
        # WINDOW-RELATIVE TIME: c_src_rel[:, 0] = f_idx and c_tgt_rel[:, 0] = f_idx + 1
        # (predict_unet_transformer.py:580-583). This is what makes the role part of the record.
        want_t = int(f_idx[blk["pair"]]) + blk["role"]
        _require(bool(np.all(role_rel[sl, 0] == want_t)),
                 "role_relative_time_wrong", crop,
                 f"pair {blk['pair']} role {blk['role']}: role_coord_rel time column is not "
                 f"{want_t}. The source sits at f_idx and the target at f_idx+1; a block whose "
                 "time column does not match is filed under the wrong role")
        cursor += blk["n"]
    _require(cursor == n_role, "role_rows_outside_every_block", crop,
             f"the pair table accounts for {cursor} role rows but role_feat has {n_role}")

    # --- the geometry cross-checks the tap cannot fake ---------------------------------------
    ws = np.asarray(data["pair_window_shape"]).astype(np.int64)
    bound = ws[role_pair, 1:]
    _require(bool(np.all(role_rel[:, 1:] >= 0)) and bool(np.all(role_rel[:, 1:] < bound)),
             "role_coord_outside_its_window", crop,
             "a role coordinate lies outside its own window_shape, so extract_pos_features "
             "normalised it out of [0, 1)")
    # role_coord_scaled is p_coords * ds_arr_t, i.e. the SAME rows as role_coord_rel[:, 1:]
    # times the downsample (predict_unet_transformer.py:576-577, 597). Both are exactly
    # representable in float32, so this is an equality, not a tolerance.
    want_scaled = role_rel[:, 1:].astype(np.float32) * downsample.astype(np.float32)
    _require(bool(np.array_equal(role_scaled.astype(np.float32), want_scaled)),
             "role_coord_scaled_disagrees_with_role_coord_rel", crop,
             "role_coord_scaled != role_coord_rel[:, 1:] * downsample. One of the two records is "
             "torn, and the head consumes both")
    _require(bool(np.asarray(role_mask).all()), "role_mask_has_a_false_slot", crop,
             f"{int((~np.asarray(role_mask)).sum())} mask slots are False; the deployed site "
             "builds torch.ones and never pads, so a False slot is not a deployed input")

    return {
        "role_nodes": n_role,
        "dual_role_nodes": len(dual),
        "dual_role_max_abs_delta": dual_delta,
        # NOTE the definition: max ELEMENTWISE |a-b| / max(|a|,|b|). FACT-0402 records a
        # 3.5% relative delta under a different normalisation; do not compare the two.
        "dual_role_max_elementwise_relative_delta": dual_rel,
        "role_digest": _digest(role_pair.astype(np.int64), role_role.astype(np.int8), role_gid),
        "features_digest": _digest(role_feat.astype(np.float64)),
        "positional_digest": _digest(role_pos.astype(np.float64)),
    }


def _validate_fusion(crop: str, data: dict) -> dict:
    """WHICH FUSION STAGES RAN, and whether the record is internally consistent.

    This is the column contract 1 did not have, and its absence is the whole reason P36's verdict
    was invalid. `FACT-0403` blamed a secondary-model blend at 0.15 because a setup-cell log
    header said so; `loeo_retarget.py:128` had disabled the secondary three lines later and the
    stage that actually ran was the bidirectional harmonic at 0.20 (`FACT-0407`). A LOG HEADER
    RECORDS AN INTENTION AT THE MOMENT IT PRINTED, NOT WHAT RAN. The tap now reads this from the
    predictor's own locals (assoc_feature_tap.py:241-274, :568-586), so the auditor's job is to
    refuse a record that contradicts itself rather than to re-infer the stages.

    NOTE what is deliberately NOT a defect: a non-zero `secondary_edge_weight` with
    `secondary_enabled` False. That is exactly the P36 state - a weight configured and the model
    then disabled - and recording both is the point.
    """
    raw = str(data["fusion_stages"])
    recorded = [] if raw == "none" else [s for s in raw.split(",") if s]
    bidirectional = float(data["fusion_bidirectional_weight"])
    secondary_enabled = bool(data["fusion_secondary_enabled"])
    secondary_weight = float(data["fusion_secondary_edge_weight"])
    reproducible = bool(data["fusion_reproducible_from_primary_cache"])

    _require(np.isfinite([bidirectional, secondary_weight]).all(),
             "fusion_weight_is_not_finite", crop,
             f"bidirectional {bidirectional}, secondary {secondary_weight}")
    _require(0.0 <= bidirectional < 1.0, "fusion_bidirectional_weight_out_of_range", crop,
             f"bidirectional weight {bidirectional} outside [0, 1). The harmonic mixes "
             "(1-w)/p_forward + w/p_reverse, so a weight at or above 1 does not blend, it "
             "replaces the forward surface with the reverse one")

    expected = []
    if bidirectional > 0.0:
        expected.append("bidirectional_harmonic")
    if secondary_enabled:
        expected.append("secondary_logit_blend")
    _require(recorded == expected, "fusion_stages_disagree_with_the_recorded_weights", crop,
             f"fusion_stages {raw!r} but bidirectional_weight {bidirectional} and "
             f"secondary_enabled {secondary_enabled} imply {expected or ['none']}. The stage "
             "list is what a reader uses to decide whether the deployed surface is "
             "reconstructible; a list that disagrees with its own weights is worse than none")
    # The tap's own definition, at assoc_feature_tap.py:262-265: the bidirectional stage re-runs
    # the SAME primary model on the SAME cached tensors with the roles swapped, so a replay
    # reproduces it exactly from this cache; the secondary stage needs a SECOND MODEL's features,
    # which this cache does not and must not carry.
    _require(reproducible == (not secondary_enabled),
             "fusion_reproducibility_claim_is_wrong", crop,
             f"fusion_reproducible_from_primary_cache {reproducible} with secondary_enabled "
             f"{secondary_enabled}. A cache cannot reconstruct a surface a second model "
             "contributed to, and it always can reconstruct one only the primary produced")
    return {
        "stages": recorded,
        "stages_recorded": raw,
        "bidirectional_weight": bidirectional,
        "secondary_enabled": secondary_enabled,
        "secondary_edge_weight": secondary_weight,
        "secondary_link_mode": str(data["fusion_secondary_link_mode"]),
        "secondary_mix_temperature": float(data["fusion_secondary_mix_temperature"]),
        "reproducible_from_primary_cache": reproducible,
        "deployed_surface_is_the_primary_surface": not recorded,
    }


def _validate_bands(crop: str, data: dict, fusion: dict) -> dict:
    n_pairs = int(data["pair_f_idx"].shape[0])
    threshold = float(data["edge_threshold"])
    floor_b = float(data["band_b_floor"])
    topk = int(data["band_b_topk"])
    activation = str(data["edge_activation"])
    node_count = int(data["node_count"])
    src_n = np.asarray(data["pair_src_n"]).astype(np.int64)
    tgt_n = np.asarray(data["pair_tgt_n"]).astype(np.int64)
    role_gid = np.asarray(data["role_gid"]).astype(np.int64)
    src_ptr = np.asarray(data["pair_src_ptr"]).astype(np.int64)
    tgt_ptr = np.asarray(data["pair_tgt_ptr"]).astype(np.int64)

    _require(activation in ("softmax", "sigmoid"), "unknown_edge_activation", crop,
             f"edge_activation {activation!r}")
    _require(0.0 < floor_b < threshold < 1.0, "band_thresholds_not_ordered", crop,
             f"band_b_floor {floor_b} / edge_threshold {threshold} do not satisfy "
             "0 < floor < threshold < 1")
    _require(topk >= 1, "band_b_topk_invalid", crop, f"band_b_topk {topk}")

    empty_because = {
        BAND_A: "nothing in this crop can be checked against the DEPLOYED surface, so a pass "
                "here certifies nothing",
        BAND_B: "the cache holds none of the learnable population - FACT-0382 measured that ALL "
                "691 fold-0 contested errors have their true parent at or below the deployed "
                "threshold",
        BAND_P: "band P is the PARITY surface. Bands A and B are selected on the post-fusion "
                "probabilities, so their membership is a property of the deployment and cannot "
                "be re-derived from a primary-only replay without changing what they mean. Band "
                "P is the only band whose membership a replay may reproduce, so an empty one "
                "leaves the gate with no membership property at all (FACT-0403)",
    }

    out = {}
    keysets = {}
    for name in BANDS:
        cols = {c: np.asarray(data[f"band_{name}_{c}"]) for c in BAND_COLUMNS}
        lens = {c: int(v.shape[0]) for c, v in cols.items()}
        _require(len(set(lens.values())) == 1, f"band_{name}_torn", crop,
                 f"band {name} column lengths {lens} disagree")
        n = lens["pair"]
        _require(n > 0, f"band_{name}_is_empty", crop, f"zero rows. {empty_because[name]}")

        # WHICH SURFACE SELECTED THIS BAND, taken from the artifact and not assumed. A mislabel
        # here IS the contract-1 defect: a post-fusion population read as a pre-fusion one.
        surface = str(data[f"band_{name}_selected_on"])
        _require(surface == BAND_SURFACE[name], f"band_{name}_selected_on_is_wrong", crop,
                 f"band {name} declares it was selected on {surface!r}; the tap selects it on "
                 f"{BAND_SURFACE[name]!r} (assoc_feature_tap.py _AFT_BAND_SURFACE). A band whose "
                 "recorded selection surface is not the one that chose it is exactly the "
                 "confusion that invalidated P36's verdict, with a contract-2 header on it")

        pair = cols["pair"].astype(np.int64)
        i = cols["i"].astype(np.int64)
        j = cols["j"].astype(np.int64)
        logit_pre = cols["primary_logit_preblend"].astype(np.float64)
        prob_pre = cols["primary_prob_preblend"].astype(np.float64)
        prob_post = cols["deployed_prob_postblend"].astype(np.float64)
        # `prob` below is THE SURFACE THAT SELECTED THIS BAND, never "the probability".
        prob = {"primary_prob_preblend": prob_pre,
                "deployed_prob_postblend": prob_post}[SURFACE_COLUMN[surface]]
        _require(bool(np.all((pair >= 0) & (pair < n_pairs))), f"band_{name}_pair_out_of_range",
                 crop, f"a band {name} row names a pair outside [0, {n_pairs})")
        _require(bool(np.all((i >= 0) & (i < src_n[pair]))), f"band_{name}_source_index_outside_block",
                 crop, f"a band {name} local source index is outside its pair's source block")
        _require(bool(np.all((j >= 0) & (j < tgt_n[pair]))), f"band_{name}_target_index_outside_block",
                 crop, f"a band {name} local target index is outside its pair's target block")
        # The recorded GLOBAL ids must be what the role table says they are. This ties the band
        # artifacts to the numbering the pre-ILP export and the ECB sidecars address.
        want_src = role_gid[src_ptr[pair] + i]
        want_tgt = role_gid[tgt_ptr[pair] + j]
        _require(bool(np.array_equal(cols["source_id"].astype(np.int64), want_src))
                 and bool(np.array_equal(cols["target_id"].astype(np.int64), want_tgt)),
                 f"band_{name}_ids_disagree_with_the_role_table", crop,
                 f"a band {name} global id is not the id the role table gives that local index; "
                 "an id in any downstream artifact would then name a different cell")
        _require(bool(np.all((cols["source_id"] >= 0) & (cols["source_id"] < node_count))
                      and np.all((cols["target_id"] >= 0) & (cols["target_id"] < node_count))),
                 f"band_{name}_id_out_of_range", crop,
                 f"a band {name} global id is outside [0, {node_count})")
        # THE SELECTING surface must lie in (0, 1]; the OTHER surface only in [0, 1]. That
        # asymmetry is measured, not cautious: with a secondary blend at 0.15 on the real tap a
        # band-A row's `primary_prob_preblend` reached down to 0.4568 and a band-B row's to
        # 0.0088, both far outside their own band's window. A softmax can also underflow to
        # exactly 0.0 on the non-selecting axis, so a strict lower bound there would reject a
        # faithful cache.
        _require(bool(np.all((prob > 0.0) & (prob <= 1.0))), f"band_{name}_prob_out_of_range",
                 crop, f"band {name} {SURFACE_COLUMN[surface]} outside (0, 1]")
        for label, arr in (("primary_prob_preblend", prob_pre),
                           ("deployed_prob_postblend", prob_post)):
            _require(bool(np.all((arr >= 0.0) & (arr <= 1.0))),
                     f"band_{name}_{label}_out_of_range", crop,
                     f"band {name} {label} outside [0, 1]")
        _require(bool(np.isfinite(logit_pre).all()), f"band_{name}_logit_not_finite", crop,
                 f"band {name} primary_logit_preblend holds a NaN or an Inf")

        if name == BAND_A:
            _require(bool(np.all(prob > threshold)), "band_a_below_the_deployed_threshold", crop,
                     f"a band A row sits at or below the deployed threshold {threshold}; band A "
                     "is defined as probs > threshold at the deployed site")
        elif name == BAND_P:
            _require(bool(np.all(prob > threshold)), "band_p_below_the_deployed_threshold", crop,
                     f"a band P row sits at or below {threshold}. Band P applies the SAME "
                     "threshold rule to the PRE-fusion surface; a row below it was not selected "
                     "by that rule")
        else:
            _require(bool(np.all((prob > floor_b) & (prob <= threshold))),
                     "band_b_outside_its_acquisition_window", crop,
                     f"a band B row sits outside ({floor_b}, {threshold}]; band B is the ECB "
                     "acquisition surface above the floor and at or below the deployed threshold")
            counts: dict[tuple[int, int], int] = {}
            for p, jj in zip(pair.tolist(), j.tolist()):
                counts[(p, jj)] = counts.get((p, jj), 0) + 1
            worst = max(counts.values())
            _require(worst <= topk, "band_b_exceeds_its_rank_cap", crop,
                     f"a target holds {worst} band-B rows, above the recorded cap {topk}; the "
                     "cache does not have the membership the ECB sidecars would record")

        # THE LOGIT COLUMN IS NOT DECORATION, AND THIS IS WHAT AUDITS IT. `primary_prob_preblend`
        # is the DEPLOYED activation of `primary_logit_preblend` and nothing else
        # (assoc_feature_tap.py:329-333). Both deployed activations - a source-axis softmax and a
        # sigmoid - are strictly increasing in the logit at fixed target, so within one
        # (pair, target) the two columns must induce the SAME ORDER. This is the only check that
        # can see a torn logit column, and the logits exist precisely to close the blind spot
        # where a per-target constant offset is invisible after the softmax.
        order = np.lexsort((logit_pre, j, pair))
        same_target = (pair[order][1:] == pair[order][:-1]) & (j[order][1:] == j[order][:-1])
        inverted = same_target & (np.diff(prob_pre[order]) < -1e-12)
        _require(not bool(inverted.any()), f"band_{name}_logit_and_probability_disagree", crop,
                 f"{int(inverted.sum())} band {name} row pairs share a (pair, target) and rank "
                 "differently by primary_logit_preblend than by primary_prob_preblend. The "
                 "probability is the deployed activation of that logit, and both deployed "
                 "activations are strictly increasing in it, so one of the two columns is torn")

        # WHEN NO FUSION STAGE RAN, `probs` IS the primary activation - the same tensor through
        # the same op - so the two columns are BITWISE equal. Measured on the real tap: exactly
        # 0.0 with no stage, and 0.31 with a secondary at 0.15.
        if fusion["deployed_surface_is_the_primary_surface"]:
            _require(bool(np.array_equal(prob_pre, prob_post)),
                     f"no_fusion_but_band_{name}_surfaces_differ", crop,
                     f"fusion_stages records no stage, yet band {name}'s "
                     f"primary_prob_preblend and deployed_prob_postblend differ by up to "
                     f"{float(np.abs(prob_pre - prob_post).max()):.6g}. With no stage the "
                     "deployed probability IS the primary activation, so either a stage ran and "
                     "was not recorded - the FACT-0403 failure exactly - or a column is torn")

        keysets[name] = set(zip(pair.tolist(), i.tolist(), j.tolist()))
        out[f"band_{name}"] = {
            "rows": n,
            "selected_on": surface,
            "min_selected_prob": float(prob.min()), "max_selected_prob": float(prob.max()),
            "max_abs_preblend_postblend_delta": float(np.abs(prob_pre - prob_post).max()),
            "digest": _digest(pair, i, j, cols["source_id"].astype(np.int64),
                              cols["target_id"].astype(np.int64),
                              logit_pre, prob_pre, prob_post),
        }
    overlap = keysets[BAND_A] & keysets[BAND_B]
    _require(not overlap, "bands_overlap", crop,
             f"{len(overlap)} (pair, i, j) keys appear in BOTH bands; the tap removes band A's "
             "members from band B, so an overlap means the two were built from different runs")

    # THE MEMBERSHIP STATEMENT THE PARITY BAND EXISTS TO MAKE. With no fusion stage, band A's rule
    # (post-fusion > threshold) and band P's rule (pre-fusion > threshold) are applied to the same
    # numbers, so their memberships are IDENTICAL. Measured on the real tap: identical with no
    # stage; 19 rows became 8 with a secondary at 0.15.
    if fusion["deployed_surface_is_the_primary_surface"]:
        _require(keysets[BAND_P] == keysets[BAND_A], "no_fusion_but_band_p_is_not_band_a", crop,
                 f"fusion_stages records no stage, but band P holds {len(keysets[BAND_P])} rows "
                 f"and band A {len(keysets[BAND_A])}, differing on "
                 f"{len(keysets[BAND_P] ^ keysets[BAND_A])} keys. Two thresholdings of the same "
                 "numbers cannot select different rows")

    # The auditor-facing union the tap writes for exactly this check. BAND P IS NOT IN IT: it is a
    # gate instrument, not a deployed candidate set, and folding it in would inflate the deployed
    # candidate surface by the rows the deployment never saw.
    union_parts = {col: tuple(f"band_{b}_{col}" for b in UNION_BANDS) for col in
                   ("source_id", "target_id")}
    for col, parts, cast in (
        ("source_id", union_parts["source_id"], np.int64),
        ("target_id", union_parts["target_id"], np.int64),
        ("deployed_edge_prob_postblend",
         tuple(f"band_{b}_deployed_prob_postblend" for b in UNION_BANDS), np.float32),
        ("primary_edge_prob_preblend",
         tuple(f"band_{b}_primary_prob_preblend" for b in UNION_BANDS), np.float32),
    ):
        want = np.concatenate([np.asarray(data[p]) for p in parts]).astype(cast)
        got = np.asarray(data[col]).astype(cast)
        _require(got.shape == want.shape and bool(np.array_equal(got, want)),
                 "union_surface_disagrees_with_its_bands", crop,
                 f"{col} is not the concatenation of {parts}; the candidate surface and the "
                 "bands it is made of have drifted apart")

    deployed = np.asarray(data["deployed_edge_prob_postblend"]).astype(np.float64)
    out.update({
        "deployed_threshold": threshold,
        "band_b_floor": floor_b,
        "band_b_topk": topk,
        "edge_activation": activation,
        "union_bands": list(UNION_BANDS),
        "above_threshold": int((deployed > threshold).sum()),
        "sub_threshold": int((deployed <= threshold).sum()),
        "total": int(deployed.shape[0]),
        "surface_digest": _digest(
            np.asarray(data["source_id"]).astype(np.int64),
            np.asarray(data["target_id"]).astype(np.int64),
            np.asarray(data["deployed_edge_prob_postblend"]).astype(np.float32),
            np.asarray(data["primary_edge_prob_preblend"]).astype(np.float32)),
    })
    return out


# --------------------------------------------------------------------------------------------
# per-crop binding
# --------------------------------------------------------------------------------------------
def crop_binding(crop: str, data: dict, deployed_floor: float,
                 *, bytes_on_disk: int | None = None,
                 expect_window: int | None = None) -> dict:
    """Validate one crop and return the record the manifest binds. Raises Reject on any defect."""
    window = int(data["window"])
    if expect_window is not None:
        _require(window == int(expect_window), "window_disagrees_with_the_manifest", crop,
                 f"cache window {window}, manifest declares {expect_window}")
    frames, starts, ends = _validate_partition(crop, data)
    n_pairs = _validate_pairs(crop, data)
    roles = _validate_roles(crop, data, frames, starts, ends)
    fusion = _validate_fusion(crop, data)
    bands = _validate_bands(crop, data, fusion)

    coords = np.asarray(data["coords"])
    role_feat = np.asarray(data["role_feat"])
    uncompressed = int(sum(int(np.asarray(v).nbytes) for v in data.values()))
    storage = {
        "bytes_on_disk": int(bytes_on_disk) if bytes_on_disk is not None else None,
        "uncompressed_bytes": uncompressed,
        "compression_ratio": (float(bytes_on_disk) / uncompressed
                              if bytes_on_disk is not None and uncompressed else None),
        # Band rows now count all THREE bands, because all three occupy BAND_ROW_BYTES on disk;
        # the union carries only the two the deployed candidate surface is made of.
        "band_rows": int(sum(bands[f"band_{b}"]["rows"] for b in BANDS)),
        "union_rows": int(sum(bands[f"band_{b}"]["rows"] for b in UNION_BANDS)),
        "role_nodes": roles["role_nodes"],
    }
    if bytes_on_disk is not None:
        storage["bytes_per_node"] = float(bytes_on_disk) / max(int(data["node_count"]), 1)
        storage["bytes_per_role_node"] = float(bytes_on_disk) / max(roles["role_nodes"], 1)
        storage["bytes_per_band_row"] = float(bytes_on_disk) / max(storage["band_rows"], 1)

    return {
        "crop": crop,
        "embryo": crop.split("_")[0],
        "nodes": int(data["node_count"]),
        "frames": [int(t) for t in frames.tolist()],
        "pairs": n_pairs,
        "window": window,
        "feature_dim": int(data["feat_dim"]),
        "pos_dim": int(data["pos_dim"]),
        "feat_dtype": str(data["feat_dtype"]),
        "downsample": [int(v) for v in np.asarray(data["downsample"]).tolist()],
        "det_threshold": float(data["det_threshold"]),
        "quantiles": [float(data["q_low"]), float(data["q_high"])],
        "node_order": {
            "convention": "positional index into coords_so_far, blocks in increasing frame order",
            "feature_key": "(pair, role) - NOT frame; FACT-0402",
            "coords_digest": _digest(coords),
            "role_digest": roles["role_digest"],
            "features_digest": roles["features_digest"],
            "positional_digest": roles["positional_digest"],
            "coord_feature_pair_digest": _digest(
                coords.astype(np.float64),
                np.asarray(data["role_gid"]).astype(np.int64),
                role_feat.astype(np.float64)),
        },
        "window_dependence": {
            "dual_role_nodes": roles["dual_role_nodes"],
            "role_nodes": roles["role_nodes"],
            "max_abs_delta": roles["dual_role_max_abs_delta"],
            "max_elementwise_relative_delta":
                roles["dual_role_max_elementwise_relative_delta"],
        },
        "bands": dict(bands, deployed_floor=deployed_floor),
        # WHICH FUSION STAGES PRODUCED THE `deployed_prob_postblend` COLUMN. Bound here so a
        # consumer never has to infer it - and never from a log header (FACT-0407).
        "fusion": fusion,
        "contract_version": int(data["contract_version"]),
        "storage": storage,
        "feature_stats": {
            "dtype": str(role_feat.dtype),
            "min": float(role_feat.min()), "max": float(role_feat.max()),
            "mean": float(role_feat.mean()), "abs_max": float(np.abs(role_feat).max()),
        },
    }


def bind_crop_file(path: Path, deployed_floor: float,
                   expect_window: int | None = None) -> dict:
    return crop_binding(path.stem, load_cache(path), deployed_floor,
                        bytes_on_disk=path.stat().st_size, expect_window=expect_window)


def git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except Exception:                                            # pragma: no cover - no git
        return None


def state_dict_shape(path: Path) -> dict:
    """What the checkpoint can say about ITSELF - measured, and deliberately not much.

    Agent 4 established, and this re-measures, that every candidate trunk is a BARE state dict:
    136 tensors and ZERO non-tensor keys. ``official_f0`` and ``stabledet_f0`` even share a byte
    size. So this function exists to record that the checkpoint carries no provenance, not to
    extract any - the manifest is the only place provenance can live.
    """
    try:
        import torch  # noqa: PLC0415
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(sd, dict):
            return {"readable": True, "is_state_dict": False}
        tensors = sum(1 for v in sd.values() if torch.is_tensor(v))
        return {"readable": True, "is_state_dict": True, "keys": len(sd),
                "tensors": tensors, "non_tensor_keys": len(sd) - tensors,
                "carries_provenance": False}
    except Exception as err:
        return {"readable": False, "why": f"{type(err).__name__}: {err}"}


# --------------------------------------------------------------------------------------------
# bind
# --------------------------------------------------------------------------------------------
# One list, in the policy both guards read - a second copy here would be a new drift point of
# exactly the kind PKT-0043 exists to remove.
PLACEHOLDER_PROVENANCE = PP.PLACEHOLDER_PROVENANCE


def build_manifest(args) -> dict:
    cache_dir = Path(args.cache_dir)
    caches = sorted(cache_dir.glob("*.npz"))
    if not caches:
        raise Reject(f"{cache_dir} holds no *.npz cache - there is nothing to bind")

    trunk = Path(args.trunk)
    if not trunk.is_file():
        raise Reject(f"trunk checkpoint {trunk} is not on disk; its identity cannot be recorded")
    if args.trunk_role not in TRUNK_ROLES:
        raise Reject(f"unknown trunk role {args.trunk_role!r}; known: {sorted(TRUNK_ROLES)}")
    provenance = (args.trunk_provenance or "").strip()
    if provenance.lower() in PLACEHOLDER_PROVENANCE:
        raise Reject(
            f"trunk_provenance {args.trunk_provenance!r} is a placeholder. The checkpoint is a "
            "bare state dict with zero non-tensor keys and two candidate trunks share a byte "
            "size, so the manifest is the ONLY place the producing trunk can be named"
        )
    fold = str(args.fold).strip()
    if fold not in FOLD_EMBRYO:
        raise Reject(f"fold must be one of {sorted(FOLD_EMBRYO)}, got {fold!r}")

    crops = [bind_crop_file(p, args.deployed_floor) for p in caches]
    windows = {c["window"] for c in crops}
    if len(windows) != 1:
        raise Reject(f"the caches disagree about the window size: {sorted(windows)}")
    window = windows.pop()

    # ONE FUSION CONFIGURATION PER CACHE SET. The tap already refuses a configuration that
    # changes part way through a crop (assoc_feature_tap.py:268-274); this is the same statement
    # one level up. A directory mixing configurations holds two different
    # `deployed_prob_postblend` surfaces under one name, and a head trained across it would be
    # fed a column whose meaning changes by crop.
    fusions = {json.dumps(c["fusion"], sort_keys=True) for c in crops}
    if len(fusions) != 1:
        raise Reject(
            "the caches disagree about the fusion configuration: "
            f"{[json.loads(f)['stages_recorded'] for f in sorted(fusions)]}. Every crop in one "
            "manifest must have been produced under one deployment configuration, or the "
            "recorded deployed surfaces are not comparable"
        )
    fusion = crops[0]["fusion"]

    notebook = Path(args.notebook) if args.notebook else None
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "assoc_feature_cache_manifest",
        "packet": "PKT-0037",
        "layout": "pair_and_role",
        "cache_dir": str(cache_dir),
        "cache_contract_version": CACHE_CONTRACT_VERSION,
        # THE FUSION PROVENANCE, BOUND. `deployed_prob_postblend` is only meaningful beside the
        # list of stages that produced it: P36's verdict was invalid because a post-fusion column
        # was compared against a pre-fusion replay, and the stage was misattributed from a log
        # header (`FACT-0403`, corrected by `FACT-0407`). `reproducible_from_primary_cache` is
        # the consumer-facing consequence: false means the deployed surface cannot be rebuilt
        # from this cache alone, because a second model contributed to it.
        "fusion": fusion,
        "trunk": {
            "role": args.trunk_role,
            "path": str(trunk),
            "sha256": sha256_file(trunk),
            "bytes": trunk.stat().st_size,
            "provenance": provenance,
            "provenance_source": "manifest",
            "checkpoint_carries_no_provenance": True,
            "state_dict": state_dict_shape(trunk),
            # The cache-intrinsic half of the binding. A cache swapped between two trunk
            # directories keeps its manifest's trunk sha256 and stops matching this.
            "features_digest": _digest(
                *[np.frombuffer(bytes.fromhex(c["node_order"]["coord_feature_pair_digest"]),
                                dtype=np.uint8) for c in crops]
            ),
        },
        "fold": {
            "fold": fold,
            "held_out_embryo": FOLD_EMBRYO[fold],
            "weights_glob": args.weights_glob,
            "crops": [c["crop"] for c in crops],
        },
        "window_contract": {
            "window": window,
            "deployed_window": DEPLOYED_WINDOW,
            "pair_stride_frames": 1,
            "feature_key": "(pair, role)",
            "why": ("TemporalUNet3D._TemporalAttention mixes across the window's time axis, so "
                    "unet_out[:, i] depends on every frame in the window and a node's feature "
                    "belongs to the pair it was computed in (FACT-0402)"),
        },
        "feature_normalisation": {
            "transform": args.feature_normalisation,
            "applied_after": "model._index_features(unet_out[:, f_idx], coords, mask)",
            "note": (
                "'none' is the deployed contract: the head consumes _index_features output "
                "directly. Any other value must name code, because a head trained on a "
                "differently scaled cache is the wrong-features branch of an uninterpretable null."
            ),
        },
        "candidate_graph": {
            "rule": args.candidate_rule,
            "deployed_floor": args.deployed_floor,
            "acquisition_floor": args.acquisition_floor,
            "rank_cap": args.rank_cap,
            "gate_um": args.gate_um,
            "det_threshold": args.det_threshold,
            "pool_kernel_um": args.pool_kernel_um,
            "softmax_axis": args.softmax_axis,
            "abstain_mass": bool(args.abstain_mass),
        },
        "storage": {
            "bytes_on_disk": sum(c["storage"]["bytes_on_disk"] for c in crops),
            "uncompressed_bytes": sum(c["storage"]["uncompressed_bytes"] for c in crops),
            "crops": len(crops),
            "nodes": sum(c["nodes"] for c in crops),
            "role_nodes": sum(c["storage"]["role_nodes"] for c in crops),
            "band_rows": sum(c["storage"]["band_rows"] for c in crops),
            "union_rows": sum(c["storage"]["union_rows"] for c in crops),
        },
        "provenance": {
            "notebook": str(notebook) if notebook else None,
            "notebook_sha256": sha256_file(notebook) if notebook and notebook.is_file() else None,
            "spec": args.spec,
            "kernel": args.kernel,
            "kernel_version": args.kernel_version,
            "source_commit": args.commit or git_commit(Path(__file__).resolve().parents[2]),
        },
        "crops": crops,
    }


# --------------------------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------------------------
def audit(cache_dir: Path, manifest_path: Path, *, trunk: Path | None = None,
          expect_trunk_sha: str | None = None, expect_fold: str | None = None,
          expect_role: str | None = None) -> dict:
    """Re-derive the binding from what is on disk. Returns a report; raises Reject on failure."""
    if not manifest_path.is_file():
        raise Reject(
            f"no manifest at {manifest_path}. An unbound cache cannot distinguish 'the head does "
            "not work' from 'we fed it the wrong features', so it is not licensed for training."
        )
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if m.get("schema_version") != SCHEMA_VERSION:
        raise Reject(
            f"manifest schema {m.get('schema_version')!r} != {SCHEMA_VERSION}. Schema 1 keyed "
            "features BY FRAME, which FACT-0402 measured to be undefined at the deployed "
            "window; a schema-1 manifest cannot describe a pair-and-role cache."
        )
    if m.get("layout") != "pair_and_role":
        raise Reject(f"manifest layout {m.get('layout')!r} != 'pair_and_role'")
    if m.get("cache_contract_version") != CACHE_CONTRACT_VERSION:
        raise Reject(
            f"manifest cache_contract_version {m.get('cache_contract_version')!r} != "
            f"{CACHE_CONTRACT_VERSION}. A contract-1 manifest binds a cache whose single "
            "probability column named no surface, so its digests describe an artifact this "
            "auditor cannot interpret (FACT-0403, FACT-0407)."
        )
    if not isinstance(m.get("fusion"), dict):
        raise Reject(
            "the manifest binds no fusion record. `deployed_prob_postblend` is meaningless "
            "without the list of stages that produced it - reading a stage off a log header is "
            "how FACT-0403 misattributed its own root cause."
        )

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            raise Reject(f"{name}: {detail}")

    # --- 1. TRUNK, AND ITS PROVENANCE ----------------------------------------------------
    tr = m["trunk"]
    trunk_path = Path(trunk) if trunk else Path(tr["path"])
    if trunk_path.is_file():
        actual = sha256_file(trunk_path)
        check("trunk_checkpoint_sha256", actual == tr["sha256"],
              f"{trunk_path} hashes {actual[:16]}..., manifest records {tr['sha256'][:16]}...")
    else:
        check("trunk_checkpoint_present", False,
              f"trunk {trunk_path} is not on disk, so the cache's producing trunk cannot be "
              "re-verified - FACT-0392's third risk is exactly this")

    check("trunk_role_known", tr["role"] in TRUNK_ROLES, f"role {tr['role']!r}")
    check("trunk_provenance_is_explicit_in_the_manifest",
          bool((tr.get("provenance") or "").strip())
          and (tr.get("provenance") or "").strip().lower() not in PLACEHOLDER_PROVENANCE
          and tr.get("provenance_source") == "manifest",
          "trunk.provenance is empty or a placeholder. Every candidate trunk is a BARE state "
          "dict - 136 tensors, ZERO non-tensor keys - and official_f0 and stabledet_f0 share a "
          "byte size, so nothing in the checkpoint can name it. Provenance must be declared "
          "here or the cache cannot answer FACT-0392")
    sd = tr.get("state_dict") or {}
    check("trunk_checkpoint_is_acknowledged_to_carry_no_provenance",
          (not sd.get("is_state_dict")) or sd.get("carries_provenance") is False,
          "the manifest claims the checkpoint carries its own provenance; it does not - the "
          "state dict holds tensors only")
    if expect_role:
        check("trunk_role_matches_expectation", tr["role"] == expect_role,
              f"cache was produced by {tr['role']!r}, consumer expects {expect_role!r}")
    if expect_trunk_sha:
        check("trunk_sha_matches_expectation", tr["sha256"] == expect_trunk_sha,
              f"cache trunk {tr['sha256'][:16]}..., consumer expects {expect_trunk_sha[:16]}...")

    # --- 2. FOLD, EMBRYO AND WEIGHTS ------------------------------------------------------
    fold = str(m["fold"]["fold"])
    check("fold_known", fold in FOLD_EMBRYO, f"fold {fold!r}")
    expected_embryo = FOLD_EMBRYO[fold]
    check("fold_declares_the_right_embryo", m["fold"]["held_out_embryo"] == expected_embryo,
          f"fold {fold} holds out {expected_embryo}, manifest says "
          f"{m['fold']['held_out_embryo']!r}")
    if expect_fold is not None:
        check("fold_matches_expectation", fold == str(expect_fold),
              f"cache is fold {fold}, consumer expects fold {expect_fold}")

    glob = m["fold"].get("weights_glob") or ""
    if fold == "1":
        check("fold1_does_not_use_the_leaky_pack_weights",
              bool(glob) and PACK_WEIGHTS_TOKEN not in glob,
              f"fold 1 weights_glob {glob!r} - the pack ships only {PACK_WEIGHTS_TOKEN}, which "
              "was TRAINED ON 6bba, the very embryo fold 1 holds out. This is the EXP-0019 defect")
    else:
        check("fold0_does_not_load_split1_weights", "split_1" not in glob,
              f"fold 0 weights_glob {glob!r} loads split_1")

    # --- 3. THE WINDOW CONTRACT -----------------------------------------------------------
    wc = m.get("window_contract") or {}
    check("window_contract_declared", bool(wc.get("window")),
          "window_contract.window is absent; a cache whose window is unknown cannot be checked "
          "for the role duplication the window creates")
    check("window_matches_the_deployed_window", int(wc["window"]) == DEPLOYED_WINDOW,
          f"cache window {wc.get('window')} != deployed window {DEPLOYED_WINDOW} "
          "(unet_transformer split_0 config.json)")
    check("feature_key_is_pair_and_role", wc.get("feature_key") == "(pair, role)",
          f"window_contract.feature_key {wc.get('feature_key')!r} - a frame key is undefined "
          "at this window (FACT-0402)")

    # --- 4. THE CACHE ON DISK STILL IS THE CACHE THAT WAS BOUND ---------------------------
    on_disk = {p.stem: p for p in sorted(cache_dir.glob("*.npz"))}
    recorded = {c["crop"]: c for c in m["crops"]}
    check("crop_sets_agree", set(on_disk) == set(recorded),
          f"on disk {sorted(set(on_disk) - set(recorded))} unbound, manifest "
          f"{sorted(set(recorded) - set(on_disk))} absent")
    check("cache_is_not_empty", bool(on_disk), "no crop caches on disk")

    pair_digests = []
    measured = []
    for crop in sorted(on_disk):
        rec = recorded[crop]
        got = bind_crop_file(on_disk[crop], m["candidate_graph"]["deployed_floor"],
                             expect_window=int(wc["window"]))
        pair_digests.append(got["node_order"]["coord_feature_pair_digest"])
        measured.append(got)

        check(f"{crop}:embryo_matches_fold", got["embryo"] == expected_embryo,
              f"crop {crop} is embryo {got['embryo']}, but fold {fold} evaluates "
              f"{expected_embryo}")
        check(f"{crop}:node_order_unchanged",
              got["node_order"]["coords_digest"] == rec["node_order"]["coords_digest"],
              "the coordinate rows no longer digest to the bound value - the NODES HAVE BEEN "
              "REORDERED, so every positional id recorded against this crop (pre-ILP export, "
              "candidate surface, any trained head) now names a different cell")
        check(f"{crop}:role_table_unchanged",
              got["node_order"]["role_digest"] == rec["node_order"]["role_digest"],
              "the (pair, role, gid) table differs from the one that was bound; a feature row "
              "is now filed under a different pair or role than it was trained against")
        check(f"{crop}:features_unchanged",
              got["node_order"]["features_digest"] == rec["node_order"]["features_digest"],
              "the same nodes carry DIFFERENT FEATURE VALUES than were bound. On an unchanged "
              "node set that is a different producing trunk - FACT-0392's confound, where "
              "official-trunk features fed to a StableDet-trained head look exactly like a "
              "wrong feature contract")
        check(f"{crop}:positional_features_unchanged",
              got["node_order"]["positional_digest"] == rec["node_order"]["positional_digest"],
              "role_pos differs from the bound value; the head's positional input changed")
        check(f"{crop}:candidate_surface_unchanged",
              got["bands"]["surface_digest"] == rec["bands"]["surface_digest"],
              "the candidate surface differs from the one that was bound")
        check(f"{crop}:node_count_unchanged", got["nodes"] == rec["nodes"],
              f"{got['nodes']} nodes on disk, {rec['nodes']} bound")
        check(f"{crop}:role_node_count_unchanged",
              got["storage"]["role_nodes"] == rec["storage"]["role_nodes"],
              f"{got['storage']['role_nodes']} role rows on disk, "
              f"{rec['storage']['role_nodes']} bound")
        # Non-vacuity, restated at the manifest level. _validate_bands already refuses an empty
        # band; this makes the same statement a named CHECK so a report reader can see it fired.
        check(f"{crop}:deployed_band_is_not_empty", got["bands"]["band_a"]["rows"] > 0,
              "zero pairs above the deployed threshold: nothing in this crop can be checked "
              "against the deployed surface, so a pass here certifies nothing")
        check(f"{crop}:learnable_band_is_not_empty", got["bands"]["band_b"]["rows"] > 0,
              "zero pairs in the sub-threshold band. FACT-0382 measured that ALL 691 fold-0 "
              "contested errors have their true parent below the deployed threshold, so a cache "
              "with an empty band B holds none of the learnable population")
        check(f"{crop}:parity_band_is_not_empty", got["bands"]["band_p"]["rows"] > 0,
              "zero pairs in band P. Bands A and B are selected on the POST-fusion surface, so "
              "their membership belongs to the deployment and a primary-only replay may not "
              "re-derive it; band P is the only band whose membership carries the parity "
              "property, and an empty one leaves the gate nothing to check (FACT-0403)")
        check(f"{crop}:fusion_unchanged", got["fusion"] == rec.get("fusion"),
              f"the crop records fusion {got['fusion'].get('stages_recorded')!r} "
              f"(bidirectional {got['fusion'].get('bidirectional_weight')}, secondary "
              f"{got['fusion'].get('secondary_enabled')}) but was bound as "
              f"{(rec.get('fusion') or {}).get('stages_recorded')!r}. The stage list decides "
              "what deployed_prob_postblend MEANS, so a change to it changes the artifact")
        check(f"{crop}:fusion_matches_the_manifest", got["fusion"] == m["fusion"],
              "this crop's fusion configuration is not the one the manifest binds for the whole "
              "cache; a directory mixing configurations holds two different deployed surfaces "
              "under one column name")
        check(f"{crop}:features_are_window_dependent",
              got["window_dependence"]["dual_role_nodes"] == 0
              or got["window_dependence"]["max_abs_delta"] > 0.0,
              "every dual-role node carries identical features in both roles - frame-keyed "
              "content in a pair-and-role container (FACT-0402)")

    # --- 5. THE TRUNK <-> FEATURES BINDING ------------------------------------------------
    features_digest = _digest(*[np.frombuffer(bytes.fromhex(d), dtype=np.uint8)
                                for d in pair_digests])
    check("features_still_bound_to_the_recorded_trunk",
          features_digest == tr["features_digest"],
          "the features on disk do not digest to the value bound against trunk "
          f"{tr['role']} ({tr['sha256'][:16]}...). Either the cache was swapped between trunk "
          "directories or it was rewritten without rebinding - both are FACT-0392's confound")

    # --- 6. PROVENANCE COMPLETENESS -------------------------------------------------------
    prov = m.get("provenance", {})
    for field in ("notebook", "spec", "source_commit"):
        check(f"provenance_declares_{field}", bool(prov.get(field)),
              f"provenance.{field} is empty; the run that produced this cache is not identifiable")
    check("feature_normalisation_declared", bool(m["feature_normalisation"]["transform"]),
          "feature_normalisation.transform is empty")
    check("candidate_rule_declared", bool(m["candidate_graph"]["rule"]),
          "candidate_graph.rule is empty")

    return {"passed": True, "checks": checks, "crops": len(on_disk), "fold": fold,
            "trunk_role": tr["role"], "trunk_sha256": tr["sha256"],
            "fusion": m["fusion"],
            "measured": [{"crop": c["crop"], "nodes": c["nodes"],
                          "role_nodes": c["storage"]["role_nodes"],
                          "band_rows": c["storage"]["band_rows"],
                          "union_rows": c["storage"]["union_rows"],
                          "bytes_on_disk": c["storage"]["bytes_on_disk"],
                          "uncompressed_bytes": c["storage"]["uncompressed_bytes"],
                          "compression_ratio": c["storage"]["compression_ratio"],
                          "feature_dim": c["feature_dim"], "pos_dim": c["pos_dim"],
                          "band_b_topk": c["bands"]["band_b_topk"],
                          "edge_activation": c["bands"]["edge_activation"],
                          "dual_role_nodes": c["window_dependence"]["dual_role_nodes"],
                          "dual_role_max_abs_delta":
                              c["window_dependence"]["max_abs_delta"]}
                         for c in measured]}


def audit_dual_trunk(dir_a: Path, dir_b: Path, man_a: Path | None = None,
                     man_b: Path | None = None, *, require_roles: bool = False) -> dict:
    """Check that two sibling caches form a VALID DUAL-TRUNK PAIR.

    FACT-0392's third risk is that trunk identity is a command-line argument recorded nowhere in
    the artifact, so an official-trunk cache fed to a StableDet-trained head looks exactly like a
    wrong feature contract. The remedy is to cache both trunks and compare them - but only if the
    pair is well formed, and "well formed" has a precise, checkable meaning:

        SAME node set        identical coordinate AND role-table digests, crop for crop. Both
                             trunks must index ONE detector pass; otherwise trunk identity is
                             confounded with node identity and the pair cannot separate the two.
        DIFFERENT features   the feature digests must differ. Two caches that agree here are not
                             two trunks - they are one trunk written twice.
        SAME fold            comparing trunks across folds re-introduces the embryo confound.
        DIFFERENT trunks     distinct checkpoint hashes and distinct declared roles.

    Both caches must independently pass `audit` first. A pair of broken caches is not a pair.

    ``require_roles`` adds FOLD LEGITIMACY, and it reads ``provenance_policy`` - the SAME module
    ``gpu_protection_contract`` DATA-2/DATA-3 read. One arm must be readable as a result on this
    fold, the other is the declared comparison arm, and no leaky checkpoint may be present in
    either arm. The retired ``PKT0029_REQUIRED_TRUNK_ROLES`` asked a different question
    (`is the pair literally official+stabledet?`) and FACT-0418 measured that it answered the
    real one backwards.
    """
    man_a = man_a or dir_a / "cache_manifest.json"
    man_b = man_b or dir_b / "cache_manifest.json"
    rep_a, rep_b = audit(dir_a, man_a), audit(dir_b, man_b)
    ma = json.loads(man_a.read_text(encoding="utf-8"))
    mb = json.loads(man_b.read_text(encoding="utf-8"))

    if ma["trunk"]["sha256"] == mb["trunk"]["sha256"]:
        raise Reject(
            "dual_trunk_checkpoints_differ: both caches name the same trunk checkpoint, so this "
            "is one trunk written twice and resolves nothing about FACT-0392's ambiguity"
        )
    if ma["trunk"]["role"] == mb["trunk"]["role"]:
        raise Reject(f"dual_trunk_roles_differ: both caches declare role {ma['trunk']['role']!r}")
    if str(ma["fold"]["fold"]) != str(mb["fold"]["fold"]):
        raise Reject(
            f"dual_trunk_same_fold: fold {ma['fold']['fold']} against fold {mb['fold']['fold']}. "
            "Comparing trunks across folds confounds trunk identity with the embryo direction, "
            "which the contract requires be reported separately in the first place"
        )
    # THE SHARED POLICY, AND NOTHING LOCAL. Evaluated after the fold check, because a pair whose
    # two arms disagree about the fold has no fold to be legitimate ON.
    policy = None
    if require_roles:
        fold = str(ma["fold"]["fold"])
        refusals = PP.pair_refusals(
            fold, (ma["trunk"]["role"], mb["trunk"]["role"]),
            digests={ma["trunk"]["role"]: ma["trunk"].get("sha256"),
                     mb["trunk"]["role"]: mb["trunk"].get("sha256")})
        for m in (ma, mb):
            refusals += PP.role_binding_refusals(
                m["trunk"]["role"], m["trunk"].get("sha256"), fold=fold,
                embryo=m["fold"].get("held_out_embryo"),
                provenance=m["trunk"].get("provenance"))
        if refusals:
            raise Reject("dual_trunk_fold_legitimacy: " + "; ".join(dict.fromkeys(refusals)))
        policy = PP.describe(fold)

    a_crops = {c["crop"]: c for c in ma["crops"]}
    b_crops = {c["crop"]: c for c in mb["crops"]}
    if set(a_crops) != set(b_crops):
        raise Reject(
            "dual_trunk_same_crops: the two caches cover different crops "
            f"({sorted(set(a_crops) ^ set(b_crops))[:5]}), so any difference between them is "
            "confounded with which crops each one saw"
        )
    for crop in sorted(a_crops):
        an, bn = a_crops[crop]["node_order"], b_crops[crop]["node_order"]
        if an["coords_digest"] != bn["coords_digest"]:
            raise Reject(
                f"dual_trunk_shares_the_node_set: {crop} has different coordinates in the two "
                "caches. Both trunks must index ONE detector pass - otherwise trunk identity is "
                "confounded with node identity and the pair cannot separate them"
            )
        if an["role_digest"] != bn["role_digest"]:
            raise Reject(
                f"dual_trunk_shares_the_role_table: {crop} pairs its nodes into different "
                "(pair, role) blocks in the two caches, so a per-role comparison would compare "
                "different objects"
            )
        if an["features_digest"] == bn["features_digest"]:
            raise Reject(
                f"dual_trunk_features_differ: {crop} has byte-identical features under two "
                "different checkpoints. Either the second trunk was never loaded or the cache "
                "was copied - either way the pair proves nothing"
            )
    return {"passed": True, "fold": str(ma["fold"]["fold"]),
            "roles": [ma["trunk"]["role"], mb["trunk"]["role"]],
            "crops": len(a_crops), "a": rep_a["trunk_sha256"], "b": rep_b["trunk_sha256"],
            # Recorded rather than implied: a receipt that does not say WHICH policy passed the
            # pair cannot be re-checked against the policy that has since moved.
            "fold_legitimacy_policy": policy}


# --------------------------------------------------------------------------------------------
# full-fold projection, with a HARD CAPACITY GUARD
# --------------------------------------------------------------------------------------------
def schema_bytes_per_node(feat_dim: int, pos_dim: int, band_b_topk: int) -> dict:
    """The WORST-CASE uncompressed bytes one detected node can contribute, by the schema.

    Every term is a bound that follows from the deployed loop, not a fitted rate:

    role rows       <= 2 per node. Windows slide with stride W-1 = 1, so a node's frame is the
                    source of at most one pair and the target of at most one
                    (predict_unet_transformer.py:352-359, seen_pairs at :558).
    band A rows     <= 1 per TARGET node. probs = softmax over the SOURCE axis
                    (dim=0, :590-600), so each column sums to 1 and at most one entry per column
                    can exceed a threshold >= 0.5. This is the arithmetic behind FACT-0369.
    band B rows     <= band_b_topk per TARGET node, by the tap's own rank cap.
    band P rows     <= 1 per TARGET node, by the SAME softmax argument - band P applies the same
                    threshold to `primary_probs_preblend`, which is the source-axis activation of
                    the pre-fusion logits and therefore also sums to 1 down each column. This is
                    the contract-2 term; contract 1 had no third band.
    union rows      = band A + band B rows. Band P is deliberately NOT in the union.
    coords          exactly 1 row per node.
    pair + partition rows  <= 1 per node, since pairs <= frames <= nodes.
    """
    role = ROLE_FIXED_BYTES + 4 * int(feat_dim) + 4 * int(pos_dim)
    band_rows = 1 + int(band_b_topk) + 1          # A + B + P
    union_rows = 1 + int(band_b_topk)             # A + B
    band_bytes = band_rows * BAND_ROW_BYTES + union_rows * UNION_ROW_BYTES
    return {
        "role_bytes": 2 * role,
        "band_bytes": band_bytes,
        "node_side_bytes": NODE_SIDE_BYTES,
        "total": 2 * role + band_bytes + NODE_SIDE_BYTES,
        "per_role_row_bytes": role,
        "per_band_row_bytes": BAND_ROW_BYTES,
        "per_union_row_bytes": UNION_ROW_BYTES,
        "band_rows_per_node": band_rows,
        "union_rows_per_node": union_rows,
    }


def read_node_counts(path: Path, column: str = "raw_nodes") -> dict[str, int]:
    """Per-crop detector node counts from a pre-ILP run_stats.csv - a MEASURED full-fold table."""
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise Reject(f"{path} holds no rows")
    if "dataset" not in rows[0] or column not in rows[0]:
        raise Reject(f"{path} has no 'dataset'/{column!r} columns; got {list(rows[0])[:6]}")
    return {r["dataset"]: int(r[column]) for r in rows}


def project_full_fold(measured: list[dict], node_counts: dict[str, int], *,
                      capacity_bytes: int, feat_dim: int | None = None,
                      pos_dim: int | None = None, band_b_topk: int | None = None,
                      label: str = "fold") -> dict:
    """Project full-fold storage from MEASURED caches and REFUSE if it will not fit.

    The binding number is the WORST CASE with NO compression credit. The measured compression
    ratio is reported beside it and is deliberately not allowed to buy headroom: a cache that
    only fits if float32 features compress as well as a smoke fixture's did is a cache that
    fails at 03:00 on a GPU session, and this guard exists to stop exactly that.
    """
    if not measured:
        raise Reject("projection_has_no_measurement: no audited cache was supplied, so every "
                     "byte figure would be an estimate. Measure a real cache first")
    if not node_counts:
        raise Reject("projection_has_no_node_counts: supply the fold's per-crop node counts")
    if capacity_bytes is None or int(capacity_bytes) <= 0:
        raise Reject("projection_has_no_capacity: state the byte budget explicitly. There is no "
                     "default, because a guard with a guessed budget is a warning")

    activations = {m["edge_activation"] for m in measured}
    if activations != {"softmax"}:
        raise Reject(
            f"projection_bound_does_not_hold: edge_activation {sorted(activations)}. The band-A "
            "bound of one row per target follows from the source-axis softmax summing to 1; "
            "under any other activation band A is unbounded and this projection is invalid"
        )
    feat_dims = {m["feature_dim"] for m in measured}
    pos_dims = {m["pos_dim"] for m in measured}
    topks = {m["band_b_topk"] for m in measured}
    if feat_dim is None:
        if len(feat_dims) != 1:
            raise Reject(f"projection_feat_dim_ambiguous: measured caches carry {sorted(feat_dims)}")
        feat_dim = feat_dims.pop()
    if pos_dim is None:
        if len(pos_dims) != 1:
            raise Reject(f"projection_pos_dim_ambiguous: measured caches carry {sorted(pos_dims)}")
        pos_dim = pos_dims.pop()
    if band_b_topk is None:
        if len(topks) != 1:
            raise Reject(f"projection_topk_ambiguous: measured caches carry {sorted(topks)}")
        band_b_topk = topks.pop()

    per_node = schema_bytes_per_node(feat_dim, pos_dim, band_b_topk)
    total_nodes = int(sum(node_counts.values()))
    n_crops = len(node_counts)

    # MEASURED, from the real artifacts only.
    m_disk = int(sum(m["bytes_on_disk"] for m in measured))
    m_unc = int(sum(m["uncompressed_bytes"] for m in measured))
    m_nodes = int(sum(m["nodes"] for m in measured))
    m_role = int(sum(m["role_nodes"] for m in measured))
    m_band = int(sum(m["band_rows"] for m in measured))
    m_union = int(sum(m["union_rows"] for m in measured))
    ratio = m_disk / m_unc if m_unc else 1.0
    # Per-crop fixed overhead: whatever the on-disk file costs beyond its variable rows. Taken as
    # the WORST observed, and floored at zero so a well-compressing fixture cannot buy credit.
    overheads = []
    for m in measured:
        variable = (m["role_nodes"] * (ROLE_FIXED_BYTES + 4 * m["feature_dim"] + 4 * m["pos_dim"])
                    + m["band_rows"] * BAND_ROW_BYTES + m["union_rows"] * UNION_ROW_BYTES
                    + m["nodes"] * NODE_SIDE_BYTES)
        overheads.append(max(int(m["uncompressed_bytes"]) - variable, 0))
    overhead = max(overheads)

    worst_case = total_nodes * per_node["total"] + n_crops * overhead
    expected = worst_case * ratio
    fits = worst_case <= int(capacity_bytes)

    report = {
        "label": label,
        "capacity_bytes": int(capacity_bytes),
        "crops": n_crops,
        "total_nodes": total_nodes,
        "feat_dim": int(feat_dim), "pos_dim": int(pos_dim), "band_b_topk": int(band_b_topk),
        "measured": {
            "caches": len(measured),
            "crops": [m["crop"] for m in measured],
            "bytes_on_disk": m_disk,
            "uncompressed_bytes": m_unc,
            "compression_ratio": ratio,
            "nodes": m_nodes, "role_nodes": m_role, "band_rows": m_band,
            "union_rows": m_union,
            "bytes_on_disk_per_node": m_disk / m_nodes if m_nodes else None,
            "bytes_on_disk_per_role_node": m_disk / m_role if m_role else None,
            "bytes_on_disk_per_band_row": m_disk / m_band if m_band else None,
            "role_nodes_per_node": m_role / m_nodes if m_nodes else None,
            "band_rows_per_node": m_band / m_nodes if m_nodes else None,
            "per_crop_fixed_overhead_bytes": overhead,
        },
        "schema_bytes_per_node": per_node,
        "worst_case_bytes": int(worst_case),
        "worst_case_gib": worst_case / float(1 << 30),
        "expected_bytes_at_measured_compression": int(expected),
        "expected_gib_at_measured_compression": expected / float(1 << 30),
        "headroom_bytes": int(capacity_bytes) - int(worst_case),
        "fits": bool(fits),
    }
    if not fits:
        raise Reject(
            f"CAPACITY: {label} projects {worst_case / float(1 << 30):.2f} GiB worst case "
            f"({total_nodes} nodes x {per_node['total']} B/node + {n_crops} x {overhead} B "
            f"overhead) against a stated budget of {int(capacity_bytes) / float(1 << 30):.2f} "
            f"GiB. REFUSED - the run would fill the disk mid-session and the GPU spend would "
            f"buy a truncated cache. Reduce the crop set, lower band_b_topk, or raise the budget "
            f"deliberately."
        )
    return report


# --------------------------------------------------------------------------------------------
# self-test: production-layout fixtures, each mutation manufactured
# --------------------------------------------------------------------------------------------
def _synth_cache(path: Path, *, crop: str, trunk_seed: int, n_frames: int = 4,
                 n_per_frame: int = 5, feat_dim: int = 32, pos_dim: int = 32,
                 window: int = DEPLOYED_WINDOW, band_b_topk: int = 8,
                 band_b_floor: float = 0.02, threshold: float = 0.5,
                 downsample=(1, 4, 4), spatial=(8, 32, 32),
                 reorder: bool = False, empty_sub_band: bool = False,
                 frame_keyed: bool = False, dedupe_roles: bool = False,
                 duplicate_role: bool = False, frame_keyed_features: bool = False,
                 bad_mask: bool = False, torn_scaled: bool = False,
                 wrong_relative_time: bool = False, band_b_over_cap: bool = False,
                 torn_union: bool = False, reordered_roles: bool = False,
                 mutate=None) -> None:
    """Manufacture ONE crop cache in the tap's exact production layout.

    `mutate` receives the finished payload dict and may edit it in place. The contract-2
    conditions are one-line payload edits - a mislabelled `selected_on`, a fusion record that
    contradicts its own weights - so a hook is a smaller and more honest surface for them than
    another seven boolean parameters.

    The coordinate grid is a property of the DETECTOR and is identical across trunks here; only
    the features depend on `trunk_seed`. That makes the swapped-trunk mutation a pure feature
    change on an unchanged node set, which is the shape FACT-0392 warns about and the shape a
    node-count check cannot see.
    """
    rng = np.random.default_rng(trunk_seed)
    down = np.asarray(downsample, dtype=np.int64)
    grid = np.asarray(spatial, dtype=np.int64)

    coords, starts, ends = [], [], []
    total = 0
    for t in range(n_frames):
        arr = np.stack([
            np.full(n_per_frame, t),
            np.arange(n_per_frame) % grid[0],
            (np.arange(n_per_frame) * 2) % grid[1],
            (np.arange(n_per_frame) * 3) % grid[2],
        ], axis=1).astype(np.int16)
        starts.append(total); total += n_per_frame; ends.append(total)
        coords.append(arr)
    coords_arr = np.concatenate(coords)
    if reorder:
        # A REORDER that leaves the frame partition, the node COUNT and every feature value
        # untouched: within the first frame, reverse the coordinate rows.
        block = coords_arr[starts[0]:ends[0]][::-1].copy()
        coords_arr = coords_arr.copy()
        coords_arr[starts[0]:ends[0]] = block

    if frame_keyed:
        # THE RETIRED SCHEMA. One feature vector per node, keyed by frame - the layout FACT-0402
        # proved undefined at the deployed window.
        src, tgt, prob = [], [], []
        for t in range(n_frames - 1):
            for i in range(n_per_frame):
                for j in range(n_per_frame):
                    src.append(starts[t] + i); tgt.append(starts[t + 1] + j)
                    prob.append(0.9 if i == j else 0.05)
        np.savez_compressed(
            path, coords=coords_arr, frames=np.arange(n_frames, dtype=np.int64),
            starts=np.asarray(starts, dtype=np.int64), ends=np.asarray(ends, dtype=np.int64),
            feat_frames=np.arange(n_frames, dtype=np.int64),
            source_id=np.asarray(src, dtype=np.int64),
            target_id=np.asarray(tgt, dtype=np.int64),
            edge_prob=np.asarray(prob, dtype=np.float32),
            **{f"feat_{t}": rng.normal(size=(n_per_frame, feat_dim)).astype(np.float32)
               for t in range(n_frames)})
        return

    # One feature vector PER FRAME, as a frame-keyed writer would hold them. The faithful path
    # perturbs the target-role copy, because a temporal-attention trunk cannot return the same
    # vector from two different windows; `frame_keyed_features` skips that perturbation, which
    # is defect 6 wearing the new schema.
    per_frame = {t: rng.normal(size=(n_per_frame, feat_dim)).astype(np.float32)
                 for t in range(n_frames)}
    tgt_shift = {t: (per_frame[t] + rng.normal(scale=0.01, size=per_frame[t].shape)
                     ).astype(np.float32) for t in range(n_frames)}

    r_pair, r_role, r_gid, r_feat, r_scaled, r_rel, r_mask, r_pos = [], [], [], [], [], [], [], []
    p_f, p_ts, p_tt, p_sp, p_sn, p_tp, p_tn, p_ws = [], [], [], [], [], [], [], []
    cursor = 0
    if dedupe_roles:
        # DEFECT 6, PART TWO, IN ITS REALISTIC FORM. A frame-keyed writer wearing the
        # pair-and-role schema emits ONE block per FRAME and points both roles at it, so an
        # interior node carries a single vector where the window makes two. Every array is
        # well formed; only the multiplicity betrays it.
        for frame in range(n_frames):
            gid = np.arange(starts[frame], ends[frame], dtype=np.int64)
            rel = coords_arr[starts[frame]:ends[frame]].astype(np.int32).copy()
            rel[:, 0] = 0
            r_pair.append(np.full(gid.shape[0], min(frame, n_frames - 2), dtype=np.int64))
            r_role.append(np.zeros(gid.shape[0], dtype=np.int8))
            r_gid.append(gid)
            r_feat.append(per_frame[frame].astype(np.float32))
            r_scaled.append(rel[:, 1:].astype(np.float32) * down.astype(np.float32))
            r_rel.append(rel.astype(np.int32))
            r_mask.append(np.ones(gid.shape[0], dtype=bool))
            r_pos.append(rng.normal(size=(gid.shape[0], pos_dim)).astype(np.float32))
        for t in range(n_frames - 1):
            p_f.append(0); p_ts.append(t); p_tt.append(t + 1)
            p_sp.append(int(starts[t])); p_sn.append(n_per_frame)
            p_tp.append(int(starts[t + 1])); p_tn.append(n_per_frame)
            p_ws.append((window, int(grid[0]), int(grid[1]), int(grid[2])))
        cursor = int(ends[-1])
    for pair, t in (() if dedupe_roles else enumerate(range(n_frames - 1))):
        for role, frame in ((0, t), (1, t + 1)):
            gid = np.arange(starts[frame], ends[frame], dtype=np.int64)
            base = per_frame[frame] if role == 0 else (
                per_frame[frame] if frame_keyed_features else tgt_shift[frame])
            rel = coords_arr[starts[frame]:ends[frame]].astype(np.int32).copy()
            rel[:, 0] = role                      # f_idx is 0 at window 2, so src=0 tgt=1
            if wrong_relative_time and role == 1 and pair == 0:
                rel[:, 0] = 0
            scaled = rel[:, 1:].astype(np.float32) * down.astype(np.float32)
            if torn_scaled and pair == 0 and role == 0:
                scaled = scaled + 1.0
            mask = np.ones(gid.shape[0], dtype=bool)
            if bad_mask and pair == 0 and role == 0:
                mask[0] = False
            n = gid.shape[0]
            r_pair.append(np.full(n, pair, dtype=np.int64))
            r_role.append(np.full(n, role, dtype=np.int8))
            r_gid.append(gid)
            r_feat.append(base.astype(np.float32))
            r_scaled.append(scaled.astype(np.float32))
            r_rel.append(rel.astype(np.int32))
            r_mask.append(mask)
            r_pos.append(rng.normal(size=(n, pos_dim)).astype(np.float32))
            if role == 0:
                p_sp.append(cursor); p_sn.append(n)
            else:
                p_tp.append(cursor); p_tn.append(n)
            cursor += n
        p_f.append(0); p_ts.append(t); p_tt.append(t + 1)
        p_ws.append((window, int(grid[0]), int(grid[1]), int(grid[2])))

    if duplicate_role:
        # A block appended twice: the pair table still points at the first copy, so the extra
        # rows sit outside every block AND inflate a node's multiplicity.
        for lst in (r_pair, r_role, r_gid, r_feat, r_scaled, r_rel, r_mask, r_pos):
            lst.append(lst[0].copy())
    if reordered_roles:
        # Permute the role rows of one block without touching the pair table.
        r_gid[0] = r_gid[0][::-1].copy()

    def cat(parts, dtype):
        return np.concatenate(parts).astype(dtype)

    role_feat = cat(r_feat, np.float32)
    n_role = int(role_feat.shape[0])

    # Bands, built to satisfy the tap's own definitions.
    a_pair, a_i, a_j, a_p = [], [], [], []
    b_pair, b_i, b_j, b_p = [], [], [], []
    for pair in range(len(p_f)):
        ns, nt = p_sn[pair], p_tn[pair]
        if ns == 0 or nt == 0:
            continue
        for j in range(nt):
            if empty_sub_band:
                # every recorded pair above the deployed threshold: band B is empty
                for i in range(ns):
                    a_pair.append(pair); a_i.append(i); a_j.append(j); a_p.append(0.90)
                continue
            a_pair.append(pair); a_i.append(j % ns); a_j.append(j); a_p.append(0.90)
            # `band_b_over_cap` is what an UNCAPPED writer emits: every source above the floor,
            # rather than the tap's top-k per target.
            limit = ns if band_b_over_cap else min(band_b_topk + 1, ns)
            for r in range(limit):
                if r == j % ns:
                    continue
                b_pair.append(pair); b_i.append(r); b_j.append(j)
                b_p.append(round(0.05 + 0.01 * r, 4))

    def band_arrays(pair_l, i_l, j_l, p_l):
        pair_a = np.asarray(pair_l, dtype=np.int64)
        i_a = np.asarray(i_l, dtype=np.int64)
        j_a = np.asarray(j_l, dtype=np.int64)
        gid_all = cat(r_gid, np.int64)
        src = gid_all[np.asarray([p_sp[p] for p in pair_l], dtype=np.int64) + i_a] \
            if len(pair_l) else np.empty(0, dtype=np.int64)
        tgt = gid_all[np.asarray([p_tp[p] for p in pair_l], dtype=np.int64) + j_a] \
            if len(pair_l) else np.empty(0, dtype=np.int64)
        prob = np.asarray(p_l, dtype=np.float64)
        # NO FUSION STAGE RUNS IN THIS FIXTURE, so the pre- and post-fusion surfaces are the SAME
        # ARRAY - which is what the real tap emits with no stage, measured at exactly 0.0. The
        # logit is log(p): monotone in the probability, which is all the rank check requires.
        logit = np.log(np.clip(prob, 1e-12, None))
        return pair_a, src, tgt, i_a, j_a, logit, prob, prob.copy()

    a = band_arrays(a_pair, a_i, a_j, a_p)
    b = band_arrays(b_pair, b_i, b_j, b_p)
    # BAND P: the same threshold rule on the PRE-fusion surface. With no fusion stage the two
    # surfaces are the same numbers, so band P's membership is band A's - the property
    # _validate_bands checks and the property FACT-0407 used as its control arm.
    p = band_arrays(a_pair, a_i, a_j, a_p)

    union_src = np.concatenate([a[1], b[1]])
    union_tgt = np.concatenate([a[2], b[2]])
    union_post = np.concatenate([a[7], b[7]]).astype(np.float32)
    union_pre = np.concatenate([a[6], b[6]]).astype(np.float32)
    if torn_union:
        union_post = union_post[::-1].copy()

    payload = {
        "schema_version": np.int64(SCHEMA_VERSION),
        "contract_version": np.int64(CACHE_CONTRACT_VERSION),
        "primary_surface": np.str_("primary: model.predict_edges forward output, before fusion"),
        "deployed_surface": np.str_("deployed: probs after every fusion stage that ran"),
        # NO STAGE RAN. A synthetic fixture must not claim one, because the claim is what
        # _validate_bands checks the two surface columns against.
        "fusion_stages": np.str_("none"),
        "fusion_bidirectional_weight": np.float64(0.0),
        "fusion_secondary_enabled": np.bool_(False),
        "fusion_secondary_edge_weight": np.float64(0.0),
        "fusion_secondary_link_mode": np.str_(""),
        "fusion_secondary_mix_temperature": np.float64(1.0),
        "fusion_reproducible_from_primary_cache": np.bool_(True),
        "crop": np.str_(crop),
        "window": np.int64(window),
        "downsample": down,
        "voxel_size": np.asarray([1.625, 0.40625 * down[1], 0.40625 * down[2]], dtype=np.float64),
        "pool_kernel": np.asarray([1, 3, 3], dtype=np.int64),
        "q_low": np.float64(50.0), "q_high": np.float64(950.0),
        "image_shape": np.asarray([n_frames, *grid.tolist()], dtype=np.int64),
        "det_threshold": np.float64(0.5), "det_tta": np.bool_(False),
        "edge_threshold": np.float64(threshold), "edge_activation": np.str_("softmax"),
        "band_b_floor": np.float64(band_b_floor), "band_b_topk": np.int64(band_b_topk),
        "feat_dtype": np.str_("float32"), "feat_dim": np.int64(feat_dim),
        "pos_dim": np.int64(pos_dim), "node_count": np.int64(len(coords_arr)),
        "coords": coords_arr,
        "frames": np.arange(n_frames, dtype=np.int64),
        "starts": np.asarray(starts, dtype=np.int64),
        "ends": np.asarray(ends, dtype=np.int64),
        "pair_f_idx": np.asarray(p_f, dtype=np.int64),
        "pair_t_src": np.asarray(p_ts, dtype=np.int64),
        "pair_t_tgt": np.asarray(p_tt, dtype=np.int64),
        "pair_src_ptr": np.asarray(p_sp, dtype=np.int64),
        "pair_src_n": np.asarray(p_sn, dtype=np.int64),
        "pair_tgt_ptr": np.asarray(p_tp, dtype=np.int64),
        "pair_tgt_n": np.asarray(p_tn, dtype=np.int64),
        "pair_window_shape": np.asarray(p_ws, dtype=np.int64),
        "role_pair": cat(r_pair, np.int64), "role_role": cat(r_role, np.int8),
        "role_gid": cat(r_gid, np.int64), "role_feat": role_feat,
        "role_coord_scaled": cat(r_scaled, np.float32),
        "role_coord_rel": cat(r_rel, np.int32),
        "role_mask": cat(r_mask, bool), "role_pos": cat(r_pos, np.float32),
        "source_id": union_src, "target_id": union_tgt,
        "deployed_edge_prob_postblend": union_post,
        "primary_edge_prob_preblend": union_pre,
    }
    for name, cols in ((BAND_A, a), (BAND_B, b), (BAND_P, p)):
        payload[f"band_{name}_selected_on"] = np.str_(BAND_SURFACE[name])
        payload[f"band_{name}_pair"] = cols[0]
        payload[f"band_{name}_source_id"] = cols[1]
        payload[f"band_{name}_target_id"] = cols[2]
        payload[f"band_{name}_i"] = cols[3]
        payload[f"band_{name}_j"] = cols[4]
        payload[f"band_{name}_primary_logit_preblend"] = cols[5]
        payload[f"band_{name}_primary_prob_preblend"] = cols[6]
        payload[f"band_{name}_deployed_prob_postblend"] = cols[7]
    assert n_role == payload["role_pair"].shape[0]
    assert set(payload) == set(REQUIRED_KEYS) | {POSITIONAL_KEY}, (
        "the self-test fixture no longer emits the schema it is meant to exercise: extra "
        f"{sorted(set(payload) - set(REQUIRED_KEYS) - {POSITIONAL_KEY})}, missing "
        f"{sorted(set(REQUIRED_KEYS) - set(payload))}"
    )
    if mutate is not None:
        mutate(payload)
    np.savez_compressed(path, **payload)


def _bind_args(cache_dir: Path, trunk: Path, *, fold: str, role: str, weights_glob: str,
               notebook: Path, provenance: str | None = None):
    return argparse.Namespace(
        cache_dir=str(cache_dir), trunk=str(trunk), trunk_role=role,
        trunk_provenance=provenance or (
            f"synthetic self-test trunk for role {role}; named here because the checkpoint is a "
            "bare state dict with zero non-tensor keys"),
        fold=fold, weights_glob=weights_glob,
        feature_normalisation="none",
        candidate_rule="softmax over source axis; band A > 0.5, band B (0.02, 0.5] top-8/target",
        deployed_floor=0.5, acquisition_floor=0.02, rank_cap=8, gate_um=None,
        det_threshold=0.96875, pool_kernel_um=3.0, softmax_axis="source", abstain_mass=False,
        notebook=str(notebook), spec="selftest_spec", kernel=None, kernel_version=None,
        commit="0" * 40,
    )


def self_test(out: Path | None) -> int:
    results = []

    def record(name: str, expect_reject: bool, fn, want: str | None = None) -> None:
        try:
            fn()
            outcome, detail = "accepted", ""
        except Reject as err:
            outcome, detail = "rejected", str(err)
        ok = (outcome == "rejected") if expect_reject else (outcome == "accepted")
        if ok and expect_reject and want:
            ok = want in detail
            if not ok:
                detail = f"WRONG CONDITION (wanted {want!r}): {detail}"
        results.append({"mutation": name, "expected": "reject" if expect_reject else "accept",
                        "outcome": outcome, "ok": ok, "condition": want, "detail": detail[:400]})
        print(f"  {'PASS' if ok else 'FAIL'}  {name:38s} expected="
              f"{'reject' if expect_reject else 'accept':8s} got={outcome}")
        if detail and expect_reject:
            print(f"          -> {detail[:180]}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        nb = root / "notebook.ipynb"; nb.write_text('{"cells": []}', encoding="utf-8")
        trunk_a = root / "trunk_official.pth"; trunk_a.write_bytes(b"OFFICIAL-TRUNK-BYTES" * 64)
        trunk_b = root / "trunk_stabledet.pth"; trunk_b.write_bytes(b"STABLEDET-TRUNK-BYTE" * 64)
        crop0 = "44b6_aaaaaaaa"

        def build(name: str, **kw) -> Path:
            d = root / name; d.mkdir(exist_ok=True)
            _synth_cache(d / f"{kw.pop('crop_name', crop0)}.npz", crop=crop0, **kw)
            return d

        def manifest_for(d: Path, trunk: Path, *, fold="0", role="official",
                         glob="/kaggle/input/*/split_0/edge_predictor_best.pth",
                         name="cache_manifest.json", provenance=None) -> Path:
            p = d / name
            p.write_text(json.dumps(build_manifest(
                _bind_args(d, trunk, fold=fold, role=role, weights_glob=glob, notebook=nb,
                           provenance=provenance)), indent=2), encoding="utf-8")
            return p

        clean = build("clean", trunk_seed=1)
        man = manifest_for(clean, trunk_a)

        print("\nSELF-TEST: accept controls first, then every manufactured defect\n")
        # --- ACCEPT CONTROLS -----------------------------------------------------------------
        record("control_clean_cache", False, lambda: audit(clean, man))

        # --- DEFECT 6, ALL THREE FORMS -------------------------------------------------------
        fk = build("frame_keyed", trunk_seed=1, frame_keyed=True)
        record("defect6_frame_keyed_cache", True,
               lambda: bind_crop_file(fk / f"{crop0}.npz", 0.5), want="frame_keyed_cache")

        dedup = build("dedupe_roles", trunk_seed=1, n_frames=5, dedupe_roles=True)
        record("defect6_role_records_missing", True,
               lambda: bind_crop_file(dedup / f"{crop0}.npz", 0.5), want="role_records_missing")

        dup = build("duplicate_role", trunk_seed=1, duplicate_role=True)
        record("defect6_duplicate_role_record", True,
               lambda: bind_crop_file(dup / f"{crop0}.npz", 0.5),
               want="duplicate_role_record")

        fkf = build("frame_keyed_features", trunk_seed=1, frame_keyed_features=True)
        record("defect6_frame_keyed_features_in_a_pair_and_role_container", True,
               lambda: bind_crop_file(fkf / f"{crop0}.npz", 0.5),
               want="role_features_frame_keyed")

        # --- STRUCTURE -----------------------------------------------------------------------
        ro = build("reordered_roles", trunk_seed=1, reordered_roles=True)
        record("reordered_role_rows", True,
               lambda: bind_crop_file(ro / f"{crop0}.npz", 0.5),
               want="role_node_order_is_not_the_deployed_frame_slice")

        bm = build("bad_mask", trunk_seed=1, bad_mask=True)
        record("padded_mask_slot", True, lambda: bind_crop_file(bm / f"{crop0}.npz", 0.5),
               want="role_mask_has_a_false_slot")

        ts = build("torn_scaled", trunk_seed=1, torn_scaled=True)
        record("torn_scaled_coordinates", True, lambda: bind_crop_file(ts / f"{crop0}.npz", 0.5),
               want="role_coord_scaled_disagrees_with_role_coord_rel")

        wrt = build("wrong_rel_time", trunk_seed=1, wrong_relative_time=True)
        record("role_filed_under_the_wrong_relative_time", True,
               lambda: bind_crop_file(wrt / f"{crop0}.npz", 0.5), want="role_relative_time_wrong")

        # --- BANDS ---------------------------------------------------------------------------
        eb = build("empty_band", trunk_seed=1, empty_sub_band=True)
        record("empty_learnable_band", True, lambda: bind_crop_file(eb / f"{crop0}.npz", 0.5),
               want="band_b_is_empty")

        oc = build("band_b_over_cap", trunk_seed=1, n_per_frame=12, band_b_over_cap=True)
        record("band_b_over_its_rank_cap", True, lambda: bind_crop_file(oc / f"{crop0}.npz", 0.5),
               want="band_b_exceeds_its_rank_cap")

        tu = build("torn_union", trunk_seed=1, torn_union=True)
        record("union_surface_torn_from_its_bands", True,
               lambda: bind_crop_file(tu / f"{crop0}.npz", 0.5),
               want="union_surface_disagrees_with_its_bands")

        # --- CONTRACT 2: THE NAMED SURFACES AND THE FUSION RECORD -----------------------------
        # Every one of these is the FACT-0403 failure in a form contract 1 could not express:
        # a probability whose surface is misdeclared, or a fusion record that does not describe
        # what produced the column beside it.
        def _contract1(payload):
            """Down-convert to contract 1: one unqualified probability per band."""
            for band in BANDS:
                for suffix in ("selected_on", "primary_logit_preblend", "primary_prob_preblend"):
                    payload.pop(f"band_{band}_{suffix}")
                payload[f"band_{band}_prob"] = payload.pop(f"band_{band}_deployed_prob_postblend")
            for key in ("schema_version", "contract_version", "primary_surface",
                        "deployed_surface", "primary_edge_prob_preblend"):
                payload.pop(key)
            for key in [k for k in payload if k.startswith("fusion_")]:
                payload.pop(key)
            payload["edge_prob"] = payload.pop("deployed_edge_prob_postblend")

        c1 = build("contract_1", trunk_seed=1, mutate=_contract1)
        record("contract_1_cache_is_refused_by_name", True,
               lambda: bind_crop_file(c1 / f"{crop0}.npz", 0.5), want="cache_contract_1")

        ver = build("contract_3", trunk_seed=1,
                    mutate=lambda p: p.__setitem__("contract_version", np.int64(3)))
        record("unknown_contract_version_is_refused", True,
               lambda: bind_crop_file(ver / f"{crop0}.npz", 0.5),
               want="cache_contract_version_unknown")

        mis = build("mislabelled_surface", trunk_seed=1,
                    mutate=lambda p: p.__setitem__(
                        "band_p_selected_on", np.str_("deployed_probs_postblend")))
        record("band_p_declares_the_wrong_selection_surface", True,
               lambda: bind_crop_file(mis / f"{crop0}.npz", 0.5),
               want="band_p_selected_on_is_wrong")

        def _no_parity_band(payload):
            for col in ("pair", "source_id", "target_id", "i", "j"):
                payload[f"band_p_{col}"] = np.empty(0, dtype=np.int64)
            for col in ("primary_logit_preblend", "primary_prob_preblend",
                        "deployed_prob_postblend"):
                payload[f"band_p_{col}"] = np.empty(0, dtype=np.float64)

        nop = build("no_parity_band", trunk_seed=1, mutate=_no_parity_band)
        record("empty_parity_band", True, lambda: bind_crop_file(nop / f"{crop0}.npz", 0.5),
               want="band_p_is_empty")

        def _claims_reproducible_with_a_secondary(payload):
            payload["fusion_stages"] = np.str_("secondary_logit_blend")
            payload["fusion_secondary_enabled"] = np.bool_(True)
            payload["fusion_secondary_edge_weight"] = np.float64(0.15)
            # the lie: a second model contributed, and the cache says it can rebuild the surface
            payload["fusion_reproducible_from_primary_cache"] = np.bool_(True)

        fr = build("fusion_repro_lie", trunk_seed=1,
                   mutate=_claims_reproducible_with_a_secondary)
        record("fusion_claims_reproducibility_it_cannot_have", True,
               lambda: bind_crop_file(fr / f"{crop0}.npz", 0.5),
               want="fusion_reproducibility_claim_is_wrong")

        fs = build("fusion_stage_unrecorded", trunk_seed=1,
                   mutate=lambda p: p.__setitem__(
                       "fusion_bidirectional_weight", np.float64(0.20)))
        record("a_fusion_stage_ran_and_was_not_listed", True,
               lambda: bind_crop_file(fs / f"{crop0}.npz", 0.5),
               want="fusion_stages_disagree_with_the_recorded_weights")

        def _postblend_moved(payload):
            arr = payload["band_a_deployed_prob_postblend"].copy()
            arr[0] = min(float(arr[0]) + 0.05, 0.999)
            payload["band_a_deployed_prob_postblend"] = arr
            payload["deployed_edge_prob_postblend"] = np.concatenate(
                [arr, payload["band_b_deployed_prob_postblend"]]).astype(np.float32)

        pb = build("postblend_without_a_stage", trunk_seed=1, mutate=_postblend_moved)
        record("post_fusion_column_moved_with_no_stage_recorded", True,
               lambda: bind_crop_file(pb / f"{crop0}.npz", 0.5),
               want="no_fusion_but_band_a_surfaces_differ")

        def _parity_band_shrunk(payload):
            keep = np.ones(payload["band_p_pair"].shape[0], dtype=bool)
            keep[0] = False
            for col in ("pair", "source_id", "target_id", "i", "j",
                        "primary_logit_preblend", "primary_prob_preblend",
                        "deployed_prob_postblend"):
                payload[f"band_p_{col}"] = payload[f"band_p_{col}"][keep]

        pp = build("parity_band_shrunk", trunk_seed=1, mutate=_parity_band_shrunk)
        record("parity_band_membership_differs_with_no_stage", True,
               lambda: bind_crop_file(pp / f"{crop0}.npz", 0.5),
               want="no_fusion_but_band_p_is_not_band_a")

        def _torn_logits(payload):
            payload["band_b_primary_logit_preblend"] = \
                payload["band_b_primary_logit_preblend"][::-1].copy()

        tl = build("torn_logits", trunk_seed=1, n_per_frame=8, mutate=_torn_logits)
        record("logit_column_torn_from_its_probability", True,
               lambda: bind_crop_file(tl / f"{crop0}.npz", 0.5),
               want="band_b_logit_and_probability_disagree")

        # --- TRUNK AND FOLD ------------------------------------------------------------------
        swapped = build("swapped", trunk_seed=2)
        man_sw = swapped / "cache_manifest.json"
        man_sw.write_text(man.read_text(encoding="utf-8"), encoding="utf-8")
        record("swapped_trunk_features", True, lambda: audit(swapped, man_sw),
               want="features_unchanged")

        m2 = json.loads(man.read_text(encoding="utf-8")); m2["trunk"]["path"] = str(trunk_b)
        man_sw2 = clean / "manifest_wrong_trunk.json"
        man_sw2.write_text(json.dumps(m2), encoding="utf-8")
        record("swapped_trunk_checkpoint", True, lambda: audit(clean, man_sw2),
               want="trunk_checkpoint_sha256")

        m3 = json.loads(man.read_text(encoding="utf-8")); m3["trunk"]["provenance"] = "unknown"
        man_np = clean / "manifest_no_provenance.json"
        man_np.write_text(json.dumps(m3), encoding="utf-8")
        record("trunk_without_explicit_provenance", True, lambda: audit(clean, man_np),
               want="trunk_provenance_is_explicit_in_the_manifest")

        reordered = build("reordered", trunk_seed=1, reorder=True)
        man_ro = reordered / "cache_manifest.json"
        man_ro.write_text(man.read_text(encoding="utf-8"), encoding="utf-8")
        record("reordered_nodes", True, lambda: audit(reordered, man_ro),
               want="node_order_unchanged")

        f1 = root / "fold1"; f1.mkdir()
        _synth_cache(f1 / "6bba_bbbbbbbb.npz", crop="6bba_bbbbbbbb", trunk_seed=1)
        man_f1 = manifest_for(f1, trunk_a, fold="1", role="oof_split1",
                              glob="/kaggle/input/*/split_0/edge_predictor_best.pth")
        record("wrong_fold_weights_split0_on_fold1", True, lambda: audit(f1, man_f1),
               want="fold1_does_not_use_the_leaky_pack_weights")

        man_f1b = manifest_for(f1, trunk_a, fold="1", role="oof_split1",
                               glob="/kaggle/input/*/edge_predictor_best_split_1.pth",
                               name="good_f1.json")
        record("control_correct_fold1_cache", False, lambda: audit(f1, man_f1b))

        mismatch = json.loads(man_f1b.read_text(encoding="utf-8"))
        mismatch["fold"].update({"fold": "0", "held_out_embryo": "44b6",
                                 "weights_glob": "/kaggle/input/*/split_0/e.pth"})
        man_f1c = f1 / "fold0_over_6bba.json"
        man_f1c.write_text(json.dumps(mismatch), encoding="utf-8")
        record("wrong_fold_declared_over_the_other_embryo", True, lambda: audit(f1, man_f1c),
               want="embryo_matches_fold")

        record("no_manifest_at_all", True, lambda: audit(clean, root / "nope.json"),
               want="no manifest")

        m1 = json.loads(man.read_text(encoding="utf-8")); m1["schema_version"] = 1
        man_v1 = clean / "manifest_v1.json"; man_v1.write_text(json.dumps(m1), encoding="utf-8")
        record("schema_1_manifest_is_refused", True, lambda: audit(clean, man_v1),
               want="manifest schema")

        # --- DUAL TRUNK, AND THE ONE FOLD-AWARE PROVENANCE POLICY ----------------------------
        pair_b = build("pair_b", trunk_seed=7)
        manifest_for(pair_b, trunk_b, role="stabledet")

        # ACCEPT CONTROL: the HONEST fold-0 pair - a legitimate CLAIM arm plus the declared
        # comparison arm. Without this the rejections below prove only that the guard refuses.
        pack0 = build("pack0", trunk_seed=1)
        manifest_for(pack0, trunk_a, role="pack_split0")
        record("control_honest_fold0_pair_is_accepted", False,
               lambda: audit_dual_trunk(pack0, pair_b, require_roles=True))

        # THE FACT-0418 PAIR. `clean` declares role 'official' on fold 0, so this is
        # ['official','stabledet'] - and it was THIS FILE'S ACCEPT CONTROL until now.
        record("dual_trunk_fact0418_pair_has_no_legitimate_arm", True,
               lambda: audit_dual_trunk(clean, pair_b, require_roles=True),
               want="pair_has_no_fold_legitimate_claim_arm")

        same = build("pair_same", trunk_seed=1)
        manifest_for(same, trunk_b, role="stabledet")
        record("dual_trunk_one_trunk_written_twice", True,
               lambda: audit_dual_trunk(clean, same), want="dual_trunk_features_differ")

        diffnodes = build("pair_diffnodes", trunk_seed=7, n_per_frame=6)
        manifest_for(diffnodes, trunk_b, role="stabledet")
        record("dual_trunk_node_sets_disagree", True,
               lambda: audit_dual_trunk(clean, diffnodes), want="dual_trunk_shares_the_node_set")

        # THE FOLD SWAP, BOTH DIRECTIONS. Each mutation takes an HONEST pair and relabels it onto
        # the other fold, where its claim arm becomes a leaky checkpoint.
        # The glob is the fold-0 one and passes `fold0_does_not_load_split1_weights`; only the
        # ROLE is swapped. So this mutation is caught by the POLICY and by nothing else, which is
        # the point - the pre-existing glob checks do not see a role/fold leak.
        swap_a = build("swap_split1_on_fold0", trunk_seed=1)
        manifest_for(swap_a, trunk_a, fold="0", role="oof_split1",
                     glob="/kaggle/input/*/split_0/edge_predictor_best.pth")
        record("fold_swap_split1_claim_arm_on_fold0", True,
               lambda: audit_dual_trunk(swap_a, pair_b, require_roles=True),
               want="leaky_checkpoint_in_the_fold")

        f1_pack = root / "swap_split0_on_fold1"; f1_pack.mkdir()
        _synth_cache(f1_pack / "6bba_bbbbbbbb.npz", crop="6bba_bbbbbbbb", trunk_seed=1)
        manifest_for(f1_pack, trunk_a, fold="1", role="pack_split0",
                     glob="/kaggle/input/*/edge_predictor_best_split_1.pth")
        f1_ctrl = root / "swap_f1_control"; f1_ctrl.mkdir()
        _synth_cache(f1_ctrl / "6bba_bbbbbbbb.npz", crop="6bba_bbbbbbbb", trunk_seed=7)
        manifest_for(f1_ctrl, trunk_b, fold="1", role="stabledet",
                     glob="/kaggle/input/*/edge_predictor_best_split_1.pth")
        record("fold_swap_split0_claim_arm_on_fold1", True,
               lambda: audit_dual_trunk(f1_pack, f1_ctrl, require_roles=True),
               want="leaky_checkpoint_in_the_fold")

        # ACCEPT CONTROL on the other fold: the honest fold-1 pair, which the RETIRED constant
        # would have refused (FACT-0418).
        f1_claim = root / "honest_f1_claim"; f1_claim.mkdir()
        _synth_cache(f1_claim / "6bba_bbbbbbbb.npz", crop="6bba_bbbbbbbb", trunk_seed=1)
        manifest_for(f1_claim, trunk_a, fold="1", role="oof_split1",
                     glob="/kaggle/input/*/edge_predictor_best_split_1.pth")
        record("control_honest_fold1_pair_is_accepted", False,
               lambda: audit_dual_trunk(f1_claim, f1_ctrl, require_roles=True))

        # --- PROJECTION AND THE CAPACITY GUARD ------------------------------------------------
        rep = audit(clean, man)
        counts = {f"44b6_{i:08x}": 25000 for i in range(71)}
        record("control_projection_inside_budget", False,
               lambda: project_full_fold(rep["measured"], counts,
                                         capacity_bytes=64 * (1 << 30), label="fold0"))
        record("projection_over_capacity_is_refused", True,
               lambda: project_full_fold(rep["measured"], counts,
                                         capacity_bytes=1 << 20, label="fold0"),
               want="CAPACITY")
        record("projection_without_a_budget_is_refused", True,
               lambda: project_full_fold(rep["measured"], counts, capacity_bytes=0),
               want="projection_has_no_capacity")
        record("projection_without_a_measurement_is_refused", True,
               lambda: project_full_fold([], counts, capacity_bytes=1 << 40),
               want="projection_has_no_measurement")

    passed = all(r["ok"] for r in results)
    payload = {"schema_version": SCHEMA_VERSION, "self_test": "audit_feature_cache",
               "all_passed": passed, "results": results}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSELF-TEST {'PASSED' if passed else 'FAILED'} - "
          f"{sum(r['ok'] for r in results)}/{len(results)} behaved as required")
    return 0 if passed else 1


# --------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bind", help="write a manifest that binds a cache to its producer")
    b.add_argument("--cache-dir", required=True)
    b.add_argument("--manifest", required=True)
    b.add_argument("--trunk", required=True)
    b.add_argument("--trunk-role", required=True, choices=sorted(TRUNK_ROLES))
    b.add_argument("--trunk-provenance", required=True,
                   help="WHERE this checkpoint came from. The checkpoint is a bare state dict "
                        "with zero non-tensor keys, so this is the only record of it.")
    b.add_argument("--fold", required=True, choices=sorted(FOLD_EMBRYO))
    b.add_argument("--weights-glob", default="")
    b.add_argument("--feature-normalisation", default="none")
    b.add_argument("--candidate-rule", required=True)
    b.add_argument("--deployed-floor", type=float, default=0.5)
    b.add_argument("--acquisition-floor", type=float, default=0.02)
    b.add_argument("--rank-cap", type=int, default=8)
    b.add_argument("--gate-um", type=float)
    b.add_argument("--det-threshold", type=float, default=0.96875)
    b.add_argument("--pool-kernel-um", type=float, default=3.0)
    b.add_argument("--softmax-axis", default="source")
    b.add_argument("--abstain-mass", action="store_true")
    b.add_argument("--notebook")
    b.add_argument("--spec")
    b.add_argument("--kernel")
    b.add_argument("--kernel-version")
    b.add_argument("--commit")

    a = sub.add_parser("audit", help="independently re-check a cache against its manifest")
    a.add_argument("--cache-dir", required=True)
    a.add_argument("--manifest")
    a.add_argument("--trunk")
    a.add_argument("--expect-trunk-sha256")
    a.add_argument("--expect-fold", choices=sorted(FOLD_EMBRYO))
    a.add_argument("--expect-role", choices=sorted(TRUNK_ROLES))
    a.add_argument("--out")

    d = sub.add_parser("audit-pair", help="check two caches form a valid dual-trunk pair")
    d.add_argument("--cache-a", required=True)
    d.add_argument("--cache-b", required=True)
    d.add_argument("--manifest-a")
    d.add_argument("--manifest-b")
    d.add_argument("--require-fold-legitimate-roles", "--require-pkt0029-roles",
                   dest="require_fold_legitimate_roles", action="store_true",
                   help="require the pair to satisfy provenance_policy for its own fold: one "
                        "legitimate CLAIM arm, the other a permitted comparison arm, and no leaky "
                        "checkpoint in either (FACT-0418). The old spelling is kept as an alias "
                        "so a committed invocation does not silently stop guarding")
    d.add_argument("--out")

    p = sub.add_parser(
        "project",
        help="project full-fold storage from a MEASURED cache and refuse if it will not fit")
    p.add_argument("--cache-dir", required=True, help="an audited cache to measure")
    p.add_argument("--manifest")
    p.add_argument("--node-counts", required=True,
                   help="pre-ILP run_stats.csv for the fold (columns dataset, raw_nodes)")
    p.add_argument("--node-count-column", default="raw_nodes")
    p.add_argument("--capacity-bytes", type=int, required=True,
                   help="the byte budget. No default: a guard with a guessed budget is a warning")
    p.add_argument("--feat-dim", type=int, help="target feature width (default: as measured)")
    p.add_argument("--pos-dim", type=int)
    p.add_argument("--band-b-topk", type=int)
    p.add_argument("--label", default="fold")
    p.add_argument("--out")

    s = sub.add_parser("self-test", help="prove the auditor rejects each manufactured defect")
    s.add_argument("--out")

    args = ap.parse_args()

    if args.cmd == "self-test":
        return self_test(Path(args.out) if args.out else None)

    def emit(payload, path):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.cmd == "audit-pair":
        try:
            report = audit_dual_trunk(
                Path(args.cache_a), Path(args.cache_b),
                Path(args.manifest_a) if args.manifest_a else None,
                Path(args.manifest_b) if args.manifest_b else None,
                require_roles=args.require_fold_legitimate_roles)
        except Reject as err:
            emit({"passed": False, "reject": str(err)}, args.out)
            print("DUAL-TRUNK PAIR REJECTED", file=sys.stderr)
            print(f"  {err}", file=sys.stderr)
            return 1
        emit(report, args.out)
        print(f"DUAL-TRUNK PAIR OK  fold={report['fold']} roles={report['roles']} "
              f"crops={report['crops']}")
        return 0

    if args.cmd == "bind":
        try:
            manifest = build_manifest(args)
        except Reject as err:
            print(f"BIND REFUSED\n  {err}", file=sys.stderr)
            return 1
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        st = manifest["storage"]
        print(f"BOUND {args.cache_dir} -> {args.manifest}\n"
              f"  trunk {manifest['trunk']['role']} {manifest['trunk']['sha256'][:16]}...\n"
              f"  fold {manifest['fold']['fold']} ({manifest['fold']['held_out_embryo']}), "
              f"{len(manifest['crops'])} crops, window {manifest['window_contract']['window']}\n"
              f"  measured {st['bytes_on_disk']} B on disk over {st['nodes']} nodes / "
              f"{st['role_nodes']} role rows / {st['band_rows']} band rows")
        return 0

    if args.cmd == "project":
        cache_dir = Path(args.cache_dir)
        manifest = Path(args.manifest) if args.manifest else cache_dir / "cache_manifest.json"
        try:
            rep = audit(cache_dir, manifest)
            counts = read_node_counts(Path(args.node_counts), args.node_count_column)
            report = project_full_fold(
                rep["measured"], counts, capacity_bytes=args.capacity_bytes,
                feat_dim=args.feat_dim, pos_dim=args.pos_dim, band_b_topk=args.band_b_topk,
                label=args.label)
        except Reject as err:
            emit({"passed": False, "reject": str(err)}, args.out)
            print(f"PROJECTION REFUSED\n  {err}", file=sys.stderr)
            return 1
        emit(report, args.out)
        print(f"PROJECTION OK  {report['label']}: {report['crops']} crops, "
              f"{report['total_nodes']} nodes\n"
              f"  measured   {report['measured']['bytes_on_disk']} B on disk over "
              f"{report['measured']['caches']} cache(s), compression "
              f"{report['measured']['compression_ratio']:.4f}\n"
              f"  worst case {report['worst_case_gib']:.3f} GiB "
              f"({report['schema_bytes_per_node']['total']} B/node, no compression credit)\n"
              f"  expected   {report['expected_gib_at_measured_compression']:.3f} GiB at the "
              f"measured ratio\n"
              f"  budget     {report['capacity_bytes'] / float(1 << 30):.3f} GiB, headroom "
              f"{report['headroom_bytes'] / float(1 << 30):.3f} GiB")
        return 0

    cache_dir = Path(args.cache_dir)
    manifest = Path(args.manifest) if args.manifest else cache_dir / "cache_manifest.json"
    try:
        report = audit(cache_dir, manifest,
                       trunk=Path(args.trunk) if args.trunk else None,
                       expect_trunk_sha=args.expect_trunk_sha256,
                       expect_fold=args.expect_fold, expect_role=args.expect_role)
    except Reject as err:
        emit({"passed": False, "reject": str(err)}, args.out)
        print(f"CACHE AUDIT REJECTED\n  {err}", file=sys.stderr)
        return 1
    emit(report, args.out)
    total = sum(m["bytes_on_disk"] for m in report["measured"])
    print(f"CACHE AUDIT PASSED  fold={report['fold']} trunk={report['trunk_role']} "
          f"crops={report['crops']} checks={len(report['checks'])} bytes_on_disk={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
