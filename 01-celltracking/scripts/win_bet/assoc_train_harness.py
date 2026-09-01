r"""THE ONE TRAINING AND EVALUATION HARNESS every parent scorer is fitted and reported through.

WHAT THIS IS FOR (PKT-0034 / LEVER-0041)
----------------------------------------
Three scorer classes are being advanced at once - a calibrated linear/listwise ranker, a tree
ranker, and a contextual head whose feature contract is DECLARED rather than invented here. If each
brings its own training loop, its own splits and its own summary, the cycle produces three
unfalsifiable claims instead of a capability ladder. So the cache licence, the grouping, the
abstention rule, the single-candidate accounting and the report path live here, once.

This module does NOT own the surface or the label contract: those are frozen in
``assoc_parent_dataset`` and are read, never rewritten. It does not own the report decomposition
either: that is ``assoc_report``. What it owns is everything between the two.

THE FOUR CONTRACTS, AND WHY EACH ONE IS A REFUSAL RATHER THAN A WARNING
----------------------------------------------------------------------

0. THE FEATURE CACHE IS KEYED BY (PAIR, ROLE), NEVER BY FRAME - SCHEMA V2 (``FACT-0402``).
   This harness previously required ``feat_frames`` and per-frame ``feat_{t}`` blocks and scattered
   them into ONE ``(n_nodes, dim)`` matrix. That object does not exist. ``TemporalUNet3D`` contains
   ``_TemporalAttention``, which mixes across the window's time axis, so ``unet_out[:, i]`` depends
   on EVERY frame in the window; at the deployed ``window_size`` 2 with stride ``W-1 = 1`` an
   interior frame ``t`` is the TARGET of pair ``(t-1, t)`` and the SOURCE of pair ``(t, t+1)`` -
   two forward passes, two different vectors for one node. Measured on the real deployed path: 66
   of 173 role-nodes carry two vectors, max abs delta 0.0123, 2.1 ORDERS OF MAGNITUDE above this
   harness's own 1e-4 tolerance, and NOT uniform across the source axis.

   So a candidate edge, not a node, is what selects a feature: its source sits at ``t`` and its
   target at ``t + 1``, which names the deployed pair and therefore names which vector the head was
   fed. ``RoleFeatureIndex.rows_for_edges`` is that lookup and ``node_features`` is a tombstone
   that refuses under ``frame_keyed_consumption``. A frame-keyed ARTIFACT is refused one layer
   earlier, on load, under ``frame_keyed_cache``.

   The reader and the structural validation are ``scripts/win_bet/audit_feature_cache.py``'s -
   IMPORTED, not reimplemented. Five of the six Gate-1 defects were reimplementations of a
   deployed step; a seventh was found in this file while migrating it (see (a) below).

1. THE CACHE IS GATED BEFORE ANY FIT. ``FACT-0387`` is the reason: Gate 1 attempt 1 compared zero
   crops and would have licensed two feature-cache sessions had its heartbeat not refused to report
   a pass. A head trained on an unlicensed cache measures the CACHE, not the head, and every number
   downstream is then uninterpretable. So ``CacheGate`` runs first and raises ``HarnessRefusal``;
   there is no ``--force``, no warn-and-continue, and no code path that trains past a failed gate.

   The gate has FOUR parts and states honestly what each can and cannot prove:

     (a) SCHEMA + STRUCTURE + COORDINATE PARITY, computed on CPU from the cache bytes alone.
         ``audit_feature_cache`` proves the cache is internally coherent: frame partition tiling,
         pair stride, role-block tiling, ROLE MULTIPLICITY (a dual-role node must appear twice),
         DUAL-ROLE CONTENT (its two vectors must actually differ), node ordering against the
         deployed frame slice, masks, positional and feature widths, band membership. What it
         cannot do is check the cache against an EXTERNAL artifact, so this harness adds pre-ILP
         parity: ``node_id`` is a positional index into ``coords_so_far`` (``FACT-0373`` check D),
         so the cache's ``coords`` must reproduce the recorded ``(t, z, y, x)`` of every node.
         THE DOWNSAMPLE IS PART OF THAT, and the schema-1 check got it wrong: the tap flushes
         BEFORE the deployed rescale, so the cache holds the downsampled grid while the pre-ILP
         export holds ``coords[:, 1:] * downsample``
         (``predict_unet_transformer.py:798-802``). The old comparison was raw and passed only
         because its fixture used downsample (1,1,1); at the deployed (1,4,4) it rejects every
         real cache. Re-measured on the archived P36 cache: raw parity False, rescaled parity
         True on all 26,169 and 33,432 nodes of both crops.
     (b) CANDIDATE-AND-PROBABILITY REPRODUCTION, through a declared ``reproducer``. Two bands, the
         same two ``assoc_feature_parity`` uses: band A is the deployed edges above 0.5, band B the
         sub-threshold ECB surface where parent ranking is actually learned (``FACT-0382`` puts all
         691 contested errors below the deployed floor, so a band-A-only check validates exactly
         the band the task does not use). An in-process reproducer is used by the synthetic tests
         and by any future CPU-exact head.
     (c) THE GATE-1 RECEIPT is the only admissible proof of PROBABILITY parity for the real cache,
         because CPU and GPU floating-point paths can differ by more than the 1e-4 tolerance and a
         CPU re-derivation would confound a numeric-backend difference with a genuine cache error.
         Stated plainly: this harness does not re-do Gate 1, it REFUSES TO TRAIN WITHOUT IT.
     (d) THE TRUNK IS BOUND BY HASH. ``FACT-0392``: which trunk produced the 32-dim features is a
         ``--weights`` argument recorded NOWHERE in the checkpoint - every candidate trunk is a
         bare state dict of 136 tensors with zero non-tensor keys, and ``official_f0`` and
         ``stabledet_f0`` have an IDENTICAL byte size (8,357,783) with different sha256. So size
         cannot discriminate them and a manifest is mandatory. The gate refuses without one under
         ``trunk_not_bound``, and ``verify_licence`` re-hashes the checkpoint at training time and
         refuses a swap under ``trunk_sha256_changed`` - saying so explicitly when the sizes match,
         because that is the case a size check would have waved through.

   The receipt is bound to bytes, not to a filename. ``gate`` writes a LICENCE recording the sha256
   of every cache file it passed AND of the trunk; ``train`` recomputes both and refuses if either
   moved. A receipt that says "the cache passed" is worthless if the cache can be edited afterwards.

2. TARGETS ARE GROUPS, AND CROP-GROUPED IS NOT EMBRYO-HELD-OUT. A random row split leaks twice: a
   target's own positive and negatives would straddle the boundary, and neighbouring frames of one
   movie are not independent samples. Splits are by CROP and the guard is mechanical - every fold
   asserts that no crop AND no ``(crop, target)`` pair appears on both sides. Said once, plainly,
   because it is easy to over-read: each fold of this campaign is a SINGLE EMBRYO (fold 0 is all
   ``44b6``, fold 1 all ``6bba``), so crop-grouped CV is WITHIN-embryo generalisation. The payload
   reports ``embryo_held_out: false`` and names the embryos rather than leaving a reader to assume.
   Cross-embryo transfer is a separate question and this harness cannot answer it on one fold.

3. ABSTENTION IS EXPLICIT. The deployed rule is an argmax over sources with a threshold
   (``FACT-0369``), i.e. a null class the pipeline can choose. Here the null is a first-class
   option scored beside the candidates: the decision is ``argmax`` over ``{candidates} u {NULL}``,
   and ties against the null resolve TO THE NULL, which is what a strict ``prob > threshold`` rule
   does. With the ``none`` policy the null score is ``-inf`` and the decision reduces exactly to the
   frozen ``evaluate``'s argmax - an equivalence this module ASSERTS at runtime rather than claims.

4. SINGLE-CANDIDATE TARGETS ARE THE SUBTLE PART, AND THIS IS THE ANSWER TO IT.
   ``FACT-0386`` recorded that single-candidate retention of 1.0000 was ARITHMETIC, not skill: a
   non-abstaining argmax cannot lose a target that has one candidate. On fold 0 that is 78.6% of
   the surface (``FACT-0381``). The moment abstention exists the protection disappears and side (b)
   of the two-sided bar becomes live for the first time. A netted single-candidate figure would
   hide exactly that, so this harness never emits one. Both available protections are implemented
   and BOTH report:

     * ``preserve_single_candidate: true`` (default) - the null is STRUCTURALLY UNAVAILABLE on
       targets with one candidate, so retention is 1.0 by construction. That alone would recreate
       the blind spot, so the harness additionally computes the SHADOW counterfactual: the identity
       of every single-candidate target the model WOULD have abstained on had it been allowed.
       Preservation by construction then prices the constraint instead of hiding it, and a shadow
       count is a measurement of how much work the constraint is doing.
     * ``preserve_single_candidate: false`` - the null is live everywhere and EVERY regression is
       listed individually by ``(crop, target, source)``. Because a decidable single-candidate
       target's one candidate IS its true parent, the only way to lose one is to abstain; the
       ledger asserts that, and any other reason is a contract violation that raises.

   ``SingleCandidateLedger.summary()`` raises if the count and the identity list disagree, so a
   summary that reports "lost 7" without seven identities cannot be produced at all.

5. ONE REPORT. Contested top-1, true-parent margin, gained/lost TP IDENTITIES, node recall, raw
   edge Jaccard, the count-adjustment contribution and division TP/FP/FN all travel through
   ``assoc_report``. Every model class emits the SAME channel keys - that is falsifier (d) of this
   packet and ``tests/test_assoc_train_harness.py`` asserts it by running all three.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
The contextual head's feature contract. ``LEVER-0034`` owns identifying it; this harness ACCEPTS a
declaration and validates it against the cache (dimension, availability, pair construction). An
undeclared contract is a refusal, not a default - falling back to representation-free features
would silently answer a different question from the one asked.

THE ONE REAL-CACHE SMOKE, READY TO RUN THE MOMENT GATE 1 (EXP-0041) PASSES
--------------------------------------------------------------------------
Nothing below trains on an unlicensed cache; the gate is the first thing that runs and it refuses.

    set PYTHONIOENCODING=utf-8 && set PYTHONUTF8=1 && .venv\Scripts\python.exe ^
      scripts\win_bet\assoc_train_harness.py gate ^
        --cache-dir C:/temp/assoc_harness/cache_f0 ^
        --manifest  C:/temp/assoc_harness/cache_f0/cache_manifest.json ^
        --receipt   C:/temp/assoc_harness/assoc_feature_tap_gate.json ^
        --preilp    C:/temp/p30_f0/preilp_split0.parquet ^
        --ecb-dir   C:/temp/p30_f0/ecb ^
        --licence   C:/temp/assoc_harness/licence_f0.json ^
        --out       C:/temp/assoc_harness/gate_f0.json

and then, only if that prints ``ASSOC_CACHE_GATE_PASSED``:

    .venv\Scripts\python.exe scripts\win_bet\assoc_train_harness.py train ^
      --spec scripts/win_bet/assoc_specs/harness_f0.json

Three operational notes so the command is runnable rather than aspirational. The cache is written
on Kaggle by the Gate-1 kernel, so ``cache_f0`` and the receipt must be FETCHED to the paths above
first; the manifest is built by ``scripts/win_bet/audit_feature_cache.py bind`` and is MANDATORY,
because the trunk is otherwise unnamed (``FACT-0392``); and the spec's ``cache.crops`` is null,
meaning every crop in the table needs a cache file - set it to the crops the gate actually covered
to smoke a subset. A crop with no cache file is a refusal, not a skip.

SCHEMA VALIDATION, WHICH IS NOT A LICENCE AND MUST NEVER BECOME ONE
------------------------------------------------------------------
    .venv\Scripts\python.exe scripts\win_bet\assoc_train_harness.py schema ^
      --cache-dir C:/temp/assoc_tournament/schema_v2/aft_cache

reads a tapped cache, validates the pair-and-role schema and exercises this harness's own role
lookup on every recorded band edge. It prints ``TRAINING_LICENCE=NOT_GRANTED`` and its report
carries NO ``passed`` key, so ``write_licence`` refuses it structurally. ``FACT-0403`` is why the
distinction is load-bearing: the archived P36 cache has PROVEN structure - passivity, pair-and-role
schema, zero integrity failures, positional features round-tripping at exactly 0.0 - and an INVALID
parity verdict, because the deployed probabilities it recorded as a reference are computed AFTER a
secondary-model logit blend at weight 0.15 while the replay re-runs the primary head only. It
therefore proves structure and NOTHING about learned quantities. No head may be trained on it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_feature_cache as AFC  # noqa: E402  the schema-v2 reference reader (PKT-0037)
from assoc_baseline_rankers import (  # noqa: E402  protocol continuity with the LEVER-0039 arms
    LINEAR_KW,
    SEED,
    TREE_KW,
    calibration,
    crop_paired_bootstrap,
)
from assoc_parent_dataset import FEATURES, evaluate  # noqa: E402  the FROZEN surface
from assoc_report import parent_conversions  # noqa: E402
import spec_restriction  # noqa: E402  FACT-0451 at the schema boundary; rules 1-3 at load, 4 at
#                                      the verdict. Imported at module scope so a missing module
#                                      is a startup error, never a silently skipped guard.

PROB_TOL = 1e-4

# SCHEMA V2 - PAIR AND ROLE. The key list is IMPORTED from the auditor rather than restated here,
# because a third copy of a schema is a third place for it to drift: the tap DECLARES it
# (`_AFT_SCHEMA_REQUIRED`), the auditor mirrors it under an ast-parsing anti-drift test, and this
# harness now reads the auditor's mirror. `role_pos` is optional to the tap and REQUIRED here, for
# the auditor's reason - the positional features are a head INPUT, so a cache without them forces
# a re-derivation in place of an artifact.
CACHE_SCHEMA_VERSION = AFC.SCHEMA_VERSION
CACHE_KEYS = AFC.REQUIRED_KEYS + (AFC.POSITIONAL_KEY,)
ROLE_SRC, ROLE_TGT = AFC.ROLE_SRC, AFC.ROLE_TGT

# THE NAMED CONDITIONS THIS MODULE REFUSES UNDER. Named, because "the harness raised" is not a
# diagnosis: FACT-0402 was found only because the failure could be attributed to a mechanism.
FRAME_KEYED_CACHE = "frame_keyed_cache"                  # the retired artifact, rejected on load
FRAME_KEYED_CONSUMPTION = "frame_keyed_consumption"      # the retired ACCESS PATTERN, tombstoned
CANDIDATE_NOT_A_DEPLOYED_PAIR = "candidate_is_not_a_deployed_frame_pair"
NO_CACHED_PAIR = "no_cached_pair_covers_this_candidate"
ROLE_LOOKUP_DISAGREES = "role_lookup_disagrees_with_the_role_table"
TRUNK_NOT_BOUND = "trunk_not_bound"
TRUNK_SHA_CHANGED = "trunk_sha256_changed"

DEPLOYED_FLOOR = 0.5          # FACT-0369, verified at source; not a tunable
NEVER = -np.inf               # the null score that makes abstention impossible


class HarnessRefusal(RuntimeError):
    """Raised wherever the harness must stop rather than produce an uninterpretable number."""


# ======================================================================================
# 1. THE CACHE GATE
# ======================================================================================

def cache_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cache(path: Path) -> dict:
    """Load one crop's SCHEMA-V2 cache, through the auditor's reader rather than a second one.

    The reader is `audit_feature_cache.load_cache`, not a reimplementation of it. That choice is
    the whole lesson of this lane: six defects in three Gate-1 attempts, every one of them a
    worker approximating a path instead of reusing it (`FACT-0402`'s ledger). Its refusals are
    re-raised as `HarnessRefusal` with the CONDITION NAME preserved verbatim, so a caller can
    still tell the three retirement conditions apart, and they want three different responses:
    `frame_keyed_cache` - retire the artifact, its addressing is undefined; `cache_contract_1` -
    the artifact is readable but its one probability column names no surface, so RE-RUN THE TAP
    at contract 2 (`FACT-0407`: the archived P36 cache is faithful and still unusable for
    training, because the pre-fusion surface it never stored cannot be rewritten into it);
    `cache_schema_incomplete` - go looking for a truncated write.
    """
    path = Path(path)
    if not path.is_file():
        raise HarnessRefusal(f"cache missing: {path}")
    try:
        return AFC.load_cache(path)
    except AFC.Reject as exc:
        raise HarnessRefusal(str(exc)) from exc


def node_features(cache: dict) -> np.ndarray:
    """RETIRED. A node does not have "a" feature vector, so this cannot return one.

    Kept as a loud tombstone rather than deleted. It was the schema-1 access pattern - scatter the
    per-frame blocks into one ``(n_nodes, dim)`` matrix indexed by ``node_id`` - and the object it
    returned does not exist: ``TemporalUNet3D._TemporalAttention`` mixes across the window's time
    axis, so at the deployed ``window_size`` 2 an interior frame ``t`` is the TARGET of pair
    ``(t-1, t)`` and the SOURCE of pair ``(t, t+1)`` and carries TWO vectors (`FACT-0402`: 66 of
    173 role-nodes, max abs delta 0.0123, 2.1 orders of magnitude above the 1e-4 gate tolerance).
    Deleting it would leave a caller with an ``AttributeError`` and no diagnosis; this names the
    mechanism and points at the replacement.
    """
    raise HarnessRefusal(
        f"{FRAME_KEYED_CONSUMPTION}: node_features() scattered one feature vector per node, which "
        "is the retired schema-1 access pattern and is UNDEFINED at the deployed window - a "
        "trunk feature belongs to a (pair, role), never to a frame (FACT-0402). Use "
        "role_features(cache) and address features through RoleFeatureIndex.rows_for_edges(), "
        "which resolves the SOURCE-role vector and the TARGET-role vector of the pair each "
        "candidate edge actually belongs to."
    )


@dataclass
class RoleFeatureIndex:
    """Role-specific node features, addressed by (frame pair, role, node id) - never by frame.

    WHY THIS SHAPE. A candidate edge is not an arbitrary pair of nodes: its source sits at frame
    ``t`` and its target at ``t + 1``, so the edge NAMES the deployed frame pair it belongs to,
    and therefore names which of a node's vectors the head was fed. That makes the lookup exact
    rather than a choice - which is precisely what the frame-keyed layout destroyed, because it
    had one vector to offer and had to guess.

    Every arithmetic step below is a deployed invariant the auditor has already checked on this
    cache, so nothing here re-derives anything:
      * a role block is exactly ``arange(starts[k], ends[k])`` - the deployed
        ``idx_src = np.arange(s_src, e_src)`` (``predict_unet_transformer.py:572-573``), asserted
        by ``role_node_order_is_not_the_deployed_frame_slice``;
      * so the row of node ``g`` in a block is ``ptr + (g - start_of_its_frame)``;
      * and pairs are consecutive frames, asserted by ``pair_stride_is_not_one_frame``.
    The identity is nevertheless RE-CHECKED against ``role_gid`` on every lookup. It costs one
    comparison and it is the difference between a computed row and a verified one.
    """

    crop: str
    feat: np.ndarray                 # (n_role, feat_dim) - role rows, NOT node rows
    dim: int
    node_count: int
    role_gid: np.ndarray
    _frame_of_node: np.ndarray       # (node_count,) absolute frame index per node id
    _start_by_frame: np.ndarray      # frame -> block start, -1 where the frame is absent
    _pair_by_t_src: np.ndarray       # source frame -> pair index, -1 where no pair was recorded
    _src_ptr: np.ndarray
    _tgt_ptr: np.ndarray

    def rows_for_edges(self, src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Role rows for candidate edges. Every failure is a named refusal, never a zero row."""
        src = np.asarray(src, dtype=np.int64)
        tgt = np.asarray(tgt, dtype=np.int64)
        if np.any((src < 0) | (src >= self.node_count) | (tgt < 0) | (tgt >= self.node_count)):
            raise HarnessRefusal(
                f"{self.crop}: a candidate names a node outside [0, {self.node_count}) - the "
                "surface and the cache do not describe the same crop"
            )
        t_src = self._frame_of_node[src]
        t_tgt = self._frame_of_node[tgt]
        bad = np.nonzero(t_tgt != t_src + 1)[0]
        if bad.size:
            i = int(bad[0])
            raise HarnessRefusal(
                f"{CANDIDATE_NOT_A_DEPLOYED_PAIR}: {self.crop}: {int(bad.size)} candidate(s) span "
                f"frames that are not consecutive, e.g. {int(src[i])}@t={int(t_src[i])} -> "
                f"{int(tgt[i])}@t={int(t_tgt[i])}. The deployed loop only ever scores consecutive "
                "frame_indices, so such an edge has no (pair, role) and no cached features exist "
                "for it - a frame-keyed cache would have silently served some vector anyway"
            )
        pair = self._pair_by_t_src[t_src]
        missing = np.nonzero(pair < 0)[0]
        if missing.size:
            i = int(missing[0])
            raise HarnessRefusal(
                f"{NO_CACHED_PAIR}: {self.crop}: {int(missing.size)} candidate(s) fall in frame "
                f"pairs this cache never recorded, e.g. t={int(t_src[i])} -> {int(t_tgt[i])}. A "
                "partially tapped cache (BIOHUB_AFT_MAX_PAIRS) covers only the pairs it captured; "
                "training over the rest would be training on features that were never computed"
            )
        src_row = self._src_ptr[pair] + (src - self._start_by_frame[t_src])
        tgt_row = self._tgt_ptr[pair] + (tgt - self._start_by_frame[t_tgt])
        if not (np.array_equal(self.role_gid[src_row], src)
                and np.array_equal(self.role_gid[tgt_row], tgt)):
            raise HarnessRefusal(
                f"{ROLE_LOOKUP_DISAGREES}: {self.crop}: a computed role row does not carry the "
                "node id it was computed for. The role block is no longer the deployed frame "
                "slice in the deployed order, so every feature row would be filed under the "
                "wrong cell"
            )
        return src_row, tgt_row


def role_features(cache: dict, crop: str | None = None) -> RoleFeatureIndex:
    """Build the (pair, role) address book for one licensed cache."""
    frames = np.asarray(cache["frames"]).astype(np.int64)
    starts = np.asarray(cache["starts"]).astype(np.int64)
    ends = np.asarray(cache["ends"]).astype(np.int64)
    role_feat = np.asarray(cache["role_feat"])
    role_gid = np.asarray(cache["role_gid"]).astype(np.int64)
    t_src = np.asarray(cache["pair_t_src"]).astype(np.int64)
    node_count = int(cache["node_count"])
    hi = int(max(int(frames.max()), int(t_src.max())) + 2) if frames.size else 2

    start_by_frame = np.full(hi, -1, dtype=np.int64)
    start_by_frame[frames] = starts
    pair_by_t_src = np.full(hi, -1, dtype=np.int64)
    pair_by_t_src[t_src] = np.arange(t_src.shape[0], dtype=np.int64)
    return RoleFeatureIndex(
        crop=str(crop if crop is not None else cache["crop"]),
        feat=role_feat,
        dim=int(role_feat.shape[1]),
        node_count=node_count,
        role_gid=role_gid,
        _frame_of_node=np.repeat(frames, (ends - starts)),
        _start_by_frame=start_by_frame,
        _pair_by_t_src=pair_by_t_src,
        _src_ptr=np.asarray(cache["pair_src_ptr"]).astype(np.int64),
        _tgt_ptr=np.asarray(cache["pair_tgt_ptr"]).astype(np.int64),
    )


def _preilp_parity(crop: str, cache: dict, pre_nodes: pl.DataFrame) -> list[str]:
    """Tie the cache to the EXTERNAL artifact every downstream id is addressed against.

    The auditor already proves the cache is internally coherent - partition tiling, role
    multiplicity, node ordering, masks, positional width, band membership. What it cannot do is
    check the cache against the pre-ILP export, because it never sees one. That is this
    function's whole job, and it is the harness's own value-add rather than a second copy of a
    check that already exists.

    THE DOWNSAMPLE IS NOT OPTIONAL, AND GETTING IT WRONG WAS A LIVE DEFECT HERE. The tap flushes
    BEFORE the deployed coordinate rescale, so the cache carries the DOWNSAMPLED grid the model
    indexed, while ``predict_video`` returns - and the pre-ILP export therefore records -
    ``coords[:, 1:] * downsample`` cast to int16 (``predict_unet_transformer.py:798-802``). The
    schema-1 check compared the two RAW and passed only because its fixture used downsample
    (1,1,1); at the deployed (1,4,4) it would have rejected every real cache. Re-measured on the
    archived P36 cache: raw parity False, rescaled parity True on all 26,169 and 33,432 nodes of
    both crops.
    """
    fail: list[str] = []
    coords = np.asarray(cache["coords"])
    frames = np.asarray(cache["frames"]).astype(np.int64)
    downsample = np.asarray(cache["downsample"]).astype(np.float32)

    rec = (pre_nodes.filter(pl.col("dataset") == crop)
                    .sort("node_id")
                    .select(["node_id", "t", "z", "y", "x"]))
    if rec.height == 0:
        return [f"{crop}: no recorded nodes in the pre-ILP export"]
    if not np.array_equal(rec["node_id"].to_numpy(), np.arange(rec.height)):
        fail.append(f"{crop}: recorded node_id is not the 0..N-1 positional index")
    rec_t = rec["t"].to_numpy().astype(np.int64)
    rec_zyx = rec.select(["z", "y", "x"]).to_numpy().astype(np.float64)

    # Per-frame node-count parity: without it a short detection silently shrinks the comparison.
    for t, s, e in zip(frames.tolist(), cache["starts"].tolist(), cache["ends"].tolist()):
        n_rec = int((rec_t == t).sum())
        if n_rec != e - s:
            fail.append(f"{crop}: frame {t} node count cached {e - s} vs recorded {n_rec}")

    sel = np.nonzero(np.isin(rec_t, frames))[0]
    if len(sel) != len(coords):
        fail.append(f"{crop}: cached coords {len(coords)} vs recorded {len(sel)} over cached frames")
        return fail
    if not np.array_equal(coords[:, 0].astype(np.int64), rec_t[sel]):
        fail.append(f"{crop}: cached frame indices disagree with the recorded nodes")
    rescaled = (coords[:, 1:].astype(np.float32) * downsample).astype(np.int16)
    if not np.array_equal(rescaled.astype(np.float64), rec_zyx[sel]):
        fail.append(f"{crop}: cached coordinates disagree with the recorded nodes after the "
                    f"deployed downsample rescale {tuple(int(d) for d in downsample)}")
    return fail


def _bands(crop: str, cache: dict, pre: pl.DataFrame, ecb_dir: Path | None) -> tuple[dict, dict]:
    """The recorded band A (deployed edges > 0.5) and band B (sub-threshold ECB) in cache scope."""
    frames = set(cache["frames"].tolist())
    starts, ends = cache["starts"].tolist(), cache["ends"].tolist()
    node_frame: dict[int, int] = {}
    for t, s, e in zip(cache["frames"].tolist(), starts, ends):
        for n in range(s, e):
            node_frame[n] = t
    in_scope = lambda a, b: node_frame.get(a) in frames and node_frame.get(b) in frames  # noqa: E731

    e = pre.filter((pl.col("dataset") == crop) & (pl.col("row_type") == "edge"))
    band_a = {(int(a), int(b)): float(p)
              for a, b, p in zip(e["source_id"], e["target_id"], e["edge_prob"])
              if in_scope(int(a), int(b))}
    band_b: dict[tuple[int, int], float] = {}
    side = (Path(ecb_dir) / f"{crop}.npz") if ecb_dir else None
    if side is not None and side.is_file():
        with np.load(side, allow_pickle=False) as z:
            for a, b, p in zip(z["source_id"].astype(np.int64).tolist(),
                               z["target_id"].astype(np.int64).tolist(),
                               z["edge_prob"].astype(np.float64).tolist()):
                if p > DEPLOYED_FLOOR or not in_scope(a, b):
                    continue
                band_b[(a, b)] = p
    return band_a, band_b


def resolve_callable(spec: str):
    """``module:function`` -> the callable. Import failure is a refusal, never a fallback."""
    if ":" not in spec:
        raise HarnessRefusal(f"expected 'module:function', got {spec!r}")
    mod, fn = spec.split(":", 1)
    try:
        return getattr(importlib.import_module(mod), fn)
    except Exception as exc:  # pragma: no cover - message is the point
        raise HarnessRefusal(f"cannot resolve {spec!r}: {type(exc).__name__}: {exc}") from exc


def gate_cache(*, cache_dir: Path, preilp: Path, ecb_dir: Path | None, crops: list[str] | None,
               receipt: Path | None, reproducer: str = "receipt",
               manifest: Path | None = None, expect_trunk_sha: str | None = None,
               expect_fold: str | None = None, expect_trunk_role: str | None = None) -> dict:
    """Run the cache gate. Returns a report; a failure is reported, never raised away.

    ``reproducer`` is ``"receipt"`` (the Gate-1 GPU proof, the only admissible probability parity
    for the real cache) or ``"callable:module:function"`` taking ``(crop, cache)`` and returning
    ``{(source_id, target_id): prob}``.

    ``manifest`` is MANDATORY and binds the TRUNK. `FACT-0392`: which trunk produced the features
    is a ``--weights`` argument recorded nowhere in the checkpoint, every candidate trunk is a
    bare state dict of 136 tensors with zero non-tensor keys, and ``official_f0`` and
    ``stabledet_f0`` have the SAME byte size with different sha256 - so size cannot discriminate
    them and provenance can only come from the manifest. Without it, a null result cannot be told
    apart from "we fed it the wrong features", which is the expensive failure on this lane.
    """
    cache_dir = Path(cache_dir)
    pre = pl.read_parquet(preilp)
    pre_nodes = pre.filter(pl.col("row_type") == "node")
    available = sorted(p.stem for p in cache_dir.glob("*.npz"))
    crops = list(crops) if crops else available
    report: dict = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "layout": "pair_and_role",
        "heartbeat": "ASSOC_CACHE_GATE_COMPLETE",
        "cache_dir": str(cache_dir),
        "preilp": str(preilp),
        "ecb_dir": str(ecb_dir) if ecb_dir else None,
        "reproducer": reproducer,
        "receipt": str(receipt) if receipt else None,
        "manifest": str(manifest) if manifest else None,
        "trunk": None,
        # Said in the artifact, not only in the docstring, so a reader of the payload can never
        # mistake an in-process reproduction for the GPU-side numeric proof.
        "probability_parity_proof": (
            "gate1_receipt" if reproducer == "receipt" else
            "in_process_reproducer - NOT admissible for a GPU-written cache, because the CPU and "
            "GPU float paths differ by more than the tolerance"
        ),
        "crops": [],
        "cache_sha256": {},
        "passed": False,
        "refusals": [],
    }
    if not crops:
        report["refusals"].append("no cache files found - a gate that compares nothing is not a gate")
        return report

    # --- THE TRUNK BINDING, AND THE WHOLE SCHEMA-V2 STRUCTURAL AUDIT, IN ONE CALL -------------
    # `audit_feature_cache.audit` is the committed reference: it re-derives the manifest binding
    # AND runs the per-crop pair-and-role validation (partition tiling, pair stride, role
    # multiplicity, dual-role content, node ordering, masks, positional and feature widths, band
    # membership). Calling it is how this harness avoids owning a second copy of those checks.
    if manifest is None:
        report["refusals"].append(
            f"{TRUNK_NOT_BOUND}: no cache manifest. Trunk identity is a command-line argument "
            "recorded nowhere in the checkpoint, and official_f0 and stabledet_f0 share a byte "
            "size with different sha256 (FACT-0392) - so an unbound cache cannot distinguish "
            "'the head does not work' from 'we fed it the wrong features', and is not licensed"
        )
        return report
    try:
        binding = AFC.audit(cache_dir, Path(manifest), expect_trunk_sha=expect_trunk_sha,
                            expect_fold=expect_fold, expect_role=expect_trunk_role)
    except AFC.Reject as exc:
        report["refusals"].append(f"cache manifest binding failed: {exc}")
        return report
    man = json.loads(Path(manifest).read_text(encoding="utf-8"))
    report["trunk"] = {
        "role": binding["trunk_role"],
        "sha256": binding["trunk_sha256"],
        "path": man["trunk"]["path"],
        "bytes": man["trunk"]["bytes"],
        "provenance": man["trunk"]["provenance"],
        "identified_by": "sha256 of the checkpoint bytes, recorded in the manifest",
        "why_not_size": (
            "official_f0 and stabledet_f0 have an IDENTICAL byte size and different sha256 "
            "(FACT-0392), so a size match is not an identity"
        ),
    }
    report["manifest_checks"] = len(binding["checks"])
    report["fold"] = binding["fold"]
    report["schema_measured"] = {m["crop"]: m for m in binding["measured"]}

    receipt_crops: dict[str, dict] = {}
    if reproducer == "receipt":
        if receipt is None or not Path(receipt).is_file():
            report["refusals"].append(
                "no Gate-1 receipt: probability parity cannot be established on CPU (the GPU/CPU "
                "float paths differ by more than the tolerance), so training is refused"
            )
            return report
        data = json.loads(Path(receipt).read_text(encoding="utf-8"))
        report["receipt_all_passed"] = bool(data.get("all_passed"))
        if not data.get("all_passed"):
            report["refusals"].append("Gate-1 receipt says all_passed is false (FACT-0387)")
        receipt_crops = {c["crop"]: c for c in data.get("crops", [])}

    repro_fn = None
    if reproducer.startswith("callable:"):
        repro_fn = resolve_callable(reproducer.split(":", 1)[1])
    elif reproducer != "receipt":
        report["refusals"].append(f"unknown reproducer {reproducer!r}")
        return report

    for crop in crops:
        path = cache_dir / f"{crop}.npz"
        entry: dict = {"crop": crop, "reasons": []}
        try:
            cache = load_cache(path)
        except HarnessRefusal as exc:
            entry["reasons"].append(str(exc))
            entry["passed"] = False
            report["crops"].append(entry)
            continue
        report["cache_sha256"][crop] = cache_digest(path)
        entry["nodes"] = int(len(cache["coords"]))
        entry["frames"] = int(len(cache["frames"]))
        # The pair-and-role facts the auditor measured, restated per crop so the gate payload
        # carries them rather than pointing at another artifact.
        meas = report["schema_measured"].get(crop, {})
        entry["role_nodes"] = meas.get("role_nodes")
        entry["dual_role_nodes"] = meas.get("dual_role_nodes")
        entry["dual_role_max_abs_delta"] = meas.get("dual_role_max_abs_delta")
        entry["feat_dim"] = meas.get("feature_dim")
        entry["pos_dim"] = meas.get("pos_dim")
        entry["pairs"] = int(np.asarray(cache["pair_f_idx"]).shape[0])
        if crop not in report["schema_measured"]:
            entry["reasons"].append(
                f"{crop}: the manifest binding did not cover this crop, so its pair-and-role "
                "structure was never audited")
        entry["reasons"].extend(_preilp_parity(crop, cache, pre_nodes))

        band_a, band_b = _bands(crop, cache, pre, ecb_dir)
        entry["band_a_recorded"] = len(band_a)
        entry["band_b_recorded"] = len(band_b)
        if reproducer == "receipt":
            rc = receipt_crops.get(crop)
            if rc is None:
                entry["reasons"].append("crop is absent from the Gate-1 receipt")
            else:
                if not rc.get("passed"):
                    entry["reasons"].append("Gate-1 receipt marks this crop failed")
                if rc.get("node_count_mismatches"):
                    entry["reasons"].append("Gate-1 receipt records node-count mismatches")
                if int(rc.get("band_b", {}).get("checked", 0)) <= 0:
                    entry["reasons"].append(
                        "Gate-1 receipt checked zero band-B pairs: it validated only the band the "
                        "learnable task does not use (FACT-0382)"
                    )
        else:
            got = {(int(a), int(b)): float(p) for (a, b), p in repro_fn(crop, cache).items()}
            got_a = {k: v for k, v in got.items() if v > DEPLOYED_FLOOR}
            miss_a = sorted(set(band_a) - set(got_a))
            extra_a = sorted(set(got_a) - set(band_a))
            shared = set(band_a) & set(got_a)
            d_a = max((abs(got_a[k] - band_a[k]) for k in shared), default=0.0)
            miss_b = [k for k in band_b if k not in got]
            d_b = max((abs(got[k] - band_b[k]) for k in band_b if k in got), default=0.0)
            checked_b = sum(1 for k in band_b if k in got)
            entry["band_a"] = {"missing": len(miss_a), "extra": len(extra_a),
                               "max_abs_prob_delta": d_a}
            entry["band_b"] = {"checked": checked_b, "missing": len(miss_b),
                               "max_abs_prob_delta": d_b}
            if miss_a or extra_a:
                entry["reasons"].append(
                    f"band A candidate set differs: {len(miss_a)} missing, {len(extra_a)} extra")
            if d_a > PROB_TOL:
                entry["reasons"].append(f"band A probability delta {d_a:.3e} exceeds {PROB_TOL:.0e}")
            if miss_b:
                entry["reasons"].append(f"band B: {len(miss_b)} recorded sub-threshold pairs "
                                        "were not reproduced")
            if d_b > PROB_TOL:
                entry["reasons"].append(f"band B probability delta {d_b:.3e} exceeds {PROB_TOL:.0e}")
            if checked_b == 0:
                entry["reasons"].append(
                    "zero band-B pairs compared: the sub-0.5 band is where parent ranking is "
                    "learned (FACT-0382), so a band-A-only pass is not Gate 1")
        entry["passed"] = not entry["reasons"]
        report["crops"].append(entry)

    report["passed"] = (
        bool(report["crops"])
        and not report["refusals"]
        and all(c["passed"] for c in report["crops"])
    )
    return report


def write_licence(gate_report: dict, out: Path) -> dict:
    """Pin the cache BYTES and the TRUNK the gate passed. A receipt naming a file licenses nothing.

    A schema-only report (``schema_report``) deliberately carries no ``passed`` key, so it cannot
    reach this function: proving that a cache is well formed is not the same as proving it may be
    trained on, and `FACT-0403` is the live example - P36's structure is sound and its parity
    verdict is INVALID.
    """
    if not gate_report.get("passed"):
        raise HarnessRefusal("refusing to write a licence for a cache gate that did not pass")
    trunk = gate_report.get("trunk") or {}
    if not trunk.get("sha256"):
        raise HarnessRefusal(
            f"{TRUNK_NOT_BOUND}: the gate report carries no trunk sha256, so a licence written "
            "from it would pin the cache bytes and leave the producing trunk unnamed - the exact "
            "ambiguity FACT-0392 records"
        )
    licence = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "layout": "pair_and_role",
        "heartbeat": "ASSOC_CACHE_LICENCE",
        "cache_dir": gate_report["cache_dir"],
        "reproducer": gate_report["reproducer"],
        "probability_parity_proof": gate_report.get("probability_parity_proof"),
        "receipt": gate_report.get("receipt"),
        "manifest": gate_report.get("manifest"),
        "trunk": trunk,
        "cache_sha256": gate_report["cache_sha256"],
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(licence, indent=2), encoding="utf-8")
    return licence


def verify_licence(licence_path: Path, cache_dir: Path, crops: list[str],
                   trunk: Path | None = None) -> dict:
    """Re-verify the pinned digests at training time. A moved byte is a refusal.

    TWO digests, not one. The cache bytes answer "is this the file that was gated"; the TRUNK
    sha256 answers "is this the checkpoint that produced it", and neither substitutes for the
    other. The trunk check is by CONTENT HASH and never by size, because the two trunks this
    campaign must tell apart share a byte size exactly (`FACT-0392`).
    """
    p = Path(licence_path)
    if not p.is_file():
        raise HarnessRefusal(
            f"no cache licence at {p}: run the `gate` subcommand first. A head trained on an "
            "unlicensed cache measures the cache (FACT-0387)."
        )
    lic = json.loads(p.read_text(encoding="utf-8"))
    if int(lic.get("schema_version", 1)) != CACHE_SCHEMA_VERSION:
        raise HarnessRefusal(
            f"cache licence schema {lic.get('schema_version')!r} != {CACHE_SCHEMA_VERSION}. "
            "Schema 1 licensed a FRAME-KEYED cache, a layout FACT-0402 measured to be undefined "
            "at the deployed window; it cannot license a pair-and-role cache"
        )
    pinned = lic.get("cache_sha256", {})
    bad, absent = [], []
    for crop in crops:
        want = pinned.get(crop)
        f = Path(cache_dir) / f"{crop}.npz"
        if want is None:
            absent.append(crop)
        elif not f.is_file() or cache_digest(f) != want:
            bad.append(crop)
    if absent:
        raise HarnessRefusal(f"crops not covered by the cache licence: {absent[:5]}")
    if bad:
        raise HarnessRefusal(f"cache bytes changed since the licence was issued: {bad[:5]}")

    lic_trunk = lic.get("trunk") or {}
    if not lic_trunk.get("sha256"):
        raise HarnessRefusal(
            f"{TRUNK_NOT_BOUND}: the licence pins cache bytes but names no trunk sha256"
        )
    tpath = Path(trunk) if trunk else Path(lic_trunk["path"])
    if not tpath.is_file():
        raise HarnessRefusal(
            f"{TRUNK_NOT_BOUND}: the licensed trunk {tpath} is not on disk, so the checkpoint "
            "that produced these features cannot be re-verified (FACT-0392)"
        )
    got = AFC.sha256_file(tpath)
    if got != lic_trunk["sha256"]:
        size_note = (
            " Its byte size MATCHES the licensed one, which is exactly why size is not the test: "
            "official_f0 and stabledet_f0 are the same size with different sha256 (FACT-0392)."
            if int(tpath.stat().st_size) == int(lic_trunk.get("bytes", -1)) else
            f" Size {tpath.stat().st_size} against licensed {lic_trunk.get('bytes')}."
        )
        raise HarnessRefusal(
            f"{TRUNK_SHA_CHANGED}: {tpath} hashes {got[:16]}..., the licence pins "
            f"{lic_trunk['sha256'][:16]}... for role {lic_trunk.get('role')!r}.{size_note}"
        )
    return {"licence": str(p), "crops_pinned": len(pinned), "verified": len(crops),
            "trunk_role": lic_trunk.get("role"), "trunk_sha256": lic_trunk["sha256"],
            "trunk_verified_by": "sha256"}


# ======================================================================================
# 1b. SCHEMA VALIDATION ONLY - which is NOT a licence, and must never become one
# ======================================================================================

def schema_report(cache_dir: Path, crops: list[str] | None = None) -> dict:
    """Validate the pair-and-role SCHEMA of a tapped cache. This licenses NOTHING.

    WHY THIS EXISTS SEPARATELY FROM THE GATE. `FACT-0403` measured the exact case it is for: the
    archived P36 cache has PROVEN structure - passivity, pair-and-role schema, zero integrity
    failures, positional features round-tripping at exactly 0.0 - and an INVALID parity verdict,
    because the reference it was compared against was downstream of a secondary-model logit blend
    at weight 0.15. Structure is therefore established and learned quantities are not. Reading
    such a cache to check that this harness can address it is legitimate; training on it is not.

    The report deliberately carries NO ``passed`` key. ``write_licence`` requires one, so a
    schema check cannot be laundered into a training licence by a caller who passes the wrong
    dict - the refusal is structural rather than a convention.
    """
    cache_dir = Path(cache_dir)
    names = list(crops) if crops else sorted(p.stem for p in cache_dir.glob("*.npz"))
    out: dict = {
        "heartbeat": "ASSOC_CACHE_SCHEMA_ONLY",
        "schema_version": CACHE_SCHEMA_VERSION,
        "layout": "pair_and_role",
        "cache_dir": str(cache_dir),
        "licenses_training": False,
        "why_not_a_licence": (
            "Schema validation proves STRUCTURE and nothing about learned quantities. It does "
            "not run the deployed head, does not bind a trunk and does not establish probability "
            "parity - FACT-0403 records a cache with sound structure whose parity verdict is "
            "INVALID. Training requires `gate` with a manifest and a Gate-1 receipt."
        ),
        "crops": [],
        "schema_ok": False,
        "refusals": [],
    }
    if not names:
        out["refusals"].append(
            "no cache files found - a schema check that reads nothing is not a check (FACT-0387)")
        return out
    for crop in names:
        path = cache_dir / f"{crop}.npz"
        row: dict = {"crop": crop, "ok": False, "reasons": []}
        try:
            data = load_cache(path)
            binding = AFC.crop_binding(crop, data, DEPLOYED_FLOOR,
                                       bytes_on_disk=path.stat().st_size)
            index = role_features(data, crop)
            row.update({
                "ok": True,
                "nodes": binding["nodes"],
                "pairs": binding["pairs"],
                "frames": len(binding["frames"]),
                "window": binding["window"],
                "role_nodes": binding["storage"]["role_nodes"],
                "feat_dim": binding["feature_dim"],
                "pos_dim": binding["pos_dim"],
                "downsample": binding["downsample"],
                "band_a_rows": binding["bands"]["band_a"]["rows"],
                "band_b_rows": binding["bands"]["band_b"]["rows"],
                "window_dependence": binding["window_dependence"],
                # The harness's OWN addressing, exercised rather than assumed: resolve every
                # recorded band edge through the (pair, role) index and require the role rows to
                # carry the ids they were computed for.
                "role_lookup": _exercise_role_lookup(data, index),
            })
        except HarnessRefusal as exc:
            row["reasons"].append(str(exc))
        except AFC.Reject as exc:
            row["reasons"].append(str(exc))
        out["crops"].append(row)
    out["schema_ok"] = bool(out["crops"]) and all(c["ok"] for c in out["crops"])
    return out


def _exercise_role_lookup(cache: dict, index: RoleFeatureIndex) -> dict:
    """Resolve every recorded band edge through the role index, and prove the roles differ.

    The second half is the part a shape check cannot do. If a node appears in both roles, the two
    vectors the index hands back MUST differ - ``_TemporalAttention`` mixes across the window, so
    they come from different forward passes (`FACT-0402`). Identical vectors would mean
    frame-keyed CONTENT inside a pair-and-role container.
    """
    src = np.concatenate([np.asarray(cache["band_a_source_id"]),
                          np.asarray(cache["band_b_source_id"])]).astype(np.int64)
    tgt = np.concatenate([np.asarray(cache["band_a_target_id"]),
                          np.asarray(cache["band_b_target_id"])]).astype(np.int64)
    s_rows, t_rows = index.rows_for_edges(src, tgt)
    role = np.asarray(cache["role_role"]).astype(np.int64)
    if not (np.all(role[s_rows] == ROLE_SRC) and np.all(role[t_rows] == ROLE_TGT)):
        raise HarnessRefusal(
            f"{ROLE_LOOKUP_DISAGREES}: {index.crop}: an edge resolved to a role row filed under "
            "the opposite role, so the head would be fed a target vector as a source"
        )
    # Nodes served in BOTH roles somewhere in this cache, and by how much the vectors differ.
    src_row_of = dict(zip(index.role_gid[s_rows].tolist(), s_rows.tolist()))
    tgt_row_of = dict(zip(index.role_gid[t_rows].tolist(), t_rows.tolist()))
    both = sorted(set(src_row_of) & set(tgt_row_of))
    delta = 0.0
    if both:
        lhs = index.feat[[src_row_of[g] for g in both]].astype(np.float64)
        rhs = index.feat[[tgt_row_of[g] for g in both]].astype(np.float64)
        delta = float(np.abs(lhs - rhs).max())
        if delta == 0.0:
            raise HarnessRefusal(
                f"{FRAME_KEYED_CACHE}: {index.crop}: every node served in both roles returns a "
                "BIT-IDENTICAL vector in each. A temporal-attention trunk cannot do that "
                "(FACT-0402); a frame-keyed writer wearing this schema does exactly that"
            )
    return {"edges_resolved": int(src.shape[0]),
            "nodes_served_in_both_roles": len(both),
            "max_abs_delta_between_the_two_roles": delta}


# ======================================================================================
# 2. GROUPING - targets are the unit, crops are the split, and the fold is one embryo
# ======================================================================================

def embryo_of(crop: str) -> str:
    return str(crop).split("_", 1)[0]


def assert_group_integrity(crops: np.ndarray, targets: np.ndarray,
                           tr: np.ndarray, te: np.ndarray, tag: str) -> None:
    """Falsifier (b) of PKT-0034, checked mechanically on every fold rather than assumed."""
    if len(tr) == 0 or len(te) == 0:
        raise HarnessRefusal(f"{tag}: degenerate split ({len(tr)} train, {len(te)} validation)")
    problems: list[str] = []
    # BOTH conditions are reported, never just the first. The target condition is the contract
    # (falsifier (b): a target's own candidates must never be split); the crop condition is the
    # stronger sufficient one. Raising on whichever happened to be checked first would leave a
    # reader unable to tell which of the two guarantees actually broke.
    key_tr = set(zip(crops[tr].tolist(), targets[tr].tolist()))
    key_te = set(zip(crops[te].tolist(), targets[te].tolist()))
    straddle = key_tr & key_te
    if straddle:
        problems.append(f"{len(straddle)} target(s) straddle the split - a target's own "
                        f"candidates must never be split, e.g. {sorted(straddle)[:3]}")
    crop_overlap = set(crops[tr]) & set(crops[te])
    if crop_overlap:
        problems.append(f"crop on both sides of the split - {sorted(crop_overlap)[:5]}")
    if problems:
        raise HarnessRefusal(f"{tag}: " + "; ".join(problems))


def grouping_block(crops: np.ndarray, cv_kind: str, n_splits: int) -> dict:
    embryos = sorted({embryo_of(c) for c in set(crops.tolist())})
    return {
        "unit": "crop",
        "cv": cv_kind,
        "n_splits": n_splits,
        "n_crops": len(set(crops.tolist())),
        "embryos": embryos,
        "embryo_held_out": len(embryos) > 1,
        "note": (
            "Crop-grouped folds are WITHIN-embryo generalisation. Each fold of this campaign is a "
            "single embryo (fold 0 all 44b6, fold 1 all 6bba), so no split here holds an embryo "
            "out and no cross-embryo transfer claim may be made from this run."
        ),
    }


def make_splitter(kind: str, n_splits: int):
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

    if kind == "GroupKFold":
        return GroupKFold(n_splits=n_splits)
    if kind == "LeaveOneCropOut":
        return LeaveOneGroupOut()
    raise HarnessRefusal(f"unknown cv kind {kind!r} - only crop-grouped splits are permitted")


# ======================================================================================
# 3. ABSTENTION AND THE SINGLE-CANDIDATE LEDGER
# ======================================================================================

@dataclass
class SingleCandidateLedger:
    """Every single-candidate movement, by IDENTITY. A netted figure cannot be produced from it.

    A decidable single-candidate target's one candidate IS its true parent, so the baseline is
    correct on all of them by arithmetic (FACT-0386) and the only way a model can lose one is by
    abstaining. ``regressions`` are the losses that actually happened; ``shadow_regressions`` are
    the ones the ``preserve_single_candidate`` constraint prevented - reported so that preservation
    by construction prices the risk instead of hiding it.
    """

    preserved_by_construction: bool
    n_targets: int = 0
    regressions: list[dict] = field(default_factory=list)
    shadow_regressions: list[dict] = field(default_factory=list)

    def record(self, crop: str, target: int, source: int, lost: bool, would_abstain: bool) -> None:
        self.n_targets += 1
        if lost:
            self.regressions.append({"crop": crop, "target": int(target), "source": int(source),
                                     "reason": "abstained"})
        elif would_abstain:
            self.shadow_regressions.append({"crop": crop, "target": int(target),
                                            "source": int(source),
                                            "reason": "would have abstained; prevented by "
                                                      "preserve_single_candidate"})

    def summary(self) -> dict:
        if self.preserved_by_construction and self.regressions:
            raise HarnessRefusal(
                "preserve_single_candidate is on yet single-candidate targets were lost - the "
                "constraint is not being applied where it is claimed"
            )
        out = {
            "n": self.n_targets,
            "preserved_by_construction": self.preserved_by_construction,
            "top1": 1.0 - (len(self.regressions) / self.n_targets) if self.n_targets else None,
            "n_regressions": len(self.regressions),
            "regressions": self.regressions,
            "n_shadow_regressions": len(self.shadow_regressions),
            "shadow_regressions": self.shadow_regressions,
            "side_b_of_the_two_sided_bar": (
                "INERT - preserved by construction; the shadow count is what abstention would "
                "have cost" if self.preserved_by_construction else
                "LIVE - abstention is permitted on single-candidate targets"
            ),
        }
        # The guard PKT-0034 falsifier (c) exists for: a count without its identities is a netted
        # figure wearing a list's clothes, and it must be impossible to emit.
        if out["n_regressions"] != len(out["regressions"]):
            raise HarnessRefusal("single-candidate regression count and identity list disagree")
        if out["n_shadow_regressions"] != len(out["shadow_regressions"]):
            raise HarnessRefusal("single-candidate shadow count and identity list disagree")
        return out


def decide(table: pl.DataFrame, score_col: str, abstain_col: str | None,
           preserve_single: bool) -> tuple[dict, SingleCandidateLedger, dict]:
    """Target-wise argmax over ``{candidates} u {NULL}``, on the frozen decidable surface.

    Returns ``(per_target_correct, ledger, aggregate)``. The candidate ordering is the frozen
    contract's rule 4 - ties break by LOWER SOURCE INDEX - reproduced here with the same lexsort as
    ``assoc_parent_dataset.evaluate`` so the two cannot diverge silently. Abstention resolves ties
    TO THE NULL, mirroring the deployed strict ``prob > threshold``; with ``abstain_col`` None the
    null score is -inf and the decision is exactly the frozen argmax.
    """
    dec = table.filter(pl.col("true_parent_is_candidate") == 1)
    ledger = SingleCandidateLedger(preserved_by_construction=preserve_single)
    per_target: dict[tuple[str, int], int] = {}
    contested_n = contested_hits = 0
    abstained_contested = abstained_single = 0
    margins: list[float] = []
    for _key, group in dec.group_by("crop", "target"):
        s = group[score_col].to_numpy()
        y = group["is_true_parent"].to_numpy()
        src = group["source"].to_numpy()
        order = np.lexsort((src, -s))
        best = order[0]
        null = float(group[abstain_col][0]) if abstain_col else NEVER
        single = len(s) == 1
        would_abstain = bool(s[best] <= null)
        allowed = not (single and preserve_single)
        abstained = would_abstain and allowed
        hit = bool((not abstained) and y[best] == 1)
        crop = str(group["crop"][0])
        target = int(group["target"][0])
        per_target[(crop, target)] = int(hit)
        if single:
            ledger.record(crop, target, int(src[best]), lost=not hit, would_abstain=would_abstain)
            abstained_single += int(abstained)
        else:
            contested_n += 1
            contested_hits += int(hit)
            abstained_contested += int(abstained)
            margins.append(float(s[order[0]] - s[order[1]]) * (1.0 if hit else -1.0))
    n_targets = len(per_target)
    aggregate = {
        "decidable_targets": n_targets,
        "parent_top1": sum(per_target.values()) / max(n_targets, 1),
        "true_parent_margin_mean": float(np.mean(margins)) if margins else None,
        "true_parent_margin_median": float(np.median(margins)) if margins else None,
        "contested": {
            "n": contested_n,
            "share": contested_n / max(n_targets, 1),
            "top1": contested_hits / max(contested_n, 1) if contested_n else None,
            "errors": contested_n - contested_hits,
            "abstained": abstained_contested,
        },
        "abstentions": {"contested": abstained_contested, "single_candidate": abstained_single},
        "degenerate_for_ranking": contested_n == 0,
    }
    return per_target, ledger, aggregate


def assert_surface_equivalence(table: pl.DataFrame, score_col: str) -> dict:
    """With abstention off this harness MUST reproduce the frozen surface, per target.

    Asserted rather than asserted-in-prose: if it ever diverges, every number the harness emits is
    incomparable to FACT-0381 and to the LEVER-0039 arms, and the run must stop.
    """
    frozen = evaluate(table, score_col)
    frozen_map = frozen.pop("per_target_correct")
    mine, _ledger, agg = decide(table, score_col, None, preserve_single=False)
    if mine != frozen_map:
        differing = [k for k in set(mine) | set(frozen_map) if mine.get(k) != frozen_map.get(k)]
        raise HarnessRefusal(
            f"harness decision differs from the frozen surface on {len(differing)} targets, "
            f"e.g. {differing[:3]} - the two are not the same evaluation"
        )
    for key in ("decidable_targets", "parent_top1", "true_parent_margin_median"):
        if agg[key] != frozen[key]:
            raise HarnessRefusal(f"harness surface differs on {key}: {agg[key]!r} vs {frozen[key]!r}")
    if (agg["contested"]["n"], agg["contested"]["top1"]) != (
            frozen["contested"]["n"], frozen["contested"]["top1"]):
        raise HarnessRefusal("harness surface differs on the contested split")
    return {"heartbeat": "SURFACE_EQUIVALENT", "targets": len(frozen_map)}


# ======================================================================================
# 4. THE THREE SCORER CLASSES
# ======================================================================================

@dataclass
class AbstainPolicy:
    """Declared before fitting, never searched after seeing a validation number.

    ``none``           the null score is -inf: abstention is impossible (the LEVER-0039 mode).
    ``fixed``          a constant threshold, mirroring the deployed rule's 0.5 (FACT-0369).
    ``train_quantile`` the q-quantile of the winning-candidate score over TRAINING targets only,
                       refitted per fold, never touching validation.
    """

    kind: str = "none"
    tau: float = DEPLOYED_FLOOR
    q: float = 0.05

    def as_dict(self) -> dict:
        return {"kind": self.kind, "tau": self.tau, "q": self.q}

    def fit(self, train_winning_scores: np.ndarray) -> float:
        if self.kind == "none":
            return NEVER
        if self.kind == "fixed":
            return float(self.tau)
        if self.kind == "train_quantile":
            if len(train_winning_scores) == 0:
                raise HarnessRefusal("train_quantile abstention has no training targets to fit on")
            return float(np.quantile(train_winning_scores, self.q))
        raise HarnessRefusal(f"unknown abstention policy {self.kind!r}")


@dataclass
class ContextContract:
    """A DECLARED contextual feature contract. This harness validates it; it does not invent it.

    ``LEVER-0034`` owns identifying the HOCT contract. Everything here is checked against the
    licensed cache, so a declaration that does not describe the cache is a refusal rather than a
    silently different experiment.
    """

    name: str
    dim: int | None
    pair_builder: str = "concat_src_tgt"          # concat_src_tgt | diff | concat_diff
    extra_features: list[str] = field(default_factory=list)
    has_null_head: bool = False
    dim_from_cache: bool = False

    BUILDERS = {"concat_src_tgt", "diff", "concat_diff"}

    @classmethod
    def from_dict(cls, d: dict | None) -> "ContextContract":
        if not d:
            raise HarnessRefusal(
                "a contextual model was requested with no declared contract. The contract comes "
                "from LEVER-0034 or is ours by construction and declared; falling back to the "
                "representation-free features would answer a different question."
            )
        missing = [k for k in ("name", "dim") if k not in d]
        if missing:
            raise HarnessRefusal(f"contextual contract is missing {missing}")
        # `"dim": "from_cache"` is the CONTRACT-OURS-BY-CONSTRUCTION case: the representation is
        # the licensed cache's own frozen-trunk node features, whose width we do not get to
        # choose. It is a weaker declaration than a number and is recorded as such in the payload,
        # never silently. A numeric dim stays a hard equality check.
        raw = d["dim"]
        adopt = isinstance(raw, str) and raw == "from_cache"
        if not adopt and isinstance(raw, str):
            raise HarnessRefusal(f"contract dim must be an integer or 'from_cache', got {raw!r}")
        c = cls(name=str(d["name"]), dim=None if adopt else int(raw),
                pair_builder=str(d.get("pair_builder", "concat_src_tgt")),
                extra_features=list(d.get("extra_features", [])),
                has_null_head=bool(d.get("has_null_head", False)),
                dim_from_cache=adopt)
        if c.pair_builder not in cls.BUILDERS:
            raise HarnessRefusal(f"unknown pair_builder {c.pair_builder!r}")
        bad = [f for f in c.extra_features if f not in FEATURES]
        if bad:
            raise HarnessRefusal(f"contract names features outside the frozen surface: {bad}")
        return c

    def width(self) -> int:
        if self.dim is None:
            raise HarnessRefusal(f"contract {self.name!r} has no resolved dim yet")
        mult = {"concat_src_tgt": 2, "diff": 1, "concat_diff": 3}[self.pair_builder]
        return mult * self.dim + len(self.extra_features)

    def validate_against_cache(self, index: "RoleFeatureIndex") -> None:
        if self.dim is None:
            self.dim = int(index.dim)
            return
        if int(index.dim) != self.dim:
            raise HarnessRefusal(
                f"contract {self.name!r} declares dim {self.dim} but the licensed cache carries "
                f"{index.dim} - the declaration does not describe this cache"
            )

    def build(self, role_index_by_crop: dict, table: pl.DataFrame) -> np.ndarray:
        """Assemble the pair features, taking each node's ROLE-SPECIFIC vector.

        The source vector comes from the SOURCE role of the pair the candidate belongs to and the
        target vector from that same pair's TARGET role. Under the retired layout there was one
        vector per node and this line silently picked the wrong one for every source in every
        pair (t, t+1) with t >= 1 - `FACT-0402`, measured at 66 of 173 role-nodes and a max abs
        delta of 0.0123 against a 1e-4 tolerance.
        """
        crops = table["crop"].to_numpy()
        src = table["source"].to_numpy().astype(np.int64)
        tgt = table["target"].to_numpy().astype(np.int64)
        for crop in np.unique(crops):
            index = role_index_by_crop.get(str(crop))
            if index is None:
                raise HarnessRefusal(f"no licensed cache features for crop {crop!r}")
            self.validate_against_cache(index)
        out = np.empty((table.height, self.width()), dtype=np.float32)
        extra = (table.select(self.extra_features).to_numpy().astype(np.float32)
                 if self.extra_features else np.empty((table.height, 0), dtype=np.float32))
        for crop in np.unique(crops):
            index = role_index_by_crop[str(crop)]
            m = crops == crop
            s_rows, t_rows = index.rows_for_edges(src[m], tgt[m])
            fs, ft = index.feat[s_rows], index.feat[t_rows]
            if not (np.all(np.isfinite(fs)) and np.all(np.isfinite(ft))):
                raise HarnessRefusal(
                    f"{crop}: a candidate resolves to a role row holding a non-finite feature"
                )
            if self.pair_builder == "concat_src_tgt":
                block = np.concatenate([fs, ft], axis=1)
            elif self.pair_builder == "diff":
                block = ft - fs
            else:
                block = np.concatenate([fs, ft, ft - fs], axis=1)
            out[m] = np.concatenate([block, extra[m]], axis=1)
        return out


def make_estimator(model_class: str, head: str | None = None):
    """The three scorer classes. Hyperparameters are the LEVER-0039 pre-registered defaults.

    SAID PRECISELY, BECAUSE "LISTWISE" IS EASY TO OVERCLAIM. The ``linear`` and ``tree`` classes are
    fitted POINTWISE on candidate rows and DECIDED LISTWISE - the argmax runs over a target's
    candidates plus the null, which is the decision the deployed pipeline makes (``FACT-0369``). A
    genuinely listwise LOSS - per-target cross-entropy over candidates and null - is deliberately
    not implemented on the nine representation-free features, because ``FACT-0386`` established
    that the head is not the constraint there and a new loss on a killed feature set would
    re-litigate a dead lever. Any listwise or contextual head belongs behind the ``head``
    entrypoint of a declared contract, where it is fitted on the representation that differs.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if model_class == "linear":
        return Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(**LINEAR_KW))])
    if model_class == "tree":
        return HistGradientBoostingClassifier(**TREE_KW)
    if model_class == "contextual":
        if head:
            return resolve_callable(head)()
        return Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(**LINEAR_KW))])
    if model_class == "declared":
        # ADDITIVE, PKT-0038. A scorer whose FITTING PROCEDURE is the thing under test - an
        # iterative hard-negative curriculum, an alternate-seed secondary, or a model frozen on
        # one fold and applied unchanged to another - cannot be expressed by `linear`/`tree`,
        # whose hyperparameters are fixed constants, and must NOT be routed through `contextual`,
        # which means "cache-derived representation" and is cache-gated for that reason. So this
        # class carries a DECLARED estimator factory and no cache claim: `needs_cache()` stays
        # False and every existing path above is byte-unchanged. The head is a `module:function`
        # entrypoint resolved through the same refusal-not-fallback rule as the contextual head,
        # so an unresolvable declaration stops the run instead of silently fitting something else.
        if not head:
            raise HarnessRefusal(
                "model class 'declared' requires a `head` entrypoint - an undeclared fitting "
                "procedure is not a comparable arm"
            )
        return resolve_callable(head)()
    raise HarnessRefusal(f"unknown model class {model_class!r}")


# ======================================================================================
# 5. THE RUN
# ======================================================================================

@dataclass
class ModelSpec:
    tag: str
    model_class: str
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    abstain: AbstainPolicy = field(default_factory=AbstainPolicy)
    contract: dict | None = None
    head: str | None = None
    population: str = "alldec"                    # alldec | contested

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        return cls(
            tag=str(d["tag"]),
            model_class=str(d["class"]),
            features=list(d.get("features", FEATURES)),
            abstain=AbstainPolicy(**d.get("abstain", {})),
            contract=d.get("contract"),
            head=d.get("head"),
            population=str(d.get("population", "alldec")),
        )

    def needs_cache(self) -> bool:
        return self.model_class == "contextual"


def fit_out_of_fold(dec: pl.DataFrame, spec: ModelSpec, x_all: np.ndarray,
                    splitter, tag: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Out-of-fold candidate scores and per-target null scores. Every fold is crop-grouped."""
    y_all = dec["is_true_parent"].to_numpy().astype(np.int64)
    crops_all = dec["crop"].to_numpy()
    targets_all = dec["target"].to_numpy().astype(np.int64)
    n_dec = dec["n_dec"].to_numpy().astype(np.int64)
    fit_mask = np.ones(len(y_all), bool) if spec.population == "alldec" else (n_dec > 1)

    score = np.full(len(y_all), np.nan)
    null = np.full(len(y_all), np.nan)
    folds: list[dict] = []
    for i, (tr, te) in enumerate(splitter.split(x_all, y_all, groups=crops_all)):
        assert_group_integrity(crops_all, targets_all, tr, te, f"{tag}/fold{i}")
        fit_rows = tr[fit_mask[tr]]
        if len(fit_rows) == 0 or len(np.unique(y_all[fit_rows])) < 2:
            raise HarnessRefusal(f"{tag}/fold{i}: training population is empty or one-class")
        model = make_estimator(spec.model_class, spec.head)
        model.fit(x_all[fit_rows], y_all[fit_rows])
        score[te] = model.predict_proba(x_all[te])[:, 1]

        train_scores = model.predict_proba(x_all[tr])[:, 1]
        winning = _winning_per_target(crops_all[tr], targets_all[tr], train_scores)
        tau = spec.abstain.fit(winning)
        null[te] = tau
        folds.append({"fold": i, "n_train_rows": int(len(fit_rows)), "n_val_rows": int(len(te)),
                      "abstain_tau": None if tau == NEVER else float(tau),
                      "val_crops": sorted(set(crops_all[te].tolist()))})
    if np.isnan(score).any() or np.isnan(null).any():
        raise HarnessRefusal(f"{tag}: rows without an out-of-fold score")
    return score, null, folds


def _winning_per_target(crops: np.ndarray, targets: np.ndarray, scores: np.ndarray) -> np.ndarray:
    best: dict[tuple[str, int], float] = {}
    for c, t, s in zip(crops.tolist(), targets.tolist(), scores.tolist()):
        k = (c, int(t))
        if s > best.get(k, -np.inf):
            best[k] = s
    return np.asarray(list(best.values()), dtype=np.float64)


def run_model(dec: pl.DataFrame, spec: ModelSpec, splitter, baseline_map: dict,
              contested_keys: set, preserve_single: bool,
              role_index_by_crop: dict | None) -> dict:
    """One model, fitted out of fold and scored through the harness surface."""
    if spec.model_class == "contextual":
        contract = ContextContract.from_dict(spec.contract)
        if role_index_by_crop is None:
            raise HarnessRefusal(
                f"{spec.tag}: a contextual model needs the licensed cache and none was gated"
            )
        x_all = contract.build(role_index_by_crop, dec)
        contract_block = {"name": contract.name, "dim": contract.dim,
                          "dim_declared": "from_cache" if contract.dim_from_cache else contract.dim,
                          "pair_builder": contract.pair_builder,
                          "extra_features": contract.extra_features,
                          "width": contract.width(), "has_null_head": contract.has_null_head}
    else:
        missing = [f for f in spec.features if f not in dec.columns]
        if missing:
            raise HarnessRefusal(f"{spec.tag}: features absent from the frozen surface: {missing}")
        x_all = dec.select(spec.features).to_numpy().astype(np.float64)
        contract_block = None

    score, null, folds = fit_out_of_fold(dec, spec, x_all, splitter, spec.tag)
    scored = dec.with_columns([pl.Series("_score", score), pl.Series("_null", null)])

    contested_rows = np.array([(c, int(t)) in contested_keys
                               for c, t in zip(dec["crop"].to_numpy().tolist(),
                                               dec["target"].to_numpy().tolist())])
    if contested_rows.any() and float(np.std(score[contested_rows])) == 0.0:
        raise HarnessRefusal(
            f"{spec.tag}: constant score on contested rows - the argmax is source order, not skill"
        )

    abstain_col = None if spec.abstain.kind == "none" else "_null"
    per_target, ledger, agg = decide(scored, "_score", abstain_col, preserve_single)
    conv_all = parent_conversions(baseline_map, per_target)
    conv_cont = parent_conversions(
        {k: v for k, v in baseline_map.items() if k in contested_keys},
        {k: v for k, v in per_target.items() if k in contested_keys},
    )
    conv_cont["crop_paired_bootstrap"] = crop_paired_bootstrap(baseline_map, per_target,
                                                               contested_keys)
    single_keys = set(baseline_map) - contested_keys
    conv_single = parent_conversions(
        {k: v for k, v in baseline_map.items() if k in single_keys},
        {k: v for k, v in per_target.items() if k in single_keys},
    ) if single_keys else None

    return {
        "tag": spec.tag,
        "model_class": spec.model_class,
        "population": spec.population,
        "features": None if contract_block else spec.features,
        "contract": contract_block,
        "abstention": spec.abstain.as_dict(),
        "folds": folds,
        "surface": agg,
        "single_candidate": ledger.summary(),
        "conversions": {"all_decidable": conv_all, "contested": conv_cont,
                        "single_candidate": conv_single},
        "calibration_oof": calibration(dec["is_true_parent"].to_numpy().astype(np.int64), score),
        "per_target_correct": per_target,
    }


def harness_verdict(model_result: dict, deployed_contested_top1: float | None,
                    full_chain: dict | None, degenerate: bool) -> dict:
    """Advisory, never automatic, and deliberately two-sided.

    Side (a): beat the deployed contested top-1 with a favourable crop-paired interval.
    Side (b): do not regress single-candidate targets - live for the first time now that
    abstention exists (FACT-0386), and satisfied only by identities, never by a net.
    """
    blockers: list[str] = []
    surf = model_result["surface"]
    single = model_result["single_candidate"]
    if degenerate:
        blockers.append("this fold is DEGENERATE for ranking (FACT-0381/FACT-0382): contested "
                        "top-1 is 1.0 for every model, so no claim may be made on it")
    if deployed_contested_top1 is not None and surf["contested"]["top1"] is not None:
        if surf["contested"]["top1"] <= deployed_contested_top1:
            blockers.append("contested top-1 does not beat the deployed bar")
    boot = model_result["conversions"]["contested"].get("crop_paired_bootstrap", {})
    if not boot.get("favourable"):
        blockers.append("crop-paired interval on contested top-1 is not favourable")
    if single["n_regressions"] > 0:
        blockers.append(
            f"{single['n_regressions']} single-candidate target(s) regressed: "
            + ", ".join(f"{r['crop']}#{r['target']}" for r in single["regressions"][:5])
            + (" ..." if single["n_regressions"] > 5 else "")
        )
    if full_chain is None:
        blockers.append("no full-chain arms supplied: the FACT-0376 quantity is unmeasured and a "
                        "pre-ILP ranking gain survives the solver only as a scoring prior "
                        "(FACT-0364)")
    elif not full_chain.get("verdict", {}).get("promotable"):
        blockers.extend(full_chain.get("verdict", {}).get("blockers", []))
    return {
        "promotable": not blockers,
        "blockers": blockers,
        "side_b_status": single["side_b_of_the_two_sided_bar"],
        "single_candidate_shadow_regressions": single["n_shadow_regressions"],
    }


def run_harness(*, table: pl.DataFrame, models: list[ModelSpec], fold: int,
                cv_kind: str = "GroupKFold", n_splits: int = 5,
                preserve_single: bool = True, cache_gate: dict | None = None,
                role_index_by_crop: dict | None = None,
                chain_arms: dict | None = None, summarise=None,
                allow_degenerate: bool = False) -> dict:
    """Fit and report every declared model into ONE comparable payload."""
    equivalence = assert_surface_equivalence(table, "prob")
    base = evaluate(table, "prob")
    baseline_map = base.pop("per_target_correct")
    degenerate = bool(base["degenerate_for_ranking"])
    if degenerate and not allow_degenerate:
        raise HarnessRefusal(
            "this fold has ZERO contested targets: top-1 is 1.0 for any model, so it cannot "
            "falsify a ranker (FACT-0381, FACT-0382). Pass allow_degenerate for diagnostics only."
        )

    dec = table.filter(pl.col("true_parent_is_candidate") == 1)
    counts = dec.group_by(["crop", "target"]).agg(pl.len().alias("n_dec"))
    dec = dec.join(counts, on=["crop", "target"], how="left")
    contested_keys = {(r[0], int(r[1])) for r in
                      counts.filter(pl.col("n_dec") > 1).select(["crop", "target"]).rows()}

    needs_cache = any(m.needs_cache() for m in models)
    if needs_cache and not (cache_gate and cache_gate.get("passed")):
        raise HarnessRefusal(
            "a contextual model was declared but the cache gate has not passed: a head trained on "
            "an unlicensed cache measures the cache (FACT-0387), so nothing is fitted"
        )
    if cache_gate is not None and not cache_gate.get("passed"):
        raise HarnessRefusal("the cache gate did not pass - refusing to train any arm, because a "
                             "licensed arm and an unlicensed arm are not comparable")

    splitter = make_splitter(cv_kind, n_splits)
    crops_all = dec["crop"].to_numpy()
    results, per_target_maps = [], {}
    for spec in models:
        r = run_model(dec, spec, splitter, baseline_map, contested_keys, preserve_single,
                      role_index_by_crop)
        per_target_maps[spec.tag] = r.pop("per_target_correct")
        if chain_arms and spec.tag in chain_arms:
            from assoc_report import build_report
            arm = chain_arms[spec.tag]
            r["full_chain"] = build_report(
                model=spec.tag, fold=fold, control=arm["control"], candidate=arm["candidate"],
                summarise=summarise, conversions=r["conversions"]["all_decidable"],
                notes=arm.get("notes", ""),
                # An arm that supplies no binding is not "unconstrained": build_report writes a
                # refusing block and the verdict blocks on it (FACT-0432).
                control_binding=arm.get("control_binding"),
            )
        else:
            r["full_chain"] = None
        r["verdict"] = harness_verdict(r, base["contested"]["top1"], r["full_chain"], degenerate)
        results.append(r)

    payload = {
        "schema_version": 1,
        "heartbeat": "ASSOC_TRAIN_HARNESS_COMPLETE",
        "fold": fold,
        "surface_equivalence": equivalence,
        "cache_gate": cache_gate or {
            "status": "NOT_REQUIRED",
            "note": "no model declared cache-derived features, so no contextual claim exists in "
                    "this run; a contextual arm would have required a passing Gate-1 licence",
        },
        "grouping": grouping_block(crops_all, cv_kind, n_splits),
        "preserve_single_candidate": preserve_single,
        "deployed_baseline": base,
        "seed": SEED,
        "models": results,
        "surface_sha256_16": {
            name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()[:16]
            for name in ("assoc_parent_dataset.py", "assoc_report.py", "assoc_train_harness.py")
        },
    }
    channels = [set(m.keys()) for m in results]
    if channels and any(c != channels[0] for c in channels):
        raise HarnessRefusal(
            "model classes emitted different channels - PKT-0034 falsifier (d): the comparison "
            "this harness exists to enable would not exist"
        )
    return payload


# ======================================================================================
# 6. CLI
# ======================================================================================

def apply_restriction_to_payload(payload: dict, resolution: dict | None) -> dict:
    """FACT-0451 rule 4 - the VERDICT LAYER refuses promotion; it does not merely label a report.

    A label on a payload nobody blocks on is prose again, which is the state FACT-0451 found on
    this exact surface. So the resolution travels INTO the payload - a downstream reader cannot
    receive the number without the mark - and every model verdict is forced non-promotable when
    the arm is restricted.

    It is a named function rather than eight lines inside ``cmd_train`` so the rule can be proved
    by a test without building a whole harness run. An unproved gate is the thing being repaired.
    """
    payload["binding_restriction"] = resolution
    if resolution is None or resolution.get(spec_restriction.FIELD, False):
        return payload
    names = ", ".join(sorted({str(h.get("name_at_registration", "?"))
                              for h in resolution.get("restricted_hits", [])})) or "unknown"
    blocker = (
        f"offline_scoreable is FALSE ({names}, FACT-0451): no LOEO control we can build can show "
        f"the held-out fold was held out of THEIR training, so this number is interpretable as "
        f"neither a pass nor a fail. SUBMISSION-ONLY JUDGEMENT."
    )
    for m in payload.get("models", []):
        verdict = m.setdefault("verdict", {})
        verdict["promotable"] = False
        verdict["blockers"] = [blocker] + list(verdict.get("blockers", []))
    return payload


def _load_spec(path: Path) -> dict:
    """Read a spec AND enforce the FACT-0451 restriction at the schema boundary.

    This is the surface FACT-0451 named as still-absent: without it, an arm using a checkpoint
    whose training data cannot be established emits an offline number carrying no mark, which is
    indistinguishable in form from a valid one. Rules 1-3 fire here (identity by hash, mandatory
    declaration, non-overridable derivation); rule 4 - refusing PROMOTION - fires wherever the
    resolution is consumed, which is why the resolution is stashed on the returned dict rather
    than discarded.
    """
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    spec["_fact_0451_resolution"] = spec_restriction.enforce_spec(spec, path=path)
    return spec


def cmd_schema(args) -> int:
    """Validate a cache's pair-and-role SCHEMA. Prints, loudly, that it licenses nothing."""
    report = schema_report(Path(args.cache_dir), args.crops)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    for c in report["crops"]:
        if c["ok"]:
            rl = c["role_lookup"]
            print(f"  {c['crop']:<20s} nodes={c['nodes']:>7,} pairs={c['pairs']:>4} "
                  f"role_nodes={c['role_nodes']:>7,} feat_dim={c['feat_dim']} "
                  f"pos_dim={c['pos_dim']} window={c['window']} "
                  f"dual_role={c['window_dependence']['dual_role_nodes']:>6,} "
                  f"dual_delta={c['window_dependence']['max_abs_delta']:.4g} "
                  f"edges_resolved={rl['edges_resolved']:,} "
                  f"both_roles={rl['nodes_served_in_both_roles']:,} "
                  f"role_delta={rl['max_abs_delta_between_the_two_roles']:.4g}", flush=True)
        else:
            print(f"  {c['crop']:<20s} REJECTED <- " + "; ".join(c["reasons"][:2]), flush=True)
    for r in report["refusals"]:
        print(f"  REFUSAL: {r}", flush=True)
    print("ASSOC_CACHE_SCHEMA_OK" if report["schema_ok"] else "ASSOC_CACHE_SCHEMA_REJECTED",
          flush=True)
    print("  TRAINING_LICENCE=NOT_GRANTED - " + report["why_not_a_licence"], flush=True)
    return 0 if report["schema_ok"] else 2


def cmd_gate(args) -> int:
    report = gate_cache(cache_dir=args.cache_dir, preilp=args.preilp, ecb_dir=args.ecb_dir,
                        crops=args.crops, receipt=args.receipt, reproducer=args.reproducer,
                        manifest=args.manifest, expect_trunk_sha=args.expect_trunk_sha,
                        expect_fold=args.expect_fold, expect_trunk_role=args.expect_trunk_role)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    if report.get("trunk"):
        t = report["trunk"]
        print(f"  trunk      role={t['role']} sha256={t['sha256'][:16]}... "
              f"bytes={t['bytes']:,} (identity is the HASH, never the size)", flush=True)
    for c in report["crops"]:
        print(f"  {c['crop']:<20s} nodes={c.get('nodes', 0):>7,} "
              f"frames={c.get('frames', 0):>4} role_nodes={c.get('role_nodes') or 0:>7,} "
              f"passed={c['passed']}"
              + ("" if c["passed"] else "  <- " + "; ".join(c["reasons"][:3])), flush=True)
    for r in report["refusals"]:
        print(f"  REFUSAL: {r}", flush=True)
    if report["passed"]:
        if args.licence:
            write_licence(report, Path(args.licence))
            print(f"  licence written -> {args.licence}", flush=True)
        print("ASSOC_CACHE_GATE_PASSED", flush=True)
        return 0
    print("ASSOC_CACHE_GATE_FAILED - no head may be trained on this cache", flush=True)
    return 2


def cmd_train(args) -> int:
    spec = _load_spec(args.spec)
    table = pl.read_parquet(spec["table"])
    models = [ModelSpec.from_dict(m) for m in spec["models"]]
    crops = sorted(table["crop"].unique().to_list())

    gate, role_index = None, None
    cache_cfg = spec.get("cache")
    if any(m.needs_cache() for m in models) or cache_cfg:
        if not cache_cfg:
            raise HarnessRefusal("a contextual model was declared with no `cache` section")
        if not cache_cfg.get("manifest"):
            raise HarnessRefusal(
                f"{TRUNK_NOT_BOUND}: the spec's `cache` section names no `manifest`, so the "
                "producing trunk would be unnamed. Build one with "
                "`audit_feature_cache.py bind` (FACT-0392)"
            )
        cache_crops = cache_cfg.get("crops") or crops
        licence = verify_licence(Path(cache_cfg["licence"]), Path(cache_cfg["dir"]), cache_crops,
                                 trunk=Path(cache_cfg["trunk"]) if cache_cfg.get("trunk") else None)
        gate = gate_cache(cache_dir=Path(cache_cfg["dir"]), preilp=Path(cache_cfg["preilp"]),
                          ecb_dir=Path(cache_cfg["ecb_dir"]) if cache_cfg.get("ecb_dir") else None,
                          crops=cache_crops, receipt=Path(cache_cfg["receipt"]),
                          reproducer=cache_cfg.get("reproducer", "receipt"),
                          manifest=Path(cache_cfg["manifest"]),
                          # The licence's trunk is the expectation, so a cache rebound to a
                          # different trunk after licensing is refused rather than adopted.
                          expect_trunk_sha=licence["trunk_sha256"],
                          expect_fold=cache_cfg.get("fold"),
                          expect_trunk_role=cache_cfg.get("trunk_role"))
        if not gate["passed"]:
            print("ASSOC_CACHE_GATE_FAILED", json.dumps(gate["refusals"]), flush=True)
            raise HarnessRefusal("cache gate failed at training time - nothing was fitted")
        role_index = {c: role_features(load_cache(Path(cache_cfg["dir"]) / f"{c}.npz"), c)
                      for c in cache_crops}
        table = table.filter(pl.col("crop").is_in(cache_crops))

    summarise = None
    chain_arms = spec.get("chain_arms")
    if chain_arms:
        summarise = resolve_callable(spec["summariser"])
        # THE CONTROL IS BOUND BEFORE IT IS READ. `v["control"]` is still a spec-supplied path -
        # that is unavoidable, something has to name the file - but the path no longer confers
        # identity. `control_binding.bind_control` re-hashes the artifact, the graph it was scored
        # from and its receipt, and matches them against the registry-bound deployed fold control
        # (FACT-0382). An arm that declares no `control_identity` binds to nothing and its verdict
        # is blocked; it is NOT refused here, because a report that records WHY it was refused is
        # worth more than a traceback (FACT-0432).
        import control_binding as _cb
        resolved = {}
        for k, v in chain_arms.items():
            rows = json.loads(Path(v["control"]).read_text(encoding="utf-8"))
            resolved[k] = {
                "control": rows,
                "candidate": json.loads(Path(v["candidate"]).read_text(encoding="utf-8")),
                "notes": v.get("notes", ""),
                "control_binding": _cb.bind_control(
                    control_path=Path(v["control"]),
                    identity=v.get("control_identity"),
                    manifest=Path(v["control_manifest"]) if v.get("control_manifest") else None,
                    n_rows=len(rows),
                ),
            }
        chain_arms = resolved

    payload = run_harness(
        table=table, models=models, fold=int(spec["fold"]),
        cv_kind=spec.get("cv", {}).get("kind", "GroupKFold"),
        n_splits=int(spec.get("cv", {}).get("n_splits", 5)),
        preserve_single=bool(spec.get("preserve_single_candidate", True)),
        cache_gate=gate, role_index_by_crop=role_index,
        chain_arms=chain_arms, summarise=summarise,
        allow_degenerate=bool(spec.get("allow_degenerate", False)),
    )
    apply_restriction_to_payload(payload, spec.get("_fact_0451_resolution"))

    out_dir = Path(spec.get("out_dir", "C:/temp/assoc_harness"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"harness_f{spec['fold']}.json"
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print_payload(payload)
    print(f"\nASSOC_TRAIN_HARNESS_COMPLETE -> {out}", flush=True)
    return 0


def print_payload(payload: dict) -> None:
    g = payload["grouping"]
    print(f"\nASSOC_TRAIN_HARNESS fold={payload['fold']} "
          f"cache_gate={payload['cache_gate'].get('passed', payload['cache_gate'].get('status'))}")
    print(f"  grouping   {g['cv']} over {g['n_crops']} crops, embryos {g['embryos']}, "
          f"embryo_held_out={g['embryo_held_out']}")
    print(f"  {payload['surface_equivalence']['heartbeat']} on "
          f"{payload['surface_equivalence']['targets']:,} targets")
    bar = payload["deployed_baseline"]["contested"]["top1"]
    print(f"\n{'arm':38s} {'contested':>10s} {'d vs bar':>9s} {'gain':>5s} {'lost':>5s} "
          f"{'net':>5s} {'churn':>6s} {'sc.reg':>7s} {'sc.shadow':>10s} {'promotable':>11s}")
    for m in payload["models"]:
        c, k, s = m["surface"]["contested"], m["conversions"]["contested"], m["single_candidate"]
        print(f"{m['tag']:38s} {c['top1']:10.4f} {c['top1'] - bar:+9.4f} {k['gained']:5d} "
              f"{k['lost']:5d} {k['net']:+5d} {k['churn']:6d} {s['n_regressions']:7d} "
              f"{s['n_shadow_regressions']:10d} {str(m['verdict']['promotable']):>11s}")
        for r in s["regressions"]:
            print(f"      SINGLE-CANDIDATE REGRESSION {r['crop']}#{r['target']} "
                  f"source={r['source']} ({r['reason']})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("schema", help="validate a cache's pair-and-role schema; licenses NOTHING")
    s.add_argument("--cache-dir", type=Path, required=True)
    s.add_argument("--crops", nargs="*")
    s.add_argument("--out", type=Path)
    s.set_defaults(func=cmd_schema)

    g = sub.add_parser("gate", help="gate a feature cache; writes the licence training requires")
    g.add_argument("--cache-dir", type=Path, required=True)
    g.add_argument("--preilp", type=Path, required=True)
    g.add_argument("--ecb-dir", type=Path)
    g.add_argument("--receipt", type=Path)
    g.add_argument("--reproducer", default="receipt")
    g.add_argument("--manifest", type=Path,
                   help="the audit_feature_cache manifest binding the producing TRUNK by sha256")
    g.add_argument("--expect-trunk-sha")
    g.add_argument("--expect-trunk-role")
    g.add_argument("--expect-fold")
    g.add_argument("--crops", nargs="*")
    g.add_argument("--licence", type=Path)
    g.add_argument("--out", type=Path)
    g.set_defaults(func=cmd_gate)

    t = sub.add_parser("train", help="fit and report every model declared in a spec")
    t.add_argument("--spec", type=Path, required=True)
    t.set_defaults(func=cmd_train)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
