"""BIOHUB-X - THE FROZEN I/O CONTRACT. Written BEFORE any model exists, and it fails CLOSED.

WHY A CONTRACT COMES FIRST
--------------------------
The campaign's recurring killer is not a wrong model, it is a green-while-wrong instrument. The
defect ledger is at thirteen and the SILENT FILL alone is at five instances:

  FACT-0432  an uncovered pair inherited 0.0 - the WORST score, not a neutral one;
  FACT-0425  ``strict=False`` loaded a 411M-parameter segmenter onto RANDOM weights with the two
             prints that would have reported it commented out;
  FACT-0454  a ``scale`` argument accepted and never applied, and 14 of 19 features zero-filled;
  FACT-0457  an output slot that is ``torch.new_zeros`` read as if it were a learned score;
  FACT-0459  a post-standardisation zero fill arriving as the constant -mean/std, which passes
             shape checks, NaN checks and "no zeros present" checks simultaneously.

Every one of those produces a plausible number instead of a crash, and every one looks exactly
like "the new model does not work". FACT-0459 states the rule this module implements: A COVERAGE
GUARD MUST INSPECT THE DATA, NOT THE ACCOUNTING. A count of features supplied is not evidence
that features were computed, so every check below reads the array.

WHAT BIOHUB-X OWNS, AND THE ARROW THAT MATTERS
----------------------------------------------
    raw 3D frames -> centers + instance representation -> candidate parents + explicit abstention
      -> continuation / division / neither -> FINAL LINEAGE GRAPH -> official scorer

The fourth arrow is the point. FACT-0428 measured that a change made only at pre-ILP ranking is
overwritten 84.6% of the time on fold 0 and 69.7% on fold 1 - the pipeline AGREES WITH its
ranking but does not FOLLOW it, and a scorer's gains live precisely in the disagreements. So the
graph in stage E is emitted by this system, and ``verify_graph`` REFUSES a graph whose declared
consumer chain contains the incumbent motion relink.

FIVE DATA STAGES, PLUS TWO COMPLETE-ARTIFACT VERIFIERS
------------------------------------------------------
  A NODES        verify_nodes       centers, instance labels, physical coordinates
  B EMBEDDINGS   verify_embeddings  the identity representation
  C CANDIDATES   verify_candidates  parents with an EXPLICIT no-parent class
  D CLASSES      folded into C      continuation / division / neither over the same pair
  E GRAPH        verify_graph       the emitted lineage, owned end to end
  P PROVENANCE   verify_consumer_provenance  source/hash-bound producer receipt
  X CROSS-STAGE  verify_cross_stage           one shared node/pair identity space

Each verifier returns a REPORT and raises ``ContractRefusal`` on violation. Nothing here scores
anything, promotes anything, or knows what a good number looks like. CLAUDE.md rule 4: tests
enforce software contracts only. Only ``verify_complete_artifact`` can assemble the required
stage set and earn the ``BIOHUBX_CONTRACT_OK`` heartbeat.

THE COORDINATE CONVENTION IS THE ONE THING MOST LIKELY TO BE GOT WRONG
---------------------------------------------------------------------
Atlas coordinates are FULL-RES ``(z, y, x)`` at ``(1.625, 0.40625, 0.40625)`` um - a 4:1:1
anisotropy, NOT isotropic 1.625. AGENTS.md section 4 records two of three distance analyses in
one day starting with the wrong convention, and FACT-0447 records the discriminating figure: the
fold-1 GT inter-frame displacement median is 1.81681 um on the correct convention and 5.13870 um
on the isotropic one. ``calibration_gate`` reproduces BOTH and refuses unless the correct one
passes AND the wrong one fails - a gate that cannot fail is not a gate.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# --------------------------------------------------------------------------------------------
# Frozen constants. Changing any of these is a NEW contract version, never an edit.
# --------------------------------------------------------------------------------------------
CONTRACT_VERSION = "biohubx_io_v1"

#: full-res (z, y, x) microns per voxel. FACT-0040's anchor is calibrated against this.
SCALE_UM: tuple[float, float, float] = (1.625, 0.40625, 0.40625)

#: the official matcher's radius. Present so a verifier can flag geometry that cannot be matched.
MATCH_RADIUS_UM = 7.0

#: FACT-0040 / FACT-0447: fold-1 GT inter-frame displacement median, and the value the WRONG
#: (isotropic) convention produces. The pair is what makes the calibration gate discriminating.
CALIBRATION_ANCHOR_UM = 1.81681
CALIBRATION_ANCHOR_TOL_UM = 0.15
CALIBRATION_ISOTROPIC_DECOY_UM = 5.13870

#: the sentinel for "no parent". It is a ROW, never a missing row - see verify_candidates.
NO_PARENT = -1

#: association classes, in the frozen slot order. `neither` is not abstention; see the note in
#: verify_candidates.
CLASSES: tuple[str, ...] = ("continuation", "division", "neither")

#: symbols that identify the INCUMBENT consumer. A Biohub-X graph declaring any of these in its
#: consumer chain is refused: FACT-0428 is the whole reason this system exists.
FORBIDDEN_CONSUMERS: tuple[str, ...] = (
    "motion_relink_edges", "motion_relink", "MOTION_RELINK_LEARNED_BONUS",
    "linefit_smooth_output_graph", "adaptive_short_track_rescue", "gap2_recovery",
)

#: below this, a float column is CONSTANT for our purposes. FACT-0459's fill arrived as the
#: finite constants -1.4052 and -0.2473, so "is it zero" is the wrong question and "does it vary"
#: is the right one.
CONSTANT_STD_EPS = 1e-9


class ContractRefusal(RuntimeError):
    """A refusal, never a warning. A silent no-op is worse than a crash (AGENTS.md section 4)."""


# --------------------------------------------------------------------------------------------
# helpers - every one of these reads the ARRAY
# --------------------------------------------------------------------------------------------
def _arr(table: dict, name: str, where: str) -> np.ndarray:
    if name not in table:
        raise ContractRefusal(f"{where}: required column {name!r} is absent")
    a = np.asarray(table[name])
    if a.ndim != 1:
        raise ContractRefusal(f"{where}.{name}: expected one dimension, got shape {a.shape}")
    return a


def _columns(table: dict, names: Sequence[str], where: str) -> dict[str, np.ndarray]:
    """Read required columns and refuse ragged tables before any zip/cast can truncate them."""
    out = {name: _arr(table, name, where) for name in names}
    lengths = {name: int(a.shape[0]) for name, a in out.items()}
    if len(set(lengths.values())) > 1:
        raise ContractRefusal(
            f"{where}: columns do not have equal lengths: {lengths}. Python zip truncates to the "
            "shortest input, so accepting this table would silently detach values from rows"
        )
    return out


def _integer(a: np.ndarray, where: str) -> np.ndarray:
    """Require an integer storage dtype *before* conversion; integral-looking floats are not IDs."""
    if a.dtype.kind not in "iu":
        raise ContractRefusal(
            f"{where}: must have an INTEGER dtype, got {a.dtype}. Casting before this check would "
            "silently truncate fractional IDs/times and make a malformed artifact look valid"
        )
    return a.astype(np.int64, copy=False)


def _table_digest(table: dict, names: Sequence[str]) -> str:
    """Stable content identity for a table, including dtype and shape rather than values alone."""
    h = hashlib.sha256()
    for name in names:
        a = np.asarray(table[name])
        h.update(name.encode("utf-8"))
        h.update(str(a.dtype).encode("ascii"))
        h.update(json.dumps(list(a.shape)).encode("ascii"))
        if a.dtype.kind in "OUS":
            h.update(json.dumps(a.astype(str).tolist(), ensure_ascii=False,
                                separators=(",", ":")).encode("utf-8"))
        else:
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _finite(a: np.ndarray, where: str) -> None:
    if a.dtype.kind == "f" and not np.isfinite(a).all():
        n = int((~np.isfinite(a)).sum())
        raise ContractRefusal(f"{where}: {n} non-finite value(s). NaN is not a neutral fill")


def _in_unit_interval(a: np.ndarray, where: str) -> None:
    if a.size and (a.min() < 0.0 or a.max() > 1.0):
        raise ContractRefusal(
            f"{where}: probabilities must lie in [0, 1]; got [{a.min():.6g}, {a.max():.6g}]. "
            f"A logit written into a probability slot passes every shape check"
        )


def constancy_report(a: np.ndarray, where: str) -> dict:
    """Is this column actually carrying information, or is it a FILL wearing a plausible value?

    FACT-0459 is the reason this is a data check and not a bookkeeping one: a post-standardisation
    zero fill arrives as the constant ``-mean/std``, on which shape, NaN and no-zeros checks all
    pass. Only the distribution can see it.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return {"where": where, "n": 0, "constant": True, "std": 0.0, "n_unique": 0}
    std = float(np.std(a))
    uniq = int(np.unique(a).size)
    return {"where": where, "n": int(a.size), "std": std, "n_unique": uniq,
            "min": float(a.min()), "max": float(a.max()),
            "constant": bool(std <= CONSTANT_STD_EPS or uniq <= 1)}


def refuse_if_constant(a: np.ndarray, where: str, *, why: str) -> dict:
    rep = constancy_report(a, where)
    if rep["constant"] and rep["n"] > 1:
        raise ContractRefusal(
            f"{where}: CONSTANT over {rep['n']} rows (std={rep['std']:.3g}, "
            f"{rep['n_unique']} unique value(s), value~{rep.get('min')}). {why} This is the "
            f"FACT-0459 shape - the fill is arithmetic, not a zero, and every structural check "
            f"passes on it"
        )
    return rep


# --------------------------------------------------------------------------------------------
# coordinates
# --------------------------------------------------------------------------------------------
def voxel_to_um(zyx_vox: np.ndarray) -> np.ndarray:
    """(N, 3) index-space (z, y, x) -> physical microns on the FROZEN anisotropic convention."""
    zyx_vox = np.asarray(zyx_vox, dtype=np.float64)
    if zyx_vox.ndim != 2 or zyx_vox.shape[1] != 3:
        raise ContractRefusal(f"voxel_to_um expects (N, 3) in (z, y, x) order, got {zyx_vox.shape}")
    return zyx_vox * np.asarray(SCALE_UM, dtype=np.float64)


def calibration_gate(gt_zyx_vox_t: np.ndarray, gt_zyx_vox_t1: np.ndarray) -> dict:
    """Reproduce FACT-0040's anchor AND show the wrong convention fails it.

    A calibration that only confirms the right answer proves nothing about whether it could have
    detected the wrong one. FACT-0447 ran exactly this pair and recorded both numbers, so this
    gate is a re-derivation rather than a new claim.
    """
    a = np.asarray(gt_zyx_vox_t, dtype=np.float64)
    b = np.asarray(gt_zyx_vox_t1, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ContractRefusal(
            f"calibration_gate: expected two matching (N, 3) arrays, got {a.shape} and {b.shape}")
    if a.shape[0] < 1000:
        raise ContractRefusal(
            f"calibration_gate: {a.shape[0]} edges is too few to reproduce a median anchor; "
            f"FACT-0447 used 109,057 on fold 1. A gate run on a toy sample is not a gate")
    d_correct = float(np.median(np.linalg.norm((b - a) * np.asarray(SCALE_UM), axis=1)))
    d_isotropic = float(np.median(np.linalg.norm((b - a) * SCALE_UM[0], axis=1)))
    correct_ok = abs(d_correct - CALIBRATION_ANCHOR_UM) <= CALIBRATION_ANCHOR_TOL_UM
    decoy_fails = abs(d_isotropic - CALIBRATION_ANCHOR_UM) > CALIBRATION_ANCHOR_TOL_UM
    rep = {
        "anchor_fact": "FACT-0040", "reproduced_by": "FACT-0447",
        "median_um_anisotropic": d_correct, "median_um_isotropic_decoy": d_isotropic,
        "anchor_um": CALIBRATION_ANCHOR_UM, "tolerance_um": CALIBRATION_ANCHOR_TOL_UM,
        "correct_convention_passes": correct_ok, "wrong_convention_fails": decoy_fails,
        "gate_discriminates": bool(correct_ok and decoy_fails),
        "n_edges": int(a.shape[0]),
    }
    if not correct_ok:
        raise ContractRefusal(
            f"calibration_gate: the anisotropic convention gives {d_correct:.5f} um against the "
            f"FACT-0040 anchor {CALIBRATION_ANCHOR_UM} +/- {CALIBRATION_ANCHOR_TOL_UM}. Distance "
            f"code that cannot reproduce the anchor is not trusted (AGENTS.md section 4)")
    if not decoy_fails:
        raise ContractRefusal(
            f"calibration_gate: the ISOTROPIC decoy also passes at {d_isotropic:.5f} um, so this "
            f"gate cannot distinguish the two conventions and proves nothing")
    return rep


# --------------------------------------------------------------------------------------------
# STAGE A - nodes
# --------------------------------------------------------------------------------------------
def verify_nodes(nodes: dict, *, where: str = "nodes") -> dict:
    """Centers plus instance representation, on the frozen coordinate convention.

    Required columns: crop, node_uid, t, instance_label, z_um, y_um, x_um, z_vox, y_vox,
    x_vox, center_confidence, volume_vox, division_prob. ``node_uid`` is stable across stages;
    ``instance_label`` is the per-frame mask identity.
    """
    names = (
        "crop", "node_uid", "t", "instance_label", "z_um", "y_um", "x_um",
        "z_vox", "y_vox", "x_vox", "center_confidence", "volume_vox", "division_prob",
    )
    cols = _columns(nodes, names, where)
    crop = cols["crop"]
    uid = _integer(cols["node_uid"], f"{where}.node_uid")
    t = _integer(cols["t"], f"{where}.t")
    label = _integer(cols["instance_label"], f"{where}.instance_label")
    if uid.size and uid.min() < 0:
        raise ContractRefusal(f"{where}.node_uid: IDs must be >= 0; {NO_PARENT} is a sentinel")
    if label.size and label.min() < 1:
        raise ContractRefusal(
            f"{where}.instance_label: labels must be >= 1; 0 is reserved for background and a "
            f"0-labelled row is indistinguishable from an unlabelled voxel")

    um = np.stack([cols[c].astype(np.float64) for c in ("z_um", "y_um", "x_um")], 1)
    vox = np.stack([cols[c].astype(np.float64)
                    for c in ("z_vox", "y_vox", "x_vox")], 1)
    _finite(um, f"{where}.[zyx]_um")
    _finite(vox, f"{where}.[zyx]_vox")

    # ROUND TRIP. The single most valuable check here: it catches a producer that wrote microns
    # into the voxel slot, an isotropic scaling, and a transposed (x, y, z) order, none of which
    # changes a shape.
    resid = np.abs(um - voxel_to_um(vox))
    max_resid = float(resid.max()) if resid.size else 0.0
    if max_resid > 1e-6:
        worst = int(np.argmax(resid.max(axis=1))) if resid.size else -1
        raise ContractRefusal(
            f"{where}: physical coordinates do not round-trip from the voxel coordinates at "
            f"scale {SCALE_UM}; max residual {max_resid:.6g} um at row {worst}. Either the "
            f"convention is isotropic, the axis order is not (z, y, x), or the two columns "
            f"describe different points"
        )

    conf = cols["center_confidence"].astype(np.float64)
    _finite(conf, f"{where}.center_confidence")
    _in_unit_interval(conf, f"{where}.center_confidence")
    dprob = cols["division_prob"].astype(np.float64)
    _finite(dprob, f"{where}.division_prob")
    _in_unit_interval(dprob, f"{where}.division_prob")

    vol = _integer(cols["volume_vox"], f"{where}.volume_vox")
    if vol.size and vol.min() < 1:
        raise ContractRefusal(
            f"{where}.volume_vox: an instance with zero voxels is not an instance. This is the "
            f"mask-derived slot, and a zero here is what a points-only producer emits when it "
            f"pretends to have masks (FACT-0454)")

    # uniqueness of (crop, t, instance_label)
    keys = np.array([f"{c}|{int(tt)}|{int(l)}" for c, tt, l in zip(crop, t, label)])
    if keys.size != np.unique(keys).size:
        dup = keys.size - np.unique(keys).size
        raise ContractRefusal(
            f"{where}: {dup} duplicate (crop, t, instance_label) key(s). Instance labels must be "
            f"unique within a frame or the graph's node identity is ambiguous")

    # Cross-stage identity is (crop, node_uid), and must be unique even when instance labels are
    # reused in later frames. Without this, candidate and graph rows can resolve to different nodes.
    uid_keys = np.array([f"{c}|{int(u)}" for c, u in zip(crop, uid)])
    if uid_keys.size != np.unique(uid_keys).size:
        raise ContractRefusal(
            f"{where}: {uid_keys.size - np.unique(uid_keys).size} duplicate "
            f"(crop, node_uid) key(s). node_uid is the cross-stage identity and must be unique")

    conf_rep = refuse_if_constant(
        conf, f"{where}.center_confidence",
        why="a detector head that returns the same confidence for every cell has not run.")
    dprob_rep = refuse_if_constant(
        dprob, f"{where}.division_prob",
        why="FACT-0457 is exactly this: an output slot that is torch.new_zeros, read as a learned "
            "score. A constant division probability is a head that was never trained or never "
            "wired.")

    return {
        "contract": CONTRACT_VERSION, "stage": "A_nodes", "n": int(t.size),
        "crops": int(np.unique(crop).size), "frames": int(np.unique(t).size),
        "coordinate_round_trip_max_um": max_resid,
        "scale_um": list(SCALE_UM),
        "center_confidence": conf_rep, "division_prob": dprob_rep,
        "volume_vox": {"min": int(vol.min()) if vol.size else 0,
                       "median": float(np.median(vol)) if vol.size else 0.0},
    }


# --------------------------------------------------------------------------------------------
# STAGE B - the identity embedding
# --------------------------------------------------------------------------------------------
def verify_embeddings(emb: np.ndarray, *, where: str = "embeddings",
                      expect_dim: int | None = None) -> dict:
    """(N, D) identity representation. Refuses a representation that carries no information.

    THE PER-DIMENSION CHECK IS THE ONE THAT MATTERS. A whole-array std is dominated by whichever
    dimensions do vary, so a partially-filled embedding - the 14-of-19 shape FACT-0454 records -
    passes it comfortably. Dead dimensions are counted individually and refused.
    """
    e = np.asarray(emb, dtype=np.float64)
    if e.ndim != 2:
        raise ContractRefusal(f"{where}: expected (N, D), got shape {e.shape}")
    n, d = e.shape
    if expect_dim is not None and d != expect_dim:
        raise ContractRefusal(f"{where}: declared dim {expect_dim}, got {d}")
    _finite(e, where)
    per_dim_std = np.std(e, axis=0) if n else np.zeros(d)
    dead = np.flatnonzero(per_dim_std <= CONSTANT_STD_EPS)
    if n > 1 and dead.size:
        raise ContractRefusal(
            f"{where}: {dead.size} of {d} embedding dimension(s) are CONSTANT across all {n} "
            f"nodes (first at index {int(dead[0])}). A partially-filled representation passes a "
            f"whole-array check and carries no identity in those slots - the FACT-0454 shape, "
            f"14 of 19 features constant behind a correctly-shaped vector")
    norms = np.linalg.norm(e, axis=1) if n else np.zeros(0)
    if n and float(norms.min()) == 0.0:
        raise ContractRefusal(
            f"{where}: {int((norms == 0).sum())} all-zero embedding row(s). A zero vector is "
            f"equidistant from everything and will be scored, not skipped")
    return {
        "contract": CONTRACT_VERSION, "stage": "B_embeddings", "n": int(n), "dim": int(d),
        "per_dim_std_min": float(per_dim_std.min()) if d else 0.0,
        "per_dim_std_median": float(np.median(per_dim_std)) if d else 0.0,
        "dead_dims": int(dead.size),
        "norm": {"min": float(norms.min()) if n else 0.0,
                 "max": float(norms.max()) if n else 0.0,
                 "median": float(np.median(norms)) if n else 0.0},
    }


# --------------------------------------------------------------------------------------------
# STAGES C and D - candidates, explicit abstention, and the three-way class
# --------------------------------------------------------------------------------------------
def verify_candidates(cand: dict, *, offered_pairs: Sequence[tuple[Any, int, int]] | None = None,
                      where: str = "candidates") -> dict:
    """Candidate parents with an EXPLICIT no-parent class and a genuine 3-way over each pair.

    Required columns: crop, t_target, target_uid, source_uid, p_continuation, p_division,
    p_neither. ``source_uid == NO_PARENT`` marks the abstention row.

    FOUR RULES, EACH TIED TO A MEASURED FAILURE.

    (1) EVERY TARGET CARRIES EXACTLY ONE NO-PARENT ROW. Abstention is a CLASS, not a threshold on
        the best candidate's score. FACT-0457 found HOCT's abstain mass is
        ``1 / (sum(exp(edge logits)) + 1)`` against a hard-coded logit-zero reference - a
        deterministic function of the edge logits with no parameters of its own. A missing
        no-parent row silently reintroduces exactly that.
    (2) THE NO-PARENT MASS MUST VARY. A constant abstain column is a head that was never trained,
        and it is the FACT-0459 shape: constant, finite, plausible, and invisible to every
        structural check.
    (3) THE THREE CLASS PROBABILITIES SUM TO 1 OVER THE SAME PAIR. `neither` is the model saying
        THIS PAIR is not linked; the no-parent ROW is the model saying THIS TARGET has no parent
        at all. They are different objects and conflating them is how abstention quietly becomes
        a threshold again.
    (4) COVERAGE REFUSES, IT DOES NOT FILL. FACT-0432: an uncovered pair took 0.0, which is the
        WORST score rather than a neutral one, and nothing downstream could tell. If
        ``offered_pairs`` is supplied, every one must be present.
    """
    names = ("crop", "t_target", "target_uid", "source_uid",
             "p_continuation", "p_division", "p_neither")
    cols = _columns(cand, names, where)
    crop = cols["crop"]
    t_t = _integer(cols["t_target"], f"{where}.t_target")
    tgt = _integer(cols["target_uid"], f"{where}.target_uid")
    src = _integer(cols["source_uid"], f"{where}.source_uid")
    if tgt.size and tgt.min() < 0:
        raise ContractRefusal(f"{where}.target_uid: IDs must be >= 0")
    if src.size and np.any((src < 0) & (src != NO_PARENT)):
        raise ContractRefusal(
            f"{where}.source_uid: the only negative identity allowed is NO_PARENT={NO_PARENT}")
    p = np.stack([cols[f"p_{c}"].astype(np.float64) for c in CLASSES], 1)
    _finite(p, f"{where}.p_*")
    _in_unit_interval(p.ravel(), f"{where}.p_*")

    # (3) a genuine three-way over each pair
    s = p.sum(axis=1)
    if s.size and float(np.abs(s - 1.0).max()) > 1e-5:
        bad = int(np.argmax(np.abs(s - 1.0)))
        raise ContractRefusal(
            f"{where}: the three class probabilities must sum to 1 over each pair; worst row "
            f"{bad} sums to {float(s[bad]):.6f}. Three independent sigmoids are not a 3-way "
            f"decision and cannot express 'this pair, but as a division rather than a "
            f"continuation'")

    # (0) each (target, source) pair appears ONCE. A pair scored twice is a pair whose two scores
    #     disagree, and whichever the consumer reads is an accident of row order - the ordering
    #     defect FACT-0386's cycle already caught once in `evaluate()`.
    pkeys = np.array([f"{c}|{int(tt)}|{int(u)}|{int(sv)}"
                      for c, tt, u, sv in zip(crop, t_t, tgt, src)])
    if pkeys.size != np.unique(pkeys).size:
        vals, counts = np.unique(pkeys, return_counts=True)
        worst = vals[int(np.argmax(counts))]
        raise ContractRefusal(
            f"{where}: {pkeys.size - np.unique(pkeys).size} duplicate (crop, t_target, "
            f"target_uid, source_uid) row(s); worst {worst!r} appears {int(counts.max())} times. "
            f"Which score the consumer reads would be an accident of row order")

    # (1) exactly one no-parent row per target
    tkeys = np.array([f"{c}|{int(tt)}|{int(u)}" for c, tt, u in zip(crop, t_t, tgt)])
    is_np = src == NO_PARENT
    targets, inverse = np.unique(tkeys, return_inverse=True)
    n_np = np.bincount(inverse[is_np], minlength=targets.size) if is_np.any() \
        else np.zeros(targets.size, dtype=np.int64)
    missing = np.flatnonzero(n_np == 0)
    extra = np.flatnonzero(n_np > 1)
    if missing.size:
        raise ContractRefusal(
            f"{where}: {missing.size} of {targets.size} target(s) carry NO explicit no-parent row "
            f"(first: {targets[missing[0]]}). Abstention must be a class the model emits, not a "
            f"threshold a caller applies afterwards - FACT-0457 records what the alternative "
            f"actually is")
    if extra.size:
        raise ContractRefusal(
            f"{where}: {extra.size} target(s) carry MORE THAN ONE no-parent row "
            f"(first: {targets[extra[0]]}). The abstain mass would be counted twice")

    # (2) the abstain mass must carry information
    #     `p_neither` on the no-parent ROW is the target's probability of having no parent.
    abstain = p[is_np, CLASSES.index("neither")]
    abstain_rep = refuse_if_constant(
        abstain, f"{where}.p_neither[source_uid == NO_PARENT]",
        why="a constant no-parent mass is a head that was never trained or never wired, and it "
            "reproduces FACT-0457's deterministic abstention rather than replacing it.")

    # (4) coverage refuses; it never fills
    coverage: dict[str, Any] = {"checked": offered_pairs is not None}
    if offered_pairs is not None:
        pairs = list(offered_pairs)
        if any(len(p) != 4 for p in pairs):
            raise ContractRefusal(
                f"{where}: offered_pairs must be (crop, t_target, target_uid, source_uid) tuples "
                f"so coverage is checked on the SAME key the table uses. A coverage check run on "
                f"a different key reports 100% and means nothing")
        have = {f"{c}|{int(tt)}|{int(u)}|{int(sv)}"
                for c, tt, u, sv in zip(crop, t_t, tgt, src)}
        want = {f"{c}|{int(tt)}|{int(u)}|{int(sv)}" for (c, tt, u, sv) in pairs}
        uncovered = sorted(want - have)
        coverage.update({"n_offered": len(want), "n_uncovered": len(uncovered)})
        if uncovered:
            raise ContractRefusal(
                f"{where}: {len(uncovered)} offered pair(s) have NO row (first: {uncovered[0]}). "
                f"REFUSING rather than filling: FACT-0432 records an uncovered pair inheriting "
                f"0.0, which is the WORST score and not a neutral one, invisibly")

    real = ~is_np
    return {
        "contract": CONTRACT_VERSION, "stage": "CD_candidates",
        "n_rows": int(tkeys.size), "n_targets": int(targets.size),
        "n_real_candidate_rows": int(real.sum()),
        "candidates_per_target_mean": float(real.sum() / max(targets.size, 1)),
        "abstain": abstain_rep,
        "argmax_class_counts": {
            c: int((p.argmax(axis=1) == i).sum()) for i, c in enumerate(CLASSES)},
        "coverage": coverage,
    }


# --------------------------------------------------------------------------------------------
# STAGE E - the emitted lineage graph, owned end to end
# --------------------------------------------------------------------------------------------
def verify_graph(graph: dict, *, consumer_chain: Sequence[str], where: str = "graph") -> dict:
    """The FINAL graph. Refuses the incumbent consumer, and refuses a float export.

    Required columns: crop, node_uid, parent_uid, t, z_vox, y_vox, x_vox.
    ``parent_uid == NO_PARENT`` marks a root.

    THE CONSUMER CHECK IS THE REASON THIS STAGE EXISTS. FACT-0428: a change made only at pre-ILP
    ranking is overwritten 84.6% of the time on fold 0 and 69.7% on fold 1, because the relink
    re-derives its answer from geometry and overrules the score exactly where the score disagrees
    - which is exactly where a better scorer's gains live. Feeding a Biohub-X score into that
    consumer re-runs an experiment this campaign has already lost.

    THE INTEGER CHECK IS A DEPLOYMENT FACT, NOT A STYLE RULE. FACT-0344: the float export scored
    0.914 against 0.928, a -0.014 reversal, and the field-validated form is integer.
    """
    bad = [c for c in consumer_chain
           if any(f.lower() in str(c).lower() for f in FORBIDDEN_CONSUMERS)]
    if bad:
        raise ContractRefusal(
            f"{where}: the declared consumer chain contains the INCUMBENT relink ({bad}). "
            f"Biohub-X owns the emitted graph; FACT-0428 measured that a score handed to that "
            f"consumer survives on only 15.4% (fold 0) and 30.3% (fold 1) of forced changes. "
            f"Declaring it here would make this a pre-ILP experiment wearing a new name")
    if not consumer_chain:
        raise ContractRefusal(
            f"{where}: consumer_chain is empty. A graph that does not say what produced it "
            f"cannot be audited, and 'no relinking' must be a positive declaration rather than "
            f"an absence someone reads into it")

    names = ("crop", "node_uid", "parent_uid", "t", "z_vox", "y_vox", "x_vox")
    cols = _columns(graph, names, where)
    crop = cols["crop"]
    uid = _integer(cols["node_uid"], f"{where}.node_uid")
    par = _integer(cols["parent_uid"], f"{where}.parent_uid")
    t = _integer(cols["t"], f"{where}.t")
    if uid.size and uid.min() < 0:
        raise ContractRefusal(f"{where}.node_uid: IDs must be >= 0")
    if par.size and np.any((par < 0) & (par != NO_PARENT)):
        raise ContractRefusal(
            f"{where}.parent_uid: the only negative identity allowed is NO_PARENT={NO_PARENT}")
    for c in ("z_vox", "y_vox", "x_vox"):
        _integer(cols[c], f"{where}.{c}")

    keys = np.array([f"{c}|{int(u)}" for c, u in zip(crop, uid)])
    if keys.size != np.unique(keys).size:
        raise ContractRefusal(
            f"{where}.node_uid: {keys.size - np.unique(keys).size} duplicate (crop, node_uid) id(s)")

    index = {(str(c), int(u)): i for i, (c, u) in enumerate(zip(crop, uid))}
    has_parent = par != NO_PARENT
    unknown = [(str(c), int(p)) for c, p in zip(crop[has_parent], par[has_parent])
               if (str(c), int(p)) not in index]
    if unknown:
        raise ContractRefusal(
            f"{where}: {len(unknown)} edge(s) point at a node that is not in the table "
            f"(first {unknown[0]}). A dangling parent is a graph the scorer cannot read")

    # in-degree is 1 by construction (one parent column); out-degree must be <= 2.
    out_deg = np.bincount([index[(str(c), int(p))]
                               for c, p in zip(crop[has_parent], par[has_parent])],
                              minlength=uid.size) \
        if has_parent.any() else np.zeros(uid.size, dtype=np.int64)
    if out_deg.size and int(out_deg.max()) > 2:
        worst = int(uid[int(np.argmax(out_deg))])
        raise ContractRefusal(
            f"{where}: node {worst} has out-degree {int(out_deg.max())}; the lineage contract "
            f"allows at most 2 (a division). A 3-way fork is not a biological event this metric "
            f"scores")

    # temporal validity: a parent sits exactly one frame earlier.
    if has_parent.any():
        child_t = t[has_parent]
        parent_t = t[[index[(str(c), int(p))]
                      for c, p in zip(crop[has_parent], par[has_parent])]]
        dt = child_t - parent_t
        if int(np.abs(dt - 1).max()) != 0:
            n_bad = int((dt != 1).sum())
            raise ContractRefusal(
                f"{where}: {n_bad} edge(s) do not span exactly one frame (dt ranges "
                f"[{int(dt.min())}, {int(dt.max())}]). A backwards or skipping edge is a cycle "
                f"risk and the scorer's matcher is per-frame")

    n_div = int((out_deg == 2).sum())
    return {
        "contract": CONTRACT_VERSION, "stage": "E_graph",
        "nodes": int(uid.size), "edges": int(has_parent.sum()),
        "roots": int((~has_parent).sum()), "divisions": n_div,
        "max_out_degree": int(out_deg.max()) if out_deg.size else 0,
        "max_in_degree": 1,
        "frames": int(np.unique(t).size),
        "consumer_chain": list(consumer_chain),
        "relink_applied": False,
        "integer_export": True,
    }


# --------------------------------------------------------------------------------------------
# COMPLETE-ARTIFACT BINDING - stages must describe the same nodes, pairs, graph, and producer
# --------------------------------------------------------------------------------------------
NODE_COLUMNS = (
    "crop", "node_uid", "t", "instance_label", "z_um", "y_um", "x_um",
    "z_vox", "y_vox", "x_vox", "center_confidence", "volume_vox", "division_prob",
)
CANDIDATE_COLUMNS = (
    "crop", "t_target", "target_uid", "source_uid",
    "p_continuation", "p_division", "p_neither",
)
GRAPH_COLUMNS = ("crop", "node_uid", "parent_uid", "t", "z_vox", "y_vox", "x_vox")


def _producer_identity(path: str | Path, symbol: str) -> dict:
    """Bind provenance to auditable source bytes containing the named producer function.

    This does not pretend Python can prove an arbitrary runtime call graph. It does replace an
    unauditable string assertion with positive, reproducible evidence: the exact source bytes,
    the function they define, and a scan that refuses every incumbent-consumer symbol.
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise ContractRefusal(f"consumer_provenance: producer source is absent: {p}")
    raw = p.read_bytes()
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=str(p))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ContractRefusal(f"consumer_provenance: producer source cannot be audited: {exc}") from exc
    definitions = {n.name for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if symbol not in definitions:
        raise ContractRefusal(
            f"consumer_provenance: producer symbol {symbol!r} is not defined in {p}")
    text = raw.decode("utf-8").lower()
    forbidden = sorted({name for name in FORBIDDEN_CONSUMERS if name.lower() in text})
    if forbidden:
        raise ContractRefusal(
            f"consumer_provenance: producer source contains forbidden incumbent consumer "
            f"symbol(s) {forbidden}. A clean chain must be mechanically separate, not a branch "
            f"selected by a self-declared flag")
    return {"producer_path": str(p), "producer_symbol": symbol,
            "producer_sha256": hashlib.sha256(raw).hexdigest()}


def make_consumer_receipt(*, producer_path: str | Path, producer_symbol: str,
                          candidates: dict, graph: dict,
                          consumer_chain: Sequence[str]) -> dict:
    """Create a content-bound receipt after a graph-owning producer run.

    The verifier re-derives every field; editing this dictionary cannot make a different source,
    candidate surface, or graph pass.
    """
    ident = _producer_identity(producer_path, producer_symbol)
    return {
        "contract": CONTRACT_VERSION,
        **ident,
        "candidate_sha256": _table_digest(candidates, CANDIDATE_COLUMNS),
        "graph_sha256": _table_digest(graph, GRAPH_COLUMNS),
        "consumer_chain": list(consumer_chain),
    }


def verify_consumer_provenance(receipt: dict, *, candidates: dict, graph: dict,
                               where: str = "consumer_provenance") -> dict:
    """Re-derive a producer receipt from source and artifact bytes; refuse declarations alone."""
    required = {
        "contract", "producer_path", "producer_symbol", "producer_sha256",
        "candidate_sha256", "graph_sha256", "consumer_chain",
    }
    missing = sorted(required - set(receipt))
    if missing:
        raise ContractRefusal(f"{where}: receipt is incomplete; missing {missing}")
    if receipt["contract"] != CONTRACT_VERSION:
        raise ContractRefusal(
            f"{where}: receipt contract {receipt['contract']!r} != {CONTRACT_VERSION!r}")
    ident = _producer_identity(receipt["producer_path"], receipt["producer_symbol"])
    if ident["producer_sha256"] != receipt["producer_sha256"]:
        raise ContractRefusal(
            f"{where}: producer source hash changed after the receipt was written")
    candidate_sha = _table_digest(candidates, CANDIDATE_COLUMNS)
    graph_sha = _table_digest(graph, GRAPH_COLUMNS)
    if candidate_sha != receipt["candidate_sha256"]:
        raise ContractRefusal(f"{where}: candidate table does not match the producer receipt")
    if graph_sha != receipt["graph_sha256"]:
        raise ContractRefusal(f"{where}: final graph does not match the producer receipt")
    # Reuse the same forbidden-chain rule as the graph verifier, but bind it to source bytes and
    # artifact bytes here. A chain string by itself is never sufficient for a complete report.
    bad = [c for c in receipt["consumer_chain"]
           if any(f.lower() in str(c).lower() for f in FORBIDDEN_CONSUMERS)]
    if bad or not receipt["consumer_chain"]:
        raise ContractRefusal(
            f"{where}: receipt carries an empty or forbidden consumer chain: {bad}")
    return {
        "contract": CONTRACT_VERSION, "stage": "P_consumer",
        "producer_path": ident["producer_path"],
        "producer_symbol": ident["producer_symbol"],
        "producer_sha256": ident["producer_sha256"],
        "candidate_sha256": candidate_sha, "graph_sha256": graph_sha,
        "consumer_chain": list(receipt["consumer_chain"]),
        "source_scanned_for_incumbent": True,
        "artifacts_content_bound": True,
    }


def verify_cross_stage(nodes: dict, embeddings: np.ndarray, candidates: dict, graph: dict,
                       *, where: str = "cross_stage") -> dict:
    """Prove that A/B/CD/E refer to the same objects rather than four plausible tables."""
    ncols = _columns(nodes, NODE_COLUMNS, "nodes")
    ccols = _columns(candidates, CANDIDATE_COLUMNS, "candidates")
    gcols = _columns(graph, GRAPH_COLUMNS, "graph")
    n_uid = _integer(ncols["node_uid"], "nodes.node_uid")
    n_t = _integer(ncols["t"], "nodes.t")
    c_t = _integer(ccols["t_target"], "candidates.t_target")
    c_tgt = _integer(ccols["target_uid"], "candidates.target_uid")
    c_src = _integer(ccols["source_uid"], "candidates.source_uid")
    g_uid = _integer(gcols["node_uid"], "graph.node_uid")
    g_par = _integer(gcols["parent_uid"], "graph.parent_uid")
    g_t = _integer(gcols["t"], "graph.t")

    e = np.asarray(embeddings)
    if e.ndim != 2 or e.shape[0] != n_uid.size:
        raise ContractRefusal(
            f"{where}: embeddings must be row-aligned with nodes; got {e.shape} for "
            f"{n_uid.size} nodes")

    node_index = {(str(c), int(u)): i
                  for i, (c, u) in enumerate(zip(ncols["crop"], n_uid))}
    if len(node_index) != n_uid.size:
        raise ContractRefusal(f"{where}: nodes do not have unique (crop, node_uid) identities")

    unknown_targets, unknown_sources, wrong_candidate_time = [], [], []
    candidate_pairs: set[tuple[str, int, int, int]] = set()
    for crop, tt, tgt, src in zip(ccols["crop"], c_t, c_tgt, c_src):
        c = str(crop); target_key = (c, int(tgt))
        if target_key not in node_index:
            unknown_targets.append(target_key)
            continue
        if int(n_t[node_index[target_key]]) != int(tt):
            wrong_candidate_time.append((target_key, int(tt)))
        if int(src) != NO_PARENT:
            source_key = (c, int(src))
            if source_key not in node_index:
                unknown_sources.append(source_key)
            elif int(n_t[node_index[source_key]]) != int(tt) - 1:
                wrong_candidate_time.append((source_key, int(tt) - 1))
        candidate_pairs.add((c, int(tt), int(tgt), int(src)))
    if unknown_targets or unknown_sources or wrong_candidate_time:
        raise ContractRefusal(
            f"{where}: candidate/node referential integrity failed: "
            f"unknown_targets={unknown_targets[:3]}, unknown_sources={unknown_sources[:3]}, "
            f"wrong_time={wrong_candidate_time[:3]}")

    graph_node_rows: dict[tuple[str, int], int] = {}
    unknown_graph_nodes, wrong_graph_time, wrong_graph_coord = [], [], []
    for i, (crop, uid, tt) in enumerate(zip(gcols["crop"], g_uid, g_t)):
        key = (str(crop), int(uid)); graph_node_rows[key] = i
        if key not in node_index:
            unknown_graph_nodes.append(key)
            continue
        ni = node_index[key]
        if int(n_t[ni]) != int(tt):
            wrong_graph_time.append(key)
        expected = np.rint([ncols[c][ni] for c in ("z_vox", "y_vox", "x_vox")]).astype(np.int64)
        actual = np.array([gcols[c][i] for c in ("z_vox", "y_vox", "x_vox")], dtype=np.int64)
        if not np.array_equal(expected, actual):
            wrong_graph_coord.append(key)
    if unknown_graph_nodes or wrong_graph_time or wrong_graph_coord:
        raise ContractRefusal(
            f"{where}: graph/node referential integrity failed: "
            f"unknown={unknown_graph_nodes[:3]}, wrong_time={wrong_graph_time[:3]}, "
            f"wrong_rounded_coordinate={wrong_graph_coord[:3]}")

    # Every emitted node after a crop's first frame is a target decision, including a birth/root.
    # Requiring a candidate target here is what makes the explicit NO_PARENT row end-to-end rather
    # than a property of whichever subset happened to reach the candidate table.
    candidate_targets = {(str(c), int(tt), int(tgt))
                         for c, tt, tgt in zip(ccols["crop"], c_t, c_tgt)}
    min_frame = {str(c): int(np.min(g_t[np.asarray(gcols["crop"]).astype(str) == str(c)]))
                 for c in np.unique(np.asarray(gcols["crop"]).astype(str))}
    missing_target_decisions = [
        (str(c), int(tt), int(uid))
        for c, tt, uid in zip(gcols["crop"], g_t, g_uid)
        if int(tt) > min_frame[str(c)] and (str(c), int(tt), int(uid)) not in candidate_targets
    ]
    if missing_target_decisions:
        raise ContractRefusal(
            f"{where}: {len(missing_target_decisions)} emitted non-initial node(s) never entered "
            f"the candidate/abstention decision (first {missing_target_decisions[0]}). A birth is "
            f"an explicit NO_PARENT outcome, not an absent row")

    unoffered_edges = []
    for crop, uid, par, tt in zip(gcols["crop"], g_uid, g_par, g_t):
        if int(par) == NO_PARENT:
            continue
        pair = (str(crop), int(tt), int(uid), int(par))
        if pair not in candidate_pairs:
            unoffered_edges.append(pair)
    if unoffered_edges:
        raise ContractRefusal(
            f"{where}: {len(unoffered_edges)} emitted edge(s) were never offered/scored "
            f"(first {unoffered_edges[0]}). A graph-owning consumer may abstain, but it may not "
            f"invent an uncovered edge")

    return {
        "contract": CONTRACT_VERSION, "stage": "X_cross_stage",
        "nodes": int(n_uid.size), "embedding_rows": int(e.shape[0]),
        "candidate_rows": int(c_tgt.size), "graph_nodes": int(g_uid.size),
        "candidate_node_references_valid": True,
        "graph_node_references_valid": True,
        "emitted_edges_offered": True,
    }


# --------------------------------------------------------------------------------------------
# the contract document
# --------------------------------------------------------------------------------------------
@dataclass
class ContractReport:
    stages: dict = field(default_factory=dict)

    REQUIRED_STAGES = frozenset({
        "A_nodes", "B_embeddings", "CD_candidates", "E_graph",
        "P_consumer", "X_cross_stage",
    })

    def add(self, rep: dict) -> "ContractReport":
        if "stage" not in rep:
            raise ContractRefusal("ContractReport.add: report has no stage identity")
        if rep["stage"] in self.stages:
            raise ContractRefusal(
                f"ContractReport.add: duplicate stage {rep['stage']!r}; later evidence may not "
                f"silently replace earlier evidence")
        self.stages[rep["stage"]] = rep
        return self

    def require_complete(self) -> "ContractReport":
        missing = sorted(self.REQUIRED_STAGES - set(self.stages))
        extra = sorted(set(self.stages) - self.REQUIRED_STAGES)
        if missing or extra:
            raise ContractRefusal(
                f"ContractReport: incomplete or unknown stage set; missing={missing}, extra={extra}. "
                f"The BIOHUBX_CONTRACT_OK heartbeat is reserved for one complete A/B/CD/E artifact")
        return self

    def to_json(self, path: str | Path) -> Path:
        self.require_complete()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "contract": CONTRACT_VERSION,
            "heartbeat": "BIOHUBX_CONTRACT_OK",
            "stages": self.stages,
        }, indent=2, default=float), encoding="utf-8")
        return p


def verify_complete_artifact(*, nodes: dict, embeddings: np.ndarray, candidates: dict,
                             graph: dict, consumer_receipt: dict,
                             offered_pairs: Sequence[tuple[Any, int, int, int]] | None = None,
                             where: str = "biohubx") -> ContractReport:
    """The only route to a complete ``biohubx_io_v1`` heartbeat."""
    chain = consumer_receipt.get("consumer_chain", [])
    report = ContractReport()
    report.add(verify_nodes(nodes, where=f"{where}.nodes"))
    report.add(verify_embeddings(embeddings, where=f"{where}.embeddings"))
    report.add(verify_candidates(candidates, offered_pairs=offered_pairs,
                                 where=f"{where}.candidates"))
    report.add(verify_graph(graph, consumer_chain=chain, where=f"{where}.graph"))
    report.add(verify_consumer_provenance(
        consumer_receipt, candidates=candidates, graph=graph,
        where=f"{where}.consumer_provenance"))
    report.add(verify_cross_stage(nodes, embeddings, candidates, graph,
                                  where=f"{where}.cross_stage"))
    return report.require_complete()
