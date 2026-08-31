r"""THE 19-FEATURE HOCT INPUT CONTRACT - fail-closed, and it REFUSES rather than zero-fills.

WHY THIS EXISTS
---------------
``FACT-0454`` blocker 2: most of HOCT's 19 input features need INSTANCE MASKS the deployed
notebook does not produce, the one points-only entry point ``create_graph_from_points`` is an
exported STUB (`_api.py:133-134`), and a missing intensity feature is SILENTLY ZERO-FILLED at
``graph.py:184-187`` / ``graph.py:283-290``. The result is a correctly-SHAPED 19-vector that
carries almost no information, and a head fed a mostly-constant vector returns a plausible
number instead of an error. That is the ``FACT-0432`` / ``FACT-0425`` false-kill shape a third
time: it would look exactly like "the foreign model does not transfer".

So this module does two separable things and neither trusts the other:

1.  It states the slot map - what each of the 19 input columns IS, where it comes from at
    ``file:line``, and whether we can compute it today - and it PROVES that map against the
    publisher's own ``_MEAN``/``_STD`` constants rather than against our reading of the source.
2.  It AUDITS a produced feature matrix and refuses one that carries a zero-filled slot. The
    audit sees ONLY the array. It cannot be fooled by correct bookkeeping over wrong data,
    which is the exact failure ``PKT-0048``'s falsifier names.

WHERE THE 19 COME FROM, at named lines of ``royerlab/hoct`` (local copy under
``C:/temp/hoct_official/meta/src/``, package tree ``src/hoct/``)

    ``_batching.py:195-199``   ``node_attrs.select(T, *spatial_cols, *properties)``
    ``_batching.py:200``       ``unpack_array_attrs`` flattens ``inertia_tensor`` (3x3) in place
    ``_api.py:178`` / ``:188`` ``properties = REGIONPROPS``
    ``features/constants.py``  ``REGIONPROPS`` order, verbatim

    -> ``t, z, y, x`` + ``equivalent_diameter_area`` + ``intensity_{min,max,mean,std}``
       + ``inertia_tensor[0..8]`` + ``border_dist``  =  4 + 1 + 4 + 9 + 1  =  19.

A CORRECTION TO ``FACT-0454``, AND IT IS A CORRECTION IN OUR FAVOUR
-------------------------------------------------------------------
``FACT-0454`` records 15 of 19 as mask-requiring. Read at source it is FOURTEEN.
``border_dist`` is explicitly REMOVED from the regionprops request - ``graph.py:172-173`` in
``create_graph`` and ``graph.py:276-281`` in ``add_features`` - and computed from ``z,y,x`` plus
the image ``shape`` by ``features.py:156-189``. It is a POINTS-ONLY feature. The mask-requiring
set is ``equivalent_diameter_area`` (1) + the four intensities (4) + ``inertia_tensor`` (9) = 14,
and the points-only set is ``t,z,y,x,border_dist`` = 5. The conclusion is unchanged - 14 of 19
still blocks - but the number must be right before anything is built on it.

THE ZERO-FILL IS NOT A ZERO AT THE MODEL INPUT
----------------------------------------------
``Standardize`` (``_transforms.py:204-211``) runs AFTER the fill, so a zero-filled slot reaches
the network as the CONSTANT ``-mean/std``, not as 0. For ``equivalent_diameter_area`` that is
-1.405; for ``inertia_tensor[0,0]`` it is -0.247. A constant is exactly the value that produces
a plausible wrong answer rather than a crash, so the audit hunts BOTH signatures.

WHAT THE PUBLISHER'S OWN CONSTANTS PROVE ABOUT ITS TRAINING DATA
----------------------------------------------------------------
``_MEAN``/``_STD`` are 19 numbers the publisher shipped, and they are evidence about the corpus
``general_v1`` was fitted on. Two structural checks fall out and BOTH pass, which is why the slot
map above is asserted rather than assumed:

  SYMMETRY       slots 9..17 must be a symmetric 3x3 in mean AND in std. They are, exactly.
  PERPENDICULAR  for a PLANAR object I_zz = I_yy + I_xx. The shipped means give
                 167.81 against 87.012 + 83.695 = 170.707, a 1.7% miss.

The second one is not a formality. It says the training corpus is dominated by masks that are
FLAT IN Z - 2D data, or 3D data one slice thick. ``graph.py:16-40`` (``convert_to_3d``) sets
``z = 0.0`` for every 2D input, and the shipped ``z`` statistics agree: mean 2.938, std 7.600.
Our crops are ``(T=100, Z=64, Y=256, X=256)`` and our nodes fill that z range. See
``hoct_scale_gate.py`` for what that does to the input distribution; it is recorded here because
it is read off the same nineteen numbers.

USAGE
-----
    python scripts/win_bet/hoct_feature_contract.py report --out <json>
    python scripts/win_bet/hoct_feature_contract.py mutate --out <json>   # the demonstrated refusal
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())

HEARTBEAT_OK = "HOCT_FEATURE_CONTRACT_COMPLETE"
HEARTBEAT_REFUSED = "HOCT_FEATURE_CONTRACT_REFUSED"

UPSTREAM = "royerlab/hoct 2ccc5040 (v0.2.0); local copy C:/temp/hoct_official/meta/src/"

# Verbatim from `hoct/_api.py:18-60`. NOT restated from prose - copied from the source file.
PUBLISHER_MEAN = (
    4.6326e02, 2.9380e00, 3.5649e02, 3.4491e02, 1.1521e01, 2.7600e-01, 9.6600e-01,
    5.7400e-01, 1.6200e-01, 1.6781e02, -2.7000e-02, 5.0000e-02, -2.7000e-02, 8.7012e01,
    -1.4010e00, 5.0000e-02, -1.4010e00, 8.3695e01, 9.0000e-03,
)
PUBLISHER_STD = (
    5.5578e02, 7.6000e00, 1.9588e02, 2.2610e02, 8.1990e00, 2.1600e-01, 2.8100e-01,
    1.9300e-01, 6.9000e-02, 6.7845e02, 3.1670e00, 2.8750e00, 3.1670e00, 5.1292e02,
    1.8274e02, 2.8750e00, 1.8274e02, 3.0608e02, 7.8000e-02,
)

POINTS = "points"          # available from any (t,z,y,x) detection
POINTS_AND_SHAPE = "points+volume_shape"
MASK = "instance_mask"
MASK_AND_IMAGE = "instance_mask+intensity_image"


class FeatureRefusal(RuntimeError):
    """Raised instead of zero-filling. Carries the slot names that could not be computed."""

    def __init__(self, reason: str, slots: list[str]) -> None:
        super().__init__(f"{reason}: {', '.join(slots)}")
        self.reason = reason
        self.slots = slots


@dataclass(frozen=True)
class Slot:
    index: int
    name: str
    column: str            # the graph node attribute this slot is unpacked from
    source: str            # what data is needed to compute it
    where: str             # file:line in the publisher's package
    computable_today: bool
    what_it_would_take: str


def _inertia(i: int, r: int, c: int) -> Slot:
    return Slot(
        index=i,
        name=f"inertia_tensor[{r},{c}]",
        column="inertia_tensor",
        source=MASK,
        where="graph.py:175-177 RegionPropsNodes; unpacked at _batching.py:200",
        computable_today=False,
        what_it_would_take=(
            "a per-instance binary mask. skimage second moments about the centroid, in the "
            "LABEL ARRAY's index space - the publisher never rescales it, so a micron-space "
            "mask would change this slot by a factor of the voxel size squared per axis"
        ),
    )


SLOTS: tuple[Slot, ...] = (
    Slot(0, "t", "t", POINTS, "_batching.py:196", True, "already have it"),
    Slot(1, "z", "z", POINTS, "_batching.py:197 via _frame_dataset.py:53-54", True,
         "already have it - but see hoct_scale_gate.py, the UNIT is the open question"),
    Slot(2, "y", "y", POINTS, "_batching.py:197 via _frame_dataset.py:53-54", True, "already have it"),
    Slot(3, "x", "x", POINTS, "_batching.py:197 via _frame_dataset.py:53-54", True, "already have it"),
    Slot(4, "equivalent_diameter_area", "equivalent_diameter_area", MASK,
         "constants.py REGIONPROPS[0]; graph.py:175-177", False,
         "a per-instance mask VOXEL COUNT. A radius guess from a detector peak is NOT this "
         "feature and must not be substituted - it would be our number in the publisher's slot"),
    Slot(5, "intensity_min", "intensity_min", MASK_AND_IMAGE,
         "constants.py REGIONPROPS[1]; zero-filled at graph.py:184-187 when images is None", False,
         "mask + the intensity crop, normalised by features.py:55-85 (min / 0.999 quantile)"),
    Slot(6, "intensity_max", "intensity_max", MASK_AND_IMAGE,
         "constants.py REGIONPROPS[2]; zero-filled at graph.py:184-187", False,
         "mask + normalised intensity crop"),
    Slot(7, "intensity_mean", "intensity_mean", MASK_AND_IMAGE,
         "constants.py REGIONPROPS[3]; zero-filled at graph.py:184-187", False,
         "mask + normalised intensity crop"),
    Slot(8, "intensity_std", "intensity_std", MASK_AND_IMAGE,
         "constants.py REGIONPROPS[4]; zero-filled at graph.py:184-187", False,
         "mask + normalised intensity crop"),
    _inertia(9, 0, 0), _inertia(10, 0, 1), _inertia(11, 0, 2),
    _inertia(12, 1, 0), _inertia(13, 1, 1), _inertia(14, 1, 2),
    _inertia(15, 2, 0), _inertia(16, 2, 1), _inertia(17, 2, 2),
    Slot(18, "border_dist", "border_dist", POINTS_AND_SHAPE,
         "features.py:8-52 + :156-189; REMOVED from regionprops at graph.py:172-173", True,
         "z,y,x and the volume shape. Ours is (100, 64, 256, 256) for all 199 crops. "
         "features.py:34 uses cutoff=5 in INDEX units, so it must be fed index coords"),
)

MASK_SLOTS = tuple(s.index for s in SLOTS if not s.computable_today)
POINT_SLOTS = tuple(s.index for s in SLOTS if s.computable_today)


# ---------------------------------------------------------------------------------------------
# 1. PROVE THE SLOT MAP, against the publisher's constants rather than against our source read
# ---------------------------------------------------------------------------------------------
def prove_slot_map() -> dict:
    """Four structural checks on ``_MEAN``/``_STD``. Every one must pass or the map is wrong.

    These do not depend on our reading of ``_batching.py`` at all. If the true column order were
    different, the inertia block would not be a symmetric 3x3 sitting exactly at 9..17 and the
    intensity block would not obey min <= mean <= max inside [0, 1].
    """
    m = np.asarray(PUBLISHER_MEAN, dtype=np.float64)
    s = np.asarray(PUBLISHER_STD, dtype=np.float64)
    checks: list[dict] = []

    checks.append({
        "check": "width",
        "why": "19 slots is the whole claim; a different width voids the map",
        "passes": bool(len(m) == 19 and len(s) == 19 and len(SLOTS) == 19),
        "detail": {"n_mean": len(m), "n_std": len(s), "n_slots": len(SLOTS)},
    })

    mm = m[9:18].reshape(3, 3)
    ss = s[9:18].reshape(3, 3)
    sym_m = float(np.abs(mm - mm.T).max())
    sym_s = float(np.abs(ss - ss.T).max())
    checks.append({
        "check": "inertia_block_is_symmetric",
        "why": "an inertia tensor is symmetric. Slots 9..17 read row-major must be too, in "
               "BOTH statistics. A shifted block breaks this immediately",
        "passes": bool(sym_m == 0.0 and sym_s == 0.0),
        "detail": {"max_abs_asymmetry_mean": sym_m, "max_abs_asymmetry_std": sym_s},
    })

    i_min, i_max, i_mean, i_std = m[5], m[6], m[7], m[8]
    ok_int = bool(
        0.0 <= i_min <= i_mean <= i_max <= 1.0 and 0.0 <= i_std <= i_mean
    )
    checks.append({
        "check": "intensity_block_order_and_range",
        "why": "features.py:55-85 normalises to [0,1], so slot 5..8 read as (min, max, mean, "
               "std) must satisfy min <= mean <= max <= 1. Any permutation of the four breaks it",
        "passes": ok_int,
        "detail": {"min": i_min, "max": i_max, "mean": i_mean, "std": i_std},
    })

    izz, iyy, ixx = mm[0, 0], mm[1, 1], mm[2, 2]
    planar_miss = float(abs(izz - (iyy + ixx)) / izz)
    checks.append({
        "check": "perpendicular_axis_theorem",
        "why": "for a PLANAR mask I_zz = I_yy + I_xx exactly. If slots 9..17 are the inertia "
               "tensor AND the corpus is dominated by flat masks, the shipped means obey it. "
               "It is simultaneously a slot-map check and a statement about the training data",
        "passes": bool(planar_miss < 0.05),
        "detail": {"I_zz": izz, "I_yy+I_xx": float(iyy + ixx), "relative_miss": planar_miss},
    })

    return {
        "upstream": UPSTREAM,
        "checks": checks,
        "all_passed": all(c["passes"] for c in checks),
        "corpus_inference": {
            "claim": "general_v1's training corpus is dominated by masks that are FLAT IN Z",
            "evidence": [
                "the perpendicular-axis theorem holds on the shipped inertia means to 1.7%",
                f"shipped z statistics are mean={m[1]}, std={s[1]} - a z distribution "
                "concentrated within a few index units of 0",
                "graph.py:16-40 convert_to_3d assigns z = 0.0 to every 2D input, which is "
                "exactly the mechanism that would produce those statistics",
            ],
            "consequence": "our crops are (T=100, Z=64, Y=256, X=256) and our nodes fill that z "
                           "range. No choice of unit puts our z inside the shipped z "
                           "distribution - see hoct_scale_gate.py",
            "provenance_ceiling": "INFERENCE FROM SHIPPED CONSTANTS. It is not a training "
                                  "manifest and must not be quoted as one. FACT-0451 stands: "
                                  "the training data CANNOT BE ESTABLISHED",
        },
    }


# ---------------------------------------------------------------------------------------------
# 2. BUILD - and refuse
# ---------------------------------------------------------------------------------------------
def build_node_feats(columns: dict[str, np.ndarray], n_nodes: int) -> np.ndarray:
    """Assemble the (N, 19) matrix. REFUSES on any absent column. Never fills.

    ``columns`` is keyed by graph node-attribute name; ``inertia_tensor`` is expected as (N, 3, 3).
    """
    missing = []
    for slot in SLOTS:
        if slot.column not in columns:
            missing.append(slot.name)
    if missing:
        raise FeatureRefusal(
            "cannot assemble the 19-vector - required inputs absent and zero-filling is "
            "forbidden (PKT-0048; FACT-0454)", sorted(set(missing))
        )

    out = np.empty((n_nodes, 19), dtype=np.float64)
    for slot in SLOTS:
        col = columns[slot.column]
        if slot.column == "inertia_tensor":
            arr = np.asarray(col, dtype=np.float64)
            if arr.shape != (n_nodes, 3, 3):
                raise FeatureRefusal(
                    f"inertia_tensor must be (N,3,3), got {arr.shape}", [slot.name]
                )
            r, c = divmod(slot.index - 9, 3)
            out[:, slot.index] = arr[:, r, c]
        else:
            arr = np.asarray(col, dtype=np.float64).reshape(-1)
            if arr.shape != (n_nodes,):
                raise FeatureRefusal(
                    f"{slot.column} must be (N,), got {arr.shape}", [slot.name]
                )
            out[:, slot.index] = arr
    return out


# ---------------------------------------------------------------------------------------------
# 3. AUDIT - the part that cannot be fooled by correct bookkeeping over wrong data
# ---------------------------------------------------------------------------------------------
ZERO_FILL_RAW = "ZERO_FILL_RAW"                    # 0.0 everywhere - graph.py:184-187 / :285-287
ZERO_FILL_STANDARDIZED = "ZERO_FILL_STANDARDIZED"  # -mean/std everywhere - the post-Standardize form
CONSTANT_OTHER = "CONSTANT_OTHER"
VARYING = "VARYING"

_ATOL = 1e-9


def _classify(col: np.ndarray, index: int, standardized: bool) -> tuple[str, dict]:
    finite = col[np.isfinite(col)]
    if finite.size == 0:
        return CONSTANT_OTHER, {"n_finite": 0}
    spread = float(finite.max() - finite.min())
    detail = {
        "n_unique": int(np.unique(np.round(finite, 12)).size),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "spread": spread,
        "std": float(finite.std()),
    }
    if spread > _ATOL:
        return VARYING, detail
    value = float(finite[0])
    detail["constant_value"] = value
    if abs(value) <= _ATOL:
        return ZERO_FILL_RAW, detail
    zf = -PUBLISHER_MEAN[index] / max(PUBLISHER_STD[index], 1e-7)
    detail["standardized_zero_fill_value"] = zf
    if standardized and abs(value - zf) <= 1e-3 * max(1.0, abs(zf)):
        return ZERO_FILL_STANDARDIZED, detail
    return CONSTANT_OTHER, detail


def audit_matrix(arr: np.ndarray, *, standardized: bool = False) -> dict:
    """Judge a produced (N, 19) matrix. Sees ONLY the array - no provenance is trusted.

    ``standardized`` says whether ``Standardize`` has already been applied, because that is what
    turns a zero fill from 0.0 into the constant ``-mean/std``.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 19:
        return {
            "verdict": "REFUSE",
            "reason": f"expected (N, 19), got {arr.shape}",
            "slots": [],
        }
    if arr.shape[0] < 2:
        return {
            "verdict": "REFUSE",
            "reason": "fewer than 2 nodes - a constant column is not distinguishable from a "
                      "zero fill on one row, so the audit cannot discharge its own job",
            "slots": [],
        }

    slots = []
    offenders = []
    for slot in SLOTS:
        kind, detail = _classify(arr[:, slot.index], slot.index, standardized)
        entry = {
            "index": slot.index,
            "name": slot.name,
            "source": slot.source,
            "computable_today": slot.computable_today,
            "classification": kind,
            **detail,
        }
        slots.append(entry)
        if kind in (ZERO_FILL_RAW, ZERO_FILL_STANDARDIZED):
            offenders.append(entry)
    verdict = "REFUSE" if offenders else "PASS"
    return {
        "verdict": verdict,
        "reason": (
            "zero-filled slot(s) detected - a 19-vector of the right SHAPE carrying no "
            "information; refusing per PKT-0048"
            if offenders else
            "every slot varies across nodes; no zero-fill signature present"
        ),
        "standardized_input": bool(standardized),
        "n_nodes": int(arr.shape[0]),
        "offending_slots": [o["name"] for o in offenders],
        "slots": slots,
    }


# ---------------------------------------------------------------------------------------------
# 4. THE COVERAGE REPORT
# ---------------------------------------------------------------------------------------------
def coverage_report() -> dict:
    rows = [asdict(s) for s in SLOTS]
    return {
        "schema_version": 1,
        "packet": "PKT-0048",
        "upstream": UPSTREAM,
        "n_slots": len(SLOTS),
        "n_computable_today": sum(1 for s in SLOTS if s.computable_today),
        "n_blocked": sum(1 for s in SLOTS if not s.computable_today),
        "blocked_slot_names": [s.name for s in SLOTS if not s.computable_today],
        "correction_to_FACT_0454": {
            "recorded": "15 of 19 require instance masks",
            "read_at_source": "14 of 19",
            "which_slot_moves": "border_dist",
            "why": "graph.py:172-173 removes border_dist from the regionprops request and "
                   "features.py:156-189 computes it from z,y,x plus the volume shape. It is a "
                   "POINTS-ONLY feature. The blocker is unchanged in kind; only the count moves",
            "consequence_for_the_route": "none - 14 blocked slots still means an official-HOCT "
                                         "run without masks is 14/19 constant",
        },
        "zero_fill_sites": [
            {"where": "graph.py:184-187",
             "what": "create_graph, images is None -> add_node_attr_key(prop, Float32, 0.0) for "
                     "every intensity property"},
            {"where": "graph.py:283-290",
             "what": "add_features, same fill, and the feature is then DROPPED from "
                     "missing_features so nothing ever overwrites it"},
            {"where": "_batching.py:202-212",
             "what": "nulls and NaNs are filled with 0 behind a LOG.warning - a warning is not "
                     "a refusal, and on a non-interactive kernel run nobody reads it"},
        ],
        "points_only_entry_point": {
            "name": "create_graph_from_points",
            "state": "unimplemented stub - body is `pass`, returns None",
            "where": "_api.py:133-134",
            "exported_in___all__": True,
            "consequence": "a caller who follows the published API gets None, then an "
                           "AttributeError far from the cause",
        },
        "slots": rows,
    }


# ---------------------------------------------------------------------------------------------
# 5. THE DEMONSTRATED REFUSAL - mutation, not assertion
# ---------------------------------------------------------------------------------------------
def _synthetic_full(n: int = 64, seed: int = 20260831) -> np.ndarray:
    """A matrix in which every slot genuinely varies. Stands in for a mask-derived matrix.

    Its VALUES are arbitrary; the audit's job is to notice a slot that stopped varying, and that
    job is value-independent. This fixture differs from production in COST, not in KIND: it is
    the same (N, 19) array in the same slot order the model is fed.
    """
    rng = np.random.default_rng(seed)
    arr = np.empty((n, 19), dtype=np.float64)
    arr[:, 0] = rng.integers(0, 100, n)
    arr[:, 1] = rng.uniform(0, 64, n)
    arr[:, 2] = rng.uniform(0, 256, n)
    arr[:, 3] = rng.uniform(0, 256, n)
    arr[:, 4] = rng.uniform(6, 18, n)
    arr[:, 5] = rng.uniform(0.0, 0.4, n)
    arr[:, 6] = rng.uniform(0.6, 1.0, n)
    arr[:, 7] = rng.uniform(0.3, 0.8, n)
    arr[:, 8] = rng.uniform(0.05, 0.3, n)
    for i in range(n):
        a = rng.normal(0, 1, (3, 3))
        sym = a @ a.T * 50.0
        arr[i, 9:18] = sym.reshape(9)
    arr[:, 18] = rng.uniform(0, 1, n)
    return arr


def mutation_demo() -> dict:
    """Prove the audit distinguishes a computed slot from a zero-filled one.

    THREE arms. The first must PASS, the next two must REFUSE and must NAME the mutated slot.
    A report that cannot do this is the failure PKT-0048's falsifier names, so it is executed
    rather than asserted in prose.
    """
    base = _synthetic_full()
    arms: list[dict] = []

    clean = audit_matrix(base)
    arms.append({
        "arm": "unmutated",
        "expect": "PASS",
        "verdict": clean["verdict"],
        "offending_slots": clean["offending_slots"],
        "correct": clean["verdict"] == "PASS",
    })

    # ARM 2: the raw zero fill - graph.py:184-187, before Standardize.
    raw = base.copy()
    for i in range(5, 9):
        raw[:, i] = 0.0
    a2 = audit_matrix(raw)
    want2 = ["intensity_min", "intensity_max", "intensity_mean", "intensity_std"]
    arms.append({
        "arm": "zero_fill_raw_intensity_block",
        "mutated_slots": want2,
        "expect": "REFUSE",
        "verdict": a2["verdict"],
        "offending_slots": a2["offending_slots"],
        "correct": a2["verdict"] == "REFUSE" and sorted(a2["offending_slots"]) == sorted(want2),
    })

    # ARM 3: the SAME fill seen after Standardize, where it is no longer zero but -mean/std.
    # This is the one a shape check and a "did we get 19 columns" check both pass.
    std_arr = (base - np.asarray(PUBLISHER_MEAN)) / np.asarray(PUBLISHER_STD)
    for i in (4, 9):
        std_arr[:, i] = -PUBLISHER_MEAN[i] / PUBLISHER_STD[i]
    a3 = audit_matrix(std_arr, standardized=True)
    want3 = ["equivalent_diameter_area", "inertia_tensor[0,0]"]
    arms.append({
        "arm": "zero_fill_after_standardize",
        "mutated_slots": want3,
        "constant_values_seen": [-PUBLISHER_MEAN[i] / PUBLISHER_STD[i] for i in (4, 9)],
        "expect": "REFUSE",
        "verdict": a3["verdict"],
        "offending_slots": a3["offending_slots"],
        "correct": a3["verdict"] == "REFUSE" and sorted(a3["offending_slots"]) == sorted(want3),
        "note": "the fill is NOT zero here - it is -1.4052 and -0.2473. A shape check, a NaN "
                "check and a 'no zeros' check all pass on this matrix",
    })

    # ARM 4: the builder refuses rather than filling, when a column is simply absent.
    cols = {
        "t": np.arange(8.0), "z": np.arange(8.0), "y": np.arange(8.0), "x": np.arange(8.0),
        "border_dist": np.zeros(8),
    }
    try:
        build_node_feats(cols, 8)
        refused, named = False, []
    except FeatureRefusal as exc:
        refused, named = True, exc.slots
    arms.append({
        "arm": "builder_with_points_only_columns",
        "expect": "FeatureRefusal",
        "refused": refused,
        "named_slots": named,
        "correct": refused and len(named) == 14,
        "note": "14 named slots, not 15 - the FACT-0454 correction, executed",
    })

    return {"arms": arms, "all_correct": all(a["correct"] for a in arms)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("report", "mutate", "all"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    payload: dict = {
        "schema_version": 1,
        "packet": "PKT-0048",
        "instrument": "scripts/win_bet/hoct_feature_contract.py",
        "binding_restriction": {
            "fact": "FACT-0451",
            "text": "general_v1's training data CANNOT BE ESTABLISHED and overlap with the "
                    "competition movies cannot be excluded. Restricted to submission-only "
                    "judgement",
            "offline_scoreable": False,
        },
    }
    if args.mode in ("report", "all"):
        payload["slot_map_proof"] = prove_slot_map()
        payload["coverage"] = coverage_report()
    if args.mode in ("mutate", "all"):
        payload["mutation"] = mutation_demo()

    ok = True
    if "slot_map_proof" in payload:
        ok = ok and payload["slot_map_proof"]["all_passed"]
    if "mutation" in payload:
        ok = ok and payload["mutation"]["all_correct"]
    payload["heartbeat"] = HEARTBEAT_OK if ok else HEARTBEAT_REFUSED
    payload["passes"] = ok

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(payload["heartbeat"])
    if "slot_map_proof" in payload:
        for c in payload["slot_map_proof"]["checks"]:
            print(f"  slot-map {c['check']:<32} {'PASS' if c['passes'] else 'FAIL'}  {c['detail']}")
        cov = payload["coverage"]
        print(f"\n  {cov['n_computable_today']}/19 computable today, "
              f"{cov['n_blocked']}/19 blocked: {', '.join(cov['blocked_slot_names'][:5])} ...")
    if "mutation" in payload:
        print()
        for a in payload["mutation"]["arms"]:
            print(f"  mutation {a['arm']:<34} {'OK' if a['correct'] else 'WRONG'}  "
                  f"-> {a.get('verdict') or a.get('refused')} {a.get('offending_slots') or a.get('named_slots')}")
    print(f"\n  -> {args.out}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
