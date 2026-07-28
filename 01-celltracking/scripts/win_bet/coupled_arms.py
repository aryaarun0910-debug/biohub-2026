"""Arm definitions for the coupled detector/ILP/wrapper decomposition (2026-07-28).

Preregistered path (ORDER-DEPENDENT, CONTAINS INTERACTIONS -- see the note below):

  A   0.990  detector + greedy selection      + E0c wrapper   (existing artifact, no GPU)
  B   0.9690 detector + greedy selection      + E0c wrapper
  B'  0.9690 detector + ILP app 0.1 / dis 0.1 + E0c wrapper
  C   0.9690 detector + ILP app 0.0 / dis 1.5 + E0c wrapper
  D   0.9690 detector + ILP app 0.0 / dis 1.5 + v122 wrapper   <- promotion candidate
  C0  0.96875 detector + ILP app 0.0 / dis 1.5 + v122 wrapper  <- promotion candidate

Attribution reading:
  A->B   detected node population under greedy selection
  B->B'  introducing the global ILP at its default survival costs
  B'->C  changing ILP appearance/disappearance costs
  C->D   wrapper / min-track / gap stage
  A->D   total coupled effect
  C0 vs D  the only threshold comparison under identical downstream processing

**This decomposition is order-dependent and its steps contain interactions.** The listed
deltas are a single sequential path through a non-commutative configuration space; a
different ordering would attribute different magnitudes to the same stages. Only D and C0
are promotion candidates. B, B' and C are preregistered diagnostics.

IMPORTANT -- E0c has NO ILP stage. `oof_clean/pred_geffs_split_{0,1}` was produced by
notebooks/kaggle_predict_score/kaggle_predict_score.py with USE_ILP = False at
det-threshold 0.99, i.e. greedy candidate selection (max_parents=1, max_children=2)
applied inside predict_video. Arm A is therefore detector + greedy + wrapper.

IMPORTANT -- the v122 wrapper constants below are taken from the RETAINED v122 source
(notebooks/kaggle_clean_v122/), not from scripts/win_bet/clean903_wrapper_run.py. Three
constants differ between them; see V122_VS_CLEAN903_DELTAS.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from biotrack import wrapper as W  # noqa: E402

# Provenance lock for the promotion-candidate wrapper. Both D and C0 MUST derive their
# wrapper constants from this retained source, never from the clean-903 port.
V122_SOURCE = ROOT / "notebooks" / "kaggle_clean_v122" / "biohub-clean-v122-reproduction.ipynb"
V122_SOURCE_SHA256 = "703A05E25483F904D3555A3BA0A38F5C1C614968C1F53C63CD0316506C7C2FBC"
V122_KERNEL = "aryaarun07/biohub-clean-v122-reproduction v1"

# The three constants whose drift this module exists to prevent. Locked by
# tests/test_coupled_arms.py so the clean-903 port's values cannot silently return.
V122_LOCKED_CONSTANTS = {
    "PREFIX_DENSITY_BLEND": 0.0,
    "SAFE_DIV_GLOBAL_FRAC_CAP": 0.00375,
    "MOTION_RELINK_LEARNED_BONUS": 1.0,
}

# Differences found between the retained v122 source and the earlier clean-903 port.
# The clean-903 port targeted a DIFFERENT public notebook, so these are not necessarily
# bugs in that script -- but they mean it must not be reused as "the v122 wrapper".
V122_VS_CLEAN903_DELTAS = {
    "PREFIX_DENSITY_BLEND": {
        "v122": 0.0, "clean903_port": 0.20,
        "why": "v122's frame_local_spacing uses pure local kNN spacing "
               "(result[node_id] = spacing). It has no prefix/embryo density prior at all; "
               "the blended transductive prior is our own addition.",
    },
    "SAFE_DIV_GLOBAL_FRAC_CAP": {"v122": 0.00375, "clean903_port": 0.00385, "why": "verbatim env value"},
    "MOTION_RELINK_LEARNED_BONUS": {"v122": 1.0, "clean903_port": 0.75, "why": "v122 sets it explicitly; port left the default"},
}


def set_e0c_wrapper() -> None:
    """E0c deployment-exact wrapper (matches scripts/win_bet/e0c_run.py::set_e0c_config)."""
    W.OUTPUT_MIN_TRACK_LEN = 7
    W.SHORT_TRACK_MIN_LEN_BY_DATASET = {}
    W.GAP_REFINE_SYNTHETIC = True
    W.TEST_DIR = ROOT / "data" / "train"
    # Everything else stays at wrapper.py defaults; density adaptation OFF.
    W.GAP_DENSITY_ADAPTIVE = False
    W.PREFIX_DENSITY_BLEND = 0.0
    W.PREFIX_DENSITY_PRIOR_UM = {}


def set_v122_wrapper() -> None:
    """v122 wrapper, verbatim from the retained notebook's environment block."""
    W.OUTPUT_FILTER_SHORT_TRACKS = True
    W.OUTPUT_MIN_TRACK_LEN = 6
    W.OUTPUT_KEEP_DIVISION_COMPONENTS = True
    W.SHORT_TRACK_MIN_LEN_BY_DATASET = {}
    W.GAP_CLOSE_MAX_GAP = 2
    W.GAP_CLOSE_UM = 5.8
    W.GAP_DENSITY_ADAPTIVE = True
    W.GAP_DENSITY_REFERENCE_UM = 6.5
    W.GAP_DENSITY_GAIN = 0.040
    W.GAP_DENSITY_MAX_STEP_DELTA_UM = 0.125
    W.GAP_DENSITY_NEIGHBORS = 3
    # v122 has NO prefix-density prior -> pure local spacing.
    W.PREFIX_DENSITY_BLEND = 0.0
    W.PREFIX_DENSITY_PRIOR_UM = {}
    W.OUTPUT_GAP2_RECOVERY = False
    W.SAFE_DIV_MAX_UM = 4.66
    W.SAFE_DIV_SISTER_MAX_UM = 8.5
    W.SAFE_DIV_EXISTING_CHILD_MAX_UM = 7.65
    W.SAFE_DIV_FRAME_FRAC_CAP = 0.0076
    W.SAFE_DIV_GLOBAL_FRAC_CAP = 0.00375
    W.MOTION_RELINK_LEARNED_BONUS = 1.0
    W.ADAPTIVE_SHORT_TRACK_RESCUE = False
    W.GAP_REFINE_SYNTHETIC = True
    W.TEST_DIR = ROOT / "data" / "train"


def set_v122_port_wrapper() -> None:
    """DIAGNOSTIC ONLY (arm D-port): v122 wrapper with the clean-903 port's three drifted
    constants restored. Identical to `set_v122_wrapper` in every other respect, so
    D-port minus D isolates the BUNDLED effect of the port's wrapper drift. It does NOT
    identify any individual constant, and it is NOT a promotion candidate.
    """
    set_v122_wrapper()
    W.PREFIX_DENSITY_BLEND = 0.20
    W.SAFE_DIV_GLOBAL_FRAC_CAP = 0.00385
    W.MOTION_RELINK_LEARNED_BONUS = 0.75
    # NOTE: a non-zero blend is inert unless W.PREFIX_DENSITY_PRIOR_UM is populated. The
    # replay driver must set it per held-out family, using the same transductive,
    # label-free estimator as clean903_wrapper_run.py::density_prior (median over
    # per-frame median kNN spacing across that fold's predicted graphs). For every other
    # arm the prior stays empty and the blend is 0.0, so this affects D-port alone.


# ILP configurations. `None` == greedy (no ILP), which is what E0c/oof_clean used.
ILP_GREEDY = None
ILP_DEFAULT = {"edge_weight": -1.0, "appearance_weight": 0.1,
               "disappearance_weight": 0.1, "division_weight": 1.0}
ILP_C1 = {"edge_weight": -1.0, "appearance_weight": 0.0,
          "disappearance_weight": 1.5, "division_weight": 1.0}

ARMS = {
    "A":  {"det": 0.990,   "ilp": ILP_GREEDY,  "wrapper": "e0c",  "role": "reference",  "source": "existing e0c_cache"},
    "B":  {"det": 0.9690,  "ilp": ILP_GREEDY,  "wrapper": "e0c",  "role": "diagnostic", "source": "cache_0.9690"},
    "Bp": {"det": 0.9690,  "ilp": ILP_DEFAULT, "wrapper": "e0c",  "role": "diagnostic", "source": "cache_0.9690"},
    "C":  {"det": 0.9690,  "ilp": ILP_C1,      "wrapper": "e0c",  "role": "diagnostic", "source": "cache_0.9690"},
    "D":  {"det": 0.9690,  "ilp": ILP_C1,      "wrapper": "v122", "role": "PROMOTION",  "source": "cache_0.9690"},
    "C0": {"det": 0.96875, "ilp": ILP_C1,      "wrapper": "v122", "role": "PROMOTION",  "source": "cache_0.96875"},
    # Seventh arm, diagnostic only. Same cached 0.9690 detections and same C1 ILP output
    # as D; differs ONLY in the three drifted wrapper constants. Must not delay D/C0 and
    # must never become a promotion candidate without explicit preregistration.
    "Dport": {"det": 0.9690, "ilp": ILP_C1, "wrapper": "v122_port", "role": "diagnostic",
              "source": "cache_0.9690"},
}

WRAPPER_SETTERS = {"e0c": set_e0c_wrapper, "v122": set_v122_wrapper,
                   "v122_port": set_v122_port_wrapper}

PROMOTION_ARMS = tuple(k for k, v in ARMS.items() if v["role"] == "PROMOTION")

# Wrapper attributes captured in the config hash (superset of everything an arm varies).
_HASHED_ATTRS = [
    "OUTPUT_MIN_TRACK_LEN", "OUTPUT_FILTER_SHORT_TRACKS", "OUTPUT_KEEP_DIVISION_COMPONENTS",
    "GAP_CLOSE_MAX_GAP", "GAP_CLOSE_UM", "GAP_DENSITY_ADAPTIVE", "GAP_DENSITY_REFERENCE_UM",
    "GAP_DENSITY_GAIN", "GAP_DENSITY_MAX_STEP_DELTA_UM", "GAP_DENSITY_NEIGHBORS",
    "PREFIX_DENSITY_BLEND", "OUTPUT_GAP2_RECOVERY", "SAFE_DIV_MAX_UM", "SAFE_DIV_SISTER_MAX_UM",
    "SAFE_DIV_EXISTING_CHILD_MAX_UM", "SAFE_DIV_FRAME_FRAC_CAP", "SAFE_DIV_GLOBAL_FRAC_CAP",
    "MOTION_RELINK_LEARNED_BONUS", "ADAPTIVE_SHORT_TRACK_RESCUE", "GAP_REFINE_SYNTHETIC",
]


def wrapper_state() -> dict:
    return {k: getattr(W, k, None) for k in _HASHED_ATTRS}


def arm_config(name: str) -> dict:
    spec = ARMS[name]
    WRAPPER_SETTERS[spec["wrapper"]]()
    return {"arm": name, "det_threshold": spec["det"], "ilp": spec["ilp"],
            "wrapper_name": spec["wrapper"], "wrapper": wrapper_state()}


def config_hash(name: str) -> str:
    cfg = arm_config(name)
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


if __name__ == "__main__":
    print(f"{'arm':<5}{'det':>9}  {'ilp(app/dis)':<14}{'wrapper':<7}{'role':<11}{'config_hash':<18}")
    print("-" * 72)
    seen = {}
    for name in ARMS:
        spec, h = ARMS[name], config_hash(name)
        ilp = "greedy" if spec["ilp"] is None else (
            f"{spec['ilp']['appearance_weight']}/{spec['ilp']['disappearance_weight']}")
        print(f"{name:<5}{spec['det']:>9}  {ilp:<14}{spec['wrapper']:<7}{spec['role']:<11}{h:<18}")
        seen.setdefault(h, []).append(name)
    dupes = {h: v for h, v in seen.items() if len(v) > 1}
    print("\nDISTINCT config hashes:", len(seen), "/", len(ARMS))
    print("COLLISIONS:", dupes if dupes else "none - all arms distinguishable")
    print("\nv122 vs clean903-port deltas:")
    for k, v in V122_VS_CLEAN903_DELTAS.items():
        print(f"  {k}: v122={v['v122']} port={v['clean903_port']}\n      {v['why']}")
