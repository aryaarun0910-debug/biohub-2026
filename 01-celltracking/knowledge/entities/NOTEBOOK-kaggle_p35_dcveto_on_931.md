---
id: NOTEBOOK-kaggle_p35_dcveto_on_931
kind: NOTEBOOK
status: leaderboard_champion+operational_base
roles: ["leaderboard_champion", "operational_base"]
tags: [notebook]
---

> [!info] Generated navigation. Do not edit.
> Built by `scripts/core/knowledge.py` from the canonical catalog. The registry is the
> only source of truth for numbers; every value below is a LINK, never a copy.


**Path:** `notebooks/kaggle_p35_dcveto_on_931/biohub-p35-dcveto-on-931.ipynb`
**Spec:** `scripts/kaggle_specs/p35_dcveto_on_931.json`
**Status basis:** declared in baseline_roles.yaml
**Environment variables:** 47

## outgoing
- `descends_from` -> [[NOTEBOOK-kaggle_p32_public931_exact]]  <sub>spec base_notebook</sub>
- `enables` -> [[FEATURE-BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_BIDIRECTIONAL_FUSION_MODE]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_DEEPCENTER_CHECKPOINT]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_DEEPCENTER_EXPECTED_EPOCH]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_DEEPCENTER_GAP_CONFIRM_MIN_SPAN_UM]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_DEEPCENTER_GAP_THRESHOLD]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_DEEPCENTER_GAP_VETO]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_DEEPCENTER_SAFE_DIV_VETO]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_DET_THRESHOLD]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_DIAGNOSTIC_ARM]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_DUAL_SEED_EDGE_THRESHOLD]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_GAP_CLOSE_MAX_GAP]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_GAP_CLOSE_UM]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_GAP_DENSITY_ADAPTIVE]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_GAP_DENSITY_GAIN]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_GAP_DENSITY_NEIGHBORS]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_GAP_DENSITY_REFERENCE_UM]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_ILP_APPEARANCE_WEIGHT]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_ILP_DISAPPEARANCE_WEIGHT]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_ILP_DIVISION_WEIGHT]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_MOTION_RELINK_LEARNED_BONUS]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_OUTPUT_FILTER_SHORT_TRACKS]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_OUTPUT_GAP2_RECOVERY]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_OUTPUT_MIN_TRACK_LEN]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_REQUIRE_DEEPCENTER_VETO]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_RUN_OUTPUT_DIAGNOSTICS]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_SAFE_DIV_FRAME_FRAC_CAP]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_SAFE_DIV_MAX_UM]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_SAFE_DIV_SISTER_MAX_UM]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_SECONDARY_DETECTION_WEIGHT]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_SECONDARY_EDGE_WEIGHT]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_SECONDARY_LINK_MODE]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_SECONDARY_LOW_MARGIN_MAX]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_SECONDARY_MIX_TEMPERATURE]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_SECONDARY_WEIGHTS]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB]]  <sub>notebook environment</sub>
- `enables` -> [[FEATURE-BIOHUB_USE_DEEPCENTER_VETO]]  <sub>notebook environment</sub>

## incoming
- `builds` <- [[SPEC-scripts__kaggle_specs__p35_dcveto_on_931.json]]  <sub>spec out_dir</sub>
- `descends_from` <- [[NOTEBOOK-kaggle_p38_relink_bonus_b2]]  <sub>spec base_notebook</sub>
- `runs` <- [[EXP-0042]]  <sub>experiment spec -> notebook</sub>
