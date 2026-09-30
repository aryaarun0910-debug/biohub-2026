# --- ARM B provenance, re-aim census and pin record ---------------------------------
# DIAGNOSTIC ONLY. Nothing in this cell writes to, or reads back into, submission.csv;
# it runs after the file has already been written and closed.
#
# Degrees are keyed on (dataset, node_id). Node ids are NOT unique across movies in this
# submission format, so a bare node_id counter fabricates hub nodes -- on the clean P0-B
# artifact it reports 15,471 "hubs" whose true maximum out-degree is 2.
# The whole body is guarded: a diagnostic must never turn a COMPLETE kernel into an
# ERROR one, because this competition can only submit from a COMPLETED kernel (trap 7).
import traceback as _armb_traceback
from pathlib import Path as _armb_path

try:
    import hashlib as _armb_hashlib
    import json as _armb_json
    import os as _armb_os
    from collections import Counter as _armb_counter
    from pathlib import Path as _armb_path

    _armb_sub = _armb_path("/kaggle/working/submission.csv")
    _armb_bytes = _armb_sub.read_bytes()
    _armb_sha = _armb_hashlib.sha256(_armb_bytes).hexdigest()

    # ---- input pinning: enumerate the mount tree to depth 2 (never recursive: trap 16) ----
    _armb_mounts = []
    _armb_root = _armb_path("/kaggle/input")
    if _armb_root.is_dir():
        for _p1 in sorted(_armb_root.iterdir()):
            _armb_mounts.append(_p1.as_posix())
            if _p1.is_dir():
                for _p2 in sorted(_p1.iterdir())[:12]:
                    if _p2.is_dir():
                        _armb_mounts.append(_p2.as_posix())

    # ---- gate re-aim census, accumulated over every crop of this run ----------------------
    _armb_census = {k: int(v) for k, v in GATE_STATS.items()}
    for _sfx in ("_tight", "_relaxed"):
        _raw = _armb_census.get("admit_raw" + _sfx, 0)
        _arm = _armb_census.get("admit_arm" + _sfx, 0)
        _armb_census["net_delta" + _sfx] = int(_arm - _raw)
        _armb_census["net_delta_frac" + _sfx] = (float(_arm - _raw) / _raw) if _raw else 0.0
        _armb_census["churn_frac" + _sfx] = (
            float(_armb_census.get("newly_admitted" + _sfx, 0)
                  + _armb_census.get("newly_excluded" + _sfx, 0)) / _raw
        ) if _raw else 0.0

    # ---- structural self-check on the emitted artifact, per dataset -----------------------
    import pandas as _armb_pd

    _armb_df = _armb_pd.read_csv(_armb_sub)
    _armb_nodes = _armb_df[_armb_df["row_type"].eq("node")]
    _armb_edges = _armb_df[_armb_df["row_type"].eq("edge")]
    _armb_indeg = _armb_counter(
        zip(_armb_edges["dataset"].astype(str), _armb_edges["target_id"].astype("int64")))
    _armb_outdeg = _armb_counter(
        zip(_armb_edges["dataset"].astype(str), _armb_edges["source_id"].astype("int64")))

    _armb_per_crop = {}
    for _ds in sorted(_armb_df["dataset"].astype(str).unique()):
        _n = _armb_nodes[_armb_nodes["dataset"].astype(str).eq(_ds)]
        _e = _armb_edges[_armb_edges["dataset"].astype(str).eq(_ds)]
        _o = _armb_counter(_e["source_id"].astype("int64").tolist())
        _i = _armb_counter(_e["target_id"].astype("int64").tolist())
        _armb_per_crop[_ds] = {
            "nodes": int(len(_n)), "edges": int(len(_e)),
            "max_indegree": int(max(_i.values(), default=0)),
            "max_outdegree": int(max(_o.values(), default=0)),
            "outdegree_3_plus": int(sum(1 for v in _o.values() if v >= 3)),
            "divisions": int(sum(1 for v in _o.values() if v == 2)),
            "t_min": int(_n["t"].min()), "t_max": int(_n["t"].max()),
            "z_max": int(_n["z"].max()), "y_max": int(_n["y"].max()),
            "x_max": int(_n["x"].max()),
        }

    _armb_struct = {
        "rows": int(len(_armb_df)),
        "nodes": int(len(_armb_nodes)),
        "edges": int(len(_armb_edges)),
        "datasets": sorted(_armb_df["dataset"].astype(str).unique()),
        "t_min": int(_armb_nodes["t"].min()),
        "t_max": int(_armb_nodes["t"].max()),
        "z_min": int(_armb_nodes["z"].min()), "z_max": int(_armb_nodes["z"].max()),
        "y_min": int(_armb_nodes["y"].min()), "y_max": int(_armb_nodes["y"].max()),
        "x_min": int(_armb_nodes["x"].min()), "x_max": int(_armb_nodes["x"].max()),
        "max_indegree": int(max(_armb_indeg.values(), default=0)),
        "max_outdegree": int(max(_armb_outdeg.values(), default=0)),
        "outdegree_3_plus": int(sum(1 for v in _armb_outdeg.values() if v >= 3)),
        "divisions": int(sum(1 for v in _armb_outdeg.values() if v == 2)),
        "duplicate_node_ids": int(
            len(_armb_nodes) - len(_armb_nodes.groupby(["dataset", "node_id"]).size())),
        "negative_time_nodes": int((_armb_nodes["t"] < 0).sum()),
        "out_of_volume_nodes": int((
            (_armb_nodes["z"] < 0) | (_armb_nodes["z"] > 63)
            | (_armb_nodes["y"] < 0) | (_armb_nodes["y"] > 255)
            | (_armb_nodes["x"] < 0) | (_armb_nodes["x"] > 255)).sum()),
        "per_crop": _armb_per_crop,
    }

    _armb_report = {
        "arm": "B_flow_compensated_relink_gate" if ARMB_FLOW_GATE else "A_baseline_raw_distance",
        "armb_flow_gate_enabled": bool(ARMB_FLOW_GATE),
        "semantic_change": (
            "motion_relink_edges candidate eligibility gates on "
            "|target - (source + kNN16 flow(source))| instead of |target - source|; "
            "SAME 6.0/10.0 um radius, SAME cost expression, SAME node population"
        ),
        "base_system": {
            "name": "P0-B",
            "kernel": "aryaarun07/biohub-p0b-clean-913-reverse-time",
            "public_score": 0.914,
            "output_sha256": (
                "4c285cae0c220a11b8ffdeeea1e02de86b5c1e7795daeb2bb27c073e1ab4ecec"
            ),
        },
        "evidence": {
            "p0strict_loeo_pooled": 0.0079822,
            "p0strict_loeo_44b6": 0.0167567,
            "p0strict_loeo_6bba": 0.0067055,
            "e0c_pooled": 0.0088059,
            "basis": "exact-pooled-OOF, complete wrapper, 144/144 crops parity-exact",
        },
        "unchanged": [
            "motion_relink_tight_um", "motion_relink_relaxed_um", "relink cost expression",
            "detector", "model weights", "ILP/association config", "wrapper constants",
            "division logic", "pruning", "submission formatting", "reverse-time block",
        ],
        "constants": {
            "motion_relink_tight_um": float(MOTION_RELINK_TIGHT_UM),
            "motion_relink_relaxed_um": float(MOTION_RELINK_RELAXED_UM),
            "motion_relink_velocity_weight": float(MOTION_RELINK_VELOCITY_WEIGHT),
            "motion_relink_learned_bonus": float(MOTION_RELINK_LEARNED_BONUS),
            "motion_relink_max_frame_nodes": int(MOTION_RELINK_MAX_FRAME_NODES),
            "output_edge_max_um": float(OUTPUT_EDGE_MAX_UM),
            "output_min_track_len": int(OUTPUT_MIN_TRACK_LEN),
            "safe_div_sister_max_um": float(SAFE_DIV_SISTER_MAX_UM),
            "det_threshold": float(DET_THRESHOLD),
            "armb_knn_k": int(ARMB_KNN_K),
            "armb_knn_min": int(ARMB_KNN_MIN),
        },
        "routing_prohibitions": {
            "BIOHUB_SHORT_TRACK_MIN_LEN_BY_DATASET": _armb_os.environ.get(
                "BIOHUB_SHORT_TRACK_MIN_LEN_BY_DATASET", ""
            ),
            "family_literal_in_gate_code": False,
            "gate_reads": "pre-wrapper prediction graph of the crop only; no GT, no image, no state",
        },
        "gate_census": _armb_census,
        "structure": _armb_struct,
        "pins": {
            "run_type": _armb_os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "unset"),
            "gpu_count": int(_torch.cuda.device_count()),
            "input_mounts": _armb_mounts,
        },
        "submission": {"sha256": _armb_sha, "bytes": int(len(_armb_bytes))},
    }
    _armb_path("/kaggle/working/armb_provenance.json").write_text(
        _armb_json.dumps(_armb_report, indent=2, sort_keys=True) + "\n"
    )
    print(_armb_json.dumps(_armb_report, indent=2, sort_keys=True))
except Exception:
    _armb_traceback.print_exc()
    _armb_path('/kaggle/working/armb_provenance_FAILED.txt').write_text(
        _armb_traceback.format_exc())
    print('ARM-B PROVENANCE CELL FAILED -- submission.csv is unaffected; see armb_provenance_FAILED.txt')
