"""OOF gate for the clean-public-0.903 wrapper delta on frozen 0.990 detections.

This is deliberately a *partial* reproduction: it evaluates the notebook's
density-adaptive gap closing and wrapper constants using the already cached raw
OOF prediction graphs.  It does not claim to test the notebook's 0.970 detector
threshold; that requires a separate GPU regeneration if this CPU gate survives.

The same-embryo density prior is transductive but label-free: for each held-out
family it is computed from all predicted graphs in that fold, matching test-time
availability without touching GT.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

from biotrack import wrapper as W  # noqa: E402
from e0c_run import (  # noqa: E402
    atomic_write_json, atomic_write_parquet, build_nodes_edges, candidate_rows,
    git_commit, graph_rows,
)

CACHE = ROOT / "artifacts" / "kaggle" / "clean903_wrapper_oof_cache"
RAW_ROOT = ROOT / "artifacts" / "kaggle" / "oof_clean"


def set_clean903_wrapper_config() -> None:
    W.GAP_CLOSE_MAX_GAP = 2  # effective implementation remains one missing frame
    W.GAP_CLOSE_UM = 5.8
    W.GAP_DENSITY_ADAPTIVE = True
    W.GAP_DENSITY_REFERENCE_UM = 6.5
    W.GAP_DENSITY_GAIN = 0.040
    W.GAP_DENSITY_MAX_STEP_DELTA_UM = 0.125
    W.GAP_DENSITY_NEIGHBORS = 3
    W.PREFIX_DENSITY_BLEND = 0.20
    W.OUTPUT_MIN_TRACK_LEN = 6
    W.SHORT_TRACK_MIN_LEN_BY_DATASET = {}
    W.OUTPUT_GAP2_RECOVERY = False
    W.SAFE_DIV_MAX_UM = 4.66
    W.SAFE_DIV_SISTER_MAX_UM = 8.5
    W.SAFE_DIV_EXISTING_CHILD_MAX_UM = 7.65
    W.SAFE_DIV_FRAME_FRAC_CAP = 0.0076
    W.SAFE_DIV_GLOBAL_FRAC_CAP = 0.00385
    W.GAP_REFINE_SYNTHETIC = True
    W.TEST_DIR = ROOT / "data" / "train"


def density_prior(paths: list[Path]) -> float:
    samples: list[float] = []
    for path in paths:
        graph = W.graph_from_geff(path)
        by_t: dict[int, list[np.ndarray]] = {}
        for row in graph.node_attrs().iter_rows(named=True):
            by_t.setdefault(int(row["t"]), []).append(np.array([
                float(row["z"]) * W.VOXEL_SCALE_UM[0],
                float(row["y"]) * W.VOXEL_SCALE_UM[1],
                float(row["x"]) * W.VOXEL_SCALE_UM[2],
            ], dtype=np.float64))
        for points in by_t.values():
            if len(points) < 4:
                continue
            arr = np.stack(points)
            k = min(len(points), max(2, W.GAP_DENSITY_NEIGHBORS + 1))
            distances, _ = cKDTree(arr).query(arr, k=k)
            if distances.ndim == 1:
                distances = distances[:, None]
            neighbours = distances[:, 1:]
            neighbours = neighbours[np.isfinite(neighbours)]
            if neighbours.size:
                samples.append(float(np.median(neighbours)))
    return float(np.median(samples)) if samples else float(W.GAP_DENSITY_REFERENCE_UM)


def config_hash(priors: dict[str, float]) -> str:
    keys = [
        "GAP_CLOSE_MAX_GAP", "GAP_CLOSE_UM", "GAP_DENSITY_ADAPTIVE",
        "GAP_DENSITY_REFERENCE_UM", "GAP_DENSITY_GAIN",
        "GAP_DENSITY_MAX_STEP_DELTA_UM", "GAP_DENSITY_NEIGHBORS",
        "PREFIX_DENSITY_BLEND", "OUTPUT_MIN_TRACK_LEN", "OUTPUT_GAP2_RECOVERY",
        "SAFE_DIV_MAX_UM", "SAFE_DIV_SISTER_MAX_UM",
        "SAFE_DIV_EXISTING_CHILD_MAX_UM", "SAFE_DIV_FRAME_FRAC_CAP",
        "SAFE_DIV_GLOBAL_FRAC_CAP", "GAP_REFINE_SYNTHETIC",
    ]
    payload = {key: getattr(W, key) for key in keys}
    payload["priors"] = priors
    payload["raw_detector_threshold"] = 0.990
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--splits", default="0,1")
    args = parser.parse_args()
    shard, nshards = (int(value) for value in args.shard.split("/"))
    splits = [int(value) for value in args.splits.split(",")]

    set_clean903_wrapper_config()
    paths_by_split = {
        split: sorted((RAW_ROOT / f"pred_geffs_split_{split}").glob("*.geff"))
        for split in splits
    }
    priors: dict[str, float] = {}
    for paths in paths_by_split.values():
        if paths:
            priors[paths[0].stem.split("_", 1)[0]] = density_prior(paths)
    # Each split contains one embryo prefix.
    W.PREFIX_DENSITY_PRIOR_UM = priors
    W.GLOBAL_DENSITY_MEDIAN_UM = float(np.median(list(priors.values())))
    cfg_hash = config_hash(priors)
    commit = git_commit()
    print(f"[clean903-wrapper] shard={shard}/{nshards} cfg={cfg_hash} priors={priors}", flush=True)

    for subdir in ("graphs", "candidates"):
        for split in splits:
            (CACHE / subdir / str(split)).mkdir(parents=True, exist_ok=True)
    (CACHE / "status").mkdir(parents=True, exist_ok=True)

    done = skipped = failed = 0
    for split, paths in paths_by_split.items():
        for index, path in enumerate(paths):
            if index % nshards != shard:
                continue
            crop = path.stem
            status_path = CACHE / "status" / f"{split}__{crop}.json"
            if status_path.exists():
                status = json.loads(status_path.read_text())
                if status.get("status") == "ok" and status.get("config_hash") == cfg_hash:
                    skipped += 1
                    continue
            try:
                graph = W.graph_from_geff(path)
                nodes, raw_edges = build_nodes_edges(graph)
                raw_nodes = len(nodes)
                W._CANDIDATE_SINK = []
                final_nodes, final_edges, stats = W.filter_output_graph(
                    copy.deepcopy(nodes), raw_edges, dataset=crop
                )
                candidates = W._CANDIDATE_SINK
                W._CANDIDATE_SINK = None
                atomic_write_parquet(graph_rows(final_nodes, final_edges),
                                     CACHE / "graphs" / str(split) / f"{crop}.parquet")
                atomic_write_parquet(candidate_rows(candidates, nodes),
                                     CACHE / "candidates" / str(split) / f"{crop}.parquet")
                atomic_write_json({
                    "crop": crop, "split": split, "status": "ok",
                    "config_hash": cfg_hash, "git_commit": commit,
                    "experiment_scope": "clean903 wrapper on frozen 0.990 detections",
                    "density_prior_um": priors[crop.split("_", 1)[0]],
                    "raw_nodes": raw_nodes, "raw_edges": len(raw_edges),
                    "final_nodes": len(final_nodes), "final_edges": len(final_edges),
                    "diagnostics": {key: int(value) for key, value in stats.items()},
                }, status_path)
                done += 1
                print(f"  ok s{split} {crop}: {raw_nodes}->{len(final_nodes)} nodes", flush=True)
            except Exception as exc:
                W._CANDIDATE_SINK = None
                atomic_write_json({"crop": crop, "split": split, "status": "failed",
                                   "config_hash": cfg_hash, "error": str(exc),
                                   "trace": traceback.format_exc()[:2000]}, status_path)
                failed += 1
                print(f"  FAIL s{split} {crop}: {exc}", flush=True)
    print(f"[clean903-wrapper] done={done} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
