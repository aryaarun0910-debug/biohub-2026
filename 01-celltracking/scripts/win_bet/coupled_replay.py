"""CPU replay of the coupled decomposition arms from cached pre-graph detections.

The GPU stage (`predict_video`) is the only threshold-dependent step; everything after it
is deterministic CPU work:

    (coords, edges)  ->  build_graph  ->  [optional ILP]  ->  wrapper  ->  exact score

Edge logits CANNOT be shared across detector thresholds: the edge head cross-attends over
the whole node set and `predict_video` applies softmax over the source axis, so both the
attention context and the normaliser change when the node population changes. Each
threshold therefore gets its own exact inference pass; this script never mixes them.

Arm A replays from the existing parity-proven `oof_clean` GEFFs (greedy, no ILP) rather
than from a cache, because that artifact *is* the reference.

Robustness (per the execution order):
  - atomic per-crop writes + a resumable status manifest;
  - each ILP solve in its own subprocess so an OOM cannot destroy completed work;
  - solver config, candidate hash, runtime, peak RSS and completion status recorded;
  - partial coverage is reported and never silently scored.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import coupled_arms as CA  # noqa: E402
from biotrack import wrapper as W  # noqa: E402
from e0c_run import (  # noqa: E402
    atomic_write_json, atomic_write_parquet, build_nodes_edges, graph_rows,
)

CACHE_ROOT = ROOT / "artifacts" / "kaggle" / "coupled_cache"
RAW_OOF = ROOT / "artifacts" / "kaggle" / "oof_clean"
FOLDS = {0: "44b6", 1: "6bba"}


def det_cache_dir(det: float) -> Path:
    return CACHE_ROOT / f"det_{det:.5f}".rstrip("0").rstrip(".")


def out_dir(arm: str) -> Path:
    return CACHE_ROOT / "arms" / arm


# --------------------------------------------------------------------------- graph build
def graph_from_cached_detections(path: Path):
    """Rebuild the pre-ILP tracksdata graph from a cached (coords, edges) parquet."""
    import tracksdata as td

    df = pl.read_parquet(path)
    nodes = df.filter(pl.col("row_type") == "node")
    edges = df.filter(pl.col("row_type") == "edge")

    graph = td.graph.RustWorkXGraph()
    graph.add_node_attr_key("t", 0)
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, 0.0)
    graph.add_edge_attr_key("edge_prob", 0.0)
    graph.add_edge_attr_key("edge_dist", 0.0)

    id_map: dict[int, int] = {}
    for row in nodes.iter_rows(named=True):
        nid = graph.add_node({"t": int(row["t"]), "z": float(row["z"]),
                              "y": float(row["y"]), "x": float(row["x"])})
        id_map[int(row["node_id"])] = nid
    for row in edges.iter_rows(named=True):
        s, t = id_map.get(int(row["source_id"])), id_map.get(int(row["target_id"]))
        if s is None or t is None:
            continue
        graph.add_edge(s, t, {"edge_prob": float(row["edge_prob"]),
                              "edge_dist": float(row["edge_dist"])})
    return graph


def solve_ilp(graph, ilp: dict):
    """Global ILP selection. Deterministic single-thread where the backend allows."""
    import tracksdata as td

    solver = td.solvers.ILPSolver(
        edge_weight=ilp["edge_weight"] * td.EdgeAttr("edge_prob"),
        appearance_weight=ilp["appearance_weight"],
        disappearance_weight=ilp["disappearance_weight"],
        division_weight=ilp["division_weight"],
    )
    return solver.solve(graph)


# ------------------------------------------------------------------ transductive prior
def density_prior(paths: list[Path]) -> float:
    """Label-free per-fold spacing prior, identical to clean903_wrapper_run.py.

    Used ONLY by arm D-port, whose PREFIX_DENSITY_BLEND is non-zero. Every other arm
    leaves the prior empty so the blend is inert.
    """
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


# --------------------------------------------------------------------------- one crop
def replay_crop(arm: str, split: int, crop: str) -> dict:
    """Run one crop through one arm. Returns a status record; writes the graph atomically."""
    spec = CA.ARMS[arm]
    t0 = time.time()

    CA.WRAPPER_SETTERS[spec["wrapper"]]()
    if arm == "A":
        src = RAW_OOF / f"pred_geffs_split_{split}" / f"{crop}.geff"
        graph = W.graph_from_geff(src)
        cand_hash = hashlib.sha256(str(src).encode()).hexdigest()[:16]
    else:
        src = det_cache_dir(spec["det"]) / str(split) / f"{crop}.parquet"
        if not src.exists():
            return {"crop": crop, "split": split, "arm": arm, "status": "missing_cache",
                    "detail": str(src)}
        cand_hash = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
        graph = graph_from_cached_detections(src)
        if spec["ilp"] is not None and graph.num_edges() > 0:
            graph = solve_ilp(graph, spec["ilp"])

    nbi, raw = build_nodes_edges(graph)
    fn, fe, stats = W.filter_output_graph(copy.deepcopy(nbi), raw, dataset=crop)

    dest = out_dir(arm) / str(split)
    dest.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(pl.DataFrame(graph_rows(fn, fe)), dest / f"{crop}.parquet")

    return {
        "crop": crop, "split": split, "arm": arm, "status": "ok",
        "config_hash": CA.config_hash(arm), "candidate_hash": cand_hash,
        "det_threshold": spec["det"], "ilp": spec["ilp"],
        "wrapper": spec["wrapper"],
        "nodes_pre_wrapper": len(nbi), "nodes_post_wrapper": len(fn),
        "edges_pre_wrapper": len(raw), "edges_post_wrapper": len(fe),
        "runtime_s": round(time.time() - t0, 2),
        "peak_rss_mb": peak_rss_mb(),
        "wrapper_stats": {k: v for k, v in stats.items() if isinstance(v, (int, float))},
    }


def peak_rss_mb() -> float:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1e6, 1)
    except Exception:
        return float("nan")


def crops_for(split: int) -> list[str]:
    return sorted(p.stem for p in (RAW_OOF / f"pred_geffs_split_{split}").glob("*.geff"))


# ------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(CA.ARMS))
    ap.add_argument("--splits", default="0,1")
    ap.add_argument("--crops", default="", help="comma-separated crop subset (preflight)")
    ap.add_argument("--isolated", action="store_true",
                    help="run each crop in its own subprocess (ILP OOM containment)")
    a = ap.parse_args()

    arm = a.arm
    splits = [int(s) for s in a.splits.split(",") if s != ""]
    subset = [c for c in a.crops.split(",") if c]

    status_dir = out_dir(arm) / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    print(f"[coupled_replay] arm={arm} config_hash={CA.config_hash(arm)} "
          f"det={CA.ARMS[arm]['det']} ilp={CA.ARMS[arm]['ilp']} "
          f"wrapper={CA.ARMS[arm]['wrapper']}", flush=True)

    # D-port is the only arm with a live prefix-density blend.
    if CA.ARMS[arm]["wrapper"] == "v122_port":
        for split in splits:
            paths = [RAW_OOF / f"pred_geffs_split_{split}" / f"{c}.geff"
                     for c in (subset or crops_for(split))]
            prior = density_prior([p for p in paths if p.exists()])
            W.PREFIX_DENSITY_PRIOR_UM[FOLDS[split]] = prior
            print(f"  transductive density prior {FOLDS[split]} = {prior:.4f} um", flush=True)

    done = failed = skipped = 0
    for split in splits:
        for crop in (subset or crops_for(split)):
            sp = status_dir / f"{split}__{crop}.json"
            if sp.exists() and json.loads(sp.read_text()).get("status") == "ok":
                skipped += 1
                continue
            try:
                if a.isolated:
                    rec = run_isolated(arm, split, crop)
                else:
                    rec = replay_crop(arm, split, crop)
            except Exception as exc:  # noqa: BLE001
                rec = {"crop": crop, "split": split, "arm": arm, "status": "failed",
                       "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()[-2000:]}
            atomic_write_json(rec, sp)
            if rec.get("status") == "ok":
                done += 1
                print(f"  ok  s{split} {crop}: {rec['nodes_pre_wrapper']}->"
                      f"{rec['nodes_post_wrapper']} nodes, {rec['runtime_s']}s", flush=True)
            else:
                failed += 1
                print(f"  FAIL s{split} {crop}: {rec.get('status')} "
                      f"{rec.get('error', rec.get('detail', ''))}", flush=True)

    print(f"[coupled_replay] arm={arm} done={done} skipped={skipped} failed={failed}")
    if failed:
        print("  NOTE: partial coverage -- do NOT score this arm until failures are resolved.")


def run_isolated(arm: str, split: int, crop: str) -> dict:
    """Re-invoke this module for a single crop so an ILP OOM cannot kill the batch."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--arm", arm,
           "--splits", str(split), "--crops", crop]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    sp = out_dir(arm) / "status" / f"{split}__{crop}.json"
    if sp.exists():
        return json.loads(sp.read_text())
    return {"crop": crop, "split": split, "arm": arm, "status": "subprocess_died",
            "returncode": proc.returncode, "stderr": proc.stderr[-2000:]}


if __name__ == "__main__":
    main()
