"""E0c Stage 1 — exact deployment wrapper pass, cached once (never repeat).

Runs the parity-verified wrapper with gap-refine ON (min-track-len 7 UNIFORM; the
6bba_05b6850b=6 public-test exception is NOT applied to OOF) over the raw OOF GEFFs,
and for every crop caches, with atomic writes and resume:
  - graphs/{split}/{crop}.parquet   post-wrapper graph (submission rows -> rebuildable)
  - candidates/{split}/{crop}.parquet  FULL pre-assignment candidate surface (features +
    wrapper composite cost + selected), the Phase-B training asset captured in one pass
  - status/{split}__{crop}.json     per-crop manifest (config hash, git commit, counts,
    wrapper diagnostics, status) -- failures are explicit, never silently dropped

Scoring is a SEPARATE stage (e0c_score.py). Shard with --shard i/n to run N parallel
workers (atomic per-crop writes => no conflicts).

Usage: e0c_run.py [--shard i/n] [--splits 0,1]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl  # noqa: E402

from biotrack import wrapper as W  # noqa: E402

CACHE = ROOT / "artifacts" / "kaggle" / "e0c_cache"
CFG_KEYS = [
    "OUTPUT_ENFORCE_NEXT_FRAME", "OUTPUT_EDGE_MAX_UM", "OUTPUT_MOTION_RELINK",
    "MOTION_RELINK_TIGHT_UM", "MOTION_RELINK_RELAXED_UM", "MOTION_RELINK_VELOCITY_WEIGHT",
    "MOTION_RELINK_LEARNED_BONUS", "MOTION_RELINK_MAX_FRAME_NODES", "OUTPUT_SINGLE_PARENT_REPAIR",
    "OUTPUT_SINGLE_CHILD_REPAIR", "OUTPUT_DIVISION_GEOMETRY_FILTER", "OUTPUT_PRUNE_ISOLATED",
    "OUTPUT_GAP_CLOSE", "GAP_CLOSE_MAX_GAP", "GAP_CLOSE_UM", "OUTPUT_GAP2_RECOVERY",
    "GAP_REFINE_SYNTHETIC", "OUTPUT_FILTER_SHORT_TRACKS", "OUTPUT_MIN_TRACK_LEN",
    "OUTPUT_SAFE_DIVISIONS", "SAFE_DIV_MAX_UM", "OUTPUT_LINEFIT_SMOOTH", "OUTPUT_LINEFIT_WEIGHT",
]


def set_e0c_config():
    W.OUTPUT_MIN_TRACK_LEN = 7
    W.SHORT_TRACK_MIN_LEN_BY_DATASET = {}
    W.GAP_REFINE_SYNTHETIC = True
    W.TEST_DIR = ROOT / "data" / "train"


def config_hash() -> str:
    cfg = {k: getattr(W, k) for k in CFG_KEYS}
    cfg["SHORT_TRACK_MIN_LEN_BY_DATASET"] = dict(W.SHORT_TRACK_MIN_LEN_BY_DATASET)
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def atomic_write_parquet(df: pl.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def atomic_write_json(obj: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def build_nodes_edges(graph):
    nbi = {}
    for r in graph.node_attrs().iter_rows(named=True):
        nid = int(r["node_id"])
        nbi[nid] = {"node_id": nid, "t": int(r["t"]), "z": float(r["z"]), "y": float(r["y"]), "x": float(r["x"])}
    raw = []
    for r in graph.edge_attrs().iter_rows(named=True):
        ep = r.get("edge_prob")
        raw.append({"source_id": int(r["source_id"]), "target_id": int(r["target_id"]),
                    "edge_prob": None if ep is None else float(ep)})
    return nbi, raw


def graph_rows(nbi: dict, edges: list) -> pl.DataFrame:
    ids = sorted(nbi)
    sub = {nid: i + 1 for i, nid in enumerate(ids)}
    rows = [{"row_type": "node", "node_id": sub[n], "t": int(nbi[n]["t"]),
             "z": float(nbi[n]["z"]), "y": float(nbi[n]["y"]), "x": float(nbi[n]["x"]),
             "source_id": -1, "target_id": -1} for n in ids]
    rows += [{"row_type": "edge", "node_id": -1, "t": -1, "z": -1.0, "y": -1.0, "x": -1.0,
              "source_id": sub[int(e["source_id"])], "target_id": sub[int(e["target_id"])]} for e in edges]
    return pl.DataFrame(rows)


def candidate_rows(cands: list, nbi: dict) -> pl.DataFrame:
    from collections import defaultdict
    by_src = defaultdict(list)
    for c in cands:
        by_src[c["source_id"]].append(c)
    rows = []
    for sid, lst in by_src.items():
        for rank, c in enumerate(sorted(lst, key=lambda c: c["cost"])):
            sn, tn = nbi[c["source_id"]], nbi[c["target_id"]]
            rows.append({
                "source_id": c["source_id"], "target_id": c["target_id"], "t": int(sn["t"]),
                "raw_um": c["raw_um"], "motion_um": c["motion_um"], "edge_prob": c["edge_prob"],
                "pass": c["pass"], "gate_um": c["gate_um"], "cost": c["cost"],
                "cost_rank": rank, "n_cand": len(lst), "selected": c["selected"],
                "sz": float(sn["z"]), "sy": float(sn["y"]), "sx": float(sn["x"]),
                "tz": float(tn["z"]), "ty": float(tn["y"]), "tx": float(tn["x"]),
            })
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={"source_id": pl.Int64})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="i/n: process crops where idx%n==i")
    ap.add_argument("--splits", default="0,1")
    a = ap.parse_args()
    i, n = (int(x) for x in a.shard.split("/"))
    splits = [int(s) for s in a.splits.split(",")]

    set_e0c_config()
    chash, commit = config_hash(), git_commit()
    for sub in ("graphs", "candidates", "status"):
        for s in splits:
            (CACHE / sub / str(s)).mkdir(parents=True, exist_ok=True)
    (CACHE / "status").mkdir(parents=True, exist_ok=True)
    print(f"[e0c_run] shard {i}/{n} splits={splits} config_hash={chash} commit={commit} gap_refine={W.GAP_REFINE_SYNTHETIC}")

    done = skipped = failed = 0
    for s in splits:
        crops = sorted((ROOT / f"artifacts/kaggle/oof_clean/pred_geffs_split_{s}").glob("*.geff"))
        for idx, pg in enumerate(crops):
            if idx % n != i:
                continue
            ds = pg.stem
            status_p = CACHE / "status" / f"{s}__{ds}.json"
            if status_p.exists():
                try:
                    st = json.loads(status_p.read_text())
                    if st.get("status") == "ok" and st.get("config_hash") == chash:
                        skipped += 1
                        continue
                except Exception:
                    pass
            try:
                g = W.graph_from_geff(pg)
                nbi, raw = build_nodes_edges(g)
                raw_nodes, raw_edges = len(nbi), len(raw)
                W._CANDIDATE_SINK = []
                fn, fe, stats = W.filter_output_graph(copy.deepcopy(nbi), raw, dataset=ds)
                cands = W._CANDIDATE_SINK
                W._CANDIDATE_SINK = None
                atomic_write_parquet(graph_rows(fn, fe), CACHE / "graphs" / str(s) / f"{ds}.parquet")
                atomic_write_parquet(candidate_rows(cands, nbi), CACHE / "candidates" / str(s) / f"{ds}.parquet")
                atomic_write_json({
                    "crop": ds, "split": s, "config_hash": chash, "git_commit": commit,
                    "raw_nodes": raw_nodes, "raw_edges": raw_edges,
                    "final_nodes": len(fn), "final_edges": len(fe), "n_candidates": len(cands),
                    "diagnostics": {k: int(v) for k, v in stats.items()}, "status": "ok",
                }, status_p)
                done += 1
                print(f"  ok  s{s} {ds}: nodes {raw_nodes}->{len(fn)} edges {len(fe)} cands {len(cands)}", flush=True)
            except Exception as e:
                W._CANDIDATE_SINK = None
                atomic_write_json({"crop": ds, "split": s, "config_hash": chash, "git_commit": commit,
                                   "status": "failed", "error": f"{e}", "trace": traceback.format_exc()[:2000]}, status_p)
                failed += 1
                print(f"  FAIL s{s} {ds}: {e}", flush=True)
    print(f"[e0c_run] shard {i}/{n} done={done} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
