"""Shared-solve CPU replay of arms B, B', C and D from one cached detection graph.

Six independent pipelines would re-solve the same ILPs. Instead, per crop:

    build pre-selection graph
      -> B    greedy selection            + E0c wrapper
      -> B'   solve DEFAULT ILP (0.1/0.1) + E0c wrapper
      -> C    solve C1 ILP (0.0/1.5)      + E0c wrapper
      -> D    REUSE C's solved graph      + faithful v122 wrapper   <- no second C1 solve

Arm A needs neither inference nor a solve: it is the existing parity-proven E0c cache.

Scheduling: the ILP solve dominates cost and peak RSS, so crops above the p90 node count
are run serially first, and only the remainder is parallelised (default 3 workers). Each
crop runs in its own subprocess so an OOM or solver abort cannot destroy completed work.
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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "win_bet"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import coupled_arms as CA  # noqa: E402
from biotrack import wrapper as W  # noqa: E402
from coupled_replay import (  # noqa: E402
    CACHE_ROOT, RAW_OOF, cache_path, graph_from_cached_detections, out_dir,
    peak_rss_mb, solve_ilp,
)
from e0c_run import atomic_write_json, atomic_write_parquet, build_nodes_edges, graph_rows  # noqa: E402

SHARED_ARMS = ["B", "Bp", "C", "D"]
DET = 0.9690
TTA = "4view"

# coupled_replay.cache_path() resolves the cache filename from a MODULE-GLOBAL TTA that
# defaults to "d4". The full-population run is 4-view (the scheme proven to reproduce
# oof_clean exactly), so pin it here. Without this the loader silently reads d4 caches
# while this module globs 4-view ones -- wrong data, no error.
import coupled_replay as _CR  # noqa: E402

_CR.TTA = TTA


def _apply_wrapper(nbi, raw, crop: str, wrapper_name: str):
    CA.WRAPPER_SETTERS[wrapper_name]()
    return W.filter_output_graph(copy.deepcopy(nbi), list(raw), dataset=crop)


def _write(arm: str, split: int, crop: str, fn, fe) -> None:
    dest = out_dir(arm) / str(split)
    dest.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(pl.DataFrame(graph_rows(fn, fe)), dest / f"{crop}.parquet")


def _record(arm, split, crop, nbi, raw, fn, fe, stats, t0, cand_hash, solver) -> dict:
    return {"crop": crop, "split": split, "arm": arm, "status": "ok",
            "config_hash": CA.config_hash(arm), "candidate_hash": cand_hash,
            "tta": "4view", "det_threshold": DET, "solver": solver,
            "ilp": CA.ARMS[arm]["ilp"], "wrapper": CA.ARMS[arm]["wrapper"],
            "nodes_pre_wrapper": len(nbi), "nodes_post_wrapper": len(fn),
            "edges_pre_wrapper": len(raw), "edges_post_wrapper": len(fe),
            "runtime_s": round(time.time() - t0, 2), "peak_rss_mb": peak_rss_mb(),
            "wrapper_stats": {k: v for k, v in stats.items() if isinstance(v, (int, float))}}


def replay_shared(split: int, crop: str) -> dict:
    """All four cache-derived arms for one crop, solving each ILP at most once."""
    src = cache_path(crop, DET)
    assert f"tta-{TTA}" in src.name, f"TTA mismatch: wanted {TTA}, got {src.name}"
    if not src.exists():
        return {"crop": crop, "status": "missing_cache", "detail": str(src)}
    cand_hash = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    out: dict[str, dict] = {}

    # ---- B : greedy (no ILP) ------------------------------------------------
    t0 = time.time()
    g = graph_from_cached_detections(src)
    nbi, raw = build_nodes_edges(g)
    fn, fe, st = _apply_wrapper(nbi, raw, crop, "e0c")
    _write("B", split, crop, fn, fe)
    out["B"] = _record("B", split, crop, nbi, raw, fn, fe, st, t0, cand_hash, "greedy")

    # ---- B' : default ILP 0.1/0.1 -------------------------------------------
    t0 = time.time()
    g = graph_from_cached_detections(src)
    if g.num_edges() > 0:
        g = solve_ilp(g, CA.ILP_DEFAULT)
    nbi, raw = build_nodes_edges(g)
    fn, fe, st = _apply_wrapper(nbi, raw, crop, "e0c")
    _write("Bp", split, crop, fn, fe)
    out["Bp"] = _record("Bp", split, crop, nbi, raw, fn, fe, st, t0, cand_hash, "ilp_default")

    # ---- C : C1 ILP 0.0/1.5 -- solved ONCE, reused by D ----------------------
    t0 = time.time()
    g = graph_from_cached_detections(src)
    if g.num_edges() > 0:
        g = solve_ilp(g, CA.ILP_C1)
    nbi_c, raw_c = build_nodes_edges(g)
    solve_s = round(time.time() - t0, 2)
    fn, fe, st = _apply_wrapper(nbi_c, raw_c, crop, "e0c")
    _write("C", split, crop, fn, fe)
    out["C"] = _record("C", split, crop, nbi_c, raw_c, fn, fe, st, t0, cand_hash, "ilp_c1")

    # ---- D : REUSE C's solved graph, swap only the wrapper -------------------
    t0 = time.time()
    fn, fe, st = _apply_wrapper(nbi_c, raw_c, crop, "v122")
    _write("D", split, crop, fn, fe)
    out["D"] = _record("D", split, crop, nbi_c, raw_c, fn, fe, st, t0, cand_hash,
                       "ilp_c1(reused)")
    out["D"]["c1_solve_shared_s"] = solve_s
    return out


def _worker(args) -> dict:
    """Subprocess entry: isolate one crop so an OOM cannot kill the batch."""
    split, crop = args
    cmd = [sys.executable, str(Path(__file__).resolve()), "--one", f"{split}:{crop}"]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    sp = CACHE_ROOT / "shared_status" / f"{split}__{crop}.json"
    if sp.exists():
        return json.loads(sp.read_text())
    return {"crop": crop, "split": split, "status": "subprocess_died",
            "returncode": proc.returncode, "stderr": proc.stderr[-1500:]}


def run_one(split: int, crop: str) -> dict:
    sp = CACHE_ROOT / "shared_status" / f"{split}__{crop}.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    try:
        rec = {"crop": crop, "split": split, "status": "ok", "arms": replay_shared(split, crop)}
        if any(a.get("status") != "ok" for a in rec["arms"].values()):
            rec["status"] = "partial"
    except Exception as exc:  # noqa: BLE001
        rec = {"crop": crop, "split": split, "status": "failed",
               "error": f"{type(exc).__name__}: {exc}",
               "traceback": traceback.format_exc()[-2500:]}
    atomic_write_json(rec, sp)
    return rec


def available_crops() -> list[tuple[int, str, int]]:
    """Cached crops with their node counts, for cost-aware scheduling."""
    out = []
    for p in sorted((CACHE_ROOT / "det").glob(f"*__tta-4view__det-{DET:g}.npz")):
        crop = p.name.split("__")[0]
        split = 0 if crop.startswith("44b6") else 1
        try:
            n = int(np.load(p)["coords"].shape[0])
        except Exception:  # noqa: BLE001
            continue
        out.append((split, crop, n))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--one", default="", help="internal: '<split>:<crop>'")
    a = ap.parse_args()

    if a.one:
        split, crop = a.one.split(":", 1)
        r = run_one(int(split), crop)
        print(f"{crop}: {r['status']}")
        return

    crops = available_crops()
    done = {p.stem for p in (CACHE_ROOT / "shared_status").glob("*.json")
            if json.loads(p.read_text()).get("status") == "ok"} \
        if (CACHE_ROOT / "shared_status").exists() else set()
    todo = [(s, c, n) for s, c, n in crops if f"{s}__{c}" not in done]
    if not todo:
        print("[shared] nothing to do (all cached crops already replayed)")
        return

    ns = np.array([n for _, _, n in todo])
    p90 = float(np.percentile(ns, 90)) if len(ns) > 4 else float("inf")
    heavy = [(s, c) for s, c, n in todo if n >= p90]
    light = [(s, c) for s, c, n in todo if n < p90]
    print(f"[shared] {len(todo)} crops to replay | p90={p90:.0f} nodes | "
          f"{len(heavy)} heavy (serial) + {len(light)} light ({a.workers} workers)")

    ok = bad = 0
    # Heavy crops first, strictly serial: never let several max-size ILPs coincide.
    for s, c in heavy:
        r = _worker((s, c))
        ok, bad = (ok + 1, bad) if r.get("status") == "ok" else (ok, bad + 1)
        print(f"  [heavy] s{s} {c}: {r.get('status')}", flush=True)
    if light:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(_worker, light):
                ok, bad = (ok + 1, bad) if r.get("status") == "ok" else (ok, bad + 1)
                print(f"  s{r.get('split')} {r.get('crop')}: {r.get('status')}", flush=True)

    print(f"[shared] done ok={ok} failed={bad}")
    if bad:
        print("  PARTIAL COVERAGE -- do NOT score until resolved.")


if __name__ == "__main__":
    main()
