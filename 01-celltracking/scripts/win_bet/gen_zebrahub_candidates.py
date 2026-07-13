"""Generate Zebrahub geometry candidates in the COMMON schema for Phase-B breadth.

For each dense Zebrahub embryo, per frame, builds kNN candidate successor edges within a
relative gate and emits [source_id, target_id, raw_um, motion_um, label] — the same
columns the competition candidate table exposes, so the SAME scale-free geometry features
(raw_rel, motion_rel, distance-rank, density) apply identically to both domains. Only
answerable source groups (source whose true successor is in the pool) are kept, matching
the competition ranking setup. Coordinates are used as-is; every downstream feature is
per-source-relative, so the embryo's absolute unit/anisotropy cancels.

Output: artifacts/kaggle/e0c_cache/zebrahub_candidates/{embryo}.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "external" / "zebrahub"
OUT = ROOT / "artifacts" / "kaggle" / "e0c_cache" / "zebrahub_candidates"
EMBRYOS = ["ZSNS003_tracks.csv", "ZSNS004_tracks.csv", "ZSNS005_tracks.csv"]
K_CAND = 6          # candidate successors per source (matches wrapper relaxed pass scale)
GATE_MULT = 3.0     # relative gate = GATE_MULT * median local step
T_WINDOW = 30       # timepoints per embryo (dense -> plenty of groups)
EPS = 1e-6


def gen(path: Path, t_lo: int) -> pl.DataFrame:
    df = pl.read_csv(path).filter((pl.col("t") >= t_lo) & (pl.col("t") < t_lo + T_WINDOW))
    df = df.with_row_index("nid")
    # predecessor position per (track,t) for velocity; and true successor (child) map
    key = {(int(r["track_id"]), int(r["t"])): int(r["nid"]) for r in df.iter_rows(named=True)}
    pred_pos: dict[int, np.ndarray] = {}
    true_succ: dict[int, set[int]] = {}
    pos = {int(r["nid"]): np.array([r["x"], r["y"], r["z"]], float) for r in df.iter_rows(named=True)}
    for r in df.iter_rows(named=True):
        tid, t, ptid, nid = int(r["track_id"]), int(r["t"]), int(r["parent_track_id"]), int(r["nid"])
        p = key.get((tid, t - 1))
        if p is None and ptid != -1:
            p = key.get((ptid, t - 1))
        if p is not None:
            pred_pos[nid] = pos[p]
            true_succ.setdefault(p, set()).add(nid)

    times = sorted(df["t"].unique().to_list())
    by_t = {t: df.filter(pl.col("t") == t) for t in times}
    rows = []
    for t in times[:-1]:
        src, dst = by_t[t], by_t.get(t + 1)
        if dst is None or dst.height == 0 or src.height == 0:
            continue
        s_ids = src["nid"].to_numpy()
        s_xyz = src.select("x", "y", "z").to_numpy().astype(float)
        d_ids = dst["nid"].to_numpy()
        d_xyz = dst.select("x", "y", "z").to_numpy().astype(float)
        tree = cKDTree(d_xyz)
        kc = min(K_CAND, len(d_ids))
        dist, idx = tree.query(s_xyz, k=kc)
        dist = np.atleast_2d(dist); idx = np.atleast_2d(idx)
        gate = float(np.median(dist[:, 0])) * GATE_MULT + EPS
        for i, sid in enumerate(s_ids):
            sid = int(sid)
            succ = true_succ.get(sid, set())
            pp = pred_pos.get(sid)
            predicted = s_xyz[i] if pp is None else s_xyz[i] + 0.5 * (s_xyz[i] - pp)
            for r in range(kc):
                j = int(idx[i, r]); tid = int(d_ids[j])
                raw = float(dist[i, r])
                if raw > gate:
                    continue
                motion = float(np.linalg.norm(d_xyz[j] - predicted))
                rows.append((sid, tid, raw, motion, 1 if tid in succ else 0))
    if not rows:
        return pl.DataFrame(schema={"source_id": pl.Int64})
    out = pl.DataFrame(rows, schema=["source_id", "target_id", "raw_um", "motion_um", "label"], orient="row")
    # keep only answerable groups (source has its true successor in the pool)
    ans = out.filter(pl.col("label") == 1)["source_id"].unique()
    return out.filter(pl.col("source_id").is_in(ans))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for e in EMBRYOS:
        p = DATA / e
        if not p.exists():
            print(f"skip {e} (missing)"); continue
        # use a mid-embryo window (developmental variety without the sparse early frames)
        df = gen(p, t_lo=200)
        name = e.replace("_tracks.csv", "")
        df.write_parquet(OUT / f"{name}.parquet")
        pos = int(df["label"].sum()) if df.height else 0
        grp = df["source_id"].n_unique() if df.height else 0
        print(f"{name}: {df.height:,} cands, {grp:,} answerable groups, {pos:,} positive")


if __name__ == "__main__":
    main()
