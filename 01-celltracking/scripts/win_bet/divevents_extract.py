"""Local data-prep for GPU Job A — extract scale-free division fork-events from Zebrahub.

Compute split: this runs LOCAL (CPU/Polars). It emits a compact training tensor the T4
notebook trains on, so no GPU hours are spent on data processing.

Each event = (mother trajectory, daughter-1 trajectory, daughter-2 trajectory) over a
short window, expressed in a SCALE-FREE mother-centric frame (translate to mother's split
position, normalize by mother's recent step length). No absolute coordinates, no embryo id.
  positive  : a real division (parent -> its two real daughter tracks)
  negative  : a real non-dividing continuation + a nearby NON-daughter as the fake 2nd
              daughter (genuine continuation, per commander spec)

Output: artifacts/kaggle/divevents/{embryo}.npz  with X (N, 3, W, 3), y (N,), embryo tag.
Also a manifest with per-embryo counts for LOEO.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "external" / "zebrahub"
OUT = ROOT / "artifacts" / "kaggle" / "divevents"
EMBRYOS = ["ZSNS001", "ZSNS003", "ZSNS004", "ZSNS005"]
W = 5          # frames each side of the split
T_WINDOW = 120  # timepoints scanned per embryo
EPS = 1e-6


def load(embryo: str, t_lo: int) -> pl.DataFrame:
    return pl.read_csv(DATA / f"{embryo}_tracks.csv").filter(
        (pl.col("t") >= t_lo) & (pl.col("t") < t_lo + T_WINDOW))


def track_pos(df: pl.DataFrame) -> dict[int, dict[int, np.ndarray]]:
    d: dict[int, dict[int, np.ndarray]] = {}
    for r in df.iter_rows(named=True):
        d.setdefault(int(r["track_id"]), {})[int(r["t"])] = np.array([r["x"], r["y"], r["z"]], float)
    return d


def seq(pos: dict[int, np.ndarray], t0: int, step: int, n: int) -> np.ndarray | None:
    """n positions at t0, t0+step, ... ; None if any missing."""
    out = []
    for k in range(n):
        p = pos.get(t0 + k * step)
        if p is None:
            return None
        out.append(p)
    return np.stack(out)


def scale_free(mother: np.ndarray, d1: np.ndarray, d2: np.ndarray):
    """Mother-centric, step-normalized event tensor (3, W, 3)."""
    origin = mother[-1]                                  # split position
    steps = np.linalg.norm(np.diff(mother, axis=0), axis=1)
    scale = float(np.median(steps)) + EPS if len(steps) else 1.0
    def norm(a):
        return (a - origin) / scale
    return np.stack([norm(mother), norm(d1), norm(d2)]).astype(np.float32)  # (3, W, 3)


def build(embryo: str) -> tuple[np.ndarray, np.ndarray]:
    df = load(embryo, 200)
    tp = track_pos(df)
    firsts = df.sort("t").group_by("track_id").first().select("track_id", "t", "parent_track_id")
    children_of: dict[int, list[int]] = {}
    birth_t: dict[int, int] = {}
    for r in firsts.iter_rows(named=True):
        birth_t[int(r["track_id"])] = int(r["t"])
        if int(r["parent_track_id"]) != -1:
            children_of.setdefault(int(r["parent_track_id"]), []).append(int(r["track_id"]))

    # per-timepoint KDTree for sampling fake second daughters
    by_t = {t: df.filter(pl.col("t") == t) for t in df["t"].unique().to_list()}
    trees = {}
    for t, sub in by_t.items():
        xyz = sub.select("x", "y", "z").to_numpy().astype(float)
        trees[t] = (cKDTree(xyz), sub["track_id"].to_numpy())

    def fake_daughter(t_next: int, at: np.ndarray, exclude: set[int]) -> np.ndarray | None:
        if t_next not in trees:
            return None
        tree, tids = trees[t_next]
        _, idx = tree.query(at, k=min(8, len(tids)))
        for j in np.atleast_1d(idx):
            c = int(tids[j])
            if c not in exclude and c in tp:
                s = seq(tp[c], t_next, 1, W)
                if s is not None:
                    return s
        return None

    X, y = [], []
    # POSITIVES: real divisions (parent -> two real daughters), split at mother's last frame
    for parent, kids in children_of.items():
        if parent not in tp or len(kids) < 2:
            continue
        tau = max(tp[parent])
        m = seq(tp[parent], tau - (W - 1), 1, W)
        da = seq(tp[kids[0]], tau + 1, 1, W) if m is not None else None
        db = seq(tp[kids[1]], tau + 1, 1, W) if m is not None else None
        if m is not None and da is not None and db is not None:
            X.append(scale_free(m, da, db)); y.append(1)
    n_pos = len(y)

    # NEGATIVES: genuine non-dividing continuations (self-continuation + nearby fake 2nd),
    # sampled to ~balance the positives.
    rng = np.random.default_rng(0)
    tids_all = list(tp.keys()); rng.shuffle(tids_all)
    n_neg = 0
    for tid in tids_all:
        if n_neg >= n_pos:
            break
        pos = tp[tid]
        taus = [t for t in pos if all((t - k) in pos for k in range(W))
                and all((t + 1 + k) in pos for k in range(W))]
        if not taus:
            continue
        tau = taus[len(taus) // 2]
        m = seq(pos, tau - (W - 1), 1, W)
        da = seq(pos, tau + 1, 1, W)                       # real self continuation
        if m is None or da is None:
            continue
        fake = fake_daughter(tau + 1, pos[tau], {tid, *children_of.get(tid, [])})
        if fake is not None:
            X.append(scale_free(m, da, fake)); y.append(0); n_neg += 1
    if not X:
        return np.zeros((0, 3, W, 3), np.float32), np.zeros((0,), np.int8)
    return np.stack(X), np.asarray(y, np.int8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for e in EMBRYOS:
        if not (DATA / f"{e}_tracks.csv").exists():
            print(f"skip {e}"); continue
        X, y = build(e)
        np.savez_compressed(OUT / f"{e}.npz", X=X, y=y)
        manifest[e] = {"events": int(len(y)), "pos": int(y.sum()), "neg": int((y == 0).sum())}
        print(f"{e}: {len(y)} events ({int(y.sum())} pos / {int((y==0).sum())} neg), X {X.shape}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
