#!/usr/bin/env python3
"""EXP-13 -- re-derive division geometry on Zebrahub, 105x more divisions than Kaggle.

Every division decision so far rests on Kaggle's 151 divisions (EXP-1's discriminators, EXP-7's
verdict that the cosine gate is pure loss). Zebrahub ZSNS003 alone has 15,868, is the same
organism tracked with the same tool (Ultrack), and is host-approved as external data.

This asks the one question that most needs a bigger sample: is the cosine gate genuinely useless,
or was 151 divisions too small to see it work?

Zebrahub coordinates are VOXELS; scale is z 1.24, y/x 0.439 um (from the ome-zarr .zattrs).
"""
import sys
from pathlib import Path
import numpy as np, polars as pl

ZSCALE = np.array([1.24, 0.439, 0.439])
SRC = Path("data/zebrahub")


def analyse(path):
    df = pl.read_csv(path)
    # child -> parent map, and the set of true divisions (a parent with 2 children)
    kids = (df.filter(pl.col("parent_track_id") != -1)
              .group_by("parent_track_id").agg(pl.col("track_id").unique().alias("ch")))
    kids = kids.filter(pl.col("ch").list.len() == 2)
    # first appearance of each track
    first = (df.sort("t").group_by("track_id")
               .agg(pl.col("t").first(), pl.col("z").first(), pl.col("y").first(),
                    pl.col("x").first()))
    fmap = {r["track_id"]: (r["t"], np.array([r["z"], r["y"], r["x"]]) * ZSCALE)
            for r in first.iter_rows(named=True)}
    # last position of each parent track
    last = (df.sort("t").group_by("track_id")
              .agg(pl.col("t").last(), pl.col("z").last(), pl.col("y").last(),
                   pl.col("x").last()))
    lmap = {r["track_id"]: (r["t"], np.array([r["z"], r["y"], r["x"]]) * ZSCALE)
            for r in last.iter_rows(named=True)}
    # per-track position at a given t, for the divergence measurement
    pos = {}
    for r in df.iter_rows(named=True):
        pos[(r["track_id"], r["t"])] = np.array([r["z"], r["y"], r["x"]]) * ZSCALE

    cos, sis, par, diverge = [], [], [], []
    for r in kids.iter_rows(named=True):
        p = r["parent_track_id"]; a, b = r["ch"]
        if p not in lmap or a not in fmap or b not in fmap:
            continue
        tp, P = lmap[p]; ta, A = fmap[a]; tb, B = fmap[b]
        if ta != tb or ta != tp + 1:
            continue                                   # not a clean one-frame division
        va, vb = A - P, B - P
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na < 1e-9 or nb < 1e-9:
            continue
        cos.append(float(va @ vb / (na * nb)))
        sis.append(float(np.linalg.norm(A - B)))
        par.append(float(max(na, nb)))
        n1, n2 = pos.get((a, ta + 1)), pos.get((b, tb + 1))
        if n1 is not None and n2 is not None:
            diverge.append(float(np.linalg.norm(n1 - n2) - np.linalg.norm(A - B)))
    return map(np.array, (cos, sis, par, diverge))


def pct(name, v, ps=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    q = np.percentile(v, ps)
    print(f"  {name:<26} n={len(v):>6}  " + "  ".join(f"p{p}={x:>7.3f}" for p, x in zip(ps, q)))


for f in sorted(SRC.glob("*_tracks.csv")):
    print(f"\n=== {f.stem} ===", flush=True)
    cos, sis, par, div = analyse(f)
    if not len(cos):
        print("  no clean divisions"); continue
    pct("cos(parent->daughters)", cos)
    pct("sister separation um", sis)
    pct("max parent arc um", par)
    if len(div): pct("divergence t+1 um", div)
    print(f"\n  KAGGLE EXP-1 measured (n=151): cos median -0.746, IQR -0.914..-0.486;"
          f" sisters 10.570um -> 13.656um by t+2")
    print(f"  ZEBRAHUB here:                 cos median {np.median(cos):+.3f},"
          f" IQR {np.percentile(cos,25):+.3f}..{np.percentile(cos,75):+.3f}")
    print(f"\n  fraction of TRUE divisions our gates would REJECT:")
    for c in (1.0, 0.0, -0.30, -0.50, -0.70):
        print(f"    fork_cos_max={c:>5}: rejects {100*(cos > c).mean():>5.1f}% of real divisions")
    for d in (0.0, 0.5, 1.0, 2.0):
        if len(div):
            print(f"    divergence>={d:>4}: rejects {100*(div < d).mean():>5.1f}% of real divisions")
    print(f"    fork_parent_um=12:  rejects {100*(par > 12).mean():>5.1f}%")
    print(f"    fork_sister_um=18:  rejects {100*(sis > 18).mean():>5.1f}%")
