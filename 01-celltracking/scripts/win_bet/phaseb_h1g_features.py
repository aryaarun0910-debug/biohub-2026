"""H1-G — geometry-only feature extraction over the frozen H0c candidate census.

NO model inference. Every feature is derivable from the E0c graph geometry/topology plus the
frozen census table, so this arm can be measured before any forward/reverse extractor exists.
It establishes the baseline that the F (forward-ambiguity) and R (reverse-consistency) arms
must beat -- and if G alone promotes, association consistency is NOT what is doing the work.

The candidate surface is IMMUTABLE (cfg hash 04eeac97500d): 15/15 generation, kNN flow,
flow-midpoint top-3 per mother. Nothing is retuned.

Features per candidate (mother m, daughters d1/d2 at t+1):
  flow_midpoint_residual   frozen ranking feature (from the census)
  rank                     0/1/2 within the mother's choice set
  parent_midpoint_um       |midpoint - mother|
  pd1_um, pd2_um           mother->daughter distances
  pd_min/pd_max/pd_ratio   symmetry of the two arms
  sister_um                daughter separation (continuous evidence, never a veto)
  cos_daughter_axis        alignment of (d2-d1) with the mother's recent velocity
  cos_split_vs_flow        alignment of (midpoint-mother) with local flow
  daughter_angle           angle subtended at the mother by the two daughters
  persist_d1/d2            daughter survives into t+2 (has an outgoing edge)
  n_persist                0/1/2
  mother_track_age         frames the mother's chain extends backwards
  mother_speed_um          |mother - its parent|
  vel_consistency          |mother displacement - local flow| (motion anomaly)
  local_density_t/t1       neighbours within 15um at t and t+1
  competing_parents        daughters already having a different parent (0/1/2)
  steal_required           any competing parent present
  best_alt_gap             residual gap to the mother's next-best pair (choice-set margin)
  resid_ratio              residual / best residual in the choice set
  bdist_um                 distance to the crop bounding box (boundary proximity)

Usage:
  .venv\\Scripts\\python.exe scripts\\win_bet\\phaseb_h1g_features.py --workers 6
"""
from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

from phaseb_d0p_proposer import CACHE, SCALE, cached_crops, load_e0c_tables, _continuations  # noqa: E402
from phaseb_h0c_replay import CFG_HASH  # noqa: E402

CENSUS = CACHE / "fork_candidates" / "h0c_top3"
FEATS = CACHE / "fork_candidates" / "h1g_features"
EPS = 1e-9


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > EPS else np.zeros(3)


def features_one(args) -> dict:
    split, crop = args
    from scipy.spatial import cKDTree
    cen = pl.read_parquet(CENSUS / str(split) / f"{crop}.parquet")
    if cen.height == 0:
        return {"split": split, "crop": crop, "rows": 0}
    sub, t, pos, edges, _nd = load_e0c_tables(split, crop)
    idx = {int(s): i for i, s in enumerate(sub)}

    parent_of, children_of = {}, {}
    for a, b in edges:
        parent_of[int(b)] = int(a)
        children_of.setdefault(int(a), set()).add(int(b))

    by_t: dict[int, list[int]] = {}
    for i, a in enumerate(t):
        by_t.setdefault(int(a), []).append(i)
    trees = {a: cKDTree(pos[ii]) for a, ii in by_t.items()}
    lo, hi = pos.min(axis=0), pos.max(axis=0)

    cont = _continuations(edges, idx, t, pos)
    alld = [d for f in cont.values() for d in f[1]]
    gmed = np.median(np.stack(alld), axis=0) if alld else np.zeros(3)
    fmed, ktree, darr = {}, {}, {}
    for a, (s_, d_) in cont.items():
        darr[a] = np.stack(d_) if d_ else None
        fmed[a] = np.median(darr[a], axis=0) if d_ else gmed
        ktree[a] = cKDTree(pos[s_]) if len(s_) >= 4 else None
        trees.setdefault(a, None)

    def flow_at(i, a):
        kt = ktree.get(a)
        if kt is not None:
            _, jj = kt.query(pos[i], k=min(16, darr[a].shape[0]))
            return np.median(darr[a][np.atleast_1d(jj)], axis=0)
        return fmed.get(a, gmed)

    # track age: walk the parent chain backwards (bounded)
    def track_age(m):
        n, cur, seen = 0, m, set()
        while cur in parent_of and cur not in seen and n < 64:
            seen.add(cur)
            cur = parent_of[cur]
            n += 1
        return n

    rows = []
    for r in cen.iter_rows(named=True):
        m, d1, d2 = int(r["mother"]), int(r["d1"]), int(r["d2"])
        im, i1, i2 = idx.get(m), idx.get(d1), idx.get(d2)
        if im is None or i1 is None or i2 is None:
            continue
        a = int(t[im])
        pm, p1, p2 = pos[im], pos[i1], pos[i2]
        mid = 0.5 * (p1 + p2)
        fl = flow_at(im, a)
        pd1 = float(np.linalg.norm(p1 - pm))
        pd2 = float(np.linalg.norm(p2 - pm))
        sis = float(np.linalg.norm(p2 - p1))
        par = parent_of.get(m)
        mv = pm - pos[idx[par]] if par is not None and par in idx else np.zeros(3)
        u1, u2 = _unit(p1 - pm), _unit(p2 - pm)
        rows.append({
            "cand_id": r["cand_id"], "crop": crop, "fold": split, "family": r["family"],
            "mother": m, "d1": d1, "d2": d2, "rank": int(r["rank"]),
            "label": r["label"], "metric_visible": bool(r["metric_visible"]),
            "flow_midpoint_residual": float(r["flow_midpoint_residual"]),
            "parent_midpoint_um": float(np.linalg.norm(mid - pm)),
            "pd1_um": min(pd1, pd2), "pd2_um": max(pd1, pd2),
            "pd_ratio": min(pd1, pd2) / (max(pd1, pd2) + EPS),
            "sister_um": sis,
            "cos_daughter_axis": float(np.dot(_unit(p2 - p1), _unit(mv))),
            "cos_split_vs_flow": float(np.dot(_unit(mid - pm), _unit(fl))),
            "daughter_angle": float(np.dot(u1, u2)),
            "persist_d1": int(d1 in children_of), "persist_d2": int(d2 in children_of),
            "n_persist": int(d1 in children_of) + int(d2 in children_of),
            "mother_track_age": track_age(m),
            "mother_speed_um": float(np.linalg.norm(mv)),
            "vel_consistency": float(np.linalg.norm(mv - fl)) if par is not None else -1.0,
            "local_density_t": len(trees[a].query_ball_point(pm, 15.0)) if trees.get(a) is not None else 0,
            "local_density_t1": len(trees[a + 1].query_ball_point(mid, 15.0)) if trees.get(a + 1) is not None else 0,
            "competing_parents": int(parent_of.get(d1, m) != m) + int(parent_of.get(d2, m) != m),
            "steal_required": int(bool(r["steal_required"])),
            "bdist_um": float(min(np.min(pm - lo), np.min(hi - pm))),
        })
    df = pl.DataFrame(rows)
    # choice-set margins (within mother)
    df = df.with_columns([
        pl.col("flow_midpoint_residual").min().over("mother").alias("_best"),
        pl.col("flow_midpoint_residual").sort().over("mother").shift(-1).alias("_next"),
    ])
    df = df.with_columns([
        (pl.col("flow_midpoint_residual") - pl.col("_best")).alias("resid_gap_to_best"),
        (pl.col("flow_midpoint_residual") / (pl.col("_best") + EPS)).alias("resid_ratio"),
        (pl.col("_next") - pl.col("flow_midpoint_residual")).fill_null(-1.0).alias("best_alt_gap"),
    ]).drop("_best", "_next")

    (FEATS / str(split)).mkdir(parents=True, exist_ok=True)
    df.write_parquet(FEATS / str(split) / f"{crop}.parquet")
    return {"split": split, "crop": crop, "rows": df.height,
            "positives": int((df["label"] == "positive").sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    print(f"H1-G geometry features over frozen census (cfg {CFG_HASH})")
    tot = pos_tot = 0
    for fold, fam in ((0, "44b6"), (1, "6bba")):
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(features_one, [(fold, c) for c in cached_crops(fold)]))
        r_ = sum(x["rows"] for x in res)
        p_ = sum(x.get("positives", 0) for x in res)
        tot += r_; pos_tot += p_
        print(f"  {fam}: rows={r_:,} positives={p_} crops={len(res)}")
    print(f"  TOTAL rows={tot:,} positives={pos_tot} -> {FEATS}")


if __name__ == "__main__":
    main()
