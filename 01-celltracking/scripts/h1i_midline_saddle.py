"""H1-I: per-candidate mid-line saddle depth.

A genuine division at time t leaves TWO separated intensity maxima at t+1 with a
saddle (an intensity dip) between them.  A spurious fork joins two unrelated
cells, or two points inside one un-divided blob; neither shows the same profile.

For every metric-visible candidate we sample the segment d1 -> d2 in the t+1
frame at 9 interior fractions, taking the local ridge value (max over a 3x3x3
voxel block) at each, and record the minimum.  Endpoint brightness comes from
the already-computed per-node peak_inner, so this pass only needs the interior.

This is per-PAIR work, so it is deliberately restricted to the metric-visible
subset (253,507 rows) rather than the full 14.37M census.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

FRACS = np.linspace(0.15, 0.85, 9)
NB = np.array([(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1)
               for dx in (-1, 0, 1)], dtype=np.int64)


def run_crop(crop: str, fold: int, root: Path, out_path: Path) -> dict:
    t0 = time.time()
    cen = pq.read_table(
        root / f"artifacts/kaggle/e0c_cache/fork_candidates/h0c_top3/{fold}/{crop}.parquet",
        columns=["cand_id", "t", "mother", "d1", "d2", "metric_visible", "label"]
    ).to_pandas()
    cen = cen[cen.metric_visible & cen.label.isin(["positive", "reliable_negative"])]
    if len(cen) == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"cand_id": pa.array([], pa.large_string())}), out_path)
        return {"crop": crop, "n": 0, "seconds": 0.0}

    g = pq.read_table(root / f"artifacts/kaggle/e0c_cache/graphs/{fold}/{crop}.parquet",
                      columns=["row_type", "node_id", "z", "y", "x"]).to_pandas()
    g = g[g.row_type == "node"].set_index("node_id")
    coord = g[["z", "y", "x"]].to_numpy()
    row_of = {n: i for i, n in enumerate(g.index.to_numpy())}

    arr = zarr.open_group(str(root / f"data/train/{crop}.zarr"), mode="r")["0"]
    T, Z, Y, X = arr.shape

    i1 = np.array([row_of[n] for n in cen.d1.to_numpy()])
    i2 = np.array([row_of[n] for n in cen.d2.to_numpy()])
    p1, p2 = coord[i1], coord[i2]
    frames = cen.t.to_numpy() + 1

    ridge_min = np.full(len(cen), np.nan, dtype=np.float32)
    ridge_mid = np.full(len(cen), np.nan, dtype=np.float32)
    bg_col = np.full(len(cen), np.nan, dtype=np.float32)

    order = np.argsort(frames, kind="stable")
    fu, starts = np.unique(frames[order], return_index=True)
    bounds = list(starts) + [len(order)]
    for k, f in enumerate(fu):
        if not (0 <= f < T):
            continue
        sel = order[bounds[k]:bounds[k + 1]]
        frame = np.asarray(arr[int(f)])
        sub = frame[::2, ::3, ::3]
        bg = float(np.percentile(sub, 20.0))
        flat = frame.ravel()
        a, b = p1[sel], p2[sel]
        # (n, 9, 3) sample points in voxel index space
        pts = a[:, None, :] + (b - a)[:, None, :] * FRACS[None, :, None]
        vi = np.rint(pts).astype(np.int64)
        nb = vi[:, :, None, :] + NB[None, None, :, :]
        nb[..., 0] = np.clip(nb[..., 0], 0, Z - 1)
        nb[..., 1] = np.clip(nb[..., 1], 0, Y - 1)
        nb[..., 2] = np.clip(nb[..., 2], 0, X - 1)
        idx = nb[..., 0] * (Y * X) + nb[..., 1] * X + nb[..., 2]
        vals = flat[idx].max(axis=2).astype(np.float32)     # (n, 9) ridge values
        vals = np.maximum(vals - np.float32(bg), 0.0)
        ridge_min[sel] = vals.min(axis=1)
        ridge_mid[sel] = vals[:, len(FRACS) // 2]
        bg_col[sel] = bg

    res = pd.DataFrame({
        "cand_id": cen.cand_id.to_numpy(),
        "ridge_min": ridge_min,
        "ridge_mid": ridge_mid,
        "line_bg": bg_col,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(res, preserve_index=False), out_path,
                   compression="zstd")
    return {"crop": crop, "n": int(len(res)), "seconds": round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--root", default=r"c:\Users\aryaa\Documents\Biohub-CellTracking-2026")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    root, out_dir = Path(args.root), Path(args.out_dir)
    specs = [l.strip() for l in Path(args.crops_file).read_text().splitlines() if l.strip()]
    tot = 0.0
    for spec in specs:
        fold_s, crop = spec.split(":", 1)
        fold = int(fold_s)
        dest = out_dir / str(fold) / f"{crop}.parquet"
        if args.skip_existing and dest.exists():
            continue
        st = run_crop(crop, fold, root, dest)
        tot += st["seconds"]
        print(json.dumps(st), flush=True)
    print(json.dumps({"total_seconds": round(tot, 2)}), flush=True)


if __name__ == "__main__":
    main()
