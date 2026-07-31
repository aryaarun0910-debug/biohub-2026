"""H1-M2: full TEMPORAL PROFILE extraction around every Zebrahub fork.

Lane C extracted appearance at a single anchor (the tracker's fork frame t0) with a
fixed one-frame forward step, and found no mitotic signature.  This script removes both
of those assumptions in one download: for every event it gathers the SAME physical 6 um
sphere, centred on the mother's position at t0, at EVERY frame in [t0+dt_lo, t0+dt_hi].

Once that profile exists, every (anchor, stride) hypothesis -- including the
track-geometry realignment produced by `h1m2_realign.py` -- can be evaluated offline at
zero further cost.  That is the point: the imagery is fetched once, the label-time
question is then answered arithmetically.

Position semantics are identical to the competition side (`h1i_node_appearance`): the
sampling point is FIXED at the mother's own frame and later frames are read at that same
point, so `massn` measures how much mass has LEFT the mother's last position.

Negatives are tracks with out-degree exactly 1 at t (the annotation says they continue),
sampled from the same frames as the positives, which is the same four-way convention
`phaseb_h1a_census.py` and `h1m_zebrahub_appearance.py` use.

Data source and licence: Zebrahub, CZ Biohub SF (Royer lab), CC BY 4.0 --
Lange, Granados, VijayKumar et al., Cell (2024), doi:10.1016/j.cell.2024.09.047.

Usage:
  .venv\\Scripts\\python.exe scripts\\h1m2_profile.py --embryo ZSNS005 \
      --forks realigned_forks.parquet --tracks <prep>/ZSNS005.parquet \
      --frame-lo 300 --frame-hi 329 --out prof_ZSNS005.parquet
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h1i_node_appearance as h1i  # noqa: E402
import h1m_features as F  # noqa: E402

BASE = "https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/"
LEVEL = 0
BBOX_MARGIN_UM = 20.0
DERIVED = F.NODE_FEATS


def zarray(embryo: str) -> dict:
    return json.load(urllib.request.urlopen(
        f"{BASE}{embryo}.ome.zarr/{LEVEL}/.zarray", timeout=120))


def fetch_chunk(embryo: str, t: int, zc: int, xc: int) -> bytes:
    return urllib.request.urlopen(
        f"{BASE}{embryo}.ome.zarr/{LEVEL}/{t}/0/{zc}/0/{xc}", timeout=900).read()


def decode(raw: bytes, shape) -> np.ndarray:
    from numcodecs import Blosc
    return np.frombuffer(Blosc().decode(raw), dtype="<u2").reshape(shape)


def build_events(emb: str, forks: pd.DataFrame, tr: pd.DataFrame,
                 lo: int, hi: int, dt_lo: int, dt_hi: int,
                 n_neg: int, seed: int) -> pd.DataFrame:
    """Positives = forks whose whole profile fits inside [lo, hi].
    Negatives = out-degree-1 mothers drawn from the same frames."""
    t_lo, t_hi = lo - dt_lo, hi - dt_hi
    fk = forks[(forks.embryo == emb) & (forks.t_orig >= t_lo)
               & (forks.t_orig <= t_hi)].copy()
    fk["label"] = 1
    fk = fk.rename(columns={"t_orig": "t"})

    # ---- negatives: track continues to t+1 and is not a fork parent at t --------
    fork_par = set(forks.loc[forks.embryo == emb, "mother_track_id"].tolist())
    sub = tr[(tr.t >= t_lo) & (tr.t <= t_hi)]
    nxt = tr[(tr.t >= t_lo + 1) & (tr.t <= t_hi + 1)][["track_id", "t"]]
    have_next = set(zip(nxt.track_id.to_numpy(), (nxt.t - 1).to_numpy()))
    cand = sub[~sub.track_id.isin(fork_par)]
    keys = list(zip(cand.track_id.to_numpy(), cand.t.to_numpy()))
    ok = np.fromiter((k in have_next for k in keys), bool, len(keys))
    cand = cand[ok]
    rng = np.random.default_rng(seed)
    take = rng.choice(len(cand), size=min(n_neg, len(cand)), replace=False)
    neg = cand.iloc[np.sort(take)].copy()
    neg = neg.rename(columns={"track_id": "mother_track_id"})
    neg["label"] = 0
    neg["z_vox"], neg["y_vox"], neg["x_vox"] = neg.z, neg.y, neg.x
    neg["stride"] = -1
    neg["realigned"] = False
    for c in ("sep_p1_um", "sep_at_stride_um", "max_sep_um", "drift_at_stride_um"):
        neg[c] = np.nan
    neg["embryo"] = emb
    keep = ["embryo", "mother_track_id", "t", "z_vox", "y_vox", "x_vox", "label",
            "stride", "realigned", "sep_p1_um", "sep_at_stride_um", "max_sep_um",
            "drift_at_stride_um"]
    ev = pd.concat([fk[keep], neg[keep]], ignore_index=True)
    return ev.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embryo", required=True)
    ap.add_argument("--forks", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--frame-lo", type=int, required=True)
    ap.add_argument("--frame-hi", type=int, required=True)
    ap.add_argument("--dt-lo", type=int, default=-4)
    ap.add_argument("--dt-hi", type=int, default=10)
    ap.add_argument("--z-slabs", default="0")
    ap.add_argument("--x-chunk", type=int, default=0)
    ap.add_argument("--n-neg", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from numcodecs import blosc
    blosc.set_nthreads(4)
    z_slabs = tuple(int(v) for v in a.z_slabs.split(","))

    kern = F.install_kernel(F.VOX_ZEBRAHUB_L0)
    HZ, HY, HX = kern["HZ"], kern["HY"], kern["HX"]

    forks = pd.read_parquet(a.forks)
    tr = pd.read_parquet(a.tracks, columns=["track_id", "t", "z", "y", "x"])
    ev = build_events(a.embryo, forks, tr, a.frame_lo, a.frame_hi,
                      a.dt_lo, a.dt_hi, a.n_neg, a.seed)

    meta = zarray(a.embryo)
    cz, cy, cx = meta["chunks"][2], meta["chunks"][3], meta["chunks"][4]
    Y = min(meta["shape"][3], cy)
    X = min(meta["shape"][4] - a.x_chunk * cx, cx)
    ZMAX = cz * len(z_slabs)
    ZOFF = z_slabs[0] * cz

    iz = np.rint(ev.z_vox.to_numpy()).astype(np.int64) - ZOFF
    iy = np.rint(ev.y_vox.to_numpy()).astype(np.int64)
    ix = np.rint(ev.x_vox.to_numpy()).astype(np.int64)
    inside = ((iz >= HZ) & (iz < ZMAX - HZ) & (iy >= HY) & (iy < Y - HY)
              & (ix >= HX) & (ix < X - HX))
    n_drop = int((~inside).sum())
    ev = ev[inside].reset_index(drop=True)
    iz, iy, ix = iz[inside], iy[inside], ix[inside]
    base = (iz * (Y * X) + iy * X + ix).astype(np.int64)
    off = (kern["VOX_OFF"][:, 0].astype(np.int64) * (Y * X)
           + kern["VOX_OFF"][:, 1].astype(np.int64) * X
           + kern["VOX_OFF"][:, 2].astype(np.int64))

    ts = ev.t.to_numpy()
    dts = list(range(a.dt_lo, a.dt_hi + 1))
    frames = [f for f in range(a.frame_lo, a.frame_hi + 1)
              if 0 <= f < meta["shape"][0]]

    q: queue.Queue = queue.Queue(maxsize=2)

    def producer():
        for f in frames:
            try:
                q.put((f, {zc: fetch_chunk(a.embryo, f, zc, a.x_chunk)
                           for zc in z_slabs}))
            except Exception as exc:                        # noqa: BLE001
                q.put((f, exc))
        q.put((None, None))

    threading.Thread(target=producer, daemon=True).start()

    cols = {k: [] for k in h1i.FEATURE_COLS}
    idx_all, out_dt, out_frame = [], [], []
    t_start, n_bytes, stats = time.time(), 0, []
    while True:
        f, raws = q.get()
        if f is None:
            break
        if isinstance(raws, Exception):
            print(json.dumps({"frame": f, "error": repr(raws)}), flush=True)
            continue
        n_bytes += sum(len(v) for v in raws.values())
        buf = np.zeros((ZMAX, Y, X), dtype=np.uint16)
        for k, zc in enumerate(z_slabs):
            arr = decode(raws[zc], (cz, cy, cx))
            buf[k * cz:(k + 1) * cz] = arr[:, :Y, :X]
        del raws, arr
        buf = buf.ravel()

        sub = tr[tr.t == f]
        vs = F.VOX_ZEBRAHUB_L0
        z0 = max(0, int(sub.z.min() - BBOX_MARGIN_UM / vs[0]) - ZOFF)
        z1 = min(ZMAX, int(sub.z.max() + BBOX_MARGIN_UM / vs[0]) + 1 - ZOFF)
        y0 = max(0, int(sub.y.min() - BBOX_MARGIN_UM / vs[1]))
        y1 = min(Y, int(sub.y.max() + BBOX_MARGIN_UM / vs[1]) + 1)
        x0 = max(0, int(sub.x.min() - BBOX_MARGIN_UM / vs[2]))
        x1 = min(X, int(sub.x.max() + BBOX_MARGIN_UM / vs[2]) + 1)
        z0, z1 = max(0, z0), max(1, z1)
        v3 = buf.reshape(ZMAX, Y, X)[z0:z1:2, y0:y1:3, x0:x1:3]
        bg = float(np.percentile(v3, 20.0))
        hi = float(np.percentile(v3, 99.5))
        del v3

        for dt in dts:
            rows = np.flatnonzero(ts + dt == f)
            if rows.size == 0:
                continue
            for s in range(0, rows.size, 2048):
                b = rows[s:s + 2048]
                box = buf[base[b][:, None] + off[None, :]]
                fv = h1i._node_features(box, bg)
                for k in h1i.FEATURE_COLS:
                    if k == "bg":
                        cols[k].append(np.full(b.size, bg, np.float32))
                    elif k == "hi":
                        cols[k].append(np.full(b.size, hi, np.float32))
                    else:
                        cols[k].append(np.asarray(fv[k], np.float32))
                idx_all.append(b)
                out_dt.append(np.full(b.size, dt, np.int8))
                out_frame.append(np.full(b.size, f, np.int16))
        del buf
        stats.append({"frame": f, "bg": bg, "hi": hi})
        print(json.dumps({"frame": f, "bg": bg, "hi": hi,
                          "MB": round(n_bytes / 1e6, 1),
                          "elapsed_s": round(time.time() - t_start, 1)}), flush=True)

    idx = np.concatenate(idx_all)
    nf = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    der = F.node_derived(nf, kern["n_core_vox"])
    der["ev"] = idx
    der["dt"] = np.concatenate(out_dt)
    der["frame"] = np.concatenate(out_frame)
    der["raw_mass_core"] = nf.mass_core.to_numpy()
    der["bg"] = nf.bg.to_numpy()
    der["hi"] = nf.hi.to_numpy()

    ev = ev.reset_index().rename(columns={"index": "ev"})
    out = der.merge(ev, on="ev", how="left")
    out["voxel_um_z"], out["voxel_um_y"], out["voxel_um_x"] = F.VOX_ZEBRAHUB_L0
    out["level"] = LEVEL
    out["feature_hash"] = F.config_hash()
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(outp, compression="zstd", index=False)

    rep = {"embryo": a.embryo, "frames": [a.frame_lo, a.frame_hi],
           "dt": [a.dt_lo, a.dt_hi], "z_slabs": list(z_slabs),
           "events": int(len(ev)), "positives": int((ev.label == 1).sum()),
           "negatives": int((ev.label == 0).sum()),
           "realigned_pos": int(((ev.label == 1) & ev.realigned).sum()),
           "profile_rows": int(len(out)),
           "dropped_outside_slab": n_drop,
           "MB_downloaded": round(n_bytes / 1e6, 1),
           "seconds": round(time.time() - t_start, 1),
           "feature_hash": F.config_hash()}
    print(json.dumps(rep, indent=2), flush=True)
    Path(str(outp) + ".report.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
