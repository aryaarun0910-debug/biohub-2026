"""H1-M: mother-centred appearance extraction from Zebrahub OME-Zarr imagery.

Streams level-0 chunks over anonymous HTTPS, decodes them, gathers the SAME physical
6 um sphere that `h1i_node_appearance` gathers on the competition crops (kernel rebuilt
for Zebrahub voxel spacing 1.24 / 0.439 / 0.439 um), and writes one row per
(embryo, mother, t, dt).

Data source and licence: Zebrahub, CZ Biohub SF (Royer lab), CC BY 4.0 --
Lange, Granados, VijayKumar et al., Cell (2024), doi:10.1016/j.cell.2024.09.047.
See reports/H1M_EXTERNAL_PROVENANCE (Lane C) and the prior agent's
EXTERNAL_DATA_PROVENANCE.md for checksums and the leakage check.

Design notes that matter for transfer:

  * NO PADDING. A mother is only sampled where the full 6 um sphere lies inside the
    z-slab that was downloaded, so no feature is ever computed against replicated
    edge voxels. Events too close to a slab or field boundary are dropped and counted.
  * `bg` / `hi` (the frame dynamic range that normalises `massn`, `peakn`, `contrast`)
    are computed over the CELL-OCCUPIED bounding box of the frame, not the whole
    embryo volume. The competition crops are already tight around cells, so this is
    the transformation that makes the frame statistic the same quantity on both
    sides; measured bg/hi are written out so the match can be checked rather than
    assumed.

Usage:
  .venv\\Scripts\\python.exe scripts\\h1m_zebrahub_appearance.py \
      --events <agent4 external_events parquet> --embryo ZSNS003 \
      --tracks <zebrahub_prep parquet> --out <parquet> --neg-per-window 6000
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

import h1m_features as F  # noqa: E402
import h1i_node_appearance as h1i  # noqa: E402

BASE = "https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/"
LEVEL = 0
Z_SLABS = (0, 1)          # z 0..255; slab 2 holds <2 % of cells and is not fetched
X_CHUNK = 0               # x 0..2303; ZSNS004/005 lose the thin far-field x chunk
BBOX_MARGIN_UM = 20.0


def zarray(embryo: str) -> dict:
    u = f"{BASE}{embryo}.ome.zarr/{LEVEL}/.zarray"
    return json.load(urllib.request.urlopen(u, timeout=120))


def fetch_chunk(embryo: str, t: int, zc: int, xc: int) -> bytes:
    u = f"{BASE}{embryo}.ome.zarr/{LEVEL}/{t}/0/{zc}/0/{xc}"
    return urllib.request.urlopen(u, timeout=900).read()


def decode(raw: bytes, shape) -> np.ndarray:
    from numcodecs import Blosc
    return np.frombuffer(Blosc().decode(raw), dtype="<u2").reshape(shape)


def build_frame(embryo: str, t: int, meta: dict, raws: dict) -> np.ndarray:
    """Assemble the (256, Y, X) uint16 slab for one frame from prefetched bytes."""
    cz, cy, cx = meta["chunks"][2], meta["chunks"][3], meta["chunks"][4]
    Y = min(meta["shape"][3], cy)
    X = min(meta["shape"][4] - X_CHUNK * cx, cx)
    buf = np.zeros((cz * len(Z_SLABS), Y, X), dtype=np.uint16)
    for k, zc in enumerate(Z_SLABS):
        a = decode(raws[zc], (cz, cy, cx))
        buf[k * cz:(k + 1) * cz] = a[:, :Y, :X]
    return buf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="agent4 external_events parquet")
    ap.add_argument("--tracks", required=True, help="zebrahub_prep track parquet")
    ap.add_argument("--embryo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--neg-per-window", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--t-lo", type=int, default=None, help="restrict events to t >= this")
    ap.add_argument("--t-hi", type=int, default=None, help="restrict events to t <= this")
    a = ap.parse_args()

    from numcodecs import blosc
    blosc.set_nthreads(4)

    kern = F.install_kernel(F.VOX_ZEBRAHUB_L0)
    HZ, HY, HX = kern["HZ"], kern["HY"], kern["HX"]

    ev = pd.read_parquet(a.events)
    if a.t_lo is not None:
        ev = ev[ev.t >= a.t_lo]
    if a.t_hi is not None:
        ev = ev[ev.t <= a.t_hi]
    ev = ev.sort_values(["t", "mother", "rank"])
    ev["_pos"] = (ev.label == "positive")
    g = ev.groupby(["t", "mother"], sort=False)
    m = pd.DataFrame({
        "mother_track_id": g.mother_track_id.first(),
        "mother_gt_outdeg": g.mother_gt_outdeg.first(),
        "n_cand": g.size(),
        "best_resid_um": g.flow_midpoint_residual.min(),
        "realisable": g._pos.max().astype(bool),
        "local_density": g.local_density_t.first(),
        "track_age": g.mother_track_age.first(),
        "speed_um": g.mother_speed_um.first(),
        "z_vox": g.mother_z_raw.first(), "y_vox": g.mother_y_raw.first(),
        "x_vox": g.mother_x_raw.first(),
        "z_um": g.mother_z_um.first(), "y_um": g.mother_y_um.first(),
        "x_um": g.mother_x_um.first(),
    }).reset_index()
    m["label"] = np.where(m.mother_gt_outdeg >= 2, 1,
                          np.where(m.mother_gt_outdeg == 1, 0, -1))

    meta = zarray(a.embryo)
    cz, cy, cx = meta["chunks"][2], meta["chunks"][3], meta["chunks"][4]
    Y = min(meta["shape"][3], cy)
    X = min(meta["shape"][4] - X_CHUNK * cx, cx)
    ZMAX = cz * len(Z_SLABS)
    iz = np.rint(m.z_vox.to_numpy()).astype(np.int64)
    iy = np.rint(m.y_vox.to_numpy()).astype(np.int64)
    ix = np.rint(m.x_vox.to_numpy()).astype(np.int64)
    inside = ((iz >= HZ) & (iz < ZMAX - HZ) & (iy >= HY) & (iy < Y - HY)
              & (ix >= HX) & (ix < X - HX))
    n_drop = int((~inside).sum())
    m, iz, iy, ix = m[inside].reset_index(drop=True), iz[inside], iy[inside], ix[inside]

    rng = np.random.default_rng(a.seed)
    pos = np.flatnonzero(m.label.to_numpy() == 1)
    neg = np.flatnonzero(m.label.to_numpy() == 0)
    n_take = min(a.neg_per_window, neg.size)
    hard_order = neg[np.argsort(m.best_resid_um.to_numpy()[neg])]
    n_hard = n_take // 2
    hard = hard_order[:n_hard]
    rest = np.setdiff1d(neg, hard, assume_unique=False)
    rand = rng.choice(rest, size=min(n_take - n_hard, rest.size), replace=False)
    keep = np.sort(np.concatenate([pos, hard, rand]))
    sel = m.iloc[keep].reset_index(drop=True)
    sel["hard_negative"] = np.isin(keep, hard)
    sel["neg_sample_frac"] = (hard.size + rand.size) / max(neg.size, 1)
    sel["neg_pool_total"] = int(neg.size)
    iz, iy, ix = iz[keep], iy[keep], ix[keep]
    base = (iz * (Y * X) + iy * X + ix).astype(np.int64)
    off = (kern["VOX_OFF"][:, 0].astype(np.int64) * (Y * X)
           + kern["VOX_OFF"][:, 1].astype(np.int64) * X
           + kern["VOX_OFF"][:, 2].astype(np.int64))

    tr = pd.read_parquet(a.tracks, columns=["t", "z", "y", "x"])
    ts = sel.t.to_numpy()
    frames = sorted({int(t) + dt for t in np.unique(ts) for dt in F.DTS})
    frames = [f for f in frames if 0 <= f < meta["shape"][0]]

    # prefetch raw chunk bytes one frame ahead of the decoder
    q: queue.Queue = queue.Queue(maxsize=2)

    def producer():
        for f in frames:
            try:
                q.put((f, {zc: fetch_chunk(a.embryo, f, zc, X_CHUNK) for zc in Z_SLABS}))
            except Exception as exc:                       # noqa: BLE001
                q.put((f, exc))
        q.put((None, None))

    threading.Thread(target=producer, daemon=True).start()

    rows_node, rows_dt, rows_frame = [], [], []
    cols = {k: [] for k in h1i.FEATURE_COLS}
    idx_all = []
    t0, n_bytes, stats = time.time(), 0, []
    while True:
        f, raws = q.get()
        if f is None:
            break
        if isinstance(raws, Exception):
            print(json.dumps({"frame": f, "error": repr(raws)}), flush=True)
            continue
        n_bytes += sum(len(v) for v in raws.values())
        buf = build_frame(a.embryo, f, meta, raws).ravel()
        del raws

        sub = tr[tr.t == f]
        vs = F.VOX_ZEBRAHUB_L0
        z0 = max(0, int(sub.z.min() - BBOX_MARGIN_UM / vs[0]))
        z1 = min(ZMAX, int(sub.z.max() + BBOX_MARGIN_UM / vs[0]) + 1)
        y0 = max(0, int(sub.y.min() - BBOX_MARGIN_UM / vs[1]))
        y1 = min(Y, int(sub.y.max() + BBOX_MARGIN_UM / vs[1]) + 1)
        x0 = max(0, int(sub.x.min() - BBOX_MARGIN_UM / vs[2]))
        x1 = min(X, int(sub.x.max() + BBOX_MARGIN_UM / vs[2]) + 1)
        v3 = buf.reshape(ZMAX, Y, X)[z0:z1:2, y0:y1:3, x0:x1:3]
        bg = float(np.percentile(v3, 20.0))
        hi = float(np.percentile(v3, 99.5))
        del v3

        for dt in F.DTS:
            selrows = np.flatnonzero(ts + dt == f)
            if selrows.size == 0:
                continue
            for s in range(0, selrows.size, 2048):
                b = selrows[s:s + 2048]
                box = buf[base[b][:, None] + off[None, :]]
                fv = h1i._node_features(box, bg)
                for k in h1i.FEATURE_COLS:
                    if k == "bg":
                        cols[k].append(np.full(b.size, bg, dtype=np.float32))
                    elif k == "hi":
                        cols[k].append(np.full(b.size, hi, dtype=np.float32))
                    else:
                        cols[k].append(np.asarray(fv[k], dtype=np.float32))
                idx_all.append(b)
                rows_dt.append(np.full(b.size, dt, dtype=np.int8))
                rows_frame.append(np.full(b.size, f, dtype=np.int16))
        del buf
        stats.append({"frame": f, "bg": bg, "hi": hi})
        print(json.dumps({"frame": f, "bg": bg, "hi": hi,
                          "MB": round(n_bytes / 1e6, 1),
                          "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    idx = np.concatenate(idx_all)
    nf = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    nf["dt"] = np.concatenate(rows_dt)
    nf["frame"] = np.concatenate(rows_frame)
    nf["ev"] = idx
    nf["inside"] = True

    der = F.node_derived(nf, kern["n_core_vox"])
    der["ev"] = idx
    der["dt"] = nf.dt.to_numpy()
    by_dt = {}
    for dt in F.DTS:
        s = der[der.dt == dt].drop_duplicates("ev").set_index("ev")
        by_dt[dt] = s.reindex(np.arange(len(sel))).reset_index(drop=True)
    feats = F.mother_event_features(by_dt, sel.index)
    sel["n_frames_seen"] = sum(by_dt[d]["conc"].notna().to_numpy().astype(int)
                               for d in F.DTS)
    sel["L_raw_massn"] = by_dt[0]["massn"].to_numpy()
    out_df = pd.concat([sel, feats], axis=1)
    out_df["embryo"] = a.embryo
    out_df["family"] = a.embryo
    out_df["window"] = f"{int(sel.t.min())}_{int(sel.t.max())}"
    out_df["voxel_um_z"], out_df["voxel_um_y"], out_df["voxel_um_x"] = F.VOX_ZEBRAHUB_L0
    out_df["level"] = LEVEL
    out_df["feature_hash"] = F.config_hash()

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(outp, compression="zstd", index=False)
    rep = {"embryo": a.embryo, "rows": int(len(out_df)),
           "positives": int((out_df.label == 1).sum()),
           "negatives": int((out_df.label == 0).sum()),
           "hard_negatives": int(out_df.hard_negative.sum()),
           "unlabeled": int((out_df.label == -1).sum()),
           "dropped_outside_slab": n_drop,
           "neg_pool_total": int(sel.neg_pool_total.iloc[0]),
           "neg_sample_frac": float(sel.neg_sample_frac.iloc[0]),
           "complete_5frame": int((out_df.n_frames_seen == 5).sum()),
           "frames": len(frames), "MB_downloaded": round(n_bytes / 1e6, 1),
           "seconds": round(time.time() - t0, 1),
           "bg_hi": stats[:3], "feature_hash": F.config_hash()}
    print(json.dumps(rep, indent=2), flush=True)
    Path(str(outp) + ".report.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
