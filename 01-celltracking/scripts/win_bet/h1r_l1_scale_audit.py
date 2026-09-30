r"""H1-R edge half -- audit the voxel scale of our OWN streamed Zebrahub level-1 imaging.

WHY THIS EXISTS
---------------
The edge half of the H1 retrain needs Zebrahub imaging resampled onto the deployed 1.625 um
isotropic grid. The resample factor was taken from the `scale_zyx` attribute our fetcher wrote
(`h1r_fetch_imaging.py:88`, read by `level_scale()` at :50-59 out of the OME-Zarr multiscales
metadata), giving [0.62, 0.2195, 0.2195] um and hence a ~7.4x lateral downsample.

`retrain_recipes_2026-08-17` section 11.4 named exactly this class of error as the highest-value
zero-GPU pre-flight: "a scale error here silently poisons the whole retrain" -- while training
looks perfectly healthy. That warning was aimed at a third party's packaged crops. This script
points the same instrument at OUR OWN fetch, which had never been checked.

It measures the level-1 voxel size from the IMAGE CONTENT, three independent ways:

  1. NUCLEUS RULER   -- per-axis intensity half-width around track nodes, in voxels, compared
                        against the same measurement on competition data where the voxel size is
                        known exactly from the zarr multiscales ([1.625, 0.40625, 0.40625] um).
                        Lateral (y/x) is the trustworthy axis: lateral PSF is near
                        diffraction-limited in both instruments. Axial is PSF-confounded and is
                        reported but NOT used for the verdict.
  2. NUCLEAR SPACING -- median nearest-neighbour distance between track nodes. Under the correct
                        scale this must land near a nuclear diameter (~5-10 um). It is a pure
                        consistency check and needs no imaging.
  3. EMBRYO EXTENT   -- bounding box of the nodes. Must be able to contain ~9.6k nuclei at the
                        implied spacing; a scale that shrinks the embryo below its own cell count
                        is self-refuting.

Exit code is non-zero when the ruler disagrees with the recorded attribute by more than
--tolerance (default 1.25x), because at that point the recorded scale must not be used to build
the resample.

Usage:
  .venv\Scripts\python.exe scripts\win_bet\h1r_l1_scale_audit.py
  .venv\Scripts\python.exe scripts\win_bet\h1r_l1_scale_audit.py --zarr <path> --tracks <parquet>
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl
import zarr

ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())

DEFAULT_ZARR = ROOT / "data" / "external" / "zebrahub" / "imaging" / "ZSNS003_L1.zarr"
DEFAULT_TRACKS = (ROOT / "_evidence" / "agent_runs" / "agent4" / "zebrahub_prep"
                  / "ZSNS003.parquet")
LEVEL1_DS = 2               # level-0 voxel track coords -> level-1 grid (h1r_train_smoke.py:32)
COMP_SCALE = (1.625, 0.40625, 0.40625)   # competition zarr multiscales, verified from attrs
PROFILE_R = 12


def axis_profile(vol: np.ndarray, pts: np.ndarray, axis: int, r: int = PROFILE_R):
    """Mean 1-D intensity profile along *axis* through node centres. Returns (profile, n_used)."""
    acc = np.zeros(2 * r + 1, dtype=np.float64)
    cnt = 0
    sh = vol.shape
    for p in pts:
        i = [int(round(float(v))) for v in p]
        if not all(r <= i[a] < sh[a] - r for a in range(3)):
            continue
        sl: list = [i[0], i[1], i[2]]
        sl[axis] = slice(i[axis] - r, i[axis] + r + 1)
        acc += vol[tuple(sl)].astype(np.float64)
        cnt += 1
    if cnt == 0:
        return None, 0
    p = acc / cnt
    p = p - p.min()
    return p / max(p[r], 1e-9), cnt


def half_width(prof: np.ndarray) -> float | None:
    """Interpolated radius (in voxels) at which the profile falls to half its peak."""
    r = len(prof) // 2
    for i in range(r + 1, len(prof)):
        if prof[i] <= 0.5:
            return (i - 1 - r) + (prof[i - 1] - 0.5) / max(prof[i - 1] - prof[i], 1e-9)
    return None


def competition_reference(n_crops: int = 1, n_frames: int = 40) -> dict:
    """Per-axis nucleus half-width in MICRONS on competition data (known voxel size)."""
    out: dict[str, float] = {}
    for zp in sorted(glob.glob(str(ROOT / "data" / "train" / "44b6_*.zarr")))[:n_crops]:
        gp = zp[:-5] + ".geff"
        arr = zarr.open_group(zp, mode="r")["0"]
        g = zarr.open(gp, mode="r")
        t = np.asarray(g["nodes/props/t/values"][:])
        z = np.asarray(g["nodes/props/z/values"][:])
        y = np.asarray(g["nodes/props/y/values"][:])
        x = np.asarray(g["nodes/props/x/values"][:])
        acc = {0: [], 1: [], 2: []}
        for tt in np.unique(t)[:n_frames]:
            m = t == tt
            if m.sum() == 0:
                continue
            vol = np.asarray(arr[int(tt)])
            pts = np.stack([z[m], y[m], x[m]], 1)
            for ax in (0, 1, 2):
                p, n = axis_profile(vol, pts, ax)
                if p is not None:
                    acc[ax].append((p, n))
        for ax, nm in ((0, "z"), (1, "y"), (2, "x")):
            if not acc[ax]:
                continue
            ps = np.array([p for p, _ in acc[ax]])
            ns = np.array([n for _, n in acc[ax]], dtype=float)
            p = (ps * ns[:, None]).sum(0) / ns.sum()
            p = p - p.min()
            p = p / max(p[PROFILE_R], 1e-9)
            hw = half_width(p)
            if hw:
                out[nm] = hw * COMP_SCALE[ax]
        break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", default=str(DEFAULT_ZARR))
    ap.add_argument("--tracks", default=str(DEFAULT_TRACKS))
    ap.add_argument("--frame", type=int, default=1, help="index into the fetched frames")
    ap.add_argument("--stride", type=int, default=12, help="probe every Nth node")
    ap.add_argument("--tolerance", type=float, default=1.25,
                    help="max acceptable ratio between ruler and recorded lateral scale")
    args = ap.parse_args()

    g = zarr.open_group(args.zarr, mode="r")
    arr = g["0"]
    recorded = list(g.attrs["zebrahub"]["scale_zyx"])
    print("== RECORDED ==")
    print(f"  array {arr.shape} {arr.dtype}")
    print(f"  scale_zyx (from OME multiscales via h1r_fetch_imaging.level_scale) = {recorded}")

    tracks = pl.read_parquet(args.tracks)
    t_probe = int(args.frame)
    d = tracks.filter(pl.col("t") == t_probe)
    coords_l1 = d.select(["z", "y", "x"]).to_numpy() / LEVEL1_DS
    print(f"  tracks at t={t_probe}: {len(d)} nodes")

    # --- 1. nucleus ruler ------------------------------------------------------------
    print("\n== 1. NUCLEUS RULER (image content) ==")
    vol = np.asarray(arr[t_probe, 0])
    sel = coords_l1[:: args.stride]
    comp = competition_reference()
    print(f"  competition half-width (known scale): "
          + "  ".join(f"{k}={v:.2f}um" for k, v in comp.items()))
    implied: dict[str, float] = {}
    for ax, nm in ((0, "z"), (1, "y"), (2, "x")):
        p, n = axis_profile(vol, sel, ax)
        if p is None:
            print(f"  {nm}: no usable nodes")
            continue
        hw = half_width(p)
        if not hw or nm not in comp:
            print(f"  {nm}: profile too flat to read a half-width")
            continue
        implied[nm] = comp[nm] / hw
        print(f"  {nm}: zebrahub half-width {hw:.2f} vox (n={n})  ->  implied "
              f"{implied[nm]:.3f} um/voxel   (recorded {recorded[ax]:.4f}, "
              f"ratio {implied[nm] / recorded[ax]:.2f}x)")
    if not implied:
        print("  INCONCLUSIVE: no axis produced a half-width")
        return 0

    lateral = [implied[k] for k in ("y", "x") if k in implied]
    lat = float(np.mean(lateral)) if lateral else None
    lat_rec = float(np.mean(recorded[1:]))

    # --- 2. nuclear spacing ----------------------------------------------------------
    print("\n== 2. NUCLEAR SPACING (tracks only) ==")
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("  SKIPPED: scipy unavailable")
        cKDTree = None
    if cKDTree is not None:
        for label, s in (("recorded", np.array(recorded) / LEVEL1_DS * LEVEL1_DS),
                         ("ruler", np.array([implied.get("z", np.nan), lat, lat]))):
            if np.any(np.isnan(s)):
                continue
            um = coords_l1 * s
            dd, _ = cKDTree(um).query(um, k=2)
            print(f"  under {label:8s} scale ({np.round(s, 3).tolist()}): "
                  f"median NN {np.median(dd[:, 1]):.2f} um "
                  f"(p10 {np.percentile(dd[:, 1], 10):.2f}, "
                  f"p90 {np.percentile(dd[:, 1], 90):.2f})")
        print("  a nuclear spacing far below ~5 um is not biologically possible")

    # --- 3. embryo extent ------------------------------------------------------------
    print("\n== 3. EMBRYO EXTENT ==")
    for label, s in (("recorded", np.array(recorded)),
                     ("ruler", np.array([implied.get("z", lat), lat, lat]))):
        ext = (coords_l1.max(0) - coords_l1.min(0)) * s
        print(f"  under {label:8s} scale: {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} um")
    print(f"  ({len(d)} nuclei must fit inside this box)")

    # --- verdict ---------------------------------------------------------------------
    print("\n== VERDICT ==")
    ratio = lat / lat_rec
    print(f"  lateral: ruler {lat:.3f} um/vox vs recorded {lat_rec:.4f} um/vox "
          f"-> {ratio:.2f}x disagreement")
    if ratio > args.tolerance or ratio < 1.0 / args.tolerance:
        print(f"  GATE FAIL: recorded scale disagrees with image content by {ratio:.2f}x "
              f"(tolerance {args.tolerance}x).")
        print("  DO NOT build the edge-half resample on the recorded attribute.")
        print("  Re-audit the OME-Zarr multiscales metadata (needs internet) and, until then,")
        print("  calibrate the resample empirically so nucleus half-width in um matches the")
        print(f"  competition reference. Implied resample to the deployed 1.625 um grid: "
              f"z {implied.get('z', float('nan')) / 1.625:.2f}x, xy {lat / 1.625:.2f}x.")
        return 1
    print("  PASS: recorded scale is consistent with the image content.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
