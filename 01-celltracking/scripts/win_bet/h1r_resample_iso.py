r"""H1-R -- resample a fetched Zebrahub pyramid level onto the deployed 1.625 um isotropic grid.

WHY THIS EXISTS
---------------
`h1r_fetch_imaging.py` streams a pyramid level into a local zarr at that level's native
anisotropic scale. The deployed detector input is 64^3 **isotropic at 1.625 um**, and the
packaged `zh001r` crops were measured (by exact registration, `h1r_zh001r_register.py`) to be on
exactly that grid. To train the edge half on our own stream -- which, unlike the packaged crops,
carries Ultrack track ids natively -- the fetched level has to be put on the same grid.

TWO THINGS THAT MUST NOT BE HARD-CODED
--------------------------------------
1. The voxel scale is **per embryo**, read from `.zattrs`. Measured from the published pyramids:
   ZSNS001 level-1 is (2.48, 0.878, 0.878) um, while the locally fetched ZSNS003 level-1 is
   (0.62, 0.2195, 0.2195) um. Against a 1.625 um target that is a 1.53x/1.85x *downsample* for
   one embryo and 2.62x/7.40x for the other. A fixed factor silently corrupts one of them.
2. Downsampling without a low-pass prefilter aliases. `scipy.ndimage.zoom` does **not**
   anti-alias on the way down; the Gaussian prefilter below is what makes the result faithful.

Usage:
  .venv\Scripts\python.exe scripts\win_bet\h1r_resample_iso.py --bench          # time the methods
  .venv\Scripts\python.exe scripts\win_bet\h1r_resample_iso.py \
      --src data/external/zebrahub/imaging/ZSNS003_L1.zarr --out .../ZSNS003_iso.zarr
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import zarr
from scipy import ndimage as ndi

ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
TARGET_UM = 1.625


def src_scale(group) -> np.ndarray:
    """Physical (z, y, x) voxel size of a zarr written by h1r_fetch_imaging.py."""
    meta = dict(group.attrs).get("zebrahub")
    if not meta or "scale_zyx" not in meta:
        raise SystemExit("source zarr has no zebrahub/scale_zyx attr -- refetch with "
                         "h1r_fetch_imaging.py so the scale travels with the data")
    return np.asarray(meta["scale_zyx"], float)


def resample_frame(vol: np.ndarray, scale_zyx: np.ndarray, target: float = TARGET_UM,
                   antialias: bool = True, order: int = 1) -> np.ndarray:
    """Resample one (Z, Y, X) frame onto an isotropic `target` um grid.

    zoom < 1 on any axis means downsampling; a Gaussian with sigma = (factor - 1) / 2 input
    voxels is applied on those axes first (the standard anti-alias prefilter -- sigma scales
    with the decimation factor, and is 0 where no decimation happens).
    """
    zoom = scale_zyx / target                      # >1 upsamples, <1 downsamples
    v = vol.astype(np.float32, copy=False)
    if antialias:
        factor = np.maximum(1.0 / zoom, 1.0)       # decimation factor per axis, >= 1
        sigma = (factor - 1.0) / 2.0
        if np.any(sigma > 0):
            v = ndi.gaussian_filter(v, sigma=sigma, mode="nearest")
    return ndi.zoom(v, zoom, order=order, mode="nearest", prefilter=False)


def bench(src: Path, t_index: int = 0) -> None:
    g = zarr.open_group(str(src), mode="r")
    arr = g["0"]
    sc = src_scale(g)
    vol = np.asarray(arr[t_index, 0])
    zoom = sc / TARGET_UM
    out_shape = tuple(int(round(s * z)) for s, z in zip(vol.shape, zoom))
    print(f"source : {src.name}  frame shape {vol.shape} {vol.dtype}  scale_zyx={sc.tolist()} um")
    print(f"         physical extent (z,y,x) = "
          f"{tuple(round(s * c, 1) for s, c in zip(sc, vol.shape))} um")
    print(f"target : 1.625 um isotropic -> {out_shape}  "
          f"(decimation z {1/zoom[0]:.2f}x, xy {1/zoom[1]:.2f}x)")
    print(f"         {np.prod(out_shape) / 64**3:.1f} x the 64^3 deployed input volume\n")

    variants = [
        ("nearest,   no antialias", dict(antialias=False, order=0)),
        ("linear,    no antialias", dict(antialias=False, order=1)),
        ("linear,    antialiased ", dict(antialias=True, order=1)),
        ("cubic,     antialiased ", dict(antialias=True, order=3)),
    ]
    ref = None
    for name, kw in variants:
        t0 = time.time()
        out = resample_frame(vol, sc, **kw)
        el = time.time() - t0
        if kw.get("antialias") and kw.get("order") == 3:
            ref = out
        print(f"  {name}: {el:6.2f} s/frame   out {out.shape}  "
              f"mean {out.mean():8.2f}  std {out.std():7.2f}")
    if ref is not None:
        print("\n  deviation from the cubic+antialias reference (RMS of intensity units):")
        for name, kw in variants[:-1]:
            out = resample_frame(vol, sc, **kw)
            print(f"    {name}: RMS {float(np.sqrt(((out - ref) ** 2).mean())):.3f}  "
                  f"(reference std {ref.std():.2f})")


def convert(src: Path, out: Path, order: int = 1, antialias: bool = True) -> None:
    g = zarr.open_group(str(src), mode="r")
    arr = g["0"]
    sc = src_scale(g)
    n = arr.shape[0]
    probe = resample_frame(np.asarray(arr[0, 0]), sc, antialias=antialias, order=order)
    og = zarr.open_group(str(out), mode="a")
    if "0" in og:
        oarr = og["0"]
    else:
        oarr = og.create_array("0", shape=(n, 1) + probe.shape,
                               chunks=(1, 1) + probe.shape, dtype=np.uint16)
    meta = dict(g.attrs).get("zebrahub", {})
    og.attrs["zebrahub"] = {**meta, "scale_zyx": [TARGET_UM] * 3,
                            "resampled_from": meta.get("scale_zyx"),
                            "resample": {"order": order, "antialias": antialias}}
    t0 = time.time()
    for i in range(n):
        v = probe if i == 0 else resample_frame(np.asarray(arr[i, 0]), sc,
                                                antialias=antialias, order=order)
        oarr[i, 0] = np.clip(v, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        print(f"  frame {i + 1}/{n}  {(time.time() - t0) / (i + 1):.2f} s/frame", flush=True)
    print(f"done -> {out}  shape {oarr.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/external/zebrahub/imaging/ZSNS003_L1.zarr"))
    ap.add_argument("--out", default="")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--order", type=int, default=1)
    ap.add_argument("--no-antialias", action="store_true")
    a = ap.parse_args()
    if a.bench or not a.out:
        bench(Path(a.src))
        return
    convert(Path(a.src), Path(a.out), order=a.order, antialias=not a.no_antialias)


if __name__ == "__main__":
    main()
