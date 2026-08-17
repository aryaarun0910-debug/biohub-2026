r"""H1-R (H1 retrain lane) -- fetch a Zebrahub OME-Zarr pyramid level into a LOCAL zarr.

Streams chunks directly over HTTP (fsspec/aiohttp not required) and assembles frames into a
local zarr group whose array "0" holds the chosen level's data, with the level's voxel scale
written to .zattrs. The competition trainer reads group["0"] strided by `downsample`; storing
the pyramid level at "0" lets the level-1 retrain reuse that path unchanged.

Level-0 is ~1.85 TB for 3 embryos and will not fit; level-1 (z-half, xy-half) is the chosen
retrain resolution (see research/01-research-direction/directional-updates.md, 2026-08-17).

Usage:
  .venv\Scripts\python.exe scripts\win_bet\h1r_fetch_imaging.py --embryo ZSNS003 --level 1 \
      --t-start 0 --t-end 8
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import numcodecs
import zarr

ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
BASE = "https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/{e}.ome.zarr"
OUT_ROOT = ROOT / "data" / "external" / "zebrahub" / "imaging"  # gitignored (data/)


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())


def _get_bytes(url: str, retries: int = 4) -> bytes:
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - transient network
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError("unreachable")


def level_scale(zattrs: dict, level: str) -> list[float]:
    """Physical voxel scale (Z, Y, X) for a pyramid level from OME-Zarr multiscales."""
    ms = zattrs["multiscales"][0]
    for ds in ms["datasets"]:
        if ds["path"] == level:
            for tf in ds.get("coordinateTransformations", []):
                if tf.get("type") == "scale":
                    s = tf["scale"]
                    return list(s[-3:])  # last 3 dims = Z, Y, X
    raise KeyError(f"no scale for level {level}")


def fetch(embryo: str, level: int, t_start: int, t_end: int) -> Path:
    base = BASE.format(e=embryo)
    zattrs = _get_json(base + "/.zattrs")
    za = _get_json(base + f"/{level}/.zarray")
    shape = za["shape"]            # (T, C, Z, Y, X)
    chunks = za["chunks"]
    dtype = np.dtype(za["dtype"])
    cname = za["compressor"]["cname"]
    codec = numcodecs.Blosc(cname=cname)
    T, C, Z, Y, X = shape
    czt, ccc, czz, cyy, cxx = chunks
    t_end = min(t_end, T)
    scale = level_scale(zattrs, str(level))
    n_zc, n_yc, n_xc = math.ceil(Z / czz), math.ceil(Y / cyy), math.ceil(X / cxx)
    if n_yc != 1 or n_xc != 1:
        # frames here are single y/x chunks; generalise only if that changes
        print(f"  note: {n_yc} y-chunks x {n_xc} x-chunks per frame")

    out = OUT_ROOT / f"{embryo}_L{level}.zarr"
    out.mkdir(parents=True, exist_ok=True)
    g = zarr.open_group(str(out), mode="a")
    n = t_end - t_start
    if "0" in g:
        arr = g["0"]
    else:
        arr = g.create_array("0", shape=(n, 1, Z, Y, X), chunks=(1, 1, czz, Y, X), dtype=dtype)
    g.attrs["zebrahub"] = {"embryo": embryo, "level": level, "scale_zyx": scale,
                           "t_start": t_start, "t_end": t_end, "source": base}

    print(f"{embryo} level {level}: frames [{t_start},{t_end}) -> {out.relative_to(ROOT)}")
    print(f"  shape/frame (Z,Y,X)=({Z},{Y},{X})  {n_zc} z-chunks/frame  scale_zyx={scale}")
    t0 = time.time()
    got_mb = 0.0
    for i, t in enumerate(range(t_start, t_end)):
        frame = np.zeros((Z, Y, X), dtype=dtype)
        for zc in range(n_zc):
            raw = _get_bytes(base + f"/{level}/{t}/0/{zc}/0/0")
            got_mb += len(raw) / 1e6
            block = np.frombuffer(codec.decode(raw), dtype=dtype).reshape(czz, cyy, cxx)
            z0 = zc * czz
            frame[z0:z0 + czz] = block[: min(czz, Z - z0), :Y, :X]
        arr[i, 0] = frame
        el = time.time() - t0
        print(f"  frame {t:4d} ({i+1}/{n})  {got_mb:7.1f} MB  {got_mb/el:5.1f} MB/s", flush=True)
    print(f"done: {got_mb/1e3:.2f} GB in {time.time()-t0:.0f}s -> {out.relative_to(ROOT)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embryo", default="ZSNS003")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--t-start", type=int, default=0)
    ap.add_argument("--t-end", type=int, default=8)
    a = ap.parse_args()
    fetch(a.embryo, a.level, a.t_start, a.t_end)


if __name__ == "__main__":
    main()
