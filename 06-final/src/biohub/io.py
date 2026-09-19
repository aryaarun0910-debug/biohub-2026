"""Direct Zarr v3 / GEFF readers.

Zarr's Python API costs ~10.4 ms per timepoint; reading the chunk file and
calling the codec directly costs ~2.1 ms. Everything downstream is a loop over
timepoints, so the 5x is worth the hundred lines.

Measured on this machine (M5 Pro, APFS internal):
    raw read+decode, cold   2.13 ms/chunk   468 timepoints/s
    zarr z[t] Python API   10.42 ms/chunk    96 timepoints/s
The GPU consumes ~0.3 timepoints/s, so a single reader thread has ~1500x headroom.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

VOXEL_SCALE_UM = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)  # (z, y, x)
MATCH_RADIUS_UM = 7.0
NODE_COUNT_PENALTY_A = 0.1
DIVISION_WEIGHT = 0.1


def _meta(path: Path) -> dict:
    return json.loads((path / "zarr.json").read_text())


def _dtype(meta: dict) -> np.dtype:
    endian = meta["codecs"][0].get("configuration", {}).get("endian", "little")
    return np.dtype(meta["data_type"]).newbyteorder("<" if endian == "little" else ">")


# --------------------------------------------------------------------------
# image volumes:  <sample>.zarr/0/c/<t>/0/0/0   blosc(zstd, bitshuffle) uint16
# --------------------------------------------------------------------------

def image_meta(zarr_path: str | Path) -> dict:
    """Shape/dtype/chunk info for the level-0 array of an image store."""
    meta = _meta(Path(zarr_path) / "0")
    return {
        "shape": tuple(meta["shape"]),
        "chunks": tuple(meta["chunk_grid"]["configuration"]["chunk_shape"]),
        "dtype": _dtype(meta),
        "separator": meta["chunk_key_encoding"].get("configuration", {}).get("separator", "/"),
    }


def read_frame(zarr_path: str | Path, t: int) -> np.ndarray:
    """One timepoint as (Z, Y, X). Raises if the chunk is absent."""
    import blosc2

    root = Path(zarr_path) / "0"
    info = image_meta(zarr_path)
    sep = info["separator"]
    blob = root / ("c" + sep.join(["", str(t), "0", "0", "0"]))
    if not blob.exists():
        raise FileNotFoundError(blob)
    raw = blosc2.decompress2(blob.read_bytes())
    return np.frombuffer(raw, dtype=info["dtype"]).reshape(info["chunks"][1:])


def iter_frames(zarr_path: str | Path, ts=None):
    info = image_meta(zarr_path)
    for t in range(info["shape"][0]) if ts is None else ts:
        yield t, read_frame(zarr_path, t)


def quantiles(zarr_path: str | Path) -> dict:
    """Precomputed intensity quantiles stored by the organisers, for normalisation."""
    attrs = _meta(Path(zarr_path)).get("attributes", {})
    return attrs.get("image_statistics", {}).get("quantiles", {})


# --------------------------------------------------------------------------
# label graphs:  <sample>.geff/{nodes/ids, nodes/props/*/values, edges/ids}
# plain zstd, no blosc
# --------------------------------------------------------------------------

def read_zstd_array(path: str | Path) -> np.ndarray:
    """A whole zarr v3 array whose codec chain is bytes->zstd, assembled from chunks."""
    import zstandard

    path = Path(path)
    meta = _meta(path)
    if meta.get("zarr_format") != 3:
        raise ValueError(f"{path}: expected zarr v3")
    codecs = [c["name"] for c in meta.get("codecs", [])]
    if codecs != ["bytes", "zstd"]:
        raise ValueError(f"{path}: unexpected codec chain {codecs}")
    dtype = _dtype(meta)
    shape = tuple(meta["shape"])
    chunks = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
    sep = meta["chunk_key_encoding"].get("configuration", {}).get("separator", "/")
    out = np.full(shape, meta.get("fill_value", 0), dtype=dtype)
    if 0 in shape:
        return out
    dec = zstandard.ZstdDecompressor()
    expected = int(np.prod(chunks)) * dtype.itemsize
    grid = tuple(int(np.ceil(s / c)) for s, c in zip(shape, chunks))
    for idx in np.ndindex(*grid):
        blob = path / ("c" + "".join(sep + str(i) for i in idx))
        if not blob.exists():
            continue  # unwritten chunk keeps the fill value
        block = np.frombuffer(
            dec.decompress(blob.read_bytes(), max_output_size=expected), dtype=dtype
        ).reshape(chunks)
        region = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, chunks, shape))
        out[region] = block[tuple(slice(0, r.stop - r.start) for r in region)]
    return out


def read_geff(geff_path: str | Path) -> dict:
    """Ground-truth graph: ids, t/z/y/x arrays, (N,2) edges, estimated node count.

    Edge column 0 is the source, per the GEFF spec.
    """
    geff_path = Path(geff_path)
    attrs = _meta(geff_path).get("attributes", {}).get("geff", {})
    if attrs.get("directed") is not True:
        raise ValueError(f"{geff_path.name}: expected an explicitly directed graph")
    nodes = read_zstd_array(geff_path / "nodes/ids")
    edges = read_zstd_array(geff_path / "edges/ids")
    props = {k: read_zstd_array(geff_path / f"nodes/props/{k}/values") for k in "tzyx"}
    if nodes.ndim != 1 or len(np.unique(nodes)) != len(nodes):
        raise ValueError(f"{geff_path.name}: invalid node ids")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"{geff_path.name}: expected (N,2) source/target edges")
    if not np.isin(edges, nodes).all():
        raise ValueError(f"{geff_path.name}: edge endpoint absent from nodes")
    est = (attrs.get("extra") or {}).get("estimated_number_of_nodes")
    return {
        "ids": nodes,
        "edges": edges,
        "estimated_number_of_nodes": float(est) if est else np.nan,
        **props,
    }


def to_um(zyx: np.ndarray) -> np.ndarray:
    """Voxel coordinates -> micrometres. Everything geometric happens in um."""
    return np.asarray(zyx, dtype=np.float64) * VOXEL_SCALE_UM


@lru_cache(maxsize=1)
def dataset_root() -> Path:
    return Path(__file__).resolve().parents[2] / "Data" / "biohub-cell-tracking-during-development"


def films(split: str = "train") -> list[str]:
    return sorted(p.stem for p in (dataset_root() / split).glob("*.zarr"))
