r"""LEVER-0021 instrument (PKT-0017): DeepCenter sub-voxel refinement of an exported LOEO submission.

What it does
------------
Every node of a submission-format export carries an INTEGER (z,y,x) (FACT-0273). For each
frame this script evaluates the DeepCenter centre-prior UNet3D on the raw zarr chunk - the
preprocessing is copied VERBATIM from the deployed notebook
(notebooks/kaggle_p3_harmonic/biohub-p3-harmonic.ipynb cell 6:117-138 and 329-355) and is
identical to the trainer's block_mean_xy / normalize_dynamic_range - and replaces each node's
coordinate with a local sub-voxel centre read off the heatmap.

Heatmap geometry: z native (64 slices x 1.625 um); XY block-mean pooled 4x (64x64 x 1.625 um).
The training target was a sigma=1 Gaussian at heatmap index (z0, y0/4, x0/4)
(train_full_frame_center_detector.py::make_heatmap), so the TARGET-CONVENTION decode is
    z_full = cz,   y_full = 4*cy,   x_full = 4*cx.
The trainer's own peak decoder adds (pool-1)/2 = 1.5 in XY; that variant is emitted as a
DIAGNOSTIC arm (`com1_off`), not as a fitted correction.

Arms (pre-registered in PKT-0017)
    com1      PRIMARY    centre of mass of max(h-0.05, 0) over z+-1, yx+-1 pooled voxels, decode 4*c
    com1_off  DIAGNOSTIC same window, decode 4*c + 1.5
    par1                 parabolic sub-voxel peak at the window argmax, decode 4*c
    com2                 centre of mass over z+-2, yx+-2, decode 4*c
Guards (all arms): window max < 0.05 -> keep the original coordinate;
                   refined point > 3.0 um from the original -> keep the original coordinate.

Peak-snap arms (PKT-0017 amendment 1, pre-registered AFTER the pilot of the arms above and BEFORE
any snap measurement; they act on cached heatmaps, no re-inference):
    snap_r4   PRIMARY    local maxima of the heatmap (3x3x3, >= 0.10); each detection takes the
                         NEAREST peak within 4.0 um and is re-localised to the com1 centre of that
                         peak; two detections wanting one peak -> the nearer wins, the other keeps
                         its original coordinate; no peak within radius -> keep original
    snap_r3, snap_r5     radius sensitivity, reported not selected
    snapu_r4             as snap_r4 but only when the peak is the UNIQUE one within the radius

Intensity-centroid arms (LEVER-0022, the 0.927 public lineage's `refine_centroids`, cell13:74-113 of
arnav170/biohub-sdw60 - no model at all, just the RAW frame):
    icom_133  PRIMARY    intensity-weighted centroid in a z+-1, y+-3, x+-3 VOXEL window, weights
                         max(patch - p20(patch), 0), max shift 2.8 um (their exact settings)
    icom_122             window z+-1, y+-2, x+-2
    icom_155             window z+-1, y+-5, x+-5
    icom_133_s5          as icom_133 with max shift 5.0 um (tail-reach sensitivity)

Subcommands
-----------
refine    run the DeepCenter model, write refined_nodes.parquet + refined_<arm>.csv.gz + timing.json
icentroid no model: intensity centroid of the raw frame per node -> the same outputs (icom_* arms)
eval     FROZEN-MATCH residual per arm against a baseline ea_atlas dump (stage (a) statistic),
         with the calibration gate that the unrefined coordinates reproduce the atlas residual.

Usage
-----
  .\.venv\Scripts\python.exe scripts\win_bet\dc_subvoxel_refine.py refine ^
      --csv _evidence\exports\loeo_f1_strict\loeo_split1_strict.csv.gz ^
      --checkpoint C:\temp\biohub_deepcenter_p10_audit\best.pt ^
      --out-dir C:\temp\dc_refine\f1_best --crop-stride 10 --threads 8

  .\.venv\Scripts\python.exe scripts\win_bet\dc_subvoxel_refine.py eval ^
      --refined C:\temp\dc_refine\f1_best\refined_nodes.parquet ^
      --atlas-dir C:\temp\edge_atlas_strict --tag s1 --json-out C:\temp\dc_refine\f1_best\eval.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())

VOXEL_SCALE_UM = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
POS_THRESH = 0.05          # DeepCenter training `pos_thresh`; also the no-signal guard
MAX_SHIFT_UM = 3.0         # refinement further than this is a neighbour, not a refinement
ARMS = ("com1", "com1_off", "par1", "com2")
ICOM_ARMS = {"icom_133": ((1, 3, 3), 2.8), "icom_122": ((1, 2, 2), 2.8), "icom_155": ((1, 5, 5), 2.8),
             "icom_133_s5": ((1, 3, 3), 5.0)}
ICOM_BASELINE_PCT = 20.0
PEAK_THRESH = 0.10         # BIOHUB_DEEPCENTER_GAP_THRESHOLD default; peaks below it are not cells
SNAP_ARMS = {"snap_r4": (4.0, False), "snap_r3": (3.0, False), "snap_r5": (5.0, False), "snapu_r4": (4.0, True)}


# --------------------------------------------------------------------------------------
# Model - copied verbatim from the deployed notebook (cell 6:222-266)
# --------------------------------------------------------------------------------------
def build_model_classes(torch):
    class _DCConvBlock3d(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            groups = min(8, out_channels)
            self.block = torch.nn.Sequential(
                torch.nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
                torch.nn.GroupNorm(groups, out_channels),
                torch.nn.SiLU(inplace=True),
                torch.nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
                torch.nn.GroupNorm(groups, out_channels),
                torch.nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class _DCDeepCenterUNet3D(torch.nn.Module):
        def __init__(self, in_channels: int = 1, base_channels: int = 24) -> None:
            super().__init__()
            c = int(base_channels)
            self.enc1 = _DCConvBlock3d(in_channels, c)
            self.down1 = torch.nn.MaxPool3d(2, 2)
            self.enc2 = _DCConvBlock3d(c, c * 2)
            self.down2 = torch.nn.MaxPool3d(2, 2)
            self.enc3 = _DCConvBlock3d(c * 2, c * 4)
            self.down3 = torch.nn.MaxPool3d(2, 2)
            self.bottleneck = _DCConvBlock3d(c * 4, c * 8)
            self.up3 = torch.nn.ConvTranspose3d(c * 8, c * 4, 2, 2)
            self.dec3 = _DCConvBlock3d(c * 8, c * 4)
            self.up2 = torch.nn.ConvTranspose3d(c * 4, c * 2, 2, 2)
            self.dec2 = _DCConvBlock3d(c * 4, c * 2)
            self.up1 = torch.nn.ConvTranspose3d(c * 2, c, 2, 2)
            self.dec1 = _DCConvBlock3d(c * 2, c)
            self.head = torch.nn.Conv3d(c, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.down1(e1))
            e3 = self.enc3(self.down2(e2))
            b = self.bottleneck(self.down3(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)

    return _DCDeepCenterUNet3D


def load_deepcenter(checkpoint: Path, threads: int):
    import torch

    torch.set_num_threads(int(threads))
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model_state" not in ck:
        raise ValueError(f"{checkpoint}: checkpoint has no model_state")
    cfg = SimpleNamespace(**ck.get("config", {}))
    net = build_model_classes(torch)(base_channels=int(getattr(cfg, "base_channels", 24)))
    net.load_state_dict(ck["model_state"], strict=True)
    net.eval()
    return {"model": net, "cfg": cfg, "torch": torch, "epoch": int(ck.get("epoch", -1)),
            "path": str(checkpoint)}


# --------------------------------------------------------------------------------------
# Preprocessing - verbatim from the deployed notebook (cell 6:46-67, 117-138)
# --------------------------------------------------------------------------------------
def read_frame(zarr_root: Path, t: int) -> np.ndarray:
    meta = json.loads((zarr_root / "0" / "zarr.json").read_text())
    shape = tuple(int(v) for v in meta["shape"])
    dtype = np.dtype(meta["data_type"])
    frame_shape = shape[1:]
    chunk_path = zarr_root / "0" / "c" / str(t) / "0" / "0" / "0"
    try:
        import blosc2

        arr = np.frombuffer(blosc2.decompress(chunk_path.read_bytes()), dtype=dtype)
        if arr.size == int(np.prod(frame_shape)):
            return arr.reshape(frame_shape).copy()
    except Exception:
        pass
    import zarr

    return np.asarray(zarr.open(zarr_root / "0", mode="r")[t])


def pool_frame_xy(volume: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return volume.astype(np.float32, copy=False)
    z, y, x = volume.shape
    y2 = (y // factor) * factor
    x2 = (x // factor) * factor
    cropped = volume[:, :y2, :x2].astype(np.float32, copy=False)
    return cropped.reshape(z, y2 // factor, factor, x2 // factor, factor).mean(axis=(2, 4))


def normalize_dynamic_range(volume: np.ndarray, cfg: object) -> np.ndarray:
    vol = np.asarray(volume, dtype=np.float32)
    lo = float(np.percentile(vol, float(getattr(cfg, "norm_lo_pct", 50.0))))
    hi = float(np.percentile(vol, float(getattr(cfg, "norm_hi_pct", 99.5))))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    ratio = (vol - lo) / (hi - lo)
    return np.clip(ratio, float(getattr(cfg, "norm_clip_lo", -0.5)),
                   float(getattr(cfg, "norm_clip_hi", 6.0))).astype(np.float32)


def heatmap_for_frame(bundle: dict, frame: np.ndarray) -> np.ndarray:
    torch = bundle["torch"]
    cfg = bundle["cfg"]
    image = normalize_dynamic_range(pool_frame_xy(frame, int(getattr(cfg, "pool_factor", 4))), cfg)
    with torch.no_grad():
        tensor = torch.from_numpy(image[None, None, ...]).to(dtype=torch.float32)
        return torch.sigmoid(bundle["model"](tensor))[0, 0].numpy().astype(np.float32, copy=False)


# --------------------------------------------------------------------------------------
# Refinement - pure numpy, unit-tested
# --------------------------------------------------------------------------------------
def _decode(c_zyx: np.ndarray, pool: int, xy_offset: float) -> np.ndarray:
    out = np.asarray(c_zyx, dtype=np.float64).copy()
    out[1] = out[1] * pool + xy_offset
    out[2] = out[2] * pool + xy_offset
    return out


def _guarded(original: np.ndarray, refined: np.ndarray, max_shift_um: float) -> tuple[np.ndarray, bool]:
    shift = float(np.sqrt((((refined - original) * VOXEL_SCALE_UM) ** 2).sum()))
    if shift > max_shift_um:
        return original.copy(), False
    return refined, True


def _com(hm: np.ndarray, iz: int, iy: int, ix: int, rz: int, ryx: int, thresh: float):
    Z, Y, X = hm.shape
    z0, z1 = max(0, iz - rz), min(Z, iz + rz + 1)
    y0, y1 = max(0, iy - ryx), min(Y, iy + ryx + 1)
    x0, x1 = max(0, ix - ryx), min(X, ix + ryx + 1)
    patch = hm[z0:z1, y0:y1, x0:x1]
    pmax = float(patch.max()) if patch.size else 0.0
    if pmax < thresh:
        return None, pmax
    w = np.maximum(patch - thresh, 0.0).astype(np.float64)
    s = w.sum()
    if s <= 0:
        return None, pmax
    zz = np.arange(z0, z1, dtype=np.float64)[:, None, None]
    yy = np.arange(y0, y1, dtype=np.float64)[None, :, None]
    xx = np.arange(x0, x1, dtype=np.float64)[None, None, :]
    return np.array([(w * zz).sum() / s, (w * yy).sum() / s, (w * xx).sum() / s]), pmax


def _parabolic(hm: np.ndarray, iz: int, iy: int, ix: int, r: int, thresh: float):
    Z, Y, X = hm.shape
    z0, z1 = max(0, iz - r), min(Z, iz + r + 1)
    y0, y1 = max(0, iy - r), min(Y, iy + r + 1)
    x0, x1 = max(0, ix - r), min(X, ix + r + 1)
    patch = hm[z0:z1, y0:y1, x0:x1]
    if patch.size == 0 or float(patch.max()) < thresh:
        return None
    pz, py, px = np.unravel_index(int(np.argmax(patch)), patch.shape)
    pk = np.array([z0 + pz, y0 + py, x0 + px], dtype=np.int64)
    out = pk.astype(np.float64)
    dims = (Z, Y, X)
    for a in range(3):
        c = int(pk[a])
        if c <= 0 or c >= dims[a] - 1:
            continue
        im = pk.copy()
        im[a] = c - 1
        ip = pk.copy()
        ip[a] = c + 1
        lm, l0, lp = float(hm[tuple(im)]), float(hm[tuple(pk)]), float(hm[tuple(ip)])
        denom = lm - 2.0 * l0 + lp
        if denom >= -1e-9:
            continue
        out[a] = c + float(np.clip(0.5 * (lm - lp) / denom, -0.5, 0.5))
    return out


def refine_points(hm: np.ndarray, zyx: np.ndarray, pool: int = 4,
                  thresh: float = POS_THRESH, max_shift_um: float = MAX_SHIFT_UM) -> dict:
    """Refine full-res points (N,3) against a pooled heatmap of shape (Z, Y//pool, X//pool).

    Returns {arm: (N,3) float64 coords}, plus 'hm_max' (N,) and 'moved_<arm>' (N,) bools.
    """
    zyx = np.asarray(zyx, dtype=np.float64)
    n = len(zyx)
    Z, Y, X = hm.shape
    out = {a: zyx.copy() for a in ARMS}
    moved = {a: np.zeros(n, dtype=bool) for a in ARMS}
    hm_max = np.zeros(n, dtype=np.float32)
    for i, (z, y, x) in enumerate(zyx):
        iz = int(np.clip(round(z), 0, Z - 1))
        iy = int(np.clip(round(y / pool), 0, Y - 1))
        ix = int(np.clip(round(x / pool), 0, X - 1))
        c1, pmax = _com(hm, iz, iy, ix, 1, 1, thresh)
        hm_max[i] = pmax
        if c1 is not None:
            out["com1"][i], moved["com1"][i] = _guarded(zyx[i], _decode(c1, pool, 0.0), max_shift_um)
            out["com1_off"][i], moved["com1_off"][i] = _guarded(
                zyx[i], _decode(c1, pool, (pool - 1) / 2.0), max_shift_um)
        c2, _ = _com(hm, iz, iy, ix, 2, 2, thresh)
        if c2 is not None:
            out["com2"][i], moved["com2"][i] = _guarded(zyx[i], _decode(c2, pool, 0.0), max_shift_um)
        p1 = _parabolic(hm, iz, iy, ix, 1, thresh)
        if p1 is not None:
            out["par1"][i], moved["par1"][i] = _guarded(zyx[i], _decode(p1, pool, 0.0), max_shift_um)
    out["hm_max"] = hm_max
    for a in ARMS:
        out[f"moved_{a}"] = moved[a]
    return out


def heatmap_peaks(hm: np.ndarray, thresh: float = PEAK_THRESH) -> np.ndarray:
    """Integer (P,3) local maxima of the pooled heatmap: 3x3x3 maximum and >= thresh."""
    from scipy.ndimage import maximum_filter

    mask = (hm >= thresh) & (hm == maximum_filter(hm, size=3, mode="nearest"))
    return np.argwhere(mask)


def snap_points(hm: np.ndarray, zyx: np.ndarray, pool: int = 4, thresh: float = PEAK_THRESH,
                arms: dict[str, tuple[float, bool]] | None = None, com_thresh: float = POS_THRESH) -> dict:
    """Snap detections to the nearest heatmap peak and re-localise there (see module docstring).

    Returns {arm: (N,3)} plus 'moved_<arm>' (N,) bools and 'n_peaks'.
    """
    arms = SNAP_ARMS if arms is None else arms
    zyx = np.asarray(zyx, dtype=np.float64)
    n = len(zyx)
    out = {a: zyx.copy() for a in arms}
    moved = {a: np.zeros(n, dtype=bool) for a in arms}
    peaks = heatmap_peaks(hm, thresh)
    out["n_peaks"] = int(len(peaks))
    if len(peaks) == 0 or n == 0:
        for a in arms:
            out[f"moved_{a}"] = moved[a]
        return out
    # centre of each peak: the com1 operator seeded AT the peak, decoded with the target convention
    centres = np.empty((len(peaks), 3), dtype=np.float64)
    for k, (pz, py, px) in enumerate(peaks):
        c, _ = _com(hm, int(pz), int(py), int(px), 1, 1, com_thresh)
        centres[k] = _decode(c if c is not None else np.array([pz, py, px], dtype=np.float64), pool, 0.0)
    peak_full = np.column_stack(
        [peaks[:, 0].astype(np.float64), peaks[:, 1] * float(pool), peaks[:, 2] * float(pool)])
    # anisotropic distances detection -> peak, in um
    d = np.sqrt((((zyx[:, None, :] - peak_full[None, :, :]) * VOXEL_SCALE_UM) ** 2).sum(axis=2))  # (N,P)
    for a, (radius, unique) in arms.items():
        within = d <= radius
        n_within = within.sum(axis=1)
        nearest = np.argmin(d, axis=1)
        want = n_within >= 1
        if unique:
            want &= n_within == 1
        order = np.argsort(d[np.arange(n), nearest])   # nearer detections claim first
        taken: set[int] = set()
        for i in order:
            if not want[i]:
                continue
            k = int(nearest[i])
            if k in taken:
                continue
            taken.add(k)
            out[a][i] = centres[k]
            moved[a][i] = True
        out[f"moved_{a}"] = moved[a]
    return out


def intensity_centroid_points(vol: np.ndarray, zyx: np.ndarray, arms: dict | None = None,
                              baseline_pct: float = ICOM_BASELINE_PCT) -> dict:
    """The 0.927 lineage's refine_centroids on a RAW full-resolution frame (Z,Y,X), per arm.

    Returns {arm: (N,3) float64} plus 'moved_<arm>' (N,) bools. A point keeps its original
    coordinate when the window has no mass above the baseline or the shift exceeds the arm's cap.
    """
    arms = ICOM_ARMS if arms is None else arms
    zyx = np.asarray(zyx, dtype=np.float64)
    n = len(zyx)
    Z, Y, X = vol.shape
    out = {a: zyx.copy() for a in arms}
    moved = {a: np.zeros(n, dtype=bool) for a in arms}
    for a, ((wz, wy, wx), max_shift) in arms.items():
        for i, original in enumerate(zyx):
            z, y, x = [int(round(v)) for v in original]
            z0, z1 = max(0, z - wz), min(Z, z + wz + 1)
            y0, y1 = max(0, y - wy), min(Y, y + wy + 1)
            x0, x1 = max(0, x - wx), min(X, x + wx + 1)
            if z0 >= z1 or y0 >= y1 or x0 >= x1:
                continue
            patch = vol[z0:z1, y0:y1, x0:x1].astype(np.float64)
            baseline = float(np.percentile(patch, baseline_pct))
            w = np.maximum(patch - baseline, 0.0)
            total = float(w.sum())
            if total <= 0:
                continue
            zz = np.arange(z0, z1, dtype=np.float64)[:, None, None]
            yy = np.arange(y0, y1, dtype=np.float64)[None, :, None]
            xx = np.arange(x0, x1, dtype=np.float64)[None, None, :]
            refined = np.array([(w * zz).sum() / total, (w * yy).sum() / total, (w * xx).sum() / total])
            if float(np.sqrt((((refined - original) * VOXEL_SCALE_UM) ** 2).sum())) <= max_shift:
                out[a][i] = refined
                moved[a][i] = True
    for a in arms:
        out[f"moved_{a}"] = moved[a]
    return out


def cmd_icentroid(args) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(args.csv))
    node_mask = df["row_type"] == "node"
    all_names = df.loc[node_mask, "dataset"].unique().tolist()
    names = select_crops(all_names, args.max_crops, args.crop_stride, args.crops)
    print(f"{len(names)} crops selected of {len(all_names)}", flush=True)
    df = df.loc[df["dataset"].isin(names)].reset_index(drop=True)
    for c in ("z", "y", "x"):
        df[c] = df[c].astype(np.float64)
    nodes = df.loc[df["row_type"] == "node", ["dataset", "node_id", "t", "z", "y", "x"]].copy()
    arms = list(ICOM_ARMS)
    for a in arms:
        for c in ("z", "y", "x"):
            nodes[f"{c}_{a}"] = nodes[c].to_numpy(copy=True)
        nodes[f"moved_{a}"] = False
    col = {c: nodes.columns.get_loc(c) for c in nodes.columns}
    idx_by = nodes.groupby(["dataset", "t"]).indices
    t_start = time.time()
    n_frames = 0
    for ci, name in enumerate(names, 1):
        zarr_root = Path(args.data_dir) / f"{name}.zarr"
        if not zarr_root.exists():
            raise SystemExit(f"missing raw frames: {zarr_root}")
        ts = sorted({int(t) for (d, t) in idx_by if d == name})
        c0 = time.time()
        for t in ts:
            rows = idx_by[(name, t)]
            vol = read_frame(zarr_root, t)
            n_frames += 1
            res = intensity_centroid_points(vol, nodes.iloc[rows][["z", "y", "x"]].to_numpy())
            for a in arms:
                nodes.iloc[rows, col[f"z_{a}"]] = res[a][:, 0]
                nodes.iloc[rows, col[f"y_{a}"]] = res[a][:, 1]
                nodes.iloc[rows, col[f"x_{a}"]] = res[a][:, 2]
                nodes.iloc[rows, col[f"moved_{a}"]] = res[f"moved_{a}"]
        sub = nodes.loc[nodes["dataset"] == name]
        print(f"  [{ci}/{len(names)}] {name} frames={len(ts)} nodes={len(sub)} "
              f"moved(icom_133)={float(sub['moved_icom_133'].mean()):.3f} {time.time() - c0:.0f}s", flush=True)
    nodes.to_parquet(out / "refined_nodes.parquet")
    write_arm_csvs(df, nodes, arms, out)
    (out / "timing.json").write_text(json.dumps({"crops": names, "n_frames": n_frames, "n_nodes": int(len(nodes)),
                                                  "wall_s": time.time() - t_start, "arms": arms,
                                                  "icom_arms": {k: [list(v[0]), v[1]] for k, v in ICOM_ARMS.items()},
                                                  "baseline_pct": ICOM_BASELINE_PCT}, indent=2))
    print(f"wrote {out} ({n_frames} frames, {time.time() - t_start:.0f}s)")
    return 0


# --------------------------------------------------------------------------------------
# refine subcommand
# --------------------------------------------------------------------------------------
def select_crops(names: list[str], max_crops: int | None, stride: int | None,
                 explicit: str | None) -> list[str]:
    names = sorted(names)
    if explicit:
        want = [s.strip() for s in explicit.split(",") if s.strip()]
        missing = [w for w in want if w not in names]
        if missing:
            raise SystemExit(f"crops not in export: {missing}")
        return want
    if stride and stride > 1:
        names = names[::stride]
    if max_crops:
        names = names[:max_crops]
    return names


def cmd_refine(args) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = load_deepcenter(Path(args.checkpoint), args.threads)
    pool = int(getattr(bundle["cfg"], "pool_factor", 4))
    print(f"DeepCenter {bundle['path']} epoch={bundle['epoch']} pool={pool} threads={args.threads}",
          flush=True)

    df = pd.read_csv(Path(args.csv))
    node_mask = df["row_type"] == "node"
    all_names = df.loc[node_mask, "dataset"].unique().tolist()
    names = select_crops(all_names, args.max_crops, args.crop_stride, args.crops)
    print(f"{len(names)} crops selected of {len(all_names)}", flush=True)

    df = df.loc[df["dataset"].isin(names)].reset_index(drop=True)
    for c in ("z", "y", "x"):
        df[c] = df[c].astype(np.float64)   # exports are int64; refined coordinates are floats
    nodes = df.loc[df["row_type"] == "node", ["dataset", "node_id", "t", "z", "y", "x"]].copy()
    for c in ("z", "y", "x"):
        nodes[c] = nodes[c].astype(np.float64)
    for a in ARMS:
        for c in ("z", "y", "x"):
            nodes[f"{c}_{a}"] = nodes[c].to_numpy(copy=True)
        nodes[f"moved_{a}"] = False
    nodes["hm_max"] = np.float32(0.0)
    col = {c: nodes.columns.get_loc(c) for c in nodes.columns}

    gt_nodes = None
    gt_idx_by = {}
    if args.gt_parquet:
        gt_nodes = pd.read_parquet(args.gt_parquet)
        gt_nodes = gt_nodes.loc[gt_nodes["dataset"].isin(names), ["dataset", "gt_id", "t", "z", "y", "x"]].copy()
        gt_nodes = gt_nodes.reset_index(drop=True)
        for c in ("z", "y", "x"):
            gt_nodes[c] = gt_nodes[c].astype(np.float64)
        for a in ARMS:
            for c in ("z", "y", "x"):
                gt_nodes[f"{c}_{a}"] = gt_nodes[c].to_numpy(copy=True)
            gt_nodes[f"moved_{a}"] = False
        gt_nodes["hm_max"] = np.float32(0.0)
        gt_idx_by = gt_nodes.groupby(["dataset", "t"]).indices
        gcol = {c: gt_nodes.columns.get_loc(c) for c in gt_nodes.columns}

    hm_dir = None
    if args.save_heatmaps:
        hm_dir = out / "heatmaps"
        hm_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    n_frames = 0
    infer_s = 0.0
    idx_by = nodes.groupby(["dataset", "t"]).indices
    for ci, name in enumerate(names, 1):
        zarr_root = Path(args.data_dir) / f"{name}.zarr"
        if not zarr_root.exists():
            raise SystemExit(f"missing raw frames: {zarr_root}")
        ts = sorted({int(t) for (d, t) in idx_by if d == name})
        c0 = time.time()
        hm_stack = None
        for t in ts:
            rows = idx_by[(name, t)]
            frame = read_frame(zarr_root, t)
            i0 = time.time()
            hm = heatmap_for_frame(bundle, frame)
            infer_s += time.time() - i0
            n_frames += 1
            if hm_dir is not None:
                if hm_stack is None:
                    n_t = int(json.loads((zarr_root / "0" / "zarr.json").read_text())["shape"][0])
                    hm_stack = np.zeros((n_t,) + hm.shape, dtype=np.float16)
                hm_stack[t] = hm.astype(np.float16)
            res = refine_points(hm, nodes.iloc[rows][["z", "y", "x"]].to_numpy(), pool=pool)
            for a in ARMS:
                nodes.iloc[rows, col[f"z_{a}"]] = res[a][:, 0]
                nodes.iloc[rows, col[f"y_{a}"]] = res[a][:, 1]
                nodes.iloc[rows, col[f"x_{a}"]] = res[a][:, 2]
                nodes.iloc[rows, col[f"moved_{a}"]] = res[f"moved_{a}"]
            nodes.iloc[rows, col["hm_max"]] = res["hm_max"]
            if gt_nodes is not None and (name, t) in gt_idx_by:
                grows = gt_idx_by[(name, t)]
                gres = refine_points(hm, gt_nodes.iloc[grows][["z", "y", "x"]].to_numpy(), pool=pool)
                for a in ARMS:
                    gt_nodes.iloc[grows, gcol[f"z_{a}"]] = gres[a][:, 0]
                    gt_nodes.iloc[grows, gcol[f"y_{a}"]] = gres[a][:, 1]
                    gt_nodes.iloc[grows, gcol[f"x_{a}"]] = gres[a][:, 2]
                    gt_nodes.iloc[grows, gcol[f"moved_{a}"]] = gres[f"moved_{a}"]
                gt_nodes.iloc[grows, gcol["hm_max"]] = gres["hm_max"]
        if hm_dir is not None and hm_stack is not None:
            np.save(hm_dir / f"{name}.npy", hm_stack)
        sub = nodes.loc[nodes["dataset"] == name]
        print(f"  [{ci}/{len(names)}] {name} frames={len(ts)} nodes={len(sub)} "
              f"moved(com1)={float(sub['moved_com1'].mean()):.3f} "
              f"hm_max median={float(sub['hm_max'].median()):.3f} {time.time() - c0:.0f}s", flush=True)

    nodes.to_parquet(out / "refined_nodes.parquet")
    if gt_nodes is not None:
        gt_nodes.to_parquet(out / "gt_seeded.parquet")
    for a in ARMS:
        arm_df = df.copy()
        nm = arm_df["row_type"] == "node"
        arm_df.loc[nm, "z"] = nodes[f"z_{a}"].to_numpy()
        arm_df.loc[nm, "y"] = nodes[f"y_{a}"].to_numpy()
        arm_df.loc[nm, "x"] = nodes[f"x_{a}"].to_numpy()
        with gzip.open(out / f"refined_{a}.csv.gz", "wt", newline="") as fh:
            arm_df.to_csv(fh, index=False)
    timing = {
        "checkpoint": bundle["path"], "epoch": bundle["epoch"], "crops": names,
        "n_frames": n_frames, "n_nodes": int(len(nodes)), "wall_s": time.time() - t_start,
        "infer_s": infer_s, "threads": args.threads, "arms": list(ARMS),
        "pos_thresh": POS_THRESH, "max_shift_um": MAX_SHIFT_UM,
    }
    (out / "timing.json").write_text(json.dumps(timing, indent=2))
    print(f"wrote {out} ({n_frames} frames, {timing['wall_s']:.0f}s wall, {infer_s:.0f}s inference)")
    return 0


# --------------------------------------------------------------------------------------
# snap subcommand - acts on cached heatmaps, no model
# --------------------------------------------------------------------------------------
def write_arm_csvs(df: pd.DataFrame, nodes: pd.DataFrame, arms, out: Path) -> None:
    for a in arms:
        arm_df = df.copy()
        nm = arm_df["row_type"] == "node"
        arm_df.loc[nm, "z"] = nodes[f"z_{a}"].to_numpy()
        arm_df.loc[nm, "y"] = nodes[f"y_{a}"].to_numpy()
        arm_df.loc[nm, "x"] = nodes[f"x_{a}"].to_numpy()
        with gzip.open(out / f"refined_{a}.csv.gz", "wt", newline="") as fh:
            arm_df.to_csv(fh, index=False)


def cmd_snap(args) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hm_dir = Path(args.heatmap_dir)
    names = sorted(p.stem for p in hm_dir.glob("*.npy"))
    if not names:
        raise SystemExit(f"no cached heatmaps in {hm_dir}")
    df = pd.read_csv(Path(args.csv))
    df = df.loc[df["dataset"].isin(names)].reset_index(drop=True)
    for c in ("z", "y", "x"):
        df[c] = df[c].astype(np.float64)
    nodes = df.loc[df["row_type"] == "node", ["dataset", "node_id", "t", "z", "y", "x"]].copy()
    arms = list(SNAP_ARMS)
    for a in arms:
        for c in ("z", "y", "x"):
            nodes[f"{c}_{a}"] = nodes[c].to_numpy(copy=True)
        nodes[f"moved_{a}"] = False
    col = {c: nodes.columns.get_loc(c) for c in nodes.columns}
    idx_by = nodes.groupby(["dataset", "t"]).indices
    t_start = time.time()
    n_peaks_total = 0
    for ci, name in enumerate(names, 1):
        stack = np.load(hm_dir / f"{name}.npy", mmap_mode="r")
        ts = sorted({int(t) for (d, t) in idx_by if d == name})
        for t in ts:
            rows = idx_by[(name, t)]
            hm = np.asarray(stack[t], dtype=np.float32)
            res = snap_points(hm, nodes.iloc[rows][["z", "y", "x"]].to_numpy(), pool=args.pool,
                              thresh=args.peak_thresh)
            n_peaks_total += res["n_peaks"]
            for a in arms:
                nodes.iloc[rows, col[f"z_{a}"]] = res[a][:, 0]
                nodes.iloc[rows, col[f"y_{a}"]] = res[a][:, 1]
                nodes.iloc[rows, col[f"x_{a}"]] = res[a][:, 2]
                nodes.iloc[rows, col[f"moved_{a}"]] = res[f"moved_{a}"]
        sub = nodes.loc[nodes["dataset"] == name]
        print(f"  [{ci}/{len(names)}] {name} nodes={len(sub)} "
              f"moved(snap_r4)={float(sub['moved_snap_r4'].mean()):.3f}", flush=True)
    nodes.to_parquet(out / "refined_nodes.parquet")
    write_arm_csvs(df, nodes, arms, out)
    (out / "timing.json").write_text(json.dumps({
        "crops": names, "n_nodes": int(len(nodes)), "wall_s": time.time() - t_start, "arms": arms,
        "peak_thresh": args.peak_thresh, "n_peaks_total": n_peaks_total, "snap_arms": SNAP_ARMS}, indent=2))
    print(f"wrote {out} ({len(names)} crops, {time.time() - t_start:.0f}s)")
    return 0


# --------------------------------------------------------------------------------------
# eval subcommand - frozen-match residual (stage a)
# --------------------------------------------------------------------------------------
def residual_um(pred_zyx: np.ndarray, gt_zyx: np.ndarray) -> np.ndarray:
    return np.sqrt((((pred_zyx - gt_zyx) * VOXEL_SCALE_UM) ** 2).sum(axis=1))


def frozen_match_eval(ref: pd.DataFrame, atlas_nodes: pd.DataFrame, atlas_gt: pd.DataFrame) -> dict:
    """Stage (a): residual per arm on the atlas's own pred->GT matching, held fixed."""
    an = atlas_nodes[atlas_nodes["gt_id"] != -1][["dataset", "node_id", "gt_id"]]
    gpos = atlas_gt[["dataset", "gt_id", "z", "y", "x"]].rename(columns={"z": "gz", "y": "gy", "x": "gx"})
    m = ref.merge(an, on=["dataset", "node_id"], how="inner").merge(gpos, on=["dataset", "gt_id"], how="inner")
    gt = m[["gz", "gy", "gx"]].to_numpy(dtype=np.float64)
    base = residual_um(m[["z", "y", "x"]].to_numpy(dtype=np.float64), gt)

    # calibration gate: unrefined coordinates must reproduce the atlas residual on these crops
    a_full = atlas_nodes[atlas_nodes["dataset"].isin(m["dataset"].unique()) & (atlas_nodes["gt_id"] != -1)]
    a_full = a_full.merge(gpos, on=["dataset", "gt_id"], how="inner")
    a_base = residual_um(a_full[["z", "y", "x"]].to_numpy(dtype=np.float64),
                         a_full[["gz", "gy", "gx"]].to_numpy(dtype=np.float64))
    gate_ok = bool(len(base) == len(a_base) and abs(float(np.median(base)) - float(np.median(a_base))) < 1e-9)

    arms = [c[2:] for c in ref.columns if c.startswith("z_") and c[2:] and f"y_{c[2:]}" in ref.columns]
    result = {
        "arms_present": arms,
        "crops": sorted(m["dataset"].unique().tolist()), "n_matched": int(len(m)),
        "calibration_gate": {"ok": gate_ok, "n_here": int(len(base)), "n_atlas": int(len(a_base)),
                             "median_here": float(np.median(base)), "median_atlas": float(np.median(a_base))},
        "baseline": {"median_um": float(np.median(base)), "mean_um": float(base.mean()),
                     "p90_um": float(np.percentile(base, 90))},
        "arms": {},
    }
    result["baseline"]["frac_gt_3um"] = float((base > 3.0).mean())
    for a in arms:
        pred = m[[f"z_{a}", f"y_{a}", f"x_{a}"]].to_numpy(dtype=np.float64)
        r = residual_um(pred, gt)
        d = pred - gt
        result["arms"][a] = {
            "median_um": float(np.median(r)), "mean_um": float(r.mean()), "p90_um": float(np.percentile(r, 90)),
            "median_change_pct": float(100.0 * (np.median(r) / np.median(base) - 1.0)),
            "mean_change_pct": float(100.0 * (r.mean() / base.mean() - 1.0)),
            "frac_gt_3um": float((r > 3.0).mean()),
            "moved_frac": float(m[f"moved_{a}"].mean()) if f"moved_{a}" in m.columns else None,
            "signed_bias_vox_zyx": [float(v) for v in d.mean(axis=0)],
            "rms_um_zyx": [float(v) for v in np.sqrt(((d * VOXEL_SCALE_UM) ** 2).mean(axis=0))],
            "frac_within_1um": float((r <= 1.0).mean()), "frac_within_2um": float((r <= 2.0).mean()),
        }
    return result


def cmd_eval(args) -> int:
    ref = pd.read_parquet(args.refined)
    atlas = Path(args.atlas_dir)
    result = frozen_match_eval(ref, pd.read_parquet(atlas / f"nodes_{args.tag}.parquet"),
                               pd.read_parquet(atlas / f"gtnodes_{args.tag}.parquet"))
    result["tag"] = args.tag
    result["refined"] = str(args.refined)
    gt_seeded = Path(args.refined).with_name("gt_seeded.parquet")
    if gt_seeded.exists():
        gs = pd.read_parquet(gt_seeded)
        gt = gs[["z", "y", "x"]].to_numpy(dtype=np.float64)
        diag = {"n_gt": int(len(gs)), "hm_max_median": float(gs["hm_max"].median()),
                "frac_hm_max_ge_thresh": float((gs["hm_max"] >= POS_THRESH).mean())}
        for a in ARMS:
            r = residual_um(gs[[f"z_{a}", f"y_{a}", f"x_{a}"]].to_numpy(dtype=np.float64), gt)
            mv = gs[f"moved_{a}"].to_numpy()
            diag[a] = {"median_um_all": float(np.median(r)), "median_um_moved": float(np.median(r[mv])) if mv.any() else None,
                       "moved_frac": float(mv.mean())}
        result["gt_seeded_self_localisation"] = diag
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=2))
    g = result["calibration_gate"]
    print(f"[{args.tag}] crops={len(result['crops'])} matched={result['n_matched']} "
          f"calibration_gate={'OK' if g['ok'] else 'FAIL'} (median here {g['median_here']:.4f} "
          f"vs atlas {g['median_atlas']:.4f}; n {g['n_here']} vs {g['n_atlas']})")
    b = result["baseline"]
    print(f"  baseline  median {b['median_um']:.4f} um  mean {b['mean_um']:.4f}  p90 {b['p90_um']:.3f}  "
          f">3um {b['frac_gt_3um']:.3f}")
    for a in result["arms_present"]:
        r = result["arms"][a]
        mv = f"{r['moved_frac']:.3f}" if r["moved_frac"] is not None else "n/a"
        print(f"  {a:9s} median {r['median_um']:.4f} ({r['median_change_pct']:+.1f}%)  "
              f"mean {r['mean_um']:.4f} ({r['mean_change_pct']:+.1f}%)  >3um {r['frac_gt_3um']:.3f}  moved {mv}  "
              f"bias_vox {np.round(r['signed_bias_vox_zyx'], 3).tolist()}  "
              f"rms_um {np.round(r['rms_um_zyx'], 3).tolist()}")
    if "gt_seeded_self_localisation" in result:
        d = result["gt_seeded_self_localisation"]
        print(f"  GT-seeded self-localisation (n={d['n_gt']}, hm_max median {d['hm_max_median']:.3f}, "
              f"signal frac {d['frac_hm_max_ge_thresh']:.3f}): " +
              "  ".join(f"{a} {d[a]['median_um_moved'] if d[a]['median_um_moved'] is not None else float('nan'):.3f}um"
                        for a in ARMS))
    return 0 if g["ok"] else 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("refine")
    r.add_argument("--csv", required=True)
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--data-dir", default=str(ROOT / "data" / "train"))
    r.add_argument("--max-crops", type=int)
    r.add_argument("--crop-stride", type=int)
    r.add_argument("--crops", help="comma-separated explicit crop list")
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--gt-parquet", help="ea_atlas gtnodes_<tag>.parquet: also refine each GT node from its own "
                                        "position -> gt_seeded.parquet (heatmap self-localisation diagnostic)")
    r.add_argument("--save-heatmaps", action="store_true",
                   help="cache each crop's heatmap stack as float16 <out-dir>/heatmaps/<crop>.npy for `snap`")
    r.set_defaults(func=cmd_refine)
    ic = sp.add_parser("icentroid")
    ic.add_argument("--csv", required=True)
    ic.add_argument("--out-dir", required=True)
    ic.add_argument("--data-dir", default=str(ROOT / "data" / "train"))
    ic.add_argument("--max-crops", type=int)
    ic.add_argument("--crop-stride", type=int)
    ic.add_argument("--crops")
    ic.set_defaults(func=cmd_icentroid)
    sn = sp.add_parser("snap")
    sn.add_argument("--csv", required=True)
    sn.add_argument("--heatmap-dir", required=True, help="<refine out-dir>/heatmaps")
    sn.add_argument("--out-dir", required=True)
    sn.add_argument("--pool", type=int, default=4)
    sn.add_argument("--peak-thresh", type=float, default=PEAK_THRESH)
    sn.set_defaults(func=cmd_snap)
    e = sp.add_parser("eval")
    e.add_argument("--refined", required=True)
    e.add_argument("--atlas-dir", required=True)
    e.add_argument("--tag", required=True)
    e.add_argument("--json-out")
    e.set_defaults(func=cmd_eval)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
