"""M1 checkpoint selection — inference stage (separately budgeted GPU job).

Runs each retained candidate checkpoint over the 12-crop 44b6 INNER VALIDATION set only,
producing pre-graph (coords, edges) caches. Selection itself happens locally with the
pinned patched scorer after E0c greedy + faithful E0c wrapper.

Downstream is held at E0c to isolate the model change:
  * detector threshold 0.990   (E0c / oof_clean, NOT the failed v122 0.9690)
  * stock 4-view TTA           (reproduces oof_clean node counts exactly)
  * greedy selection           (max_parents=1, max_children=2 -- no ILP)

THE HELD-OUT 6bba FAMILY IS NOT TOUCHED HERE. It is evaluated exactly once, later, after a
single checkpoint has been chosen from this inner-validation evidence alone.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path("/kaggle/working/m1_select")
OUT.mkdir(parents=True, exist_ok=True)

DET_THRESHOLD = 0.990          # E0c operating point
TTA = "4view"
CANDIDATE_EPOCHS = [10, 15, 20, 25, 30]
FORBIDDEN_FAMILY = "6bba"
EXPECT_CONFIG_HASH = "fc7e4644ea37a90a"
EXPECT_AUG = "368908ecc44c0214"


def find_one(pattern: str) -> Path:
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    return Path(hits[0])


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


repo = find_one("/kaggle/input/**/scripts/predict_unet_transformer.py").parents[1]
data_dir = find_one("/kaggle/input/**/train")
m1_dir = find_one("/kaggle/input/**/m1_config.py").parent
ck_dir = find_one("/kaggle/input/**/m1_fold1_epoch10.pt").parent
print(f"repo={repo}\ndata={data_dir}\nm1={m1_dir}\nckpts={ck_dir}", flush=True)

sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "scripts"))
sys.path.insert(0, str(m1_dir))
os.environ.setdefault("BIOHUB_DATA_DIR", str(data_dir))
os.environ.setdefault("POLARS_PREFER_PKG", "32")

_SPECS = ["tracksdata", "zarr>=3.0.10,<4", "pyscipopt", "geff>=1.1.3.1.1", "geff-spec<1.2",
          "ilpy>=0.5.1", "polars>=1.36", "polars-runtime-32", "blosc2", "dask", "imagecodecs",
          "scikit-image>=0.24", "pyarrow", "rustworkx>=0.17.1", "sqlalchemy>=2",
          "numcodecs>=0.13,<0.16", "donfig>=0.8", "google-crc32c>=1.5", "bidict>=0.23.1",
          "psygnal>=0.14", "rich", "networkx>=3.2.1", "pydantic>=2.11", "pydantic-core",
          "annotated-types", "typing-extensions>=4.13", "typing-inspection", "markdown-it-py",
          "pygments", "click", "cloudpickle", "fsspec", "partd", "locket", "toolz", "pyyaml",
          "ndindex", "msgpack", "numexpr", "deprecated", "wrapt", "imageio", "pillow",
          "tifffile", "lazy-loader", "tqdm"]
_dirs = sorted({str(Path(w).parent)
                for w in glob.glob("/kaggle/input/**/wheels/*.whl", recursive=True)})
_cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q"]
for _d in _dirs:
    _cmd += ["--find-links", _d]
subprocess.check_call(_cmd + _SPECS)


def polars_ready() -> bool:
    try:
        import polars as _pl
        from polars._plr import PySeries as _P  # noqa: F401
        return hasattr(_pl, "Float16") and _pl.Series([1.0], dtype=_pl.Float64).dtype == _pl.Float64
    except Exception:
        return False


if not polars_ready():
    for m in [m for m in list(sys.modules) if m.startswith("polars")]:
        del sys.modules[m]
    subprocess.check_call(_cmd + ["--force-reinstall", "polars>=1.36", "polars-runtime-32"])
    for m in [m for m in list(sys.modules) if m.startswith("polars")]:
        del sys.modules[m]
    if not polars_ready():
        raise SystemExit("ABORT -- polars runtime unusable")
print("deps OK", flush=True)

import torch  # noqa: E402

import predict_unet_transformer as P  # noqa: E402

try:
    from tracking_cellmot.io import open_dataset
except ModuleNotFoundError:
    from biohub_tracking.io import open_dataset
import zarr  # noqa: E402

import m1_augment as M1A  # noqa: E402
from m1_config import config_hash  # noqa: E402

# ---------------------------------------------------------------- guards
manifest = json.loads((m1_dir / "val_manifests.json").read_text())
d1 = manifest["directions"]["1"]
VAL_CROPS = sorted(d1["inner_val_crops"])
if config_hash() != EXPECT_CONFIG_HASH or M1A.config_fingerprint() != EXPECT_AUG:
    raise SystemExit("ABORT -- config/augmentation hash mismatch")
leak = [c for c in VAL_CROPS if c.startswith(FORBIDDEN_FAMILY)]
if leak:
    raise SystemExit(f"ABORT -- held-out family in selection set: {leak}")
print(f"selection set: {len(VAL_CROPS)} inner-validation 44b6 crops; "
      f"zero {FORBIDDEN_FAMILY} paths", flush=True)
print(f"downstream held at E0c: det={DET_THRESHOLD}, {TTA} TTA, greedy (no ILP)", flush=True)

if not torch.cuda.is_available():
    raise SystemExit("ABORT -- no CUDA")
device = torch.device("cuda")
print("device:", torch.cuda.get_device_name(0), flush=True)


def load_m1_model(ck_path: Path):
    """Rebuild the architecture and load an M1 checkpoint's model weights."""
    import train_unet_transformer as T
    blob = torch.load(ck_path, map_location="cpu", weights_only=False)
    if blob.get("config_hash") != EXPECT_CONFIG_HASH:
        raise SystemExit(f"ABORT -- {ck_path.name} config hash {blob.get('config_hash')}")
    unet = T.TemporalUNet3D(in_channels=1, out_channels=32, layers=[32, 64, 128])
    m = T.UNetNodeTransformer(unet=unet, unet_out_channels=32,
                              pos_feat_dim=4 * T._POS_EMBED_DIM)
    m.load_state_dict(blob["model"])
    return m.to(device).eval(), blob


def infer_crop(model, crop: str, window_size: int = 2, downsample=(1, 4, 4)) -> dict:
    """Stock 4-view TTA detection + edge head at det=0.990, greedy candidate capping."""
    ds = open_dataset(data_dir / f"{crop}.zarr", normalize=False, load_image=False,
                      downsample=downsample)
    arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_lo, q_hi = float(ds.quantiles["0.001"]), float(ds.quantiles["0.999"])
    T_n = ds.image_shape[0]
    image_shape = (T_n,) + ds.image_shape[1:]
    target_shape = list(image_shape[1:])
    ds_arr = np.array(downsample, dtype=np.float32)
    ds_arr_t = torch.from_numpy(ds_arr).to(device)
    voxel = tuple(s * d for s, d in zip(ds.scale, downsample))
    pool_k = P.pool_kernel_from_um(P.PredictConfig().pool_kernel_um, voxel)
    W = window_size

    coord_lists, coord_offset, n_nodes, edges = [], {}, 0, []
    seen_f, seen_p = set(), set()
    stride = max(W - 1, 1)
    starts = list(range(0, T_n - W + 1, stride))
    if not starts or starts[-1] + W < T_n:
        starts.append(max(T_n - W, 0))

    for ws in starts:
        frames = list(range(ws, ws + W))
        imgs = torch.stack([P._load_frame(arr, t, target_shape, downsample) for t in frames])
        imgs = ((imgs - q_lo) / (q_hi - q_lo + 1e-6)).clamp(0.0).unsqueeze(0).to(device)
        with torch.inference_mode():
            unet_out, det = model.encode(imgs)
            acc = [d.clone() for d in det]
            for dims in [(-1,), (-2,), (-2, -1)]:      # stock 4-view TTA
                _, df = model.encode(imgs.flip(dims))
                for f in range(W):
                    acc[f] = acc[f] + df[f].flip(dims)
            det = [a / 4 for a in acc]

            for fi, t in enumerate(frames):
                if t in seen_f:
                    continue
                a = P._detect_cells_pooled(det[fi][0], t, DET_THRESHOLD, pool_k)
                coord_offset[t] = (n_nodes, n_nodes + len(a))
                n_nodes += len(a)
                coord_lists.append(a)
                seen_f.add(t)
            cs = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), np.int16)

            for fi in range(W - 1):
                ts, tt = frames[fi], frames[fi + 1]
                if (ts, tt) in seen_p:
                    continue
                seen_p.add((ts, tt))
                if ts not in coord_offset or tt not in coord_offset:
                    continue
                s0, s1 = coord_offset[ts]
                t0, t1 = coord_offset[tt]
                if s1 == s0 or t1 == t0:
                    continue
                c_s, c_t = cs[s0:s1], cs[t0:t1]
                n_s, n_t = len(c_s), len(c_t)
                pc_s = torch.from_numpy(c_s[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
                pc_t = torch.from_numpy(c_t[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
                win = (W,) + image_shape[1:]
                a_s, a_t = c_s.copy(), c_t.copy()
                a_s[:, 0], a_t[:, 0] = fi, fi + 1
                pp_s = torch.from_numpy(P.extract_pos_features(a_s, win)).unsqueeze(0).to(device)
                pp_t = torch.from_numpy(P.extract_pos_features(a_t, win)).unsqueeze(0).to(device)
                m_s = torch.ones(1, n_s, dtype=torch.bool, device=device)
                m_t = torch.ones(1, n_t, dtype=torch.bool, device=device)
                f_s = model._index_features(unet_out[:, fi], pc_s, m_s)
                f_t = model._index_features(unet_out[:, fi + 1], pc_t, m_t)
                logits = model.predict_edges(f_s, f_t, pc_s * ds_arr_t, pc_t * ds_arr_t,
                                             pp_s, pp_t, m_s, m_t)
                probs = torch.softmax(logits[0], dim=0).cpu().numpy()
                cand = sorted(((probs[i, j], i, j) for i in range(n_s) for j in range(n_t)
                               if probs[i, j] > P.PredictConfig().threshold), reverse=True)
                kids, pars = {}, {}
                is_ = np.arange(s0, s1, dtype=np.int64)
                it_ = np.arange(t0, t1, dtype=np.int64)
                for pr, i, j in cand:                 # greedy: <=2 children, <=1 parent
                    if kids.get(i, 0) >= 2 or pars.get(j, 0) >= 1:
                        continue
                    gi, gj = int(is_[i]), int(it_[j])
                    d_um = float(np.linalg.norm(cs[gi, 1:].astype(np.float32)
                                                - cs[gj, 1:].astype(np.float32)))
                    edges.append((gi, gj, float(pr), d_um))
                    kids[i] = kids.get(i, 0) + 1
                    pars[j] = pars.get(j, 0) + 1
        del unet_out, det, imgs

    coords = (np.concatenate(coord_lists) if coord_lists
              else np.empty((0, 4), np.int16)).astype(np.float32)
    coords[:, 1:] *= ds_arr
    return {"coords": coords.astype(np.int16),
            "edge_src": np.asarray([e[0] for e in edges], np.int64),
            "edge_tgt": np.asarray([e[1] for e in edges], np.int64),
            "edge_prob": np.asarray([e[2] for e in edges], np.float32),
            "edge_dist": np.asarray([e[3] for e in edges], np.float32)}


manifest_out = {"det_threshold": DET_THRESHOLD, "tta": TTA, "downstream": "E0c greedy (no ILP)",
                "val_crops": VAL_CROPS, "config_hash": config_hash(),
                "aug_fingerprint": M1A.config_fingerprint(),
                "gpu": torch.cuda.get_device_name(0), "results": []}
t_all = time.time()
for ep in CANDIDATE_EPOCHS:
    cks = sorted(glob.glob(f"/kaggle/input/**/m1_fold1_epoch{ep:02d}.pt", recursive=True))
    if not cks:
        print(f"  epoch {ep}: checkpoint not attached, skipping", flush=True)
        continue
    ck = Path(cks[0])
    model, blob = load_m1_model(ck)
    print(f"\n=== checkpoint epoch {ep} (step {blob['global_step']}, "
          f"sha {sha256(ck)[:16]}) ===", flush=True)
    dest = OUT / f"epoch{ep:02d}"
    dest.mkdir(parents=True, exist_ok=True)
    for crop in VAL_CROPS:
        t0 = time.time()
        out = infer_crop(model, crop)
        p = dest / f"{crop}__det-{DET_THRESHOLD:g}.npz"
        tmp = p.parent / (p.stem + ".tmp.npz")
        np.savez_compressed(tmp, **out)
        os.replace(tmp, p)
        manifest_out["results"].append({
            "epoch": ep, "crop": crop, "n_coords": int(out["coords"].shape[0]),
            "n_edges": int(out["edge_src"].shape[0]), "seconds": round(time.time() - t0, 1),
            "sha256": sha256(p), "checkpoint_sha256": sha256(ck)})
        print(f"  {crop}: {out['coords'].shape[0]} coords, {out['edge_src'].shape[0]} edges, "
              f"{time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()

manifest_out["total_hours"] = round((time.time() - t_all) / 3600, 2)
(OUT / "select_manifest.json").write_text(json.dumps(manifest_out, indent=2))
print(f"\nDONE in {manifest_out['total_hours']} h; "
      f"{len(manifest_out['results'])} (checkpoint, crop) caches written", flush=True)
print("Scoring and single-checkpoint selection happen LOCALLY with the pinned patched "
      "scorer. Held-out 6bba remains untouched.", flush=True)
