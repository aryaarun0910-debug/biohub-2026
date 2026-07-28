"""Exact pre-graph (coords, edges) cache for the coupled C0/C1 decomposition — PREFLIGHT.

Caches the ONLY threshold-dependent stage (`predict_video`) so that every ILP-weight and
wrapper variant can be replayed exactly on CPU. Edge logits are NOT reusable across
detector thresholds: the edge head cross-attends over the whole node set and the source
axis is softmax-normalised, so both the attention context and the normaliser change with
the node population. One exact inference per (TTA scheme, detector threshold).

LOEO weight routing is explicit and mandatory:
    44b6_* crops -> split_0 checkpoint (trained on 6bba, never saw 44b6)
    6bba_* crops -> split_1 checkpoint (trained on 44b6, never saw 6bba)

TTA: E0c/oof_clean used the stock 4-view detector TTA; the deployed v122 kernel patches it
to spatial D4 (8 views). That is a fourth pipeline stage the decomposition has no arm for,
so this preflight emits BOTH schemes from the same 8 encodes (identity + 3 flips give the
4-view average; all 8 give D4). Cost is identical to running D4 alone.

Images only. The inference path never opens a .geff.
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
import torch

OUT = Path("/kaggle/working/coupled_cache")
OUT.mkdir(parents=True, exist_ok=True)

# Preregistered preflight crops: the same two used for Arm A parity.
CROPS = ["44b6_0113de3b", "6bba_05b6850b"]
CACHE_THRESHOLDS = [0.9690, 0.96875]   # full (coords, edges) caches
PEAK_ONLY_THRESHOLD = 0.990            # coords only, for the superset check
TTA_SCHEMES = ["4view", "d4"]


def find_one(pattern: str) -> Path:
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    return Path(hits[0])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- environment discovery
repo = find_one("/kaggle/input/**/scripts/predict_unet_transformer.py").parents[1]
weights_root = find_one("/kaggle/input/**/edge_predictor_best_split_0.pth").parent
data_dir = find_one("/kaggle/input/**/train")
print(f"repo={repo}\nweights={weights_root}\ndata={data_dir}", flush=True)

sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "scripts"))
os.environ.setdefault("BIOHUB_DATA_DIR", str(data_dir))

# Kaggle's July-2026 GPU image lacks zarr AND tracksdata (plus their transitive deps), and
# predict_unet_transformer imports both at module level. Install the support pack's full
# pinned offline wheel set in one pass; --no-index/--no-deps keeps internet off and pins
# exactly the versions the deployed kernels used (incl. tracksdata 0.1.0rc6.dev3+g980c2d30a).
_wheels = sorted(glob.glob("/kaggle/input/**/wheels/*.whl", recursive=True))
if not _wheels:
    raise FileNotFoundError("no offline wheels found under /kaggle/input/**/wheels/")
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "--no-index", "--no-deps", "-q", *_wheels])
print(f"offline wheels installed: {len(_wheels)}", flush=True)
for _m in ("zarr", "tracksdata", "polars"):
    __import__(_m)
print("dependency preflight OK: zarr, tracksdata, polars importable", flush=True)

import predict_unet_transformer as P  # noqa: E402
import zarr  # noqa: E402
from tracking_cellmot.io import open_dataset  # noqa: E402

device = torch.device("cuda")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable; refusing to run detector inference on CPU.")
print("device:", torch.cuda.get_device_name(0), flush=True)


def stage_weights(split: int) -> Path:
    """load_model() reads config.json beside the weights, so stage them together."""
    dst = Path(f"/kaggle/working/wts_split_{split}")
    dst.mkdir(parents=True, exist_ok=True)
    wsrc = weights_root / f"edge_predictor_best_split_{split}.pth"
    csrc = weights_root / f"config_split_{split}.json"
    (dst / "edge_predictor.pth").write_bytes(wsrc.read_bytes())
    (dst / "config.json").write_bytes(csrc.read_bytes())
    return dst / "edge_predictor.pth"


def fold_for(crop: str) -> int:
    """LOEO routing: score each crop with the model that never saw its family."""
    fam = crop.split("_")[0]
    if fam == "44b6":
        return 0   # split_0 trained on 6bba
    if fam == "6bba":
        return 1   # split_1 trained on 44b6
    raise ValueError(f"unknown family for crop {crop}")


# --------------------------------------------------------------------------- TTA encode
def encode_with_tta(model, imgs, W):
    """Return {'4view': det_logits, 'd4': det_logits} plus the shared unet_out.

    The 4-view average is identity + 3 flips (stock predict_video). D4 additionally uses
    rot90 k=1,3, transpose and rot90-then-transpose, matching the deployed v122 patch
    verbatim, and divides by 8.
    """
    unet_out, det_base = model.encode(imgs)
    acc = [d.clone() for d in det_base]          # running sum, starts at identity
    n_views = 1

    for dims in [(-1,), (-2,), (-2, -1)]:
        _, det_f = model.encode(imgs.flip(dims))
        for f in range(W):
            acc[f] = acc[f] + det_f[f].flip(dims)
        n_views += 1
    four = [a / n_views for a in acc]            # exactly the stock 4-view mean

    for k in (1, 3):
        _, det_r = model.encode(torch.rot90(imgs, k, dims=(-2, -1)))
        for f in range(W):
            acc[f] = acc[f] + torch.rot90(det_r[f], -k, dims=(-2, -1))
        n_views += 1
    _, det_t = model.encode(imgs.transpose(-1, -2))
    for f in range(W):
        acc[f] = acc[f] + det_t[f].transpose(-1, -2)
    n_views += 1
    _, det_at = model.encode(torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2))
    for f in range(W):
        acc[f] = acc[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims=(-2, -1))
    n_views += 1
    assert n_views == 8, n_views
    d4 = [a / n_views for a in acc]

    return unet_out, {"4view": four, "d4": d4}


# ------------------------------------------------------------------------------ one crop
def run_crop(crop: str) -> dict:
    t_start = time.time()
    split = fold_for(crop)
    weights_path = stage_weights(split)
    model, window_size, downsample = P.load_model(weights_path, device)
    model.eval()

    ds_path = Path(data_dir) / f"{crop}.zarr"
    ds = open_dataset(ds_path, normalize=False, load_image=False, downsample=tuple(downsample))
    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low, q_high = float(ds.quantiles["0.001"]), float(ds.quantiles["0.999"])

    W = window_size
    T = ds.image_shape[0]
    image_shape = (T,) + ds.image_shape[1:]
    target_shape = list(image_shape[1:])
    ds_arr = np.array(downsample, dtype=np.float32)
    ds_arr_t = torch.from_numpy(ds_arr).to(device)
    voxel_size = tuple(s * d for s, d in zip(ds.scale, downsample))
    pool_k = P.pool_kernel_from_um(P.PredictConfig().pool_kernel_um, voxel_size)

    # One independent registry per (tta, threshold) combination.
    combos = [(t, thr) for t in TTA_SCHEMES for thr in CACHE_THRESHOLDS]
    peaks_only = [(t, PEAK_ONLY_THRESHOLD) for t in TTA_SCHEMES]
    state = {k: {"coord_lists": [], "coord_offset": {}, "n": 0, "edges": [],
                 "seen_f": set(), "seen_p": set()} for k in combos + peaks_only}

    stride = max(W - 1, 1)
    starts = list(range(0, T - W + 1, stride))
    if not starts or starts[-1] + W < T:
        last = max(T - W, 0)
        if not starts or last != starts[-1]:
            starts.append(last)

    for ws in starts:
        frames = list(range(ws, ws + W))
        imgs = torch.stack([P._load_frame(zarr_arr, t, target_shape, tuple(downsample))
                            for t in frames])
        imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0).unsqueeze(0).to(device)

        with torch.inference_mode():
            unet_out, det_by_tta = encode_with_tta(model, imgs, W)

            # ---- detection per combo (dedup frames across overlapping windows)
            for key in state:
                tta, thr = key
                st = state[key]
                for f_idx, t in enumerate(frames):
                    if t in st["seen_f"]:
                        continue
                    arr = P._detect_cells_pooled(det_by_tta[tta][f_idx][0], t, thr, pool_k)
                    st["coord_offset"][t] = (st["n"], st["n"] + len(arr))
                    st["n"] += len(arr)
                    st["coord_lists"].append(arr)
                    st["seen_f"].add(t)

            # ---- edge inference per cache combo (peaks-only combos skip this)
            for key in combos:
                st = state[key]
                coords_so_far = (np.concatenate(st["coord_lists"]) if st["coord_lists"]
                                 else np.empty((0, 4), dtype=np.int16))
                for f_idx in range(W - 1):
                    t_src, t_tgt = frames[f_idx], frames[f_idx + 1]
                    if (t_src, t_tgt) in st["seen_p"]:
                        continue
                    st["seen_p"].add((t_src, t_tgt))
                    if t_src not in st["coord_offset"] or t_tgt not in st["coord_offset"]:
                        continue
                    s_s, e_s = st["coord_offset"][t_src]
                    s_t, e_t = st["coord_offset"][t_tgt]
                    if e_s == s_s or e_t == s_t:
                        continue
                    c_src, c_tgt = coords_so_far[s_s:e_s], coords_so_far[s_t:e_t]
                    n_src, n_tgt = len(c_src), len(c_tgt)

                    pc_s = torch.from_numpy(c_src[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
                    pc_t = torch.from_numpy(c_tgt[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
                    win_shape = (W,) + image_shape[1:]
                    cs, ct = c_src.copy(), c_tgt.copy()
                    cs[:, 0], ct[:, 0] = f_idx, f_idx + 1
                    pp_s = torch.from_numpy(P.extract_pos_features(cs, win_shape)).unsqueeze(0).to(device)
                    pp_t = torch.from_numpy(P.extract_pos_features(ct, win_shape)).unsqueeze(0).to(device)
                    m_s = torch.ones(1, n_src, dtype=torch.bool, device=device)
                    m_t = torch.ones(1, n_tgt, dtype=torch.bool, device=device)

                    uf_s = model._index_features(unet_out[:, f_idx], pc_s, m_s)
                    uf_t = model._index_features(unet_out[:, f_idx + 1], pc_t, m_t)
                    logits = model.predict_edges(uf_s, uf_t, pc_s * ds_arr_t, pc_t * ds_arr_t,
                                                 pp_s, pp_t, m_s, m_t)
                    probs = torch.softmax(logits[0], dim=0).cpu().numpy()

                    cand = sorted(((probs[i, j], i, j)
                                   for i in range(n_src) for j in range(n_tgt)
                                   if probs[i, j] > P.PredictConfig().threshold), reverse=True)
                    kids: dict[int, int] = {}
                    pars: dict[int, int] = {}
                    idx_s = np.arange(s_s, e_s, dtype=np.int64)
                    idx_t = np.arange(s_t, e_t, dtype=np.int64)
                    for prob, i, j in cand:
                        if kids.get(i, 0) >= 2 or pars.get(j, 0) >= 1:
                            continue
                        gi, gj = int(idx_s[i]), int(idx_t[j])
                        dist = float(np.linalg.norm(
                            coords_so_far[gi, 1:].astype(np.float32)
                            - coords_so_far[gj, 1:].astype(np.float32)))
                        st["edges"].append((gi, gj, float(prob), dist))
                        kids[i] = kids.get(i, 0) + 1
                        pars[j] = pars.get(j, 0) + 1
        del unet_out, det_by_tta, imgs

    # ------------------------------------------------------------------ write per combo
    import polars as pl
    records = {}
    for key, st in state.items():
        tta, thr = key
        coords = (np.concatenate(st["coord_lists"]) if st["coord_lists"]
                  else np.empty((0, 4), dtype=np.int16)).astype(np.float32)
        coords[:, 1:] *= ds_arr
        coords = coords.astype(np.int16)

        rows = [{"row_type": "node", "node_id": i, "t": int(c[0]), "z": float(c[1]),
                 "y": float(c[2]), "x": float(c[3]), "source_id": -1, "target_id": -1,
                 "edge_prob": 0.0, "edge_dist": 0.0} for i, c in enumerate(coords)]
        rows += [{"row_type": "edge", "node_id": -1, "t": -1, "z": 0.0, "y": 0.0, "x": 0.0,
                  "source_id": int(s), "target_id": int(t), "edge_prob": float(p),
                  "edge_dist": float(d)} for s, t, p, d in st["edges"]]

        tag = f"{crop}__tta-{tta}__det-{thr:g}"
        dest = OUT / ("peaks" if thr == PEAK_ONLY_THRESHOLD else "cache")
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / f"{tag}.parquet"
        pl.DataFrame(rows).write_parquet(path)
        records[tag] = {"crop": crop, "family": crop.split("_")[0], "tta": tta,
                        "det_threshold": thr, "split_used": split,
                        "n_coords": int(len(coords)), "n_edges": int(len(st["edges"])),
                        "output_sha256": sha256(path), "output_bytes": path.stat().st_size,
                        "path": str(path.relative_to(OUT))}
        print(f"  {tag}: {len(coords)} coords, {len(st['edges'])} edges", flush=True)

    return {"crop": crop, "split_used": split, "status": "ok",
            "runtime_s": round(time.time() - t_start, 1), "outputs": records,
            "weights_sha256": sha256(weights_root / f"edge_predictor_best_split_{split}.pth"),
            "config_sha256": sha256(weights_root / f"config_split_{split}.json")}


# ----------------------------------------------------------------------------- run all
manifest = {
    "purpose": "coupled C0/C1 preflight pre-graph cache (2 crops, both TTA schemes)",
    "crops": CROPS, "cache_thresholds": CACHE_THRESHOLDS,
    "peak_only_threshold": PEAK_ONLY_THRESHOLD, "tta_schemes": TTA_SCHEMES,
    "fold_routing": {"44b6": "split_0 (trained on 6bba)", "6bba": "split_1 (trained on 44b6)"},
    "repo": str(repo), "weights_root": str(weights_root), "data_dir": str(data_dir),
    "gpu": torch.cuda.get_device_name(0),
    "versions": {"torch": torch.__version__, "numpy": np.__version__,
                 "python": sys.version.split()[0]},
    "results": [],
}
for crop in CROPS:
    print(f"\n=== {crop} (split_{fold_for(crop)}) ===", flush=True)
    try:
        manifest["results"].append(run_crop(crop))
    except Exception as exc:  # noqa: BLE001
        import traceback
        manifest["results"].append({"crop": crop, "status": "failed",
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc()[-3000:]})
        print(f"FAILED {crop}: {exc}", flush=True)

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("\n" + json.dumps({r["crop"]: r.get("status") for r in manifest["results"]}, indent=2))
print("DONE", flush=True)
