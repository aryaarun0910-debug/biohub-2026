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

# FULL POPULATION RUN. Preflight nulls drove three deliberate resource kills:
#   * C0 / det 0.96875 -- detector-level no-op vs 0.9690 (9 nodes on 44b6, 0 on 6bba)
#   * D-port           -- clean-903 wrapper drift immaterial (1-3 nodes, +/-0.0000)
#   * D4 TTA           -- not in the parity-proven baseline; would confound this gate
# Stock 4-view TTA is retained: it reproduces oof_clean node counts EXACTLY
# (28,119 / 6,847), so A and B differ only by detector threshold.
SHARD = 0
CACHE_THRESHOLDS = [0.9690]
PEAK_ONLY_THRESHOLD = None
TTA_SCHEMES = ["4view"]
SHARDS = {"0": [["44b6_12dfb391", 0], ["44b6_144b256d", 0], ["44b6_1574802b", 0], ["44b6_267148e4", 0], ["44b6_2f31fc2f", 0], ["44b6_3a861e03", 0], ["44b6_40c45f5a", 0], ["44b6_551a5dba", 0], ["44b6_5740d24b", 0], ["44b6_5f15d135", 0], ["44b6_668e0cc7", 0], ["44b6_706092f0", 0], ["44b6_7e557709", 0], ["44b6_81c256f0", 0], ["44b6_8cc6506c", 0], ["44b6_90724892", 0], ["44b6_95029e92", 0], ["44b6_97725144", 0], ["44b6_996155de", 0], ["44b6_9be80b04", 0], ["44b6_a2bb48bb", 0], ["44b6_aaf8b0ea", 0], ["44b6_abf82518", 0], ["44b6_c15fded2", 0], ["44b6_c50204e0", 0], ["44b6_c771cb04", 0], ["44b6_c8e2a523", 0], ["44b6_c96cfa10", 0], ["44b6_d2f34f90", 0], ["44b6_d754aa59", 0], ["44b6_db3c847b", 0], ["44b6_ddf577ad", 0], ["44b6_deabac95", 0], ["44b6_e29f0176", 0], ["44b6_e31261b4", 0], ["44b6_e35b117d", 0], ["44b6_eb2880fc", 0], ["44b6_f28707c6", 0], ["6bba_07e24132", 1], ["6bba_12665c0e", 1], ["6bba_15403b9a", 1], ["6bba_1f58c2f6", 1], ["6bba_23af9eeb", 1], ["6bba_2540cd90", 1], ["6bba_2819ca14", 1], ["6bba_337b1b3a", 1], ["6bba_372c8cb8", 1], ["6bba_3a1849c2", 1], ["6bba_3abfe10a", 1], ["6bba_3c5691b6", 1], ["6bba_3db54e20", 1], ["6bba_3fda6b25", 1], ["6bba_4aa14c89", 1], ["6bba_4ffd3da3", 1], ["6bba_55b7eebe", 1], ["6bba_55c70843", 1], ["6bba_57b7cc1e", 1], ["6bba_5c039895", 1], ["6bba_5f89039d", 1], ["6bba_61dd1e0d", 1], ["6bba_61ecbe65", 1], ["6bba_6321a359", 1], ["6bba_67ebd073", 1], ["6bba_6db0e9b4", 1], ["6bba_6feb10f0", 1], ["6bba_74686d6a", 1], ["6bba_76db78c1", 1], ["6bba_784a78c9", 1], ["6bba_789f8168", 1], ["6bba_7b5d3b2c", 1], ["6bba_7d3058ae", 1], ["6bba_80d12824", 1], ["6bba_91951b3a", 1], ["6bba_971fa5e0", 1], ["6bba_9a41d029", 1], ["6bba_9e23430b", 1], ["6bba_ab78413d", 1], ["6bba_ae82a791", 1], ["6bba_af149c94", 1], ["6bba_b1ae37b9", 1], ["6bba_b204cac7", 1], ["6bba_b329af44", 1], ["6bba_b693381b", 1], ["6bba_bbb708ca", 1], ["6bba_c73a1d11", 1], ["6bba_d1acb6ff", 1], ["6bba_d2b9fc0c", 1], ["6bba_d6ecebbb", 1], ["6bba_d82a4fc6", 1], ["6bba_df673a83", 1], ["6bba_e5e44988", 1], ["6bba_ebff6e76", 1], ["6bba_edf14583", 1], ["6bba_f17befbc", 1], ["6bba_f1fde7e0", 1], ["6bba_f20478e9", 1], ["6bba_f8ffd5e7", 1], ["6bba_fbc898dc", 1], ["6bba_fc5f39dc", 1], ["6bba_fc83837d", 1]], "1": [["44b6_0113de3b", 0], ["44b6_0b24845f", 0], ["44b6_0c582fdc", 0], ["44b6_0db75fae", 0], ["44b6_18ced818", 0], ["44b6_1d530831", 0], ["44b6_24264f12", 0], ["44b6_2a2eff9f", 0], ["44b6_33b596bf", 0], ["44b6_341df25f", 0], ["44b6_3bb3690f", 0], ["44b6_415c0a3a", 0], ["44b6_53f95252", 0], ["44b6_587a1e22", 0], ["44b6_66f9292d", 0], ["44b6_71a4179f", 0], ["44b6_74d0c52e", 0], ["44b6_7a302da0", 0], ["44b6_808952d6", 0], ["44b6_87bba6c4", 0], ["44b6_8f5ab931", 0], ["44b6_8f9ecab4", 0], ["44b6_949adeb1", 0], ["44b6_9bfa6a0a", 0], ["44b6_a21120c2", 0], ["44b6_b2c44266", 0], ["44b6_cf2536e8", 0], ["44b6_cf8fed6b", 0], ["44b6_d29c9ab2", 0], ["44b6_d5e7d891", 0], ["44b6_d78e09d9", 0], ["44b6_e28840c6", 0], ["44b6_e57ff5c6", 0], ["6bba_05b6850b", 1], ["6bba_05db0fb1", 1], ["6bba_062c8d37", 1], ["6bba_07477033", 1], ["6bba_085bf656", 1], ["6bba_09961292", 1], ["6bba_0c7fa718", 1], ["6bba_0e7c0d07", 1], ["6bba_13f19531", 1], ["6bba_1d0d8384", 1], ["6bba_1ebfb80d", 1], ["6bba_207c6aaf", 1], ["6bba_20852818", 1], ["6bba_2312ac41", 1], ["6bba_2646afc7", 1], ["6bba_268e1230", 1], ["6bba_283bf9f1", 1], ["6bba_312f0dc3", 1], ["6bba_32db13fc", 1], ["6bba_43fea39d", 1], ["6bba_474be664", 1], ["6bba_48816121", 1], ["6bba_4f99ce20", 1], ["6bba_5a4d9360", 1], ["6bba_5b28472a", 1], ["6bba_5c824876", 1], ["6bba_5dfe9ad1", 1], ["6bba_6479435d", 1], ["6bba_6ca87370", 1], ["6bba_6feeb0b1", 1], ["6bba_705ec2c9", 1], ["6bba_718b21f9", 1], ["6bba_767a1e17", 1], ["6bba_786893ac", 1], ["6bba_78a7bd97", 1], ["6bba_7af54fde", 1], ["6bba_7f87b3d8", 1], ["6bba_825bd1c6", 1], ["6bba_87289e13", 1], ["6bba_8b7818bf", 1], ["6bba_907271db", 1], ["6bba_96833384", 1], ["6bba_969618f6", 1], ["6bba_a5e926bb", 1], ["6bba_a90a0b9c", 1], ["6bba_acd782a8", 1], ["6bba_aeee7805", 1], ["6bba_afb141ff", 1], ["6bba_bb9f20c3", 1], ["6bba_c27cba08", 1], ["6bba_c328f2fd", 1], ["6bba_cdcfe533", 1], ["6bba_cf35214c", 1], ["6bba_cff5865f", 1], ["6bba_d0fc38b5", 1], ["6bba_d3da753b", 1], ["6bba_d5eae175", 1], ["6bba_debd7bfa", 1], ["6bba_e16ffc58", 1], ["6bba_ebdf3b34", 1], ["6bba_ed9377fd", 1], ["6bba_eebc57a5", 1], ["6bba_ef7b4f7e", 1], ["6bba_f4ae811c", 1], ["6bba_fc516dc6", 1], ["6bba_fe670320", 1]]}
CROPS = [c for c, _ in SHARDS[str(SHARD)]]


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

# Kaggle's July-2026 GPU image lacks zarr and tracksdata; predict_unet_transformer imports
# both at module level. Install ONLY what is genuinely missing, from the support pack's
# pinned wheels, with internet off.
#
# Do NOT blanket-install the whole wheels/ directory: that overwrites the image's numpy and
# breaks the preinstalled scipy's ABI (`cannot import name '_center' from numpy._core.umath`).
# This allowlist mirrors the deployed v122 kernel's PACKAGE_SPECS and deliberately contains
# no numpy/scipy/torch entry, so ABI-sensitive image packages are never touched.
# This mirrors the deployed v122 kernel's bootstrap, which is the only version proven to
# work on this image. Two properties matter and were the cause of three earlier failures:
#   * install by SPEC via --find-links, not by raw wheel path -- and the spec list contains
#     no numpy/scipy/torch, so ABI-sensitive image packages are never replaced. Blanket-
#     installing the 62 wheels DOES replace numpy 2.4.6 / scipy 1.18.0 and breaks the image
#     ("cannot import name '_center' from numpy._core.umath").
#   * install unconditionally rather than only-if-missing. The image ships an OLD polars;
#     skipping it while installing polars-runtime-32 yields "Polars binary is missing!" and
#     a missing pl.Float16 that tracksdata needs at import.
os.environ.setdefault("POLARS_PREFER_PKG", "32")

_SPECS = [
    "tracksdata", "zarr>=3.0.10,<4", "pyscipopt", "geff>=1.1.3.1.1", "geff-spec<1.2",
    "ilpy>=0.5.1", "polars>=1.36", "polars-runtime-32", "blosc2", "dask", "imagecodecs",
    "scikit-image>=0.24", "pyarrow", "rustworkx>=0.17.1", "sqlalchemy>=2",
    "numcodecs>=0.13,<0.16", "donfig>=0.8", "google-crc32c>=1.5", "bidict>=0.23.1",
    "psygnal>=0.14", "rich", "networkx>=3.2.1", "pydantic>=2.11", "pydantic-core",
    "annotated-types", "typing-extensions>=4.13", "typing-inspection", "markdown-it-py",
    "pygments", "click", "cloudpickle", "fsspec", "partd", "locket", "toolz", "pyyaml",
    "ndindex", "msgpack", "numexpr", "deprecated", "wrapt", "imageio", "pillow",
    "tifffile", "lazy-loader", "tqdm",
]
_wheel_dirs = sorted({str(Path(w).parent)
                      for w in glob.glob("/kaggle/input/**/wheels/*.whl", recursive=True)})
if not _wheel_dirs:
    raise FileNotFoundError("no offline wheels found under /kaggle/input/**/wheels/")

_cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q"]
for _d in _wheel_dirs:
    _cmd += ["--find-links", _d]
_cmd += _SPECS
subprocess.check_call(_cmd)
print(f"offline deps installed from {_wheel_dirs}", flush=True)

# Drop any stale pre-install imports before verifying.
for _root in ("polars", "zarr", "tracksdata", "numcodecs", "geff"):
    for _name in [n for n in list(sys.modules) if n == _root or n.startswith(_root + ".")]:
        del sys.modules[_name]

import numpy as _np_check  # noqa: E402
import scipy.spatial as _sp_check  # noqa: E402
import polars as _pl_check  # noqa: E402
import zarr as _zarr_check  # noqa: E402
import tracksdata as _td_check  # noqa: E402
assert hasattr(_pl_check, "Float16"), "polars too old for tracksdata"
print(f"dependency preflight OK: numpy {_np_check.__version__}, scipy {_sp_check.__name__}, "
      f"polars {_pl_check.__version__}, zarr {_zarr_check.__version__}, "
      f"tracksdata {getattr(_td_check, '__version__', '?')}", flush=True)

import predict_unet_transformer as P  # noqa: E402
import zarr  # noqa: E402

# The support pack names this package `biohub_tracking`; our local vendored checkout calls
# the same package `tracking_cellmot`. Accept either and record which one is live.
try:
    from tracking_cellmot.io import open_dataset  # noqa: E402
    import tracking_cellmot.io as inference_io  # noqa: E402
    INFERENCE_PACKAGE = "tracking_cellmot"
except ModuleNotFoundError:
    from biohub_tracking.io import open_dataset  # noqa: E402
    import biohub_tracking.io as inference_io  # noqa: E402
    INFERENCE_PACKAGE = "biohub_tracking"

import inspect as _inspect  # noqa: E402

print("\n" + "=" * 78)
print("!! SUPPORT PACK IS INFERENCE-ONLY !!")
print("  Its repository predates official metric patch 075fc5f (pack dated 2026-07-08).")
print("  It MAY produce coords/edges. It MUST NEVER score, validate or select an")
print("  experiment. Authoritative scoring happens locally with the pinned patched")
print("  scorer; nothing this kernel emits is a score.")
print("=" * 78)
print(f"INFERENCE_PACKAGE   : {INFERENCE_PACKAGE}")
print(f"inference_io.__file__: {inference_io.__file__}")
print(f"open_dataset sig     : {_inspect.signature(open_dataset)}")

# Provenance of the attached artifacts.
_pack_root = Path(str(repo)).parent
_pack_manifest = _pack_root / "ARTIFACT_MANIFEST.json"
if _pack_manifest.exists():
    _pm = json.loads(_pack_manifest.read_text())
    print(f"pack manifest        : {json.dumps(_pm)[:400]}")
else:
    print(f"pack manifest        : absent; repo mtime="
          f"{time.strftime('%Y-%m-%d', time.gmtime(Path(P.__file__).stat().st_mtime))}")
for _s in (0, 1):
    _wp = weights_root / f"edge_predictor_best_split_{_s}.pth"
    print(f"weight split_{_s} sha256 : {sha256(_wp)}  ({_wp.stat().st_size} bytes)")
print("pack weights          : NOT USED (LOEO fold weights come from the private dataset)")

# Scoring-module audit. `predict_unet_transformer` transitively imports the pack's
# `evaluate`/`metrics`, so they can be PRESENT; what matters is that this kernel never
# calls them and never emits a score. Assert the local authoritative scorer is absent.
_scoring_present = sorted(m for m in sys.modules
                          if "metric" in m.lower() or m.endswith("evaluate"))
print(f"scoring modules present (transitive, NEVER CALLED): {_scoring_present}")
assert "biotrack" not in sys.modules, "local authoritative scorer must not load here"
assert not any(m.startswith("biotrack") for m in sys.modules)
print("confirmed: no authoritative scoring module imported; kernel emits caches only")
print("=" * 78 + "\n", flush=True)

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
    torch.cuda.reset_peak_memory_stats()
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
    peaks_only = []
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
    # Serialised with numpy, NOT polars: the image's polars python package is present but
    # its compiled runtime is not loadable here ("Polars binary is missing!" ->
    # NameError: PyDataFrame). numpy is verified working, and this keeps the cache format
    # independent of the pack's dependency state. Inference is unaffected.
    records = {}
    for key, st in state.items():
        tta, thr = key
        coords = (np.concatenate(st["coord_lists"]) if st["coord_lists"]
                  else np.empty((0, 4), dtype=np.int16)).astype(np.float32)
        coords[:, 1:] *= ds_arr
        coords = coords.astype(np.int16)

        e = st["edges"]
        src = np.asarray([x[0] for x in e], dtype=np.int64)
        tgt = np.asarray([x[1] for x in e], dtype=np.int64)
        prob = np.asarray([x[2] for x in e], dtype=np.float32)
        dist = np.asarray([x[3] for x in e], dtype=np.float32)

        tag = f"{crop}__tta-{tta}__det-{thr:g}"
        dest = OUT / ("peaks" if thr == PEAK_ONLY_THRESHOLD else "cache")
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / f"{tag}.npz"
        _tmp = path.with_suffix(".npz.tmp")
        np.savez_compressed(_tmp, coords=coords, edge_src=src, edge_tgt=tgt,
                            edge_prob=prob, edge_dist=dist)
        os.replace(_tmp, path)
        records[tag] = {"crop": crop, "family": crop.split("_")[0], "tta": tta,
                        "det_threshold": thr, "split_used": split,
                        "n_coords": int(len(coords)), "n_edges": int(len(e)),
                        "output_sha256": sha256(path), "output_bytes": path.stat().st_size,
                        "path": str(path.relative_to(OUT))}
        print(f"  {tag}: {len(coords)} coords, {len(e)} edges", flush=True)

    return {"crop": crop, "split_used": split, "status": "ok",
            "runtime_s": round(time.time() - t_start, 1), "outputs": records,
            "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
            "shard": SHARD,
            "weights_sha256": sha256(weights_root / f"edge_predictor_best_split_{split}.pth"),
            "config_sha256": sha256(weights_root / f"config_split_{split}.json")}


# ----------------------------------------------------------------------------- run all
manifest = {
    "purpose": "coupled full-population pre-graph cache (det 0.9690, stock 4-view TTA)",
    "shard": SHARD, "n_crops": len(CROPS),
    "cache_thresholds": CACHE_THRESHOLDS, "tta_schemes": TTA_SCHEMES,
    "dropped_after_preflight": {
        "C0_det_0.96875": "detector-level no-op vs 0.9690",
        "D_port": "clean-903 wrapper drift immaterial",
        "D4_TTA": "not in the parity-proven baseline; would confound the gate"},
    "fold_routing": {"44b6": "split_0 (trained on 6bba)", "6bba": "split_1 (trained on 44b6)"},
    "repo": str(repo), "weights_root": str(weights_root), "data_dir": str(data_dir),
    "gpu": torch.cuda.get_device_name(0),
    "versions": {"torch": torch.__version__, "numpy": np.__version__,
                 "python": sys.version.split()[0]},
    "results": [],
}

STATUS = OUT / "status"
STATUS.mkdir(parents=True, exist_ok=True)
_t0 = time.time()

for _i, crop in enumerate(CROPS):
    _sp = STATUS / f"{crop}.json"
    if _sp.exists():
        _prev = json.loads(_sp.read_text())
        if _prev.get("status") == "ok":
            print(f"skip [{_i+1}/{len(CROPS)}] {crop} (already ok)", flush=True)
            manifest["results"].append(_prev)
            continue
    el = (time.time() - _t0) / 60
    print("")
    print(f"=== [{_i+1}/{len(CROPS)}] {crop} (split_{fold_for(crop)}) "
          f"elapsed={el:.1f}m ===", flush=True)
    try:
        _rec = run_crop(crop)
    except Exception as exc:  # noqa: BLE001
        import traceback
        _rec = {"crop": crop, "status": "failed", "shard": SHARD,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-3000:]}
        print(f"FAILED {crop}: {exc}", flush=True)
    _tmp = _sp.with_suffix(".json.tmp")
    _tmp.write_text(json.dumps(_rec))
    os.replace(_tmp, _sp)
    manifest["results"].append(_rec)

(OUT / f"manifest_shard{SHARD}.json").write_text(json.dumps(manifest, indent=2))
_ok = sum(1 for r in manifest["results"] if r.get("status") == "ok")
_bad = [r["crop"] for r in manifest["results"] if r.get("status") != "ok"]
print("")
print(f"SHARD {SHARD}: ok={_ok}/{len(CROPS)} failed={len(_bad)} "
      f"wall={(time.time()-_t0)/60:.1f}m", flush=True)
if _bad:
    print("FAILED CROPS (partial coverage -- do NOT score):", _bad, flush=True)
print("DONE", flush=True)
