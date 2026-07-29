"""M1 fold-1 ONE-EPOCH SMOKE — measurement only, no held-out access, no full training.

Measures the real fold-1 training path so the 24,000-step budget can be projected from
MEASURED backward-step time rather than extrapolated from inference cost (an earlier
extrapolation was wrong by ~4x because inference runs at full 256^2 while training runs
downsampled 4x in y/x).

Budget matches the baseline OOF model exactly so the training change stays the only
variable: batch 1, 800 iters, LR 1e-4. This smoke runs ONE epoch = 800 optimizer steps.

SAFETY (aborts before any training if violated):
  * held-out family 6bba is not reachable -- the loader is restricted to the manifest's
    train crops, and a guard asserts no 6bba path is ever constructed;
  * seed, augmentation fingerprint, config hash and manifest hash must match the committed
    values;
  * startup self-tests cover determinism, augmentation and checkpoint round-trip.
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

OUT = Path("/kaggle/working/m1_smoke")
OUT.mkdir(parents=True, exist_ok=True)

# ---- values pinned from the committed repo; mismatch aborts before training -------------
EXPECT = {
    "seed": 20260729,
    "aug_fingerprint": "368908ecc44c0214",
    "config_hash": "fc7e4644ea37a90a",
    "direction": 1, "train_family": "44b6", "held_out_family": "6bba",
    "batch_size": 1, "max_iters": 800, "lr": 1e-4,
}
FORBIDDEN_FAMILY = "6bba"


def find_one(pattern: str) -> Path:
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    return Path(hits[0])


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


repo = find_one("/kaggle/input/**/scripts/train_unet_transformer.py").parents[1]
data_dir = find_one("/kaggle/input/**/train")
m1_dir = find_one("/kaggle/input/**/m1_config.py").parent
print(f"repo={repo}\ndata={data_dir}\nm1={m1_dir}", flush=True)

sys.path.insert(0, str(repo / "src"))
sys.path.insert(0, str(repo / "scripts"))
sys.path.insert(0, str(m1_dir))
os.environ.setdefault("BIOHUB_DATA_DIR", str(data_dir))

# ---- proven offline dependency bootstrap (spec-based; never replaces numpy/scipy) -------
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
_dirs = sorted({str(Path(w).parent) for w in
                glob.glob("/kaggle/input/**/wheels/*.whl", recursive=True)})
if not _dirs:
    raise FileNotFoundError("no offline wheels found")
_cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q"]
for _d in _dirs:
    _cmd += ["--find-links", _d]
subprocess.check_call(_cmd + _SPECS)


def polars_runtime_ready() -> bool:
    """FUNCTIONAL polars check, not just importability.

    The image ships a polars whose compiled backend does not load: `import polars` succeeds
    but `polars._plr.PySeries` is undefined, so any real Series/DataFrame construction dies
    with `NameError: PySeries`. That is invisible to an import test and only surfaced when
    tracksdata tried to read a GT geff -- i.e. minutes into the run. This mirrors the
    deployed v122 kernel's own readiness probe.
    """
    try:
        import polars as _pl
        from polars._plr import PySeries as _PySeries  # noqa: F401
        return (hasattr(_pl, "Float16")
                and _pl.Series([-999999.0], dtype=_pl.Float64).dtype == _pl.Float64)
    except Exception:
        return False


if not polars_runtime_ready():
    print("polars runtime NOT ready -> force-reinstalling polars + runtime", flush=True)
    for _mod in [m for m in list(sys.modules) if m == "polars" or m.startswith("polars.")]:
        del sys.modules[_mod]
    subprocess.check_call(_cmd + ["--force-reinstall", "polars>=1.36", "polars-runtime-32"])
    for _mod in [m for m in list(sys.modules) if m == "polars" or m.startswith("polars.")]:
        del sys.modules[_mod]
    if not polars_runtime_ready():
        raise SystemExit("ABORT -- polars runtime still unusable after force-reinstall; "
                         "tracksdata cannot read GT geffs without it")
    print("polars runtime repaired", flush=True)

for _m in ("numpy", "scipy.spatial", "polars", "zarr", "tracksdata"):
    __import__(_m)
import polars as _pl_check  # noqa: E402
assert _pl_check.DataFrame({"a": [1.0, 2.0]}).height == 2, "polars DataFrame unusable"
print(f"deps OK (polars {_pl_check.__version__} functional)", flush=True)

import torch  # noqa: E402
import m1_augment as M1A  # noqa: E402
import m1_determinism as DT  # noqa: E402
from m1_config import M1_CONFIG, config_hash  # noqa: E402

# ---------------------------------------------------------------- hash / safety guards
manifest = json.loads((m1_dir / "val_manifests.json").read_text())
d1 = manifest["directions"]["1"]
checks = {
    "config_hash": (config_hash(), EXPECT["config_hash"]),
    "aug_fingerprint": (M1A.config_fingerprint(), EXPECT["aug_fingerprint"]),
    "seed": (M1_CONFIG["seed"], EXPECT["seed"]),
    "batch_size": (M1_CONFIG["batch_size"], EXPECT["batch_size"]),
    "max_iters": (M1_CONFIG["max_iters_per_epoch"], EXPECT["max_iters"]),
    "lr": (M1_CONFIG["lr"], EXPECT["lr"]),
    "train_family": (d1["train_family"], EXPECT["train_family"]),
    "held_out_family": (d1["held_out_family"], EXPECT["held_out_family"]),
}
bad = {k: v for k, v in checks.items() if v[0] != v[1]}
for k, (got, want) in checks.items():
    print(f"  guard {k:<18} got={got!r} want={want!r} {'OK' if got == want else 'MISMATCH'}")
if bad:
    raise SystemExit(f"ABORT before training -- hash/config mismatch: {bad}")

TRAIN_CROPS = list(d1["train_crops"])
VAL_CROPS = list(d1["inner_val_crops"])
leak = [c for c in TRAIN_CROPS + VAL_CROPS if c.startswith(FORBIDDEN_FAMILY)]
if leak:
    raise SystemExit(f"ABORT -- held-out family reachable in loader: {leak[:5]}")
print(f"  guard held-out isolation: 0 {FORBIDDEN_FAMILY} crops in loader  OK")
print(f"  train crops={len(TRAIN_CROPS)}  inner-val (NOT used here)={len(VAL_CROPS)}",
      flush=True)

# ---------------------------------------------------------------- startup self-tests
det_info = DT.seed_everything(EXPECT["seed"])
print("  determinism:", json.dumps(det_info), flush=True)
_a = M1A.apply_augmentations(np.random.default_rng(0).random((2, 4, 8, 8), dtype=np.float32),
                             DT.sample_rng(EXPECT["seed"], 0, 0))
_b = M1A.apply_augmentations(np.random.default_rng(0).random((2, 4, 8, 8), dtype=np.float32),
                             DT.sample_rng(EXPECT["seed"], 0, 0))
assert np.array_equal(_a, _b), "augmentation not reproducible"
_ck = OUT / "_probe.pt"
torch.save({"x": torch.zeros(3)}, _ck)
assert torch.load(_ck, weights_only=True)["x"].shape == (3,)
_ck.unlink()
print("  self-tests OK: determinism, augmentation reproducibility, checkpoint round-trip",
      flush=True)

if not torch.cuda.is_available():
    raise SystemExit("ABORT -- no CUDA")
print("device:", torch.cuda.get_device_name(0), flush=True)

# ------------------------------------------------- explicit train/val disjointness report
VAL_SET, TRAIN_SET = set(VAL_CROPS), set(TRAIN_CROPS)
inter = sorted(TRAIN_SET & VAL_SET)
print("")
print("=== TRAIN / INNER-VALIDATION SETS (direction 1) ===")
print(f"  train crops ({len(TRAIN_SET)}): {sorted(TRAIN_SET)}")
print(f"  inner-val crops ({len(VAL_SET)}): {sorted(VAL_SET)}")
print(f"  INTERSECTION: {inter}  -> {'EMPTY (correct)' if not inter else 'LEAK!'}")
if inter:
    raise SystemExit(f"ABORT -- validation crops present in training loader: {inter}")
allc = sorted(TRAIN_SET | VAL_SET)
print(f"  union={len(allc)} (train family total)")
bad = [c for c in allc if not c.startswith("44b6")]
if bad:
    raise SystemExit(f"ABORT -- non-train-family crop in scope: {bad[:5]}")
print(f"  every crop in scope is 44b6; zero {FORBIDDEN_FAMILY} paths constructible  OK")

# ------------------------------------------------- patch trainer (determinism + telemetry)
import shutil  # noqa: E402

import m1_driver as MD  # noqa: E402

work_repo = Path("/kaggle/working/repo")
work_repo.mkdir(parents=True, exist_ok=True)
if not (work_repo / "scripts").exists():
    shutil.copytree(repo / "scripts", work_repo / "scripts")
    shutil.copytree(repo / "src", work_repo / "src")
tgt = work_repo / "scripts" / "train_unet_transformer.py"
tgt.write_text(MD.patch_trainer_source(tgt.read_text(), EXPECT["seed"], M1A))
print("trainer patched: seeded per-sample RNG + per-step telemetry", flush=True)
sys.path.insert(0, str(work_repo / "src"))
sys.path.insert(0, str(work_repo / "scripts"))
sys.path.insert(0, str(m1_dir))

from torch.utils.data import DataLoader  # noqa: E402

import train_unet_transformer as T  # noqa: E402

device = torch.device("cuda")
manifest_sha = sha256(m1_dir / "val_manifests.json")

# ------------------------------------------------- construct exactly as train() does
t_index = time.time()
train_files = [data_dir / c for c in sorted(TRAIN_SET)]
video_data = []
for f in train_files:
    vm, w = T.load_dataset_windows(f, window_size=2, max_frames=None, downsample=(1, 4, 4))
    video_data.append((vm, w))
n_windows = sum(len(w) for _, w in video_data)
max_nodes = max(max(w.node_counts) for _, ws in video_data for w in ws)
index_s = time.time() - t_index
print(f"indexed {len(train_files)} crops -> {n_windows} windows, max_nodes={max_nodes}, "
      f"{index_s:.1f}s", flush=True)


def build(seed):
    DT.seed_everything(seed)
    ds = T.FrameWindowDataset(video_data, max_nodes=max_nodes,
                              augmentations=[T.flip_augment, M1A.as_trainer_augmentation()])
    g = torch.Generator()
    g.manual_seed(seed)
    ld = DataLoader(ds, batch_size=EXPECT["batch_size"], shuffle=True, num_workers=2,
                    prefetch_factor=2, persistent_workers=True, pin_memory=False,
                    generator=g, worker_init_fn=DT.worker_init_fn(seed))
    unet = T.TemporalUNet3D(in_channels=1, out_channels=32, layers=[32, 64, 128])
    m = T.UNetNodeTransformer(unet=unet, unet_out_channels=32,
                              pos_feat_dim=4 * T._POS_EMBED_DIM).to(device)
    o = torch.optim.AdamW(m.parameters(), lr=EXPECT["lr"])
    return ds, ld, m, o


# ------------------------------------------------- MAIN: one epoch = 800 steps
ds, loader, model, opt = build(EXPECT["seed"])
init_hash = DT.state_hash(model)
MD.STEP_LOG.clear()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
edge_l, det_l = T.train_epoch(model, loader, opt, device, max_iters=EXPECT["max_iters"])
epoch_s = time.time() - t0
peak_gpu = torch.cuda.max_memory_allocated() / 1e6
stats = MD.step_stats(MD.STEP_LOG)
ck = OUT / "m1_fold1_epoch1.pt"
ck_sha = MD.save_checkpoint(ck, model=model, optimizer=opt, epoch=1,
                            global_step=EXPECT["max_iters"], seed=EXPECT["seed"],
                            config_hash=config_hash(),
                            aug_fingerprint=M1A.config_fingerprint(),
                            manifest_sha=manifest_sha, torch_mod=torch)
print(f"epoch done: {epoch_s:.1f}s edge={edge_l:.4f} det={det_l:.4f}", flush=True)

# ------------------------------------------------- OPERATIONAL RESUME TEST
N = 40
print("")
print(f"=== OPERATIONAL RESUME TEST: control {N} steps vs {N // 2}+resume+{N // 2} ===",
      flush=True)
_, ldC, mC, oC = build(4242)
MD.STEP_LOG.clear()
T.train_epoch(mC, ldC, oC, device, max_iters=N)
ctrl_losses = [r["edge_loss"] for r in MD.STEP_LOG]
ctrl_hash = DT.state_hash(mC)

_, ldI, mI, oI = build(4242)
MD.STEP_LOG.clear()
T.train_epoch(mI, ldI, oI, device, max_iters=N // 2)
first_losses = [r["edge_loss"] for r in MD.STEP_LOG]
mid = OUT / "_resume_probe.pt"
MD.save_checkpoint(mid, model=mI, optimizer=oI, epoch=1, global_step=N // 2, seed=4242,
                   config_hash=config_hash(), aug_fingerprint=M1A.config_fingerprint(),
                   manifest_sha=manifest_sha, torch_mod=torch)

_, ldR, mR, oR = build(4242)
restored = MD.load_checkpoint(mid, model=mR, optimizer=oR, torch_mod=torch,
                              expect_config_hash=config_hash())
resume_weights_match = DT.state_hash(mR) == DT.state_hash(mI)
MD.STEP_LOG.clear()
T.train_epoch(mR, ldR, oR, device, max_iters=N // 2)
second_losses = [r["edge_loss"] for r in MD.STEP_LOG]
resumed = first_losses + second_losses
final_match = DT.state_hash(mR) == ctrl_hash
max_dev = max(abs(a - b) for a, b in zip(ctrl_losses, resumed)) if ctrl_losses else float("nan")

resume_report = {
    "counters_restored": {"epoch": restored["epoch"], "global_step": restored["global_step"],
                          "aug_epoch": restored["aug_epoch"]},
    "config_hash_verified": restored["config_hash"] == config_hash(),
    "manifest_sha_verified": restored["manifest_sha256"] == manifest_sha,
    "optimizer_state_restored": bool(oR.state_dict()["state"]),
    "model_weights_match_at_resume_point": resume_weights_match,
    "control_steps": len(ctrl_losses), "resumed_steps": len(resumed),
    "max_abs_loss_deviation_vs_control": max_dev,
    "final_weights_match_control": final_match,
    "note": ("Bitwise GPU parity is NOT claimed: some cuDNN 3D-conv backward kernels have "
             "no deterministic implementation, so use_deterministic_algorithms runs "
             "warn_only. Counter, optimizer and RNG restoration are exact; the loss "
             "deviation quantifies residual kernel nondeterminism."),
}
print(json.dumps(resume_report, indent=2, default=str), flush=True)

per_step = epoch_s / EXPECT["max_iters"]
full = per_step * 24000
report = {
    "train_crops": sorted(TRAIN_SET), "inner_val_crops": sorted(VAL_SET),
    "train_val_intersection": inter,
    "n_train_crops": len(TRAIN_SET), "n_inner_val_crops": len(VAL_SET),
    "windows_indexed": n_windows, "max_nodes": max_nodes,
    "index_seconds": round(index_s, 1),
    "optimizer_steps": EXPECT["max_iters"], "batch_size": EXPECT["batch_size"],
    "lr": EXPECT["lr"], "epoch_wall_seconds": round(epoch_s, 1),
    "avg_edge_loss": float(edge_l), "avg_det_loss": float(det_l),
    "peak_gpu_mb": round(peak_gpu, 1),
    "checkpoint_bytes": ck.stat().st_size, "checkpoint_sha256": ck_sha,
    "init_state_hash": init_hash,
    "aug_fingerprint": M1A.config_fingerprint(), "aug_ranges": M1A.DEFAULT.to_dict(),
    "config_hash": config_hash(), "determinism": det_info,
    "step_stats": stats, "resume_test": resume_report,
    "projection_24000_steps": {
        "train_only_h": round(full / 3600, 2),
        "with_index_and_ckpt_h": round((full + index_s + 5 * 30) / 3600, 2),
        "with_20pct_margin_h": round((full + index_s + 5 * 30) * 1.2 / 3600, 2),
        "fits_7_5h_single_session": bool((full + index_s + 150) * 1.2 / 3600 <= 7.5),
        "two_session_hours_each": round(full / 2 / 3600, 2),
    },
}
(OUT / "smoke_report.json").write_text(json.dumps(report, indent=2, default=str))
print("")
print(json.dumps({k: v for k, v in report.items()
                  if k not in ("train_crops", "inner_val_crops", "aug_ranges", "determinism")},
                 indent=2, default=str))
print("DONE", flush=True)
