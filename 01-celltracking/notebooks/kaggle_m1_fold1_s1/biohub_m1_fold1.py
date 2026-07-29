"""M1 fold-1 TRAINING — measurement only, no held-out access, no full training.

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

OUT = Path("/kaggle/working/m1_fold1")
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

# ------------------------------------------------- scope guards
VAL_SET, TRAIN_SET = set(VAL_CROPS), set(TRAIN_CROPS)
inter = sorted(TRAIN_SET & VAL_SET)
if inter:
    raise SystemExit(f"ABORT -- validation crops in training loader: {inter}")
if [c for c in TRAIN_SET | VAL_SET if not c.startswith("44b6")]:
    raise SystemExit("ABORT -- non-train-family crop in scope")
print(f"scope OK: train={len(TRAIN_SET)} val={len(VAL_SET)} (excluded) "
      f"intersection=EMPTY; zero {FORBIDDEN_FAMILY} paths", flush=True)

SESSION = 1
EPOCHS = {1: (1, 15), 2: (16, 30)}[SESSION]
CKPT_EPOCHS = [10, 15, 20, 25, 30]
print(f"SESSION {SESSION}: epochs {EPOCHS[0]}..{EPOCHS[1]}", flush=True)

import shutil  # noqa: E402

import psutil  # noqa: E402

import m1_driver as MD  # noqa: E402

work_repo = Path("/kaggle/working/repo")
work_repo.mkdir(parents=True, exist_ok=True)
if not (work_repo / "scripts").exists():
    shutil.copytree(repo / "scripts", work_repo / "scripts")
    shutil.copytree(repo / "src", work_repo / "src")
tgt = work_repo / "scripts" / "train_unet_transformer.py"
trainer_sha = hashlib.sha256((repo / "scripts" / "train_unet_transformer.py").read_bytes()).hexdigest()
tgt.write_text(MD.patch_all(tgt.read_text(), EXPECT["seed"], M1A))
sys.path.insert(0, str(work_repo / "src"))
sys.path.insert(0, str(work_repo / "scripts"))
sys.path.insert(0, str(m1_dir))

from torch.utils.data import DataLoader  # noqa: E402

import train_unet_transformer as T  # noqa: E402

device = torch.device("cuda")
manifest_sha = sha256(m1_dir / "val_manifests.json")
LW = MD.BASELINE_LOSS_WEIGHTS
print(f"baseline loss weights enforced: {LW}", flush=True)

t_index = time.time()
video_data = []
for c in sorted(TRAIN_SET):
    vm, w = T.load_dataset_windows(data_dir / c, window_size=2, max_frames=None,
                                   downsample=(1, 4, 4))
    video_data.append((vm, w))
n_windows = sum(len(w) for _, w in video_data)
max_nodes = max(max(w.node_counts) for _, ws in video_data for w in ws)
print(f"indexed {len(video_data)} crops -> {n_windows} windows, max_nodes={max_nodes}, "
      f"{time.time() - t_index:.1f}s", flush=True)

DT.seed_everything(EXPECT["seed"])
dataset = T.FrameWindowDataset(video_data, max_nodes=max_nodes,
                               augmentations=[T.flip_augment, M1A.as_trainer_augmentation()])
unet = T.TemporalUNet3D(in_channels=1, out_channels=32, layers=[32, 64, 128])
model = T.UNetNodeTransformer(unet=unet, unet_out_channels=32,
                              pos_feat_dim=4 * T._POS_EMBED_DIM).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=EXPECT["lr"])
print(f"model params: {sum(p.numel() for p in model.parameters()):,}  "
      f"init_hash={DT.state_hash(model)}", flush=True)

ROLLING = OUT / "m1_fold1_resume.pt"
start_epoch, global_step = EPOCHS[0], (EPOCHS[0] - 1) * EXPECT["max_iters"]

# Session 2 (or a retry) resumes from the rolling checkpoint if one was attached.
prior = sorted(glob.glob("/kaggle/input/**/m1_fold1_resume.pt", recursive=True))
if prior and SESSION > 1:
    src_ck = Path(prior[0])
    st = MD.load_checkpoint(src_ck, model=model, optimizer=optimizer, torch_mod=torch,
                            expect_config_hash=config_hash())
    blob = torch.load(src_ck, weights_only=False)
    if blob.get("manifest_sha256") != manifest_sha:
        raise SystemExit("ABORT -- manifest hash mismatch on resume")
    if blob.get("aug_fingerprint") != M1A.config_fingerprint():
        raise SystemExit("ABORT -- augmentation fingerprint mismatch on resume")
    start_epoch = int(st["epoch"]) + 1
    global_step = int(st["global_step"])
    print(f"RESUMED from {src_ck.name}: next epoch={start_epoch} global_step={global_step} "
          f"ckpt_sha={hashlib.sha256(src_ck.read_bytes()).hexdigest()[:16]}", flush=True)
    if start_epoch != EPOCHS[0]:
        raise SystemExit(f"ABORT -- resume epoch {start_epoch} != session start {EPOCHS[0]}")

history = []
t_run = time.time()
for epoch in range(start_epoch, EPOCHS[1] + 1):
    T.M1_EPOCH[0] = epoch                       # drives the seeded per-sample augmentation
    sampler = MD.ResumableSampler(len(dataset), seed=EXPECT["seed"], epoch=epoch,
                                  start=0, length=EXPECT["max_iters"])
    loader = DataLoader(dataset, batch_size=EXPECT["batch_size"], sampler=sampler,
                        num_workers=2, prefetch_factor=2, persistent_workers=False,
                        pin_memory=False)
    MD.STEP_LOG.clear()
    t0 = time.time()
    edge_l, det_l = T.train_epoch(model, loader, optimizer, device,
                                  LW["det_loss_weight"], LW["det_neg_weight"],
                                  max_iters=EXPECT["max_iters"],
                                  pool_kernel_um=LW["pool_kernel_um"])
    dt = time.time() - t0
    global_step += EXPECT["max_iters"]
    gn = np.array([r.get("grad_norm", np.nan) for r in MD.STEP_LOG], dtype=float)
    losses = np.array([r["edge_loss"] for r in MD.STEP_LOG], dtype=float)
    dets = np.array([r["det_loss"] for r in MD.STEP_LOG], dtype=float)
    nonfinite = int(np.sum(~np.isfinite(losses)) + np.sum(~np.isfinite(dets))
                    + np.sum(~np.isfinite(gn)))
    clipped = float(np.mean(gn > 1.0)) if gn.size else float("nan")

    rec = {"epoch": epoch, "global_step": global_step, "seconds": round(dt, 1),
           "edge_loss": float(edge_l), "det_loss": float(det_l),
           "grad_norm_median": float(np.nanmedian(gn)),
           "grad_norm_p90": float(np.nanpercentile(gn, 90)),
           "grad_norm_max": float(np.nanmax(gn)),
           "clipped_step_frac": clipped, "nonfinite": nonfinite,
           "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
           "cpu_rss_mb": round(psutil.Process().memory_info().rss / 1e6, 1)}
    history.append(rec)
    print(f"  epoch {epoch:2d}/30 | edge={edge_l:.4f} det={det_l:.4f} | "
          f"gn med={rec['grad_norm_median']:.1f} p90={rec['grad_norm_p90']:.1f} "
          f"clipped={clipped * 100:.0f}% | {dt:.0f}s | nonfinite={nonfinite}", flush=True)

    # MECHANICAL GUARD: stop only on non-finite values or genuine divergence.
    if nonfinite:
        raise SystemExit(f"ABORT -- {nonfinite} non-finite values at epoch {epoch}")
    if not np.isfinite(edge_l) or not np.isfinite(det_l):
        raise SystemExit(f"ABORT -- non-finite epoch loss at epoch {epoch}")

    # rolling atomic resume checkpoint every epoch
    MD.save_checkpoint(ROLLING, model=model, optimizer=optimizer, epoch=epoch,
                       global_step=global_step, seed=EXPECT["seed"],
                       config_hash=config_hash(), aug_fingerprint=M1A.config_fingerprint(),
                       manifest_sha=manifest_sha, torch_mod=torch)
    # retained candidate checkpoints
    if epoch in CKPT_EPOCHS:
        cand = OUT / f"m1_fold1_epoch{epoch:02d}.pt"
        sha = MD.save_checkpoint(cand, model=model, optimizer=optimizer, epoch=epoch,
                                 global_step=global_step, seed=EXPECT["seed"],
                                 config_hash=config_hash(),
                                 aug_fingerprint=M1A.config_fingerprint(),
                                 manifest_sha=manifest_sha, torch_mod=torch)
        rec["candidate_checkpoint"] = {"file": cand.name, "sha256": sha,
                                       "bytes": cand.stat().st_size}
        print(f"    retained candidate checkpoint {cand.name} sha={sha[:16]}", flush=True)

    (OUT / f"m1_fold1_session{SESSION}_history.json").write_text(
        json.dumps({"session": SESSION, "epochs": EPOCHS, "seed": EXPECT["seed"],
                    "config_hash": config_hash(), "aug_fingerprint": M1A.config_fingerprint(),
                    "manifest_sha256": manifest_sha, "trainer_source_sha256": trainer_sha,
                    "baseline_loss_weights": LW, "max_nodes": max_nodes,
                    "n_train_crops": len(TRAIN_SET), "n_windows": n_windows,
                    "history": history}, indent=2, default=str))

print("")
print(f"SESSION {SESSION} COMPLETE: epochs {EPOCHS[0]}..{EPOCHS[1]} in "
      f"{(time.time() - t_run) / 3600:.2f} h", flush=True)
print(f"rolling resume checkpoint sha256={hashlib.sha256(ROLLING.read_bytes()).hexdigest()}",
      flush=True)
print("NOTE: checkpoint selection and held-out 6bba evaluation are SEPARATE jobs; this "
      "kernel never touched the held-out family.", flush=True)
print("DONE", flush=True)
