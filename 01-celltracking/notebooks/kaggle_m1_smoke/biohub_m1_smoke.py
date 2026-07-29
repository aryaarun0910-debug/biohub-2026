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

# ------------------------------------------------- patch trainer RNG + build fold-1 split
train_src = repo / "scripts" / "train_unet_transformer.py"
work_repo = Path("/kaggle/working/repo"); work_repo.mkdir(parents=True, exist_ok=True)
import shutil
if not (work_repo / "scripts").exists():
    shutil.copytree(repo / "scripts", work_repo / "scripts")
    shutil.copytree(repo / "src", work_repo / "src")
tgt = work_repo / "scripts" / "train_unet_transformer.py"
src_txt = tgt.read_text()
if M1A.RNG_PATCH_OLD not in src_txt:
    raise SystemExit("ABORT -- unseeded RNG line not found; determinism patch would no-op")
src_txt = src_txt.replace(M1A.RNG_PATCH_OLD, M1A.RNG_PATCH_NEW, 1)
_nl = chr(10)
_inject = _nl.join(["import numpy as np",
                    "M1_SEED = %d" % EXPECT["seed"],
                    "M1_EPOCH = [0]"])
src_txt = src_txt.replace("import numpy as np", _inject, 1)
tgt.write_text(src_txt)
print("determinism patch applied to the working copy of train_unet_transformer.py", flush=True)
sys.path.insert(0, str(work_repo / "src")); sys.path.insert(0, str(work_repo / "scripts"))

# Fold-1 splits file. `test` is a single TRAIN-FAMILY crop purely to satisfy the trainer
# API -- it is NOT the inner-validation manifest and NOT used for any selection. The
# held-out 6bba family appears nowhere.
splits = work_repo / "m1_fold1_splits.json"
holdout_probe = TRAIN_CROPS[:1]
splits.write_text(json.dumps([{"split": 0, "train": TRAIN_CROPS, "test": holdout_probe}]))
assert not any(c.startswith(FORBIDDEN_FAMILY)
               for c in TRAIN_CROPS + holdout_probe), "held-out leak in splits"
print(f"splits: train={len(TRAIN_CROPS)} api-test={len(holdout_probe)} "
      f"(no {FORBIDDEN_FAMILY} anywhere)", flush=True)

import train_unet_transformer as T  # noqa: E402

# ---------------------------------------------------------------- ONE epoch = 800 steps
t0 = time.time()
torch.cuda.reset_peak_memory_stats()
model = T.train(
    data_dir=data_dir, splits_file=splits, fold=0, method="unet_transformer",
    n_epochs=1, lr=EXPECT["lr"], batch_size=EXPECT["batch_size"], num_workers=2,
    max_iters=EXPECT["max_iters"], seed=EXPECT["seed"],
    augmentations=[T.flip_augment, M1A.as_trainer_augmentation()],
    data_parallel=False,
)
epoch_s = time.time() - t0
peak_gpu = torch.cuda.max_memory_allocated() / 1e6

ck = OUT / "m1_fold1_smoke_epoch1.pt"
tmp = ck.with_suffix(".pt.tmp")
torch.save({"model": model.state_dict(), "epoch": 1, "global_step": EXPECT["max_iters"],
            "rng_python": __import__("random").getstate(),
            "rng_numpy": np.random.get_state(), "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all(),
            "seed": EXPECT["seed"], "aug_fingerprint": M1A.config_fingerprint(),
            "config_hash": config_hash(),
            "manifest_sha256": sha256(m1_dir / "val_manifests.json")}, tmp)
os.replace(tmp, ck)
blob = torch.load(ck, weights_only=False)
resume_ok = {"model": "model" in blob, "epoch": blob["epoch"],
             "global_step": blob["global_step"],
             "rng_python": blob["rng_python"] is not None,
             "rng_numpy": blob["rng_numpy"] is not None,
             "rng_torch": blob["rng_torch"] is not None,
             "rng_cuda": len(blob["rng_cuda"]) > 0,
             "config_hash": blob["config_hash"],
             "manifest_sha256": blob["manifest_sha256"][:16],
             "OPTIMIZER_STATE": "NOT SAVED BY train(); full-run kernel must save it"}

per_step = epoch_s / EXPECT["max_iters"]
full = per_step * 24000
report = {
    "crops_train": len(TRAIN_CROPS), "optimizer_steps": EXPECT["max_iters"],
    "batch_size": EXPECT["batch_size"], "lr": EXPECT["lr"],
    "epoch_wall_seconds": round(epoch_s, 1),
    "seconds_per_step_mean": round(per_step, 4),
    "peak_gpu_mb": round(peak_gpu, 1),
    "checkpoint_bytes": ck.stat().st_size, "checkpoint_sha256": sha256(ck),
    "aug_fingerprint": M1A.config_fingerprint(), "config_hash": config_hash(),
    "resume": resume_ok, "determinism": det_info,
    "projection_24000_steps": {
        "train_only_h": round(full / 3600, 2),
        "with_ckpt_overhead_h": round((full + 5 * 30) / 3600, 2),
        "with_20pct_margin_h": round((full + 5 * 30) * 1.2 / 3600, 2),
        "fits_7_5h_single_session": bool((full + 5 * 30) * 1.2 / 3600 <= 7.5),
    },
}
(OUT / "smoke_report.json").write_text(json.dumps(report, indent=2, default=str))
print("")
print(json.dumps(report, indent=2, default=str))
print("DONE", flush=True)
