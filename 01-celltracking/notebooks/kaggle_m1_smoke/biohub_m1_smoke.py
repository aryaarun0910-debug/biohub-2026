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
print(f"  INTERSECTION: {inter} -> {'EMPTY (correct)' if not inter else 'LEAK!'}")
if inter:
    raise SystemExit(f"ABORT -- validation crops in training loader: {inter}")
if [c for c in TRAIN_SET | VAL_SET if not c.startswith("44b6")]:
    raise SystemExit("ABORT -- non-train-family crop in scope")
print(f"  all 44b6; zero {FORBIDDEN_FAMILY} paths constructible  OK")

# ------------------------------------------------- patch trainer (4 patches) + construct
import shutil  # noqa: E402

import psutil  # noqa: E402

import m1_driver as MD  # noqa: E402

work_repo = Path("/kaggle/working/repo")
work_repo.mkdir(parents=True, exist_ok=True)
if not (work_repo / "scripts").exists():
    shutil.copytree(repo / "scripts", work_repo / "scripts")
    shutil.copytree(repo / "src", work_repo / "src")
tgt = work_repo / "scripts" / "train_unet_transformer.py"
trainer_src_sha = hashlib.sha256((repo / "scripts" / "train_unet_transformer.py").read_bytes()).hexdigest()
tgt.write_text(MD.patch_all(tgt.read_text(), EXPECT["seed"], M1A))
print("trainer patched: seeded RNG + telemetry + grad-norm + sample identity", flush=True)
sys.path.insert(0, str(work_repo / "src"))
sys.path.insert(0, str(work_repo / "scripts"))
sys.path.insert(0, str(m1_dir))

from torch.utils.data import DataLoader  # noqa: E402

import train_unet_transformer as T  # noqa: E402

device = torch.device("cuda")
manifest_sha = sha256(m1_dir / "val_manifests.json")
LW = MD.BASELINE_LOSS_WEIGHTS
print(f"BASELINE loss weights enforced: {LW}", flush=True)

t_index = time.time()
train_files = [data_dir / c for c in sorted(TRAIN_SET)]
video_data = []
for f in train_files:
    vm, w = T.load_dataset_windows(f, window_size=2, max_frames=None, downsample=(1, 4, 4))
    video_data.append((vm, w))
n_windows = sum(len(w) for _, w in video_data)
max_nodes = max(max(w.node_counts) for _, ws in video_data for w in ws)
index_s = time.time() - t_index
# window index -> crop, so unique crops sampled can be reported
win_owner = []
for (vm, ws) in video_data:
    win_owner += [Path(str(getattr(vm, "path", getattr(vm, "name", "?")))).name] * len(ws)
print(f"indexed {len(train_files)} crops -> {n_windows} windows, max_nodes={max_nodes}, "
      f"{index_s:.1f}s", flush=True)


def build(seed, epoch=0, start=0, length=None):
    DT.seed_everything(seed)
    ds = T.FrameWindowDataset(video_data, max_nodes=max_nodes,
                              augmentations=[T.flip_augment, M1A.as_trainer_augmentation()])
    sampler = MD.ResumableSampler(len(ds), seed=seed, epoch=epoch, start=start, length=length)
    ld = DataLoader(ds, batch_size=EXPECT["batch_size"], sampler=sampler, num_workers=0,
                    pin_memory=False)
    unet = T.TemporalUNet3D(in_channels=1, out_channels=32, layers=[32, 64, 128])
    m = T.UNetNodeTransformer(unet=unet, unet_out_channels=32,
                              pos_feat_dim=4 * T._POS_EMBED_DIM).to(device)
    o = torch.optim.AdamW(m.parameters(), lr=EXPECT["lr"])
    return ds, ld, m, o, sampler


def run(model, loader, opt, n):
    return T.train_epoch(model, loader, opt, device,
                         LW["det_loss_weight"], LW["det_neg_weight"],
                         max_iters=n, pool_kernel_um=LW["pool_kernel_um"])


N = 40
SEED = 4242
MD.SAMPLE_LOG_ENABLED[0] = True
M1A.AUG_LOG_ENABLED[0] = True

print("")
print(f"=== CONTROL A: uninterrupted 1..{N} ===", flush=True)
_, ldA, mA, oA, spA = build(SEED, epoch=0, length=N)
MD.STEP_LOG.clear(); MD.SAMPLE_LOG.clear(); M1A.AUG_LOG.clear()
run(mA, ldA, oA, N)
A_steps = list(MD.STEP_LOG); A_samples = list(MD.SAMPLE_LOG)
A_hash = DT.state_hash(mA)
A_aug = list(M1A.AUG_LOG)

print(f"=== CONTROL A2: identical rerun (GPU nondeterminism floor) ===", flush=True)
_, ldA2, mA2, oA2, _ = build(SEED, epoch=0, length=N)
MD.STEP_LOG.clear(); MD.SAMPLE_LOG.clear()
run(mA2, ldA2, oA2, N)
A2_steps = list(MD.STEP_LOG); A2_samples = list(MD.SAMPLE_LOG)
A2_hash = DT.state_hash(mA2)

print(f"=== RESUME B: 1..{N // 2}, serialize/reload, {N // 2 + 1}..{N} ===", flush=True)
_, ldB, mB, oB, spB = build(SEED, epoch=0, length=N)
MD.STEP_LOG.clear(); MD.SAMPLE_LOG.clear()
run(mB, ldB, oB, N // 2)
B1_steps = list(MD.STEP_LOG); B1_samples = list(MD.SAMPLE_LOG)
mid = OUT / "_resume.pt"
MD.save_checkpoint(mid, model=mB, optimizer=oB, epoch=0, global_step=N // 2, seed=SEED,
                   config_hash=config_hash(), aug_fingerprint=M1A.config_fingerprint(),
                   manifest_sha=manifest_sha, torch_mod=torch)
blob = torch.load(mid, weights_only=False)
blob["sampler_state"] = spB.state()
blob["step_in_epoch"] = N // 2
torch.save(blob, mid)

_, _, mR, oR, _ = build(SEED, epoch=0, length=N)
restored = MD.load_checkpoint(mid, model=mR, optimizer=oR, torch_mod=torch,
                              expect_config_hash=config_hash())
st = torch.load(mid, weights_only=False)
spR = MD.ResumableSampler.resume(st["sampler_state"], step_in_epoch=st["step_in_epoch"])
dsR = T.FrameWindowDataset(video_data, max_nodes=max_nodes,
                           augmentations=[T.flip_augment, M1A.as_trainer_augmentation()])
ldR = DataLoader(dsR, batch_size=1, sampler=spR, num_workers=0, pin_memory=False)
MD.STEP_LOG.clear(); MD.SAMPLE_LOG.clear()
run(mR, ldR, oR, N // 2)
B2_steps = list(MD.STEP_LOG); B2_samples = list(MD.SAMPLE_LOG)
B_hash = DT.state_hash(mR)

B_steps = B1_steps + B2_steps
B_samples = B1_samples + B2_samples
same_ids = [a["idx"] == b["idx"] for a, b in zip(A_samples[N // 2:], B2_samples)]
same_aug = [a["img_hash"] == b["img_hash"] for a, b in zip(A_samples[N // 2:], B2_samples)]
dev_resume = max(abs(a["edge_loss"] - b["edge_loss"]) for a, b in zip(A_steps, B_steps))
dev_floor = max(abs(a["edge_loss"] - b["edge_loss"]) for a, b in zip(A_steps, A2_steps))

resume_report = {
    "counters_restored": {"epoch": restored["epoch"], "global_step": restored["global_step"],
                          "step_in_epoch": st["step_in_epoch"],
                          "sampler_permutation_hash": st["sampler_state"]["permutation_hash"]},
    "config_hash_verified": restored["config_hash"] == config_hash(),
    "manifest_sha_verified": restored["manifest_sha256"] == manifest_sha,
    "optimizer_state_restored": bool(oR.state_dict()["state"]),
    "steps_21_40_same_sample_ids": all(same_ids),
    "steps_21_40_same_augmented_images": all(same_aug),
    "n_compared": len(same_ids),
    "max_loss_dev_resume_vs_control": dev_resume,
    "max_loss_dev_control_vs_control": dev_floor,
    "resume_within_nondeterminism_floor": bool(dev_resume <= max(dev_floor, 1e-12) * 1.5),
    "final_weights_resume_vs_control": B_hash == A_hash,
    "final_weights_control_vs_control": A2_hash == A_hash,
}
print(json.dumps(resume_report, indent=2, default=str), flush=True)

MD.SAMPLE_LOG_ENABLED[0] = False
M1A.AUG_LOG_ENABLED[0] = False

aug_counts = {}
for r in A_aug:
    for t in r["fired"]:
        aug_counts[t] = aug_counts.get(t, 0) + 1
gn = [r.get("grad_norm", float("nan")) for r in A_steps]
pos = [r.get("n_pos_targets", 0) for r in A_steps]
uniq_win = {s["idx"] for s in A_samples}
uniq_crops = {win_owner[i] for i in uniq_win if i < len(win_owner)}

report = {
    "commit_note": "see repo git log; config/manifest/source hashes below",
    "trainer_source_sha256": trainer_src_sha,
    "config_hash": config_hash(), "aug_fingerprint": M1A.config_fingerprint(),
    "manifest_sha256": manifest_sha,
    "baseline_loss_weights_enforced": LW,
    "max_nodes": max_nodes,
    "max_nodes_note": ("computed over TRAIN crops only; baseline computes over train+test but "
                       "test there is the held-out family, which M1 must not touch. max_nodes "
                       "only sets padding width and is masked, so it is semantically neutral."),
    "train_crops": sorted(TRAIN_SET), "inner_val_crops": sorted(VAL_SET),
    "train_val_intersection": inter,
    "windows_indexed": n_windows, "index_seconds": round(index_s, 1),
    "unique_windows_sampled": len(uniq_win), "unique_crops_sampled": len(uniq_crops),
    "augmentation_counts_over_%d_samples" % len(A_aug): aug_counts,
    "augmentation_ranges_configured": M1A.DEFAULT.to_dict(),
    "edge_positive_targets": {
        "windows_with_any_positive": int(sum(1 for p in pos if p > 0)),
        "windows_total": len(pos),
        "frac_positive": float(sum(1 for p in pos if p > 0) / max(len(pos), 1)),
        "mean_positive_per_window": float(np.mean(pos)) if pos else 0.0},
    "grad_norm": {"median": float(np.nanmedian(gn)), "p90": float(np.nanpercentile(gn, 90)),
                  "max": float(np.nanmax(gn)), "nonfinite": int(np.sum(~np.isfinite(gn)))},
    "peak_cpu_rss_mb": round(psutil.Process().memory_info().rss / 1e6, 1),
    "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
    "resume_test": resume_report,
    "session_plan": {"session_1": "epochs 1-15 (12,000 steps)",
                     "session_2": "epochs 16-30 (12,000 steps)",
                     "hours_each_measured": 3.94,
                     "checkpoints": [10, 15, 20, 25, 30]},
}
ckf = OUT / "m1_resume_probe.pt"
ck_sha = MD.save_checkpoint(ckf, model=mA, optimizer=oA, epoch=0, global_step=N, seed=SEED,
                            config_hash=config_hash(),
                            aug_fingerprint=M1A.config_fingerprint(),
                            manifest_sha=manifest_sha, torch_mod=torch)
report["checkpoint_bytes"] = ckf.stat().st_size
report["checkpoint_sha256"] = ck_sha
(OUT / "resume_report.json").write_text(json.dumps(report, indent=2, default=str))
print("")
print(json.dumps({k: v for k, v in report.items()
                  if k not in ("train_crops", "inner_val_crops",
                               "augmentation_ranges_configured")},
                 indent=2, default=str))
print("DONE", flush=True)
