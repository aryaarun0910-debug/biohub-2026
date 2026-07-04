"""Kaggle kernel: train the organizer UNet+transformer EMBRYO-HELD-OUT (clean OOF baseline = step 3).

This is the "sole optimization target" from the Codex-reviewed plan: train on one embryo family, hold the
other out entirely, so the score predicts the disjoint HIDDEN embryo far better than the leakable public
board (all 4 visible test movies have labeled train copies). Uses the public organizer code + data:
  - repo/code: pilkwang/biohub-tracking-support-pack-50ep-v1  (CC0; its own pinned biohub_tracking repo)
  - train+GT: kms111201/biohub-cell-tracking-data             (CC0; 79GB, chunked train_chunk_*/train/)

WHAT THIS DOES
  1. finds the pack repo + copies it to writable /kaggle/working/repo (so WEIGHTS_PATH is writable)
  2. installs training deps (internet ON — enable it for this training kernel)
  3. gathers the chunked train crops (zarr+geff) into one dir via symlinks (glob-discovered; layout-robust)
  4. writes the EMBRYO-HELD-OUT splits: fold 0 = train 6bba / test 44b6 ; fold 1 = train 44b6 / test 6bba
  5. runs train_unet_transformer.py with pool_kernel_um=5.0 (the value the model is trained/served with)

DEFAULT = a fast SMOKE run (2 epochs, 50 iters) to prove the whole pipeline runs on GPU before you commit
hours. Set SMOKE=False for the real run and watch the per-epoch timing print to stay under the 12h limit.

Notes / caveats (see reports/RUNBOOK_step3_clean_oof_baseline_2026-07-04.md):
  - crops have DIFFERENT spatial shapes -> BATCH_SIZE must be 1 (so effectively single-T4; DataParallel
    needs batch>=2 same-shape). Real 50ep may need >1 session or --max-iters.
  - the trainer selects "best" epoch by acc*recall ON the held-out embryo -> that number is OPTIMISTIC
    (model-selection leakage). For a strictly clean OOF, carve a few TRAIN-embryo crops as val, or use a
    fixed epoch count, then score ONCE with the exact metric (biotrack.metric / the pack's --evaluate).
  - after training, predict the held-out fold with predict_unet_transformer.py (SET pool_kernel_um=5.0,
    it defaults to 3.0!) --use-ilp --det-threshold 0.99, then score. That OOF is the real target.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ============================ CONFIG ============================
FOLD = 0                 # 0 = train 6bba / test 44b6 (Fold A);  1 = train 44b6 / test 6bba (Fold B)
SMOKE = True             # True: fast end-to-end check. Set False for the real run.
EPOCHS = 2 if SMOKE else 50
MAX_ITERS = 50 if SMOKE else 1500   # cap train iters/epoch to bound runtime; raise/remove for full epochs
BATCH_SIZE = 1           # keep 1: train crops have varying spatial shapes
POOL_KERNEL_UM = 5.0     # MUST match training/serving (config.json says 5.0)
LR = 1e-4
WORK = Path("/kaggle/working")
# ===============================================================


def sh(cmd, **kw):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def find_pack_repo() -> Path:
    """Locate the support pack's repo/ under /kaggle/input (glob = layout-robust)."""
    hits = glob.glob("/kaggle/input/**/repo/scripts/train_unet_transformer.py", recursive=True)
    if not hits:
        raise SystemExit("Support pack repo not found. Add dataset "
                         "'pilkwang/biohub-tracking-support-pack-50ep-v1' as input.")
    return Path(hits[0]).parents[1]  # .../repo


def gather_train_dir() -> Path:
    """Symlink every train crop (.zarr + matching .geff), wherever it is mounted, into one dir.

    Handles the chunked layout (train_chunk_*/train/<crop>.zarr|.geff) and any mount path.
    A crop is kept only if BOTH its .zarr and .geff exist (GT required for training)."""
    dst = WORK / "train"
    dst.mkdir(exist_ok=True)
    geffs = glob.glob("/kaggle/input/**/train/*.geff", recursive=True)
    n = 0
    for g in geffs:
        stem = Path(g).stem
        z = str(Path(g).with_suffix(".zarr"))
        if not os.path.exists(z):
            continue
        for src in (g, z):
            link = dst / Path(src).name
            if not link.exists():
                try:
                    os.symlink(src, link)
                except OSError:
                    # fall back to copy if symlinks are disallowed
                    (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, link)
        n += 1
    if n == 0:
        raise SystemExit("No train crops (.zarr + .geff) found. Add dataset "
                         "'kms111201/biohub-cell-tracking-data' as input.")
    print(f"Gathered {n} train crops -> {dst}", flush=True)
    return dst


def write_embryo_held_out_splits(train_dir: Path) -> Path:
    """fold 0 = train 6bba / test 44b6 ; fold 1 = train 44b6 / test 6bba."""
    crops = sorted(p.stem for p in train_dir.glob("*.geff")
                   if (train_dir / f"{p.stem}.zarr").exists())
    by_fam: dict[str, list[str]] = {}
    for c in crops:
        by_fam.setdefault(c.split("_")[0], []).append(c)
    fams = sorted(by_fam)
    assert len(fams) == 2, f"expected 2 embryo families, got {fams}"
    a, b = fams  # '44b6', '6bba'
    folds = [
        {"train": by_fam[b], "test": by_fam[a]},   # fold 0: train 6bba, test 44b6
        {"train": by_fam[a], "test": by_fam[b]},   # fold 1: train 44b6, test 6bba
    ]
    for i, f in enumerate(folds):
        print(f"fold {i}: train {len(f['train'])} ({f['train'][0].split('_')[0]}) / "
              f"test {len(f['test'])} ({f['test'][0].split('_')[0]})", flush=True)
    p = WORK / "dataset_splits.json"
    p.write_text(json.dumps(folds))
    return p


def main():
    # 1. writable copy of the pack repo
    pack_repo = find_pack_repo()
    repo = WORK / "repo"
    if not repo.exists():
        shutil.copytree(pack_repo, repo)
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "scripts"))
    print(f"repo -> {repo}", flush=True)

    # 2. training deps (internet ON). torch is preinstalled on the Kaggle GPU image.
    sh([sys.executable, "-m", "pip", "install", "-q",
        "tracksdata", "geff>=1.1.3.1.1", "zarr>=3.0.10,<4", "polars>=1.36",
        "numcodecs>=0.13", "blosc2", "imagecodecs", "rustworkx>=0.17.1", "tqdm"])

    # 3-4. data + splits
    train_dir = gather_train_dir()
    splits = write_embryo_held_out_splits(train_dir)

    # 5. train the held-out fold
    env = dict(os.environ, BIOHUB_DATA_DIR=str(train_dir))
    cmd = [sys.executable, str(repo / "scripts" / "train_unet_transformer.py"),
           "--split", str(FOLD),
           "--data-dir", str(train_dir),
           "--splits", str(splits),
           "--epochs", str(EPOCHS),
           "--lr", str(LR),
           "--batch-size", str(BATCH_SIZE),
           "--pool-kernel-um", str(POOL_KERNEL_UM)]
    if MAX_ITERS is not None:
        cmd += ["--max-iters", str(MAX_ITERS)]
    sh(cmd, env=env)

    out = repo / "weights" / "unet_transformer" / f"split_{FOLD}"
    print(f"\nDONE. Weights -> {out} (copy to /kaggle/working to download).", flush=True)
    ckpt = out / "edge_predictor_best.pth"
    if ckpt.exists():
        shutil.copy(ckpt, WORK / f"edge_predictor_best_split_{FOLD}.pth")
        shutil.copy(out / "config.json", WORK / f"config_split_{FOLD}.json")
        print(f"Copied checkpoint + config to {WORK} for download.", flush=True)
    print("\nNext: set SMOKE=False for the real run; then predict the held-out fold with "
          "predict_unet_transformer.py (SET PredictConfig.pool_kernel_um=5.0!) --use-ilp "
          "--det-threshold 0.99, and score with biotrack.metric = the clean OOF target.", flush=True)


if __name__ == "__main__":
    main()
