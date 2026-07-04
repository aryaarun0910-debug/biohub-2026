"""Kaggle kernel: train the organizer UNet+transformer for a CLEAN embryo-held-out OOF baseline (step 3).

The "sole optimization target" from the Codex-reviewed plan. One embryo family is held out ENTIRELY from
training and scored separately (by the predict-and-score kernel) — that OOF predicts the disjoint HIDDEN
embryo far better than the leakable public board. Public organizer code + data (both CC0):
  - repo/code: pilkwang/biohub-tracking-support-pack-50ep-v1
  - train+GT: kms111201/biohub-cell-tracking-data   (79GB, chunked train_chunk_*/train/)

CLEAN-OOF DESIGN (fixes the two traps found in the code):
  - The HELD-OUT embryo is NOT used in training at all -> scored once, later, by the predict kernel.
  - The trainer selects its best epoch by scoring its `test` set each epoch. We set that to a small
    VALIDATION subset of the TRAINING embryo (NOT the held-out embryo), so (a) best-epoch selection never
    peeks at the held-out embryo, and (b) per-epoch eval is cheap enough to fit the 12h limit. (The
    original code evaluated the full 71-crop held-out embryo every epoch -> ~15-20h, would time out.)
  - pool_kernel_um=5.0 (the value the model is trained AND must be served with; predict defaults to 3.0!).

RUN ORDER: keep SMOKE=True for a fast pipeline check first; then SMOKE=False for the real run; run once
per FOLD (0 and 1). After each fold, run the predict-and-score kernel to get that fold's OOF number.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ============================ CONFIG ============================
FOLD = 1                 # 0 = HOLD OUT 44b6 (train on 6bba);  1 = HOLD OUT 6bba (train on 44b6)
SMOKE = False            # True: tiny fast pipeline check.  False: real budget-bounded run.
VAL_N = 6                # train-embryo crops used as validation for best-epoch selection (NOT held-out embryo)
EPOCHS = 2 if SMOKE else 30
MAX_ITERS = 50 if SMOKE else 800   # cap train iters/epoch. Watch epoch-0 timing: if EPOCHS*(train+test)
                                   # projects > ~10h, lower EPOCHS or MAX_ITERS (12h hard limit).
BATCH_SIZE = 1           # keep 1: train crops have varying spatial shapes (so effectively single-T4)
POOL_KERNEL_UM = 5.0
LR = 1e-4
WORK = Path("/kaggle/working")
# ===============================================================

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def sh(cmd, **kw):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def find_pack_repo() -> Path:
    hits = glob.glob("/kaggle/input/**/repo/scripts/train_unet_transformer.py", recursive=True)
    if not hits:
        raise SystemExit("Support pack repo not found. Add dataset "
                         "'pilkwang/biohub-tracking-support-pack-50ep-v1' as input.")
    return Path(hits[0]).parents[1]


def gather_train_dir() -> Path:
    """Symlink every crop (.zarr + matching .geff) into one dir. Fixed-depth globs (NOT recursive **),
    which would walk millions of .zarr chunk files across the 79GB mount."""
    dst = WORK / "train"
    dst.mkdir(exist_ok=True)
    geffs: list[str] = []
    for pat in ("/kaggle/input/*/train/*.geff",
                "/kaggle/input/*/*/train/*.geff",
                "/kaggle/input/*/*/*/train/*.geff"):
        geffs += glob.glob(pat)
    geffs = sorted(set(geffs))
    n = 0
    for g in geffs:
        z = str(Path(g).with_suffix(".zarr"))
        if not os.path.exists(z):
            continue
        for src in (g, z):
            link = dst / Path(src).name
            if not link.exists():
                try:
                    os.symlink(src, link)
                except OSError:
                    (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, link)
        n += 1
    if n == 0:
        raise SystemExit("No train crops (.zarr + .geff) found. Add dataset "
                         "'kms111201/biohub-cell-tracking-data' as input.")
    print(f"Gathered {n} crops -> {dst}", flush=True)
    return dst


def load_corrupt_stems() -> set[str]:
    """Crops flagged in the dataset's metadata_files/corrupt_files.txt (some chunks unreadable).
    Excluding whole crops avoids a mid-run crash. Returns a set of crop stems."""
    import re
    hits = (glob.glob("/kaggle/input/*/metadata_files/corrupt_files.txt")
            + glob.glob("/kaggle/input/*/*/corrupt_files.txt"))
    stems: set[str] = set()
    for h in hits:
        stems |= set(re.findall(r"(?:44b6|6bba)_[0-9a-f]+", Path(h).read_text()))
    if stems:
        print(f"Excluding {len(stems)} corrupt crops: {sorted(stems)}", flush=True)
    return stems


def write_train_val_splits(train_dir: Path, val_n: int, smoke: bool, corrupt: set[str]) -> tuple[Path, str]:
    """Write a 2-fold splits file where each fold's train/test are BOTH from the TRAINING embryo
    (train = most crops, test = a small val subset for epoch selection). The held-out embryo is
    excluded here and scored later. Returns (splits_path, held_out_family_for_FOLD)."""
    crops = sorted(p.stem for p in train_dir.glob("*.geff")
                   if (train_dir / f"{p.stem}.zarr").exists() and p.stem not in corrupt)
    by_fam: dict[str, list[str]] = {}
    for c in crops:
        by_fam.setdefault(c.split("_")[0], []).append(c)
    fams = sorted(by_fam)
    assert len(fams) == 2, f"expected 2 embryo families, got {fams}"
    a, b = fams  # a='44b6', b='6bba'

    def fold_for(train_fam: str) -> dict:
        cf = by_fam[train_fam]
        if smoke:
            cf = cf[:8]           # 8 crops -> 2 val + 6 train, fast
        vn = 2 if smoke else val_n
        return {"train": cf[vn:], "test": cf[:vn]}   # test = validation subset (same embryo)

    folds = [fold_for(b), fold_for(a)]   # fold 0 trains 6bba (holds out 44b6); fold 1 trains 44b6
    held_out = {0: a, 1: b}
    for i, f in enumerate(folds):
        tr_fam = f["train"][0].split("_")[0]
        print(f"fold {i}: train {len(f['train'])} + val {len(f['test'])} ({tr_fam})  "
              f"| HELD OUT (scored separately): {held_out[i]}", flush=True)
    p = WORK / "dataset_splits.json"
    p.write_text(json.dumps(folds))
    return p, held_out[FOLD]


def main():
    pack_repo = find_pack_repo()
    repo = WORK / "repo"
    if not repo.exists():
        shutil.copytree(pack_repo, repo)
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "scripts"))
    print(f"repo -> {repo}", flush=True)

    # The trainer only saves the epoch that maximizes acc*recall on the val set — a proxy that saturates
    # early (best often = epoch 0, an UNDERtrained detector) and is measured at det_threshold 0.3, not the
    # 0.99 predict uses. So ALSO save the last (fully-trained) epoch each epoch; we score THAT.
    tf = repo / "scripts" / "train_unet_transformer.py"
    tsrc = tf.read_text()
    anchor = '        marker = "*" if is_best else " "'
    save_last = ('        torch.save({k.replace("unet.module.", "unet.", 1): v for k, v in '
                 'model.state_dict().items()}, output_dir / "edge_predictor_last.pth")\n')
    assert anchor in tsrc, "trainer anchor for last-epoch save not found (script version changed?)"
    tf.write_text(tsrc.replace(anchor, save_last + anchor, 1))
    print("Patched trainer: also save edge_predictor_last.pth each epoch (we score the trained model).", flush=True)

    sh([sys.executable, "-m", "pip", "install", "-q",
        "tracksdata", "geff>=1.1.3.1.1", "zarr>=3.0.10,<4", "polars>=1.36",
        "numcodecs>=0.13", "blosc2", "imagecodecs", "rustworkx>=0.17.1", "tqdm"])

    train_dir = gather_train_dir()
    corrupt = load_corrupt_stems()
    splits, held_out = write_train_val_splits(train_dir, VAL_N, SMOKE, corrupt)
    print(f"Training FOLD {FOLD} (holds out {held_out}); {'SMOKE' if SMOKE else 'REAL'} run: "
          f"{EPOCHS} epochs x <= {MAX_ITERS} iters.", flush=True)

    # trainer runs as a SUBPROCESS (fresh Python) -> put repo/src (biohub_tracking) + repo/scripts on
    # PYTHONPATH; -u + PYTHONUNBUFFERED so its logs stream live.
    pypath = os.pathsep.join([str(repo / "src"), str(repo / "scripts"), os.environ.get("PYTHONPATH", "")])
    env = dict(os.environ, BIOHUB_DATA_DIR=str(train_dir), PYTHONUNBUFFERED="1", PYTHONPATH=pypath)
    cmd = [sys.executable, "-u", str(repo / "scripts" / "train_unet_transformer.py"),
           "--split", str(FOLD), "--data-dir", str(train_dir), "--splits", str(splits),
           "--epochs", str(EPOCHS), "--lr", str(LR), "--batch-size", str(BATCH_SIZE),
           "--pool-kernel-um", str(POOL_KERNEL_UM)]
    if MAX_ITERS is not None:
        cmd += ["--max-iters", str(MAX_ITERS)]
    sh(cmd, env=env)

    out = repo / "weights" / "unet_transformer" / f"split_{FOLD}"
    # Export the LAST (fully-trained) epoch under the name the predict kernel looks for; the proxy-"best"
    # (often undertrained epoch 0) is kept separately for reference.
    last, best = out / "edge_predictor_last.pth", out / "edge_predictor_best.pth"
    src = last if last.exists() else best
    if src.exists():
        shutil.copy(src, WORK / f"edge_predictor_best_split_{FOLD}.pth")   # <- this is the LAST epoch
        shutil.copy(out / "config.json", WORK / f"config_split_{FOLD}.json")
        if best.exists():
            shutil.copy(best, WORK / f"edge_predictor_proxybest_split_{FOLD}.pth")
        print(f"\nDONE. Scored checkpoint (last epoch) -> {WORK}/edge_predictor_best_split_{FOLD}.pth", flush=True)
    print(f"\nNext: run the predict-and-score kernel for FOLD {FOLD} to score the HELD-OUT {held_out} "
          f"embryo -> the clean OOF number. Then repeat with FOLD={1 - FOLD}.", flush=True)


if __name__ == "__main__":
    main()
