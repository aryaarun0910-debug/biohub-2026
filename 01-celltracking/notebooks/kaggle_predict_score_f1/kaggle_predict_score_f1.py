"""Kaggle kernel: predict the HELD-OUT embryo with a trained fold model and SCORE it = the clean OOF number.

Run this AFTER the training kernel (aryaarun07/biohub-train-oof) has produced weights for the same FOLD.
It reads those weights (attached as a kernel data source), predicts every crop of the held-out embryo, and
scores with the organizer's exact evaluator (edge adj-Jaccard + 0.1*division-Jaccard, count-adjusted) via
--evaluate. THAT printed score is the sole optimization target from the Codex-reviewed plan.

Inputs (attach all): pilkwang/biohub-tracking-support-pack-50ep-v1 (code), kms111201/biohub-cell-tracking-data
(held-out embryo zarr+GT), aryaarun07/biohub-train-oof (the trained checkpoint, as kernel output).

Fixes baked in:
  - pool_kernel_um patched 3.0 -> 5.0 in the predict script (it defaults to 3.0 and is NOT on the CLI;
    the model is trained/served at 5.0 -> using 3.0 silently mis-scores).
  - held-out embryo = the family NOT trained on for this FOLD; corrupt crops excluded.
  - USE_ILP defaults False for a first run (greedy still emits divisions via max_children=2). Flip True for
    the organizer's global SCIP ILP (installs pyscipopt+ilpy) once the greedy OOF is confirmed.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ============================ CONFIG ============================
FOLD = 1                 # must match a trained fold: 0 = score held-out 44b6; 1 = score held-out 6bba
USE_ILP = False          # False: greedy (fewer deps). True: organizer SCIP ILP (pyscipopt+ilpy).
DET_THRESHOLD = 0.99     # predict default; high-precision on sparse GT
ILP_DIVISION_WEIGHT = 1.0
POOL_KERNEL_UM = 5.0     # the trained/served value (predict script defaults to 3.0 -> we patch it)
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
    hits = glob.glob("/kaggle/input/**/repo/scripts/predict_unet_transformer.py", recursive=True)
    if not hits:
        raise SystemExit("Support pack repo not found. Add 'pilkwang/biohub-tracking-support-pack-50ep-v1'.")
    return Path(hits[0]).parents[1]


def gather_data_dir() -> Path:
    dst = WORK / "data"
    dst.mkdir(exist_ok=True)
    geffs: list[str] = []
    for pat in ("/kaggle/input/*/train/*.geff", "/kaggle/input/*/*/train/*.geff",
                "/kaggle/input/*/*/*/train/*.geff"):
        geffs += glob.glob(pat)
    n = 0
    for g in sorted(set(geffs)):
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
        raise SystemExit("No crops found. Add 'kms111201/biohub-cell-tracking-data'.")
    print(f"Gathered {n} crops -> {dst}", flush=True)
    return dst


def load_corrupt() -> set[str]:
    stems: set[str] = set()
    for h in (glob.glob("/kaggle/input/*/metadata_files/corrupt_files.txt")
              + glob.glob("/kaggle/input/*/*/corrupt_files.txt")):
        stems |= set(re.findall(r"(?:44b6|6bba)_[0-9a-f]+", Path(h).read_text()))
    return stems


def find_weights() -> tuple[Path, Path | None]:
    # Prefer the top-level copy; fall back to the trainer's native path (survives even if the training
    # kernel's final copy step was skipped, e.g. a timeout) so a long unattended run isn't wasted.
    w = (glob.glob(f"/kaggle/input/*/edge_predictor_best_split_{FOLD}.pth")
         + glob.glob(f"/kaggle/input/*/repo/weights/*/split_{FOLD}/edge_predictor_best.pth"))
    if not w:
        raise SystemExit(f"No trained weights for FOLD {FOLD}. Run the training kernel "
                         f"(aryaarun07/biohub-train-oof) with FOLD={FOLD} first, then attach it here.")
    c = (glob.glob(f"/kaggle/input/*/config_split_{FOLD}.json")
         + glob.glob(f"/kaggle/input/*/repo/weights/*/split_{FOLD}/config.json"))
    return Path(w[0]), (Path(c[0]) if c else None)


def main():
    pack_repo = find_pack_repo()
    repo = WORK / "repo"
    if not repo.exists():
        shutil.copytree(pack_repo, repo)
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "scripts"))
    print(f"repo -> {repo}", flush=True)

    # Patch the pool_kernel_um train/serve trap (3.0 default -> 5.0) in the writable copy.
    pf = repo / "scripts" / "predict_unet_transformer.py"
    src = pf.read_text()
    patched = src.replace("pool_kernel_um: float = 3.0", f"pool_kernel_um: float = {POOL_KERNEL_UM}")
    assert patched != src, "pool_kernel_um patch target not found — check the predict script version!"
    pf.write_text(patched)
    print(f"Patched PredictConfig.pool_kernel_um -> {POOL_KERNEL_UM}", flush=True)

    deps = ["tracksdata", "geff>=1.1.3.1.1", "zarr>=3.0.10,<4", "polars>=1.36",
            "numcodecs>=0.13", "blosc2", "imagecodecs", "rustworkx>=0.17.1", "tqdm"]
    if USE_ILP:
        deps += ["pyscipopt", "ilpy>=0.5.1"]
    sh([sys.executable, "-m", "pip", "install", "-q", *deps])

    data_dir = gather_data_dir()
    corrupt = load_corrupt()
    held_fam = {0: "44b6", 1: "6bba"}[FOLD]
    held = sorted(p.stem for p in data_dir.glob(f"{held_fam}_*.geff")
                  if (data_dir / f"{p.stem}.zarr").exists() and p.stem not in corrupt)
    print(f"FOLD {FOLD}: scoring {len(held)} held-out {held_fam} crops "
          f"({len(corrupt & set(p.stem for p in data_dir.glob(held_fam+'_*.geff')))} corrupt excluded)", flush=True)

    # predict uses folds[FOLD]['test']; evaluate_run reads DATA_DIR/dataset_splits.json -> put it there.
    folds = [{"train": [], "test": []}, {"train": [], "test": []}]
    folds[FOLD]["test"] = held
    splits = data_dir / "dataset_splits.json"
    splits.write_text(json.dumps(folds))

    # stage weights + config in one dir (load_model reads config.json next to the weights).
    wpath, cpath = find_weights()
    wdir = WORK / "wts"
    wdir.mkdir(exist_ok=True)
    shutil.copy(wpath, wdir / "edge_predictor.pth")
    if cpath:
        shutil.copy(cpath, wdir / "config.json")
    print(f"weights -> {wpath.name}", flush=True)

    pypath = os.pathsep.join([str(repo / "src"), str(repo / "scripts"), os.environ.get("PYTHONPATH", "")])
    env = dict(os.environ, BIOHUB_DATA_DIR=str(data_dir), PYTHONUNBUFFERED="1", PYTHONPATH=pypath)
    cmd = [sys.executable, "-u", str(pf), "--split", str(FOLD),
           "--data-dir", str(data_dir), "--splits", str(splits),
           "--weights", str(wdir / "edge_predictor.pth"),
           "--det-threshold", str(DET_THRESHOLD), "--evaluate"]
    if USE_ILP:
        cmd += ["--use-ilp", "--ilp-division-weight", str(ILP_DIVISION_WEIGHT)]
    print("\n===== PREDICT + SCORE (the Evaluation line below = the clean OOF number) =====", flush=True)
    sh(cmd, env=env)

    # export the predicted geffs so the OOF can be re-scored offline with the hardened biotrack.metric.
    pred = list((repo / "predictions").rglob(f"split_{FOLD}"))
    for d in pred:
        dest = WORK / f"pred_geffs_split_{FOLD}"
        if not dest.exists():
            shutil.copytree(d, dest)
            print(f"Predicted geffs -> {dest}", flush=True)
    print(f"\nDONE. The 'Evaluation (... score=X ...)' line above is the FOLD {FOLD} OOF on held-out "
          f"{held_fam}. Repeat with FOLD={1 - FOLD}. Flip USE_ILP=True for the global ILP once confirmed.", flush=True)


if __name__ == "__main__":
    main()
