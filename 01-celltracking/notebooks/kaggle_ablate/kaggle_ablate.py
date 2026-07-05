"""Kaggle kernel: step-4 ablation sweep — tune the EXISTING knobs on a trained fold checkpoint.

Run AFTER we have a fold checkpoint + a baseline OOF. Codex step 4: on frozen learned detections, sweep
greedy-vs-ILP, detection threshold, and ILP division weight — the organizer stack already exposes all of
these (--det-threshold, --use-ilp, --ilp-division-weight), so no new machinery. This runs predict once per
grid config (separate --method so outputs don't collide), exports each config's geffs, and (optionally)
prints the pack's --evaluate number. Authoritative comparison = score each config's geffs locally with
scripts/score_oof.py and pick the best by clean OOF.

Attach: the pack + kms111201 data + BOTH training kernels (weights for either fold). NOT pushed until a
real checkpoint exists (a push auto-runs and burns a GPU session).
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
FOLD = 1                 # ablate the WEAK fold (1 -> held-out 6bba; greedy 0.559 + 10,786 division FPs)
SUBSET = 20              # score on the first N held-out crops for a FAST sweep (None = full embryo)
POOL_KERNEL_UM = 5.0
# (name, det_threshold, use_ilp, ilp_division_weight)
GRID = [
    ("greedy_t99",      0.99, False, 1.0),
    ("greedy_t95",      0.95, False, 1.0),
    ("ilp_div1.0_t99",  0.99, True,  1.0),
    ("ilp_div0.4_t99",  0.99, True,  0.4),
    ("ilp_div0.2_t99",  0.99, True,  0.2),
]
WORK = Path("/kaggle/working")
# ===============================================================

try:
    sys.stdout.reconfigure(line_buffering=True); sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


def sh(cmd, **kw):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def main():
    hits = glob.glob("/kaggle/input/**/repo/scripts/predict_unet_transformer.py", recursive=True)
    if not hits:
        raise SystemExit("Add 'pilkwang/biohub-tracking-support-pack-50ep-v1'.")
    repo = WORK / "repo"
    if not repo.exists():
        shutil.copytree(Path(hits[0]).parents[1], repo)
    sys.path.insert(0, str(repo / "src")); sys.path.insert(0, str(repo / "scripts"))

    # pool_kernel_um 3.0 -> 5.0 (train/serve match)
    pf = repo / "scripts" / "predict_unet_transformer.py"
    s = pf.read_text()
    assert "pool_kernel_um: float = 3.0" in s, "pool_kernel_um patch target missing"
    pf.write_text(s.replace("pool_kernel_um: float = 3.0", f"pool_kernel_um: float = {POOL_KERNEL_UM}"))

    need_ilp = any(c[2] for c in GRID)
    deps = ["tracksdata", "geff>=1.1.3.1.1", "zarr>=3.0.10,<4", "polars>=1.36",
            "numcodecs>=0.13", "blosc2", "imagecodecs", "rustworkx>=0.17.1", "tqdm"]
    if need_ilp:
        deps += ["pyscipopt", "ilpy>=0.5.1"]
    sh([sys.executable, "-m", "pip", "install", "-q", *deps])

    # data + corrupt-filtered held-out list (optionally subset for a fast sweep)
    data = WORK / "data"; data.mkdir(exist_ok=True)
    geffs = []
    for pat in ("/kaggle/input/*/train/*.geff", "/kaggle/input/*/*/train/*.geff", "/kaggle/input/*/*/*/train/*.geff"):
        geffs += glob.glob(pat)
    for g in sorted(set(geffs)):
        z = str(Path(g).with_suffix(".zarr"))
        if os.path.exists(z):
            for src in (g, z):
                link = data / Path(src).name
                if not link.exists():
                    try: os.symlink(src, link)
                    except OSError: (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, link)
    corrupt = set()
    for h in (glob.glob("/kaggle/input/*/metadata_files/corrupt_files.txt") + glob.glob("/kaggle/input/*/*/corrupt_files.txt")):
        corrupt |= set(re.findall(r"(?:44b6|6bba)_[0-9a-f]+", Path(h).read_text()))
    held_fam = {0: "44b6", 1: "6bba"}[FOLD]
    held = sorted(p.stem for p in data.glob(f"{held_fam}_*.geff")
                  if (data / f"{p.stem}.zarr").exists() and p.stem not in corrupt)
    if SUBSET:
        held = held[:SUBSET]
    folds = [{"train": [], "test": []}, {"train": [], "test": []}]
    folds[FOLD]["test"] = held
    splits = data / "dataset_splits.json"; splits.write_text(json.dumps(folds))
    print(f"Ablating FOLD {FOLD} on {len(held)} held-out {held_fam} crops; {len(GRID)} configs.", flush=True)

    w = (glob.glob(f"/kaggle/input/*/edge_predictor_best_split_{FOLD}.pth")
         + glob.glob(f"/kaggle/input/*/repo/weights/*/split_{FOLD}/edge_predictor_best.pth"))
    if not w:
        raise SystemExit(f"No weights for FOLD {FOLD}; run/attach the training kernel first.")
    c = (glob.glob(f"/kaggle/input/*/config_split_{FOLD}.json")
         + glob.glob(f"/kaggle/input/*/repo/weights/*/split_{FOLD}/config.json"))
    wdir = WORK / "wts"; wdir.mkdir(exist_ok=True)
    shutil.copy(w[0], wdir / "edge_predictor.pth")
    if c:
        shutil.copy(c[0], wdir / "config.json")

    pypath = os.pathsep.join([str(repo / "src"), str(repo / "scripts"), os.environ.get("PYTHONPATH", "")])
    env = dict(os.environ, BIOHUB_DATA_DIR=str(data), PYTHONUNBUFFERED="1", PYTHONPATH=pypath)

    for name, det, use_ilp, divw in GRID:
        print(f"\n===== CONFIG {name}: det={det} ilp={use_ilp} div_w={divw} =====", flush=True)
        cmd = [sys.executable, "-u", str(pf), "--split", str(FOLD), "--method", name,
               "--data-dir", str(data), "--splits", str(splits),
               "--weights", str(wdir / "edge_predictor.pth"),
               "--det-threshold", str(det), "--evaluate"]
        if use_ilp:
            cmd += ["--use-ilp", "--ilp-division-weight", str(divw)]
        try:
            sh(cmd, env=env)
        except subprocess.CalledProcessError as e:
            print(f"  CONFIG {name} FAILED: {e}", flush=True)
            continue
        for d in (repo / "predictions").rglob(f"split_{FOLD}"):
            if f"/{name}/" in d.as_posix():
                dest = WORK / f"abl_{name}_split_{FOLD}"
                if not dest.exists():
                    shutil.copytree(d, dest); print(f"  geffs -> {dest}", flush=True)

    print("\nDONE. Score each abl_<name>_split_%d/ locally: "
          "scripts/score_oof.py --pred-dir <abl dir> --gt-dir data/train ; pick best clean OOF." % FOLD, flush=True)


if __name__ == "__main__":
    main()
