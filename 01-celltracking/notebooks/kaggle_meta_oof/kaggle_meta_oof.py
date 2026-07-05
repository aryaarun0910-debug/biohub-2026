"""Kaggle kernel: measure the PUBLIC 50ep meta weights' clean OOF on a held-out embryo = the SCALE / BAR.

The number that answers "is our 0.559 far behind, or is the OOF scale just brutal?". Runs the public pack's
own checkpoint (pilkwang pack, weights/unet_transformer/split_0/edge_predictor_best.pth) with the ILP (the
meta's real config) on the SAME 20 held-out 6bba crops our fold-1 uses, exports geffs -> score locally with
score_oof.py and compare head-to-head.

CAVEAT: the pack's single split_0 training set is undisclosed; it may be partly in-sample for 6bba, which
would INFLATE this number. So treat it as a scale anchor / upper-ish reference, not a clean held-out for
the meta. Still tells us the metric scale and roughly how far our model is.
"""
import glob, json, os, re, shutil, subprocess, sys
from pathlib import Path

FOLD = 1                 # score held-out 6bba (matches our fold-1)
SUBSET = 20              # same 20 crops as the ablation, for a fair head-to-head
POOL_KERNEL_UM = 5.0
WORK = Path("/kaggle/working")

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
    pack = Path(hits[0]).parents[1]
    repo = WORK / "repo"
    if not repo.exists():
        shutil.copytree(pack, repo)
    sys.path.insert(0, str(repo / "src")); sys.path.insert(0, str(repo / "scripts"))

    pf = repo / "scripts" / "predict_unet_transformer.py"
    s = pf.read_text()
    assert "pool_kernel_um: float = 3.0" in s, "pool_kernel_um patch target missing"
    pf.write_text(s.replace("pool_kernel_um: float = 3.0", f"pool_kernel_um: float = {POOL_KERNEL_UM}"))

    sh([sys.executable, "-m", "pip", "install", "-q", "tracksdata", "geff>=1.1.3.1.1", "zarr>=3.0.10,<4",
        "polars>=1.36", "numcodecs>=0.13", "blosc2", "imagecodecs", "rustworkx>=0.17.1", "tqdm"])

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
                  if (data / f"{p.stem}.zarr").exists() and p.stem not in corrupt)[:SUBSET]
    folds = [{"train": [], "test": []}, {"train": [], "test": []}]
    folds[FOLD]["test"] = held
    splits = data / "dataset_splits.json"; splits.write_text(json.dumps(folds))
    print(f"META OOF: {len(held)} held-out {held_fam} crops.", flush=True)

    # the PACK's own checkpoint (public meta). RECURSIVE glob under the pack mount root (found_pack_repo
    # is recursive and works, so the pack nests deeper than a fixed-depth glob expects). load_model reads
    # config.json from the weights' parent dir.
    mount_root = Path(hits[0]).parents[2]   # .../<pack>/[nesting]/repo/scripts/predict.py -> pack root
    w = glob.glob(str(mount_root / "**" / "edge_predictor_best.pth"), recursive=True)
    if not w:
        seen = glob.glob(str(mount_root / "**" / "*.pth"), recursive=True)[:20]
        raise SystemExit(f"pack meta weights not found under {mount_root}; .pth seen: {seen}")
    print(f"meta weights: {w[0]}", flush=True)

    pypath = os.pathsep.join([str(repo / "src"), str(repo / "scripts"), os.environ.get("PYTHONPATH", "")])
    env = dict(os.environ, BIOHUB_DATA_DIR=str(data), PYTHONUNBUFFERED="1", PYTHONPATH=pypath)
    # GREEDY (no --use-ilp) -> like-for-like vs our greedy 0.559, and avoids the big-crop ILP OOM.
    sh([sys.executable, "-u", str(pf), "--split", str(FOLD), "--method", "meta",
        "--data-dir", str(data), "--splits", str(splits), "--weights", w[0],
        "--det-threshold", "0.99", "--evaluate"], env=env)

    for d in (repo / "predictions").rglob(f"split_{FOLD}"):
        dest = WORK / f"meta_geffs_split_{FOLD}"
        if not dest.exists():
            shutil.copytree(d, dest); print(f"meta geffs -> {dest}", flush=True)
    print("\nDONE. Score meta_geffs locally: scripts/score_oof.py --pred-dir <meta dir> --gt-dir data/train "
          "-> the SCALE/BAR to compare our fold-1 OOF against.", flush=True)


if __name__ == "__main__":
    main()
