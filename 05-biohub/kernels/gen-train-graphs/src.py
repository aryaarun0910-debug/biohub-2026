# Generate linked prediction graphs for the TRAIN split using the public 0.947 model.
# Train has ground truth, so these graphs let us sweep division selection offline
# against our own LOEO folds instead of spending 2-hour submissions on each variant.
import json, os, subprocess, sys, shutil, time
from pathlib import Path

OUT = Path("/kaggle/working")

# The dataset mount name is not guaranteed; find the pack by its CONTENTS instead of guessing.
def find_pack():
    # mounts are nested: /kaggle/input/datasets/<owner>/<name>/... , so search by CONTENTS
    root = Path("/kaggle/input")
    for depth in range(1, 6):
        for cand in root.glob("/".join(["*"] * depth)):
            if cand.is_dir() and (cand / "repo").is_dir() and (cand / "weights").is_dir():
                return cand
    return None

print("INPUT mounts:", [str(p) for p in Path("/kaggle/input").glob("*")], flush=True)
PACK = find_pack()
print("PACK:", PACK, flush=True)
assert PACK is not None, "support pack not found under /kaggle/input"

# competition mount: the path differs between "competitions/<slug>" and "<slug>"
def find_train():
    root = Path("/kaggle/input")
    for depth in range(1, 6):
        for cand in root.glob("/".join(["*"] * depth)):
            if cand.is_dir() and cand.name == "train" and any(cand.glob("*.zarr")):
                return cand
    return None

TRAIN = find_train()
print("TRAIN mount:", TRAIN, flush=True)
assert TRAIN is not None, [str(p) for p in Path("/kaggle/input").glob("*")]

# offline deps from the pack's bundled wheels
wheels = PACK / "wheels"
if wheels.exists():
    req = PACK / "requirements-unet-ilp.txt"
    cmd = [sys.executable, "-m", "pip", "install", "--no-index", f"--find-links={wheels}", "-q"]
    cmd += (["-r", str(req)] if req.exists() else [str(w) for w in wheels.glob("*.whl")])
    print("installing offline deps...", flush=True)
    subprocess.run(cmd, check=False)

# the repo is read-only under /kaggle/input; copy it so weights can be linked in
repo = OUT / "repo"
if not repo.exists():
    shutil.copytree(PACK / "repo", repo)
wdst = repo / "weights"
if not wdst.exists():
    shutil.copytree(PACK / "weights", wdst)

# our leave-one-embryo-out folds, in their splits format. BARE STEMS: open_dataset strips a
# .zarr/.geff suffix but evaluate.py appends .geff to the same string.
stems = sorted(p.stem for p in TRAIN.glob("*.zarr"))
a = [s for s in stems if s.startswith("44b6")]
b = [s for s in stems if s.startswith("6bba")]
splits = OUT / "dataset_splits.json"
splits.write_text(json.dumps([{"train": b, "test": a}, {"train": a, "test": b}]))
print(f"datasets: {len(stems)}  44b6={len(a)}  6bba={len(b)}", flush=True)

env = dict(os.environ, BIOHUB_DATA_DIR=str(TRAIN),
           PYTHONPATH=f"{repo/'src'}:{repo/'scripts'}")
for fold in (0, 1):
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, str(repo / "scripts" / "predict_unet_transformer.py"),
         "--data-dir", str(TRAIN), "--splits", str(splits), "--split", str(fold),
         "--weights", str(wdst / "unet_transformer/split_0/edge_predictor_best.pth"),
         "--unet-batch-size", "4", "--det-threshold", "0.965"],
        cwd=repo, env=env, capture_output=True, text=True)
    print(f"fold {fold}: rc={r.returncode} in {time.time()-t0:.0f}s", flush=True)
    print(r.stdout[-2500:], flush=True)
    if r.returncode: print("STDERR:", r.stderr[-4000:], flush=True)

# collect the prediction graphs into one archive for download
preds = list((repo / "predictions").rglob("*.geff"))
print("prediction geffs:", len(preds), flush=True)
if preds:
    root = repo / "predictions"
    shutil.make_archive(str(OUT / "train_pred_graphs"), "zip", root_dir=root)
    print("archived:", (OUT / "train_pred_graphs.zip").stat().st_size / 1e6, "MB", flush=True)
shutil.rmtree(repo, ignore_errors=True)      # keep only the archive in the output
