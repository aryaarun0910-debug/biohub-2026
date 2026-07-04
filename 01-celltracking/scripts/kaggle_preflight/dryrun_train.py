"""Local CPU dry-run of the PATCHED trainer on real local crops — validates the full train path
(imports, open_dataset, windows, model fwd/bwd, det+edge loss, eval, BOTH checkpoint saves) with no GPU."""
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import torch
torch.cuda.synchronize = lambda *a, **k: None  # CPU has no CUDA; the trainer calls this unconditionally

SCR = Path(__file__).resolve().parent
REPO = SCR / "packfull" / "repo"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
DATA = Path(r"c:/Users/aryaa/Documents/Biohub-CellTracking-2026/data/train")
os.environ["BIOHUB_DATA_DIR"] = str(DATA)

splits = SCR / "tiny_train_splits.json"
splits.write_text(json.dumps([{"train": ["6bba_07477033", "6bba_062c8d37"], "test": ["6bba_05b6850b"]}]))

import train_unet_transformer as T

print(">>> running train() 1 epoch, 2 iters, 3 crops, max_frames=40, CPU ...", flush=True)
T.train(
    data_dir=DATA, fold=0, splits_file=splits,
    n_epochs=1, max_iters=2, batch_size=1, num_workers=0,
    max_frames=40, pool_kernel_um=5.0, data_parallel=False, augmentations=None, seed=0,
)

out = REPO / "weights" / "unet_transformer" / "split_0"
best, last = out / "edge_predictor_best.pth", out / "edge_predictor_last.pth"
print("\n=== RESULT ===")
print("best checkpoint saved:", best.exists())
print("last checkpoint saved (MY PATCH):", last.exists())
print("config:", (out / "config.json").read_text() if (out / "config.json").exists() else "MISSING")
print("DRYRUN_TRAIN_OK" if (best.exists() and last.exists()) else "DRYRUN_TRAIN_FAIL")
