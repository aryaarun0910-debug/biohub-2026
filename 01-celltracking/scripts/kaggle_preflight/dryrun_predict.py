"""Local CPU dry-run of the PREDICT + SCORE path (never run anywhere) — validates model load from the
trained checkpoint, the pool_kernel_um=5.0 patch, detection/linking/graph-build, and --evaluate scoring."""
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import torch
torch.cuda.synchronize = lambda *a, **k: None

SCR = Path(__file__).resolve().parent
REPO = SCR / "packfull" / "repo"

# apply the same pool_kernel_um 3.0 -> 5.0 patch the kernel does, BEFORE import
pf = REPO / "scripts" / "predict_unet_transformer.py"
s = pf.read_text()
assert "pool_kernel_um: float = 3.0" in s, "patch target missing"
pf.write_text(s.replace("pool_kernel_um: float = 3.0", "pool_kernel_um: float = 5.0"))

sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
DATA = Path(r"c:/Users/aryaa/Documents/Biohub-CellTracking-2026/data/train")
os.environ["BIOHUB_DATA_DIR"] = str(DATA)

import predict_unet_transformer as P
assert P.PredictConfig().pool_kernel_um == 5.0, "pool_kernel_um patch did NOT take"
print("pool_kernel_um patch verified: PredictConfig default = 5.0", flush=True)

splits = SCR / "tiny_predict_splits.json"
splits.write_text(json.dumps([{"train": [], "test": ["6bba_05b6850b"]}]))
wpath = REPO / "weights" / "unet_transformer" / "split_0" / "edge_predictor_last.pth"
assert wpath.exists(), "no trained checkpoint from the train dry-run"

print(">>> running predict() on 1 crop, det_threshold=0.99, greedy, --evaluate, CPU ...", flush=True)
P.predict(
    data_dir=DATA, fold=0, splits_file=splits, weights_path=wpath,
    cfg=P.PredictConfig(det_threshold=0.99), evaluate=True,
)
print("DRYRUN_PREDICT_OK")
