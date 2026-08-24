"""Final cell for the H1-R detector training kernels.

The patch and driver are embedded in earlier notebook cells. This cell resolves mounted
datasets without assuming Kaggle's directory layout, applies the trainer patch to the
ephemeral repository copy, runs S1, and exposes the promotion artifacts at output root.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch


if not torch.cuda.is_available():
    raise RuntimeError("H1-R requires a Kaggle GPU; refusing a silent CPU training run")
print("CUDA:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])


def _unique_parent(name: str) -> Path:
    hits = sorted(Path("/kaggle/input").rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name!r} under /kaggle/input, got {hits}")
    return hits[0].parent


zh_root = _unique_parent("zh001r_iso.npy")
nodes_root = _unique_parent("zh001r_nodes.npz")
if nodes_root != zh_root:
    raise RuntimeError({"iso_root": str(zh_root), "nodes_root": str(nodes_root)})

trainer_path = REPO_DIR / "scripts" / "train_unet_transformer.py"
init_weights = REPO_DIR / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
for required in (trainer_path, init_weights, zh_root / "zh001r_nodes.npz"):
    if not required.is_file():
        raise FileNotFoundError(required)

apply_h1r_trainer_patch(trainer_path)
os.environ["H1R_ROOT"] = str(zh_root)
os.environ["H1R_INIT_WEIGHTS"] = str(init_weights)
os.environ["H1R_OUT"] = "/kaggle/working/h1r_detector"

result = run_h1r_detector(
    trainer_dir=trainer_path.parent,
    root=zh_root,
    out_dir=Path(os.environ["H1R_OUT"]),
)

out = Path(os.environ["H1R_OUT"])
for name in ("edge_predictor_best.pth", "detector_last.pth", "config.json",
             "metrics.json", "summary.json", "amp_benchmark.json"):
    source = out / name
    if source.is_file():
        shutil.copy2(source, Path("/kaggle/working") / name)

print("H1R_RESULT", json.dumps(result, sort_keys=True))
