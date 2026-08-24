"""Final cell for the H1-R S5 association-training kernels.

The preceding notebook cells embed h1r_edge_data and h1r_edge_train. This cell
verifies both Zh001r mounts, resolves the public full-model checkpoint, runs the
configured training arm, and publishes only training artifacts. It never submits.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch


if not torch.cuda.is_available():
    raise RuntimeError("H1-R S5 requires a Kaggle GPU; refusing a silent CPU run")
print("CUDA:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

edge_assets = discover_edge_assets([Path("/kaggle/input")])
iso_hits = sorted(Path("/kaggle/input").rglob("zh001r_iso.npy"))
if len(iso_hits) != 1:
    raise RuntimeError(f"expected exactly one zh001r_iso.npy, got {iso_hits}")
print("S5_ASSETS", json.dumps({
    "nodes": str(edge_assets.nodes),
    "identity": str(edge_assets.identity),
    "iso": str(iso_hits[0]),
}, sort_keys=True))

trainer_path = REPO_DIR / "scripts" / "train_unet_transformer.py"
init_weights = (
    REPO_DIR / "weights" / "unet_transformer" / "split_0" /
    "edge_predictor_best.pth"
)
for required in (trainer_path, init_weights):
    if not required.is_file():
        raise FileNotFoundError(required)

env = os.environ.get
out = Path("/kaggle/working/h1r_edge")
args = SimpleNamespace(
    trainer_dir=str(trainer_path.parent),
    root=["/kaggle/input"],
    weights=str(init_weights),
    out=str(out),
    epochs=int(env("H1R_EDGE_EPOCHS", "20")),
    batch_size=int(env("H1R_EDGE_BS", "1")),
    node_cap=int(env("H1R_EDGE_NODE_CAP", "256")),
    val_mod=int(env("H1R_EDGE_VAL_MOD", "6")),
    seed=int(env("H1R_EDGE_SEED", "0")),
    lr=float(env("H1R_EDGE_LR", "1e-4")),
    max_steps=int(env("H1R_EDGE_MAX_STEPS", "0")),
    appearance_dim=int(env("H1R_APPEARANCE_DIM", "64")),
    triplet_weight=float(env("H1R_TRIPLET_WEIGHT", "0.05")),
    resume=env("H1R_EDGE_RESUME", "1") == "1",
    amp=env("H1R_EDGE_AMP", "1") == "1",
)
result = run(args)
summary = {
    "best": result["best"],
    "epochs_requested": args.epochs,
    "max_steps": args.max_steps,
    "appearance_dim": args.appearance_dim,
    "triplet_weight": args.triplet_weight,
    "nodes": str(edge_assets.nodes),
    "identity": str(edge_assets.identity),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2))

for name in (
    "edge_predictor_best.pth", "appearance_projector_best.pth",
    "edge_resume.pth", "config.json", "metrics.json", "summary.json",
):
    source = out / name
    if source.is_file():
        shutil.copy2(source, Path("/kaggle/working") / name)

print("H1R_EDGE_RESULT", json.dumps(summary, sort_keys=True))
