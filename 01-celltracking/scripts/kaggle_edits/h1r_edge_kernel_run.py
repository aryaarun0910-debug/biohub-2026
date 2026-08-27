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


def _find_input(pattern: str) -> list[str]:
    """Resolve an input glob at whatever depth Kaggle mounted the dataset (owner-qualified
    mounts put datasets at /kaggle/input/datasets/<owner>/<slug>/...). Bounded, never
    recursive: a recursive walk would descend the competition zarr tree."""
    import glob as _glob
    hits = list(_glob.glob(pattern))
    base = Path(pattern).name
    for depth in (1, 2, 3, 4):
        hits += _glob.glob("/kaggle/input/" + "*/" * depth + base)
    return sorted(set(hits))


# Initialisation weights. Default: the pack's split_0 (the smoke's contract). A full run that
# will be JUDGED on a LOEO fold must initialise from the out-of-fold weights for that fold
# (FACT-0329): set H1R_EDGE_INIT_WEIGHTS_GLOB + H1R_EDGE_INIT_CONFIG_GLOB and attach the
# oof-weights dataset. Fail closed on anything but exactly one hit for each - a silent
# fallback to the pack here would reproduce the EXP-0019 class of leak one stage upstream.
_init_glob = os.environ.get("H1R_EDGE_INIT_WEIGHTS_GLOB", "").strip()
_init_cfg_glob = os.environ.get("H1R_EDGE_INIT_CONFIG_GLOB", "").strip()
if bool(_init_glob) != bool(_init_cfg_glob):
    raise RuntimeError("H1R_EDGE_INIT_WEIGHTS_GLOB and H1R_EDGE_INIT_CONFIG_GLOB must be set together")
if _init_glob:
    import hashlib
    _w_hits = _find_input(_init_glob)
    _c_hits = _find_input(_init_cfg_glob)
    if len(_w_hits) != 1 or len(_c_hits) != 1:
        raise RuntimeError(
            f"init weights glob {_init_glob!r} matched {_w_hits}; config glob "
            f"{_init_cfg_glob!r} matched {_c_hits} - exactly one hit each is required"
        )
    _stage = Path("/kaggle/working/h1r_init")
    _stage.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_w_hits[0], _stage / "edge_predictor_best.pth")
    shutil.copy2(_c_hits[0], _stage / "config.json")
    init_weights = _stage / "edge_predictor_best.pth"
    print("H1R_INIT_WEIGHTS", json.dumps({
        "source": _w_hits[0],
        "config": _c_hits[0],
        "sha256": hashlib.sha256(init_weights.read_bytes()).hexdigest(),
        "origin": "override",
    }, sort_keys=True))
else:
    init_weights = (
        REPO_DIR / "weights" / "unet_transformer" / "split_0" /
        "edge_predictor_best.pth"
    )
    print("H1R_INIT_WEIGHTS", json.dumps({"source": str(init_weights), "origin": "pack_default"}, sort_keys=True))
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
    "pos_encoding": POS_ENCODING,
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

# Uniquely-named copies for the LOEO consumer (deploy_h1r_edge_s5_loeo_f0.json): loeo_retarget.py
# resolves BIOHUB_LOEO_WEIGHTS_GLOB with a depth ladder on the BASENAME, and the pack also
# ships an `edge_predictor_best.pth`, so the plain name would match twice and fail closed.
# The rope4d arm exports under its OWN basenames so deploy_h1r_edge_s5_rope4d_loeo_f0.json can
# never pick up a sinusoidal checkpoint (and vice versa); POS_ENCODING comes from the trainer cell.
_unique_names = {
    "sinusoidal": ("edge_predictor_best_h1r_s5.pth", "config_h1r_s5.json"),
    "rope4d": ("edge_predictor_best_h1r_s5_rope4d.pth", "config_h1r_s5_rope4d.json"),
}[POS_ENCODING]
for src_name, unique_name in (
    ("edge_predictor_best.pth", _unique_names[0]),
    ("config.json", _unique_names[1]),
):
    source = out / src_name
    if source.is_file():
        shutil.copy2(source, Path("/kaggle/working") / unique_name)

print("H1R_EDGE_RESULT", json.dumps(summary, sort_keys=True))
