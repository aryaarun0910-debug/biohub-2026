r"""H1-R detector-half trainer -- dense-Zebrahub detection training on `kkunizaw/biohub-zh001r`.

Implements recipe B1-Stage-1 for the DETECTOR half only (the packaged nodes carry no track
identity -- verified by `scripts/win_bet/h1r_zh001r_audit.py` -- so the edge head cannot be
supervised here; its weights ride along untouched and the saved checkpoint stays loadable by
the deployed `predict_unet_transformer.load_model`).

Requires `h1r_trainer_patch.apply_h1r_trainer_patch` to have been applied to the
train_unet_transformer.py COPY on sys.path first (AMP inside the UNet forward, GradScaler
pattern; this driver has its own loop but reuses the patched module's primitives:
TemporalUNet3D / UNetNodeTransformer / compute_detection_loss / detect_and_match).

Checkpoint selection is detection F1 on held-out crops at the DEPLOYED operating point
(sigmoid 0.96875 == BIOHUB_DET_THRESHOLD in the P0-B kernel; logit 3.434), which is the A3 fix:
the vendored `acc*recall` score has no precision term and selects the junkiest detector.

The default trunk contract is deliberately conservative: the shared UNet (including BatchNorm
buffers) is frozen and kept in eval mode, so detector-head tuning cannot silently rewrite the
features consumed by the already-trained association transformer.  `adabn_control` is a
diagnostic-only contract that requires LR=0 and permits BatchNorm running-stat adaptation.

All knobs are env-driven so kernel smoke/pilot specs differ only in their env cell:
  H1R_ROOT            dir holding zh001r_iso.npy / zh001r_nodes.npz
  H1R_OUT             output dir (checkpoints + metrics.json)
  H1R_INIT_WEIGHTS    public 50-ep full checkpoint (edge_predictor_best.pth); "" = scratch
  H1R_EPOCHS / H1R_BS / H1R_LR / H1R_NEG_WEIGHT / H1R_MAX_STEPS (0 = full epoch)
  H1R_VAL_MOD         crop_index % H1R_VAL_MOD == 0 -> validation crop (default 6 -> 12/72)
  H1R_AMP             shared with the trainer patch (default 1)
  H1R_ROT90           in-plane 90-degree rotation augmentation (default 1; recipe section 5:
                      grid is isotropic but the PSF is not -- xy rotations only, never z-swaps)
  H1R_RESUME          resume from H1R_OUT/detector_last.pth if present (default 1)

Local CLI (CPU smoke):
  .venv\Scripts\python.exe scripts\kaggle_edits\h1r_det_train.py --trainer-dir <dir-with-patched-trainer> \
      --root <zh001r-dir> --out /tmp/h1r --epochs 1 --max-steps 5
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ZH_VOX_UM = 1.625          # exact registration geometry; supersedes the old 1.677 um ruler
DEPLOYED_SIGMOID_THR = 0.96875   # BIOHUB_DET_THRESHOLD in the deployed P0-B kernel env cell
POOL_KERNEL_UM = 3.0       # deployed PredictConfig.pool_kernel_um
MATCH_UM = 5.0             # matches trainer detect_and_match default


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def eval_thresholds(raw: str) -> dict[str, float]:
    """Parse probability operating points into named logit thresholds."""
    probs = sorted({float(x.strip()) for x in raw.split(",") if x.strip()})
    if not probs or any(not 0.0 < p < 1.0 for p in probs):
        raise ValueError(f"H1R_EVAL_PROBS must contain probabilities in (0,1), got {raw!r}")
    if DEPLOYED_SIGMOID_THR not in probs:
        probs.append(DEPLOYED_SIGMOID_THR)
        probs.sort()
    return {
        ("deployed" if p == DEPLOYED_SIGMOID_THR else f"p{p:g}"): _logit(p)
        for p in probs
    }


def best_operating_point(val: dict[str, dict]) -> tuple[str, dict]:
    """Diagnostic best point in a sweep; never used for checkpoint promotion."""
    return max(val.items(), key=lambda item: (item[1]["f1"], item[1]["precision"]))


def selection_at_threshold(val: dict[str, dict], name: str = "deployed") -> tuple[str, dict]:
    """Return the predeclared deployment operating point, failing closed if absent."""
    if name not in val:
        raise KeyError(f"selection threshold {name!r} missing from {sorted(val)}")
    return name, val[name]


# =============================================================================
# Data
# =============================================================================

class Zh001rWindows(Dataset):
    """(crop, t0) -> W consecutive frames + padded GT node coords.

    Layout mirrors the trainer's batch dict subset the detector loss needs:
    imgs (W, 64,64,64) float32 quantile-normalised; coords (W, M, 3); masks (W, M).
    Intensity: per-crop 0.1%/99.9% quantile min-max with clamp(0) -- the same robust
    scheme the deployed loader applies to competition uint16 volumes
    (train_unet_transformer.FrameWindowDataset.__getitem__), applied to their uint8.
    """

    def __init__(self, root: Path, crop_ids: list[int], window: int = 2,
                 max_nodes: int | None = None, augment: bool = False,
                 rot90: bool = True):
        self.iso = np.load(Path(root) / "zh001r_iso.npy", mmap_mode="r")
        # Eager materialisation avoids sharing a live ZipFile handle across DataLoader
        # worker processes. The pack is small (~9 MB compressed) relative to the images.
        with np.load(Path(root) / "zh001r_nodes.npz", allow_pickle=False) as node_pack:
            self.nodes = {key: node_pack[key] for key in node_pack.files}
        self.window = window
        self.n_t = int(self.iso.shape[1])
        self.spatial = tuple(int(s) for s in self.iso.shape[2:])
        self.crop_ids = list(crop_ids)
        self.augment = augment
        self.rot90 = rot90
        self.items = [(c, t) for c in self.crop_ids
                      for t in range(self.n_t - window + 1)]
        self.qlo: dict[int, float] = {}
        self.qhi: dict[int, float] = {}
        for c in self.crop_ids:
            v = np.asarray(self.iso[c]).astype(np.float32)
            self.qlo[c] = float(np.quantile(v, 0.001))
            self.qhi[c] = float(np.quantile(v, 0.999))
        if max_nodes is None:
            max_nodes = max(int(self.nodes[f"f{c * self.n_t + t}"].shape[0])
                            for c in self.crop_ids for t in range(self.n_t))
        self.max_nodes = int(max_nodes)

    def __len__(self) -> int:
        return len(self.items)

    def _coords(self, c: int, t: int) -> np.ndarray:
        a = self.nodes[f"f{c * self.n_t + t}"][:, 1:].astype(np.float32)  # z, y, x
        inside = np.all((a >= 0) & (a < np.array(self.spatial)), axis=1)
        a = a[inside]
        return a[: self.max_nodes]

    def __getitem__(self, i: int) -> dict:
        c, t0 = self.items[i]
        W, M = self.window, self.max_nodes
        raw = np.asarray(self.iso[c, t0:t0 + W]).astype(np.float32)
        imgs_np = np.clip((raw - self.qlo[c]) / (self.qhi[c] - self.qlo[c] + 1e-6), 0.0, None)
        imgs = torch.from_numpy(imgs_np)
        coords = torch.zeros(W, M, 3)
        masks = torch.zeros(W, M, dtype=torch.bool)
        for w in range(W):
            a = self._coords(c, t0 + w)
            n = len(a)
            coords[w, :n] = torch.from_numpy(a)
            masks[w, :n] = True

        if self.augment:
            rng = np.random.default_rng()
            imgs = imgs + float(rng.uniform(-0.1, 0.1))            # brightness (vendored)
            flip = rng.random(3) < 0.5                              # 8 axis flips (vendored)
            dims = [1 + d for d, f in enumerate(flip) if f]
            if dims:
                imgs = imgs.flip(dims=dims)
                coords = coords.clone()
                for d in range(3):
                    if flip[d]:
                        coords[masks, d] = imgs.shape[1 + d] - coords[masks, d] - 1
            if self.rot90:
                k = int(rng.integers(0, 4))                         # in-plane 90-degree rots
                if k:
                    imgs = torch.rot90(imgs, k, dims=(2, 3))        # (W, z, y, x): rotate y,x
                    coords = coords.clone()
                    X = self.spatial[2]
                    for _ in range(k):
                        y = coords[..., 1].clone()
                        x = coords[..., 2].clone()
                        coords[..., 1] = X - 1 - x
                        coords[..., 2] = y
        return {"imgs": imgs, "coords": coords, "masks": masks}


def global_max_nodes(root: Path) -> int:
    """Maximum nodes in any frame; a train-only maximum can truncate validation GT."""
    with np.load(Path(root) / "zh001r_nodes.npz", allow_pickle=False) as nodes:
        return max(int(nodes[key].shape[0]) for key in nodes.files)


# =============================================================================
# Train / eval
# =============================================================================

def encode_detection_logits(model, imgs: torch.Tensor, use_tta: bool) -> list[torch.Tensor]:
    """Encode detector logits with the exact deployed eight-view planar transform set."""
    _, det_logits = model.encode(imgs)
    if not use_tta:
        return det_logits

    n_views = 1
    for dims in [(-1,), (-2,), (-2, -1)]:
        _, transformed = model.encode(imgs.flip(dims))
        for f in range(len(det_logits)):
            det_logits[f] = det_logits[f] + transformed[f].flip(dims)
        n_views += 1
    for k in (1, 3):
        _, transformed = model.encode(torch.rot90(imgs, k, dims=(-2, -1)))
        for f in range(len(det_logits)):
            det_logits[f] = det_logits[f] + torch.rot90(
                transformed[f], -k, dims=(-2, -1)
            )
        n_views += 1
    _, transformed = model.encode(imgs.transpose(-1, -2))
    for f in range(len(det_logits)):
        det_logits[f] = det_logits[f] + transformed[f].transpose(-1, -2)
    n_views += 1
    anti = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)
    _, transformed = model.encode(anti)
    for f in range(len(det_logits)):
        det_logits[f] = det_logits[f] + torch.rot90(
            transformed[f].transpose(-1, -2), -1, dims=(-2, -1)
        )
        det_logits[f] = det_logits[f] / (n_views + 1)
    return det_logits


@torch.no_grad()
def eval_detection(T, model, loader, device, thresholds: dict[str, float],
                   use_tta: bool = True) -> dict:
    """Detection P/R/F1 vs zh001r nodes at each logit threshold in *thresholds*.

    Non-overlapping eval windows count every frame exactly once. Matching is the
    trainer's own greedy one-to-one detect_and_match at MATCH_UM physical microns.
    """
    model.eval()
    stats = {k: {"det": 0, "gt": 0, "matched": 0} for k in thresholds}
    for batch in loader:
        imgs = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        B, W = imgs.shape[:2]
        det_logits = encode_detection_logits(model, imgs, use_tta=use_tta)
        for name, thr in thresholds.items():
            for i in range(W):
                _, _, det_m, matches = T.detect_and_match(
                    det_logits[i], coords[:, i], masks[:, i],
                    (W,) + tuple(imgs.shape[2:]),
                    det_threshold=thr,
                    pool_kernel_um=POOL_KERNEL_UM,
                    max_match_distance=MATCH_UM,
                    voxel_size=(ZH_VOX_UM,) * 3,
                    frame_index=i, window_size=W,
                )
                for b in range(B):
                    stats[name]["det"] += int(matches[b].shape[0])
                    stats[name]["gt"] += int(masks[b, i].sum().item())
                    stats[name]["matched"] += int((matches[b] >= 0).sum().item())
    out = {}
    for name, s in stats.items():
        p = s["matched"] / max(s["det"], 1)
        r = s["matched"] / max(s["gt"], 1)
        out[name] = {"precision": p, "recall": r,
                     "f1": 2 * p * r / max(p + r, 1e-9),
                     "n_det": s["det"], "n_gt": s["gt"]}
    return out


def _strip_dp(state: dict) -> dict:
    return {k.replace("unet.module.", "unet.", 1): v for k, v in state.items()}


def optimizer_step_index(optimizer: torch.optim.Optimizer) -> int:
    """Largest per-parameter Adam step; unlike LR, this advances only if AdamW ran."""
    steps: list[int] = []
    for state in optimizer.state.values():
        value = state.get("step")
        if value is not None:
            steps.append(int(value.item() if torch.is_tensor(value) else value))
    return max(steps, default=0)


def _train_step(T, model, batch, device, neg_weight, optimizer, scaler) -> dict:
    imgs = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
    coords = batch["coords"].to(device, non_blocking=True)
    masks = batch["masks"].to(device, non_blocking=True)
    B, W = imgs.shape[:2]
    _, det_logits = model.encode(imgs)
    loss = sum(
        T.compute_detection_loss(det_logits[i], coords[:, i], masks[:, i], neg_weight)
        for i in range(W)
    ) / W
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        (p for p in model.parameters() if p.requires_grad), 1.0
    )
    finite_gradients = bool(torch.isfinite(grad_norm).item())
    step_before = optimizer_step_index(optimizer)
    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    step_after = optimizer_step_index(optimizer)
    scale_after = float(scaler.get_scale())
    return {
        "loss": float(loss.detach()), "batch_size": B,
        "finite_loss": bool(torch.isfinite(loss.detach()).item()),
        "finite_gradients": finite_gradients,
        "grad_norm": float(grad_norm.detach()),
        "optimizer_step_before": step_before,
        "optimizer_step_after": step_after,
        "optimizer_step_occurred": step_after > step_before,
        "scaler_scale_before": scale_before,
        "scaler_scale_after": scale_after,
        "scaler_backoff": scale_after < scale_before,
    }


def configure_trunk_contract(model, contract: str, lr: float) -> None:
    """Apply the explicit shared-representation safety contract."""
    if contract not in {"freeze", "adabn_control", "joint_unsafe"}:
        raise ValueError(f"unknown H1R_TRUNK_CONTRACT={contract!r}")
    if contract == "adabn_control" and lr != 0.0:
        raise ValueError("adabn_control is diagnostic-only and requires H1R_LR=0")
    if contract == "freeze":
        for parameter in model.unet.parameters():
            parameter.requires_grad_(False)


def enforce_trunk_mode(model, contract: str) -> None:
    """`model.train()` also toggles frozen BN modules, so reassert eval mode each epoch."""
    if contract == "freeze":
        model.unet.eval()


def trunk_snapshot(model) -> dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu().clone()
        for k, v in _strip_dp(model.state_dict()).items()
        if k.startswith("unet.")
    }


def assert_trunk_unchanged(model, before: dict[str, torch.Tensor]) -> None:
    after = trunk_snapshot(model)
    if after.keys() != before.keys():
        raise RuntimeError("frozen trunk state keys changed")
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    if changed:
        raise RuntimeError(f"frozen trunk contract violated: {changed[:5]}")


def benchmark_amp(T, model, loader, device, neg_weight, lr, steps: int) -> dict:
    """Paired end-to-end fp32/fp16 timing on the exact DataParallel training path."""
    if device.type != "cuda" or steps <= 0:
        return {"skipped": True, "reason": "no CUDA or H1R_BENCH_STEPS=0"}
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all()
    timings = {}
    for use_amp in (False, True):
        model.load_state_dict(initial)
        T.TemporalUNet3D._h1r_amp_enabled = use_amp
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        iterator = iter(loader)
        model.train()
        torch.cuda.synchronize()
        t0 = time.monotonic()
        done = 0
        while done < steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            _train_step(T, model, batch, device, neg_weight, optimizer, scaler)
            done += 1
        torch.cuda.synchronize()
        timings["fp16" if use_amp else "fp32"] = {
            "steps": done, "seconds": time.monotonic() - t0,
        }
        del optimizer, scaler
    model.load_state_dict(initial)
    torch.set_rng_state(rng_cpu)
    torch.cuda.set_rng_state_all(rng_cuda)
    fp32_s = timings["fp32"]["seconds"]
    fp16_s = timings["fp16"]["seconds"]
    timings["speedup_fp32_over_fp16"] = fp32_s / max(fp16_s, 1e-9)
    return timings


def run_h1r_detector(trainer_dir: Path, root: Path, out_dir: Path) -> dict:
    trainer_dir = Path(trainer_dir)
    # The vendored scripts are normally launched with PYTHONPATH=src by the inference
    # notebook. Importing the trainer in-process must recreate both search roots.
    sys.path.insert(0, str(trainer_dir.parent / "src"))
    sys.path.insert(0, str(trainer_dir))
    import train_unet_transformer as T  # patched copy (h1r_trainer_patch applied)
    assert getattr(T.TemporalUNet3D, "_h1r_amp", False), \
        "apply h1r_trainer_patch to the trainer copy before importing it"

    env = os.environ.get
    epochs = int(env("H1R_EPOCHS", "20"))
    bs = int(env("H1R_BS", "16"))
    lr = float(env("H1R_LR", "1e-4"))
    neg_weight = float(env("H1R_NEG_WEIGHT", "0.1"))   # dense labels: 10x the deployed 1e-2
    max_steps = int(env("H1R_MAX_STEPS", "0"))
    val_mod = int(env("H1R_VAL_MOD", "6"))
    amp = env("H1R_AMP", "1") == "1" and torch.cuda.is_available()
    rot90 = env("H1R_ROT90", "1") == "1"
    resume = env("H1R_RESUME", "1") == "1"
    init_weights = env("H1R_INIT_WEIGHTS", "")
    num_workers = int(env("H1R_WORKERS", "3"))
    seed = int(env("H1R_SEED", "0"))
    bench_steps = int(env("H1R_BENCH_STEPS", "50"))
    min_amp_speedup = float(env("H1R_MIN_AMP_SPEEDUP", "1.3"))
    eval_tta = env("H1R_EVAL_TTA", "1") == "1"
    selection_threshold = env("H1R_SELECT_THRESHOLD", "deployed")
    trunk_contract = env("H1R_TRUNK_CONTRACT", "freeze")
    require_optimizer_steps = env("H1R_REQUIRE_OPT_STEPS", "1") == "1"
    patience = int(env("H1R_PATIENCE", "3"))
    min_delta = float(env("H1R_MIN_DELTA", "1e-4"))
    thresholds = eval_thresholds(env(
        "H1R_EVAL_PROBS", "0.50,0.75,0.90,0.95,0.96875,0.98,0.99"
    ))

    torch.manual_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # predict_unet_transformer.load_model reads config.json next to the weights;
    # write it so the retrained checkpoint is deploy-loadable as-is.
    (out_dir / "config.json").write_text(json.dumps({
        "unet_out_channels": 32, "unet_layers": [32, 64, 128],
        "downsample": [1, 4, 4], "window_size": 2, "pool_kernel_um": 3.0,
    }, indent=2))

    iso = np.load(Path(root) / "zh001r_iso.npy", mmap_mode="r")
    n_crop = int(iso.shape[0])
    val_ids = [c for c in range(n_crop) if c % val_mod == 0]
    train_ids = [c for c in range(n_crop) if c % val_mod != 0]
    print(f"zh001r: {n_crop} crops -> {len(train_ids)} train / {len(val_ids)} val "
          f"(val = idx % {val_mod} == 0)", flush=True)

    max_nodes = global_max_nodes(root)
    train_ds = Zh001rWindows(root, train_ids, max_nodes=max_nodes,
                             augment=True, rot90=rot90)
    val_ds = Zh001rWindows(root, val_ids, augment=False, max_nodes=max_nodes)
    # Non-overlapping val windows: stride = window so each frame is counted once.
    val_ds.items = [(c, t) for c in val_ids for t in range(0, val_ds.n_t - 1, 2)]
    print(f"windows: train={len(train_ds)} val={len(val_ds)} max_nodes={train_ds.max_nodes}",
          flush=True)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=num_workers, pin_memory=False,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=num_workers, pin_memory=False,
                            persistent_workers=num_workers > 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = T.TemporalUNet3D(in_channels=1, out_channels=32, layers=[32, 64, 128])
    model = T.UNetNodeTransformer(unet=unet, unet_out_channels=32,
                                  pos_feat_dim=4 * T._POS_EMBED_DIM)
    if init_weights:
        state = torch.load(init_weights, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"init from {init_weights}: {len(missing)} missing, "
              f"{len(unexpected)} unexpected", flush=True)
        if missing or unexpected:
            raise RuntimeError({"missing": missing, "unexpected": unexpected})
    model.to(device)
    configure_trunk_contract(model, trunk_contract, lr)

    start_epoch = 0
    history: list[dict] = []
    last_path = out_dir / "detector_last.pth"
    resume_ck = None
    if resume and last_path.exists():
        resume_ck = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(resume_ck["model"])
        start_epoch = int(resume_ck["epoch"]) + 1
        history = json.loads((out_dir / "metrics.json").read_text()) \
            if (out_dir / "metrics.json").exists() else []
        print(f"resumed at epoch {start_epoch} from {last_path}", flush=True)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model.unet = torch.nn.DataParallel(model.unet)
        print(f"DataParallel UNet across {torch.cuda.device_count()} GPUs", flush=True)
    # Snapshot after a possible resume load and DataParallel wrapping. The normalised
    # state keys still match deployment, and the guard now checks the state actually trained.
    frozen_trunk = trunk_snapshot(model) if trunk_contract == "freeze" else None

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("trunk contract left no trainable parameters")
    opt = torch.optim.AdamW(trainable, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if resume_ck is not None and "optimizer" in resume_ck:
        opt.load_state_dict(resume_ck["optimizer"])
        sched.load_state_dict(resume_ck["scheduler"])
        scaler.load_state_dict(resume_ck["scaler"])
        torch.set_rng_state(resume_ck["rng_cpu"].cpu())
        if device.type == "cuda" and resume_ck.get("rng_cuda"):
            torch.cuda.set_rng_state_all(resume_ck["rng_cuda"])
    else:
        for _ in range(start_epoch):
            sched.step()

    if start_epoch == 0:
        bench = benchmark_amp(T, model, train_loader, device, neg_weight, lr, bench_steps)
        T.TemporalUNet3D._h1r_amp_enabled = amp
        (out_dir / "amp_benchmark.json").write_text(json.dumps(bench, indent=2))
        print(f"AMP BENCHMARK: {json.dumps(bench, sort_keys=True)}", flush=True)
        if amp and not bench.get("skipped") \
                and bench["speedup_fp32_over_fp16"] < min_amp_speedup:
            raise RuntimeError(
                f"AMP speedup {bench['speedup_fp32_over_fp16']:.3f}x is below "
                f"H1R_MIN_AMP_SPEEDUP={min_amp_speedup:.3f}; aborting before full training")

    # The zero-epoch checkpoint is the scientific baseline and a safe fallback if every
    # fine-tuned epoch is worse. It is measured across a real operating-point sweep.
    if not history:
        baseline_val = eval_detection(
            T, model, val_loader, device, thresholds, use_tta=eval_tta
        )
        baseline_name, baseline_best = selection_at_threshold(
            baseline_val, selection_threshold
        )
        baseline = {"epoch": -1, "stage": "zero_epoch", "val": baseline_val,
                    "selection": {"threshold": baseline_name, **baseline_best}}
        history.append(baseline)
        torch.save(_strip_dp(model.state_dict()), out_dir / "edge_predictor_best.pth")
        (out_dir / "metrics.json").write_text(json.dumps(history, indent=1))
        print(f"zero epoch | best={baseline_name} P={baseline_best['precision']:.3f} "
              f"R={baseline_best['recall']:.3f} F1={baseline_best['f1']:.3f}", flush=True)

    best_f1 = max(
        (selection_at_threshold(h["val"], selection_threshold)[1]["f1"] for h in history),
        default=0.0,
    )
    stale_epochs = 0
    total_attempted_steps = 0
    total_optimizer_steps = 0
    total_finite_optimizer_steps = 0
    total_nonfinite_steps = 0
    epoch_audits: list[dict] = []
    for epoch in range(start_epoch, epochs):
        model.train()
        enforce_trunk_mode(model, trunk_contract)
        t0 = time.monotonic()
        tot, n = 0.0, 0
        epoch_step_records = []
        for step, batch in enumerate(train_loader):
            if max_steps and step >= max_steps:
                break
            step_record = _train_step(
                T, model, batch, device, neg_weight, opt, scaler
            )
            epoch_step_records.append(step_record)
            tot += step_record["loss"] * step_record["batch_size"]
            n += step_record["batch_size"]
        train_time = time.monotonic() - t0

        attempted = len(epoch_step_records)
        optimizer_steps = sum(r["optimizer_step_occurred"] for r in epoch_step_records)
        finite_optimizer_steps = sum(
            r["optimizer_step_occurred"] and r["finite_loss"] and r["finite_gradients"]
            for r in epoch_step_records
        )
        nonfinite = sum(not (r["finite_loss"] and r["finite_gradients"])
                        for r in epoch_step_records)
        total_attempted_steps += attempted
        total_optimizer_steps += optimizer_steps
        total_finite_optimizer_steps += finite_optimizer_steps
        total_nonfinite_steps += nonfinite
        audit = {
            "epoch": epoch, "attempted_steps": attempted,
            "optimizer_steps": optimizer_steps, "skipped_steps": attempted - optimizer_steps,
            "finite_optimizer_steps": finite_optimizer_steps,
            "nonfinite_steps": nonfinite,
            "final_optimizer_step_index": optimizer_step_index(opt),
            "initial_scaler_scale": (epoch_step_records[0]["scaler_scale_before"]
                                     if epoch_step_records else float(scaler.get_scale())),
            "final_scaler_scale": float(scaler.get_scale()),
            "max_grad_norm": max((r["grad_norm"] for r in epoch_step_records), default=0.0),
        }
        epoch_audits.append(audit)
        training_audit = {
            "contract": trunk_contract, "lr": lr, "eval_tta": eval_tta,
            "selection_threshold": selection_threshold,
            "attempted_steps": total_attempted_steps,
            "optimizer_steps": total_optimizer_steps,
            "finite_optimizer_steps": total_finite_optimizer_steps,
            "skipped_steps": total_attempted_steps - total_optimizer_steps,
            "nonfinite_steps": total_nonfinite_steps,
            "epochs": epoch_audits,
        }
        (out_dir / "training_audit.json").write_text(json.dumps(training_audit, indent=2))
        if require_optimizer_steps and attempted and finite_optimizer_steps == 0:
            raise RuntimeError(
                f"optimizer-step contract failed in epoch {epoch}: "
                f"0/{attempted} finite AdamW steps; "
                f"audit={out_dir / 'training_audit.json'}"
            )

        t0 = time.monotonic()
        val = eval_detection(T, model, val_loader, device, thresholds, use_tta=eval_tta)
        val_time = time.monotonic() - t0
        sched.step()

        d = val["deployed"]
        selected_name, selected = selection_at_threshold(val, selection_threshold)
        is_best = selected["f1"] > best_f1 + min_delta
        state = _strip_dp(model.state_dict())
        if frozen_trunk is not None:
            assert_trunk_unchanged(model, frozen_trunk)
        torch.save({
            "model": state, "epoch": epoch, "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }, last_path)
        if is_best:
            best_f1 = selected["f1"]
            torch.save(state, out_dir / "edge_predictor_best.pth")  # deploy-compatible name
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append({"epoch": epoch, "det_loss": tot / max(n, 1), "val": val,
                        "selection": {"threshold": selected_name, **selected},
                        "lr": sched.get_last_lr()[0],
                        "step_audit": audit,
                        "train_s": train_time, "val_s": val_time})
        (out_dir / "metrics.json").write_text(json.dumps(history, indent=1))
        print(f"epoch {epoch:3d} | det_loss={tot / max(n, 1):.4f} | "
              f"deployed-thr P={d['precision']:.3f} R={d['recall']:.3f} F1={d['f1']:.3f} "
              f"(n_det={d['n_det']} n_gt={d['n_gt']}) | "
              f"selected={selected_name} F1={selected['f1']:.3f} | "
              f"best={best_f1:.3f}{' *' if is_best else ''} | "
              f"train={train_time:.0f}s val={val_time:.0f}s", flush=True)
        if patience > 0 and stale_epochs >= patience:
            print(f"early stop after {stale_epochs} non-improving epochs", flush=True)
            break
    baseline_name, baseline_best = selection_at_threshold(
        history[0]["val"], selection_threshold
    )
    diagnostic_name, diagnostic_best = best_operating_point(history[-1]["val"])
    result = {
        "baseline": {"threshold": baseline_name, **baseline_best},
        "selection_threshold": selection_threshold,
        "best_f1_at_selection_threshold": best_f1,
        "diagnostic_best_final_sweep": {"threshold": diagnostic_name, **diagnostic_best},
        "improvement_over_baseline": best_f1 - baseline_best["f1"],
        "epochs_run": sum(h.get("stage") != "zero_epoch" for h in history),
        "trunk_contract": trunk_contract, "eval_tta": eval_tta,
        "training_audit": training_audit if epoch_audits else {
            "attempted_steps": 0, "optimizer_steps": 0,
        },
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    if os.environ.get("H1R_KERNEL") != "1":   # notebook cells run as __main__; stay inert
        import argparse

        ap = argparse.ArgumentParser()
        ap.add_argument("--trainer-dir", required=True,
                        help="dir holding the PATCHED train_unet_transformer.py copy")
        ap.add_argument("--root", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--epochs", type=int, default=None)
        ap.add_argument("--max-steps", type=int, default=None)
        ap.add_argument("--bs", type=int, default=None)
        a = ap.parse_args()
        if a.epochs is not None:
            os.environ["H1R_EPOCHS"] = str(a.epochs)
        if a.max_steps is not None:
            os.environ["H1R_MAX_STEPS"] = str(a.max_steps)
        if a.bs is not None:
            os.environ["H1R_BS"] = str(a.bs)
        run_h1r_detector(Path(a.trainer_dir), Path(a.root), Path(a.out))
