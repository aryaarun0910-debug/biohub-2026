"""Runnable S5 Zh001r association/appearance trainer.

Uses registered identities from :mod:`h1r_edge_data` and the vendored
``UNetNodeTransformer`` primitives.  Detection is never supervised here: the
public detection head is frozen byte-for-byte while the UNet representation and
association transformer train on adjacent two-frame windows.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from h1r_edge_data import Zh001rEdgeData, build_transition_target, discover_edge_assets


DEPLOY_DOWNSAMPLE = (1.0, 4.0, 4.0)
DEFAULT_NODE_CAP = 256


def epoch_sample_indices(n: int, cap: int, *, seed: int, epoch: int,
                         crop: int, frame: int) -> np.ndarray:
    """Deterministic within an epoch, deliberately different between epochs."""
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    ss = np.random.SeedSequence([int(seed), int(epoch), int(crop), int(frame)])
    return np.sort(np.random.default_rng(ss).choice(n, cap, replace=False)).astype(np.int64)


class Zh001rEdgeWindows(Dataset):
    """Adjacent two-frame images, capped identities, and row-aligned targets."""

    def __init__(self, data: Zh001rEdgeData, iso_path: str | Path,
                 crop_ids: Iterable[int], *, node_cap: int = DEFAULT_NODE_CAP,
                 seed: int = 0):
        self.data = data
        self.iso = np.load(Path(iso_path), mmap_mode="r")
        expected = (data.n_crops, data.n_frames)
        if self.iso.ndim != 5 or tuple(self.iso.shape[:2]) != expected:
            raise ValueError(f"image/node corpus mismatch: images={self.iso.shape[:2]} nodes={expected}")
        self.crop_ids = tuple(sorted(int(c) for c in crop_ids))
        if not self.crop_ids or any(c not in data.crops for c in self.crop_ids):
            raise ValueError(f"invalid/empty crop selection: {self.crop_ids}")
        self.node_cap = int(node_cap)
        if self.node_cap < 1:
            raise ValueError("node_cap must be positive")
        self.seed = int(seed)
        self.epoch = 0
        self.items = [(c, f) for c in self.crop_ids for f in range(data.n_frames - 1)]
        self._quantiles: dict[int, tuple[float, float]] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def _select(self, crop: int, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fr = self.data.frame(crop, frame)
        take = epoch_sample_indices(len(fr.track_ids), self.node_cap, seed=self.seed,
                                    epoch=self.epoch, crop=crop, frame=frame)
        return fr.coords[take], fr.track_ids[take], fr.parent_track_ids[take]

    def __getitem__(self, index: int) -> dict:
        crop, frame = self.items[index]
        if crop not in self._quantiles:
            whole = np.asarray(self.iso[crop], dtype=np.float32)
            self._quantiles[crop] = (float(np.quantile(whole, 0.001)),
                                     float(np.quantile(whole, 0.999)))
        qlo, qhi = self._quantiles[crop]
        raw = np.asarray(self.iso[crop, frame:frame + 2], dtype=np.float32)
        imgs = np.clip((raw - qlo) / (qhi - qlo + 1e-6), 0.0, None)
        c0, tid0, _ = self._select(crop, frame)
        c1, tid1, pid1 = self._select(crop, frame + 1)
        target, _, _ = build_transition_target(tid0, tid1, pid1)
        # The negative mask is defined from the FULL source frame, not the sampled
        # source subset.  If the 256-node cap happens to drop a mother, her daughters
        # remain division daughters and must not silently become no-parent negatives.
        full_src_set = set(map(int, self.data.frame(crop, frame).track_ids))
        division_cols = np.array([
            int(p) in full_src_set and int(t) not in full_src_set
            for t, p in zip(tid1, pid1, strict=True)
        ], dtype=bool)
        continuation = target.clone()
        if len(division_cols):
            continuation[:, torch.from_numpy(division_cols)] = 0.0
        return {
            "crop": crop, "frame": frame, "imgs": torch.from_numpy(imgs),
            "coords0": torch.from_numpy(c0.copy()), "coords1": torch.from_numpy(c1.copy()),
            "tid0": torch.from_numpy(tid0.copy()), "tid1": torch.from_numpy(tid1.copy()),
            "target": continuation, "division_cols": torch.from_numpy(division_cols),
        }


def collate_edge_windows(rows: list[dict]) -> dict:
    b = len(rows); m0 = max(len(r["tid0"]) for r in rows); m1 = max(len(r["tid1"]) for r in rows)
    coords0 = torch.zeros(b, m0, 3); coords1 = torch.zeros(b, m1, 3)
    mask0 = torch.zeros(b, m0, dtype=torch.bool); mask1 = torch.zeros(b, m1, dtype=torch.bool)
    tid0 = torch.full((b, m0), -1, dtype=torch.long); tid1 = torch.full((b, m1), -1, dtype=torch.long)
    targets = torch.zeros(b, m0, m1); div_cols = torch.zeros(b, m1, dtype=torch.bool)
    for i, r in enumerate(rows):
        n0, n1 = len(r["tid0"]), len(r["tid1"])
        coords0[i, :n0] = r["coords0"]; coords1[i, :n1] = r["coords1"]
        mask0[i, :n0] = True; mask1[i, :n1] = True
        tid0[i, :n0] = r["tid0"]; tid1[i, :n1] = r["tid1"]
        targets[i, :n0, :n1] = r["target"]; div_cols[i, :n1] = r["division_cols"]
    return {"imgs": torch.stack([r["imgs"] for r in rows]), "coords0": coords0,
            "coords1": coords1, "mask0": mask0, "mask1": mask1, "tid0": tid0,
            "tid1": tid1, "targets": targets, "division_cols": div_cols,
            "crop": torch.tensor([r["crop"] for r in rows]),
            "frame": torch.tensor([r["frame"] for r in rows])}


class EdgeTrainingModel(nn.Module):
    """Public full model plus an optional auxiliary appearance projector."""

    def __init__(self, base: nn.Module, appearance_dim: int = 0):
        super().__init__(); self.base = base
        channels = int(base.unet_out_channels)
        self.projector = (nn.Linear(channels, appearance_dim) if appearance_dim > 0 else None)
        for p in self.base.detect_head.parameters():
            p.requires_grad_(False)


def load_public_full_model(T, weights: str | Path, device: torch.device,
                           *, appearance_dim: int = 0) -> EdgeTrainingModel:
    """Reconstruct and strictly load the exact full public checkpoint."""
    weights = Path(weights); cfg_path = weights.parent / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"public config missing next to weights: {cfg_path}")
    cfg = json.loads(cfg_path.read_text())
    out_c = int(cfg["unet_out_channels"]); layers = list(cfg["unet_layers"])
    unet = T.TemporalUNet3D(in_channels=1, out_channels=out_c, layers=layers)
    base = T.UNetNodeTransformer(unet=unet, unet_out_channels=out_c,
                                 pos_feat_dim=4 * T._POS_EMBED_DIM)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("initial weights must be a bare full-model state_dict")
    base.load_state_dict(state, strict=True)
    return EdgeTrainingModel(base, appearance_dim).to(device)


def continuation_loss(logits: torch.Tensor, target: torch.Tensor,
                      source_mask: torch.Tensor, target_mask: torch.Tensor,
                      division_cols: torch.Tensor) -> torch.Tensor:
    """Parental-softmax BCE; division-daughter columns contribute zero gradient."""
    losses = []
    for b in range(logits.shape[0]):
        ns = int(source_mask[b].sum()); nt = int(target_mask[b].sum())
        if not ns or not nt:
            continue
        z = logits[b, :ns, :nt]; y = target[b, :ns, :nt]
        keep = ~division_cols[b, :nt]
        if not bool(keep.any()):
            continue
        z, y = z[:, keep], y[:, keep]
        bg = torch.zeros(1, z.shape[1], device=z.device, dtype=z.dtype)
        prob = torch.softmax(torch.cat([z, bg], 0), 0)[:-1].clamp(1e-7, 1 - 1e-7)
        # Match the vendored sparse-lineage semantics: supervise cells touching an
        # annotated continuation row or column, not the entire N x M sea of unknowns.
        active_rows = y.sum(1) > 0
        active_cols = y.sum(0) > 0
        supervised = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)
        if not bool(supervised.any()):
            continue
        bce = F.binary_cross_entropy(prob, y, reduction="none")
        p_t = prob * y + (1.0 - prob) * (1.0 - y)
        losses.append((((1.0 - p_t) ** 2) * bce)[supervised].mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def appearance_triplet_loss(projector: nn.Module | None, feat0: torch.Tensor,
                            feat1: torch.Tensor, tid0: torch.Tensor, tid1: torch.Tensor,
                            mask0: torch.Tensor, mask1: torch.Tensor,
                            margin: float = 0.2) -> torch.Tensor:
    """Hard-negative temporal identity triplets; no identity match means zero loss."""
    if projector is None:
        return feat0.sum() * 0.0
    losses = []
    e0 = F.normalize(projector(feat0.float()), dim=-1)
    e1 = F.normalize(projector(feat1.float()), dim=-1)
    for b in range(feat0.shape[0]):
        n0, n1 = int(mask0[b].sum()), int(mask1[b].sum())
        ids0 = tid0[b, :n0]; ids1 = tid1[b, :n1]
        lookup = {int(v): i for i, v in enumerate(ids0.tolist())}
        for j, identity in enumerate(ids1.tolist()):
            i = lookup.get(int(identity))
            if i is None or n0 < 2:
                continue
            negative = torch.arange(n0, device=feat0.device) != i
            distances = 1.0 - e1[b, j].unsqueeze(0) @ e0[b, :n0].T
            k = torch.arange(n0, device=feat0.device)[negative][distances[0, negative].argmin()]
            losses.append(F.triplet_margin_loss(e1[b, j:j + 1], e0[b, i:i + 1],
                                                 e0[b, k:k + 1], margin=margin))
    return torch.stack(losses).mean() if losses else (e0.sum() + e1.sum()) * 0.0


def forward_batch(T, model: EdgeTrainingModel, batch: dict, device: torch.device,
                  *, triplet_weight: float = 0.0) -> tuple[torch.Tensor, dict]:
    imgs = batch["imgs"].to(device, dtype=torch.float32)
    c0 = batch["coords0"].to(device); c1 = batch["coords1"].to(device)
    m0 = batch["mask0"].to(device); m1 = batch["mask1"].to(device)
    target = batch["targets"].to(device); div = batch["division_cols"].to(device)
    tid0 = batch["tid0"].to(device); tid1 = batch["tid1"].to(device)
    # Bypass detect_head completely: this lane trains association representations only.
    maps = model.base.unet(imgs.unsqueeze(2))
    f0 = model.base._index_features(maps[:, 0], c0, m0)
    f1 = model.base._index_features(maps[:, 1], c1, m1)
    scale = torch.tensor(DEPLOY_DOWNSAMPLE, device=device)
    pc0, pc1 = c0 * scale, c1 * scale
    shape = (2, imgs.shape[2], imgs.shape[3] * 4, imgs.shape[4] * 4)
    t0 = torch.zeros((*pc0.shape[:-1], 1), device=device)
    t1 = torch.ones((*pc1.shape[:-1], 1), device=device)
    p0 = T._pos_embed_torch(torch.cat([t0, pc0], -1), shape)
    p1 = T._pos_embed_torch(torch.cat([t1, pc1], -1), shape)
    logits = model.base.predict_edges(f0, f1, pc0, pc1, p0, p1, m0, m1)
    assoc = continuation_loss(logits, target, m0, m1, div)
    triplet = (appearance_triplet_loss(model.projector, f0, f1, tid0, tid1, m0, m1)
               if triplet_weight > 0.0 else f0.sum() * 0.0)
    loss = assoc + float(triplet_weight) * triplet
    return loss, {"logits": logits, "association_loss": assoc, "triplet_loss": triplet}


@torch.no_grad()
def evaluate(T, model: EdgeTrainingModel, loader: DataLoader, device: torch.device,
             *, candidate_threshold: float = 0.48) -> dict:
    model.eval(); correct = total = reached = 0; losses = []
    for batch in loader:
        loss, out = forward_batch(T, model, batch, device)
        losses.append(float(loss)); logits = out["logits"]
        targets = batch["targets"].to(device); div = batch["division_cols"].to(device)
        for b in range(logits.shape[0]):
            ns, nt = int(batch["mask0"][b].sum()), int(batch["mask1"][b].sum())
            z = logits[b, :ns, :nt]; y = targets[b, :ns, :nt]
            bg = torch.zeros(1, nt, device=device, dtype=z.dtype)
            probs = torch.softmax(torch.cat([z, bg], 0), 0)[:-1]
            for j in range(nt):
                if bool(div[b, j]) or not bool(y[:, j].any()):
                    continue
                i = int(y[:, j].argmax()); total += 1
                correct += int(int(probs[:, j].argmax()) == i)
                reached += int(float(probs[i, j]) >= candidate_threshold)
    return {"loss": float(np.mean(losses)) if losses else float("nan"),
            "link_top1": correct / max(total, 1),
            "candidate_recall": reached / max(total, 1), "n_links": total}


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def save_resume(path: str | Path, model: EdgeTrainingModel, optimizer, scheduler, scaler,
                *, epoch: int, best: float, history: list[dict]) -> None:
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": int(epoch), "best": float(best), "history": history,
                "rng": _rng_state()}, tmp)
    tmp.replace(path)


def load_resume(path: str | Path, model: EdgeTrainingModel, optimizer, scheduler, scaler,
                device: torch.device) -> tuple[int, float, list[dict]]:
    ck = torch.load(path, map_location=device, weights_only=False)
    for key in ("model", "optimizer", "scheduler", "scaler", "epoch", "best", "history", "rng"):
        if key not in ck: raise ValueError(f"resume checkpoint missing {key}")
    model.load_state_dict(ck["model"], strict=True); optimizer.load_state_dict(ck["optimizer"])
    scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
    random.setstate(ck["rng"]["python"]); np.random.set_state(ck["rng"]["numpy"])
    torch.set_rng_state(ck["rng"]["torch"])
    if torch.cuda.is_available() and ck["rng"]["cuda"]:
        torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
    return int(ck["epoch"]) + 1, float(ck["best"]), list(ck["history"])


def run(args) -> dict:
    trainer_dir = Path(args.trainer_dir)
    # The vendored trainer imports biohub_tracking from its sibling src tree.
    # Notebook execution must recreate the PYTHONPATH used by the CLI scripts.
    sys.path.insert(0, str(trainer_dir.parent / "src"))
    sys.path.insert(0, str(trainer_dir))
    import train_unet_transformer as T
    assets = discover_edge_assets(args.root)
    data = Zh001rEdgeData(assets.nodes, assets.identity)
    iso = next((p for r in args.root for p in Path(r).rglob("zh001r_iso.npy")), None)
    if iso is None: raise FileNotFoundError("zh001r_iso.npy not found")
    val_ids = [c for c in data.crops if c % args.val_mod == 0]
    train_ids = [c for c in data.crops if c % args.val_mod != 0]
    train_ds = Zh001rEdgeWindows(data, iso, train_ids, node_cap=args.node_cap, seed=args.seed)
    val_ds = Zh001rEdgeWindows(data, iso, val_ids, node_cap=args.node_cap, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_edge_windows)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_edge_windows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    model = load_public_full_model(T, args.weights, device, appearance_dim=args.appearance_dim)
    detect_before = {k: v.detach().cpu().clone() for k, v in model.base.detect_head.state_dict().items()}
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start, best, history = 0, -1.0, []
    resume = Path(args.out) / "edge_resume.pth"
    Path(args.out).mkdir(parents=True, exist_ok=True)
    public_config = Path(args.weights).parent / "config.json"
    (Path(args.out) / "config.json").write_text(public_config.read_text())
    if args.resume and resume.exists(): start, best, history = load_resume(resume, model, opt, sched, scaler, device)
    if not history:
        val_ds.set_epoch(0)
        zero_val = evaluate(T, model, val_loader, device)
        history.append({"epoch": -1, "val": zero_val})
        best = zero_val["link_top1"] * zero_val["candidate_recall"]
        torch.save(model.base.state_dict(), Path(args.out) / "edge_predictor_best.pth")
        if model.projector is not None:
            torch.save(model.projector.state_dict(), Path(args.out) / "appearance_projector_best.pth")
        (Path(args.out) / "metrics.json").write_text(json.dumps(history, indent=1))
    for epoch in range(start, args.epochs):
        train_ds.set_epoch(epoch); model.train(); vals = []
        for step, batch in enumerate(train_loader):
            if args.max_steps and step >= args.max_steps: break
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16,
                                enabled=args.amp and device.type == "cuda"):
                loss, _ = forward_batch(T, model, batch, device, triplet_weight=args.triplet_weight)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update()
            vals.append(float(loss.detach()))
        # Validation keeps the epoch-0 cap sample fixed so checkpoint metrics are paired.
        sched.step(); metrics = evaluate(T, model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(vals)), "val": metrics}; history.append(row)
        score = metrics["link_top1"] * metrics["candidate_recall"]
        if score >= best:
            best = score; torch.save(model.base.state_dict(), Path(args.out) / "edge_predictor_best.pth")
            if model.projector is not None: torch.save(model.projector.state_dict(), Path(args.out) / "appearance_projector_best.pth")
        save_resume(resume, model, opt, sched, scaler, epoch=epoch, best=best, history=history)
        (Path(args.out) / "metrics.json").write_text(json.dumps(history, indent=1))
    for k, v in detect_before.items():
        if not torch.equal(v, model.base.detect_head.state_dict()[k].detach().cpu()):
            raise RuntimeError("frozen detection-head bytes changed")
    return {"best": best, "history": history}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--trainer-dir", required=True); p.add_argument("--root", action="append", required=True)
    p.add_argument("--weights", required=True); p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--node-cap", type=int, default=256); p.add_argument("--val-mod", type=int, default=6)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-steps", type=int, default=0); p.add_argument("--appearance-dim", type=int, default=64)
    p.add_argument("--triplet-weight", type=float, default=0.05); p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return 0 if run(p.parse_args(argv)) else 1


if __name__ == "__main__" and os.environ.get("H1R_KERNEL") != "1":
    raise SystemExit(main())
