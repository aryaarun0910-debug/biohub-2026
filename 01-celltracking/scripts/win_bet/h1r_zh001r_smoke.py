r"""H1-R pilot -- one CPU training step on the packaged Zebrahub crops (`kkunizaw/biohub-zh001r`).

WHY THIS EXISTS
---------------
`h1r_zh001r_audit.py` establishes that the packaged crops are geometry-compatible with our
deployed detector input (64^3 isotropic, ~1.68 um vs our 1.625 um) and that their labels sit on
real nuclei. This script closes the remaining plumbing question: does that data actually flow
through the *deployed* training surface -- `vendor/.../train_unet_transformer.py` -- and produce
a gradient that reaches the UNet?

Scope is deliberately the DETECTOR half only. The audit shows the node arrays are (N, 4) =
[t, z, y, x] with no track identity, so no GT transition matrix can be built and the edge head
cannot be supervised from this dataset alone. `h1r_train_smoke.py` already proves the edge path
on our own streamed level-1 data (which carries track ids); this proves the detection path on
the packaged data. Together they cover both halves of the H1 retrain.

A passing smoke is NOT scientific evidence (CLAUDE.md rule 5). It proves plumbing. The pilot that
follows must be LOEO-validated against a competition-only baseline before any promotion, and its
falsification is unchanged: no bilateral LOEO gain from the Zebrahub augmentation => the external
imaging does not transfer, kill the bet.

Usage:
  .venv\Scripts\python.exe scripts\win_bet\h1r_zh001r_audit.py --root <dir>   # run this first
  .venv\Scripts\python.exe scripts\win_bet\h1r_zh001r_smoke.py  --root <dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "vendor" / "kaggle-cell-tracking" / "scripts"))
import train_unet_transformer as T  # noqa: E402

THEIR_VOX_UM = 1.677     # measured by h1r_zh001r_audit.py (ratio 1.032x our 1.625 um grid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory holding zh001r_iso.npy / _nodes.npz")
    ap.add_argument("--crop", type=int, default=0, help="crop index into the 72 packaged crops")
    ap.add_argument("--t0", type=int, default=0, help="first frame of the pair")
    ap.add_argument("--max-nodes", type=int, default=512, help="cap nodes for CPU memory")
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(0)
    root = Path(args.root)
    iso = np.load(root / "zh001r_iso.npy", mmap_mode="r")
    nodes = np.load(root / "zh001r_nodes.npz", allow_pickle=True)
    n_crop, n_t = iso.shape[0], iso.shape[1]
    if not (0 <= args.crop < n_crop and 0 <= args.t0 < n_t - 1):
        raise SystemExit(f"crop must be in [0,{n_crop}) and t0 in [0,{n_t - 1})")
    spatial = tuple(iso.shape[2:])
    print(f"packaged crops: {iso.shape}  using crop={args.crop} frames t={args.t0},{args.t0 + 1}")

    def frame(t):
        return np.asarray(iso[args.crop, t], dtype=np.float32)

    def frame_nodes(t):
        a = nodes[f"f{args.crop * n_t + t}"]
        c = a[:, 1:].astype(np.float32)                      # z, y, x in the 64^3 grid
        inside = np.all((c >= 0) & (c < np.array(spatial)), axis=1)
        c = c[inside]
        if len(c) > args.max_nodes:
            sel = np.random.default_rng(1).choice(len(c), args.max_nodes, replace=False)
            c = c[sel]
        return c

    c0, c1 = frame_nodes(args.t0), frame_nodes(args.t0 + 1)
    print(f"nodes: t0={len(c0)}  t1={len(c1)}  (capped at {args.max_nodes})")
    if len(c0) < 8:
        raise SystemExit("too few nodes in this crop/frame")

    # Their intensity is uint8; the competition volumes are uint16. Standardise per-window, which
    # is what the deployed loader does downstream -- this is the renormalisation the pilot needs.
    imgs = np.stack([frame(args.t0), frame(args.t0 + 1)])          # (W=2, z, y, x)
    imgs = (imgs - imgs.mean()) / (imgs.std() + 1e-6)
    imgs_t = torch.from_numpy(imgs).unsqueeze(0)                   # (B=1, W=2, z, y, x)

    def pack(c):
        M = len(c)
        coords = torch.from_numpy(c).unsqueeze(0)                  # (1, M, 3)
        mask = torch.ones(1, M, dtype=torch.bool)
        return coords, mask

    cs, ms = pack(c0)
    ct, mt = pack(c1)

    unet = T.TemporalUNet3D(in_channels=1, out_channels=32, layers=[32, 64, 128])
    model = T.UNetNodeTransformer(unet=unet, unet_out_channels=32,
                                  pos_feat_dim=4 * T._POS_EMBED_DIM)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("encode (UNet forward on packaged Zebrahub crop)...", flush=True)
    losses = []
    for step in range(args.steps):
        unet_out, det_logits = model.encode(imgs_t)                # det_logits: list of W (B,1,z,y,x)
        loss = (T.compute_detection_loss(det_logits[0], cs, ms)
                + T.compute_detection_loss(det_logits[1], ct, mt)) / 2.0
        opt.zero_grad()
        loss.backward()
        gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()
                               if p.grad is not None))
        unet_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in unet.parameters())
        head_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.detect_head.parameters())
        opt.step()
        losses.append(float(loss.detach()))
        print(f"  step {step}: det_loss={loss.item():.4f}  grad-norm={gnorm.item():.4f}  "
              f"UNet grad={unet_grad}  detect_head grad={head_grad}")
        if not (unet_grad and head_grad):
            raise SystemExit("GATE FAIL: gradient did not reach the UNet and detection head")

    print(f"  det_logits shape: {tuple(det_logits[0].shape)}  (spatial {spatial})")
    if losses[-1] >= losses[0]:
        print(f"  NOTE: loss did not decrease over {args.steps} steps "
              f"({losses[0]:.4f} -> {losses[-1]:.4f}); at lr=1e-4 on 2 steps this is not "
              f"informative, it is a plumbing check only.")
    print("\nSMOKE OK: packaged Zebrahub imaging -> deployed UNet -> detection loss -> "
          "backward -> step")
    print("This is plumbing only. Detection supervision ONLY -- the packaged nodes carry no "
          "track identity, so the edge head cannot be trained from this dataset.")


if __name__ == "__main__":
    sys.exit(main())
