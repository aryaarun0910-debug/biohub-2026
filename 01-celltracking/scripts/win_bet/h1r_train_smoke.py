r"""H1-R smoke -- prove level-1 Zebrahub imaging + tracks flow through the deployed model.

Minimal end-to-end plumbing check on a small CPU-sized crop of one frame pair:
  local level-1 zarr crop  ->  TemporalUNet3D.encode  ->  index features at Zebrahub GT nodes
  ->  SimpleNodeTransformer.predict_edges  ->  compute_loss(GT transition)  ->  backward.

It does NOT reproduce the full detect-and-match training step (deployed competition code, not the
new risk). It confirms the UNet runs on level-1 data, Zebrahub nodes index into its features, the
edge head produces logits, and gradients flow. Requires a local crop fetched by h1r_fetch_imaging.

Usage:
  .venv\Scripts\python.exe scripts\win_bet\h1r_fetch_imaging.py --embryo ZSNS003 --t-start 0 --t-end 2
  .venv\Scripts\python.exe scripts\win_bet\h1r_train_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import zarr

ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "vendor" / "kaggle-cell-tracking" / "scripts"))
import train_unet_transformer as T  # noqa: E402

ZARR = ROOT / "data" / "external" / "zebrahub" / "imaging" / "ZSNS003_L1.zarr"
TRACKS = ROOT / "_evidence" / "agent_runs" / "agent4" / "zebrahub_prep" / "ZSNS003.parquet"
CROP = (64, 160, 160)          # z, y, x subvolume (CPU-sized)
LEVEL1_DS = 2                  # level-1 is 2x downsampled vs level-0 voxel track coords


def main() -> None:
    torch.manual_seed(0)
    g = zarr.open_group(str(ZARR), mode="r")
    arr = g["0"]                                   # (n, 1, Z, Y, X)
    scale = list(g.attrs["zebrahub"]["scale_zyx"])
    n = arr.shape[0]
    assert n >= 2, "need >=2 frames; fetch t-start 0 t-end 2"

    tracks = pl.read_parquet(TRACKS)
    Z, Y, X = arr.shape[2:]
    c0_all = tracks.filter(pl.col("t") == 0).select(["z", "y", "x"]).to_numpy() / LEVEL1_DS
    hi = np.array([Z, Y, X]) - np.array(CROP)
    # nuclei sit on a shell; the centroid is in the hollow interior. Scan node-centred
    # origins and keep the densest crop so the smoke has real nodes.
    rng = np.random.default_rng(0)
    best = (0, np.zeros(3, int))
    for _ in range(3000):
        p = np.clip(c0_all[rng.integers(len(c0_all))] - np.array(CROP) / 2, 0, hi).astype(int)
        k = int(np.all((c0_all - p >= 0) & (c0_all - p < np.array(CROP)), axis=1).sum())
        if k > best[0]:
            best = (k, p)
    origin = tuple(int(v) for v in best[1])
    print(f"crop {CROP} at densest level-1 origin {origin} (~{best[0]} t0 nodes; frame ZYX={(Z,Y,X)}) scale_zyx={scale}")

    def crop_frame(i):
        z0, y0, x0 = origin
        sub = arr[i, 0, z0:z0 + CROP[0], y0:y0 + CROP[1], x0:x0 + CROP[2]]
        return np.asarray(sub, dtype=np.float32)

    def nodes(t, cap=256):
        d = tracks.filter(pl.col("t") == t)
        c = d.select(["z", "y", "x"]).to_numpy() / LEVEL1_DS - np.array(origin)  # crop-local level-1 grid
        keep = np.all((c >= 0) & (c < np.array(CROP)), axis=1)
        d, c = d.filter(pl.Series(keep)), c[keep]
        if len(c) > cap:  # cap all-pairs attention / CPU memory for the smoke
            m = np.zeros(len(c), bool)
            m[np.random.default_rng(1).choice(len(c), cap, replace=False)] = True
            d, c = d.filter(pl.Series(m)), c[m]
        return d, c

    d0, c0 = nodes(0)
    d1, c1 = nodes(1)
    print(f"nodes in crop: t0={len(c0)}  t1={len(c1)}")
    if len(c0) < 2 or len(c1) < 2:
        raise SystemExit("too few nodes in crop; widen CROP or pick another region")

    # normalise intensity like the deployed loader does downstream (simple per-frame standardise)
    imgs = np.stack([crop_frame(0), crop_frame(1)])                     # (2, z, y, x)
    imgs = (imgs - imgs.mean()) / (imgs.std() + 1e-6)
    imgs_t = torch.from_numpy(imgs).unsqueeze(0)                        # (B=1, W=2, z, y, x)

    # GT transition (N_t0 x N_t1): same track_id (continuation) or parent link (division)
    tid0 = d0["track_id"].to_numpy(); tid1 = d1["track_id"].to_numpy(); par1 = d1["parent_track_id"].to_numpy()
    gt = torch.zeros(len(c0), len(c1), dtype=torch.float32)
    idx0 = {int(t): i for i, t in enumerate(tid0)}
    for j in range(len(c1)):
        for src in (int(tid1[j]), int(par1[j])):
            if src in idx0:
                gt[idx0[src], j] = 1.0
    print(f"GT edges in crop: {int(gt.sum())} (density {gt.mean():.4f})")

    def coords_mask_pos(c, t):
        M = len(c)
        coords3 = torch.from_numpy(c.astype(np.float32)).unsqueeze(0)     # (1,M,3) z,y,x for indexing
        mask = torch.ones(1, M, dtype=torch.bool)
        c4 = np.concatenate([np.full((M, 1), t, np.float32), c.astype(np.float32)], axis=1)  # t,z,y,x
        pos = torch.from_numpy(T.extract_pos_features(c4, (2,) + CROP)).unsqueeze(0)          # (1,M,D)
        return coords3, mask, pos

    cs, ms, ps = coords_mask_pos(c0, 0)
    ct, mt, pt = coords_mask_pos(c1, 1)

    unet = T.TemporalUNet3D(in_channels=1, out_channels=32, layers=[32, 64, 128])
    model = T.UNetNodeTransformer(unet=unet, unet_out_channels=32, pos_feat_dim=4 * T._POS_EMBED_DIM)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    print("encode (UNet forward on level-1 crop)...", flush=True)
    unet_out, det_logits = model.encode(imgs_t)                        # (B,W,C,z,y,x)
    fs = model._index_features(unet_out[:, 0], cs, ms)                 # (B,N0,C)
    ft = model._index_features(unet_out[:, 1], ct, mt)                 # (B,N1,C)
    vox = torch.tensor(scale, dtype=torch.float32)
    edge_logits = model.predict_edges(fs, ft, cs * vox, ct * vox, ps, pt, ms, mt)  # (B,N0,N1)
    print("edge_logits:", tuple(edge_logits.shape), " det_logits[0]:", tuple(det_logits[0].shape))

    loss = T.compute_loss(edge_logits[0], gt)
    opt.zero_grad(); loss.backward()
    gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None))
    unet_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in unet.parameters())
    print(f"loss={loss.item():.4f}  total grad-norm={gnorm.item():.4f}  UNet received grad={unet_grad}")
    opt.step()
    print("SMOKE OK: level-1 Zebrahub imaging + tracks -> model -> loss -> backward -> step")


if __name__ == "__main__":
    main()
