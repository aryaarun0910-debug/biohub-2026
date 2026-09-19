"""G2: cache detections + node-sampled features for all 199 training films.

One forward pass buys a corpus where the probe, the oracle, the miner and every
threshold sweep run in seconds instead of hours. Feature VOLUMES are impossible
to cache (~5 TB); features sampled AT NODE POSITIONS are ~9 MB for the corpus.
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np, torch
from biohub import io, model as M

DEV = "mps"
OUT = Path("artifacts/cache"); OUT.mkdir(parents=True, exist_ok=True)
m, cfg = M.load("primary", DEV)
films = io.films("train")
t_start = time.perf_counter()

for n, film in enumerate(films):
    dst = OUT / f"{film}.npz"
    if dst.exists():
        continue
    zp = io.dataset_root() / "train" / f"{film}.zarr"
    gp = io.dataset_root() / "train" / f"{film}.geff"
    q = io.quantiles(zp); q_low, q_high = float(q["0.001"]), float(q["0.999"])
    T = io.image_meta(zp)["shape"][0]
    gt = io.read_geff(gp)

    # GT coords in the downsampled (1,4,4) grid, kept float for sub-voxel sampling
    gt_grid = np.stack([gt["z"], gt["y"] / 4.0, gt["x"] / 4.0], 1).astype(np.float32)
    gt_t = gt["t"].astype(np.int32)

    det_t, det_zyx, det_p, det_feat = [], [], [], []
    gt_feat = np.zeros((len(gt["ids"]), cfg["unet_out_channels"]), np.float16)

    for ws in range(0, T - M.WINDOW + 1):
        ts = list(range(ws, ws + M.WINDOW))
        imgs = torch.stack([M.normalise(io.read_frame(zp, t), q_low, q_high) for t in ts])
        with torch.no_grad():
            unet_out, det = m.encode(imgs.unsqueeze(0).to(DEV))
        for i, t in enumerate(ts):
            if ws > 0 and i == 0:          # dedup: frame already seen
                continue
            fmap = unet_out[0, i]
            pk = M.peaks(det[i][0])
            if len(pk):
                prob = torch.sigmoid(det[i][0, 0][pk[:, 0], pk[:, 1], pk[:, 2]])
                det_t.append(np.full(len(pk), t, np.int16))
                det_zyx.append(pk.to(torch.int16).cpu().numpy())
                det_p.append(prob.to(torch.float16).cpu().numpy())
                det_feat.append(M.sample_features(fmap, pk.float()).to(torch.float16).cpu().numpy())
            sel = np.where(gt_t == t)[0]
            if len(sel):
                c = torch.from_numpy(gt_grid[sel])
                gt_feat[sel] = M.sample_features(fmap, c).to(torch.float16).cpu().numpy()

    np.savez_compressed(
        dst,
        det_t=np.concatenate(det_t) if det_t else np.zeros(0, np.int16),
        det_zyx=np.concatenate(det_zyx) if det_zyx else np.zeros((0, 3), np.int16),
        det_p=np.concatenate(det_p) if det_p else np.zeros(0, np.float16),
        det_feat=np.concatenate(det_feat) if det_feat else np.zeros((0, 32), np.float16),
        gt_ids=gt["ids"], gt_t=gt_t, gt_grid=gt_grid, gt_feat=gt_feat,
        gt_edges=gt["edges"], est_nodes=gt["estimated_number_of_nodes"],
    )
    el = time.perf_counter() - t_start
    print(f"[{n+1:>3}/{len(films)}] {film}  {el/ (n+1):.1f}s/film  eta {el/(n+1)*(len(films)-n-1)/60:.0f}min",
          flush=True)

print(f"DONE in {(time.perf_counter()-t_start)/60:.1f} min")
