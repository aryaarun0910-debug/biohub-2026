"""Run the borrowed detector on one film, validate recall against GT, time it.

Correctness gate: if this reimplementation is faithful, node recall against the
sparse annotation should land near what the corpus reports (~0.99 at the
metric's own 7 um radius).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np, torch
from scipy.optimize import linear_sum_assignment
from biohub import io, model as M

DEV = "mps"
film = sys.argv[1] if len(sys.argv) > 1 else "6bba_05db0fb1"
zarr_p = io.dataset_root() / "train" / f"{film}.zarr"
geff_p = io.dataset_root() / "train" / f"{film}.geff"

m, cfg = M.load("primary", DEV)
q = io.quantiles(zarr_p)
q_low, q_high = float(q["0.001"]), float(q["0.999"])
info = io.image_meta(zarr_p)
T = info["shape"][0]
print(f"{film}: T={T} shape={info['shape']} q=({q_low:.1f},{q_high:.1f})")

gt = io.read_geff(geff_p)
gt_um = np.stack([gt["z"] * 1.625, gt["y"] * 0.40625, gt["x"] * 0.40625], 1)

# forward pass, sliding window of 2 with stride 1, dedup by frame
det_by_t, feat_by_t = {}, {}
t_io = t_gpu = 0.0
t0 = time.perf_counter()
for ws in range(0, T - M.WINDOW + 1):
    ts = list(range(ws, ws + M.WINDOW))
    a = time.perf_counter()
    imgs = torch.stack([M.normalise(io.read_frame(zarr_p, t), q_low, q_high) for t in ts])
    t_io += time.perf_counter() - a
    a = time.perf_counter()
    with torch.no_grad():
        unet_out, det = m.encode(imgs.unsqueeze(0).to(DEV))
    torch.mps.synchronize()
    t_gpu += time.perf_counter() - a
    for i, t in enumerate(ts):
        if t in det_by_t:
            continue
        det_by_t[t] = M.peaks(det[i][0]).cpu().numpy()
        feat_by_t[t] = unet_out[0, i]          # kept on device for the sample below
wall = time.perf_counter() - t0

n_pred = sum(len(v) for v in det_by_t.values())
print(f"\nwall {wall:.1f}s  ({wall/T*1000:.0f} ms/frame) | io {t_io:.1f}s  gpu {t_gpu:.1f}s")
print(f"predicted nodes {n_pred:,}  | est_true {gt['estimated_number_of_nodes']:,.0f}"
      f"  | ratio {n_pred/gt['estimated_number_of_nodes']:.3f}")

# recall of the sparse annotation, Hungarian at the metric's own 7 um
hit = tot = 0; dists = []
for t in np.unique(gt["t"]):
    gi = np.where(gt["t"] == t)[0]
    p = det_by_t.get(int(t), np.empty((0, 3)))
    tot += len(gi)
    if len(p) == 0:
        continue
    p_um = p.astype(np.float64) * 1.625
    d = np.linalg.norm(gt_um[gi][:, None] - p_um[None], axis=-1)
    r, c = linear_sum_assignment(d)
    for i, j in zip(r, c):
        if d[i, j] <= 7.0:
            hit += 1; dists.append(d[i, j])
print(f"node recall @7um: {hit}/{tot} = {hit/tot:.4f} | median match dist {np.median(dists):.2f} um")
print(f"\nEXTRAPOLATION -> 199-film hidden set: {wall*199/3600:.2f} h (no TTA, 1 GPU)")
