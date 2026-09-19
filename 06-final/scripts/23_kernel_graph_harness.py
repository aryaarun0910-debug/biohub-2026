"""Score the kernel's own prediction graphs locally, against real ground truth.

The kernel keeps 12 .geff predictions; 8 are TRAIN films, so we have GT for
deployed-quality graphs (the thing the local reimplementation never reached).
Coordinates here are ORIGINAL voxel space (z 0-63, y/x 0-252), so distances use
the anisotropic scale, not the isotropic downsampled grid.

Calibration target -- the kernel's own validator on these 8 films:
    adjusted_edge_jaccard 0.92583    division_jaccard 0.1111 (3 TP / 15 FP / 9 FN)
(and the unmodified pipeline: adj 0.9260, divJ 0.2308 = 3 TP / 1 FP / 9 FN)
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from biohub import io

PRED = Path("artifacts/s01_output/tracking_repo/predictions/unknown/unet_transformer_val/split_0")
SCALE = np.array([1.625, 0.40625, 0.40625])
RADIUS = 7.0


def load_pred(p):
    """Prediction graph: ids, t, zyx (voxels), edges, edge_prob, edge_dist."""
    d = io.read_geff(p)
    ids = d["ids"]
    idx = {int(i): k for k, i in enumerate(ids)}
    e = np.array([[idx[int(a)], idx[int(b)]] for a, b in d["edges"]], np.int64).reshape(-1, 2)
    prob = io.read_zstd_array(p / "edges/props/edge_prob/values")
    dist = io.read_zstd_array(p / "edges/props/edge_dist/values")
    zyx = np.stack([d["z"], d["y"], d["x"]], 1).astype(np.float64)
    return dict(t=d["t"].astype(np.int64), zyx=zyx, edges=e,
                prob=np.asarray(prob, np.float64), dist=np.asarray(dist, np.float64))


def load_gt(stem):
    g = io.read_geff(io.dataset_root() / "train" / f"{stem}.geff")
    idx = {int(i): k for k, i in enumerate(g["ids"])}
    e = np.array([[idx[int(a)], idx[int(b)]] for a, b in g["edges"]], np.int64).reshape(-1, 2)
    zyx = np.stack([g["z"], g["y"], g["x"]], 1).astype(np.float64)
    return dict(t=g["t"].astype(np.int64), zyx=zyx, edges=e,
                n_est=float(g["estimated_number_of_nodes"]))


stems = sorted(p.stem for p in PRED.glob("*.geff"))
print(f"{len(stems)} validator films with GT\n")
print(f"{'film':<18}{'pred nodes':>11}{'pred edges':>11}{'forks':>7}{'GT nodes':>10}"
      f"{'GT div':>8}{'n_est':>10}{'ratio':>7}")
tot = {}
for s in stems:
    P, G = load_pred(PRED / f"{s}.geff"), load_gt(s)
    src, cnt = np.unique(P["edges"][:, 0], return_counts=True)
    gsrc, gcnt = np.unique(G["edges"][:, 0], return_counts=True)
    print(f"{s:<18}{len(P['t']):>11,}{len(P['edges']):>11,}{int((cnt>=2).sum()):>7}"
          f"{len(G['t']):>10,}{int((gcnt>=2).sum()):>8}{G['n_est']:>10,.0f}"
          f"{len(P['t'])/G['n_est']:>7.3f}")
    tot[s] = (P, G)
np.save("artifacts/kernel_graphs.npy", np.array([0]))  # marker
print(f"\ntotal GT divisions across the 8 validator films: "
      f"{sum(int((np.unique(G['edges'][:,0], return_counts=True)[1] >= 2).sum()) for _, G in tot.values())}")
print(f"total predicted nodes: {sum(len(P['t']) for P, _ in tot.values()):,}")
print(f"mean n_pred/n_est: {np.mean([len(P['t'])/G['n_est'] for P, G in tot.values()]):.3f}")
