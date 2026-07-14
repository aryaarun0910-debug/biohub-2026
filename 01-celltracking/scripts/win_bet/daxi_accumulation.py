"""Moonshot falsification — does TEMPORAL ACCUMULATION of raw DAXI response reveal isolated
misses that single-frame thresholding misses?

For each isolated missed GT node, accumulate the raw DAXI foreground response in a small
window over a short temporal tube [t-K..t+K], and compare against deployment-matched CONTROL
locations (same timepoints, >7um from any GT node, i.e. plausible non-cells). If isolated
misses separate from controls on ACCUMULATED response far better than on SINGLE-FRAME
response, then weak sub-threshold evidence exists over time -> track-before-detect has a
mechanism. If not, the detection ceiling is real and ~0.90-0.91 is the honest end.

Cheap: reuses the DAXI detector, no new model. Bounded sample per crop.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402

CACHE = ROOT / "artifacts/kaggle/e0c_cache"
SCALE = np.array([1.625, 0.40625, 0.40625])
K = 3          # tube half-length (frames)
WZ, WYX = 1, 3  # accumulation window (voxels)
N_SAMPLE = 60


def isolated_and_gt(split, crop):
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, load_graph
    from biotrack.submission import submission_to_graphs
    import polars as pl
    gdf = (pl.read_parquet(CACHE / "graphs" / str(split) / f"{crop}.parquet")
           .with_columns(pl.lit(crop).alias("dataset")).with_row_index("id"))
    pred = submission_to_graphs(gdf)[crop]
    gt = load_graph(str(ROOT / "data" / "train" / f"{crop}.geff"))
    pred.match(gt, matching=DistanceMatching(max_distance=MAX_DISTANCE, scale=DEFAULT_SCALE))
    na = pred.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    matched = {int(m) for m in na[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list() if m not in (None, -1)}
    gna = gt.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t", "z", "y", "x"])
    pos = {int(r[td.DEFAULT_ATTR_KEYS.NODE_ID]): (int(r["t"]), r["z"], r["y"], r["x"]) for r in gna.iter_rows(named=True)}
    iso = []
    gt_by_t = {}
    for g in map(int, gt.node_ids()):
        t, z, y, x = pos[g]; gt_by_t.setdefault(t, []).append((z, y, x))
        if g in matched:
            continue
        pr = [int(p) for p in gt.predecessors(g)]; su = [int(s) for s in gt.successors(g)]
        if not (any(p in matched for p in pr) or any(s in matched for s in su)):
            iso.append((t, z, y, x))
    return iso, gt_by_t


def acc_response(fgvols, t, z, y, x, shape):
    single = 0.0; tube = []
    for dt in range(-K, K + 1):
        tt = t + dt
        if tt not in fgvols:
            continue
        z0, z1 = max(0, int(z) - WZ), min(shape[0], int(z) + WZ + 1)
        y0, y1 = max(0, int(y) - WYX), min(shape[1], int(y) + WYX + 1)
        x0, x1 = max(0, int(x) - WYX), min(shape[2], int(x) + WYX + 1)
        v = float(fgvols[tt][z0:z1, y0:y1, x0:x1].max())
        tube.append(v)
        if dt == 0:
            single = v
    return single, float(np.mean(tube)) if tube else 0.0


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s); order = np.argsort(s)
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg) if npos and nneg else float("nan")


def run(split, crop, net, rng):
    iso, gt_by_t = isolated_and_gt(split, crop)
    if not iso:
        return None
    iso = [iso[i] for i in rng.choice(len(iso), min(N_SAMPLE, len(iso)), replace=False)]
    zp = ROOT / "data" / "train" / f"{crop}.zarr"
    arr = zarr.open(zp / "0", mode="r"); shape = arr.shape[1:]
    # controls: same timepoints, >7um from any GT node
    ctrl = []
    for (t, _, _, _) in iso:
        for _ in range(20):
            z, y, x = rng.integers(0, shape[0]), rng.integers(0, shape[1]), rng.integers(0, shape[2])
            gm = np.array([z, y, x]) * SCALE
            near = gt_by_t.get(t, [])
            if not near or min(np.linalg.norm(np.array(near) * SCALE - gm, axis=1)) > 7.0:
                ctrl.append((t, z, y, x)); break
    frames = sorted({tt for (t, *_ ) in iso + ctrl for tt in range(t - K, t + K + 1) if 0 <= tt < arr.shape[0]})
    fgvols = {}
    for t in frames:
        frame = np.asarray(arr[t]).astype(np.float32)
        lo, hi = np.percentile(frame, 1), np.percentile(frame, 99.9)
        fn = np.clip((frame - lo) / (hi - lo + 1e-6), 0, 1)
        with torch.no_grad():
            fgvols[t] = net(torch.tensor(fn[None, None]))[0, 0].numpy()
    ys, single, accum = [], [], []
    for (t, z, y, x) in iso:
        s, a = acc_response(fgvols, t, z, y, x, shape); ys.append(1); single.append(s); accum.append(a)
    for (t, z, y, x) in ctrl:
        s, a = acc_response(fgvols, t, z, y, x, shape); ys.append(0); single.append(s); accum.append(a)
    return ys, single, accum


def main():
    net = torch.jit.load(str(ROOT / "weights" / "unet-daxi.pt"), map_location="cpu"); net.eval()
    rng = np.random.default_rng(0)
    for spec in ["1:6bba_05db0fb1", "0:44b6_0b24845f"]:
        split, crop = spec.split(":"); split = int(split)
        r = run(split, crop, net, rng)
        if r is None:
            print(f"{crop}: no isolated misses"); continue
        ys, single, accum = r
        print(f"{crop}: {int(sum(ys))} isolated vs {len(ys)-int(sum(ys))} controls | "
              f"single-frame sep AUC={auc(ys, single):.3f} | ACCUMULATED sep AUC={auc(ys, accum):.3f} | "
              f"iso mean single={np.mean([s for s,y in zip(single,ys) if y]):.3f} "
              f"accum={np.mean([a for a,y in zip(accum,ys) if y]):.3f} | "
              f"ctrl accum={np.mean([a for a,y in zip(accum,ys) if not y]):.3f}", flush=True)
    print("\nInterpretation: if ACCUMULATED sep AUC >> single-frame AUC and iso-accum >> ctrl-accum, "
          "temporal integration reveals weak isolated misses -> track-before-detect has a mechanism. "
          "If both AUCs are near 0.5, the isolated misses carry no sub-threshold signal -> detection "
          "ceiling is real and ~0.90-0.91 is the honest end.")


if __name__ == "__main__":
    main()
