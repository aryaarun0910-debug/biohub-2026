"""Motion-compensated DAXI accumulation pilot for isolated missed GT nodes.

This is an oracle-motion *signal existence* test, not a deployable detector:
positive tubes follow the GT lineage; paired hard controls start at cached low-
threshold DAXI peaks and receive the same displacement tube. It compares center-
frame response, static accumulation, and motion-compensated accumulation.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

CACHE = ROOT / "artifacts/kaggle/e0c_cache"
DAXI = ROOT / "artifacts/kaggle/daxi_cand"
SCALE = np.array([1.625, 0.40625, 0.40625])
K = 3
WZ, WYX = 1, 3
N_SAMPLE = 60


def isolated_records(split: int, crop: str):
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching
    from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, load_graph
    from biotrack.submission import submission_to_graphs

    gdf = (pl.read_parquet(CACHE / "graphs" / str(split) / f"{crop}.parquet")
           .with_columns(pl.lit(crop).alias("dataset")).with_row_index("id"))
    pred = submission_to_graphs(gdf)[crop]
    gt = load_graph(ROOT / "data" / "train" / f"{crop}.geff")
    pred.match(gt, matching=DistanceMatching(max_distance=MAX_DISTANCE, scale=DEFAULT_SCALE))
    matched_attr = pred.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID])
    matched = {int(v) for v in matched_attr[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list()
               if v not in (None, -1)}
    attrs = gt.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t", "z", "y", "x"])
    pos = {int(r[td.DEFAULT_ATTR_KEYS.NODE_ID]):
           (int(r["t"]), np.array([r["z"], r["y"], r["x"]], float))
           for r in attrs.iter_rows(named=True)}
    gt_by_t: dict[int, list[np.ndarray]] = {}
    for t, xyz in pos.values():
        gt_by_t.setdefault(t, []).append(xyz)
    isolated = []
    for gid in map(int, gt.node_ids()):
        if gid in matched:
            continue
        pred_ids = [int(v) for v in gt.predecessors(gid)]
        succ_ids = [int(v) for v in gt.successors(gid)]
        if any(v in matched for v in pred_ids) or any(v in matched for v in succ_ids):
            continue
        isolated.append(gid)
    return gt, pos, gt_by_t, isolated


def lineage_tube(gt, pos, center: int) -> dict[int, np.ndarray] | None:
    """Return an unambiguous GT path around center; skip division ambiguity."""
    t0, _ = pos[center]
    tube = {t0: pos[center][1]}
    cur = center
    for _ in range(K):
        parents = [int(v) for v in gt.predecessors(cur)]
        if len(parents) != 1:
            break
        cur = parents[0]
        tube[pos[cur][0]] = pos[cur][1]
    cur = center
    for _ in range(K):
        children = [int(v) for v in gt.successors(cur)]
        if len(children) != 1:
            break
        cur = children[0]
        tube[pos[cur][0]] = pos[cur][1]
    return tube if len(tube) >= 3 else None


def hard_control_pool(crop: str, t: int, gt_at_t: list[np.ndarray], rng):
    p = DAXI / f"{crop}.npz"
    if not p.exists():
        return []
    d = np.load(p)
    idx = np.flatnonzero(d["t"] == t)
    rng.shuffle(idx)
    gt_um = np.asarray(gt_at_t) * SCALE if gt_at_t else np.zeros((0, 3))
    # Cached entries are actual low-threshold local maxima: deployment-matched controls.
    controls = []
    for j in idx[:1000]:
        xyz = np.array([d["z"][j], d["y"][j], d["x"][j]], float)
        if len(gt_um) and np.linalg.norm(gt_um - xyz * SCALE, axis=1).min() <= 7.0:
            continue
        controls.append((xyz, float(d["resp"][j])))
        if len(controls) >= 200:
            break
    return controls


def patch_max(volume: np.ndarray, xyz: np.ndarray) -> float:
    z, y, x = np.rint(xyz).astype(int)
    z0, z1 = max(0, z - WZ), min(volume.shape[0], z + WZ + 1)
    y0, y1 = max(0, y - WYX), min(volume.shape[1], y + WYX + 1)
    x0, x1 = max(0, x - WYX), min(volume.shape[2], x + WYX + 1)
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        return 0.0
    return float(volume[z0:z1, y0:y1, x0:x1].max())


def infer_queries(net, arr, queries):
    """Infer only 16x64x64 patches around requested (key,t,xyz) points."""
    from collections import defaultdict
    by_t = defaultdict(list)
    for key, t, xyz in queries:
        if 0 <= t < arr.shape[0]:
            by_t[int(t)].append((key, np.asarray(xyz, float)))
    result = {}
    pz, pyx = 8, 32
    for t, items in by_t.items():
        frame = np.asarray(arr[t]).astype(np.float32)
        lo, hi = np.percentile(frame, 1), np.percentile(frame, 99.9)
        frame = np.clip((frame - lo) / (hi - lo + 1e-6), 0, 1)
        padded = np.pad(frame, ((pz, pz), (pyx, pyx), (pyx, pyx)), mode="reflect")
        patches = []
        for _, xyz in items:
            z, y, x = np.rint(xyz).astype(int) + np.array([pz, pyx, pyx])
            patches.append(padded[z-pz:z+pz, y-pyx:y+pyx, x-pyx:x+pyx])
        for start in range(0, len(patches), 8):
            batch = torch.from_numpy(np.stack(patches[start:start + 8]))[:, None]
            with torch.no_grad():
                out = net(batch)[:, 0].numpy()
            for j, vol in enumerate(out):
                key = items[start + j][0]
                result[key] = float(vol[pz-WZ:pz+WZ+1,
                                        pyx-WYX:pyx+WYX+1,
                                        pyx-WYX:pyx+WYX+1].max())
    return result


def auc(y, score) -> float:
    y = np.asarray(y, int); score = np.asarray(score, float)
    ranks = rankdata(score, method="average")
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if not npos or not nneg:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def run(split: int, crop: str, net, rng):
    gt, pos, gt_by_t, isolated = isolated_records(split, crop)
    rng.shuffle(isolated)
    events = []
    for gid in isolated:
        if len(events) >= N_SAMPLE:
            break
        tube = lineage_tube(gt, pos, gid)
        if tube is None:
            continue
        t0, center = pos[gid]
        controls = hard_control_pool(crop, t0, gt_by_t.get(t0, []), rng)
        if not controls:
            continue
        events.append((t0, tube, controls))
    if not events:
        return None

    arr = zarr.open(ROOT / "data" / "train" / f"{crop}.zarr" / "0", mode="r")
    center_queries = [(i, t0, tube[t0]) for i, (t0, tube, _) in enumerate(events)]
    positive_center = infer_queries(net, arr, center_queries)

    paired = []
    for i, (t0, positive, controls) in enumerate(events):
        ps = positive_center[i]
        ctrl_center, _ = min(controls, key=lambda item: abs(item[1] - ps))
        center = positive[t0]
        control = {t: ctrl_center + (xyz - center) for t, xyz in positive.items()}
        paired.append((t0, positive, control))

    queries = []
    for i, (t0, positive, control) in enumerate(paired):
        for kind, tube in (("p", positive), ("c", control)):
            center = tube[t0]
            for t in tube:
                queries.append(((i, kind, "s", t), t, center))
                queries.append(((i, kind, "m", t), t, tube[t]))
    response = infer_queries(net, arr, queries)

    y, single, static, motion = [], [], [], []
    for i, (t0, positive, control) in enumerate(paired):
        vals_by_kind = {}
        for kind, tube in (("p", positive), ("c", control)):
            times = list(tube)
            one = response[(i, kind, "m", t0)]
            static_mean = float(np.mean([response[(i, kind, "s", t)] for t in times]))
            motion_mean = float(np.mean([response[(i, kind, "m", t)] for t in times]))
            vals_by_kind[kind] = (one, static_mean, motion_mean)
        for label, vals in ((1, vals_by_kind["p"]), (0, vals_by_kind["c"])):
            y.append(label); single.append(vals[0]); static.append(vals[1]); motion.append(vals[2])
    return len(events), auc(y, single), auc(y, static), auc(y, motion)


def main():
    net = torch.jit.load(str(ROOT / "weights" / "unet-daxi.pt"), map_location="cpu"); net.eval()
    rng = np.random.default_rng(20260714)
    specs = [(0, "44b6_0113de3b"), (1, "6bba_05db0fb1")]
    print("Oracle-motion accumulation pilot; hard controls are cached DAXI peaks", flush=True)
    for split, crop in specs:
        result = run(split, crop, net, rng)
        if result is None:
            print(f"{crop}: no eligible paired events", flush=True)
            continue
        n, a_single, a_static, a_motion = result
        print(f"{crop}: pairs={n} single_AUC={a_single:.4f} static_AUC={a_static:.4f} "
              f"GT_motion_AUC={a_motion:.4f} delta_motion_vs_single={a_motion-a_single:+.4f}", flush=True)


if __name__ == "__main__":
    main()
