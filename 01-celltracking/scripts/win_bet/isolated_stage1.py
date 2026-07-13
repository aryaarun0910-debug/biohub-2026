"""Isolated-miss gate — Stage 1: full-volume candidate-generation REACHABILITY.

Decision kill-gate for the moonshot. Runs the DAXI U-Net detector on raw train volumes at a
LOW threshold (NO GT coordinates used to generate proposals), extracts multi-scale local
maxima as candidate cell centres (physical microns), and measures whether generated
proposals land within 7um of ISOLATED missed GT nodes (nodes the competition detector
missed AND with no tracked neighbour on either side). If DAXI proposals cannot even reach
the isolated misses, de-novo detection is dead before any tracklet/classifier work.

Bounded: processes a few crops, only the frames that contain isolated misses (capped).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from scipy.ndimage import maximum_filter  # noqa: E402

CACHE = ROOT / "artifacts/kaggle/e0c_cache"
SCALE = np.array([1.625, 0.40625, 0.40625])  # z,y,x um
LOW_THRESH = 0.20  # frozen low detector threshold (candidate generation)
NMS = 3            # local-maxima window (voxels)


def isolated_misses(split: int, crop: str):
    """Return list of (t, z, y, x) voxel coords of ISOLATED missed GT nodes."""
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
    out = []
    for g in map(int, gt.node_ids()):
        if g in matched:
            continue
        preds = [int(p) for p in gt.predecessors(g)]
        succs = [int(s) for s in gt.successors(g)]
        if not (any(p in matched for p in preds) or any(s in matched for s in succs)):
            out.append(pos[g])  # isolated
    return out


def daxi_candidates(zpath: Path, t: int, net) -> np.ndarray:
    frame = np.asarray(zarr.open(zpath / "0", mode="r")[t]).astype(np.float32)
    lo, hi = np.percentile(frame, 1), np.percentile(frame, 99.9)
    fn = np.clip((frame - lo) / (hi - lo + 1e-6), 0, 1)
    with torch.no_grad():
        fg = net(torch.tensor(fn[None, None]))[0, 0].numpy()
    mx = maximum_filter(fg, size=NMS)
    peaks = np.argwhere((fg == mx) & (fg > LOW_THRESH))  # (K,3) z,y,x voxels
    return peaks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", nargs="+", required=True, help="split:crop e.g. 1:6bba_05db0fb1")
    ap.add_argument("--max-frames", type=int, default=40)
    a = ap.parse_args()
    net = torch.jit.load(str(ROOT / "weights" / "unet-daxi.pt"), map_location="cpu"); net.eval()
    tot_iso = tot_reach = 0
    for spec in a.crops:
        split, crop = spec.split(":"); split = int(split)
        iso = isolated_misses(split, crop)
        by_t = {}
        for (t, z, y, x) in iso:
            by_t.setdefault(t, []).append((z, y, x))
        frames = sorted(by_t)[: a.max_frames]
        zpath = ROOT / "data" / "train" / f"{crop}.zarr"
        c_iso = c_reach = 0
        for t in frames:
            peaks = daxi_candidates(zpath, t, net)
            if len(peaks) == 0:
                c_iso += len(by_t[t]); continue
            pk_um = peaks * SCALE
            for (z, y, x) in by_t[t]:
                gm = np.array([z, y, x]) * SCALE
                dmin = np.linalg.norm(pk_um - gm, axis=1).min()
                c_iso += 1; c_reach += int(dmin <= 7.0)
        print(f"{crop} (split{split}): {len(frames)} frames, isolated {c_iso}, "
              f"reached<=7um {c_reach} ({100*c_reach/max(1,c_iso):.1f}%), avg peaks/frame "
              f"{len(daxi_candidates(zpath, frames[0], net)) if frames else 0}", flush=True)
        tot_iso += c_iso; tot_reach += c_reach
    print(f"\n=== STAGE-1 reachability: {tot_reach}/{tot_iso} isolated misses reached by DAXI "
          f"low-thresh proposals = {100*tot_reach/max(1,tot_iso):.1f}% ===")
    print("Gate: if reachability is high, de-novo proposals CAN see the isolated misses "
          "(then tracklet+PU+exact-graph decide value). If low, isolated detection is dead.")


if __name__ == "__main__":
    main()
