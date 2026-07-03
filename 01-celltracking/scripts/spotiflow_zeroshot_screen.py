"""FIRST KILL GATE: zero-shot Spotiflow-3D complement-recall screen.

Question: of the GT cells that loose-NMS DoG MISSES (no candidate within 7um), how many does a
PRETRAINED Spotiflow-3D model (synth_3d / smfish_3d) recover zero-shot? Gate = >=8% at a
manageable proposal ratio. If neither model clears it, do NOT invest in Spotiflow fine-tuning
(pivot to DAXI union + redetection).

Run on the A100 (needs spotiflow + GPU + internet for pretrained weights):
    pip install spotiflow
    python scripts/spotiflow_zeroshot_screen.py --per-embryo 6

Spotiflow-3D expects ~isotropic input; we resample each frame to ~0.8125 um isotropic
(XY downsample x2 -> 0.8125; Z upsample x2 -> 0.8125), predict, then map points back to RAW voxels.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from biotrack import propose  # noqa: E402
from biotrack.metric import estimated_nodes  # noqa: E402
from biotrack.metric_numpy import SCALE, Sample, gt_candidate_within  # noqa: E402
from run_v3_taxonomy import geff_to_sample  # noqa: E402

TRAIN = ROOT / "data" / "train"
ISO_UM = 0.8125
RAW = np.array(SCALE)                       # (1.625, 0.40625, 0.40625)
NORM_PCT = (1.0, 99.8)                       # spotiflow normalization


def resample_iso(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(Z,Y,X) raw -> ~0.8125um isotropic via XY/2 (mean-pool) and Z x2 (linear). Returns
    (iso_vol, zoom) where zoom maps iso-voxel -> raw-voxel per axis."""
    from scipy.ndimage import zoom as ndzoom
    Z, Y, X = vol.shape
    xy = vol[:, : (Y // 2) * 2, : (X // 2) * 2].astype(np.float32)
    xy = xy.reshape(Z, Y // 2, 2, X // 2, 2).mean(axis=(2, 4))     # XY /2
    iso = ndzoom(xy, (2.0, 1.0, 1.0), order=1)                     # Z x2
    # iso voxel sizes: z = 1.625/2 = 0.8125; y,x = 0.40625*2 = 0.8125  (isotropic)
    zoom_to_raw = np.array([1.0 / 2.0, 2.0, 2.0])                  # iso-idx * this = raw-idx
    return iso, zoom_to_raw


def normalize(v: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(v, NORM_PCT)
    return np.clip((v - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def spotiflow_points(model, iso_vol: np.ndarray) -> np.ndarray:
    """Return (N,3) predicted points in ISO-voxel coords."""
    pts, _ = model.predict(normalize(iso_vol), verbose=False)
    return np.asarray(pts, float).reshape(-1, 3)


def _near(a_zyx, b_zyx, gate_um=7.0):
    """Boolean mask over a: is there a b within gate_um (physical)?"""
    if len(a_zyx) == 0:
        return np.zeros(0, bool)
    if len(b_zyx) == 0:
        return np.zeros(len(a_zyx), bool)
    from scipy.spatial import cKDTree
    tree = cKDTree(np.asarray(b_zyx) * RAW)
    d, _ = tree.query(np.asarray(a_zyx) * RAW)
    return d <= gate_um


def screen_crop(crop: str, models: dict, frame_stride: int = 1) -> list[dict]:
    from biotrack.propose import downsample_xy, open_volume, read_frame, propose_frame, to_raw, ProposeConfig
    arr = open_volume(TRAIN / f"{crop}.zarr")
    gt = geff_to_sample(str(TRAIN / f"{crop}.geff"))
    cfg = ProposeConfig(nms_dist_um=1.0)
    T = arr.shape[0]
    ts = list(range(0, T, frame_stride))

    # per-frame: GT nodes, DoG candidates, DoG-missed GT
    per = {}
    total_missed = 0
    for t in ts:
        gt_t = gt.zyx[gt.t == t]
        raw = read_frame(arr, t)
        dog = to_raw(propose_frame(downsample_xy(raw, 4), cfg), 4)[:, :3]
        missed_mask = ~_near(gt_t, dog) if len(gt_t) else np.zeros(0, bool)
        per[t] = {"raw": raw, "gt_missed": gt_t[missed_mask]}
        total_missed += int(missed_mask.sum())

    rows = []
    for name, model in models.items():
        recovered = sf_pts = 0
        for t in ts:
            iso, zoom = resample_iso(per[t]["raw"])
            p = spotiflow_points(model, iso)
            if len(p):
                p = p * zoom
            sf_pts += len(p)
            miss = per[t]["gt_missed"]
            if len(miss):
                recovered += int(_near(miss, p).sum())
        rows.append({"crop": crop, "fam": crop.split("_")[0], "model": name,
                     "dog_missed": total_missed, "recovered": recovered,
                     "complement_recall": round(recovered / max(total_missed, 1), 4),
                     "sf_points_per_frame": round(sf_pts / max(len(ts), 1), 1)})
    return rows


def _cand_sample(frames):
    t, zyx = [], []
    for ti, f in enumerate(frames):
        if len(f):
            t.extend([ti] * len(f)); zyx.append(f[:, :3])
    zyx = np.concatenate(zyx, 0) if zyx else np.zeros((0, 3))
    return Sample(node_ids=np.arange(len(zyx)), t=np.asarray(t), zyx=zyx.astype(float),
                  edges=np.zeros((0, 2), np.int64))


def _pts_sample(pts_by_t):
    t, zyx = [], []
    for ti, p in enumerate(pts_by_t):
        if len(p):
            t.extend([ti] * len(p)); zyx.append(p)
    zyx = np.concatenate(zyx, 0) if zyx else np.zeros((0, 3))
    return Sample(node_ids=np.arange(len(zyx)), t=np.asarray(t), zyx=zyx.astype(float),
                  edges=np.zeros((0, 2), np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-embryo", type=int, default=6)
    ap.add_argument("--frame-stride", type=int, default=1, help="subsample frames (speed; use ~20 on CPU)")
    args = ap.parse_args()
    from spotiflow.model import Spotiflow

    models = {}
    for name in ("synth_3d", "smfish_3d"):
        try:
            models[name] = Spotiflow.from_pretrained(name)
            print(f"loaded {name}")
        except Exception as e:
            print(f"could not load {name}: {e}")
    if not models:
        print("no spotiflow models loaded; aborting"); return

    crops = sorted(p.stem for p in TRAIN.glob("*.geff") if (TRAIN / f"{p.stem}.zarr").exists())
    by_fam = defaultdict(list)
    for c in crops:
        by_fam[c.split("_")[0]].append(c)
    picks = [c for names in by_fam.values() for c in names[: args.per_embryo]]

    allrows = []
    for crop in picks:
        allrows.extend(screen_crop(crop, models, args.frame_stride))
        print("done", crop, allrows[-len(models):])

    import csv
    out = ROOT / "reports" / "inventory" / "spotiflow_zeroshot.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    print(f"\nWrote {out}")
    print("\n=== complement recall of DoG-missed GT (pooled over crops) ===")
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0, 0]))
    for r in allrows:
        a = agg[r["model"]][r["fam"]]
        a[0] += r["recovered"]; a[1] += r["dog_missed"]; a[2] += r["sf_points_per_frame"]; a[3] += 1
    for model in agg:
        for fam in sorted(agg[model]):
            rec, miss, ppf, n = agg[model][fam]
            print(f"  {model:10s} {fam}: complement_recall={rec/max(miss,1):.3f} (target>=0.08)  "
                  f"sf_pts/frame~{ppf/n:.0f}  (missed={miss})")
    print("\nGATE: >=8% complement recall on BOTH embryos -> Spotiflow fine-tune is worth it.")


if __name__ == "__main__":
    main()
