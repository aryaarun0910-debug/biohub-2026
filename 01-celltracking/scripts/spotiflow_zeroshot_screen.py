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


def screen_crop(crop: str, models: dict) -> list[dict]:
    from spotiflow.utils import imread as _  # noqa: F401  (ensures spotiflow importable)
    from biotrack.propose import open_volume, read_frame
    arr = open_volume(TRAIN / f"{crop}.zarr")
    gt = geff_to_sample(str(TRAIN / f"{crop}.geff"))
    n_est = estimated_nodes(str(TRAIN / f"{crop}.geff"))

    # DoG candidates (loose NMS) -> which GT nodes DoG MISSES
    dog_frames = propose.propose_volume(TRAIN / f"{crop}.zarr", propose.ProposeConfig(nms_dist_um=1.0))
    dog = _cand_sample(dog_frames)
    dog_hit = gt_candidate_within(dog, gt)                 # gt_id -> DoG candidate within 7um?
    missed = {g for g, h in dog_hit.items() if not h}

    rows = []
    for name, model in models.items():
        pts_by_t = []
        for t in range(arr.shape[0]):
            iso, zoom_to_raw = resample_iso(read_frame(arr, t))
            p = spotiflow_points(model, iso)
            if len(p):
                p = p * zoom_to_raw                        # iso-idx -> raw-idx
            pts_by_t.append(p)
        sf = _pts_sample(pts_by_t)
        sf_hit = gt_candidate_within(sf, gt)               # gt_id -> spotiflow point within 7um?
        recovered = sum(1 for g in missed if sf_hit.get(g, False))
        rows.append({"crop": crop, "fam": crop.split("_")[0], "model": name,
                     "dog_missed": len(missed), "recovered": recovered,
                     "complement_recall": round(recovered / max(len(missed), 1), 4),
                     "sf_points": int(sf.zyx.shape[0]),
                     "sf_ratio": round(sf.zyx.shape[0] / n_est, 3) if n_est else float("nan")})
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
        allrows.extend(screen_crop(crop, models))
        print("done", crop)

    import csv
    out = ROOT / "reports" / "inventory" / "spotiflow_zeroshot.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys())); w.writeheader(); w.writerows(allrows)
    print(f"\nWrote {out}")
    print("\n=== complement recall of DoG-missed GT (mean over crops) ===")
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0, 0]))
    for r in allrows:
        a = agg[r["model"]][r["fam"]]
        a[0] += r["recovered"]; a[1] += r["dog_missed"]; a[2] += r["sf_ratio"]; a[3] += 1
    for model in agg:
        for fam in sorted(agg[model]):
            rec, miss, rat, n = agg[model][fam]
            print(f"  {model:10s} {fam}: complement_recall={rec/max(miss,1):.3f} (target>=0.08)  sf_ratio~{rat/n:.2f}")
    print("\nGATE: >=8% complement recall on BOTH embryos -> Spotiflow fine-tune is worth it.")


if __name__ == "__main__":
    main()
