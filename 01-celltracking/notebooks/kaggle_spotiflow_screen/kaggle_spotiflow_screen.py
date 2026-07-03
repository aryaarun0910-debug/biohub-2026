"""Kaggle T4 zero-shot Spotiflow-3D complement-recall screen (the FIRST KILL GATE).

Self-contained: reads competition volumes + GT geffs via tensorstore (Kaggle has no zarr),
computes loose-NMS DoG candidates, runs pretrained synth_3d/smfish_3d on FULL frames, and reports
what fraction of DoG-MISSED GT nodes each model recovers within 7 um. Gate = >=8% on BOTH embryos.

Kaggle: GPU=T4, INTERNET=ON (pip install spotiflow + pretrained weights). Not a submission.
"""
import subprocess, sys
print("installing spotiflow...", flush=True)
r = subprocess.run([sys.executable, "-m", "pip", "install", "spotiflow"], capture_output=True, text=True)
print("pip rc:", r.returncode, flush=True)
if r.returncode != 0:
    print("PIP STDERR:", r.stderr[-3000:], flush=True)
try:
    import spotiflow; print("spotiflow", spotiflow.__version__, flush=True)
except Exception as e:
    print("SPOTIFLOW IMPORT FAILED:", repr(e), flush=True)

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorstore as ts
from scipy.ndimage import gaussian_filter, zoom as ndzoom
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

COMP = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train")
SCALE = np.array([1.625, 0.40625, 0.40625]); ISO = 1.625; ISO_SF = 0.8125
PER_EMBRYO = 6
GATE_UM = 7.0


def open_vol(zarr_dir: Path):
    return ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(zarr_dir / "0")}}).result()


def read_frame(arr, t):
    return np.asarray(arr[t].read().result())


def read_geff_nodes(geff: Path):
    def col(name):
        p = geff / "nodes" / "props" / name / "values"
        return np.asarray(ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(p)}}).result().read().result())
    return col("t").astype(int), np.stack([col("z"), col("y"), col("x")], 1).astype(float)


# --- DoG over-proposal (loose NMS), mirrors src/biotrack/propose.py core ---
def downsample_xy(vol, f=4):
    Z, Y, X = vol.shape
    return vol[:, :(Y // f) * f, :(X // f) * f].astype(np.float32).reshape(Z, Y // f, f, X // f, f).mean((2, 4))


def normalize(vol, q=(0.01, 0.997)):
    lo, hi = np.quantile(vol, q); return np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def dog_candidates(raw, nms_um=1.0, scale_pairs=((1.5, 4.0), (2.2, 5.5)), rel=0.02):
    iso = normalize(downsample_xy(raw, 4))
    resp = None
    for s1, s2 in scale_pairs:
        d = gaussian_filter(iso, s1 / ISO) - gaussian_filter(iso, s2 / ISO)
        resp = d if resp is None else np.maximum(resp, d)
    thr = rel * float(resp.max()) if resp.max() > 0 else rel
    mind = max(1, int(round(nms_um / ISO)))
    pk = peak_local_max(resp, min_distance=mind, threshold_abs=thr, exclude_border=False, num_peaks=60000)
    if len(pk) == 0:
        return np.zeros((0, 3))
    out = pk.astype(np.float32); out[:, 1] = out[:, 1] * 4 + 1.5; out[:, 2] = out[:, 2] * 4 + 1.5
    return out


def resample_iso(vol):
    Z, Y, X = vol.shape
    xy = vol[:, :(Y // 2) * 2, :(X // 2) * 2].astype(np.float32).reshape(Z, Y // 2, 2, X // 2, 2).mean((2, 4))
    iso = ndzoom(xy, (2.0, 1.0, 1.0), order=1)
    return iso, np.array([0.5, 2.0, 2.0])


def near(a, b, gate=GATE_UM):
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), bool)
    d, _ = cKDTree(np.asarray(b) * SCALE).query(np.asarray(a) * SCALE)
    return d <= gate


def sf_norm(v, p=(1.0, 99.8)):
    lo, hi = np.percentile(v, p); return np.clip((v - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def main():
    from spotiflow.model import Spotiflow
    models = {}
    for n in ("synth_3d", "smfish_3d"):
        try:
            models[n] = Spotiflow.from_pretrained(n); print("loaded", n)
        except Exception as e:
            print("skip", n, e)

    crops = sorted({p.name[:-5] for p in COMP.glob("*.zarr")} & {p.name[:-5] for p in COMP.glob("*.geff")})
    by = defaultdict(list)
    for c in crops:
        by[c.split("_")[0]].append(c)
    picks = [c for names in by.values() for c in names[:PER_EMBRYO]]
    print(f"{len(picks)} crops:", picks)

    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # model -> fam -> [recovered, missed]
    for crop in picks:
        arr = open_vol(COMP / f"{crop}.zarr")
        gt_t, gt_zyx = read_geff_nodes(COMP / f"{crop}.geff")
        fam = crop.split("_")[0]; T = arr.shape[0]
        missed_by_t = {}
        for t in range(T):
            gt_f = gt_zyx[gt_t == t]
            if len(gt_f) == 0:
                continue
            dog = dog_candidates(read_frame(arr, t))
            missed_by_t[t] = gt_f[~near(gt_f, dog)]
        n_missed = sum(len(v) for v in missed_by_t.values())
        for name, model in models.items():
            rec = 0
            for t, miss in missed_by_t.items():
                if len(miss) == 0:
                    continue
                iso, zoom = resample_iso(read_frame(arr, t))
                pts, _ = model.predict(sf_norm(iso), verbose=False)
                pts = np.asarray(pts, float).reshape(-1, 3) * zoom if len(pts) else np.zeros((0, 3))
                rec += int(near(miss, pts).sum())
            agg[name][fam][0] += rec; agg[name][fam][1] += n_missed
            print(f"  {crop} {name}: recovered {rec}/{n_missed}")

    print("\n=== complement recall (pooled) ===")
    rows = []
    for name in agg:
        for fam in sorted(agg[name]):
            rec, miss = agg[name][fam]
            cr = rec / max(miss, 1)
            print(f"  {name:10s} {fam}: {cr:.3f}  (recovered {rec}/{miss})  target>=0.08")
            rows.append({"model": name, "fam": fam, "recovered": rec, "missed": miss, "complement_recall": round(cr, 4)})
    import csv
    with open("/kaggle/working/spotiflow_zeroshot.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\nGATE: >=8% on BOTH embryos -> fine-tune worthwhile; else pivot to DAXI union + redetection.")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.stdout.flush()
