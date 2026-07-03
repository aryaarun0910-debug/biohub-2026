"""Phase-1 ablation harness: score any pipeline CONFIG across both embryo folds on the exact
numpy metric, writing a per-crop CSV. Configs plug in as the ladder progresses (v3 baseline now;
overpropose / arbitrate / redetect added in steps 3-6). Comparing two configs = diffing their CSVs.

Reports per-crop adjJ / recall / count ratio / edge TP-FP-FN / 3-way taxonomy, aggregated to the
per-embryo (fold) min-adjJ (the selection gate). Parallel across crops.

Usage:
    python scripts/run_phase1_ablation.py --config v3 --all --workers 7
    python scripts/run_phase1_ablation.py --config v3 --per-embryo 10
"""

import argparse
import csv
import importlib.util
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from biotrack.metric import estimated_nodes  # noqa: E402
from biotrack.metric_numpy import Sample, score_sample  # noqa: E402
from run_v3_taxonomy import classify_edges, geff_to_sample, rows_to_sample  # noqa: E402

_spec = importlib.util.spec_from_file_location("dog_p1", ROOT / "notebooks" / "kaggle_dog_infer.py")
dog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dog)

TRAIN = ROOT / "data" / "train"
OUTDIR = ROOT / "reports" / "inventory"
FIELDS = ["crop", "fam", "n_pred", "n_est", "ratio", "recall", "adjJ",
          "edge_tp", "edge_fp", "edge_fn", "tp", "no_cand", "lost_assign", "assoc"]


# --------------------------------------------------------------------------- pipeline configs
def cfg_v3(crop: str) -> Sample:
    """Baseline: the reproduced public V3 pipeline."""
    return rows_to_sample(dog.infer_dataset(TRAIN / f"{crop}.zarr"))


def _op_dedup(crop: str, nms_um: float, r_same_um: float, rep: str) -> Sample:
    """Over-propose (loose NMS) -> same-cell conflict sets -> ONE representative per set -> V3 link.
    rep: 'bright' (max DoG response) or 'central' (min mean physical dist within set)."""
    from biotrack import arbitrate, propose
    frames = propose.propose_volume(TRAIN / f"{crop}.zarr", propose.ProposeConfig(nms_dist_um=nms_um))
    cents_by_t = []
    for f in frames:
        if len(f) == 0:
            cents_by_t.append(np.zeros((0, 3), np.float32)); continue
        coords = f[:, :3]
        labels = arbitrate.same_cell_sets(coords, r_same_um=r_same_um)
        reps = []
        for lab in np.unique(labels):
            idx = np.where(labels == lab)[0]
            if rep == "bright":
                reps.append(idx[np.argmax(f[idx, 3])])            # max response
            else:
                p = coords[idx] * np.array(arbitrate.SCALE)
                reps.append(idx[np.argmin(np.linalg.norm(p - p.mean(0), axis=1))])
        cents_by_t.append(coords[reps])
    rows = dog.build_rows(crop, cents_by_t, dog.link_frames(cents_by_t))
    return rows_to_sample(rows)


def cfg_op_bright(crop: str) -> Sample:
    return _op_dedup(crop, nms_um=1.0, r_same_um=3.0, rep="bright")


def smooth_sample(s: Sample, w: float = 0.7) -> Sample:
    """Temporal coordinate smoothing (public V11 lever, ~0.854): blend each linked node's position
    toward its edge-neighbours' mean. Smoother tracks localize better -> the 7um 1/(1+d) matcher
    rewards it. w = weight on the node's own position."""
    if len(s.node_ids) == 0:
        return s
    id_to_i = {int(n): i for i, n in enumerate(s.node_ids)}
    nbr = {i: [] for i in range(len(s.node_ids))}
    for a, b in s.edges:
        ia, ib = id_to_i.get(int(a)), id_to_i.get(int(b))
        if ia is not None and ib is not None:
            nbr[ia].append(ib); nbr[ib].append(ia)
    zyx = s.zyx.copy()
    out = s.zyx.copy()
    for i, ns in nbr.items():
        if ns:
            out[i] = w * zyx[i] + (1 - w) * zyx[ns].mean(axis=0)
    return Sample(node_ids=s.node_ids, t=s.t, zyx=out, edges=s.edges)


def cfg_v3_smooth(crop: str) -> Sample:
    return smooth_sample(cfg_v3(crop))


def cfg_op_bright_smooth(crop: str) -> Sample:
    return smooth_sample(cfg_op_bright(crop))


CONFIGS = {"v3": cfg_v3, "op_bright": cfg_op_bright,
           "v3_smooth": cfg_v3_smooth, "op_bright_smooth": cfg_op_bright_smooth}
# steps 5-6 add: arbitration-scored representatives, track-conditioned redetection


def score_crop(crop: str, pred: Sample) -> dict:
    gt = geff_to_sample(str(TRAIN / f"{crop}.geff"))
    n_est = estimated_nodes(str(TRAIN / f"{crop}.geff"))
    s = score_sample(pred, gt, n_est)
    tax = classify_edges(pred, gt)
    return {"crop": crop, "fam": crop.split("_")[0], "n_pred": s["num_pred_nodes"],
            "n_est": int(n_est), "ratio": round(s["num_pred_nodes"] / n_est, 4) if n_est else float("nan"),
            "recall": round(s["node_recall"], 4), "adjJ": round(s["adj_edge_jaccard"], 4),
            "edge_tp": s["edge_tp"], "edge_fp": s["edge_fp"], "edge_fn": s["edge_fn"], **tax}


def worker(crop: str, config: str) -> dict:
    return score_crop(crop, CONFIGS[config](crop))


def fold_summary(rows: list[dict]) -> dict:
    agg = defaultdict(lambda: defaultdict(float))
    for r in rows:
        a = agg[r["fam"]]
        w = r["edge_tp"] + r["edge_fp"] + r["edge_fn"]
        a["adjw"] += r["adjJ"] * w; a["w"] += w; a["recall"] += r["recall"]; a["n"] += 1
    out = {f: {"adjJ": a["adjw"] / a["w"], "recall": a["recall"] / a["n"], "n": int(a["n"])}
           for f, a in agg.items()}
    out["min_fold_adjJ"] = min(v["adjJ"] for f, v in out.items() if f != "min_fold_adjJ")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="v3", choices=list(CONFIGS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--per-embryo", type=int, default=10)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()

    crops = sorted(p.stem for p in TRAIN.glob("*.geff") if (TRAIN / f"{p.stem}.zarr").exists())
    if not args.all:
        by_fam = defaultdict(list)
        for c in crops:
            by_fam[c.split("_")[0]].append(c)
        crops = [c for names in by_fam.values() for c in names[: args.per_embryo]]

    print(f"[{args.config}] on {len(crops)} crops, {args.workers} workers...")
    fn = partial(worker, config=args.config)
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            rows = list(pool.imap_unordered(fn, crops))
    else:
        rows = [fn(c) for c in crops]
    rows.sort(key=lambda r: r["crop"])

    out = OUTDIR / f"phase1_{args.config}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    fs = fold_summary(rows)
    print(f"Wrote {out} ({len(rows)} crops)")
    for f in sorted(k for k in fs if k != "min_fold_adjJ"):
        print(f"  fold {f}: adjJ={fs[f]['adjJ']:.4f} recall={fs[f]['recall']:.3f} (n={fs[f]['n']})")
    print(f"  MIN-FOLD adjJ = {fs['min_fold_adjJ']:.4f}  <- the selection gate")


if __name__ == "__main__":
    main()
