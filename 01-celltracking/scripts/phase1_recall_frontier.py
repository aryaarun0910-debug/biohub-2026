"""Phase-1 step 1a: measure the over-proposal recall-vs-count frontier.

For each crop, over-propose ONCE (cached), then sweep the response threshold; at each threshold
compute node recall (fraction of GT nodes with a candidate within 7um) and count ratio
(total candidates / estimated_number_of_nodes). This answers: can over-proposal reach recall
>0.95 at ratio <=2x? -> the step-1a go/no-go for building arbitration.

Writes reports/inventory/phase1_recall_frontier.csv and reports/figures/fig7_recall_frontier.png.

Usage: python scripts/phase1_recall_frontier.py --per-embryo 12 --workers 7
"""

import argparse
import csv
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from biotrack import cache, propose  # noqa: E402
from biotrack.metric import estimated_nodes  # noqa: E402
from biotrack.metric_numpy import Sample, gt_candidate_within  # noqa: E402
from run_v3_taxonomy import geff_to_sample  # noqa: E402

TRAIN = ROOT / "data" / "train"
OUT = ROOT / "reports" / "inventory" / "phase1_recall_frontier.csv"
# NMS radius (um) is the recall lever (finding step 1a); sweep it. V3 detector uses 4.0.
NMS_SWEEP = [4.0, 3.2, 2.0, 1.5, 1.0]


def cand_sample(frames: list[np.ndarray]) -> Sample:
    t, zyx = [], []
    for ti, f in enumerate(frames):
        if len(f):
            t.extend([ti] * len(f)); zyx.append(f[:, :3])
    zyx = np.concatenate(zyx, 0) if zyx else np.zeros((0, 3))
    return Sample(node_ids=np.arange(len(zyx), dtype=np.int64), t=np.asarray(t, np.int64),
                  zyx=zyx.astype(float), edges=np.zeros((0, 2), np.int64))


def worker(crop: str) -> list[dict]:
    gt = geff_to_sample(str(TRAIN / f"{crop}.geff"))
    n_est = estimated_nodes(str(TRAIN / f"{crop}.geff"))
    rows = []
    for nms in NMS_SWEEP:
        cfg = propose.ProposeConfig(nms_dist_um=nms)
        frames = propose.propose_volume(TRAIN / f"{crop}.zarr", cfg)
        pred = cand_sample(frames)
        cand = gt_candidate_within(pred, gt)             # gt_id -> candidate within 7um?
        recall = np.mean(list(cand.values())) if cand else float("nan")
        n_cand = int(pred.zyx.shape[0])
        rows.append({"crop": crop, "fam": crop.split("_")[0], "nms_um": nms,
                     "recall": round(float(recall), 4), "n_cand": n_cand,
                     "ratio": round(n_cand / n_est, 4) if n_est else float("nan")})
    # cache the loosest (max-recall) proposal for downstream arbitration steps
    cache.save(crop, propose.propose_volume(TRAIN / f"{crop}.zarr", propose.ProposeConfig(nms_dist_um=1.0)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-embryo", type=int, default=12)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()

    crops = sorted(p.stem for p in TRAIN.glob("*.geff") if (TRAIN / f"{p.stem}.zarr").exists())
    by_fam = defaultdict(list)
    for c in crops:
        by_fam[c.split("_")[0]].append(c)
    picks = [c for names in by_fam.values() for c in names[: args.per_embryo]]

    print(f"Recall frontier on {len(picks)} crops, {args.workers} workers...")
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            allrows = [r for rs in pool.imap_unordered(worker, picks) for r in rs]
    else:
        allrows = [r for c in picks for r in worker(c)]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["crop", "fam", "nms_um", "recall", "n_cand", "ratio"])
        w.writeheader(); w.writerows(allrows)
    print(f"Wrote {OUT} ({len(allrows)} rows)")

    print("\n=== recall frontier over NMS (mean over crops) ===")
    print(f"{'fam':6s} {'nms_um':>6s} {'recall':>7s} {'ratio':>7s}")
    agg = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))
    for r in allrows:
        a = agg[r["fam"]][r["nms_um"]]; a[0] += r["recall"]; a[1] += r["ratio"]; a[2] += 1
    for fam in sorted(agg):
        for nms in NMS_SWEEP:
            s, rt, n = agg[fam][nms]
            print(f"{fam:6s} {nms:6.1f} {s/n:7.3f} {rt/n:7.2f}")
    print("\nMax recall at ratio<=2.0 per embryo (the step-1a ceiling):")
    for fam in sorted(agg):
        ok = [agg[fam][t][0]/agg[fam][t][2] for t in NMS_SWEEP if agg[fam][t][1]/agg[fam][t][2] <= 2.0]
        print(f"  {fam}: {max(ok, default=float('nan')):.3f}  (target >0.95)")


if __name__ == "__main__":
    main()
