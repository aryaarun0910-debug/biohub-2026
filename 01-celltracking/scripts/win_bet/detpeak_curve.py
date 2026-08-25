r"""Replay the ENTIRE detection-threshold curve from one exported peak superset.

WHY
---
The deployed detector threshold is ``BIOHUB_DET_THRESHOLD = 0.96875``, inherited and never
selected. Four independent lines say it is too high:

* The metric charges over-prediction only ``1 - 0.1 x over_prediction`` -- a predicted node
  matching no GT node is never a false positive (Kaggle discussion/733877, VERIFIED).
* ``edge recall ~ node_recall^2 x conditional_linking_accuracy`` (discussion/734604), so node
  recall enters QUADRATICALLY while over-detection is taxed 0.1x linearly.
* Cutting nodes 17% cost ~0.18 edge Jaccard (discussion/724917) -- a steep curve.
* Our own S1 sweep found detection F1 monotone DECREASING in threshold, optimum at or below
  p0.50.

But measuring it one submission per threshold is wasteful: our first arm (0.96875 -> 0.90)
moved node count only **+2.73%**, so the interesting region is far lower and would cost many
slots to find pointwise.

``scripts/kaggle_edits/detpeak_export.py`` solves that. The local-max test
``is_peak = (logits == pooled) & (sigmoid(logits) > det_threshold)`` has a threshold-INDEPENDENT
first half, so the peak set at T=0.5 is a strict superset of the set at any higher T. Exporting
every peak at T=0.5 with its raw logit makes the whole curve [0.5, 1.0] a **pure CPU replay**.
This module is that replay.

WHAT IT MEASURES
----------------
Per threshold: node recall / precision / F1 under the OFFICIAL scorer's matching semantics
(one-to-one bipartite within ``MAX_DISTANCE`` = 7.0 um at scale (1.625, 0.40625, 0.40625)), the
predicted node count, and ``N_pred/N_est``.

One-to-one matters and is not a detail: *"Duplicating a detection costs ~9% and buys nothing.
Node matching is a one-to-one bipartite assignment, so a twin one voxel away matches nothing"*
(discussion/733877). A greedy or many-to-one matcher would overstate recall and hide exactly
the duplicate penalty we need to respect.

WHAT IT DOES NOT MEASURE
------------------------
It does NOT produce a score. It produces the DETECTION curve plus a projection of adjusted edge
Jaccard under the forum's law. The projection assumes conditional linking accuracy is constant
in the threshold, which is the assumption most likely to be wrong: a lower threshold enlarges
the junk candidate pool the linker must reject (*"the detector is finding the real cells, but it
is also producing an enormous junk candidate pool"*, Moawiz, discussion/734604). Treat the
projected optimum as a place to AIM a submission, never as a predicted score.

COORDINATES
-----------
The export writes ``zyx`` in the DOWNSAMPLED grid, before ``coords[:, 1:] *= ds_arr``. So
microns = ``zyx * downsample * scale`` = ``zyx * [1,4,4] * (1.625, 0.40625, 0.40625)`` =
``zyx * 1.625`` isotropically. That identity is asserted at import.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Iterable, Sequence

import numpy as np

MAX_DISTANCE_UM = 7.0                       # src/biotrack/metric.py:29, organizer default
SCALE_UM = (1.625, 0.40625, 0.40625)        # src/biotrack/metric.py:28
DOWNSAMPLE = (1, 4, 4)                      # predict_unet_transformer.py:157
OVER_PREDICTION_TAX = 0.1                   # score line: 1 - 0.1 * over_prediction

_ISO_UM = tuple(s * d for s, d in zip(SCALE_UM, DOWNSAMPLE))
assert all(abs(v - 1.625) < 1e-9 for v in _ISO_UM), (
    f"downsampled peak grid must be isotropic 1.625 um, got {_ISO_UM}"
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def peaks_to_um(zyx: np.ndarray) -> np.ndarray:
    """Downsampled integer peak grid -> microns (isotropic 1.625)."""
    return np.asarray(zyx, dtype=np.float64) * np.asarray(_ISO_UM, dtype=np.float64)


def match_one_to_one(pred_um: np.ndarray, gt_um: np.ndarray,
                     max_distance: float = MAX_DISTANCE_UM) -> int:
    """Count matched pairs under the scorer's one-to-one bipartite rule.

    Uses a KD-tree to bound the problem, then a Hungarian assignment on the surviving pairs.
    A greedy nearest-neighbour matcher would overcount when two predictions crowd one GT node,
    which is precisely the duplicate case the metric refuses to reward.
    """
    if len(pred_um) == 0 or len(gt_um) == 0:
        return 0
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    tree = cKDTree(pred_um)
    pairs = tree.query_ball_point(gt_um, r=max_distance)
    gi = [g for g, ps in enumerate(pairs) for _ in ps]
    pi = [p for ps in pairs for p in ps]
    if not gi:
        return 0
    gi_u = sorted(set(gi)); pi_u = sorted(set(pi))
    g_index = {g: i for i, g in enumerate(gi_u)}
    p_index = {p: i for i, p in enumerate(pi_u)}
    big = max_distance * 10.0
    cost = np.full((len(gi_u), len(pi_u)), big, dtype=np.float64)
    for g, p in zip(gi, pi):
        cost[g_index[g], p_index[p]] = float(np.linalg.norm(gt_um[g] - pred_um[p]))
    rows, cols = linear_sum_assignment(cost)
    return int(sum(1 for r, c in zip(rows, cols) if cost[r, c] <= max_distance))


def curve(peaks: dict[int, tuple[np.ndarray, np.ndarray]],
          gt: dict[int, np.ndarray],
          thresholds: Sequence[float],
          n_est: int | None = None) -> list[dict]:
    """Detection curve over ``thresholds``.

    ``peaks``: frame -> (zyx int array, raw logit array). ``gt``: frame -> (N,3) microns.
    """
    n_gt = sum(len(v) for v in gt.values())
    rows = []
    for thr in thresholds:
        n_pred = matched = 0
        for t, (zyx, logit) in peaks.items():
            keep = sigmoid(np.asarray(logit)) > thr
            if not keep.any():
                continue
            pred_um = peaks_to_um(np.asarray(zyx)[keep])
            n_pred += len(pred_um)
            g = gt.get(t)
            if g is not None and len(g):
                matched += match_one_to_one(pred_um, g)
        recall = matched / n_gt if n_gt else 0.0
        precision = matched / n_pred if n_pred else 0.0
        f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0
        est = n_est if n_est else n_gt
        over = max(0.0, (n_pred / est) - 1.0) if est else 0.0
        rows.append({
            "threshold": float(thr), "n_pred": int(n_pred), "n_gt": int(n_gt),
            "matched": int(matched), "node_recall": recall, "node_precision": precision,
            "node_f1": f1, "n_pred_over_n_est": (n_pred / est) if est else float("nan"),
            "over_prediction": over,
            # forum law: edge recall ~ node_recall^2, over-prediction taxed 0.1x linearly.
            # The constant conditional-linking-accuracy factor is omitted, so this is a
            # RELATIVE curve for locating an optimum, not an absolute score.
            "projected_edge_index": (recall ** 2) * (1.0 - OVER_PREDICTION_TAX * over),
        })
    return rows


def load_export(path: pathlib.Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    with np.load(path, allow_pickle=False) as z:
        t, zyx, logit = z["t"], z["zyx"], z["logit"]
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for frame in np.unique(t):
        m = t == frame
        out[int(frame)] = (zyx[m], logit[m])
    return out


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--peaks", required=True, type=pathlib.Path,
                    help="detpeaks .npz written by detpeak_export.py")
    ap.add_argument("--gt-npz", type=pathlib.Path,
                    help="optional GT frame->coords npz in MICRONS")
    ap.add_argument("--thresholds",
                    default="0.50,0.60,0.70,0.80,0.85,0.90,0.94,0.96875,0.98,0.99")
    ap.add_argument("--n-est", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args(list(argv) if argv is not None else None)

    peaks = load_export(args.peaks)
    gt: dict[int, np.ndarray] = {}
    if args.gt_npz:
        with np.load(args.gt_npz, allow_pickle=False) as z:
            for k in z.files:
                gt[int(k)] = z[k]
    thr = [float(x) for x in args.thresholds.split(",") if x.strip()]
    rows = curve(peaks, gt, thr, n_est=args.n_est or None)

    print(f"{'thr':>8} {'n_pred':>9} {'recall':>8} {'prec':>8} {'F1':>8} "
          f"{'N/N_est':>8} {'proj':>8}")
    for r in rows:
        star = "  <- DEPLOYED" if abs(r["threshold"] - 0.96875) < 1e-9 else ""
        print(f"{r['threshold']:>8.5f} {r['n_pred']:>9,} {r['node_recall']:>8.4f} "
              f"{r['node_precision']:>8.4f} {r['node_f1']:>8.4f} "
              f"{r['n_pred_over_n_est']:>8.3f} {r['projected_edge_index']:>8.4f}{star}")
    if rows:
        best = max(rows, key=lambda r: r["projected_edge_index"])
        print(f"\nprojected optimum at threshold {best['threshold']:.5f} "
              f"(index {best['projected_edge_index']:.4f})")
        print("NOTE: the projection assumes conditional linking accuracy is CONSTANT in the "
              "threshold. A lower threshold enlarges the junk pool the linker must reject, so "
              "this locates where to AIM a submission -- it is not a predicted score.")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
