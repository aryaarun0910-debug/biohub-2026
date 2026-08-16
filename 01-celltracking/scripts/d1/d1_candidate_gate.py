"""Gate a fitted D1 head on the deployable rejected-local-maximum population.

`d1f_probe.py` answers whether a 33-parameter head transfers between families at the
sampled-voxel level.  That is necessary but not sufficient for re-acceptance.  This gate
asks the candidate-consistent question:

* positives: T-class GT nodes (unmatched, with an unaccepted local maximum <= 7 um),
  represented at that maximum;
* reliable negatives: rejected local maxima farther than 7 um from every GT node;
* negative mass: Horvitz-corrected from the uniform per-frame sample of rejected maxima;
* head: frozen on the source family by `d1f_probe`, evaluated only on its target family.

This is still a ranking gate, not a scoring claim.  Any survivor must be installed before
edge inference and scored through the complete P3 graph pipeline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


RADIUS_UM = 7.0
ARMS = ("H0", "LIN_HEAD")
PRIMARY_SOURCE_PRECISION = 0.70


def _head(direction: dict, arm: str) -> tuple[np.ndarray, float]:
    rec = direction["arms"][arm]["deployed_head"]
    w = np.asarray(rec["w"], dtype=np.float64)
    b = float(rec["b"])
    if w.shape != (32,) or not np.isfinite(w).all() or not np.isfinite(b):
        raise ValueError(f"{direction['direction']}/{arm}: invalid deployed head")
    return w, b


def _weighted_auc(pos: np.ndarray, neg: np.ndarray, neg_w: np.ndarray) -> float:
    """P(score_positive > score_negative), with half credit for ties."""
    if not len(pos) or not len(neg) or float(neg_w.sum()) <= 0:
        return float("nan")
    order = np.argsort(neg, kind="mergesort")
    x, w = neg[order], neg_w[order]
    cw = np.cumsum(w)
    total = float(cw[-1])
    lo = np.searchsorted(x, pos, side="left")
    hi = np.searchsorted(x, pos, side="right")
    below = np.where(lo > 0, cw[np.maximum(lo - 1, 0)], 0.0)
    at = np.where(hi > lo, cw[hi - 1] - below, 0.0)
    return float(np.mean((below + 0.5 * at) / total))


def _frontier(pos: np.ndarray, neg: np.ndarray, neg_w: np.ndarray,
              *, total_positive: int | None = None) -> dict:
    """Best recall attainable at fixed precision and compact operating points."""
    if not len(pos):
        return {"n_positive": int(total_positive or 0), "n_positive_represented": 0,
                "operating_points": [], "recall_at_precision": {}}
    score = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)])
    wt = np.concatenate([np.ones(len(pos)), neg_w])
    order = np.argsort(-score, kind="mergesort")
    score, y, wt = score[order], y[order], wt[order]
    tp = np.cumsum(wt * y)
    fp = np.cumsum(wt * (1 - y))
    precision = tp / np.maximum(tp + fp, 1e-12)
    denom = int(len(pos) if total_positive is None else total_positive)
    if denom < len(pos):
        raise ValueError("total_positive cannot be smaller than represented positives")
    recall = tp / max(denom, 1)

    at_precision = {}
    for p in (0.55, 0.60, 0.70, 0.80, 0.90):
        good = precision >= p
        at_precision[f"{p:.2f}"] = float(recall[good].max()) if good.any() else 0.0

    points = []
    for target_recall in (0.05, 0.10, 0.20, 0.30, 0.50, 0.70):
        idx = np.flatnonzero(recall >= target_recall)
        if not len(idx):
            continue
        i = int(idx[0])
        points.append({
            "target_recall": target_recall,
            "threshold": float(score[i]),
            "tp": float(tp[i]),
            "fp_ht": float(fp[i]),
            "precision_ht": float(precision[i]),
            "recall": float(recall[i]),
        })
    return {
        "n_positive": int(denom),
        "n_positive_represented": int(len(pos)),
        "negative_mass_ht": float(neg_w.sum()),
        "recall_at_precision": at_precision,
        "operating_points": points,
    }


def _cell_index(cell_dirs: list[str]) -> dict[tuple[int, str], Path]:
    out = {}
    for raw in cell_dirs:
        root = Path(raw)
        rec = json.loads((root / "d1_audit" / "d1_cell.json").read_text())
        key = (int(rec["split"]), str(rec["family"]))
        if key in out:
            raise ValueError(f"duplicate cell {key}: {out[key]} and {root}")
        out[key] = root
    return out


def _derived_index(derived_dirs: list[str]) -> dict[tuple[int, str], Path]:
    out = {}
    for raw in derived_dirs:
        root = Path(raw)
        reports = list(root.glob("d1_derived_split*.json"))
        if len(reports) != 1:
            raise ValueError(f"{root}: expected one derived report, got {reports}")
        rep = json.loads(reports[0].read_text())
        split = int(rep["fold"])
        families = sorted({str(c).split("_")[0] for c in rep["crops"]})
        if len(families) != 1:
            raise ValueError(f"{root}: derived cell spans families {families}")
        out[(split, families[0])] = root
    return out


def _candidate_arrays(cell: Path, derived: Path, split: int) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, int]:
    """Return exact candidate scores/features, HT weights, crop ids, and T denominator.

    The v6 `feat_max` array is the strongest maximum inside 15 um.  It represents the
    T-defining <=7 um maximum only when `best15_dist_um <= 7`; otherwise it is a different
    point and must not be passed to a learned head.  H0 does not need a feature vector:
    `best7_logit` is the exact score of the T-defining candidate and remains available for
    every T row.
    """
    dp = derived / f"d1_derived_split{split}.parquet"
    xp = derived / f"d1_derived_split{split}__feat_tta_mean_max.npy"
    ddf, xmax = pl.read_parquet(dp), np.load(xp)
    if len(ddf) != len(xmax):
        raise ValueError(f"{derived}: derived feature alignment failed")
    audit = cell / "d1_audit"
    raw_frames = []
    for rp in sorted(audit.glob("*__rows.parquet")):
        raw_frames.append((rp.name.split("__", 1)[0], pl.read_parquet(rp)))
    raw_gt = pl.concat([
        rows.filter(pl.col("kind") == "gt_centre").select(
            "dataset", "row_id", "best15_dist_um", "best7_logit")
        for _, rows in raw_frames
    ], how="vertical_relaxed")
    ddf = ddf.join(raw_gt, on=["dataset", "row_id"], how="left", validate="1:1")
    tall = ddf["d1_class"].to_numpy() == "T"
    best15_d = ddf["best15_dist_um"].to_numpy().astype(np.float64)
    tmask = tall & np.isfinite(xmax).all(axis=1) & (best15_d <= RADIUS_UM)
    pos = xmax[tmask].astype(np.float64, copy=False)
    pos_crop = ddf["dataset"].to_numpy()[tmask].astype(str)
    h0_pos = ddf["best7_logit"].to_numpy().astype(np.float64)[tall]
    h0_pos_crop = ddf["dataset"].to_numpy()[tall].astype(str)
    if not np.isfinite(h0_pos).all():
        raise ValueError(f"{derived}: T row without a finite best7_logit")

    neg_x, neg_w, neg_crop, h0_neg = [], [], [], []
    for crop, rows in raw_frames:
        feat = np.load(audit / f"{crop}__feat_tta_mean_gt.npy")
        if len(rows) != len(feat):
            raise ValueError(f"{crop}: raw feature alignment failed")
        cand = (rows["kind"].to_numpy() == "subthr_localmax")
        reliable = rows["dist_to_nearest_gt_um"].to_numpy() > RADIUS_UM
        use = cand & reliable
        neg_x.append(feat[use].astype(np.float64, copy=False))
        neg_crop.append(np.full(int(use.sum()), crop, dtype=object))
        h0_neg.append(rows["logit"].to_numpy().astype(np.float64)[use])

        # Sampling is uniform without replacement within each frame.  Every sampled row
        # carries the full frame population, so pi = n_sampled_frame / n_total_frame.
        ts = rows["t"].to_numpy()
        totals = rows["n_subthr_localmax_in_frame"].to_numpy().astype(np.float64)
        w = np.zeros(len(rows), dtype=np.float64)
        for t in np.unique(ts[cand]):
            frame = cand & (ts == t)
            n = int(frame.sum())
            if n:
                w[frame] = float(totals[np.flatnonzero(frame)[0]]) / n
        neg_w.append(w[use])
    if not neg_x:
        raise ValueError(f"{cell}: no rejected-local-maximum rows")
    return (pos, np.concatenate(neg_x), np.concatenate(neg_w), pos_crop,
            np.concatenate(neg_crop).astype(str), h0_pos, np.concatenate(h0_neg),
            h0_pos_crop, int(tall.sum()))


def _select_source_threshold(pos: np.ndarray, neg: np.ndarray, neg_w: np.ndarray,
                             *, precision_floor: float) -> dict:
    """Freeze the most permissive source-only threshold meeting a precision floor."""
    if not len(pos):
        return {"eligible": False, "reason": "no source T positives"}
    score = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos), dtype=np.int8), np.zeros(len(neg), dtype=np.int8)])
    wt = np.concatenate([np.ones(len(pos)), neg_w])
    order = np.argsort(-score, kind="mergesort")
    score, y, wt = score[order], y[order], wt[order]
    tp = np.cumsum(wt * y)
    fp = np.cumsum(wt * (1 - y))
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / len(pos)
    good = np.flatnonzero(precision >= precision_floor)
    if not len(good):
        return {"eligible": False, "reason": "precision floor unattainable",
                "precision_floor": precision_floor}
    # Maximise recall, then use the highest threshold among ties.  The decision is made
    # wholly on source held-out crops and is applied byte-for-byte to the target family.
    best_recall = float(recall[good].max())
    tied = good[np.isclose(recall[good], best_recall, rtol=0.0, atol=1e-12)]
    i = int(tied[0])
    return {"eligible": True, "precision_floor": precision_floor,
            "threshold_logit": float(score[i]), "tp": float(tp[i]),
            "fp_ht": float(fp[i]), "precision_ht": float(precision[i]),
            "recall": float(recall[i]), "n_positive": int(len(pos)),
            "negative_mass_ht": float(neg_w.sum())}


def _fixed_operating_point(pos: np.ndarray, neg: np.ndarray, neg_w: np.ndarray,
                           threshold: float, *, total_positive: int | None = None) -> dict:
    tp = float((pos >= threshold).sum())
    fp = float(neg_w[neg >= threshold].sum())
    denom = int(len(pos) if total_positive is None else total_positive)
    return {"threshold_logit": float(threshold), "tp": tp, "fp_ht": fp,
            "precision_ht": tp / max(tp + fp, 1e-12),
            "recall": tp / max(denom, 1), "n_positive": denom,
            "n_positive_represented": int(len(pos)),
            "negative_mass_ht": float(neg_w.sum())}


def run(probe_json: str, cell_dirs: list[str], derived_dirs: list[str]) -> dict:
    probe = json.loads(Path(probe_json).read_text())
    cells, derived = _cell_index(cell_dirs), _derived_index(derived_dirs)
    reports = []
    for direction in probe["directions"]:
        split = int(direction["checkpoint_basis_split"])
        family = str(direction["target_family"])
        source_family = str(direction["source_family"])
        key, source_key = (split, family), (split, source_family)
        if key not in cells or key not in derived or source_key not in cells or source_key not in derived:
            raise ValueError(f"missing source/target artifacts for {source_key} -> {key}")
        (pos_x, neg_x, neg_w, _, _, h0_pos, h0_neg, _, n_t) = _candidate_arrays(
            cells[key], derived[key], split)
        (spos_x, sneg_x, sneg_w, spos_crop, sneg_crop, sh0_pos, sh0_neg,
         sh0_crop, _) = _candidate_arrays(
            cells[source_key], derived[source_key], split)
        held = set(direction["arms"]["LIN_HEAD"]["same_family_heldout_crops"]["crops"])
        sp = np.isin(spos_crop, sorted(held)); sn = np.isin(sneg_crop, sorted(held))
        arms = {}
        for arm in ARMS:
            w, b = _head(direction, arm)
            if arm == "H0":
                pos, neg = h0_pos, h0_neg
                source_pos = sh0_pos[np.isin(sh0_crop, sorted(held))]
                source_neg = sh0_neg[sn]
                total = n_t
            else:
                pos, neg = pos_x @ w + b, neg_x @ w + b
                source_pos, source_neg = spos_x[sp] @ w + b, sneg_x[sn] @ w + b
                total = n_t
            selected = _select_source_threshold(
                source_pos, source_neg, sneg_w[sn], precision_floor=PRIMARY_SOURCE_PRECISION)
            fixed = (None if not selected["eligible"] else
                     _fixed_operating_point(pos, neg, neg_w, selected["threshold_logit"],
                                            total_positive=total))
            arms[arm] = {
                "auc_ht": _weighted_auc(pos, neg, neg_w),
                **_frontier(pos, neg, neg_w, total_positive=total),
                "source_heldout_selection": selected,
                "target_at_frozen_source_threshold": fixed,
                "feature_coverage": len(pos) / max(total, 1),
            }
        reports.append({
            "direction": direction["direction"],
            "checkpoint_basis_split": split,
            "target_family": family,
            "n_T_total": int(n_t),
            "n_T_with_exact_exported_feature": int(len(pos_x)),
            "n_reliable_negative_samples": int(len(neg_x)),
            "arms": arms,
        })
    return {
        "kind": "d1_candidate_consistent_gate",
        "basis": (
            "T-class max features versus Horvitz-corrected rejected-local-max negatives; "
            "source-fitted head and source-heldout threshold frozen before target family; "
            "ranking gate, not graph score"
        ),
        "primary_source_precision_floor": PRIMARY_SOURCE_PRECISION,
        "directions": reports,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe-json", required=True)
    ap.add_argument("--cell-dir", action="append", required=True)
    ap.add_argument("--derived-dir", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    result = run(a.probe_json, a.cell_dir, a.derived_dir)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for d in result["directions"]:
        print(d["direction"], "T=", d["n_T_total"],
              "exact-feature=", d["n_T_with_exact_exported_feature"])
        for arm, row in d["arms"].items():
            r60 = row["recall_at_precision"].get("0.60", 0.0)
            print(" ", arm, "AUC", f"{row['auc_ht']:.4f}",
                  "R@P.60", f"{r60:.3f}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
