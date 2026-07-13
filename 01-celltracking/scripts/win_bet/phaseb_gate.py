"""Phase-B competition-transfer gate on sparse, answerable source groups.

Train on one competition embryo family and evaluate on the other. The unit of
learning and evaluation is a source candidate group containing an annotated true
successor. Unannotated groups never become synthetic negatives.

This is the competition-only control. External breadth must improve on this model,
not merely on nearest-neighbour or pooled candidate AUC.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
LABELED = ROOT / "artifacts/kaggle/e0c_cache/candidates_labeled_v2"

FEATURES = [
    "raw_gate", "motion_gate", "raw_rel", "motion_rel", "rank_frac",
    "cost_delta", "edge_prob", "tight", "log_n_cand",
]


def load_fold(split: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    costs: list[np.ndarray] = []
    groups: list[int] = []
    crops: list[str] = []
    gid = 0
    for path in sorted((LABELED / str(split)).glob("*.parquet")):
        d = pl.read_parquet(path).filter(pl.col("answerable_group"))
        if d.height == 0:
            continue
        d = d.with_columns(
            pl.col("raw_um").median().over("source_id").alias("raw_med"),
            pl.col("motion_um").median().over("source_id").alias("motion_med"),
            pl.col("cost").min().over("source_id").alias("cost_min"),
        ).with_columns(
            (pl.col("raw_um") / pl.col("gate_um").clip(1e-6, None)).alias("raw_gate"),
            (pl.col("motion_um") / pl.col("gate_um").clip(1e-6, None)).alias("motion_gate"),
            (pl.col("raw_um") / pl.col("raw_med").clip(1e-6, None)).alias("raw_rel"),
            (pl.col("motion_um") / pl.col("motion_med").clip(1e-6, None)).alias("motion_rel"),
            (pl.col("cost_rank") / (pl.col("n_cand") - 1).clip(1, None)).alias("rank_frac"),
            (pl.col("cost") - pl.col("cost_min")).alias("cost_delta"),
            pl.col("edge_prob").fill_null(0.0).fill_nan(0.0),
            (pl.col("pass") == "tight").cast(pl.Float64).alias("tight"),
            pl.col("n_cand").cast(pl.Float64).log1p().alias("log_n_cand"),
        ).sort(["source_id", "cost_rank"])
        for part in d.partition_by("source_id", maintain_order=True):
            xs.append(part.select(FEATURES).to_numpy().astype(np.float32))
            ys.append(part["rank_label"].to_numpy().astype(np.int8))
            costs.append(part["cost"].to_numpy().astype(np.float32))
            groups.extend([gid] * part.height)
            crops.append(path.stem)
            gid += 1
    if not xs:
        raise RuntimeError(f"no answerable groups for split {split}")
    return np.vstack(xs), np.concatenate(ys), np.concatenate(costs), np.asarray(groups), crops


def ranking_metrics(y: np.ndarray, score: np.ndarray, group: np.ndarray) -> dict[str, float]:
    top1 = 0
    reciprocal: list[float] = []
    for gid in np.unique(group):
        idx = np.flatnonzero(group == gid)
        order = np.argsort(-score[idx], kind="mergesort")
        hits = np.flatnonzero(y[idx][order] == 1)
        if not len(hits):
            raise AssertionError("gate contains a non-answerable group")
        top1 += int(hits[0] == 0)
        reciprocal.append(1.0 / (int(hits[0]) + 1))
    n = len(reciprocal)
    return {"groups": n, "top1": top1 / n, "mrr": float(np.mean(reciprocal))}


def group_sizes(group: np.ndarray) -> np.ndarray:
    _, counts = np.unique(group, return_counts=True)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/inventory/phaseb_gate.txt")
    args = ap.parse_args()
    lines: list[str] = []
    for train_split, test_split, family in ((1, 0, "44b6"), (0, 1, "6bba")):
        Xtr, ytr, _, gtr, _ = load_fold(train_split)
        Xte, yte, cost, gte, _ = load_fold(test_split)
        model = lgb.LGBMRanker(
            objective="lambdarank", n_estimators=args.trees, learning_rate=0.04,
            num_leaves=15, min_child_samples=100, subsample=0.8,
            colsample_bytree=0.9, reg_lambda=2.0, random_state=20260713,
            verbosity=-1, n_jobs=-1,
        )
        model.fit(Xtr, ytr, group=group_sizes(gtr))
        pred = model.predict(Xte)
        base = ranking_metrics(yte, -cost, gte)
        learned = ranking_metrics(yte, pred, gte)
        importance = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
        block = [
            f"=== held-out {family}; train split {train_split}, test split {test_split} ===",
            f"train rows={len(ytr):,} groups={len(np.unique(gtr)):,}; "
            f"test rows={len(yte):,} groups={len(np.unique(gte)):,}",
            f"wrapper composite: top1={base['top1']:.6f} mrr={base['mrr']:.6f}",
            f"competition ranker: top1={learned['top1']:.6f} mrr={learned['mrr']:.6f}",
            f"delta: top1={learned['top1']-base['top1']:+.6f} "
            f"mrr={learned['mrr']-base['mrr']:+.6f}",
            "importance: " + ", ".join(f"{k}={v}" for k, v in importance), "",
        ]
        print("\n".join(block), flush=True)
        lines.extend(block)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
