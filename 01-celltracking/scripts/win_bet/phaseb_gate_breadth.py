"""Phase-B controlled gate: does EXTERNAL BREADTH fix the reverse-fold asymmetry?

Three systems ranked on the SAME scale-free geometry features (no edge_prob, no cost --
transferable across domains), evaluated on held-out competition answerable source groups
(per-source top-1 / MRR):
  (1) wrapper composite   -- rank by -cost (the frozen baseline; it DOES use edge_prob)
  (2) competition-only    -- geometry ranker trained on the other competition family
  (3) zebrahub+competition -- geometry ranker trained on 3 Zebrahub embryos + the other family

The competition-only control already fails the both-fold gate (6bba regresses). The test:
does adding Zebrahub breadth lift BOTH folds over the wrapper, especially 6bba.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
LABELED = ROOT / "artifacts/kaggle/e0c_cache/candidates_labeled_v2"
ZEB = ROOT / "artifacts/kaggle/e0c_cache/zebrahub_candidates"
EPS = 1e-6
GEOM = ["raw_rel", "motion_rel", "raw_min_ratio", "motion_min_ratio", "rank_frac", "log_n_cand"]


def geom(d: pl.DataFrame) -> pl.DataFrame:
    return d.with_columns(
        pl.col("raw_um").median().over("source_id").alias("rmed"),
        pl.col("raw_um").min().over("source_id").alias("rmin"),
        pl.col("motion_um").median().over("source_id").alias("mmed"),
        pl.col("motion_um").min().over("source_id").alias("mmin"),
        pl.len().over("source_id").alias("ncand"),
        pl.col("raw_um").rank("ordinal").over("source_id").alias("rrank"),
    ).with_columns(
        (pl.col("raw_um") / pl.col("rmed").clip(EPS, None)).alias("raw_rel"),
        (pl.col("motion_um") / pl.col("mmed").clip(EPS, None)).alias("motion_rel"),
        (pl.col("raw_um") / pl.col("rmin").clip(EPS, None)).alias("raw_min_ratio"),
        (pl.col("motion_um") / pl.col("mmin").clip(EPS, None)).alias("motion_min_ratio"),
        ((pl.col("rrank") - 1) / (pl.col("ncand") - 1).clip(1, None)).alias("rank_frac"),
        pl.col("ncand").cast(pl.Float64).log1p().alias("log_n_cand"),
    )


def blocks(df: pl.DataFrame, label_col: str, extra: list[str] | None = None):
    """Partition by source_id -> per-group feature/label blocks (+ optional extra cols)."""
    xs, ys, gs, ex = [], [], [], []
    for part in df.partition_by("source_id", maintain_order=True):
        xs.append(part.select(GEOM).to_numpy().astype(np.float32))
        ys.append(part[label_col].to_numpy().astype(np.int8))
        gs.append(part.height)
        if extra:
            ex.append({c: part[c].to_numpy() for c in extra})
    return xs, ys, gs, ex


def load_competition(split: int, need_cost: bool):
    xs, ys, gs, ex = [], [], [], []
    for path in sorted((LABELED / str(split)).glob("*.parquet")):
        d = pl.read_parquet(path).filter(pl.col("answerable_group"))
        if d.height == 0:
            continue
        d = geom(d)
        bx, by, bg, be = blocks(d, "rank_label", ["cost"] if need_cost else None)
        xs += bx; ys += by; gs += bg; ex += be
    return xs, ys, gs, ex


def load_zebrahub():
    xs, ys, gs = [], [], []
    for path in sorted(ZEB.glob("*.parquet")):
        d = pl.read_parquet(path)
        if d.height == 0:
            continue
        d = geom(d)
        bx, by, bg, _ = blocks(d, "label")
        xs += bx; ys += by; gs += bg
    return xs, ys, gs


def train(xs, ys, gs, trees):
    model = lgb.LGBMRanker(objective="lambdarank", n_estimators=trees, learning_rate=0.04,
                           num_leaves=15, min_child_samples=100, subsample=0.8,
                           colsample_bytree=0.9, reg_lambda=2.0, random_state=20260713,
                           verbosity=-1, n_jobs=-1)
    model.fit(np.vstack(xs), np.concatenate(ys), group=np.asarray(gs))
    return model


def rank_metrics(blocks_x, blocks_y, score_fn):
    top1 = 0; rr = []
    for bx, by in zip(blocks_x, blocks_y):
        order = np.argsort(-score_fn(bx), kind="mergesort")
        hit = np.flatnonzero(by[order] == 1)
        assert len(hit), "non-answerable group leaked in"
        top1 += int(hit[0] == 0); rr.append(1.0 / (hit[0] + 1))
    n = len(rr)
    return {"groups": n, "top1": top1 / n, "mrr": float(np.mean(rr))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/inventory/phaseb_gate_breadth.txt")
    args = ap.parse_args()

    zx, zy, zg = load_zebrahub()
    print(f"zebrahub: {len(zg):,} groups, {sum(zg):,} candidates", flush=True)
    lines = [f"zebrahub train: {len(zg):,} groups"]

    for tr, te, fam in ((1, 0, "44b6"), (0, 1, "6bba")):
        cxs, cys, cgs, _ = load_competition(tr, need_cost=False)          # comp train fold
        txs, tys, tgs, tex = load_competition(te, need_cost=True)          # comp test fold (+cost)
        comp = train(cxs, cys, cgs, args.trees)
        breadth = train(zx + cxs, zy + cys, zg + cgs, args.trees)

        # wrapper composite = rank by ascending cost (lower cost = better)
        w = {"top1": 0, "rr": []}
        for bx, by, be in zip(txs, tys, tex):
            order = np.argsort(-(-be["cost"]), kind="mergesort")  # ascending cost
            hit = np.flatnonzero(by[order] == 1); assert len(hit)
            w["top1"] += int(hit[0] == 0); w["rr"].append(1.0 / (hit[0] + 1))
        nW = len(w["rr"]); wrapper = {"top1": w["top1"] / nW, "mrr": float(np.mean(w["rr"]))}
        m_comp = rank_metrics(txs, tys, comp.predict)
        m_breadth = rank_metrics(txs, tys, breadth.predict)

        block = [
            f"=== held-out {fam} (test split {te}, {len(tgs):,} answerable groups) ===",
            f"  (1) wrapper composite    : top1={wrapper['top1']:.4f} mrr={wrapper['mrr']:.4f}",
            f"  (2) competition-only geom: top1={m_comp['top1']:.4f} mrr={m_comp['mrr']:.4f}  "
            f"(dvs wrapper {m_comp['top1']-wrapper['top1']:+.4f})",
            f"  (3) zebrahub+comp geom   : top1={m_breadth['top1']:.4f} mrr={m_breadth['mrr']:.4f}  "
            f"(dvs wrapper {m_breadth['top1']-wrapper['top1']:+.4f}, dvs comp-only "
            f"{m_breadth['top1']-m_comp['top1']:+.4f})", "",
        ]
        print("\n".join(block), flush=True)
        lines += block
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
