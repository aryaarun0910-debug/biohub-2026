"""H1-N -- REVERSE-DIRECTION ASSOCIATION DISAGREEMENT over the frozen H0c census.

Motivation. The forward proposer asks, for each mother, which daughter pair best matches
its predicted flow. The reverse question -- for each DAUGHTER, which mother claims it best
-- is a different piece of evidence, and it is exactly the evidence a mother-only gate
cannot see. A true mitotic mother should win BOTH its daughters against every competing
mother; a false mother typically borrows daughters that some other mother explains better.

Computed entirely from the frozen census (cfg hash 04eeac97500d). No GT, no model, no
retuning. One row per (crop, mother, t) using the mother's RANK-0 pair.

  r0            the mother's own rank-0 flow-midpoint residual
  r_alt_d1/d2   min residual over all OTHER mothers on the census proposing that daughter
  rtd_margin    min(r_alt_d1, r_alt_d2) - r0        HIGHER == mother wins its daughters
  rtd_nclaim    number of distinct other mothers proposing d1 or d2
  rtd_loses     1 if some other mother claims a daughter with a strictly lower residual

Usage:
  .venv\\Scripts\\python.exe scripts\\h1n_reverse_contest.py --out <path>.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT / "artifacts/kaggle/e0c_cache/fork_candidates/h0c_top3"
BIG = 1e9


def one(path: Path) -> pl.DataFrame:
    d = pl.read_parquet(path, columns=["crop", "t", "mother", "d1", "d2", "rank",
                                       "flow_midpoint_residual"])
    if d.height == 0:
        return pl.DataFrame()
    # daughter-side index over the WHOLE census for this crop (all ranks, all mothers)
    long = pl.concat([
        d.select(pl.col("d1").alias("d"), "mother", "t", "flow_midpoint_residual"),
        d.select(pl.col("d2").alias("d"), "mother", "t", "flow_midpoint_residual"),
    ])
    # best and second-best residual claiming each daughter, plus the best claimant's mother
    long = long.sort("flow_midpoint_residual")
    agg = long.group_by(["t", "d"]).agg(
        pl.col("flow_midpoint_residual").min().alias("r_best"),
        pl.col("mother").first().alias("m_best"),
        pl.col("flow_midpoint_residual").sort().slice(1, 1).first().alias("r_second"),
        pl.col("mother").n_unique().alias("n_claim"),
    )
    r0 = d.filter(pl.col("rank") == 0)
    out = r0
    for k in ("d1", "d2"):
        out = out.join(agg.rename({"d": k, "r_best": f"rb_{k}", "m_best": f"mb_{k}",
                                   "r_second": f"rs_{k}", "n_claim": f"nc_{k}"}),
                       on=["t", k], how="left")
    # residual of the best OTHER mother for each daughter
    for k in ("d1", "d2"):
        out = out.with_columns(
            pl.when(pl.col(f"mb_{k}") != pl.col("mother"))
              .then(pl.col(f"rb_{k}"))
              .otherwise(pl.col(f"rs_{k}").fill_null(BIG))
              .alias(f"ralt_{k}"))
    out = out.with_columns(
        pl.min_horizontal("ralt_d1", "ralt_d2").alias("r_alt_min"),
        (pl.col("nc_d1") + pl.col("nc_d2") - 2).alias("rtd_nclaim"),
    ).with_columns(
        (pl.col("r_alt_min") - pl.col("flow_midpoint_residual")).alias("rtd_margin"),
    ).with_columns(
        (pl.col("rtd_margin") < 0).cast(pl.Int8).alias("rtd_loses"),
    )
    return out.select("crop", "t", "mother",
                      pl.col("flow_midpoint_residual").alias("rtd_r0"),
                      "r_alt_min", "rtd_margin", "rtd_nclaim", "rtd_loses")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N crops per fold")
    a = ap.parse_args()
    t0 = time.time()
    frames = []
    for fold in (0, 1, 2):
        fs = sorted((CENSUS / str(fold)).glob("*.parquet"))
        if a.limit:
            fs = fs[:a.limit]
        for f in fs:
            r = one(f)
            if r.height:
                frames.append(r)
        print(f"  fold {fold}: {len(fs)} crops  cum_rows={sum(x.height for x in frames):,} "
              f"t={time.time()-t0:.0f}s", flush=True)
    D = pl.concat(frames)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    D.write_parquet(a.out)
    m = D["rtd_margin"].to_numpy()
    print(f"rows={D.height:,} crops={D['crop'].n_unique()} "
          f"loses_frac={float(np.mean(D['rtd_loses'].to_numpy())):.4f} "
          f"margin_med={float(np.nanmedian(m[np.isfinite(m)])):.4f}  -> {a.out} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
