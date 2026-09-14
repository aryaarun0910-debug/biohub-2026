#!/usr/bin/env python3
"""Select a post-process config by LEAVE-ONE-EMBRYO-OUT, not by a pooled proxy.

The hidden test is a THIRD embryo (host: "no overlap in embryo_ids between train and test"), so
the question a config must answer is transfer to an embryo never seen -- not "does it win on a
pooled mix of the two we have". Their sweep pools all 8 validator stems, and our reweighting
experiment pushed that pooling further toward 6bba, which was reasoning from the four PUBLIC
EXAMPLE films and is now known to be the wrong target.

validator_results.csv is per-stem, so the right selection can be computed offline from any run we
already have. For each config, score it separately on the 44b6 stems and on the 6bba stems, and
rank by the MINIMUM. A config that wins on one embryo and loses on the other has not shown
transfer; it has shown it fits one embryo.

Scoring mirrors metrics.summarise: adj weight-averaged by each sample's annotated edge count,
divJ micro-averaged, score = adj + 0.1*divJ.
"""
import sys
from pathlib import Path
import numpy as np, polars as pl

path = Path(sys.argv[1] if len(sys.argv) > 1 else "work/ppgrid_out/validator_results.csv")
df = pl.read_csv(path)
need = {"stem", "config", "adjusted_edge_jaccard", "edge_tp", "edge_fp", "edge_fn",
        "div_tp", "div_fp", "div_fn"}
missing = need - set(df.columns)
if missing:
    sys.exit(f"missing columns: {sorted(missing)}")


def score(d):
    w = (d["edge_tp"] + d["edge_fp"] + d["edge_fn"]).to_numpy().astype(float)
    a = d["adjusted_edge_jaccard"].to_numpy()
    ok = ~np.isnan(a) & (w > 0)
    if not ok.any(): return float("nan")
    adj = float(np.average(a[ok], weights=w[ok]))
    dt, dfp, dfn = d["div_tp"].sum(), d["div_fp"].sum(), d["div_fn"].sum()
    dj = dt / max(dt + dfp + dfn, 1)
    return adj + 0.1 * dj


base = df.filter(pl.col("config") == "base")
b_all, b_44, b_6b = (score(base),
                     score(base.filter(pl.col("stem").str.starts_with("44b6"))),
                     score(base.filter(pl.col("stem").str.starts_with("6bba"))))
print(f"  {path}\n")
print(f"  {'config':<34}{'pooled':>9}{'44b6':>9}{'6bba':>9}{'MIN d':>9}{'pooled d':>10}")
print("  " + "-" * 82)
rows = []
for cfg in df["config"].unique().to_list():
    d = df.filter(pl.col("config") == cfg)
    s_all = score(d)
    s44 = score(d.filter(pl.col("stem").str.starts_with("44b6")))
    s6b = score(d.filter(pl.col("stem").str.starts_with("6bba")))
    rows.append((cfg, s_all, s44, s6b, min(s44 - b_44, s6b - b_6b), s_all - b_all))
rows.sort(key=lambda r: -r[4])
for cfg, s_all, s44, s6b, dmin, dall in rows:
    star = "  <-- best by MIN" if cfg == rows[0][0] else ""
    print(f"  {cfg[:33]:<34}{s_all:>9.5f}{s44:>9.5f}{s6b:>9.5f}{dmin:>+9.5f}{dall:>+10.5f}{star}")

pooled_best = max(rows, key=lambda r: r[5])
print(f"\n  best by POOLED proxy : {pooled_best[0]}   (min delta {pooled_best[4]:+.5f})")
print(f"  best by MIN-EMBRYO   : {rows[0][0]}   (pooled delta {rows[0][5]:+.5f})")
if pooled_best[0] != rows[0][0]:
    print("\n  THE TWO RULES DISAGREE. The pooled winner does not transfer best across embryos,")
    print("  and the hidden test is a third embryo.")
