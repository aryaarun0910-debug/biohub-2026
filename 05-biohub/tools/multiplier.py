#!/usr/bin/env python3
"""Compute the EXACT multiplier term of a submission, offline, with no leaderboard feedback.

The test set is four datasets and `estimated_number_of_nodes` for every one of them ships in the
organisers' released GEFF metadata (data/train_geff/<stem>.geff). The score applies

    adj_i = J_i * (1 - 0.1 * (N_pred_i - n_total_i) / n_total_i)

per dataset, so counting node rows per dataset in a submission.csv fixes the multiplier exactly.
Only J_i and the per-sample weights stay unknown. Run this on any candidate before submitting.

    python tools/multiplier.py work/repro_out/submission.csv
"""
import sys
from pathlib import Path
import polars as pl

def n_totals():
    out = {}
    from geff import GeffMetadata
    for s in ("44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"):
        p = Path("data/train_geff") / f"{s}.geff"
        out[s] = (GeffMetadata.read(p).extra or {})["estimated_number_of_nodes"]
    return out

def main(path):
    NT = n_totals()
    df = pl.read_csv(path)
    n = df.filter(pl.col("row_type") == "node").group_by("dataset").len().sort("dataset")
    print(f"  {'dataset':<16}{'N_pred':>9}{'n_total':>9}{'ratio':>9}{'mult':>9}")
    ms = []
    for r in n.iter_rows(named=True):
        d, p = r["dataset"], r["len"]
        if d not in NT:
            print(f"  {d:<16}{p:>9,}   -- no n_total on record"); continue
        ratio = (p - NT[d]) / NT[d]; m = 1 - 0.1 * ratio
        ms.append(m)
        print(f"  {d:<16}{p:>9,}{NT[d]:>9,}{ratio:>+9.3f}{m:>9.4f}")
    if ms:
        print(f"\n  unweighted mean multiplier {sum(ms)/len(ms):.4f}   (max possible 1.1000)")
        print("  NB the score weights each dataset by its ANNOTATED edge count, which is hidden,")
        print("  so this is the multiplier per dataset exactly and the blend only approximately.")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "work/repro_out/submission.csv")
