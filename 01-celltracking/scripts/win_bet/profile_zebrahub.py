"""Profile the Zebrahub dense lineage CSVs to ground WIN_BET Milestone 0.

Reports, per embryo: rows, unique tracks, timepoint range, births, divisions
(a track that is the parent of >=2 distinct child tracks), mean track length,
per-timepoint node density, and coordinate ranges. Pure polars, CPU, read-only.
"""

from pathlib import Path

import polars as pl

DATA = Path("data/external/zebrahub")
FILES = ["ZSNS001_tracks.csv", "ZSNS003_tracks.csv", "ZSNS004_tracks.csv", "ZSNS005_tracks.csv"]


def profile(path: Path) -> dict:
    df = pl.read_csv(path)
    n_rows = df.height
    n_tracks = df["track_id"].n_unique()
    t_min, t_max = int(df["t"].min()), int(df["t"].max())
    n_t = df["t"].n_unique()

    # First appearance of each track = its birth row (min t per track).
    firsts = (
        df.sort("t").group_by("track_id").first().select("track_id", "parent_track_id")
    )
    n_births = firsts.filter(pl.col("parent_track_id") == -1).height

    # Divisions: parent_track_id that is the parent of >=2 distinct child tracks.
    child_parents = firsts.filter(pl.col("parent_track_id") != -1)
    div = (
        child_parents.group_by("parent_track_id")
        .agg(pl.col("track_id").n_unique().alias("n_children"))
    )
    n_divisions = div.filter(pl.col("n_children") >= 2).height
    n_children_gt2 = div.filter(pl.col("n_children") > 2).height

    # Track length distribution (n timepoints per track).
    tlen = df.group_by("track_id").agg(pl.len().alias("n")).select("n")
    # Per-timepoint density.
    dens = df.group_by("t").agg(pl.len().alias("n")).select("n")

    return {
        "file": path.name,
        "rows": n_rows,
        "tracks": n_tracks,
        "t_range": (t_min, t_max),
        "n_timepoints": n_t,
        "births": n_births,
        "divisions(>=2)": n_divisions,
        "multi_children(>2)": n_children_gt2,
        "track_len_mean": round(float(tlen["n"].mean()), 1),
        "track_len_median": int(tlen["n"].median()),
        "nodes_per_t_mean": round(float(dens["n"].mean()), 1),
        "nodes_per_t_max": int(dens["n"].max()),
        "z_range": (round(float(df["z"].min()), 1), round(float(df["z"].max()), 1)),
        "y_range": (round(float(df["y"].min()), 1), round(float(df["y"].max()), 1)),
        "x_range": (round(float(df["x"].min()), 1), round(float(df["x"].max()), 1)),
    }


def main() -> None:
    for f in FILES:
        p = DATA / f
        if not p.exists():
            print(f"MISSING {f}")
            continue
        r = profile(p)
        print("=" * 60)
        for k, v in r.items():
            print(f"  {k:22s} {v}")


if __name__ == "__main__":
    main()
