"""H1-T — export the external event set as a Kaggle dataset for the training kernel.

Writes NPZ, not parquet: the Kaggle image ships a polars whose compiled backend does not
load (reports/ENVIRONMENT_TRAPS.md), so the kernel must not depend on it.

Emits the feature matrix, labels, and the grouping keys the leave-one-embryo-out design and
the per-mother selection rule need, plus a provenance blob carried inside the archive so the
dataset cannot be separated from its licence and config hash.

Usage:
  .venv\\Scripts\\python.exe scripts\\win_bet\\h1t_export_kaggle.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl  # noqa: E402

from h1t_external_critic import EVENTS, FEATS  # noqa: E402
from h1t_zebrahub_events import COORD_SPACE, LINK_GATE_UM, ZARR_SCALE_UM  # noqa: E402
from phaseb_h0c_replay import CFG, CFG_HASH  # noqa: E402

OUT = Path(r"C:\Users\aryaa\Documents\Biohub-CellTracking-2026_RESEARCH\agent_runs\agent4"
           r"\kaggle_dataset")

PROVENANCE = {
    "source": "Zebrahub (CZ Biohub SF, Royer lab)",
    "publication": "Lange et al., Cell 2024, doi:10.1016/j.cell.2024.09.047",
    "licence": "CC BY 4.0 - attribution required",
    "url": "https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/",
    "files_sha256": {
        "ZSNS003_tracks.csv": "12474e015f2b65273d423f88a818a0a5b9196ef96194e76ae751f3bab3019ea6",
        "ZSNS004_tracks.csv": "23cbb173f7917120fff2827a81ad8db92373c6ef33a5593663e8d724a7bfdd90",
        "ZSNS005_tracks.csv": "0747389dd9f906ea95c7d48c967c8fae433d7a4403945893211c62f5be452163",
    },
    "excluded": {"ZSNS001": "division rate 3.2%/frame vs 0.44-0.70% in ZSNS003/4/5 and "
                            "mean track length 13.5 frames -> fragmentation re-linked as "
                            "forks; unusable as division labels"},
    "transformations": [
        "CSV -> parquet (columns only)",
        f"voxel -> micron via published OME-Zarr scale {ZARR_SCALE_UM}; coord space per "
        f"embryo {COORD_SPACE}",
        f"GT-free greedy flow-corrected one-to-one linking, gate {LINK_GATE_UM} um",
        f"candidates from frozen proposer phaseb_h0c_replay.shortlist, cfg {CFG_HASH}",
        "features from h1g_features (byte-identical to deployed phaseb_h1g_features)",
        "labels from Zebrahub lineage only, phaseb_h1a_census convention",
    ],
    "proposer_config": CFG,
    "proposer_config_hash": CFG_HASH,
}


def main() -> None:
    files = sorted(EVENTS.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no shards in {EVENTS}")
    df = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    df = df.with_columns(pl.col("steal_required").cast(pl.Int64))
    df = df.filter(pl.col("label").is_in(["positive", "reliable_negative"]))

    embryos = sorted(df["embryo"].unique().to_list())
    emb_idx = {e: i for i, e in enumerate(embryos)}
    X = df.select(FEATS).to_numpy().astype(np.float32)
    y = (df["label"] == "positive").to_numpy().astype(np.int8)
    emb = np.asarray([emb_idx[e] for e in df["embryo"].to_list()], np.int8)
    t_lo = df["t_lo"].to_numpy().astype(np.int32)
    mother = df["mother"].to_numpy().astype(np.int64)
    # unique per (embryo, window, mother) so the per-mother selection rule is well defined
    group = emb.astype(np.int64) * (1 << 48) + t_lo.astype(np.int64) * (1 << 32) + mother

    OUT.mkdir(parents=True, exist_ok=True)
    npz = OUT / "h1t_external_events.npz"
    np.savez_compressed(npz, X=X, y=y, embryo=emb, t_lo=t_lo, mother=mother, group=group,
                        feats=np.asarray(FEATS), embryos=np.asarray(embryos))
    sha = hashlib.sha256(npz.read_bytes()).hexdigest()
    meta = dict(PROVENANCE)
    meta.update({"rows": int(len(y)), "positives": int(y.sum()), "embryos": embryos,
                 "features": FEATS, "npz_sha256": sha,
                 "shards": [f.name for f in files]})
    (OUT / "PROVENANCE.json").write_text(json.dumps(meta, indent=2))
    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "biohub-h1t-external-forkevents",
        "id": "aryaarun07/biohub-h1t-external-forkevents",
        "licenses": [{"name": "CC-BY-4.0"}]}, indent=2))
    print(f"rows={len(y):,} positives={int(y.sum()):,} embryos={embryos}")
    print(f"wrote {npz} sha256={sha}")


if __name__ == "__main__":
    main()
