# =====================================================================================
# LOEO EXPORT / SELF-AUDIT CELL  (replaces P0-A's frame-retention audit cell)
#
# The base cell asserts the dual-seed frame-retention contract, which the `strict` arm
# deliberately does not satisfy (the secondary model is switched off because it was
# trained on all 199 train crops). Asserting a contract the run intentionally broke
# would be theatre, so this cell asserts the contract this run DOES have:
#
#   * schema and contiguity identical to a real submission,
#   * emitted dataset set == the fold's crop list exactly,
#   * lineage validity (t(target) == t(source)+1, in-degree <= 1, out-degree <= 2),
#   * per-crop node/edge/fork counts recorded for cross-checking against a known-good
#     artifact -- the module-global-leak trap is caught by comparing counts, not by
#     trusting an "ok" log line.
#
# It then gzips the artifact and removes the bulky working-dir entries, because
# `kaggle kernels output` pulls the ENTIRE working directory (168 files including
# weights) and times out. What survives is a handful of small named files that
# scripts/kaggle_factory.py fetches by URL.
# =====================================================================================
from collections import Counter
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

_LOEO_COLUMNS = [
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id",
]
_loeo_csv = Path("/kaggle/working/submission.csv")
if not _loeo_csv.is_file():
    raise FileNotFoundError(_loeo_csv)

_loeo_df = pd.read_csv(_loeo_csv)
if _loeo_df.empty or _loeo_df.columns.tolist() != _LOEO_COLUMNS:
    raise RuntimeError(f"LOEO schema changed: {_loeo_df.columns.tolist()}")
if _loeo_df["id"].tolist() != list(range(len(_loeo_df))):
    raise RuntimeError("LOEO row ids are not contiguous")
if set(_loeo_df["row_type"].unique()) != {"node", "edge"}:
    raise RuntimeError("LOEO row types changed")

_loeo_seen = sorted(_loeo_df["dataset"].astype(str).unique())
if _loeo_seen != sorted(LOEO_STEMS):
    raise RuntimeError({"expected": sorted(LOEO_STEMS), "actual": _loeo_seen})

_loeo_rows = []
for _crop, _grp in _loeo_df.groupby("dataset", sort=True):
    _n = _grp[_grp["row_type"].eq("node")]
    _e = _grp[_grp["row_type"].eq("edge")]
    if _n.empty:
        raise RuntimeError(f"{_crop}: no nodes")
    if _n["t"].lt(0).any() or _n[["z", "y", "x"]].lt(0).any().any():
        raise RuntimeError(f"{_crop}: negative time or coordinate")
    _t = dict(zip(_n["node_id"].astype(int), _n["t"].astype(int)))
    _indeg, _outdeg = Counter(), Counter()
    for _row in _e.itertuples():
        _s, _d = int(_row.source_id), int(_row.target_id)
        if _s not in _t or _d not in _t or _t[_d] != _t[_s] + 1:
            raise RuntimeError(f"{_crop}: invalid lineage edge {_s}->{_d}")
        _indeg[_d] += 1
        _outdeg[_s] += 1
    if _indeg and max(_indeg.values()) > 1:
        raise RuntimeError(f"{_crop}: in-degree > 1")
    if _outdeg and max(_outdeg.values()) > 2:
        raise RuntimeError(f"{_crop}: out-degree > 2")
    _loeo_rows.append({
        "dataset": _crop,
        "nodes": int(len(_n)),
        "edges": int(len(_e)),
        "forks": int(sum(1 for v in _outdeg.values() if v == 2)),
        "t_max": int(_n["t"].max()),
        "z_max": int(_n["z"].max()),
        "y_max": int(_n["y"].max()),
        "x_max": int(_n["x"].max()),
    })

_loeo_summary = {
    **LOEO_MANIFEST,
    "rows": len(_loeo_df),
    "total_nodes": int(sum(r["nodes"] for r in _loeo_rows)),
    "total_edges": int(sum(r["edges"] for r in _loeo_rows)),
    "total_forks": int(sum(r["forks"] for r in _loeo_rows)),
    "submission_sha256": hashlib.sha256(_loeo_csv.read_bytes()).hexdigest(),
    "per_crop": _loeo_rows,
}
_loeo_out = Path(f"/kaggle/working/loeo_split{LOEO_FOLD}_{LOEO_ARM}.json")
_loeo_out.write_text(json.dumps(_loeo_summary, indent=2))
print(json.dumps({k: v for k, v in _loeo_summary.items()
                  if k not in ("per_crop", "crops")}, indent=2))

# gzip the artifact (a 199-crop CSV is ~600 MB raw, ~50 MB gzipped)
_loeo_gz = Path(f"/kaggle/working/loeo_split{LOEO_FOLD}_{LOEO_ARM}.csv.gz")
with _loeo_csv.open("rb") as _fin, gzip.open(_loeo_gz, "wb", compresslevel=6) as _fout:
    shutil.copyfileobj(_fin, _fout)
print(f"gzipped -> {_loeo_gz} ({_loeo_gz.stat().st_size:,} bytes, "
      f"sha256 {hashlib.sha256(_loeo_gz.read_bytes()).hexdigest()})")

# Trap 11: keep the output listing tiny so it can be fetched by URL.
_LOEO_KEEP = {_loeo_gz.name, _loeo_out.name, "loeo_manifest.json", "run_stats.csv"}
for _p in sorted(Path("/kaggle/working").iterdir()):
    if _p.name in _LOEO_KEEP:
        continue
    try:
        shutil.rmtree(_p) if _p.is_dir() else _p.unlink()
    except OSError as _exc:
        print(f"  could not remove {_p}: {_exc}")
print("kept:", sorted(p.name for p in Path("/kaggle/working").iterdir()))
