# =====================================================================================
# PRE-WRAPPER PREDICTION-GRAPH EXPORT
#
# Injected immediately after `print(f"Found {len(geffs)} prediction graphs")`, i.e. after
# inference has produced the geffs and BEFORE `filter_output_graph` consumes them.
#
# WHY. Every post-processing question this project asks -- the motion-relink gate quantity,
# the 6/10 um radii, the flow estimator, the gap-close threshold -- is a function of the
# PRE-wrapper graph: raw detector nodes (unsmoothed floats) plus raw edges carrying
# `edge_prob`. The submission CSV is the POST-wrapper graph: coordinates linefit-smoothed
# then written as max(0, int(round(v))), nodes already pruned by prune-isolated and the
# short-track component filter, edges already replaced wholesale by the motion relink, and
# `edge_prob` dropped entirely. A post-wrapper cache therefore cannot express a pre-wrapper
# stage change, and no amount of local compute recovers what the CSV never carried.
#
# The fold-0 LOEO kernel FAILED at its export cell, so its /kaggle/working survived and its
# 71 geffs were recoverable with zero GPU. The fold-1 kernel COMPLETED, so the cleanup block
# ran and its geffs are gone -- 128 crops, 6bba, 85.06% of the objective's edge mass, with
# no pre-wrapper surface anywhere. This cell exists so that never happens again: one small
# parquet makes every future gate replay a CPU job.
#
# Trap 23: the geff list is the notebook's own, already asserted equal to `test_stems`
# above; this cell re-asserts it rather than trusting a glob, and dies naming any crop whose
# tables come back empty.
# =====================================================================================
import hashlib as _pg_hashlib
from pathlib import Path as _PgPath

import polars as _pg_pl

_pg_out = _PgPath("/kaggle/working") / f"pregraphs_split{LOEO_FOLD}.parquet"
_pg_nodes: list = []
_pg_edges: list = []
_pg_rows: list = []

_pg_expect = sorted(str(s) for s in test_stems)
_pg_seen = sorted(p.stem for p in geffs)
if _pg_seen != _pg_expect:
    raise RuntimeError({"expected": _pg_expect[:8], "actual": _pg_seen[:8],
                        "n_expected": len(_pg_expect), "n_actual": len(_pg_seen)})

for _pg_path in geffs:
    _pg_ds = _pg_path.stem
    _pg_g = graph_from_geff(_pg_path)
    _pg_na = _pg_g.node_attrs()
    _pg_ea = _pg_g.edge_attrs()
    if _pg_na.height == 0:
        raise RuntimeError(f"{_pg_ds}: pre-wrapper graph has no nodes")
    _pg_nodes.append(_pg_pl.DataFrame({
        "dataset": [_pg_ds] * _pg_na.height,
        "node_id": [int(v) for v in _pg_na["node_id"]],
        "t": [int(v) for v in _pg_na["t"]],
        "z": [float(v) for v in _pg_na["z"]],
        "y": [float(v) for v in _pg_na["y"]],
        "x": [float(v) for v in _pg_na["x"]],
    }, schema={"dataset": _pg_pl.String, "node_id": _pg_pl.Int64, "t": _pg_pl.Int64,
               "z": _pg_pl.Float64, "y": _pg_pl.Float64, "x": _pg_pl.Float64}))
    _pg_has_prob = "edge_prob" in _pg_ea.columns
    _pg_prob = ([None if v is None else float(v) for v in _pg_ea["edge_prob"]]
                if _pg_has_prob else [None] * _pg_ea.height)
    _pg_edges.append(_pg_pl.DataFrame({
        "dataset": [_pg_ds] * _pg_ea.height,
        "source_id": [int(v) for v in _pg_ea["source_id"]],
        "target_id": [int(v) for v in _pg_ea["target_id"]],
        "edge_prob": _pg_prob,
    }, schema={"dataset": _pg_pl.String, "source_id": _pg_pl.Int64,
               "target_id": _pg_pl.Int64, "edge_prob": _pg_pl.Float64}))
    _pg_rows.append({"dataset": _pg_ds, "nodes": int(_pg_na.height),
                     "edges": int(_pg_ea.height), "has_edge_prob": bool(_pg_has_prob),
                     "n_edge_prob_null": int(sum(v is None for v in _pg_prob))})

_pg_nd = _pg_pl.concat(_pg_nodes)
_pg_ed = _pg_pl.concat(_pg_edges)
del _pg_nodes, _pg_edges

# One file, two row groups' worth of schema -- nodes and edges are unioned with a row_type
# discriminator so a single parquet round-trips both without a second artifact.
_pg_frame = _pg_pl.concat([
    _pg_nd.with_columns([
        _pg_pl.lit("node").alias("row_type"),
        _pg_pl.lit(None, dtype=_pg_pl.Int64).alias("source_id"),
        _pg_pl.lit(None, dtype=_pg_pl.Int64).alias("target_id"),
        _pg_pl.lit(None, dtype=_pg_pl.Float64).alias("edge_prob"),
    ]).select("dataset", "row_type", "node_id", "t", "z", "y", "x",
              "source_id", "target_id", "edge_prob"),
    _pg_ed.with_columns([
        _pg_pl.lit("edge").alias("row_type"),
        _pg_pl.lit(None, dtype=_pg_pl.Int64).alias("node_id"),
        _pg_pl.lit(None, dtype=_pg_pl.Int64).alias("t"),
        _pg_pl.lit(None, dtype=_pg_pl.Float64).alias("z"),
        _pg_pl.lit(None, dtype=_pg_pl.Float64).alias("y"),
        _pg_pl.lit(None, dtype=_pg_pl.Float64).alias("x"),
    ]).select("dataset", "row_type", "node_id", "t", "z", "y", "x",
              "source_id", "target_id", "edge_prob"),
])
_pg_frame.write_parquet(_pg_out, compression="zstd")

_pg_summary = {
    "fold": LOEO_FOLD,
    "arm": LOEO_ARM,
    "n_crops": len(_pg_rows),
    "total_nodes": int(sum(r["nodes"] for r in _pg_rows)),
    "total_edges": int(sum(r["edges"] for r in _pg_rows)),
    "crops_without_edge_prob": [r["dataset"] for r in _pg_rows if not r["has_edge_prob"]],
    "bytes": int(_pg_out.stat().st_size),
    "sha256": _pg_hashlib.sha256(_pg_out.read_bytes()).hexdigest(),
    "per_crop": _pg_rows,
}
_pg_json = _PgPath("/kaggle/working") / f"pregraphs_split{LOEO_FOLD}.json"
_pg_json.write_text(json.dumps(_pg_summary, indent=2))
print(f"PRE-WRAPPER EXPORT: {_pg_summary['n_crops']} crops, "
      f"{_pg_summary['total_nodes']:,} nodes, {_pg_summary['total_edges']:,} edges -> "
      f"{_pg_out} ({_pg_summary['bytes']:,} bytes, sha256 {_pg_summary['sha256'][:16]})")
if _pg_summary["crops_without_edge_prob"]:
    print("  !! crops missing edge_prob:", _pg_summary["crops_without_edge_prob"][:10])
del _pg_frame, _pg_nd, _pg_ed
