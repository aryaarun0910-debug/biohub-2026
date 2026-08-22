# =====================================================================================
# PRE-ILP ROLL-UP  ->  one small parquet that survives kernel cleanup
#
# WHY. `pre_ilp_export.py` writes the candidate graph the ILP solver is handed, per crop, to
# /kaggle/working/preilp/*.geff. Those geffs are large and, on a COMPLETED kernel, the cleanup
# block deletes /kaggle/working leftovers -- exactly how the fold-1 geffs were lost once before
# (see the header of loeo_pregraph_export.py). This cell rolls them into one parquet, matching
# that file's schema, so the measurement survives regardless.
#
# WHAT IT MEASURES. The single number the linker-division lane turns on:
#   out_deg_ge2  -- sources carrying >= 2 CANDIDATE edges in the graph handed to the solver.
# The deployed post-ILP graph has out-degree {1: N} everywhere, so every such source is a
# division the solver DECLINED on cost grounds rather than one the model never proposed.
# The GT-matched subset is computed offline against data/train (no GT is read on Kaggle).
#
# Emitted per candidate edge: dataset, source_id, target_id, edge_prob, plus node rows with
# t/z/y/x, so the offline audit can match sources to GT dividing mothers within 7 um and count
# how many declined candidates land on a true second daughter.
# =====================================================================================
import hashlib as _pi_hashlib
from pathlib import Path as _PiPath

import polars as _pi_pl

_pi_src = _PiPath("/kaggle/working/preilp")
_pi_out = _PiPath("/kaggle/working") / f"preilp_split{LOEO_FOLD}.parquet"
_pi_files = sorted(_pi_src.glob("*.geff")) if _pi_src.exists() else []
print(f"pre-ILP roll-up: found {len(_pi_files)} candidate graphs in {_pi_src}")

if not _pi_files:
    print("pre-ILP roll-up: NOTHING TO ROLL UP -- the export patch did not run")
else:
    _pi_nodes: list = []
    _pi_edges: list = []
    _pi_summary: list = []
    for _pi_path in _pi_files:
        _pi_ds = _pi_path.stem
        _pi_g = graph_from_geff(_pi_path)
        _pi_na = _pi_g.node_attrs()
        _pi_ea = _pi_g.edge_attrs()
        if _pi_na.height == 0:
            raise RuntimeError(f"{_pi_ds}: pre-ILP graph has no nodes")
        _pi_nodes.append(_pi_pl.DataFrame({
            "dataset": [_pi_ds] * _pi_na.height,
            "node_id": [int(v) for v in _pi_na["node_id"]],
            "t": [int(v) for v in _pi_na["t"]],
            "z": [float(v) for v in _pi_na["z"]],
            "y": [float(v) for v in _pi_na["y"]],
            "x": [float(v) for v in _pi_na["x"]],
        }, schema={"dataset": _pi_pl.String, "node_id": _pi_pl.Int64, "t": _pi_pl.Int64,
                   "z": _pi_pl.Float64, "y": _pi_pl.Float64, "x": _pi_pl.Float64}))
        _pi_has_prob = "edge_prob" in _pi_ea.columns
        _pi_prob = ([None if v is None else float(v) for v in _pi_ea["edge_prob"]]
                    if _pi_has_prob else [None] * _pi_ea.height)
        _pi_srcs = [int(v) for v in _pi_ea["source_id"]]
        _pi_tgts = [int(v) for v in _pi_ea["target_id"]]
        _pi_edges.append(_pi_pl.DataFrame({
            "dataset": [_pi_ds] * _pi_ea.height,
            "source_id": _pi_srcs,
            "target_id": _pi_tgts,
            "edge_prob": _pi_prob,
        }, schema={"dataset": _pi_pl.String, "source_id": _pi_pl.Int64,
                   "target_id": _pi_pl.Int64, "edge_prob": _pi_pl.Float64}))
        # the headline count, printed per crop so it is visible in the kernel log even if the
        # parquet is somehow lost
        _pi_deg: dict = {}
        for _pi_s in _pi_srcs:
            _pi_deg[_pi_s] = _pi_deg.get(_pi_s, 0) + 1
        _pi_ge2 = sum(1 for _v in _pi_deg.values() if _v >= 2)
        _pi_summary.append((_pi_ds, _pi_na.height, _pi_ea.height, _pi_ge2))
        print(f"  {_pi_ds}: nodes={_pi_na.height} cand_edges={_pi_ea.height} "
              f"sources_with_outdeg>=2={_pi_ge2}", flush=True)

    _pi_nodes_df = _pi_pl.concat(_pi_nodes).with_columns(
        _pi_pl.lit("node").alias("row_type"),
        _pi_pl.lit(None, dtype=_pi_pl.Int64).alias("source_id"),
        _pi_pl.lit(None, dtype=_pi_pl.Int64).alias("target_id"),
        _pi_pl.lit(None, dtype=_pi_pl.Float64).alias("edge_prob"),
    )
    _pi_edges_df = _pi_pl.concat(_pi_edges).with_columns(
        _pi_pl.lit("edge").alias("row_type"),
        _pi_pl.lit(None, dtype=_pi_pl.Int64).alias("node_id"),
        _pi_pl.lit(None, dtype=_pi_pl.Int64).alias("t"),
        _pi_pl.lit(None, dtype=_pi_pl.Float64).alias("z"),
        _pi_pl.lit(None, dtype=_pi_pl.Float64).alias("y"),
        _pi_pl.lit(None, dtype=_pi_pl.Float64).alias("x"),
    )
    _pi_cols = ["dataset", "row_type", "node_id", "t", "z", "y", "x",
                "source_id", "target_id", "edge_prob"]
    _pi_all = _pi_pl.concat([_pi_nodes_df.select(_pi_cols), _pi_edges_df.select(_pi_cols)])
    _pi_all.write_parquet(_pi_out)
    _pi_sha = _pi_hashlib.sha256(_pi_out.read_bytes()).hexdigest()[:16]
    _pi_tot = sum(r[3] for r in _pi_summary)
    print(f"pre-ILP roll-up: wrote {_pi_out} rows={_pi_all.height:,} sha256={_pi_sha}")
    print(f"pre-ILP roll-up: TOTAL sources with candidate out-degree >= 2 = {_pi_tot:,}")
    print("  (post-ILP out-degree is 1 everywhere, so each is a division the solver DECLINED)")
