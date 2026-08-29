r"""Fail-closed audit of the P30 acquisition, before either replay is allowed to mean anything.

PKT-0027 states three falsifiers, and the host added a fourth check that counts and heartbeats
cannot catch. All four run here, and every one of them is fail-closed.

  A  CONTROL PARITY. P30 must reproduce the P28 fold-0 control graph. If the acquisition changed
     what it observes, the control is destroyed and neither lever can be replayed against it.
  B  PATCH HEARTBEAT. `ECB_PATCH_APPLIED` must appear, one `ECB_EXPORT` line per crop, and the
     fail-closed assertion must not have fired. A silent no-op looks exactly like "the richer
     candidates do not exist", which is the wrong conclusion to reach for free.
  C  DEPLOYED RECONSTRUCTION. Filtering the exported surface at the deployed threshold must
     recover the candidate set the pipeline actually consumed. If it does not, the export is not
     a superset of what ran and no reachability claim from it is safe.
  D  NODE-ID STABILITY (host, 2026-08-29). Every exported alternative edge must reference the
     same node ids the pre-ILP graph consumes. This is the check counts and heartbeats pass
     while an id-remapping boundary silently corrupts the richer surface.

WHY D IS A REAL RISK AND NOT A FORMALITY
----------------------------------------
The sidecar records `idx_src`/`idx_tgt`, which are positional indices into `coords_so_far`.
The pre-ILP graph is built by `build_graph`, which calls `bulk_add_nodes` and then addresses
edges through the ids THAT returns - a remap. The two coincide only if `bulk_add_nodes` hands
back `range(N)` in coords order. Measured on the fold-1 substrate that identity holds exactly:
pre-ILP node k is row-for-row the k-th surviving detector peak. But it holds as an observed
property of one tracksdata version on one export, not as a guarantee, so it is re-proven here
against P30's own artifacts rather than assumed.

ON CHECK C AND THE DEGREE CAPS
------------------------------
Equality is NOT the right contract. The deployed path filters at `cfg.threshold` and then applies
`max_children_per_node` / `max_parents_per_node` greedily, so the graph's edges are a SUBSET of
the thresholded candidate list. The audit therefore requires containment - every pre-ILP edge
must appear in the exported surface above the deployed threshold - and reports the residual,
which should be attributable to the children cap. A missing pre-ILP edge is a hard failure; an
extra exported candidate is the entire point of the acquisition.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import polars as pl

DEPLOYED_MAX_CHILDREN = 2
DEPLOYED_MAX_PARENTS = 1


def check_control_parity(p30_csv: Path, control_csv: Path) -> dict:
    a = pl.read_csv(p30_csv).drop("id", strict=False)
    b = pl.read_csv(control_csv).drop("id", strict=False)
    same_shape = a.shape == b.shape
    identical = same_shape and a.equals(b)
    out = {
        "p30_rows": a.height,
        "control_rows": b.height,
        "same_shape": same_shape,
        "graph_identical": bool(identical),
        "passed": bool(identical),
    }
    if same_shape and not identical:
        for col in a.columns:
            if not a[col].equals(b[col]):
                out.setdefault("differing_columns", []).append(col)
    return out


def check_heartbeat(log_text: str, expected_crops: int) -> dict:
    applied = "ECB_PATCH_APPLIED" in log_text
    exports = re.findall(r"ECB_EXPORT crop=(\S+) .*?exported_candidates=(\d+)", log_text)
    crops = {c for c, _ in exports}
    empty = [c for c, n in exports if int(n) == 0]
    treatment_unset = "treatment_threshold=unset" in log_text and "treatment_topk=unset" in log_text
    return {
        "patch_applied_heartbeat": applied,
        "export_heartbeats": len(exports),
        "distinct_crops": len(crops),
        "expected_crops": expected_crops,
        "crops_with_zero_exported": empty,
        "treatment_variables_unset": treatment_unset,
        "assertion_fired": "primary path was NOT patched" in log_text,
        "passed": bool(
            applied
            and treatment_unset
            and len(crops) == expected_crops
            and not empty
            and "primary path was NOT patched" not in log_text
        ),
    }


def load_sidecar(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {
            "source_id": z["source_id"].astype(np.int64),
            "target_id": z["target_id"].astype(np.int64),
            "edge_prob": z["edge_prob"].astype(np.float64),
            "deployed_threshold": float(z["deployed_threshold"]),
            "export_threshold": float(z["export_threshold"]),
            "export_topk": int(z["export_topk"]),
            "deployed_candidate_count": int(z["deployed_candidate_count"]),
        }


def check_crop(crop: str, side: dict, pre: pl.DataFrame) -> dict:
    nodes = pre.filter(pl.col("row_type") == "node")
    edges = pre.filter(pl.col("row_type") == "edge")
    node_ids = nodes["node_id"].to_numpy().astype(np.int64)
    n_nodes = len(node_ids)

    # --- D: node-id stability -------------------------------------------------------
    ids_are_positional = bool(np.array_equal(np.sort(node_ids), np.arange(n_nodes)))
    src, tgt = side["source_id"], side["target_id"]
    in_range = bool(
        (src >= 0).all() and (src < n_nodes).all()
        and (tgt >= 0).all() and (tgt < n_nodes).all()
    )
    known = set(node_ids.tolist())
    unknown = int(sum(1 for v in np.concatenate([src, tgt]).tolist() if v not in known))

    # Frame adjacency: every candidate must join consecutive frames under the SAME ids.
    t_by_id = dict(zip(node_ids.tolist(), nodes["t"].to_numpy().astype(np.int64).tolist()))
    bad_frame = 0
    if in_range and unknown == 0:
        ts, tt = np.array([t_by_id[v] for v in src.tolist()]), np.array([t_by_id[v] for v in tgt.tolist()])
        bad_frame = int((tt - ts != 1).sum())

    # --- C: deployed reconstruction --------------------------------------------------
    deployed_pairs = set(
        zip(edges["source_id"].to_numpy().astype(np.int64).tolist(),
            edges["target_id"].to_numpy().astype(np.int64).tolist())
    )
    above = side["edge_prob"] > side["deployed_threshold"]
    exported_above = set(zip(src[above].tolist(), tgt[above].tolist()))
    missing = deployed_pairs - exported_above
    extra = exported_above - deployed_pairs

    # Extras above the deployed threshold should be edges the DEGREE CAPS dropped, not
    # evidence of a broken export. Count how many are explained by the children cap.
    children = {}
    for s, _t in sorted(deployed_pairs):
        children[s] = children.get(s, 0) + 1
    extra_at_capped_source = sum(
        1 for s, _t in extra if children.get(s, 0) >= DEPLOYED_MAX_CHILDREN
    )
    return {
        "crop": crop,
        "pre_nodes": n_nodes,
        "pre_edges": len(deployed_pairs),
        "exported_edges": int(len(src)),
        "exported_above_deployed_threshold": int(len(exported_above)),
        "ids_are_positional": ids_are_positional,
        "ids_in_range": in_range,
        "unknown_ids": unknown,
        "edges_not_frame_adjacent": bad_frame,
        "deployed_edges_missing_from_export": len(missing),
        "extra_above_threshold": len(extra),
        "extra_explained_by_children_cap": extra_at_capped_source,
        "passed": bool(
            ids_are_positional and in_range and unknown == 0
            and bad_frame == 0 and not missing
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--p30-csv", type=Path, required=True)
    ap.add_argument("--control-csv", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--preilp", type=Path, required=True)
    ap.add_argument("--ecb-dir", type=Path, required=True)
    ap.add_argument("--max-crops", type=int)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pre = pl.read_parquet(args.preilp)
    crops = sorted(pre["dataset"].unique().to_list())
    if args.max_crops:
        crops = crops[: args.max_crops]

    parity = check_control_parity(args.p30_csv, args.control_csv)
    heartbeat = check_heartbeat(args.log.read_text(encoding="utf-8", errors="replace"), len(crops))

    rows = []
    for i, crop in enumerate(crops, 1):
        side_path = args.ecb_dir / f"{crop}.npz"
        if not side_path.exists():
            rows.append({"crop": crop, "passed": False, "error": "sidecar missing"})
            continue
        rows.append(check_crop(crop, load_sidecar(side_path), pre.filter(pl.col("dataset") == crop)))
        print(f"  [{i}/{len(crops)}] {crop} passed={rows[-1]['passed']}", flush=True)

    failing = [r for r in rows if not r.get("passed")]
    result = {
        "schema_version": 1,
        "A_control_parity": parity,
        "B_patch_heartbeat": heartbeat,
        "CD_per_crop": rows,
        "CD_crops_failing": [r["crop"] for r in failing],
        "totals": {
            "crops": len(rows),
            "pre_edges": int(sum(r.get("pre_edges", 0) for r in rows)),
            "exported_edges": int(sum(r.get("exported_edges", 0) for r in rows)),
            "deployed_edges_missing_from_export": int(
                sum(r.get("deployed_edges_missing_from_export", 0) for r in rows)
            ),
            "edges_not_frame_adjacent": int(sum(r.get("edges_not_frame_adjacent", 0) for r in rows)),
            "unknown_ids": int(sum(r.get("unknown_ids", 0) for r in rows)),
        },
        "passed": bool(parity["passed"] and heartbeat["passed"] and not failing),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")

    t = result["totals"]
    print(
        f"\nP30_ACQUISITION_AUDIT passed={result['passed']}\n"
        f"  A control parity            : {parity['passed']} "
        f"(p30 {parity['p30_rows']:,} rows vs control {parity['control_rows']:,})\n"
        f"  B patch heartbeat           : {heartbeat['passed']} "
        f"(applied={heartbeat['patch_applied_heartbeat']}, "
        f"crops={heartbeat['distinct_crops']}/{heartbeat['expected_crops']}, "
        f"treatment_unset={heartbeat['treatment_variables_unset']})\n"
        f"  C deployed reconstruction   : missing={t['deployed_edges_missing_from_export']} "
        f"(must be 0)\n"
        f"  D node-id stability         : unknown_ids={t['unknown_ids']} "
        f"non_adjacent={t['edges_not_frame_adjacent']} (both must be 0)\n"
        f"  exported {t['exported_edges']:,} candidates vs {t['pre_edges']:,} deployed edges "
        f"({t['exported_edges'] / max(t['pre_edges'], 1):.2f}x)"
    )
    if not result["passed"]:
        raise SystemExit("P30 ACQUISITION AUDIT FAILED - no replay may be run from these artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
