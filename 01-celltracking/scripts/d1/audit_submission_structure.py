"""Structural audit and cross-arm graph diff for Kaggle submission.csv files.

Audit gates (all must PASS before an artifact is considered submission-ready):

  A1  schema           exact columns; contiguous ids; finite integral numeric fields; node/edge
                       sentinel contract; non-empty dataset; row_type in {node, edge}
  A2  node_time        every node has 0 <= t <= T_MAX and t is integral
  A3  volume           every node lies inside the plausible acquisition volume
  A4  node_ids         node_id unique within a dataset; no node row is also an edge row
  A5  edge_endpoints   every edge endpoint resolves to a node of the same dataset
  A6  cross_dataset    no edge references a node id that only exists in another dataset
  A7  consecutive      t(target) == t(source) + 1 for every edge
  A8  in_degree        max in-degree <= 1
  A9  out_degree       max out-degree <= 2
  A10 datasets         the emitted dataset set equals the expected test set

Optional baseline plausibility gates (enabled only by ``--baseline``):

  B0  baseline         the explicitly named baseline itself passes A1-A10
  B1  total_retention  candidate node and edge counts retain the configured baseline fractions
  B2  dataset_coverage every baseline dataset independently retains the configured fractions

Volume bounds come from the competition test zarrs: T=100, Z=64, Y=256, X=256,
so valid index ranges are t in [0, 99], z in [0, 63], y in [0, 255], x in [0, 255].

Usage:
  python scripts/d1/audit_submission_structure.py audit  <submission.csv> [more.csv ...]
      [--baseline BASE.csv] [--min-node-retention 0.5] [--min-edge-retention 0.5]
  python scripts/d1/audit_submission_structure.py diff   <base.csv> <arm.csv>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
    "source_id", "target_id",
]
T_MAX = 99
Z_MAX = 63
Y_MAX = 255
X_MAX = 255
EXPECTED_DATASETS = [
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
]
NUMERIC_COLUMNS = [
    "id", "node_id", "t", "z", "y", "x", "source_id", "target_id",
]
DEFAULT_MIN_RETENTION = 0.5


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _numeric_contract(df: pd.DataFrame, float_coords=frozenset()) -> tuple[pd.DataFrame, int, int]:
    """Return a safe numeric view plus counts of non-finite and fractional cells.

    ``astype(int)`` silently truncates fractional values, which previously let malformed
    coordinates and times pass the topology checks.  Coercing once here lets the audit record
    the defect while replacing unsafe cells with a sentinel so later checks still produce a
    complete FAIL report instead of crashing.
    """
    numeric = pd.DataFrame(index=df.index)
    nonfinite = 0
    fractional = 0
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce")
        array = values.to_numpy(dtype=np.float64, na_value=np.nan)
        finite = np.isfinite(array)
        nonfinite += int((~finite).sum())
        if column in float_coords:
            numeric[column] = np.where(finite, array, -1.0).astype(np.float64)
            continue
        integral = finite & (array == np.trunc(array))
        fractional += int((finite & ~integral).sum())
        numeric[column] = np.where(integral, array, -1).astype(np.int64)
    return numeric, nonfinite, fractional


def _validated_fraction(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return value


def _retention(candidate: int, baseline: int) -> float:
    """Fraction retained; an empty baseline has no population that can be lost."""
    return float(candidate / baseline) if baseline else 1.0


def _apply_baseline_plausibility(
    report: dict,
    baseline_path: Path,
    *,
    min_node_retention: float,
    min_edge_retention: float,
    min_dataset_node_retention: float,
    min_dataset_edge_retention: float,
) -> None:
    """Append generic count-retention gates against an explicitly selected baseline."""
    thresholds = {
        "min_node_retention": _validated_fraction(
            "min_node_retention", min_node_retention
        ),
        "min_edge_retention": _validated_fraction(
            "min_edge_retention", min_edge_retention
        ),
        "min_dataset_node_retention": _validated_fraction(
            "min_dataset_node_retention", min_dataset_node_retention
        ),
        "min_dataset_edge_retention": _validated_fraction(
            "min_dataset_edge_retention", min_dataset_edge_retention
        ),
    }
    baseline = audit(Path(baseline_path))
    baseline_ok = baseline["verdict"] == "PASS"
    report["checks"].append({
        "check": "B0 baseline",
        "pass": baseline_ok,
        "detail": (
            f"path={baseline_path}; sha256={baseline['sha256']}; "
            f"structural_verdict={baseline['verdict']}"
        ),
    })

    node_retention = _retention(report["nodes"], baseline["nodes"])
    edge_retention = _retention(report["edges"], baseline["edges"])
    totals_ok = (
        node_retention >= thresholds["min_node_retention"]
        and edge_retention >= thresholds["min_edge_retention"]
    )
    report["checks"].append({
        "check": "B1 total_retention",
        "pass": totals_ok,
        "detail": (
            f"nodes={report['nodes']}/{baseline['nodes']}={node_retention:.6f} "
            f"(min={thresholds['min_node_retention']:.6f}); "
            f"edges={report['edges']}/{baseline['edges']}={edge_retention:.6f} "
            f"(min={thresholds['min_edge_retention']:.6f})"
        ),
    })

    per_dataset = {}
    failed_datasets = []
    for dataset, base_counts in baseline["per_dataset"].items():
        candidate_counts = report["per_dataset"].get(dataset, {"nodes": 0, "edges": 0})
        ds_node_retention = _retention(candidate_counts["nodes"], base_counts["nodes"])
        ds_edge_retention = _retention(candidate_counts["edges"], base_counts["edges"])
        passed = (
            ds_node_retention >= thresholds["min_dataset_node_retention"]
            and ds_edge_retention >= thresholds["min_dataset_edge_retention"]
        )
        if not passed:
            failed_datasets.append(dataset)
        per_dataset[dataset] = {
            "baseline_nodes": base_counts["nodes"],
            "candidate_nodes": candidate_counts["nodes"],
            "node_retention": ds_node_retention,
            "baseline_edges": base_counts["edges"],
            "candidate_edges": candidate_counts["edges"],
            "edge_retention": ds_edge_retention,
            "pass": passed,
        }
    report["checks"].append({
        "check": "B2 dataset_coverage",
        "pass": not failed_datasets,
        "detail": (
            f"datasets_checked={len(per_dataset)}; failed={failed_datasets}; "
            f"min_nodes={thresholds['min_dataset_node_retention']:.6f}; "
            f"min_edges={thresholds['min_dataset_edge_retention']:.6f}"
        ),
    })
    report["baseline_plausibility"] = {
        "baseline_path": str(baseline_path),
        "baseline_sha256": baseline["sha256"],
        "thresholds": thresholds,
        "node_retention": node_retention,
        "edge_retention": edge_retention,
        "per_dataset": per_dataset,
    }
    report["verdict"] = (
        "PASS" if all(check["pass"] for check in report["checks"]) else "FAIL"
    )


def audit(
    path: Path,
    *,
    baseline: Path | None = None,
    min_node_retention: float = DEFAULT_MIN_RETENTION,
    min_edge_retention: float = DEFAULT_MIN_RETENTION,
    min_dataset_node_retention: float = DEFAULT_MIN_RETENTION,
    min_dataset_edge_retention: float = DEFAULT_MIN_RETENTION,
    allow_float_coords: bool = False,
) -> dict:
    raw = path.read_bytes()
    df = load(path)
    checks: list[tuple[str, bool, str]] = []

    ok = df.columns.tolist() == COLUMNS
    detail = "columns match" if ok else f"columns={df.columns.tolist()}"
    if not ok:
        raise ValueError(f"submission schema mismatch: {detail}")

    float_coords = frozenset({"z", "y", "x"}) if allow_float_coords else frozenset()
    numeric, nonfinite, fractional = _numeric_contract(df, float_coords)
    # All later graph checks operate on this validated/sanitised integer view. Invalid cells
    # remain represented by -1, guaranteeing a FAIL without unsafe float-to-int truncation.
    work = df.copy()
    work[NUMERIC_COLUMNS] = numeric

    contiguous = df["id"].tolist() == list(range(len(df)))
    rowtypes = set(df["row_type"].dropna().unique())
    rowtypes_valid = bool(df["row_type"].notna().all() and rowtypes <= {"node", "edge"})
    ok = ok and contiguous and rowtypes_valid
    checks.append((
        "A1 schema", ok,
        f"{detail}; contiguous_ids={contiguous}; row_types={sorted(map(str, rowtypes))}; "
        f"missing_row_type={int(df['row_type'].isna().sum())}",
    ))

    checks.append((
        "A1 numeric", nonfinite == 0 and fractional == 0,
        f"non_numeric_or_nonfinite={nonfinite}; fractional={fractional}",
    ))

    dataset_missing = int(df["dataset"].isna().sum())
    dataset_blank = int((
        df["dataset"].notna() & df["dataset"].astype(str).str.strip().eq("")
    ).sum())
    node_mask = df["row_type"].eq("node")
    edge_mask = df["row_type"].eq("edge")
    unknown_rows = int((~(node_mask | edge_mask)).sum())
    node_contract_bad = int((
        (numeric.loc[node_mask, "node_id"] < 0)
        | (numeric.loc[node_mask, ["t", "z", "y", "x"]] < 0).any(axis=1)
        | numeric.loc[node_mask, "source_id"].ne(-1)
        | numeric.loc[node_mask, "target_id"].ne(-1)
    ).sum())
    edge_contract_bad = int((
        numeric.loc[edge_mask, ["node_id", "t", "z", "y", "x"]].ne(-1).any(axis=1)
        | (numeric.loc[edge_mask, ["source_id", "target_id"]] < 0).any(axis=1)
    ).sum())
    checks.append((
        "A1 row_contract",
        dataset_missing == 0 and dataset_blank == 0 and unknown_rows == 0
        and node_contract_bad == 0 and edge_contract_bad == 0,
        f"missing_dataset={dataset_missing}; blank_dataset={dataset_blank}; "
        f"unknown_row_type={unknown_rows}; bad_node_rows={node_contract_bad}; "
        f"bad_edge_rows={edge_contract_bad}",
    ))

    nodes = work[node_mask]
    edges = work[edge_mask]

    t_bad = int(((nodes["t"] < 0) | (nodes["t"] > T_MAX)).sum())
    checks.append((
        "A2 node_time", t_bad == 0,
        f"t range [{int(nodes['t'].min())}, {int(nodes['t'].max())}]; out_of_range={t_bad}",
    ))

    vol_bad = int((
        (nodes["z"] < 0) | (nodes["z"] > Z_MAX)
        | (nodes["y"] < 0) | (nodes["y"] > Y_MAX)
        | (nodes["x"] < 0) | (nodes["x"] > X_MAX)
    ).sum())
    checks.append((
        "A3 volume", vol_bad == 0,
        f"z [{float(nodes['z'].min()):.3g}, {float(nodes['z'].max()):.3g}] "
        f"y [{float(nodes['y'].min()):.3g}, {float(nodes['y'].max()):.3g}] "
        f"x [{float(nodes['x'].min()):.3g}, {float(nodes['x'].max()):.3g}]; outside={vol_bad}",
    ))

    dup = int(nodes.duplicated(["dataset", "node_id"]).sum())
    checks.append(("A4 node_ids", dup == 0, f"duplicate (dataset,node_id) rows={dup}"))

    # node index: (dataset, node_id) -> t ; and a global id -> datasets map for A6
    node_t: dict[tuple[str, int], int] = {}
    id_owners: dict[int, set[str]] = defaultdict(set)
    for ds, nid, t in zip(
        nodes["dataset"].astype(str), nodes["node_id"].astype("int64"), nodes["t"].astype("int64")
    ):
        node_t[(ds, int(nid))] = int(t)
        id_owners[int(nid)].add(ds)

    missing = 0
    cross = 0
    nonconsec = 0
    indeg: Counter = Counter()
    outdeg: Counter = Counter()
    for ds, s, tg in zip(
        edges["dataset"].astype(str),
        edges["source_id"].astype("int64"),
        edges["target_id"].astype("int64"),
    ):
        ds = str(ds)
        ks, kt = (ds, int(s)), (ds, int(tg))
        s_ok, t_ok = ks in node_t, kt in node_t
        if not (s_ok and t_ok):
            missing += 1
            # only counts as cross-dataset if the id exists under a different movie
            if (not s_ok and id_owners.get(int(s))) or (not t_ok and id_owners.get(int(tg))):
                cross += 1
            continue
        if node_t[kt] != node_t[ks] + 1:
            nonconsec += 1
        indeg[kt] += 1
        outdeg[ks] += 1

    checks.append((
        "A5 edge_endpoints", missing == 0, f"edges with an unresolved endpoint={missing}",
    ))
    checks.append((
        "A6 cross_dataset", cross == 0, f"edges resolving into another movie={cross}",
    ))
    checks.append((
        "A7 consecutive", nonconsec == 0, f"edges not spanning t -> t+1={nonconsec}",
    ))
    max_in = max(indeg.values(), default=0)
    max_out = max(outdeg.values(), default=0)
    checks.append(("A8 in_degree", max_in <= 1, f"max in-degree={max_in}"))
    checks.append(("A9 out_degree", max_out <= 2, f"max out-degree={max_out}"))

    got_ds = sorted({str(value) for value in df["dataset"].dropna().unique()})
    checks.append((
        "A10 datasets", got_ds == EXPECTED_DATASETS, f"datasets={got_ds}",
    ))

    per_ds = {}
    for ds, grp in work.groupby("dataset", sort=True):
        n = grp[grp["row_type"].eq("node")]
        e = grp[grp["row_type"].eq("edge")]
        d_out = Counter()
        for s in e["source_id"].astype("int64"):
            d_out[int(s)] += 1
        per_ds[str(ds)] = {
            "nodes": int(len(n)),
            "edges": int(len(e)),
            "divisions": int(sum(v == 2 for v in d_out.values())),
            "t_min": int(n["t"].min()),
            "t_max": int(n["t"].max()),
        }

    report = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "rows": int(len(df)),
        "nodes": int(len(nodes)),
        "edges": int(len(edges)),
        "n_datasets": int(df["dataset"].nunique()),
        "datasets": got_ds,
        "divisions": int(sum(v == 2 for v in outdeg.values())),
        "max_indegree": int(max_in),
        "max_outdegree": int(max_out),
        "t_min": int(nodes["t"].min()),
        "t_max": int(nodes["t"].max()),
        "z_min": int(nodes["z"].min()), "z_max": int(nodes["z"].max()),
        "y_min": int(nodes["y"].min()), "y_max": int(nodes["y"].max()),
        "x_min": int(nodes["x"].min()), "x_max": int(nodes["x"].max()),
        "per_dataset": per_ds,
        "checks": [{"check": c, "pass": bool(p), "detail": d} for c, p, d in checks],
        "verdict": "PASS" if all(p for _, p, _ in checks) else "FAIL",
    }
    if baseline is not None:
        _apply_baseline_plausibility(
            report,
            baseline,
            min_node_retention=min_node_retention,
            min_edge_retention=min_edge_retention,
            min_dataset_node_retention=min_dataset_node_retention,
            min_dataset_edge_retention=min_dataset_edge_retention,
        )
    return report


def _graph(path: Path):
    """Return coordinate-keyed node set, parent map, and out-degree map."""
    df = load(path)
    nodes = df[df["row_type"].eq("node")]
    edges = df[df["row_type"].eq("edge")]
    key: dict[tuple[str, int], tuple] = {}
    node_keys: set[tuple] = set()
    for ds, nid, t, z, y, x in zip(
        nodes["dataset"].astype(str), nodes["node_id"].astype("int64"),
        nodes["t"].astype("int64"), nodes["z"].astype("int64"),
        nodes["y"].astype("int64"), nodes["x"].astype("int64"),
    ):
        k = (str(ds), int(t), int(z), int(y), int(x))
        key[(str(ds), int(nid))] = k
        node_keys.add(k)
    parent: dict[tuple, tuple] = {}
    edge_keys: set[tuple[tuple, tuple]] = set()
    out = Counter()
    for ds, s, tg in zip(
        edges["dataset"].astype(str), edges["source_id"].astype("int64"),
        edges["target_id"].astype("int64"),
    ):
        ks = key.get((str(ds), int(s)))
        kt = key.get((str(ds), int(tg)))
        if ks is None or kt is None:
            continue
        parent[kt] = ks
        edge_keys.add((ks, kt))
        out[ks] += 1
    divisions = {k for k, v in out.items() if v == 2}
    return node_keys, edge_keys, parent, divisions


def diff(base: Path, arm: Path) -> dict:
    bn, be, bp, bd = _graph(base)
    an, ae, ap, ad = _graph(arm)
    shared_targets = set(bp) & set(ap) & bn & an
    parent_changed = sum(1 for k in shared_targets if bp[k] != ap[k])
    return {
        "base": str(base),
        "arm": str(arm),
        "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "arm_sha256": hashlib.sha256(arm.read_bytes()).hexdigest(),
        "identity": "exact (dataset, t, z, y, x) voxel key",
        "nodes_base": len(bn),
        "nodes_arm": len(an),
        "nodes_added": len(an - bn),
        "nodes_removed": len(bn - an),
        "nodes_net": len(an) - len(bn),
        "edges_base": len(be),
        "edges_arm": len(ae),
        "edges_added": len(ae - be),
        "edges_removed": len(be - ae),
        "edges_net": len(ae) - len(be),
        "divisions_base": len(bd),
        "divisions_arm": len(ad),
        "divisions_added": len(ad - bd),
        "divisions_removed": len(bd - ad),
        "divisions_net": len(ad) - len(bd),
        "targets_in_both_with_parent": len(shared_targets),
        "edges_parent_reassigned": parent_changed,
    }


def print_audit(rep: dict) -> None:
    print(f"\n=== STRUCTURAL AUDIT: {rep['path']} ===")
    print(f"sha256 {rep['sha256']}  bytes {rep['bytes']}")
    print(
        f"rows {rep['rows']} = nodes {rep['nodes']} + edges {rep['edges']} | "
        f"datasets {rep['n_datasets']} | divisions {rep['divisions']}"
    )
    print(
        f"t [{rep['t_min']},{rep['t_max']}] z [{rep['z_min']},{rep['z_max']}] "
        f"y [{rep['y_min']},{rep['y_max']}] x [{rep['x_min']},{rep['x_max']}]"
    )
    print(f"{'check':<20} {'result':<6} detail")
    for c in rep["checks"]:
        print(f"{c['check']:<20} {'PASS' if c['pass'] else 'FAIL':<6} {c['detail']}")
    print(f"VERDICT: {rep['verdict']}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit")
    a.add_argument("paths", nargs="+", type=Path)
    a.add_argument("--json-out", type=Path)
    a.add_argument(
        "--baseline", type=Path,
        help="explicit baseline submission used for optional count-retention gates",
    )
    a.add_argument("--min-node-retention", type=float, default=DEFAULT_MIN_RETENTION)
    a.add_argument("--min-edge-retention", type=float, default=DEFAULT_MIN_RETENTION)
    a.add_argument(
        "--min-dataset-node-retention", type=float, default=DEFAULT_MIN_RETENTION,
    )
    a.add_argument(
        "--min-dataset-edge-retention", type=float, default=DEFAULT_MIN_RETENTION,
    )
    a.add_argument("--allow-float-coords", action="store_true")
    d = sub.add_parser("diff")
    d.add_argument("base", type=Path)
    d.add_argument("arm", type=Path)
    d.add_argument("--json-out", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "audit":
        reports = [audit(
            p,
            baseline=args.baseline,
            min_node_retention=args.min_node_retention,
            min_edge_retention=args.min_edge_retention,
            min_dataset_node_retention=args.min_dataset_node_retention,
            min_dataset_edge_retention=args.min_dataset_edge_retention,
            allow_float_coords=args.allow_float_coords,
        ) for p in args.paths]
        for rep in reports:
            print_audit(rep)
        if args.json_out:
            args.json_out.write_text(json.dumps(reports, indent=2) + "\n")
        return 0 if all(r["verdict"] == "PASS" for r in reports) else 1

    rep = diff(args.base, args.arm)
    print(json.dumps(rep, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(rep, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
