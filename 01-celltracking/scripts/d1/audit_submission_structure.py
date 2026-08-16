"""Structural audit and cross-arm graph diff for Kaggle submission.csv files.

Audit gates (all must PASS before an artifact is considered submission-ready):

  A1  schema           exactly the 10 required columns, contiguous ids, row_type in {node, edge}
  A2  node_time        every node has 0 <= t <= T_MAX and t is integral
  A3  volume           every node lies inside the plausible acquisition volume
  A4  node_ids         node_id unique within a dataset; no node row is also an edge row
  A5  edge_endpoints   every edge endpoint resolves to a node of the same dataset
  A6  cross_dataset    no edge references a node id that only exists in another dataset
  A7  consecutive      t(target) == t(source) + 1 for every edge
  A8  in_degree        max in-degree <= 1
  A9  out_degree       max out-degree <= 2
  A10 datasets         the emitted dataset set equals the expected test set

Volume bounds come from the competition test zarrs: T=100, Z=64, Y=256, X=256,
so valid index ranges are t in [0, 99], z in [0, 63], y in [0, 255], x in [0, 255].

Usage:
  python scripts/d1/audit_submission_structure.py audit  <submission.csv> [more.csv ...]
  python scripts/d1/audit_submission_structure.py diff   <base.csv> <arm.csv>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

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


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def audit(path: Path) -> dict:
    raw = path.read_bytes()
    df = load(path)
    checks: list[tuple[str, bool, str]] = []

    ok = df.columns.tolist() == COLUMNS
    detail = "columns match" if ok else f"columns={df.columns.tolist()}"
    contiguous = df["id"].tolist() == list(range(len(df)))
    rowtypes = set(df["row_type"].unique())
    ok = ok and contiguous and rowtypes <= {"node", "edge"}
    checks.append((
        "A1 schema", ok,
        f"{detail}; contiguous_ids={contiguous}; row_types={sorted(rowtypes)}",
    ))

    nodes = df[df["row_type"].eq("node")]
    edges = df[df["row_type"].eq("edge")]

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
        f"z [{int(nodes['z'].min())}, {int(nodes['z'].max())}] "
        f"y [{int(nodes['y'].min())}, {int(nodes['y'].max())}] "
        f"x [{int(nodes['x'].min())}, {int(nodes['x'].max())}]; outside={vol_bad}",
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

    got_ds = sorted(df["dataset"].astype(str).unique())
    checks.append((
        "A10 datasets", got_ds == EXPECTED_DATASETS, f"datasets={got_ds}",
    ))

    per_ds = {}
    for ds, grp in df.groupby("dataset", sort=True):
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

    return {
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
    d = sub.add_parser("diff")
    d.add_argument("base", type=Path)
    d.add_argument("arm", type=Path)
    d.add_argument("--json-out", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "audit":
        reports = [audit(p) for p in args.paths]
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
