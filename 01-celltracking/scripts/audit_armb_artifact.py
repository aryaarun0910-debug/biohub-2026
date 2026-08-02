"""Independent audit of the arm-B artifact + structural delta vs the P0-B baseline.

Degrees keyed on (dataset, node_id) per trap 24 -- node_id is NOT unique across movies.
"""
import hashlib
import json
import pathlib
from collections import Counter, defaultdict

import polars as pl

REPO = pathlib.Path(r"c:\Users\aryaa\Documents\Biohub-CellTracking-2026")
PATHS = {
    "baseline": REPO / "notebooks/kaggle_p2_armb_baseline/_out/submission.csv",
    "armB": REPO / "notebooks/kaggle_p2_armb_flowgate/_out/submission.csv",
}
VOXEL = (1.625, 0.40625, 0.40625)


def load(p):
    return pl.read_csv(p)


def audit(tag, df):
    nodes = df.filter(pl.col("row_type") == "node")
    edges = df.filter(pl.col("row_type") == "edge")
    out = {}
    out["rows"] = len(df)
    out["nodes"] = len(nodes)
    out["edges"] = len(edges)
    out["datasets"] = sorted(df["dataset"].unique().to_list())

    key_t = {}
    for d, n, t in zip(nodes["dataset"], nodes["node_id"], nodes["t"]):
        key_t[(d, int(n))] = int(t)

    outd, ind = Counter(), Counter()
    nonconsec = 0
    missing = 0
    for d, s, tg in zip(edges["dataset"], edges["source_id"], edges["target_id"]):
        ks, kt = (d, int(s)), (d, int(tg))
        if ks not in key_t or kt not in key_t:
            missing += 1
            continue
        if key_t[kt] != key_t[ks] + 1:
            nonconsec += 1
        outd[ks] += 1
        ind[kt] += 1
    out["max_out_degree"] = max(outd.values(), default=0)
    out["max_in_degree"] = max(ind.values(), default=0)
    out["out_degree_3_plus"] = sum(1 for v in outd.values() if v > 2)
    out["in_degree_2_plus"] = sum(1 for v in ind.values() if v > 1)
    out["division_parents"] = sum(1 for v in outd.values() if v == 2)
    out["nonconsecutive_edges"] = nonconsec
    out["dangling_edges"] = missing
    out["dup_nodes"] = len(nodes) - nodes.select(["dataset", "node_id"]).n_unique()
    out["dup_edges"] = len(edges) - edges.select(["dataset", "source_id", "target_id"]).n_unique()
    out["neg_t"] = int((nodes["t"] < 0).sum())
    out["neg_coord"] = int(((nodes["z"] < 0) | (nodes["y"] < 0) | (nodes["x"] < 0)).sum())
    out["_outd"] = outd
    out["_keyt"] = key_t
    out["_edgeset"] = {
        (d, int(s), int(tg))
        for d, s, tg in zip(edges["dataset"], edges["source_id"], edges["target_id"])
    }
    out["_nodes"] = nodes
    return out


res = {}
for tag, p in PATHS.items():
    df = load(p)
    a = audit(tag, df)
    a["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    a["bytes"] = p.stat().st_size
    res[tag] = a

print("=" * 78)
print("ARTIFACT AUDIT")
print("=" * 78)
fields = ["sha256", "bytes", "rows", "nodes", "edges", "max_out_degree", "max_in_degree",
          "out_degree_3_plus", "in_degree_2_plus", "division_parents",
          "nonconsecutive_edges", "dangling_edges", "dup_nodes", "dup_edges",
          "neg_t", "neg_coord"]
print(f"{'field':22s} {'baseline':>34s} {'armB':>34s}")
for f in fields:
    b, a = res["baseline"][f], res["armB"][f]
    mark = "" if b == a else "   <-- differs"
    print(f"{f:22s} {str(b):>34s} {str(a):>34s}{mark}")

print(f"\ndatasets: {res['armB']['datasets']}")

print("\n" + "=" * 78)
print("STRUCTURAL DELTA  baseline (P0-B) -> arm B")
print("=" * 78)
eb, ea = res["baseline"]["_edgeset"], res["armB"]["_edgeset"]
print(f"  edges only in P0-B : {len(eb - ea)}")
print(f"  edges only in armB : {len(ea - eb)}")
print(f"  shared edges       : {len(eb & ea)}")
print(f"  churn vs P0-B      : {100.0*len(eb ^ ea)/max(len(eb),1):.3f}%")

# edge length distribution (the property PRIMITIVE_MATRIX flagged to watch)
print("\n" + "=" * 78)
print("EDGE LENGTH — arm B raises the admissible max (gate is on the residual)")
print("=" * 78)
for tag in ("baseline", "armB"):
    nodes = res[tag]["_nodes"]
    pos = {}
    for d, n, z, y, x in zip(nodes["dataset"], nodes["node_id"], nodes["z"], nodes["y"], nodes["x"]):
        pos[(d, int(n))] = (float(z) * VOXEL[0], float(y) * VOXEL[1], float(x) * VOXEL[2])
    lens = []
    for (d, s, t) in res[tag]["_edgeset"]:
        a1, b1 = pos.get((d, s)), pos.get((d, t))
        if a1 and b1:
            lens.append(sum((p - q) ** 2 for p, q in zip(a1, b1)) ** 0.5)
    lens.sort()
    n = len(lens)
    print(f"  {tag:9s} n={n}  max={lens[-1]:.3f}um  p99={lens[int(0.99*n)]:.3f}  "
          f"p50={lens[n//2]:.3f}  >6um={sum(l>6 for l in lens)}  "
          f">10um={sum(l>10 for l in lens)}  >14um={sum(l>14 for l in lens)}")

print("\n" + "=" * 78)
print("PER-DATASET DELTA")
print("=" * 78)
by_ds = defaultdict(lambda: [0, 0, 0])
for e in eb - ea:
    by_ds[e[0]][0] += 1
for e in ea - eb:
    by_ds[e[0]][1] += 1
for e in eb & ea:
    by_ds[e[0]][2] += 1
print(f"  {'dataset':22s} {'-P0B':>8s} {'+armB':>8s} {'shared':>9s} {'churn%':>8s}")
for d in sorted(by_ds):
    lost, gained, shared = by_ds[d]
    tot = lost + shared
    print(f"  {d:22s} {lost:8d} {gained:8d} {shared:9d} {100.0*(lost+gained)/max(tot,1):8.3f}")

summary = {k: {f: v[f] for f in fields} for k, v in res.items()}
summary["delta"] = {"only_p0b": len(eb - ea), "only_armb": len(ea - eb), "shared": len(eb & ea)}
json.dump(summary, open(pathlib.Path(__file__).parent / "armb_artifact_audit.json", "w"), indent=2)
print("\nwrote armb_artifact_audit.json")
