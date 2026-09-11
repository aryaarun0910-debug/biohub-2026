"""Stage 6 — submit: Graph -> submission.csv, validated BEFORE the file is written.

A non-consecutive edge scores an exact 0.0 with no error raised: the scorer drops dt!=1 edges
before counting, so TP is structurally zero however good the detection was, and the CSV is
accepted as well-formed. Validation therefore happens here, not at submission time, and the
invariants are enforced rather than checked — a violation raises.
"""
from __future__ import annotations
import csv, io
from collections import Counter
import numpy as np
from .contracts import Graph, BOUNDS

HEADER = ["id","dataset","row_type","node_id","t","z","y","x","source_id","target_id"]

class SubmissionError(AssertionError):
    """Raised when a graph would produce a silently-wrong submission."""

def check(g: Graph) -> list[str]:
    """Every invariant is measured, not conventional. Returns the list of violations."""
    bad = []
    t, zyx, E = g.t, g.zyx, g.edges
    if not len(E):
        bad.append("EDGELESS: evaluate() returns early and raises; the graph is unscorable")
    if len(E):
        dt = t[E[:,1]] - t[E[:,0]]
        n = int((dt != 1).sum())
        if n: bad.append(f"{n} edges with dt != 1 — dropped before counting, silently scores 0.0")
        od = Counter(E[:,0].tolist())
        over = sum(1 for c in od.values() if c > 2)
        if over: bad.append(f"{over} nodes with out-degree > 2 — every GT fork is strictly binary")
    for k, v in (("t", t), ("z", zyx[:,0]), ("y", zyx[:,1]), ("x", zyx[:,2])):
        lo, hi = BOUNDS[k]
        r = np.rint(v)
        n = int(((r < lo) | (r > hi)).sum())
        if n: bad.append(f"{n} nodes outside {k} bounds {lo}..{hi}")
    return bad

def to_rows(g: Graph, start_id: int = 0):
    """Coordinates are emitted as INTEGER voxel indices: the ground truth is int64 and there
    is no sub-voxel signal the metric can read."""
    i = start_id
    zyx = np.rint(g.zyx).astype(int)
    for nid, (tt, (z, y, x)) in enumerate(zip(g.t, zyx)):
        yield [i, g.dataset, "node", nid, int(tt), int(z), int(y), int(x), -1, -1]; i += 1
    for s, d in g.edges:
        yield [i, g.dataset, "edge", -1, -1, -1, -1, -1, int(s), int(d)]; i += 1

def write(graphs: list[Graph], path: str, expect_datasets=None, strict: bool = True) -> dict:
    """Write submission.csv. Raises rather than emitting a graph that would score wrong."""
    problems = {}
    for g in graphs:
        v = check(g)
        if v: problems[g.dataset] = v
    if expect_datasets:
        missing = set(expect_datasets) - {g.dataset for g in graphs}
        if missing: problems["_coverage"] = [f"missing datasets: {sorted(missing)}"]
    if problems and strict:
        raise SubmissionError("; ".join(f"{k}: {' | '.join(v)}" for k, v in problems.items()))

    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(HEADER)
        for g in graphs:
            for row in to_rows(g, n): w.writerow(row); n += 1
    return {"rows": n, "datasets": len(graphs),
            "nodes": sum(len(g.t) for g in graphs),
            "edges": sum(len(g.edges) for g in graphs),
            "forks": sum(g.forks() for g in graphs),
            "problems": problems}
