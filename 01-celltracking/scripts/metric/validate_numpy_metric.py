"""Validate biotrack.metric_numpy against the tracksdata harness (biotrack.metric).

Builds diverse predicted graphs by perturbing real GT graphs (identity, drop edges,
jitter coords past 7um, add spurious nodes, drop nodes) and asserts both scorers agree
on edge_tp/fp/fn, num_pred_nodes, and adj_edge_jaccard. If they match, the numpy metric
is a faithful, tracksdata-free drop-in usable on Kaggle.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import tracksdata as td

sys.path.insert(0, str(next(_p for _p in Path(__file__).resolve().parents if (_p / 'pyproject.toml').exists()) / "src"))

from biotrack.metric import estimated_nodes, load_graph, score_pred_graph  # tracksdata reference
from biotrack.metric_numpy import Sample, score_sample                     # numpy under test

NID = td.DEFAULT_ATTR_KEYS.NODE_ID
ESRC = td.DEFAULT_ATTR_KEYS.EDGE_SOURCE
ETGT = td.DEFAULT_ATTR_KEYS.EDGE_TARGET
SCALE = (1.625, 0.40625, 0.40625)


def geff_to_sample(geff: str) -> Sample:
    g = load_graph(geff)
    na = g.node_attrs(attr_keys=[NID, "t", "z", "y", "x"])
    node_ids = np.asarray(na[NID].to_list(), dtype=np.int64)
    t = np.asarray(na["t"].to_list(), dtype=np.int64)
    zyx = np.stack([na["z"].to_numpy(), na["y"].to_numpy(), na["x"].to_numpy()], axis=1).astype(float)
    if g.num_edges() > 0:
        ea = g.edge_attrs(attr_keys=[ESRC, ETGT])
        edges = np.stack([np.asarray(ea[ESRC].to_list()), np.asarray(ea[ETGT].to_list())], axis=1).astype(np.int64)
    else:
        edges = np.zeros((0, 2), np.int64)
    return Sample(node_ids=node_ids, t=t, zyx=zyx, edges=edges)


def sample_to_graph(s: Sample) -> td.graph.BaseGraph:
    g = td.graph.InMemoryGraph()
    for k in ["z", "y", "x"]:
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    internal = g.bulk_add_nodes([
        {"t": int(tt), "z": float(z), "y": float(y), "x": float(x)}
        for tt, (z, y, x) in zip(s.t, s.zyx)
    ])
    id_map = {int(sid): internal[i] for i, sid in enumerate(s.node_ids)}
    if len(s.edges):
        g.bulk_add_edges([{"source_id": id_map[int(a)], "target_id": id_map[int(b)]} for a, b in s.edges])
    return g


def perturb(gt: Sample, kind: str, rng: np.random.Generator) -> Sample:
    nid, t, zyx, edges = gt.node_ids.copy(), gt.t.copy(), gt.zyx.copy(), gt.edges.copy()
    if kind == "identity":
        pass
    elif kind == "drop_edges":
        keep = rng.random(len(edges)) > 0.3
        edges = edges[keep]
    elif kind == "jitter":
        # push ~30% of nodes well past 7um (in y at 0.40625 um -> ~25 vox = ~10um)
        mask = rng.random(len(nid)) < 0.3
        zyx[mask, 1] += 25.0
    elif kind == "add_spurious":
        n_new = max(5, len(nid) // 5)
        new_ids = np.arange(nid.max() + 1, nid.max() + 1 + n_new)
        new_t = rng.integers(t.min(), t.max() + 1, n_new)
        new_zyx = zyx[rng.integers(0, len(nid), n_new)] + rng.normal(0, 40, (n_new, 3))
        nid = np.concatenate([nid, new_ids]); t = np.concatenate([t, new_t])
        zyx = np.concatenate([zyx, new_zyx])
    elif kind == "drop_nodes":
        keep = rng.random(len(nid)) > 0.2
        kept_ids = set(nid[keep].tolist())
        nid, t, zyx = nid[keep], t[keep], zyx[keep]
        edges = np.array([[a, b] for a, b in edges if a in kept_ids and b in kept_ids], np.int64).reshape(-1, 2)
    return Sample(node_ids=nid, t=t, zyx=zyx, edges=edges)


def main():
    root = next(_p for _p in Path(__file__).resolve().parents if (_p / 'pyproject.toml').exists())
    geffs = sorted((root / "data/train").glob("*.geff"))[:4]
    rng = np.random.default_rng(0)
    kinds = ["identity", "drop_edges", "jitter", "add_spurious", "drop_nodes"]
    keys = ["edge_tp", "edge_fp", "edge_fn", "num_pred_nodes"]
    n_ok = n_fail = 0
    for geff in geffs:
        gt_sample = geff_to_sample(str(geff))
        n_est = estimated_nodes(str(geff))
        for kind in kinds:
            pred_sample = perturb(gt_sample, kind, rng)
            ref = score_pred_graph(sample_to_graph(pred_sample), str(geff))
            got = score_sample(pred_sample, gt_sample, n_est)
            diffs = {k: (ref[k], got[k]) for k in keys if ref[k] != got[k]}
            adj_close = abs((ref["adj_edge_jaccard"] or 0) - (got["adj_edge_jaccard"] or 0)) < 1e-9 \
                if ref["adj_edge_jaccard"] == ref["adj_edge_jaccard"] else True
            if diffs or not adj_close:
                n_fail += 1
                print(f"MISMATCH {geff.stem[:16]} [{kind}]: counts {diffs} "
                      f"adj ref={ref['adj_edge_jaccard']:.6f} got={got['adj_edge_jaccard']:.6f}")
            else:
                n_ok += 1
                print(f"OK {geff.stem[:16]} [{kind:12s}] tp={got['edge_tp']} fp={got['edge_fp']} "
                      f"fn={got['edge_fn']} adjJ={got['adj_edge_jaccard']:.4f}")
    print(f"\n{n_ok} OK, {n_fail} MISMATCH")


if __name__ == "__main__":
    main()
