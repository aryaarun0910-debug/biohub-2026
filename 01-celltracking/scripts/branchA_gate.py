"""Branch A gate (reports/NEXT_DECISION.md) -- CPU only, no GPU, no Kaggle session.

Tests whether expanded pre-ILP candidate breadth (top-two transformer parents down to
p=0.25 under a 10 um motion cap) is a genuinely unmeasured mechanism, on the two
preregistered highest-edge-mass crops: 44b6_d29c9ab2 (split 0) and 6bba_bb9f20c3 (split 1).

Stages (all read cached artifacts; each writes one JSON to reports/inventory/):

  surface   Gate A0.2 -- does the organizer OOF GEFF already carry top-two/0.25 candidates?
  ceiling   Gate A1.1 -- oracle upper bound on the share of E0c matched-edge FNs that ANY
            within-cap candidate mechanism could recover, split by blocking cause.
  widen     Gate A1.2 -- exact clean score of a single global 10 um enumeration vs E0c.
  oracleprob        ceiling for `widen` given a PERFECT edge probability (ORACLE, never
            submitted): if this loses, no attainable probability rescues the mechanism.
  control   diagnostic -- repeats the oracle comparison with the short-track filter off,
            to rule out a downstream-filter artifact.

Usage:
  .venv\\Scripts\\python.exe scripts\\branchA_gate.py --stage all
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tracksdata as td  # noqa: E402
from tracksdata.metrics import DistanceMatching  # noqa: E402

from biotrack import wrapper as W  # noqa: E402
from biotrack.metric import DEFAULT_SCALE, MAX_DISTANCE, estimated_nodes, load_graph  # noqa: E402
from tracking_cellmot.metrics import (  # noqa: E402
    evaluate, node_recall as _node_recall, per_sample_metrics,
)

K = td.DEFAULT_ATTR_KEYS
OUT = ROOT / "reports" / "inventory"
CROPS = [(0, "44b6_d29c9ab2"), (1, "6bba_bb9f20c3")]
MOTION_CAP_UM = 10.0
TIGHT_UM = 6.0

_ORIG_RELINK = W.motion_relink_edges
_ORACLE: dict[tuple[int, int], float] = {}


# --------------------------------------------------------------------------- helpers
def set_e0c_config() -> None:
    W.OUTPUT_MIN_TRACK_LEN = 7
    W.SHORT_TRACK_MIN_LEN_BY_DATASET = {}
    W.GAP_REFINE_SYNTHETIC = True
    W.MOTION_RELINK_TIGHT_UM = TIGHT_UM
    W.MOTION_RELINK_RELAXED_UM = MOTION_CAP_UM
    W.MOTION_RELINK_LEARNED_BONUS = 0.75
    W.OUTPUT_FILTER_SHORT_TRACKS = True
    W.TEST_DIR = ROOT / "data" / "train"


def build_graph(nodes: list[dict], edges: list[tuple[int, int]]):
    g = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        g.add_node_attr_key(key, pl.Float64, -999999.0)
    internal = g.bulk_add_nodes([
        {"t": int(n["t"]), "z": float(n["z"]), "y": float(n["y"]), "x": float(n["x"])}
        for n in nodes
    ])
    ext_to_int = {int(n["node_id"]): internal[i] for i, n in enumerate(nodes)}
    if edges:
        g.bulk_add_edges([{"source_id": ext_to_int[s], "target_id": ext_to_int[t]}
                          for s, t in edges])
    return g, ext_to_int


def gt_path(crop: str) -> Path:
    return ROOT / "data" / "train" / f"{crop}.geff"


def geff_path(split: int, crop: str) -> Path:
    return ROOT / "artifacts" / "kaggle" / "oof_clean" / f"pred_geffs_split_{split}" / f"{crop}.geff"


def load_e0c(split: int, crop: str):
    df = pl.read_parquet(ROOT / f"artifacts/kaggle/e0c_cache/graphs/{split}/{crop}.parquet")
    nodes = [{"node_id": r["node_id"], "t": r["t"], "z": r["z"], "y": r["y"], "x": r["x"]}
             for r in df.filter(pl.col("row_type") == "node").iter_rows(named=True)]
    edges = [(int(r["source_id"]), int(r["target_id"]))
             for r in df.filter(pl.col("row_type") == "edge").iter_rows(named=True)]
    return build_graph(nodes, edges)[0]


def load_raw(split: int, crop: str):
    g = W.graph_from_geff(geff_path(split, crop))
    nodes = [{"node_id": int(r["node_id"]), "t": int(r["t"]), "z": float(r["z"]),
              "y": float(r["y"]), "x": float(r["x"])}
             for r in g.node_attrs().iter_rows(named=True)]
    raw_edges = {(int(r["source_id"]), int(r["target_id"]))
                 for r in g.edge_attrs().iter_rows(named=True)}
    graph, ext_to_int = build_graph(nodes, sorted(raw_edges))
    return graph, ext_to_int, nodes, raw_edges


def matched_map(pred) -> dict[int, int]:
    na = pred.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID])
    return {int(n): int(g) for n, g in zip(na[K.NODE_ID].to_list(), na[K.MATCHED_NODE_ID].to_list())
            if g is not None and int(g) != -1}


def covered_gt_edges(pred) -> set[tuple[int, int]]:
    ea = pred.edge_attrs(attr_keys=[K.EDGE_SOURCE, K.EDGE_TARGET, K.MATCHED_EDGE_MASK])
    m = matched_map(pred)
    return {(m[int(s)], m[int(t)])
            for s, t, ok in zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list(),
                                ea[K.MATCHED_EDGE_MASK].to_list())
            if ok and int(s) in m and int(t) in m}


def pos_um(n: dict) -> np.ndarray:
    return np.array([n["z"] * DEFAULT_SCALE[0], n["y"] * DEFAULT_SCALE[1],
                     n["x"] * DEFAULT_SCALE[2]])


def score(graph, gt_geff: Path) -> dict:
    gt = load_graph(gt_geff)
    er = evaluate(graph, gt, scale=DEFAULT_SCALE, max_distance=MAX_DISTANCE)
    rec = _node_recall(graph, gt) if graph.num_edges() > 0 and graph.num_nodes() > 0 else 0.0
    return per_sample_metrics(er, estimated_nodes(gt_geff), rec)


def _relink_with_oracle(nodes_by_id, stats, learned_edge_probs=None):
    merged = dict(learned_edge_probs or {})
    merged.update(_ORACLE)
    return _ORIG_RELINK(nodes_by_id, stats, merged)


def run_wrapper(split: int, crop: str, *, tight_um: float, bonus: float = 0.75,
                oracle: bool = False, short_filter: bool = True):
    set_e0c_config()
    W.MOTION_RELINK_TIGHT_UM = tight_um
    W.MOTION_RELINK_LEARNED_BONUS = bonus
    W.OUTPUT_FILTER_SHORT_TRACKS = short_filter
    W.motion_relink_edges = _relink_with_oracle if oracle else _ORIG_RELINK
    g = W.graph_from_geff(geff_path(split, crop))
    nbi = {int(r["node_id"]): {"node_id": int(r["node_id"]), "t": int(r["t"]), "z": float(r["z"]),
                               "y": float(r["y"]), "x": float(r["x"])}
           for r in g.node_attrs().iter_rows(named=True)}
    raw = [{"source_id": int(r["source_id"]), "target_id": int(r["target_id"]),
            "edge_prob": None if r.get("edge_prob") is None else float(r["edge_prob"])}
           for r in g.edge_attrs().iter_rows(named=True)]
    try:
        fn, fe, stats = W.filter_output_graph(copy.deepcopy(nbi), raw, dataset=crop)
    finally:
        W.motion_relink_edges = _ORIG_RELINK
        set_e0c_config()
    nodes = [{"node_id": nid, **{k: fn[nid][k] for k in ("t", "z", "y", "x")}} for nid in sorted(fn)]
    graph, _ = build_graph(nodes, [(int(e["source_id"]), int(e["target_id"])) for e in fe])
    return graph, stats


def fn_context(split: int, crop: str):
    """E0c's matched-edge FN GT edges plus the raw detection surface they live on."""
    gt = load_graph(gt_path(crop))
    gt_edges = {(int(r[K.EDGE_SOURCE]), int(r[K.EDGE_TARGET]))
                for r in gt.edge_attrs().iter_rows(named=True)}
    e0c = load_e0c(split, crop)
    er = evaluate(e0c, gt, scale=DEFAULT_SCALE, max_distance=MAX_DISTANCE)
    fn_edges = sorted(gt_edges - covered_gt_edges(e0c))

    raw_g, raw_ext_to_int, raw_nodes, raw_edge_set = load_raw(split, crop)
    raw_g.match(gt, matching=DistanceMatching(max_distance=MAX_DISTANCE, scale=DEFAULT_SCALE))
    int_to_ext = {v: k for k, v in raw_ext_to_int.items()}
    gt_to_raw = {gid: int_to_ext[nid] for nid, gid in matched_map(raw_g).items()}
    return {"gt": gt, "gt_edges": gt_edges, "er": er, "fn_edges": fn_edges,
            "gt_to_raw": gt_to_raw, "raw_by_ext": {int(n["node_id"]): n for n in raw_nodes},
            "raw_edge_set": raw_edge_set, "n_gt_nodes": gt.num_nodes()}


def build_oracle(split: int, crop: str) -> dict[tuple[int, int], float]:
    ctx = fn_context(split, crop)
    out = {}
    for r in ctx["gt"].edge_attrs().iter_rows(named=True):
        s = ctx["gt_to_raw"].get(int(r[K.EDGE_SOURCE]))
        t = ctx["gt_to_raw"].get(int(r[K.EDGE_TARGET]))
        if s is not None and t is not None:
            out[(s, t)] = 1.0
    return out


# --------------------------------------------------------------------------- stages
def stage_surface() -> list[dict]:
    """Gate A0.2: is the top-two / p>=0.25 candidate set present in the cached GEFF?"""
    rows = []
    for split, crop in CROPS:
        g = W.graph_from_geff(geff_path(split, crop))
        edges = list(g.edge_attrs().iter_rows(named=True))
        probs = np.array([float(e["edge_prob"]) for e in edges if e.get("edge_prob") is not None])
        in_deg = Counter(int(e["target_id"]) for e in edges)
        print(f"\n=== {crop} (split {split}) ===")
        print(f"  edges={len(edges)}  min edge_prob={probs.min():.4f}  max={probs.max():.4f}")
        print(f"  in-degree histogram (parents per target): {dict(sorted(Counter(in_deg.values()).items()))}")
        print(f"  targets with >=2 parents: {sum(1 for v in in_deg.values() if v >= 2)}")
        print(f"  edges below p=0.5: {int((probs < 0.5).sum())}")
        rows.append({"split": split, "crop": crop, "n_edges": len(edges),
                     "min_edge_prob": float(probs.min()), "max_edge_prob": float(probs.max()),
                     "in_degree_hist": {str(k): v for k, v in sorted(Counter(in_deg.values()).items())},
                     "targets_with_two_parents": sum(1 for v in in_deg.values() if v >= 2),
                     "edges_below_0p5": int((probs < 0.5).sum())})
    return rows


def stage_ceiling() -> list[dict]:
    """Gate A1.1: oracle recoverable share of E0c matched-edge FNs, by blocking cause."""
    rows = []
    for split, crop in CROPS:
        ctx = fn_context(split, crop)
        cand = pl.read_parquet(ROOT / f"artifacts/kaggle/e0c_cache/candidates/{split}/{crop}.parquet")
        pairs, sel_src, sel_tgt = set(), {}, {}
        for r in cand.iter_rows(named=True):
            s, t = int(r["source_id"]), int(r["target_id"])
            pairs.add((s, t))
            if int(r["selected"]):
                sel_src.setdefault(s, []).append(t)
                sel_tgt.setdefault(t, []).append(s)

        c = Counter()
        for gs, gtt in ctx["fn_edges"]:
            rs, rt = ctx["gt_to_raw"].get(gs), ctx["gt_to_raw"].get(gtt)
            if rs is None or rt is None:
                c["endpoint_not_detected"] += 1
                continue
            ns, nt = ctx["raw_by_ext"][rs], ctx["raw_by_ext"][rt]
            if int(nt["t"]) - int(ns["t"]) != 1:
                c["endpoints_not_consecutive"] += 1
                continue
            d = float(np.linalg.norm(pos_um(nt) - pos_um(ns)))
            if d > MOTION_CAP_UM:
                c["beyond_10um_cap"] += 1
                continue
            if (rs, rt) in ctx["raw_edge_set"]:
                c["recoverable_already_a_transformer_edge"] += 1
                continue
            if (rs, rt) not in pairs:
                c["recoverable_new_beyond_tight_6um"] += 1 if d > TIGHT_UM else 0
                c["recoverable_new_other"] += 1 if d <= TIGHT_UM else 0
                continue
            s_used = rs in sel_src and rt not in sel_src[rs]
            t_used = rt in sel_tgt and rs not in sel_tgt[rt]
            key = ("both_consumed" if s_used and t_used else "source_consumed" if s_used
                   else "target_consumed" if t_used else "lost_on_cost")
            c[f"recoverable_enumerated_{key}"] += 1

        n_fn = len(ctx["fn_edges"])
        new_cand = c["recoverable_new_beyond_tight_6um"] + c["recoverable_new_other"]
        recoverable = n_fn - c["endpoint_not_detected"] - c["endpoints_not_consecutive"] - c["beyond_10um_cap"]
        print(f"\n=== {crop} (split {split}) ===")
        print(f"  E0c edge_tp={ctx['er'].edge_tp} fp={ctx['er'].edge_fp} fn={ctx['er'].edge_fn}")
        print(f"  matched-edge FN GT edges = {n_fn}")
        for k, v in c.most_common():
            print(f"    {k:<46}: {v:>4} ({v/n_fn:.4f})")
        print(f"  recoverable within cap = {recoverable} ({recoverable/n_fn:.4f})")
        print(f"  of which NEW candidates = {new_cand} ({new_cand/n_fn:.4f})  <- Gate A1.1 vs 0.10")
        rows.append({"split": split, "crop": crop, "e0c_edge_tp": ctx["er"].edge_tp,
                     "e0c_edge_fp": ctx["er"].edge_fp, "e0c_edge_fn": ctx["er"].edge_fn,
                     "fn_gt_edges": n_fn, "breakdown": dict(c),
                     "recoverable_within_cap": recoverable,
                     "recoverable_frac": recoverable / n_fn,
                     "new_candidate_frac": new_cand / n_fn})
    return rows


def stage_widen() -> list[dict]:
    """Gate A1.2: exact clean score of one global 10 um enumeration vs E0c."""
    rows = []
    for split, crop in CROPS:
        gtp = gt_path(crop)
        base = score(load_e0c(split, crop), gtp)
        repro = score(run_wrapper(split, crop, tight_um=TIGHT_UM)[0], gtp)
        parity = (repro["edge_tp"] == base["edge_tp"] and repro["edge_fp"] == base["edge_fp"]
                  and repro["num_pred_nodes"] == base["num_pred_nodes"])
        wide = score(run_wrapper(split, crop, tight_um=MOTION_CAP_UM)[0], gtp)
        d = wide["adj_edge_jaccard"] - base["adj_edge_jaccard"]
        print(f"\n=== {crop} (split {split}) ===")
        print(f"  E0c cached   adjJ={base['adj_edge_jaccard']:.4f} tp={base['edge_tp']} fp={base['edge_fp']}")
        print(f"  E0c rerun    parity={'OK' if parity else 'MISMATCH'}")
        print(f"  widened 10um adjJ={wide['adj_edge_jaccard']:.4f} ({d:+.4f}) "
              f"tp={wide['edge_tp']} fp={wide['edge_fp']}")
        rows.append({"split": split, "crop": crop, "parity_with_cache": parity,
                     "baseline": base, "widened_10um": wide, "delta_adj_edge_jaccard": d})
    return rows


def stage_oracleprob() -> list[dict]:
    """Ceiling for `widen` under a PERFECT edge probability. ORACLE -- never submitted."""
    global _ORACLE
    rows = []
    for split, crop in CROPS:
        gtp = gt_path(crop)
        base = score(load_e0c(split, crop), gtp)
        _ORACLE = build_oracle(split, crop)
        print(f"\n=== {crop} (split {split}) ===")
        print(f"  E0c baseline adjJ={base['adj_edge_jaccard']:.4f}  oracle edges={len(_ORACLE)}")
        variants = {}
        for label, bonus in (("oracle_p_bonus0.75", 0.75), ("oracle_p_bonus40", 40.0)):
            m = score(run_wrapper(split, crop, tight_um=MOTION_CAP_UM, bonus=bonus, oracle=True)[0], gtp)
            d = m["adj_edge_jaccard"] - base["adj_edge_jaccard"]
            print(f"  {label:<20} adjJ={m['adj_edge_jaccard']:.4f} ({d:+.4f}) "
                  f"tp={m['edge_tp']} fp={m['edge_fp']}")
            variants[label] = {**m, "delta_adj_edge_jaccard": d}
        _ORACLE = {}
        rows.append({"split": split, "crop": crop, "baseline": base, "variants": variants})
    return rows


def stage_control() -> list[dict]:
    """Diagnostic: repeat the oracle comparison with the short-track filter disabled."""
    global _ORACLE
    rows = []
    for split, crop in CROPS:
        gtp = gt_path(crop)
        _ORACLE = {}
        base = score(run_wrapper(split, crop, tight_um=TIGHT_UM, short_filter=False)[0], gtp)
        _ORACLE = build_oracle(split, crop)
        wide = score(run_wrapper(split, crop, tight_um=MOTION_CAP_UM, bonus=40.0,
                                 oracle=True, short_filter=False)[0], gtp)
        _ORACLE = {}
        d = wide["adj_edge_jaccard"] - base["adj_edge_jaccard"]
        print(f"\n=== {crop} (split {split}) short-track filter OFF ===")
        print(f"  E0c gate 6um   adjJ={base['adj_edge_jaccard']:.4f} tp={base['edge_tp']} fp={base['edge_fp']}")
        print(f"  widened+oracle adjJ={wide['adj_edge_jaccard']:.4f} ({d:+.4f}) "
              f"tp={wide['edge_tp']} fp={wide['edge_fp']}")
        rows.append({"split": split, "crop": crop, "short_track_filter": False,
                     "baseline": base, "widened_oracle_bonus40": wide,
                     "delta_adj_edge_jaccard": d})
    return rows


STAGES = {"surface": (stage_surface, "branchA_surface_audit.json"),
          "ceiling": (stage_ceiling, "branchA_oracle_ceiling.json"),
          "widen": (stage_widen, "branchA_widen10um_twocrop.json"),
          "oracleprob": (stage_oracleprob, "branchA_oracle_prob_twocrop.json"),
          "control": (stage_control, "branchA_control_nofilter.json")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=[*STAGES, "all"])
    args = ap.parse_args()
    names = list(STAGES) if args.stage == "all" else [args.stage]
    for name in names:
        fn, out_name = STAGES[name]
        print(f"\n########## stage: {name} ##########")
        rows = fn()
        dest = OUT / out_name
        dest.write_text(json.dumps(rows, indent=2, default=float))
        print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
