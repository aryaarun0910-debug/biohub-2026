"""Arm-B residual-error census on the OOF (P0-strict) substrate, where GT exists.

Runs the CORRECTED arm-B wrapper over cached pre-wrapper graphs, exports coordinates exactly
as the deployed kernel does (`max(0, round(v))`), scores with the exact patched scorer, and
decomposes what is LEFT WRONG into buckets with a per-bucket oracle ceiling:

    score = adj_edge_jaccard + 0.1 * division_jaccard
    adj_edge_jaccard = weighted mean of  J * (1 - 0.1 * total_node_ratio)

Each ceiling answers: "if a future primitive fixed this bucket PERFECTLY, what is the most it
could be worth?" -- the project's oracle-vs-deployable discipline, applied before spending GPU.
"""
import argparse
import glob
import importlib.util
import json
import os
import pathlib
import sys

REPO = pathlib.Path(r"c:\Users\aryaa\Documents\Biohub-CellTracking-2026")
SP = pathlib.Path(__file__).parent

DEPLOY_ENV = {
    "BIOHUB_OUTPUT_FILTER_SHORT_TRACKS": "1", "BIOHUB_MOTION_RELINK_LEARNED_BONUS": "1.0",
    "BIOHUB_GAP_CLOSE_MAX_GAP": "2", "BIOHUB_GAP_CLOSE_UM": "5.8",
    "BIOHUB_GAP_DENSITY_ADAPTIVE": "1", "BIOHUB_GAP_DENSITY_REFERENCE_UM": "6.5",
    "BIOHUB_GAP_DENSITY_GAIN": "0.040", "BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM": "0.125",
    "BIOHUB_GAP_DENSITY_NEIGHBORS": "3", "BIOHUB_OUTPUT_MIN_TRACK_LEN": "6",
    "BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS": "1", "BIOHUB_OUTPUT_GAP2_RECOVERY": "0",
    "BIOHUB_SAFE_DIV_MAX_UM": "4.66", "BIOHUB_SAFE_DIV_SISTER_MAX_UM": "8.5",
    "BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM": "7.65",
    "BIOHUB_SAFE_DIV_FRAME_FRAC_CAP": "0.0076", "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP": "0.00375",
    "BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE": "0",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
}
_S = {}


def _build(gate_on):
    os.environ.update(DEPLOY_ENV)
    spec = json.loads((REPO / "scripts/kaggle_specs/p2_armb_flowgate.json").read_text("utf-8"))
    edits = [e for e in spec["edits"]
             if e.get("code_file", "").endswith("armb_flow_gate.py")
             or (e["kind"] == "replace" and "GATE_COMPENSATION" in e.get("new", ""))]
    text = (REPO / "src/biotrack/wrapper.py").read_text("utf-8")
    for e in edits:
        if e["kind"] == "insert_before":
            text = text.replace(
                e["anchor"], "\n" + (REPO / e["code_file"]).read_text("utf-8") + "\n" + e["anchor"], 1)
        else:
            text = text.replace(e["old"], e["new"], 1)
    text = text.replace('ARMB_FLOW_GATE = os.environ.get("BIOHUB_ARMB_FLOW_GATE", "1") != "0"',
                        f"ARMB_FLOW_GATE = {gate_on!r}")
    p = SP / f"_census_{'B' if gate_on else 'A'}.py"
    p.write_text(text, encoding="utf-8")
    s = importlib.util.spec_from_file_location(f"census_{gate_on}", p)
    m = importlib.util.module_from_spec(s)
    sys.modules[s.name] = m
    s.loader.exec_module(m)
    m.TEST_DIR = None
    return m


def init_worker(gate_on):
    global _S
    sys.path.insert(0, str(REPO / "src"))
    _S = {"m": _build(gate_on)}


def work(path):
    import polars as pl
    import tracksdata as td
    from biotrack.metric import score_pred_graph

    m = _S["m"]
    name = pathlib.Path(path).stem
    gt = REPO / "data" / "train" / f"{name}.geff"
    if not gt.exists():
        return None

    d = pl.read_parquet(path)
    nd = d.filter(pl.col("row_type") == "node")
    ed = d.filter(pl.col("row_type") == "edge")
    nodes = {int(r["node_id"]): {"node_id": int(r["node_id"]), "t": int(r["t"]),
                                 "z": float(r["z"]), "y": float(r["y"]), "x": float(r["x"])}
             for r in nd.iter_rows(named=True)}
    edges = [{"source_id": int(r["source_id"]), "target_id": int(r["target_id"])}
             for r in ed.iter_rows(named=True)]
    out_nodes, out_edges, stats = m.filter_output_graph(nodes, edges, dataset=name)

    # export exactly as the deployed kernel does
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    ids = sorted(out_nodes)
    internal = g.bulk_add_nodes([
        {"t": int(out_nodes[i]["t"]),
         "z": float(max(0, int(round(float(out_nodes[i]["z"]))))),
         "y": float(max(0, int(round(float(out_nodes[i]["y"]))))),
         "x": float(max(0, int(round(float(out_nodes[i]["x"])))))}
        for i in ids])
    remap = {sid: internal[k] for k, sid in enumerate(ids)}
    if out_edges:
        g.bulk_add_edges([{"source_id": remap[int(e["source_id"])],
                           "target_id": remap[int(e["target_id"])]} for e in out_edges])

    row = score_pred_graph(g, gt)
    row["crop"] = name
    row["family"] = name.split("_")[0]
    row["safe_div_added"] = stats.get("safe_divisions_added", 0)
    row["skipped_outdeg"] = stats.get("safe_division_skipped_outdegree", 0)
    row["gap_synthetic"] = stats.get("gap_inserted_synthetic", 0)
    return row


ALPHA, DIVW = 0.1, 0.1


def aggregate(rows, edge_fp_scale=1.0, edge_fn_scale=1.0,
              div_fp_scale=1.0, div_fn_scale=1.0, zero_node_ratio=False):
    """Re-aggregate with a bucket scaled down; scale 0 = that bucket perfectly fixed."""
    adj_num = adj_den = 0.0
    dtp = dfp = dfn = 0
    for r in rows:
        tp, fp, fn = r["edge_tp"], r["edge_fp"] * edge_fp_scale, r["edge_fn"] * edge_fn_scale
        den = tp + fp + fn
        if den <= 0:
            continue
        j = tp / den
        tnr = 0.0 if zero_node_ratio else r["total_node_ratio"]
        adj = max(0.0, j * (1 - ALPHA * tnr))
        w = tp + fp + fn
        adj_num += w * adj
        adj_den += w
        dtp += r["division_tp"]
        dfp += r["division_fp"] * div_fp_scale
        dfn += r["division_fn"] * div_fn_scale
    adj = adj_num / adj_den if adj_den else float("nan")
    ddem = dtp + dfp + dfn
    dj = dtp / ddem if ddem else float("nan")
    return adj + DIVW * dj, adj, dj


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--gate", default="B")
    a = ap.parse_args()

    crops = sorted(glob.glob(str(REPO / "artifacts/kaggle/p0strict_cache/graphs/*/*.parquet")))[::a.stride]
    print(f"crops={len(crops)} gate={a.gate}", flush=True)
    rows = []
    with mp.Pool(a.workers, initializer=init_worker, initargs=(a.gate.upper() == "B",)) as pool:
        for i, r in enumerate(pool.imap_unordered(work, crops), 1):
            if r:
                rows.append(r)
            if i % 5 == 0 or i == len(crops):
                print(f"  {i}/{len(crops)}", flush=True)

    base, adj, dj = aggregate(rows)
    tot = {k: sum(r[k] for r in rows) for k in
           ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn")}

    print("\n" + "=" * 78)
    print(f"ARM-B RESIDUAL-ERROR CENSUS — {len(rows)} OOF crops, exact scorer")
    print("=" * 78)
    print(f"  score               : {base:.6f}   (adj_edge_J {adj:.6f} + 0.1 * div_J {dj:.6f})")
    print(f"  edge   TP/FP/FN     : {tot['edge_tp']} / {tot['edge_fp']} / {tot['edge_fn']}")
    print(f"  division TP/FP/FN   : {tot['division_tp']} / {tot['division_fp']} / {tot['division_fn']}")
    print(f"  mean node_recall    : {sum(r['node_recall'] for r in rows)/len(rows):.6f}")
    print(f"  mean total_node_ratio: {sum(r['total_node_ratio'] for r in rows)/len(rows):+.6f}")

    print("\n  PER-BUCKET ORACLE CEILING (perfect fix of that bucket alone)")
    print(f"  {'bucket':28s} {'score':>10s} {'delta':>11s}")
    for label, kw in [
        ("edge FP -> 0", dict(edge_fp_scale=0.0)),
        ("edge FN -> 0", dict(edge_fn_scale=0.0)),
        ("edge FP halved", dict(edge_fp_scale=0.5)),
        ("edge FN halved", dict(edge_fn_scale=0.5)),
        ("division FP -> 0", dict(div_fp_scale=0.0)),
        ("division FN -> 0", dict(div_fn_scale=0.0)),
        ("division FP+FN -> 0", dict(div_fp_scale=0.0, div_fn_scale=0.0)),
        ("node ratio -> 0", dict(zero_node_ratio=True)),
    ]:
        s, _, _ = aggregate(rows, **kw)
        print(f"  {label:28s} {s:10.6f} {s-base:+11.6f}")

    print("\n  BY FAMILY")
    for fam in sorted({r["family"] for r in rows}):
        sub = [r for r in rows if r["family"] == fam]
        s, aa, dd = aggregate(sub)
        t = {k: sum(r[k] for r in sub) for k in ("edge_fp", "edge_fn", "division_fp", "division_fn")}
        print(f"    {fam}  n={len(sub):3d}  score={s:.6f}  adjJ={aa:.6f}  divJ={dd:.6f}"
              f"  eFP={t['edge_fp']} eFN={t['edge_fn']} dFP={t['division_fp']} dFN={t['division_fn']}")

    print("\n  WORST 8 CROPS BY adj_edge_jaccard")
    for r in sorted(rows, key=lambda r: r["adj_edge_jaccard"])[:8]:
        print(f"    {r['crop']:22s} adjJ={r['adj_edge_jaccard']:.5f} "
              f"J={r['edge_jaccard']:.5f} eFP={r['edge_fp']:5d} eFN={r['edge_fn']:5d} "
              f"dTP/FP/FN={r['division_tp']}/{r['division_fp']}/{r['division_fn']} "
              f"nodeRatio={r['total_node_ratio']:+.4f}")

    json.dump(rows, open(SP / f"census_rows_{a.gate}.json", "w"), indent=2)
    print(f"\nwrote census_rows_{a.gate}.json")


if __name__ == "__main__":
    main()
