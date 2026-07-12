"""Run the V3 DoG pipeline on train crops, score with the exact numpy metric, and write a
machine-readable per-crop edge-error taxonomy CSV (the source of truth for figures/journal).

Per missed GT edge:
  - no_cand:     an endpoint has NO pred node within 7um (true detection miss);
  - lost_assign: an endpoint has a candidate within 7um but lost the 1-to-1 match (arbitration);
  - assoc:       both endpoints matched but no pred edge connects them (linking).
Also per-crop: Npred, Nest, count ratio, node recall, adjusted edge Jaccard.

Writes reports/inventory/v3_taxonomy.csv. Parallel across crops.

Usage:
    python scripts/run_v3_taxonomy.py --all --workers 8
    python scripts/run_v3_taxonomy.py --per-embryo 10 --workers 6
"""

import argparse
import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biotrack.metric import estimated_nodes  # noqa: E402
from biotrack.metric_numpy import Sample, gt_candidate_within, match_nodes, score_sample  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "dogv3", ROOT / "notebooks" / "kaggle_submit" / "kaggle_dog_infer.py"
)
dog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dog)

TRAIN = ROOT / "data" / "train"
OUT = ROOT / "reports" / "inventory" / "v3_taxonomy.csv"
FIELDS = ["crop", "fam", "n_pred", "n_est", "ratio", "recall", "adjJ",
          "edge_tp", "edge_fp", "edge_fn", "tp", "no_cand", "lost_assign", "assoc"]


def rows_to_sample(rows: list[dict]) -> Sample:
    nid, t, zyx, edges = [], [], [], []
    for r in rows:
        if r["row_type"] == "node":
            nid.append(r["node_id"]); t.append(r["t"]); zyx.append((r["z"], r["y"], r["x"]))
        elif r["row_type"] == "edge":
            edges.append((r["source_id"], r["target_id"]))
    return Sample(node_ids=np.asarray(nid, np.int64), t=np.asarray(t, np.int64),
                  zyx=np.asarray(zyx, float).reshape(-1, 3), edges=np.asarray(edges, np.int64).reshape(-1, 2))


def geff_to_sample(geff: str) -> Sample:
    import tracksdata as td
    from biotrack.metric import load_graph
    g = load_graph(geff)
    na = g.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t", "z", "y", "x"])
    ids = np.asarray(na[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(), np.int64)
    t = np.asarray(na["t"].to_list(), np.int64)
    zyx = np.stack([na["z"].to_numpy(), na["y"].to_numpy(), na["x"].to_numpy()], 1).astype(float)
    if g.num_edges() > 0:
        ea = g.edge_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET])
        edges = np.stack([np.asarray(ea[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE].to_list()),
                          np.asarray(ea[td.DEFAULT_ATTR_KEYS.EDGE_TARGET].to_list())], 1).astype(np.int64)
    else:
        edges = np.zeros((0, 2), np.int64)
    return Sample(node_ids=ids, t=t, zyx=zyx, edges=edges)


def classify_edges(pred: Sample, gt: Sample) -> dict:
    matched = match_nodes(pred, gt)
    gt_to_pred = {g: p for p, g in matched.items()}
    cand = gt_candidate_within(pred, gt)
    pred_edge_set = {(int(a), int(b)) for a, b in pred.edges}
    tp = no_cand = lost_assign = assoc = 0
    for gs, gt_ in gt.edges:
        gs, gt_ = int(gs), int(gt_)
        ps, pt = gt_to_pred.get(gs), gt_to_pred.get(gt_)
        if ps is not None and pt is not None:
            if (ps, pt) in pred_edge_set:
                tp += 1
            else:
                assoc += 1
        else:
            unmatched = [g for g, p in ((gs, ps), (gt_, pt)) if p is None]
            if any(not cand.get(g, False) for g in unmatched):
                no_cand += 1
            else:
                lost_assign += 1
    return {"tp": tp, "no_cand": no_cand, "lost_assign": lost_assign, "assoc": assoc}


def process_crop(name: str) -> dict:
    rows = dog.infer_dataset(TRAIN / f"{name}.zarr")
    pred = rows_to_sample(rows)
    gt = geff_to_sample(str(TRAIN / f"{name}.geff"))
    n_est = estimated_nodes(str(TRAIN / f"{name}.geff"))
    s = score_sample(pred, gt, n_est)
    tax = classify_edges(pred, gt)
    return {"crop": name, "fam": name.split("_")[0], "n_pred": s["num_pred_nodes"],
            "n_est": int(n_est), "ratio": round(s["num_pred_nodes"] / n_est, 4) if n_est else float("nan"),
            "recall": round(s["node_recall"], 4), "adjJ": round(s["adj_edge_jaccard"], 4),
            "edge_tp": s["edge_tp"], "edge_fp": s["edge_fp"], "edge_fn": s["edge_fn"], **tax}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--per-embryo", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    both = sorted(p.stem for p in TRAIN.glob("*.geff") if (TRAIN / f"{p.stem}.zarr").exists())
    if not args.all:
        by_fam = defaultdict(list)
        for n in both:
            by_fam[n.split("_")[0]].append(n)
        both = [n for names in by_fam.values() for n in names[: args.per_embryo]]

    print(f"Running V3 on {len(both)} crops with {args.workers} workers...")
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(process_crop, both), 1):
                results.append(r)
                if i % 10 == 0 or i == len(both):
                    print(f"  {i}/{len(both)}")
    else:
        results = [process_crop(n) for n in both]

    results.sort(key=lambda r: r["crop"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(results)
    print(f"Wrote {OUT} ({len(results)} crops)")

    agg = defaultdict(lambda: defaultdict(float))
    for r in results:
        a = agg[r["fam"]]
        w = r["edge_tp"] + r["edge_fp"] + r["edge_fn"]
        a["adjw"] += r["adjJ"] * w; a["w"] += w; a["recall"] += r["recall"]; a["n"] += 1
        for k in ("no_cand", "lost_assign", "assoc"):
            a["tax_" + k] += r[k]
    print("\n=== per-embryo (fold) summary ===")
    for fam, a in sorted(agg.items()):
        nc, la, asc = a["tax_no_cand"], a["tax_lost_assign"], a["tax_assoc"]
        tot = nc + la + asc or 1
        print(f"  {fam}: adjJ={a['adjw']/a['w']:.4f} recall={a['recall']/a['n']:.3f} | "
              f"no-candidate={nc/tot*100:.0f}% lost-assignment={la/tot*100:.0f}% association={asc/tot*100:.0f}%")


if __name__ == "__main__":
    main()
