"""Run the V3 DoG pipeline on train crops, score with the numpy metric, and build
the edge-error taxonomy that decides detector-vs-linker investment.

For each crop: run V3 (detect+link) on the image, match pred->GT nodes, then split
every GT edge into:
  - TP: both endpoints matched AND a pred edge connects the matched pair;
  - DETECTION-miss: >=1 endpoint has no pred node within 7um (a detection failure);
  - ASSOCIATION-miss: both endpoints detected but no pred edge (a linking failure).

If DETECTION-miss dominates -> fund a learned detector (Spotiflow). If ASSOCIATION-miss
dominates -> fix linking, don't build a detector. Also reports node_recall, count ratio,
and adjusted edge Jaccard per crop and per embryo (44b6 vs 6bba = the CV folds).

Usage: python scripts/run_v3_taxonomy.py --per-embryo 6
"""

import argparse
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biotrack.metric import estimated_nodes  # noqa: E402
from biotrack.metric_numpy import Sample, gt_candidate_within, match_nodes, score_sample  # noqa: E402

# import the V3 notebook module
_spec = importlib.util.spec_from_file_location("dogv3", ROOT / "notebooks" / "kaggle_dog_infer.py")
dog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dog)

NID = None  # set below via tracksdata keys


def rows_to_sample(rows: list[dict]) -> Sample:
    nid, t, zyx, edges = [], [], [], []
    for r in rows:
        if r["row_type"] == "node":
            nid.append(r["node_id"]); t.append(r["t"]); zyx.append((r["z"], r["y"], r["x"]))
    for r in rows:
        if r["row_type"] == "edge":
            edges.append((r["source_id"], r["target_id"]))
    return Sample(
        node_ids=np.asarray(nid, np.int64),
        t=np.asarray(t, np.int64),
        zyx=np.asarray(zyx, float).reshape(-1, 3),
        edges=np.asarray(edges, np.int64).reshape(-1, 2),
    )


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
    """3-way FN split (corrected): a GT edge FN is attributed to
       - no_cand:     an endpoint has NO pred node within 7um (true detection miss);
       - lost_assign: an endpoint has a candidate within 7um but lost the 1-to-1 match
                      (arbitration/assignment-stealing failure, NOT a detector failure);
       - assoc:       both endpoints matched but no pred edge connects them (linking failure).
    """
    matched = match_nodes(pred, gt)                       # pred_id -> gt_id
    gt_to_pred = {g: p for p, g in matched.items()}       # bipartite -> clean inverse
    cand = gt_candidate_within(pred, gt)                  # gt_id -> pred within 7um exists?
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
                no_cand += 1          # undetected -> better detector / redetection
            else:
                lost_assign += 1      # detected but lost the match -> arbitration
    return {"tp": tp, "no_cand": no_cand, "lost_assign": lost_assign, "assoc": assoc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-embryo", type=int, default=6, help="crops per embryo family")
    args = ap.parse_args()

    geffs = {p.stem: p for p in (ROOT / "data/train").glob("*.geff")}
    zarrs = {p.stem: p for p in (ROOT / "data/train").glob("*.zarr")}
    both = sorted(set(geffs) & set(zarrs))
    by_fam = defaultdict(list)
    for name in both:
        by_fam[name.split("_")[0]].append(name)

    picks = []
    for fam, names in by_fam.items():
        picks += names[: args.per_embryo]

    agg = defaultdict(lambda: defaultdict(float))
    print(f"Running V3 on {len(picks)} crops ({args.per_embryo}/embryo)...\n")
    print(f"{'crop':18s} {'fam':5s} {'Npred':>6} {'Nest':>6} {'ratio':>5} {'recall':>6} "
          f"{'adjJ':>6} {'tp':>5} {'noCand':>6} {'lostA':>6} {'assoc':>6}")
    for name in picks:
        rows = dog.infer_dataset(zarrs[name])
        pred = rows_to_sample(rows)
        gt = geff_to_sample(str(geffs[name]))
        n_est = estimated_nodes(str(geffs[name]))
        s = score_sample(pred, gt, n_est)
        tax = classify_edges(pred, gt)
        fam = name.split("_")[0]
        ratio = s["num_pred_nodes"] / n_est if n_est else float("nan")
        print(f"{name[:18]:18s} {fam:5s} {s['num_pred_nodes']:6d} {int(n_est):6d} {ratio:5.2f} "
              f"{s['node_recall']:6.3f} {s['adj_edge_jaccard']:6.3f} "
              f"{tax['tp']:5d} {tax['no_cand']:6d} {tax['lost_assign']:6d} {tax['assoc']:6d}")
        for k in ("edge_tp", "edge_fp", "edge_fn"):
            agg[fam][k] += s[k]
        for k in ("tp", "no_cand", "lost_assign", "assoc"):
            agg[fam]["tax_" + k] += tax[k]
        agg[fam]["adjw"] += s["adj_edge_jaccard"] * (s["edge_tp"] + s["edge_fp"] + s["edge_fn"])
        agg[fam]["w"] += s["edge_tp"] + s["edge_fp"] + s["edge_fn"]
        agg[fam]["recall_sum"] += s["node_recall"]; agg[fam]["n"] += 1

    print("\n=== per-embryo (fold) summary ===")
    for fam, a in sorted(agg.items()):
        tp, fp, fn = a["edge_tp"], a["edge_fp"], a["edge_fn"]
        micro_J = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
        adjJ = a["adjw"] / a["w"] if a["w"] else float("nan")
        nc, la, asc = a["tax_no_cand"], a["tax_lost_assign"], a["tax_assoc"]
        tot = nc + la + asc or 1
        print(f"  {fam}: adjJ={adjJ:.4f} microJ={micro_J:.4f} recall={a['recall_sum']/a['n']:.3f} "
              f"| FN no-candidate={nc:.0f} ({nc/tot*100:.0f}%) lost-assignment={la:.0f} ({la/tot*100:.0f}%) "
              f"association={asc:.0f} ({asc/tot*100:.0f}%)")
    print("\nno-candidate dominant -> detector/recall (Spotiflow). lost-assignment dominant -> ARBITRATION. "
          "association dominant -> linking.")


if __name__ == "__main__":
    main()
