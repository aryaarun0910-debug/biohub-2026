r"""Error atlas part 3: what the ILP left on the table.

Three graphs, one GT-edge key (no cross-run node-id join is needed, because every
statement is keyed on the ground-truth edge identity that the metric itself assigns):

  AVAILABLE   GT edges covered by an UNCAPPED pre-ILP candidate edge
              (i.e. some candidate (s,t) has matched_gt(s)=gs and matched_gt(t)=gt)
  CHOSEN      GT edges that are TP in the deployed export
  DETECTABLE  GT edges whose two endpoints are both matched by some predicted node

  DETECTABLE >= AVAILABLE >= (what a perfect linker over these candidates could score)

Reports the linker headroom (AVAILABLE \ CHOSEN) and the detector headroom
(DETECTABLE \ AVAILABLE), plus what distinguishes a chosen-wrong edge from an
available-right one.

Usage
-----
  .\.venv\Scripts\python.exe scripts\win_bet\ea_solver_gap.py --dir c:\temp\error_atlas ^
      --pre-tag pre1 --dep-tag f1 --preilp c:\temp\preilp_f1_v2\preilp_split1.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pre-tag", required=True)
    ap.add_argument("--dep-tag", required=True)
    ap.add_argument("--preilp", required=True)
    ap.add_argument("--json-out")
    args = ap.parse_args()
    D = Path(args.dir)

    pn = pd.read_parquet(D / f"nodes_{args.pre_tag}.parquet")[["dataset", "node_id", "gt_id"]]
    pre = pd.read_parquet(args.preilp)
    ce = pre[pre.row_type == "edge"][["dataset", "source_id", "target_id", "edge_prob"]].copy()
    ce["source_id"] = ce.source_id.astype("int64")
    ce["target_id"] = ce.target_id.astype("int64")

    m = pn.set_index(["dataset", "node_id"]).gt_id
    ce["gs"] = m.reindex(pd.MultiIndex.from_arrays([ce.dataset, ce.source_id])).values
    ce["gt_"] = m.reindex(pd.MultiIndex.from_arrays([ce.dataset, ce.target_id])).values
    ce["maps_gt_edge"] = (ce.gs != -1) & (ce.gt_ != -1)

    gpre = pd.read_parquet(D / f"gtedges_{args.pre_tag}.parquet")
    gdep = pd.read_parquet(D / f"gtedges_{args.dep_tag}.parquet")

    # AVAILABLE: GT edge appears as a candidate mapping
    avail = set(map(tuple, ce.loc[ce.maps_gt_edge, ["dataset", "gs", "gt_"]]
                    .astype({"gs": "int64", "gt_": "int64"}).values))
    key_pre = list(map(tuple, gpre[["dataset", "gt_source", "gt_target"]].values))
    gpre = gpre.copy()
    gpre["available"] = [k in avail for k in key_pre]

    res: dict = {}
    res["n_gt_edges_pre_run"] = int(len(gpre))
    res["n_gt_edges_dep_run"] = int(len(gdep))
    res["detectable_pre_run"] = float((gpre.src_detected & gpre.tgt_detected).mean())
    res["available_pre_run"] = float(gpre.available.mean())
    res["capped_candidate_tp_pre_run"] = float(gpre.is_tp.mean())

    dep = gdep.set_index(["dataset", "gt_source", "gt_target"]).is_tp
    gpre = gpre.set_index(["dataset", "gt_source", "gt_target"])
    gpre["chosen"] = dep.reindex(gpre.index).fillna(False).values
    gpre = gpre.reset_index()

    A, C = gpre.available.values, gpre.chosen.values
    Dt = (gpre.src_detected & gpre.tgt_detected).values
    res["confusion"] = {
        "available_and_chosen": int((A & C).sum()),
        "available_not_chosen": int((A & ~C).sum()),
        "chosen_not_available": int((~A & C).sum()),
        "neither": int((~A & ~C).sum()),
        "detectable_not_available": int((Dt & ~A).sum()),
        "not_detectable": int((~Dt).sum()),
    }
    res["linker_headroom_frac_of_gt_edges"] = float((A & ~C).mean())
    res["detector_headroom_frac_of_gt_edges"] = float((Dt & ~A).mean())
    res["run_difference_noise_frac"] = float((~A & C).mean())

    # ---- ceiling scores -------------------------------------------------------
    # a perfect linker over the available candidates: every AVAILABLE GT edge becomes TP
    crops = pd.read_parquet(D / f"crops_{args.dep_tag}.parquet").set_index("dataset")
    per = gpre.groupby("dataset").agg(avail=("available", "sum"), chosen=("chosen", "sum"),
                                      n=("available", "size"))
    per = per.reindex(crops.index).fillna(0)
    tp = crops.edge_tp.values.astype(float)
    fp = crops.edge_fp.values.astype(float)
    fn = crops.edge_fn.values.astype(float)
    r = crops.total_node_ratio.values

    def pooled(tp, fp, fn, r):
        w = tp + fp + fn
        return float((w * np.maximum(0, tp / np.maximum(w, 1) * (1 - .1 * r))).sum() / w.sum())

    res["pooled_adjJ_deployed"] = pooled(tp, fp, fn, r)
    gain = np.maximum(0.0, per.avail.values - per.chosen.values)   # extra TP a perfect linker gets
    # a perfect linker also emits nothing wrong: FP -> 0 is a separate, looser bound
    res["pooled_adjJ_perfect_linker_over_candidates"] = pooled(
        tp + gain, fp, fn - gain, r)
    res["pooled_adjJ_perfect_linker_and_no_FP"] = pooled(tp + gain, fp * 0, fn - gain, r)
    res["pooled_adjJ_no_FP_only"] = pooled(tp, fp * 0, fn, r)

    # ---- what separates chosen-wrong from available-right ---------------------
    ce2 = ce[ce.maps_gt_edge].copy()
    kk = list(map(tuple, ce2[["dataset", "gs", "gt_"]].astype({"gs": "int64", "gt_": "int64"}).values))
    chosen_set = set(map(tuple, gpre.loc[gpre.chosen, ["dataset", "gt_source", "gt_target"]].values))
    ce2["is_gt_edge_chosen"] = [k in chosen_set for k in kk]
    res["edge_prob_of_correct_candidates"] = {
        "chosen": ce2.loc[ce2.is_gt_edge_chosen, "edge_prob"].describe().to_dict(),
        "not_chosen": ce2.loc[~ce2.is_gt_edge_chosen, "edge_prob"].describe().to_dict(),
    }
    q = [1, 5, 10, 25, 50, 75, 90]
    res["edge_prob_percentiles_missed_correct_candidates"] = dict(zip(
        map(str, q), np.percentile(ce2.loc[~ce2.is_gt_edge_chosen, "edge_prob"], q)))

    # candidate edges that are WRONG (map to a non-GT-edge pair or dangle)
    ce["correct"] = [k in avail if ok else False for k, ok in
                     zip(map(tuple, ce[["dataset", "gs", "gt_"]].values), ce.maps_gt_edge)]
    res["candidate_edge_prob_stats"] = {
        "n": int(len(ce)),
        "n_mapping_onto_a_gt_edge": int(ce.maps_gt_edge.sum()),
        "mean_prob_gt_edge": float(ce.loc[ce.maps_gt_edge, "edge_prob"].mean()),
        "mean_prob_other": float(ce.loc[~ce.maps_gt_edge, "edge_prob"].mean()),
    }

    out = json.dumps(res, indent=2, default=float)
    print(out)
    if args.json_out:
        Path(args.json_out).write_text(out)
    gpre.to_parquet(D / f"gtedge_avail_{args.pre_tag}.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
