r"""Error atlas part 1 + 4: exact loss decomposition, concentration, per-crop pattern.

Consumes the tables written by ``scripts/win_bet/ea_atlas.py``.

Exact identity used throughout (no approximation; verified against the scorer):

    pooled_shortfall = 1 - adj_edge_jaccard
                     = sum_i [ FP_i + FN_i + 0.1 * r_i * TP_i ] / sum_i w_i
      with w_i = TP_i+FP_i+FN_i and r_i = (N_pred_i - N_est_i)/N_est_i,
      valid while adjJ_i is not clipped at 0.

So every crop's contribution to the shortfall is FP_i + FN_i + node-count term,
in *edge units*, directly comparable across crops.

Usage
-----
  .\.venv\Scripts\python.exe scripts\win_bet\ea_analyze.py --tag f0 --dir c:\temp\error_atlas
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCALE = np.array([1.625, 0.40625, 0.40625])


def lorenz(v: np.ndarray) -> dict:
    v = np.sort(np.asarray(v, dtype=float))[::-1]
    tot = v.sum()
    n = len(v)
    cum = np.cumsum(v) / tot
    out = {}
    for q in (0.05, 0.10, 0.20, 0.25, 0.50):
        k = max(1, int(round(q * n)))
        out[f"top{int(q*100)}pct_share"] = float(cum[k - 1])
        out[f"top{int(q*100)}pct_ncrops"] = k
    # gini
    idx = np.arange(1, n + 1)
    gini = (2 * (idx * np.sort(v)).sum()) / (n * v.sum()) - (n + 1) / n
    out["gini"] = float(gini)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--json-out")
    args = ap.parse_args()
    D = Path(args.dir)
    tag = args.tag

    crops = pd.read_parquet(D / f"crops_{tag}.parquet")
    edges = pd.read_parquet(D / f"edges_{tag}.parquet")
    gtedges = pd.read_parquet(D / f"gtedges_{tag}.parquet")
    gtnodes = pd.read_parquet(D / f"gtnodes_{tag}.parquet")
    nodes = pd.read_parquet(D / f"nodes_{tag}.parquet")

    res: dict = {"tag": tag}

    # ---------------- 0. reproduce the official pooled numbers ----------------
    w = crops.edge_tp + crops.edge_fp + crops.edge_fn
    pooled_adj = float((w * crops.adj_edge_jaccard).sum() / w.sum())
    micro_J = float(crops.edge_tp.sum() / w.sum())
    res["pooled_adj_edge_jaccard"] = pooled_adj
    res["micro_edge_jaccard"] = micro_J
    res["n_crops"] = int(len(crops))
    res["totals"] = {k: int(crops[k].sum()) for k in
                     ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn")}
    dt, dfp, dfn = (res["totals"][k] for k in ("division_tp", "division_fp", "division_fn"))
    res["division_jaccard"] = dt / (dt + dfp + dfn) if (dt + dfp + dfn) else float("nan")
    res["SCORE"] = pooled_adj + 0.1 * res["division_jaccard"]

    # ---------------- 1. exact shortfall decomposition ------------------------
    crops = crops.copy()
    crops["w"] = w
    crops["node_term"] = 0.1 * crops.total_node_ratio * crops.edge_tp
    crops["loss_units"] = crops.edge_fp + crops.edge_fn + crops.node_term
    # clipping check
    crops["clipped"] = crops.adj_edge_jaccard <= 0
    ident = float(1 - crops.loss_units.sum() / crops.w.sum())
    res["identity_check_pooled_adjJ"] = ident
    res["identity_error"] = abs(ident - pooled_adj)

    W = float(crops.w.sum())
    res["shortfall"] = {
        "total": float(crops.loss_units.sum() / W),
        "from_FP": float(crops.edge_fp.sum() / W),
        "from_FN": float(crops.edge_fn.sum() / W),
        "from_node_count_penalty": float(crops.node_term.sum() / W),
    }

    # FN split: both endpoints detected (linker's fault) vs endpoint missing (detector's fault)
    fn = gtedges[~gtedges.is_tp]
    both = fn.src_detected & fn.tgt_detected
    res["fn_split"] = {
        "n_fn": int(len(fn)),
        "both_endpoints_detected_unlinked": int(both.sum()),
        "src_undetected": int((~fn.src_detected & fn.tgt_detected).sum()),
        "tgt_undetected": int((fn.src_detected & ~fn.tgt_detected).sum()),
        "both_undetected": int((~fn.src_detected & ~fn.tgt_detected).sum()),
    }
    res["fn_split"]["frac_linker_fault"] = float(both.mean()) if len(fn) else float("nan")
    # detection ceiling: GT edges with both endpoints detected
    ceil = (gtedges.src_detected & gtedges.tgt_detected)
    res["detection_ceiling"] = float(ceil.mean())
    res["gt_edges"] = int(len(gtedges))

    # FP split: does the FP edge land on GT nodes at all?
    fp = edges[edges.pred_valid & ~edges.matched]
    res["fp_split"] = {
        "n_fp": int(len(fp)),
        "both_ends_matched_wrong_pair": int(((fp.gt_src != -1) & (fp.gt_tgt != -1)).sum()),
        "src_only_matched": int(((fp.gt_src != -1) & (fp.gt_tgt == -1)).sum()),
        "tgt_only_matched": int(((fp.gt_src == -1) & (fp.gt_tgt != -1)).sum()),
        "neither_matched_but_valid": int(((fp.gt_src == -1) & (fp.gt_tgt == -1)).sum()),
    }
    res["edges_emitted_total"] = int(len(edges))
    res["edges_free_ignored"] = int((~edges.pred_valid).sum())

    # FP mechanism: is the FP a SECOND edge from a source that already has its TP
    # ("spurious fork", pure surplus) or the source's only / wrong link ("substitution")?
    tp_src = set(map(tuple, edges.loc[edges.matched, ["dataset", "source_id"]].values))
    tp_tgt = set(map(tuple, edges.loc[edges.matched, ["dataset", "target_id"]].values))
    fp_src_has_tp = np.array([(d, s) in tp_src for d, s in zip(fp.dataset, fp.source_id)])
    fp_tgt_has_tp = np.array([(d, s) in tp_tgt for d, s in zip(fp.dataset, fp.target_id)])
    res["fp_mechanism"] = {
        "surplus_source_already_has_a_TP": int(fp_src_has_tp.sum()),
        "surplus_target_already_has_a_TP": int(fp_tgt_has_tp.sum()),
        "surplus_either": int((fp_src_has_tp | fp_tgt_has_tp).sum()),
        "substitution_neither": int((~fp_src_has_tp & ~fp_tgt_has_tp).sum()),
    }
    # out-degree of the emitting source in the emitted graph, for FP vs TP
    od = edges.groupby(["dataset", "source_id"]).size().rename("outdeg")
    e2 = edges.join(od, on=["dataset", "source_id"])
    res["outdeg_of_emitting_source"] = {
        "TP": e2.loc[e2.matched, "outdeg"].value_counts().head(4).to_dict(),
        "FP": e2.loc[e2.pred_valid & ~e2.matched, "outdeg"].value_counts().head(4).to_dict(),
        "free": e2.loc[~e2.pred_valid, "outdeg"].value_counts().head(4).to_dict(),
    }

    # FN mechanism: for a GT edge with BOTH endpoints detected, did the pipeline emit
    # any outgoing edge from the predicted node holding the GT source?
    p_of_gt = nodes[nodes.gt_id != -1].drop_duplicates(["dataset", "gt_id"]).set_index(
        ["dataset", "gt_id"]).node_id
    fnb = fn[both].copy()
    pid = p_of_gt.reindex(pd.MultiIndex.from_arrays([fnb.dataset, fnb.gt_source])).values
    emit_src = set(zip(edges.dataset.tolist(), edges.source_id.tolist()))
    has_out = np.array([(d, s) in emit_src for d, s in zip(fnb.dataset, pid)])         if len(fnb) else np.array([], dtype=bool)
    res["fn_mechanism_both_detected"] = {
        "n": int(len(fnb)),
        "source_emitted_some_edge_mislink": int(has_out.sum()) if len(fnb) else 0,
        "source_emitted_nothing_abstained": int((~has_out).sum()) if len(fnb) else 0,
    }

    # ---------------- 2. concentration ---------------------------------------
    res["concentration_loss"] = lorenz(crops.loss_units.clip(lower=0).values)
    res["concentration_weight"] = lorenz(crops.w.values)
    top = crops.sort_values("loss_units", ascending=False)
    res["top10_crops"] = top.head(10)[
        ["dataset", "edge_tp", "edge_fp", "edge_fn", "adj_edge_jaccard",
         "total_node_ratio", "w", "loss_units"]].to_dict("records")
    # counterfactual: what would pooled adjJ be if the worst decile were perfect?
    n10 = max(1, int(round(0.10 * len(crops))))
    worst = set(top.head(n10).dataset)
    c2 = crops.copy()
    c2.loc[c2.dataset.isin(worst), "adj_edge_jaccard"] = 1.0
    res["counterfactual_top_decile_perfect"] = float(
        (c2.w * c2.adj_edge_jaccard).sum() / c2.w.sum())
    # counterfactual: worst decile lifted to the median crop's adjJ
    med = float(crops.adj_edge_jaccard.median())
    c3 = crops.copy()
    c3.loc[c3.dataset.isin(worst), "adj_edge_jaccard"] = med
    res["counterfactual_top_decile_to_median"] = float(
        (c3.w * c3.adj_edge_jaccard).sum() / c3.w.sum())
    res["median_crop_adjJ"] = med

    # ---------------- 3. structure inside crops ------------------------------
    def binned(df: pd.DataFrame, col: str, bins, label: str, num, den):
        g = pd.cut(df[col], bins=bins, include_lowest=True)
        out = df.groupby(g, observed=False).apply(
            lambda d: pd.Series({"n": len(d), "rate": float(d[num].sum()) / max(1, len(d))})
        )
        return {label: {str(k): {"n": int(v["n"]), "rate": float(v["rate"])}
                        for k, v in out.iterrows()}}

    gtedges = gtedges.copy()
    gtedges["disp_um"] = np.sqrt(
        ((gtedges.s_z - gtedges.t_z) * SCALE[0]) ** 2
        + ((gtedges.s_y - gtedges.t_y) * SCALE[1]) ** 2
        + ((gtedges.s_x - gtedges.t_x) * SCALE[2]) ** 2)
    gtedges["miss"] = ~gtedges.is_tp

    # local GT density: number of GT nodes in the same frame within 15 um
    dens = []
    for (ds, t), g in gtnodes.groupby(["dataset", "t"]):
        p = np.c_[g.z * SCALE[0], g.y * SCALE[1], g.x * SCALE[2]]
        if len(p) == 1:
            d = np.array([0])
        else:
            from scipy.spatial import cKDTree
            d = cKDTree(p).query_ball_point(p, r=15.0, return_length=True) - 1
        dens.append(pd.DataFrame({"dataset": ds, "gt_id": g.gt_id.values, "dens15": d}))
    dens = pd.concat(dens, ignore_index=True)
    gtedges = gtedges.merge(dens.rename(columns={"gt_id": "gt_source", "dens15": "src_dens15"}),
                            on=["dataset", "gt_source"], how="left")

    # normalised time within crop
    tmax = gtedges.groupby("dataset").s_t.transform("max")
    gtedges["t_frac"] = gtedges.s_t / tmax.replace(0, np.nan)

    struct = {}
    struct.update(binned(gtedges, "disp_um", [0, 1, 2, 3, 4, 5, 7, 10, 1e9],
                         "miss_rate_by_displacement_um", "miss", None))
    struct.update(binned(gtedges, "src_dens15", [-0.1, 0, 1, 2, 3, 5, 8, 12, 1e9],
                         "miss_rate_by_local_gt_density_r15um", "miss", None))
    struct.update(binned(gtedges, "s_z", [-0.1, 8, 16, 24, 32, 40, 48, 56, 1e9],
                         "miss_rate_by_z_voxel", "miss", None))
    struct.update(binned(gtedges, "t_frac", [-0.01, .1, .25, .5, .75, .9, 1.01],
                         "miss_rate_by_time_fraction", "miss", None))
    struct.update(binned(gtedges, "src_outdeg", [-0.1, 1.1, 2.1, 1e9],
                         "miss_rate_by_gt_source_outdegree", "miss", None))
    res["gt_edge_structure"] = struct

    # FP structure
    edges = edges.copy()
    edges["disp_um"] = np.sqrt(
        ((edges.s_z - edges.t_z) * SCALE[0]) ** 2
        + ((edges.s_y - edges.t_y) * SCALE[1]) ** 2
        + ((edges.s_x - edges.t_x) * SCALE[2]) ** 2)
    sc = edges[edges.pred_valid].copy()
    sc["is_fp"] = ~sc.matched
    fs = {}
    fs.update(binned(sc, "disp_um", [0, 1, 2, 3, 4, 5, 7, 10, 1e9],
                     "fp_rate_by_displacement_um", "is_fp", None))
    fs.update(binned(sc, "s_z", [-0.1, 8, 16, 24, 32, 40, 48, 56, 1e9],
                     "fp_rate_by_z_voxel", "is_fp", None))
    res["pred_edge_structure"] = fs

    # ---------------- 4. per-crop pattern ------------------------------------
    gsum = gtnodes.groupby("dataset").agg(
        gt_nodes=("gt_id", "size"), gt_div=("out_degree", lambda s: int((s >= 2).sum())),
        gt_tmax=("t", "max"), det_rate=("detected", "mean")).reset_index()
    esum = gtedges.groupby("dataset").agg(gt_edges=("is_tp", "size"),
                                          mean_disp=("disp_um", "mean"),
                                          mean_dens=("src_dens15", "mean")).reset_index()
    nsum = nodes.groupby("dataset").agg(pred_nodes=("node_id", "size")).reset_index()
    cc = crops.merge(gsum, on="dataset").merge(esum, on="dataset").merge(nsum, on="dataset")
    cc["annot_density"] = cc.gt_nodes / cc.pred_nodes
    cc["gt_nodes_per_frame"] = cc.gt_nodes / (cc.gt_tmax + 1)
    cc.to_parquet(D / f"cropstats_{tag}.parquet")
    corr_cols = ["gt_nodes", "gt_div", "det_rate", "gt_edges", "mean_disp", "mean_dens",
                 "pred_nodes", "annot_density", "gt_nodes_per_frame", "total_node_ratio", "w"]
    res["percrop_spearman_vs_adjJ"] = {
        c: float(cc[[c, "adj_edge_jaccard"]].corr(method="spearman").iloc[0, 1])
        for c in corr_cols}
    res["percrop_adjJ_quantiles"] = {str(q): float(cc.adj_edge_jaccard.quantile(q))
                                     for q in (0, .1, .25, .5, .75, .9, 1)}
    if "family" not in cc:
        cc["family"] = cc.dataset.str.split("_").str[0]
    res["by_family"] = {
        fam: {"n": int(len(g)), "w": float(g.w.sum()),
              "pooled_adjJ": float((g.w * g.adj_edge_jaccard).sum() / g.w.sum()),
              "fp": int(g.edge_fp.sum()), "fn": int(g.edge_fn.sum()), "tp": int(g.edge_tp.sum())}
        for fam, g in cc.groupby("family")}

    out = json.dumps(res, indent=2, default=float)
    print(out)
    if args.json_out:
        Path(args.json_out).write_text(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
