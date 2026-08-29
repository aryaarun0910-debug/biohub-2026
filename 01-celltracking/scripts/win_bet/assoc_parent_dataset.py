r"""The parent-selection dataset and the FROZEN evaluation surface every ranker reports into.

WHY ONE SHARED SURFACE
----------------------
Three learned discriminators are being built in parallel - a calibrated linear/listwise model, a
tree ranker, and the known-contract HOCT head. If each brings its own evaluation, they cannot be
compared and the cycle produces three unfalsifiable claims instead of a capability ladder. So the
dataset construction, the label contract and the metrics live here, once, and every ranker is
scored by calling into this module.

THE TASK, STATED AS THE PIPELINE ACTUALLY POSES IT
--------------------------------------------------
This is NOT independent binary edge classification. ``FACT-0369`` established that the deployed
rule takes a softmax over the SOURCE axis and keeps pairs above 0.5, so at most one parent per
target survives - the model is deployed as a hard argmax with abstention. The learnable task is
therefore TARGET-WISE: for each node in frame t+1, choose its parent among the candidate sources
in frame t, or abstain. Every metric below is per target, not per edge.

THE LABEL CONTRACT, FROZEN BEFORE ANY TRAINING (PKT-0029 gate 2)
----------------------------------------------------------------
  1. GT nodes are matched to pre-ILP nodes ONCE, per frame, by the official one-to-one 7 um rule -
     the same matcher as FACT-0354/0355/0357, so a cell counted here is the same cell counted there.
  2. A target's TRUE PARENT is the pre-ILP node matched to the GT predecessor of the GT cell that
     target is matched to.
  3. A target gets label ABSTAIN when it has no matched GT cell, its GT cell has no predecessor
     (a track start), or the true parent is not among its candidates - the last being the
     FACT-0370 unreachable case, which no scorer can win and which is therefore scored separately
     rather than counted as a ranking failure.
  4. Ambiguity rules are FIXED here and not revisited after seeing results: ties in candidate score
     are broken by lower source index, and a GT cell with two predecessors (which the GT does not
     contain) would be dropped rather than guessed.

FEATURES ARE DELIBERATELY REPRESENTATION-FREE
---------------------------------------------
This baseline uses only what already exists without the trunk feature cache: the deployed model's
own probability, candidate rank and margin within the target, and geometry. That is the point -
``PKT-0029``'s learnability question is whether the EXISTING representation already separates true
parents. If a linear or tree model on these features converts true parents, the representation is
sufficient and HOCT's learned node context is not the bottleneck; if they fail, that justifies the
richer context. Neither answer is assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detpeak_curve import MAX_DISTANCE_UM, SCALE_UM, match_one_to_one_pairs  # noqa: E402

SCALE = np.asarray(SCALE_UM, dtype=np.float64)
FEATURES = [
    "prob",            # the deployed model's own softmax score for this pair
    "rank",            # 1-based rank of this candidate within its target
    "margin_to_best",  # prob(best for this target) - prob(this)
    "n_candidates",    # how many parents this target is offered
    "dist_um",         # 3-D displacement in microns
    "dz_um", "dy_um", "dx_um",
    "src_out_degree",  # how many targets this source is a candidate for
]


def build_crop(crop: str, gt_geff: Path, pre: pl.DataFrame,
               sidecar: Path | None, floor: float, rank_cap: int) -> list[dict]:
    """One row per (target, candidate) pair, with the frozen label contract applied."""
    from biotrack.metric import load_graph

    nodes = pre.filter(pl.col("row_type") == "node").sort("node_id")
    if nodes.height == 0:
        return []
    node_id = nodes["node_id"].to_numpy().astype(np.int64)
    node_t = nodes["t"].to_numpy().astype(np.int64)
    node_zyx = nodes.select(["z", "y", "x"]).to_numpy().astype(np.float64)
    pos = {int(v): i for i, v in enumerate(node_id)}

    # Candidate set: the richer sidecar surface when available, else the deployed edges.
    if sidecar is not None and sidecar.exists():
        with np.load(sidecar, allow_pickle=False) as z:
            src = z["source_id"].astype(np.int64)
            tgt = z["target_id"].astype(np.int64)
            prob = z["edge_prob"].astype(np.float64)
        keep = prob > floor
        src, tgt, prob = src[keep], tgt[keep], prob[keep]
    else:
        e = pre.filter(pl.col("row_type") == "edge")
        src = e["source_id"].to_numpy().astype(np.int64)
        tgt = e["target_id"].to_numpy().astype(np.int64)
        prob = e["edge_prob"].to_numpy().astype(np.float64)

    order = np.lexsort((-prob, tgt))
    ranks = np.empty(len(src), dtype=np.int64)
    best = {}
    cur, seen = None, 0
    for p in order:
        if tgt[p] != cur:
            cur, seen = tgt[p], 0
            best[int(cur)] = float(prob[p])
        seen += 1
        ranks[p] = seen
    keep = ranks <= rank_cap
    src, tgt, prob, ranks = src[keep], tgt[keep], prob[keep], ranks[keep]

    n_cand: dict[int, int] = {}
    out_deg: dict[int, int] = {}
    for s, t in zip(src.tolist(), tgt.tolist()):
        n_cand[t] = n_cand.get(t, 0) + 1
        out_deg[s] = out_deg.get(s, 0) + 1

    # --- label contract -----------------------------------------------------------------
    gt = load_graph(gt_geff)
    gtn = gt.node_attrs().to_pandas()
    gt_ids = gtn["node_id"].to_numpy().astype(np.int64)
    gt_row = {int(v): i for i, v in enumerate(gt_ids)}
    gtc = gtn[["t", "z", "y", "x"]].to_numpy().astype(np.float64)
    gte = gt.edge_attrs().to_pandas()
    predecessor = {int(t): int(s) for s, t in zip(gte["source_id"], gte["target_id"])}

    matched: dict[int, int] = {}                      # gt row -> pre-ILP node id
    for frame in np.unique(gtc[:, 0]).astype(np.int64):
        gm = np.nonzero(gtc[:, 0] == frame)[0]
        pm = np.nonzero(node_t == frame)[0]
        if not len(pm):
            continue
        for g, p, _d in match_one_to_one_pairs(
            node_zyx[pm] * SCALE, gtc[gm, 1:] * SCALE, MAX_DISTANCE_UM
        ):
            matched[int(gm[g])] = int(node_id[pm][p])
    node_to_gt = {v: k for k, v in matched.items()}

    true_parent: dict[int, int] = {}                  # target node id -> true parent node id
    for gt_row_idx, pre_id in matched.items():
        gt_node = int(gt_ids[gt_row_idx])
        pred_gt = predecessor.get(gt_node)
        if pred_gt is None:
            continue
        pred_row = gt_row.get(pred_gt)
        if pred_row is None or pred_row not in matched:
            continue
        true_parent[pre_id] = matched[pred_row]

    rows = []
    for s, t, p, r in zip(src.tolist(), tgt.tolist(), prob.tolist(), ranks.tolist()):
        si, ti = pos.get(s), pos.get(t)
        if si is None or ti is None:
            continue
        d = (node_zyx[ti] - node_zyx[si]) * SCALE
        rows.append({
            "crop": crop, "target": t, "source": s,
            "prob": p, "rank": r,
            "margin_to_best": best.get(t, p) - p,
            "n_candidates": n_cand.get(t, 1),
            "dist_um": float(np.linalg.norm(d)),
            "dz_um": float(d[0]), "dy_um": float(d[1]), "dx_um": float(d[2]),
            "src_out_degree": out_deg.get(s, 1),
            "is_true_parent": int(true_parent.get(t, -1) == s),
            "target_has_true_parent": int(t in true_parent),
            "true_parent_is_candidate": int(
                t in true_parent and any(
                    ss == true_parent[t] for ss, tt in zip(src.tolist(), tgt.tolist()) if tt == t
                )
            ),
            "target_matched_gt": int(t in node_to_gt),
        })
    return rows


def evaluate(table: pl.DataFrame, score_col: str) -> dict:
    """THE FROZEN SURFACE. Target-wise, and identical for every ranker.

    `decidable` targets are those whose true parent is actually among the candidates - the only
    ones a scorer can win. Targets whose true parent was never offered are reported separately
    (FACT-0370) rather than charged to the model as ranking failures.
    """
    decidable = table.filter(pl.col("true_parent_is_candidate") == 1)
    if decidable.height == 0:
        return {"decidable_targets": 0}

    top1, margins = 0, []
    # CONTESTED vs SINGLE-CANDIDATE, split here rather than by each caller (FACT-0381). A target
    # offered one candidate scores top-1 = 1.0 by construction, not by skill, and on fold 0 that is
    # 78.6% of the surface - so a pooled top-1 is dominated by decisions no model can get wrong and
    # a real ranking failure barely moves it. The bar every ranker is held to is TWO-SIDED: beat the
    # deployed top-1 on CONTESTED targets while not losing the single-candidate ones, which a model
    # scoring the whole surface CAN lose by re-ranking a lone candidate below an abstain threshold.
    contested_hits, contested_n, single_hits, single_n = 0, 0, 0, 0
    per_target: dict[tuple[str, int], int] = {}
    for _key, group in decidable.group_by("crop", "target"):
        s = group[score_col].to_numpy()
        y = group["is_true_parent"].to_numpy()
        order = np.argsort(-s, kind="stable")
        chosen = order[0]
        hit = bool(y[chosen] == 1)
        top1 += int(hit)
        per_target[(str(group["crop"][0]), int(group["target"][0]))] = int(hit)
        if len(s) > 1:
            best, second = s[order[0]], s[order[1]]
            margins.append(float(best - second) * (1.0 if hit else -1.0))
            contested_n += 1
            contested_hits += int(hit)
        else:
            single_n += 1
            single_hits += int(hit)
    n_targets = decidable.select(["crop", "target"]).unique().height
    unreachable = table.filter(
        (pl.col("target_has_true_parent") == 1) & (pl.col("true_parent_is_candidate") == 0)
    ).select(["crop", "target"]).unique().height
    return {
        "decidable_targets": n_targets,
        "parent_top1": top1 / max(n_targets, 1),
        "true_parent_margin_mean": float(np.mean(margins)) if margins else None,
        "true_parent_margin_median": float(np.median(margins)) if margins else None,
        "targets_true_parent_not_offered": unreachable,
        "contested": {
            "n": contested_n,
            "share": contested_n / max(n_targets, 1),
            "top1": contested_hits / max(contested_n, 1) if contested_n else None,
            "errors": contested_n - contested_hits,
        },
        "single_candidate": {
            "n": single_n,
            "share": single_n / max(n_targets, 1),
            "top1": single_hits / max(single_n, 1) if single_n else None,
            "note": "1.0 by construction for any model that ranks; not skill",
        },
        # A fold whose contested population is empty CANNOT falsify a ranker - top-1 is 1.0 there
        # for every model. FACT-0381 measured exactly that on fold 1, and a metric that cannot fail
        # is not evidence, so the surface says so itself rather than leaving a caller to notice.
        "degenerate_for_ranking": contested_n == 0,
        "per_target_correct": per_target,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preilp", type=Path, required=True)
    ap.add_argument("--ecb-dir", type=Path)
    ap.add_argument("--gt-dir", type=Path, default=ROOT / "data" / "train")
    ap.add_argument("--floor", type=float, default=0.1)
    ap.add_argument("--rank-cap", type=int, default=4)
    ap.add_argument("--max-crops", type=int)
    ap.add_argument("--out-table", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pre = pl.read_parquet(args.preilp)
    crops = sorted(pre["dataset"].unique().to_list())
    if args.max_crops:
        crops = crops[: args.max_crops]

    rows: list[dict] = []
    for i, crop in enumerate(crops, 1):
        gt_geff = args.gt_dir / f"{crop}.geff"
        if not gt_geff.exists():
            continue
        side = (args.ecb_dir / f"{crop}.npz") if args.ecb_dir else None
        rows.extend(build_crop(crop, gt_geff, pre.filter(pl.col("dataset") == crop),
                               side, args.floor, args.rank_cap))
        print(f"  [{i}/{len(crops)}] {crop} rows={len(rows):,}", flush=True)

    table = pl.DataFrame(rows)
    args.out_table.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(args.out_table)

    baseline = evaluate(table, "prob")   # the DEPLOYED scorer on the same surface
    # The per-target correctness map is for in-process comparison (assoc_report.parent_conversions);
    # its tuple keys are not JSON, and dumping 19k rows into a summary would bury the summary.
    baseline.pop("per_target_correct", None)
    result = {
        "schema_version": 1,
        "preilp": str(args.preilp),
        "candidate_surface": {"floor": args.floor, "rank_cap": args.rank_cap,
                              "from_sidecar": args.ecb_dir is not None},
        "rows": table.height,
        "features": FEATURES,
        "deployed_baseline": baseline,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    b = baseline
    contested = b.get("contested", {})
    single = b.get("single_candidate", {})
    print(
        f"\nASSOC_PARENT_DATASET rows={table.height:,} crops={len(crops)}\n"
        f"  decidable targets            {b.get('decidable_targets', 0):,}\n"
        f"  DEPLOYED parent top-1        {b.get('parent_top1')}\n"
        f"  CONTESTED targets            {contested.get('n', 0):,} "
        f"({contested.get('share', 0):.1%})  top-1 {contested.get('top1')}  "
        f"errors {contested.get('errors', 0):,}\n"
        f"  single-candidate targets     {single.get('n', 0):,} "
        f"({single.get('share', 0):.1%})  top-1 1.0 by construction\n"
        f"  true-parent margin (median)  {b.get('true_parent_margin_median')}\n"
        f"  true parent never offered    {b.get('targets_true_parent_not_offered', 0):,}\n"
        f"  DEGENERATE FOR RANKING       {b.get('degenerate_for_ranking')}"
    )
    if b.get("degenerate_for_ranking"):
        print(
            "  ^ no contested targets on this surface: top-1 is 1.0 for ANY model here, so this "
            "fold cannot falsify a ranker and no ranker claim may be made on it (FACT-0381).",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
