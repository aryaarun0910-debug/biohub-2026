"""Aggregate assoc_preilp_survival crop payloads into the per-fold category-5 ledger row.

Two quantities the per-crop payloads cannot state on their own:

  1. The CONTROL outcome for the SAME sampled targets. A target whose node disappears in the
     treated arm may have disappeared in the control arm too, in which case its loss is
     baseline category 2, not something the perturbation caused. Reading the control graph
     for the sampled ids separates the two, so category 5 is never credited with mass that
     belongs to category 2.
  2. The per-fold pooled counts and their denominators, per movie (the crop prefix is the
     embryo under LOEO, so a fold IS one embryo direction) and per fold.

Read-only over the payloads and the persisted control graphs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))

HEARTBEAT = "ASSOC_PREILP_SURVIVAL_PANEL_COMPLETE"
BUCKETS = ("node_gone", "survived", "reverted", "reassigned", "no_parent")

# Survival against the GEOMETRIC HANDICAP the forced parent carries, in um. The relink cost is
# `motion + 0.05*raw - BONUS*prob`, so this is the axis the learned term competes on.
GAP_EDGES = (-1e9, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 1e9)


def _control_state(path: Path):
    frame = pl.read_parquet(path)
    edges = frame.filter(pl.col("row_type") == "edge")
    nodes = frame.filter(pl.col("row_type") == "node")
    parents = {int(t): int(s) for s, t in zip(edges["source_id"], edges["target_id"])}
    return parents, {int(n) for n in nodes["node_id"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--control-dir-f0", type=Path, required=True)
    ap.add_argument("--control-dir-f1", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--relink-bonus", type=float, required=True,
                    help="MOTION_RELINK_LEARNED_BONUS as deployed; stated, not defaulted")
    args = ap.parse_args()

    payloads = sorted(args.results_dir.glob("f*_*.json"))
    if not payloads:
        raise SystemExit(f"no survival payloads under {args.results_dir}")

    per_crop = []
    for path in payloads:
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("heartbeat") != "ASSOC_PREILP_SURVIVAL_COMPLETE":
            raise SystemExit(f"{path.name} carries no completion heartbeat")
        fold = str(d["fold"])
        ctl_dir = args.control_dir_f0 if fold == "0" else args.control_dir_f1
        ctl_path = ctl_dir / f"{d['crop']}.parquet"
        if not ctl_path.exists():
            raise SystemExit(f"control graph missing for {d['crop']}: {ctl_path}")
        parents, nodes = _control_state(ctl_path)

        # The same sampled targets, scored against the UNPERTURBED final graph.
        control_sample = {"node_gone": 0, "kept_deployed_parent": 0,
                          "other_parent": 0, "no_parent": 0}
        for rec in d["forced"]:
            t = rec["target"]
            if t not in nodes:
                control_sample["node_gone"] += 1
            elif t not in parents:
                control_sample["no_parent"] += 1
            elif parents[t] == rec["deployed_parent"]:
                control_sample["kept_deployed_parent"] += 1
            else:
                control_sample["other_parent"] += 1

        # The relink cost is `motion + 0.05*raw - BONUS*prob`, so the learned term can only
        # flip a pair when the two candidates are within about BONUS of each other in cost.
        # Raw distance is the observable proxy for that gap; this bins by it.
        band = {"in_band": 0, "in_band_survived": 0, "out_band": 0, "out_band_survived": 0}
        for rec in d["forced"]:
            if rec.get("outcome") == "node_gone":
                continue
            gap = rec["forced_dist_um"] - rec["deployed_dist_um"]
            key = "in_band" if gap <= args.relink_bonus else "out_band"
            band[key] += 1
            if rec.get("outcome") == "survived":
                band[key + "_survived"] += 1

        curve = {}
        for lo, hi in zip(GAP_EDGES[:-1], GAP_EDGES[1:]):
            key = f"({lo:g},{hi:g}]"
            n = s = 0
            for rec in d["forced"]:
                if rec.get("outcome") == "node_gone":
                    continue
                gap = rec["forced_dist_um"] - rec["deployed_dist_um"]
                if lo < gap <= hi:
                    n += 1
                    s += rec.get("outcome") == "survived"
            curve[key] = {"n": n, "survived": s}

        a = d["attribution"]
        per_crop.append({
            "bonus_band": band,
            "survival_by_geometric_handicap_um": curve,
            "crop": d["crop"],
            "movie_prefix": d["crop"].split("_")[0],
            "fold": fold,
            "n_forced": a["n_forced"],
            "buckets": a["buckets"],
            "n_forced_within_relink_gate": a["n_forced_within_relink_gate"],
            "buckets_within_relink_gate": a["buckets_within_relink_gate"],
            "control_sample": control_sample,
            "baseline_unperturbed": d["baseline_unperturbed"],
            "control_exact": d.get("control_exact"),
        })

    folds = {}
    for fold in sorted({row["fold"] for row in per_crop}):
        rows = [r for r in per_crop if r["fold"] == fold]
        agg = {k: sum(r["buckets"][k] for r in rows) for k in BUCKETS}
        gated = {k: sum(r["buckets_within_relink_gate"][k] for r in rows) for k in BUCKETS}
        ctl = {k: sum(r["control_sample"][k] for r in rows)
               for k in ("node_gone", "kept_deployed_parent", "other_parent", "no_parent")}
        band = {k: sum(r["bonus_band"][k] for r in rows)
                for k in ("in_band", "in_band_survived", "out_band", "out_band_survived")}
        base = {k: sum(r["baseline_unperturbed"][k] for r in rows)
                for k in ("n_preilp_parents", "node_gone", "same", "different", "no_parent")}
        n = sum(r["n_forced"] for r in rows)
        n_gated = sum(r["n_forced_within_relink_gate"] for r in rows)
        # Category 5 = a parent choice made upstream that the emitted graph does not carry,
        # for a target whose node SURVIVED. node_gone is category 2 and is excluded.
        survived_node = n - agg["node_gone"]
        overwritten = agg["reverted"] + agg["reassigned"] + agg["no_parent"]
        folds[fold] = {
            "movies": sorted({r["movie_prefix"] for r in rows}),
            "crops": len(rows),
            "n_forced": n,
            "treated_buckets": agg,
            "n_forced_within_relink_gate": n_gated,
            "treated_buckets_within_relink_gate": gated,
            "control_outcome_on_same_targets": ctl,
            "survival_by_geometric_handicap_um": {
                key: {
                    "n": sum(r["survival_by_geometric_handicap_um"][key]["n"] for r in rows),
                    "survived": sum(
                        r["survival_by_geometric_handicap_um"][key]["survived"] for r in rows),
                }
                for key in rows[0]["survival_by_geometric_handicap_um"]
            },
            "bonus_band": {
                **band,
                "relink_bonus_um_equivalent": args.relink_bonus,
                "in_band_survival": band["in_band_survived"] / band["in_band"]
                if band["in_band"] else None,
                "out_band_survival": band["out_band_survived"] / band["out_band"]
                if band["out_band"] else None,
            },
            "survival_rate_of_all_forced": agg["survived"] / n if n else None,
            "category_5_count": overwritten,
            "category_5_denominator_targets_whose_node_survived": survived_node,
            "category_5_rate": overwritten / survived_node if survived_node else None,
            "survival_rate_given_node_survived": (
                agg["survived"] / survived_node if survived_node else None),
            "survival_rate_within_relink_gate": (
                gated["survived"] / (n_gated - gated["node_gone"])
                if (n_gated - gated["node_gone"]) else None),
            "baseline_unperturbed_all_preilp_parents": {
                **base,
                "agreement_rate": base["same"] / base["n_preilp_parents"]
                if base["n_preilp_parents"] else None,
                "category_2_rate": base["node_gone"] / base["n_preilp_parents"]
                if base["n_preilp_parents"] else None,
            },
        }

    result = {
        "schema_version": 1,
        "heartbeat": HEARTBEAT,
        "definition": {
            "category_5": "a parent selected at pre-ILP candidate ranking that the EMITTED graph "
                          "does not carry, for a target whose node survived to emission",
            "excluded_from_category_5": "targets whose node is absent from the emitted graph - "
                                        "those are category 2 and are counted there",
            "order": "category 2 is tested first, so the two categories are disjoint",
        },
        "folds": folds,
        "per_crop": per_crop,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for fold, f in folds.items():
        print(f"fold {fold}: crops={f['crops']} n={f['n_forced']} "
              f"survived={f['treated_buckets']['survived']} "
              f"cat5={f['category_5_count']}/{f['category_5_denominator_targets_whose_node_survived']} "
              f"({f['category_5_rate']:.4f}) "
              f"survival|node_survived={f['survival_rate_given_node_survived']:.4f}")
    print(HEARTBEAT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
