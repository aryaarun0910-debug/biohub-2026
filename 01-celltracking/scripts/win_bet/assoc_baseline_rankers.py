r"""LEARNABILITY BASELINES for parent discrimination on the FROZEN target-wise surface (PKT-0031).

WHAT THIS ANSWERS
-----------------
``FACT-0381`` sized the parent-choice task honestly: the learnable headroom on the honest fold is
691 wrong parent choices among 4,157 CONTESTED targets, not "87% of the gap". This module asks the
one question that decides whether richer node context is justified - does the EXISTING
representation-free feature set already separate true parents? It fits the two cheapest
discriminators (a calibrated linear model and a gradient-boosted tree) on the nine features already
in ``assoc_parent_dataset.FEATURES`` and scores them through the frozen surface and nothing else.

A CLEAN FAILURE IS A RESULT. If neither model beats the deployed argmax on its own surface, the
representation - not the head - is the constraint, which is precisely what the HOCT lane needs to
know. Nothing here goes hunting for a feature that rescues the number.

WHY EVERY SPLIT IS GROUPED BY CROP
----------------------------------
A random row split leaks twice over. Candidates for one target share a target (so the positive and
its own negatives would straddle the split), and targets in one crop share a movie, so neighbouring
frames of one embryo are not independent samples. Every split here is by CROP and the guard is
mechanical: ``_assert_disjoint`` fails the run if any crop appears on both sides of a fold.

WHAT IS PRE-REGISTERED, AND WHY THAT MATTERS MORE THAN THE HYPERPARAMETERS
-------------------------------------------------------------------------
Hyperparameters are library defaults, fixed in ``LINEAR_KW``/``TREE_KW``, and no search is run. The
point is not that defaults are optimal - it is that a number produced by a search over the same
targets it is reported on is not held out, and this task has 691 decidable errors in total, which a
search would overfit trivially. Both training populations (all decidable rows; contested rows only)
and all five feature sets are declared here and ALL are reported, so no arm can be selected after
the fact. Early stopping is switched OFF on the tree because sklearn's default would carve an
internal RANDOM validation split out of a training fold - the exact leak this module exists to
avoid, one level down.

WHAT THE TWO-SIDED BAR REALLY COSTS (read before interpreting a retention number)
--------------------------------------------------------------------------------
The frozen ``evaluate`` takes an argmax with NO abstention, so a single-candidate target is correct
for ANY score column - including a constant. Single-candidate retention is therefore 1.0 by
arithmetic for every model here, and reporting it as though it were earned would be false progress.
It can only be LOST by a model that adds an abstain threshold, which none of these do. Side (b) of
the bar is live for any future ranker that abstains; on these models it is untestable, and this
module says so rather than banking it.

Likewise a monotone recalibration cannot change a within-target argmax, so calibration is reported
here as a property of the fitted probabilities, never as a source of top-1 movement.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(Path(__file__).resolve().parent))

from assoc_parent_dataset import FEATURES, evaluate  # noqa: E402
from assoc_report import parent_conversions  # noqa: E402

# FACT-0381, restated here ONLY as a fail-closed guard. The packet's method is explicit: if the
# harness does not reproduce the deployed baseline exactly, stop - the surface is not what the
# registry says it is and nothing downstream would be interpretable. This is the one place a
# literal is correct, because an assertion cannot cite an id.
EXPECTED_BASELINE = {
    "decidable_targets": 19444,
    "contested_n": 4157,
    "contested_top1": 0.8337743565070964,
    "contested_errors": 691,
    "single_n": 15287,
    "overall_top1": 0.9644620448467394,
}

SEED = 20260829

# Pre-registered feature sets. `prob_only` must reproduce the deployed ordering for any monotone
# model and is the self-check that the learning harness is wired to the same decision.
FEATURE_SETS: dict[str, list[str]] = {
    "prob_only": ["prob"],
    "score_context": ["prob", "rank", "margin_to_best", "n_candidates"],
    "score_context_geom": ["prob", "rank", "margin_to_best", "n_candidates",
                           "dist_um", "dz_um", "dy_um", "dx_um"],
    "all_nine": list(FEATURES),
    "geometry_only": ["dist_um", "dz_um", "dy_um", "dx_um"],
}

LINEAR_KW = dict(C=1.0, max_iter=5000, solver="lbfgs")
TREE_KW = dict(max_iter=100, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20,
               l2_regularization=0.0, early_stopping=False, random_state=SEED)


def _make(kind: str):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "linear":
        return Pipeline([("scale", StandardScaler()),
                         ("lr", LogisticRegression(**LINEAR_KW))])
    if kind == "tree":
        return HistGradientBoostingClassifier(**TREE_KW)
    raise ValueError(f"unknown model kind {kind!r}")


def _assert_disjoint(crops: np.ndarray, tr: np.ndarray, te: np.ndarray, tag: str) -> None:
    """Fail closed. A crop on both sides of a fold is the leak this whole design exists to stop."""
    overlap = set(crops[tr]) & set(crops[te])
    if overlap:
        raise AssertionError(f"{tag}: crop leak across the split - {sorted(overlap)[:5]}")
    if len(te) == 0 or len(tr) == 0:
        raise AssertionError(f"{tag}: degenerate split ({len(tr)} train, {len(te)} test)")


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    """Brier, log loss and equal-count ECE. Monotone in nothing the argmax can see - by design."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    order = np.argsort(p, kind="stable")
    ece, n = 0.0, len(p)
    for chunk in np.array_split(order, bins):
        if not len(chunk):
            continue
        ece += (len(chunk) / n) * abs(float(p[chunk].mean()) - float(y[chunk].mean()))
    return {"brier": brier, "log_loss": logloss, "ece_10bin": float(ece),
            "mean_predicted": float(p.mean()), "base_rate": float(y.mean())}


def _mcnemar_exact(gained: int, lost: int) -> float:
    """Two-sided exact McNemar on the discordant targets. Paired, which the design already is."""
    from scipy.stats import binomtest

    if gained + lost == 0:
        return 1.0
    return float(binomtest(gained, gained + lost, 0.5, alternative="two-sided").pvalue)


def crop_paired_bootstrap(before: dict, after: dict, keys: set,
                          draws: int = 2000, seed: int = 20260829) -> dict:
    """Paired-over-CROPS interval on the contested top-1 delta. The unit is the crop, not the target.

    A p-value on discordant targets says whether a coin is fair; it does not say whether the gain
    would survive a different set of embryos, and the crop is the level at which these samples are
    actually independent. ``assoc_report.verdict`` refuses to promote anything whose paired interval
    is not favourable, so a ranker reported without one cannot be judged by the standard it will
    eventually be held to. Resampling is with replacement over crops, exactly as
    ``assoc_report.paired_bootstrap_score`` resamples crops for the score.
    """
    per_crop: dict[str, list[int]] = {}
    for k in keys:
        row = per_crop.setdefault(k[0], [0, 0, 0])
        row[0] += 1
        row[1] += int(before[k])
        row[2] += int(after[k])
    crops = sorted(per_crop)
    n = np.array([per_crop[c][0] for c in crops], dtype=np.float64)
    hb = np.array([per_crop[c][1] for c in crops], dtype=np.float64)
    ha = np.array([per_crop[c][2] for c in crops], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(crops), (draws, len(crops)))
    tot = n[idx].sum(axis=1)
    deltas = (ha[idx].sum(axis=1) - hb[idx].sum(axis=1)) / np.where(tot > 0, tot, np.nan)
    lo, hi = float(np.nanpercentile(deltas, 2.5)), float(np.nanpercentile(deltas, 97.5))
    return {
        "unit": "crop", "n_crops_with_contested": len(crops),
        "mean": float(np.nanmean(deltas)), "ci95": [lo, hi],
        "draws": draws, "seed": seed,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "favourable": bool(lo > 0),
    }


def scope_check(dec: pl.DataFrame) -> dict:
    """WHERE THE CONTESTED POPULATION COMES FROM, AND WHO CAN REACH IT. Read before quoting a gain.

    The 4,157 contested targets are a property of the P30 ACQUISITION surface (floor 0.1), not of
    the deployed pipeline. Sweeping the floor back up shows the population evaporating, and at the
    deployed 0.5 there is at most one candidate per target - fold 0 is then exactly as degenerate
    for ranking as FACT-0381 found fold 1 to be. So this function also asks where the deployed
    errors sit: if every one of them has its true parent below the deployed threshold, then no
    re-ranking of today's DEPLOYED surface can win a single target, and a contested top-1 gain is
    evidence about the representation's separability rather than a score gain in waiting.
    """
    rows = []
    for floor in (0.1, 0.2, 0.3, 0.4, 0.5):
        s = dec.filter(pl.col("prob") > floor)
        alive = s.filter(pl.col("is_true_parent") == 1).select(["crop", "target"]).unique()
        g = (s.join(alive, on=["crop", "target"], how="semi")
              .group_by(["crop", "target"]).agg(pl.len().alias("n")))
        rows.append({"floor": floor, "decidable": g.height,
                     "contested": g.filter(pl.col("n") > 1).height,
                     "max_candidates_per_target": int(g["n"].max())})

    counts = dec.group_by(["crop", "target"]).agg(pl.len().alias("n"))
    cont = dec.join(counts.filter(pl.col("n") > 1).select(["crop", "target"]),
                    on=["crop", "target"], how="semi")
    best = cont.group_by(["crop", "target"]).agg(pl.col("prob").max().alias("pmax"))
    true = cont.filter(pl.col("is_true_parent") == 1).select(
        ["crop", "target", pl.col("prob").alias("ptrue")])
    err = best.join(true, on=["crop", "target"], how="inner").filter(pl.col("ptrue") < pl.col("pmax"))
    pt, pm = err["ptrue"].to_numpy(), err["pmax"].to_numpy()
    deployed_floor = 0.5
    return {
        "floor_sweep": rows,
        "deployed_floor": deployed_floor,
        "contested_errors": {
            "n": err.height,
            "true_parent_prob": {"median": float(np.median(pt)), "max": float(pt.max()),
                                 "share_at_or_below_deployed_floor": float((pt <= deployed_floor).mean())},
            "winning_wrong_prob": {"median": float(np.median(pm)), "min": float(pm.min()),
                                   "n_above_deployed_floor": int((pm > deployed_floor).sum()),
                                   "n_at_or_below": int((pm <= deployed_floor).sum())},
        },
    }


def run_arm(dec: pl.DataFrame, fit_mask: np.ndarray, feat: list[str],
            kind: str, splitter, tag: str, baseline_map: dict, contested_keys: set) -> dict:
    """One (training population x feature set x model) cell, scored only through the frozen surface."""
    x_all = dec.select(feat).to_numpy().astype(np.float64)
    y_all = dec["is_true_parent"].to_numpy().astype(np.int64)
    crops_all = dec["crop"].to_numpy()

    # Train on the declared population; PREDICT out-of-fold for every decidable row, because the
    # frozen surface needs a score on all of them and a training-set prediction is not held out.
    out = np.full(len(y_all), np.nan)
    for i, (tr, te) in enumerate(splitter.split(x_all, y_all, groups=crops_all)):
        _assert_disjoint(crops_all, tr, te, f"{tag}/fold{i}")
        tr = tr[fit_mask[tr]]
        if len(tr) == 0 or len(np.unique(y_all[tr])) < 2:
            raise AssertionError(f"{tag}/fold{i}: training population is empty or one-class")
        model = _make(kind)
        model.fit(x_all[tr], y_all[tr])
        out[te] = model.predict_proba(x_all[te])[:, 1]
    if np.isnan(out).any():
        raise AssertionError(f"{tag}: rows without an OOF score")

    # `evaluate` filters to the decidable rows itself, so scoring the decidable frame is the SAME
    # surface, not a cheaper approximation of it - and main() proves that equivalence on the
    # deployed column before any arm runs rather than asserting it here in prose. The only field
    # that differs is `targets_true_parent_not_offered`, which belongs to the fold and not to a
    # model; it is reported once from the full-table baseline and never per arm. Joining the OOF
    # score back onto all 2.96M candidate rows 24 times over is what made the first run thrash.
    col = f"oof_{tag}"
    res = evaluate(dec.with_columns(pl.Series(col, out)), col)
    res.pop("targets_true_parent_not_offered", None)
    per_target = res.pop("per_target_correct")

    # Variation guard: a constant score silently degenerates the argmax into row order.
    contested_rows = dec.select(["crop", "target"]).to_numpy()
    is_cont = np.array([(c, int(t)) in contested_keys for c, t in contested_rows])
    if float(np.std(out[is_cont])) == 0.0:
        raise AssertionError(f"{tag}: constant score on contested rows - argmax is row order, not skill")

    conv_all = parent_conversions(baseline_map, per_target)
    conv_cont = parent_conversions(
        {k: v for k, v in baseline_map.items() if k in contested_keys},
        {k: v for k, v in per_target.items() if k in contested_keys},
    )
    conv_cont["mcnemar_exact_p"] = _mcnemar_exact(conv_cont["gained"], conv_cont["lost"])
    conv_cont["crop_paired_bootstrap"] = crop_paired_bootstrap(baseline_map, per_target,
                                                               contested_keys)
    return {
        "tag": tag, "model": kind, "features": feat, "n_features": len(feat),
        "surface": {k: v for k, v in res.items()},
        "conversions_all_decidable": conv_all,
        "conversions_contested": conv_cont,
        "calibration_oof": calibration(y_all, out),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--table", type=Path, default=Path("C:/temp/assoc/f0.parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("C:/temp/assoc_baselines"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--loco", action="store_true", help="also run leave-one-crop-out on all_nine")
    args = ap.parse_args()

    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

    full = pl.read_parquet(args.table)

    # ---- STAGE 0: the surface must be what the registry says it is, or nothing else is readable.
    base = evaluate(full, "prob")
    baseline_map = base.pop("per_target_correct")
    got = {
        "decidable_targets": base["decidable_targets"],
        "contested_n": base["contested"]["n"],
        "contested_top1": base["contested"]["top1"],
        "contested_errors": base["contested"]["errors"],
        "single_n": base["single_candidate"]["n"],
        "overall_top1": base["parent_top1"],
    }
    mismatch = {k: (EXPECTED_BASELINE[k], got[k]) for k in EXPECTED_BASELINE
                if EXPECTED_BASELINE[k] != got[k]}
    if mismatch:
        print("SELF_CHECK_FAILED - the frozen surface does not reproduce FACT-0381:", flush=True)
        for k, (want, have) in mismatch.items():
            print(f"  {k}: registry {want!r}  harness {have!r}", flush=True)
        raise SystemExit(2)
    if base["degenerate_for_ranking"]:
        raise SystemExit("this fold is DEGENERATE for ranking (FACT-0381) - no claim may be made")
    print(f"SELF_CHECK_OK decidable={got['decidable_targets']:,} "
          f"contested={got['contested_n']:,} top1={got['contested_top1']!r} "
          f"errors={got['contested_errors']:,}", flush=True)

    dec = full.filter(pl.col("true_parent_is_candidate") == 1)
    counts = dec.group_by(["crop", "target"]).agg(pl.len().alias("n_dec"))
    dec = dec.join(counts, on=["crop", "target"], how="left")
    contested_keys = {(r[0], int(r[1])) for r in
                      counts.filter(pl.col("n_dec") > 1).select(["crop", "target"]).rows()}
    if len(contested_keys) != EXPECTED_BASELINE["contested_n"]:
        raise AssertionError(f"contested key set is {len(contested_keys)}, not the frozen count")

    # Every arm is scored on `dec` rather than the full candidate table. PROVE that is the same
    # surface instead of assuming it: the deployed column must give identical per-target verdicts
    # and identical headline figures on both. A silently different surface would make every arm
    # below incomparable to the baseline they are measured against.
    sub = evaluate(dec, "prob")
    sub_map = sub.pop("per_target_correct")
    if sub_map != baseline_map:
        raise AssertionError("decidable-subset surface disagrees with the full table per target")
    for path, want in (("decidable_targets", base["decidable_targets"]),
                       ("parent_top1", base["parent_top1"]),
                       ("true_parent_margin_median", base["true_parent_margin_median"])):
        if sub[path] != want:
            raise AssertionError(f"subset surface differs on {path}: {sub[path]!r} vs {want!r}")
    if sub["contested"] != base["contested"] or sub["single_candidate"] != base["single_candidate"]:
        raise AssertionError("subset surface differs on the contested/single split")
    print("SUBSET_SURFACE_IDENTICAL", flush=True)
    del full

    scope = scope_check(dec)
    sweep = {r["floor"]: r for r in scope["floor_sweep"]}
    print(f"SCOPE floor 0.1 contested={sweep[0.1]['contested']:,}  "
          f"deployed floor 0.5 contested={sweep[0.5]['contested']:,} "
          f"(max {sweep[0.5]['max_candidates_per_target']} candidate/target)  "
          f"contested errors with true parent at or below the deployed floor: "
          f"{scope['contested_errors']['true_parent_prob']['share_at_or_below_deployed_floor']:.4f}",
          flush=True)

    populations = {
        "alldec": np.ones(dec.height, dtype=bool),                       # PRIMARY
        "contested": (dec["n_dec"].to_numpy() > 1),                      # declared sensitivity
    }

    results = []
    gkf = GroupKFold(n_splits=args.folds)
    for pop_name, mask in populations.items():
        for fs_name, feat in FEATURE_SETS.items():
            for kind in ("linear", "tree"):
                tag = f"{pop_name}.{fs_name}.{kind}"
                print(f"  fitting {tag} ...", flush=True)
                results.append(
                    run_arm(dec, mask, feat, kind, gkf, tag, baseline_map, contested_keys)
                    | {"population": pop_name, "feature_set": fs_name, "cv": f"GroupKFold{args.folds}"}
                )

    if args.loco:
        logo = LeaveOneGroupOut()
        for kind in ("linear", "tree"):
            tag = f"alldec.all_nine.{kind}.LOCO"
            print(f"  fitting {tag} ...", flush=True)
            results.append(
                run_arm(dec, populations["alldec"], FEATURE_SETS["all_nine"], kind, logo,
                        tag, baseline_map, contested_keys)
                | {"population": "alldec", "feature_set": "all_nine", "cv": "LeaveOneCropOut"}
            )

    # Pin the surface this run was scored against. The tie-break in `evaluate` changed under an
    # in-flight run on 2026-08-29 (row order -> lower source index, the frozen contract's rule 4),
    # which is latent for the deployed probability and LIVE for a tree emitting identical leaf
    # scores. A result that does not say which surface produced it cannot be told apart from one
    # that predates the fix, so the digest travels with the numbers.
    import hashlib

    surface = {
        name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()[:16]
        for name in ("assoc_parent_dataset.py", "assoc_report.py", "assoc_baseline_rankers.py")
    }

    payload = {
        "schema_version": 1,
        "heartbeat": "ASSOC_BASELINE_RANKERS_COMPLETE",
        "table": str(args.table),
        "surface_sha256_16": surface,
        "seed": SEED,
        "preregistered": {"linear": LINEAR_KW, "tree": {k: str(v) for k, v in TREE_KW.items()},
                          "feature_sets": FEATURE_SETS,
                          "populations": list(populations),
                          "search": "none - library defaults, fixed before any result was seen"},
        "deployed_baseline": base,
        "scope": scope,
        "arms": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "baselines_f0.json"
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    bar = EXPECTED_BASELINE["contested_top1"]
    # gained / lost / churn are printed BESIDE net, never net alone: a 10-gained-10-lost wash is
    # churn 20 and must read as churn 20, which is the whole reason parent_conversions exists.
    print(f"\n{'arm':46s} {'contested':>10s} {'d vs bar':>9s} {'single':>7s} "
          f"{'gain':>5s} {'lost':>5s} {'net':>5s} {'churn':>6s} {'McNemar p':>10s} "
          f"{'crop-paired ci95':>26s} {'fav':>4s}")
    for r in results:
        c = r["surface"]["contested"]
        s = r["surface"]["single_candidate"]
        k = r["conversions_contested"]
        b = k["crop_paired_bootstrap"]
        print(f"{r['tag']:46s} {c['top1']:10.4f} {c['top1'] - bar:+9.4f} "
              f"{s['top1']:7.4f} {k['gained']:5d} {k['lost']:5d} {k['net']:+5d} {k['churn']:6d} "
              f"{k['mcnemar_exact_p']:10.4f} "
              f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]".rjust(26)
              + f" {str(b['favourable']):>5s}")
    print(f"\nASSOC_BASELINE_RANKERS_COMPLETE -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
