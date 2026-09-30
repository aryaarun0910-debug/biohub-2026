r"""WHY DID GATE A RETURN EXACTLY ZERO? - the fold-transfer pathology (PKT-0038 / LEVER-0041).

THE OBJECT OF THE INVESTIGATION
-------------------------------
``FACT-0413`` measured +0.0538 contested top-1 on the fold-1 discovery surface with a favourable
crop-paired interval and a MEASURED NEGATIVE binning floor. ``FACT-0414`` then measured the fold-0
gate twice and both refused: GATE_B (weight transfer) at -0.0166 with an interval excluding zero on
the WRONG side, and GATE_A (the identical recipe REFITTED out of fold on fold 0's own crop-grouped
splits) at delta EXACTLY 0.000000 - gained 36, lost 36, churn 72, contested top-1 equal to the
deployed 0.8337743565070964 to all sixteen digits.

NEITHER AVAILABLE READING EXPLAINS GATE A. "The tournament fitted the discovery fold" (falsifier g)
predicts a SMALLER POSITIVE delta on a refit, not zero. "The folds are incomparable substrates"
(different acquisition checkpoints per fold, verified at source in FACT-0414) is removed BY
CONSTRUCTION when the recipe is refitted on fold 0's own data. A refit that reshuffles 72 targets
and nets precisely nothing is the quantity to explain, and until it is explained we do not know
whether the fold-1 gain is capability or a selection artifact.

FOLD 0 IS A DIAGNOSTIC SURFACE HERE, NEVER A SELECTION SURFACE
--------------------------------------------------------------
Nothing in this module chooses an arm, a threshold, a feature set or a hyperparameter by a fold-0
number. The feature sets are the preregistered ones (``assoc_baseline_rankers.FEATURE_SETS`` and
``assoc_tournament.BIDIR``); the recipe is the frozen A4.bidir.tree; the splitter, seed and
tie-break are the harness's. The 2x2's fold-0-fitted cell is fitted on ALL of fold 0 and TESTED ON
FOLD 1, which is the exact mirror of GATE_B and not a selection.

WHAT IT MEASURES
----------------
  matrix   The 2x2 train/test matrix, crop-grouped OOF throughout. Three cells already exist
           (FACT-0413, FACT-0414 GATE_A, FACT-0414 GATE_B); the fourth - train fold 0, test fold 1 -
           is the discriminating one and is MISSING from the registry. If a fold-0-fitted model also
           fails on fold 1 the problem is fold 0's substrate or labels; if it WINS on fold 1 the
           fold-1 gain is a property of the fold-1 SURFACE and not of the learned head.
           INSTRUMENT CHECK, and it is asserted rather than reported: the same direct-predict path
           must reproduce GATE_B's published fold-0 numbers exactly before the mirror cell is
           trusted. ``matrix`` REFUSES if it does not.
  strata   The fold-0 GATE_A result decomposed by candidate count, distance, primary margin, target
           density, division involvement and crop. The question the strata answer is whether the
           exact zero is a CANCELLATION of heterogeneous strata or UNIFORM NOTHING everywhere -
           different diagnoses that only the strata separate. Its own instrument check re-derives
           GATE_A's 36/36/0.8337743565070964 through this module's code path and REFUSES on any
           disagreement, so a stratum table cannot describe a run that did not happen.
  ablate   Four information classes on BOTH folds under one protocol: raw probability only,
           target-relative (rank / margin / within-target normalised, no raw scale), geometry only,
           and probability-free contextual (the bidirectional columns without the raw scale). The
           point is to locate WHICH information class transfers.
  census   The contested-population composition of both folds, side by side, plus the per-fold
           univariate sign of every feature against ``is_true_parent``. This is the E3 test: a
           column whose sign flips between folds is a representation inversion.

READING ANY OF IT
-----------------
Every number here is PRE-ILP candidate ranking. ``FACT-0364`` measured that motion relink replaces
the solver's whole edge list at a median 99.9% coverage, so none of it is the ``FACT-0376``
final-graph quantity, and ``FACT-0382`` adds that this contested population exists only at the 0.1
acquisition floor. Nothing here is promotable and nothing here is read as division recovery
(``FACT-0371``).

    python scripts/win_bet/assoc_fold_pathology.py matrix  --out-dir C:/temp/assoc_pathology
    python scripts/win_bet/assoc_fold_pathology.py strata  --out-dir C:/temp/assoc_pathology
    python scripts/win_bet/assoc_fold_pathology.py ablate  --out-dir C:/temp/assoc_pathology
    python scripts/win_bet/assoc_fold_pathology.py census  --out-dir C:/temp/assoc_pathology
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

from assoc_baseline_rankers import FEATURE_SETS, SEED, crop_paired_bootstrap  # noqa: E402
from assoc_parent_dataset import FEATURES, evaluate  # noqa: E402
from assoc_report import parent_conversions  # noqa: E402
from assoc_train_harness import (  # noqa: E402
    HarnessRefusal,
    ModelSpec,
    decide,
    fit_out_of_fold,
    make_splitter,
)
import assoc_tournament as TOURN  # noqa: E402  the frozen recipe, imported and NOT forked

# The published values this module must reproduce before any new number it emits is trusted.
# They are FACT-0414's, quoted here as an ASSERTION TARGET rather than as a restatement: the run
# fails if it does not hit them, which is the only use a literal has in an instrument.
PUBLISHED_GATE = {
    "deployed_contested_top1_f0": 0.8337743565070964,
    "gate_a": {"contested_top1": 0.8337743565070964, "gained": 36, "lost": 36, "churn": 72},
    "gate_b": {"contested_top1": 0.8171758479672842, "gained": 133, "lost": 202, "churn": 335},
    "deployed_contested_top1_f1": 0.7083944212623585,
    "discovery_a4": {"contested_top1": 0.7622269839479718, "gained": 2316, "lost": 884},
}

# THE FOUR INFORMATION CLASSES. Preregistered, and three of the four are literally the sets
# ``assoc_baseline_rankers`` declared for FACT-0386, so the ablation is comparable to it rather
# than merely adjacent. None of them was chosen after seeing a fold-0 number.
BIDIR = list(TOURN.BIDIR)
ABLATIONS: dict[str, list[str]] = {
    # (a) raw probability only - the deployed scalar and nothing else. Its OWN delta is the binning
    #     floor, not skill (FACT-0386): a monotone transform of one feature cannot move an argmax.
    "raw_prob_only": list(FEATURE_SETS["prob_only"]),
    # (b) TARGET-RELATIVE probability - rank, margin and the within-target normalised share, with
    #     the raw scale removed. This is the E1 probe: if this class transfers where raw scale does
    #     not, the failure is a per-fold calibration shift and the fix is named by the result.
    "target_relative": ["rank", "margin_to_best", "n_candidates", "fwd_prob_norm"],
    # (c) geometry only - no probability at any remove.
    "geometry_only": list(FEATURE_SETS["geometry_only"]),
    # (d) probability-FREE contextual - the six bidirectional columns plus source competition. Every
    #     column is derived from prob but NONE carries its raw scale, so this isolates the context
    #     that lifted fold 1 above the tree from the scalar the deployed argmax already uses.
    "prob_free_context": BIDIR + ["src_out_degree"],
}

FULL_RECIPE = list(TOURN.BIDIR_FEATURES)   # the frozen A4.bidir.tree feature list
PATHOLOGY_HEAD = "assoc_fold_pathology:_pathology_head"
_THIS_FILE = str(Path(__file__).resolve())


# ==============================================================================================
# Surfaces
# ==============================================================================================

def prepare_fold(fold: int) -> dict:
    """Load one fold's surface through the tournament's own loader and derive everything once."""
    path = TOURN.DISCOVERY_TABLE if fold == TOURN.DISCOVERY_FOLD else TOURN.GATE_TABLE
    table = TOURN.load_surface(path, fold)
    base = evaluate(table, "prob")
    baseline_map = base.pop("per_target_correct")
    dec = table.filter(pl.col("true_parent_is_candidate") == 1)
    counts = dec.group_by(["crop", "target"]).agg(pl.len().alias("n_dec"))
    dec = dec.join(counts, on=["crop", "target"], how="left")
    contested_keys = {(r[0], int(r[1])) for r in
                      counts.filter(pl.col("n_dec") > 1).select(["crop", "target"]).rows()}
    return {"fold": fold, "path": path, "table": table, "dec": dec,
            "baseline_map": baseline_map, "contested_keys": contested_keys,
            "bar": base["contested"]["top1"], "deployed": base,
            "provenance": TOURN.surface_provenance(path, fold)}


def _fit_full(dec: pl.DataFrame, features: list[str], tag: str):
    """Fit the frozen recipe on ALL decidable rows of one fold - the mirror of ``freeze``."""
    spec_features = list(TOURN.META) + list(features)
    model = TOURN.MinedRanker("tree", list(features), rounds=1, tag=tag)
    x = dec.select(spec_features).to_numpy().astype(np.float64)
    y = dec["is_true_parent"].to_numpy().astype(np.int64)
    model.fit(x, y)
    return model, spec_features


def _score_frozen(model, dec: pl.DataFrame, spec_features: list[str]) -> np.ndarray:
    x = dec.select(spec_features).to_numpy().astype(np.float64)
    return model.predict_proba(x)[:, 1]


def _score_oof(dec: pl.DataFrame, features: list[str], tag: str,
               declared: bool = True) -> np.ndarray:
    """Crop-grouped out-of-fold scores through the HARNESS's own splitter and integrity asserts."""
    if declared:
        arm = {"tag": tag, "class": "declared", "head": PATHOLOGY_HEAD,
               "features": list(TOURN.META) + list(features),
               "abstain": {"kind": "none"}, "population": "alldec"}
    else:
        arm = {"tag": tag, "class": "tree", "features": list(features),
               "abstain": {"kind": "none"}, "population": "alldec"}
    spec = ModelSpec.from_dict(arm)
    x = dec.select(spec.features).to_numpy().astype(np.float64)
    score, _null, _folds = fit_out_of_fold(dec, spec, x, make_splitter("GroupKFold", 5), tag)
    return score


# The declared-head factory the harness resolves by name. Module-level state, set immediately
# before each ``_score_oof`` call, is the only channel ``resolve_callable`` leaves open - and the
# ``_module_copies`` defect (PKT-0038 result) is why it is written through the tournament's helper
# rather than assigned here directly.
_PATHOLOGY_FEATURES: list[str] = []


def _pathology_head():
    if not _PATHOLOGY_FEATURES:
        raise HarnessRefusal("_pathology_head constructed with no feature list bound")
    return TOURN.MinedRanker("tree", list(_PATHOLOGY_FEATURES), rounds=1, tag="pathology")


def _bind_features(features: list[str]) -> None:
    """Bind on EVERY live copy of this module - the harness imports it by name (FACT-0060 class)."""
    import importlib
    try:
        importlib.import_module("assoc_fold_pathology")
    except Exception:                       # pragma: no cover - a missing copy is not fatal
        pass
    bound = 0
    for name in ("__main__", "assoc_fold_pathology"):
        mod = sys.modules.get(name)
        same = mod is not None and str(Path(getattr(mod, "__file__", "") or "x").resolve()
                                       ) == _THIS_FILE
        if same:
            mod._PATHOLOGY_FEATURES = list(features)
            bound += 1
    if not bound:
        raise HarnessRefusal("_bind_features found no copy of this module to bind")


def _outcome(surf: dict, score: np.ndarray, tag: str) -> dict:
    """One arm's decision, its conversions and its crop-paired interval, on one fold."""
    dec, baseline_map = surf["dec"], surf["baseline_map"]
    contested_keys = surf["contested_keys"]
    scored = dec.with_columns([pl.Series("_score", score)])
    per_target, ledger, agg = decide(scored, "_score", None, preserve_single=True)
    conv = parent_conversions(
        {k: v for k, v in baseline_map.items() if k in contested_keys},
        {k: v for k, v in per_target.items() if k in contested_keys})
    conv["crop_paired_bootstrap"] = crop_paired_bootstrap(baseline_map, per_target, contested_keys)
    return {
        "tag": tag,
        "contested_top1": agg["contested"]["top1"],
        "delta_vs_deployed": agg["contested"]["top1"] - surf["bar"],
        "gained": conv["gained"], "lost": conv["lost"], "net": conv["net"],
        "churn": conv["churn"],
        "ci95": conv["crop_paired_bootstrap"]["ci95"],
        "favourable": conv["crop_paired_bootstrap"]["favourable"],
        "excludes_zero": conv["crop_paired_bootstrap"]["excludes_zero"],
        "single_candidate_regressions": ledger.summary()["n_regressions"],
        "_per_target": per_target,
    }


def _strip(row: dict) -> dict:
    return {k: v for k, v in row.items() if not k.startswith("_")}


# ==============================================================================================
# 1. THE 2x2
# ==============================================================================================

def cmd_matrix(args) -> int:
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f0, f1 = prepare_fold(0), prepare_fold(1)

    # --- the instrument check. The published GATE_B was produced by the harness's OOF loop wrapped
    # round a frozen model; a frozen model gives the same answer on every fold, so a DIRECT predict
    # must land on the identical numbers. If it does not, this module's transfer path is not the
    # one that produced FACT-0414 and the mirror cell below is meaningless.
    import pickle
    rec = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
    frozen_f1 = pickle.loads(Path(rec["model_pickle"]).read_bytes())
    gb = _outcome(f0, _score_frozen(frozen_f1, f0["dec"], list(TOURN.META) + FULL_RECIPE),
                  "train_f1__test_f0.frozen")
    pub = PUBLISHED_GATE["gate_b"]
    if (abs(gb["contested_top1"] - pub["contested_top1"]) > 1e-12
            or gb["gained"] != pub["gained"] or gb["lost"] != pub["lost"]):
        raise HarnessRefusal(
            "INSTRUMENT CHECK FAILED: the direct-predict transfer path does not reproduce the "
            f"published GATE_B ({gb['contested_top1']!r}/{gb['gained']}/{gb['lost']} against "
            f"{pub['contested_top1']!r}/{pub['gained']}/{pub['lost']}). The mirror cell computed "
            "by the same path therefore cannot be trusted and is not written."
        )

    # --- the MISSING cell: fit on ALL of fold 0, apply frozen to fold 1.
    model_f0, feats = _fit_full(f0["dec"], FULL_RECIPE, "F0_FULL.bidir.tree")
    fb = _outcome(f1, _score_frozen(model_f0, f1["dec"], feats), "train_f0__test_f1.frozen")

    # --- the two same-fold refits, recomputed here so all four cells come from ONE code path.
    _bind_features(FULL_RECIPE)
    aa = _outcome(f0, _score_oof(f0["dec"], FULL_RECIPE, "train_f0__test_f0.oof"),
                  "train_f0__test_f0.oof")
    _bind_features(FULL_RECIPE)
    dd = _outcome(f1, _score_oof(f1["dec"], FULL_RECIPE, "train_f1__test_f1.oof"),
                  "train_f1__test_f1.oof")

    pa = PUBLISHED_GATE["gate_a"]
    if (abs(aa["contested_top1"] - pa["contested_top1"]) > 1e-12
            or aa["gained"] != pa["gained"] or aa["lost"] != pa["lost"]):
        raise HarnessRefusal(
            "INSTRUMENT CHECK FAILED: the refit path does not reproduce the published GATE_A "
            f"({aa['contested_top1']!r}/{aa['gained']}/{aa['lost']} against "
            f"{pa['contested_top1']!r}/{pa['gained']}/{pa['lost']})."
        )
    pd_ = PUBLISHED_GATE["discovery_a4"]
    if (abs(dd["contested_top1"] - pd_["contested_top1"]) > 1e-12
            or dd["gained"] != pd_["gained"] or dd["lost"] != pd_["lost"]):
        raise HarnessRefusal(
            "INSTRUMENT CHECK FAILED: the refit path does not reproduce the published fold-1 "
            f"A4.bidir.tree ({dd['contested_top1']!r}/{dd['gained']}/{dd['lost']} against "
            f"{pd_['contested_top1']!r}/{pd_['gained']}/{pd_['lost']})."
        )

    payload = {
        "schema_version": 1,
        "heartbeat": "ASSOC_FOLD_PATHOLOGY_MATRIX_COMPLETE",
        "packet": "PKT-0038", "lever": "LEVER-0041",
        "recipe": {"arm": "A4.bidir.tree", "features": FULL_RECIPE,
                   "estimator": "HistGradientBoostingClassifier", "rounds": 1, "seed": SEED},
        "deployed_bar": {"fold0": f0["bar"], "fold1": f1["bar"]},
        "instrument_checks": {
            "direct_predict_reproduces_published_gate_b": True,
            "refit_reproduces_published_gate_a": True,
            "refit_reproduces_published_discovery_a4": True,
            "why": "all four cells are produced by ONE code path, and three of them have published "
                   "values (FACT-0413, FACT-0414); reproducing those exactly is what licenses the "
                   "fourth, which has none",
        },
        "cells": {
            "train_f1__test_f1": _strip(dd),
            "train_f1__test_f0": _strip(gb),
            "train_f0__test_f0": _strip(aa),
            "train_f0__test_f1": _strip(fb),
        },
        "reading": "train_f0__test_f1 is the discriminating cell. If a fold-0-fitted model also "
                   "fails on fold 1 the problem is fold 0's substrate or labels; if it WINS on "
                   "fold 1 the fold-1 gain is a property of the fold-1 SURFACE and not of the "
                   "learned head, which reframes LEVER-0041.",
        "stage": "pre_ILP_candidate_ranking",
        "promotable": False,
        "surfaces": {"fold0": f0["provenance"], "fold1": f1["provenance"]},
        "elapsed_s": round(time.time() - t0, 1),
    }
    p = out_dir / "matrix_2x2.json"
    p.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\n{'cell':28s} {'contested':>10s} {'delta':>10s} {'gain':>6s} {'lost':>6s} "
          f"{'net':>6s} {'churn':>6s} {'ci95_lo':>10s} {'ci95_hi':>10s} {'fav':>6s}")
    for name, r in payload["cells"].items():
        lo, hi = r["ci95"]
        print(f"{name:28s} {r['contested_top1']:10.6f} {r['delta_vs_deployed']:+10.6f} "
              f"{r['gained']:6d} {r['lost']:6d} {r['net']:+6d} {r['churn']:6d} "
              f"{lo:+10.5f} {hi:+10.5f} {str(r['favourable']):>6s}")
    print(f"\nASSOC_FOLD_PATHOLOGY_MATRIX_COMPLETE -> {p}")
    return 0


# ==============================================================================================
# 2. STRATA
# ==============================================================================================

def target_frame(surf: dict) -> pl.DataFrame:
    """One row per decidable target, carrying every stratifier. Label-free except the outcome."""
    dec = surf["dec"]
    g = dec.group_by(["crop", "target"]).agg([
        pl.len().alias("n_dec"),
        pl.col("prob").max().alias("p_best"),
        pl.col("prob").sort(descending=True).slice(1, 1).first().alias("p_second"),
        pl.col("dist_um").min().alias("dist_min_um"),
        pl.col("src_out_degree").max().alias("max_src_out_degree"),
        # the deployed winner's distance: the row with the highest prob, ties to lower source
        pl.col("dist_um").sort_by(["prob", "source"], descending=[True, False]).first()
        .alias("winner_dist_um"),
        pl.col("src_out_degree").sort_by(["prob", "source"], descending=[True, False]).first()
        .alias("winner_src_out_degree"),
        pl.col("source").sort_by(["prob", "source"], descending=[True, False]).first()
        .alias("winner_source"),
        pl.col("dist_um").filter(pl.col("is_true_parent") == 1).first().alias("true_dist_um"),
    ])
    g = g.with_columns([
        (pl.col("p_best") - pl.col("p_second").fill_null(0.0)).alias("primary_margin"),
    ])
    # DIVISION INVOLVEMENT, the FACT-0371 failure mode, computed on the deployed decision: the
    # winning source of this target also wins another target, so choosing it forks that source.
    wins = g.group_by(["crop", "winner_source"]).agg(pl.len().alias("winner_wins"))
    g = g.join(wins, on=["crop", "winner_source"], how="left")
    return g.with_columns((pl.col("winner_wins") > 1).cast(pl.Int64).alias("division_involved"))


def _bin(values: np.ndarray, edges: list[float], labels: list[str]) -> list[str]:
    idx = np.digitize(values, edges, right=False)
    return [labels[min(i, len(labels) - 1)] for i in idx]


def stratify(surf: dict, before: dict, after: dict) -> dict:
    tf = target_frame(surf).filter(pl.col("n_dec") > 1)
    keys = [(c, int(t)) for c, t in zip(tf["crop"].to_list(), tf["target"].to_list())]
    b = np.array([before[k] for k in keys], dtype=np.int64)
    a = np.array([after[k] for k in keys], dtype=np.int64)

    ncand = tf["n_dec"].to_numpy()
    wdist = tf["winner_dist_um"].to_numpy()
    marg = tf["primary_margin"].to_numpy()
    dens = tf["max_src_out_degree"].to_numpy()
    divi = tf["division_involved"].to_numpy()
    crops = np.array(tf["crop"].to_list())

    dist_edges = [float(np.quantile(wdist, q)) for q in (0.25, 0.5, 0.75)]
    marg_edges = [float(np.quantile(marg, q)) for q in (0.25, 0.5, 0.75)]
    facets = {
        "candidate_count": [f"n={min(int(v), 4)}" for v in ncand],
        "winner_distance_um_quartile": _bin(wdist, dist_edges, ["q1", "q2", "q3", "q4"]),
        "primary_margin_quartile": _bin(marg, marg_edges, ["q1", "q2", "q3", "q4"]),
        "target_density_max_src_out_degree": [f"d={min(int(v), 4)}" for v in dens],
        "division_involved": ["fork" if v else "no_fork" for v in divi],
    }

    out: dict = {"n_contested": len(keys),
                 "bin_edges": {"winner_distance_um": dist_edges, "primary_margin": marg_edges}}
    for name, lab in facets.items():
        lab = np.array(lab)
        rows = []
        for level in sorted(set(lab.tolist())):
            m = lab == level
            rows.append({
                "level": level, "n": int(m.sum()),
                "deployed_top1": float(b[m].mean()),
                "model_top1": float(a[m].mean()),
                "delta": float(a[m].mean() - b[m].mean()),
                "gained": int(((b[m] == 0) & (a[m] == 1)).sum()),
                "lost": int(((b[m] == 1) & (a[m] == 0)).sum()),
                "net": int(((b[m] == 0) & (a[m] == 1)).sum() - ((b[m] == 1) & (a[m] == 0)).sum()),
            })
        out[name] = rows

    # PER CROP, and specifically whether the gained and the lost are the SAME crops. If they are,
    # the zero is a local reshuffle; if they are disjoint, whole crops move in opposite directions.
    per_crop = []
    for c in sorted(set(crops.tolist())):
        m = crops == c
        gained = int(((b[m] == 0) & (a[m] == 1)).sum())
        lost = int(((b[m] == 1) & (a[m] == 0)).sum())
        if m.sum() == 0:
            continue
        per_crop.append({"crop": c, "n": int(m.sum()), "gained": gained, "lost": lost,
                         "net": gained - lost, "deployed_top1": float(b[m].mean()),
                         "model_top1": float(a[m].mean())})
    gcrops = {r["crop"] for r in per_crop if r["gained"] > 0}
    lcrops = {r["crop"] for r in per_crop if r["lost"] > 0}
    out["per_crop"] = per_crop
    out["crop_overlap"] = {
        "n_crops_with_contested": len(per_crop),
        "n_crops_with_a_gain": len(gcrops),
        "n_crops_with_a_loss": len(lcrops),
        "n_crops_with_both": len(gcrops & lcrops),
        "n_crops_gain_only": len(gcrops - lcrops),
        "n_crops_loss_only": len(lcrops - gcrops),
        "n_crops_with_net_zero_and_movement": sum(
            1 for r in per_crop if r["net"] == 0 and (r["gained"] + r["lost"]) > 0),
        "crops_net_positive": sorted(r["crop"] for r in per_crop if r["net"] > 0),
        "crops_net_negative": sorted(r["crop"] for r in per_crop if r["net"] < 0),
    }
    # Are the moved targets distinct, or is a gain paired with a loss on the same target? They
    # cannot be the same target by construction (a target is gained or lost, never both) - what
    # this checks is whether the moved targets are scattered or concentrated.
    moved = [k for k, bb, aa_ in zip(keys, b, a) if bb != aa_]
    out["moved_targets"] = {
        "n": len(moved),
        "n_distinct": len(set(moved)),
        "n_distinct_crops": len({k[0] for k in moved}),
        "examples": [{"crop": k[0], "target": k[1],
                      "deployed": int(before[k]), "model": int(after[k])} for k in moved[:12]],
    }
    return out


def cmd_strata(args) -> int:
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f0 = prepare_fold(0)
    _bind_features(FULL_RECIPE)
    row = _outcome(f0, _score_oof(f0["dec"], FULL_RECIPE, "GATE_A.refit"), "GATE_A.refit")
    pa = PUBLISHED_GATE["gate_a"]
    if (abs(row["contested_top1"] - pa["contested_top1"]) > 1e-12
            or row["gained"] != pa["gained"] or row["lost"] != pa["lost"]):
        raise HarnessRefusal(
            "INSTRUMENT CHECK FAILED: this module's GATE_A refit does not reproduce the published "
            f"one ({row['contested_top1']!r}/{row['gained']}/{row['lost']} against "
            f"{pa['contested_top1']!r}/{pa['gained']}/{pa['lost']}) - the strata below would "
            "describe a run that did not happen, so none are written."
        )
    strat = stratify(f0, f0["baseline_map"], row["_per_target"])

    # The same decomposition on the FOLD-1 winner, so "which strata win on fold 1" is answerable
    # rather than assumed - the comparison is the whole point of E2.
    f1 = prepare_fold(1)
    _bind_features(FULL_RECIPE)
    row1 = _outcome(f1, _score_oof(f1["dec"], FULL_RECIPE, "F1.refit"), "F1.refit")
    strat1 = stratify(f1, f1["baseline_map"], row1["_per_target"])

    payload = {
        "schema_version": 1,
        "heartbeat": "ASSOC_FOLD_PATHOLOGY_STRATA_COMPLETE",
        "packet": "PKT-0038", "lever": "LEVER-0041",
        "instrument_check": {"reproduces_published_gate_a": True,
                             "reproduces_published_fold1_a4": bool(
                                 abs(row1["contested_top1"]
                                     - PUBLISHED_GATE["discovery_a4"]["contested_top1"]) < 1e-12)},
        "gate_a_fold0": {"summary": _strip(row), "strata": strat},
        "fold1_a4": {"summary": _strip(row1), "strata": strat1},
        "question": "is the exact zero a CANCELLATION of heterogeneous strata, or UNIFORM NOTHING "
                    "everywhere? Those are different diagnoses and only the strata separate them.",
        "stage": "pre_ILP_candidate_ranking",
        "promotable": False,
        "elapsed_s": round(time.time() - t0, 1),
    }
    p = out_dir / "strata_gate_a.json"
    p.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    for fold_name, s in (("FOLD 0 (GATE_A refit)", strat), ("FOLD 1 (A4 refit)", strat1)):
        print(f"\n=== {fold_name} ===")
        for facet in ("candidate_count", "winner_distance_um_quartile",
                      "primary_margin_quartile", "target_density_max_src_out_degree",
                      "division_involved"):
            print(f"  {facet}")
            for r in s[facet]:
                print(f"    {r['level']:12s} n={r['n']:6d} deployed={r['deployed_top1']:.4f} "
                      f"model={r['model_top1']:.4f} delta={r['delta']:+.4f} "
                      f"gained={r['gained']:5d} lost={r['lost']:5d} net={r['net']:+5d}")
        print(f"  crop_overlap {json.dumps({k: v for k, v in s['crop_overlap'].items() if not isinstance(v, list)})}")
    print(f"\nASSOC_FOLD_PATHOLOGY_STRATA_COMPLETE -> {p}")
    return 0


# ==============================================================================================
# 3. ABLATIONS
# ==============================================================================================

def cmd_ablate(args) -> int:
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    surfs = {0: prepare_fold(0), 1: prepare_fold(1)}
    results: dict[str, dict] = {}
    for name, feats in ABLATIONS.items():
        entry: dict = {"features": feats}
        fits = {}
        for fold in (0, 1):
            _bind_features(feats)
            r = _outcome(surfs[fold], _score_oof(surfs[fold]["dec"], feats, f"{name}.f{fold}.oof"),
                         f"{name}.f{fold}.oof")
            entry[f"refit_f{fold}"] = _strip(r)
            fits[fold] = _fit_full(surfs[fold]["dec"], feats, f"{name}.f{fold}.full")
        # both transfer directions, same class, so "which information class transfers" is a
        # measured statement rather than an inference from the refits alone
        m1, sp1 = fits[1]
        entry["transfer_f1_to_f0"] = _strip(
            _outcome(surfs[0], _score_frozen(m1, surfs[0]["dec"], sp1), f"{name}.f1->f0"))
        m0, sp0 = fits[0]
        entry["transfer_f0_to_f1"] = _strip(
            _outcome(surfs[1], _score_frozen(m0, surfs[1]["dec"], sp0), f"{name}.f0->f1"))
        results[name] = entry
        print(f"\n{name}  {feats}")
        for k in ("refit_f0", "refit_f1", "transfer_f1_to_f0", "transfer_f0_to_f1"):
            r = entry[k]
            lo, hi = r["ci95"]
            print(f"  {k:20s} top1={r['contested_top1']:.6f} delta={r['delta_vs_deployed']:+.6f} "
                  f"g={r['gained']:5d} l={r['lost']:5d} net={r['net']:+5d} churn={r['churn']:5d} "
                  f"ci95=[{lo:+.5f},{hi:+.5f}] fav={r['favourable']}")

    payload = {
        "schema_version": 1,
        "heartbeat": "ASSOC_FOLD_PATHOLOGY_ABLATE_COMPLETE",
        "packet": "PKT-0038", "lever": "LEVER-0041",
        "deployed_bar": {"fold0": surfs[0]["bar"], "fold1": surfs[1]["bar"]},
        "protocol": "identical on both folds: crop-grouped GroupKFold(5) out-of-fold refit, plus a "
                    "full-fold fit transferred frozen in each direction. No fold-0 number selects "
                    "anything.",
        "ablations": results,
        "reading": "if the TARGET-RELATIVE class transfers where the RAW-SCALE class does not, "
                   "that is E1 (calibration shift) confirmed and it also names the fix.",
        "stage": "pre_ILP_candidate_ranking",
        "promotable": False,
        "elapsed_s": round(time.time() - t0, 1),
    }
    p = out_dir / "ablations.json"
    p.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\nASSOC_FOLD_PATHOLOGY_ABLATE_COMPLETE -> {p}")
    return 0


# ==============================================================================================
# 4. CENSUS - the E2 mixture and the E3 sign test
# ==============================================================================================

def cmd_census(args) -> int:
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(FEATURES) + BIDIR
    per_fold: dict[str, dict] = {}
    for fold in (0, 1):
        surf = prepare_fold(fold)
        dec = surf["dec"]
        cont = dec.filter(pl.col("n_dec") > 1)
        tf = target_frame(surf).filter(pl.col("n_dec") > 1)
        y = cont["is_true_parent"].to_numpy().astype(np.float64)
        stats = {}
        for c in cols:
            v = cont[c].to_numpy().astype(np.float64)
            sd = float(np.std(v))
            # point-biserial correlation with the label, on CONTESTED rows only: the sign is the
            # E3 test and the scale is the E1 one.
            r = float(np.corrcoef(v, y)[0, 1]) if sd > 0 else float("nan")
            stats[c] = {
                "mean": float(np.mean(v)), "std": sd,
                "p05": float(np.quantile(v, 0.05)), "p50": float(np.quantile(v, 0.5)),
                "p95": float(np.quantile(v, 0.95)),
                "corr_with_is_true_parent": r,
                "mean_pos": float(v[y == 1].mean()) if (y == 1).any() else None,
                "mean_neg": float(v[y == 0].mean()) if (y == 0).any() else None,
            }
        per_fold[f"fold{fold}"] = {
            "surface": surf["provenance"],
            "decidable_targets": len(surf["baseline_map"]),
            "contested_targets": len(surf["contested_keys"]),
            "contested_share": len(surf["contested_keys"]) / max(len(surf["baseline_map"]), 1),
            "deployed_contested_top1": surf["bar"],
            "deployed_contested_errors": int(round(
                (1 - surf["bar"]) * len(surf["contested_keys"]))),
            "contested_rows": cont.height,
            "mixture": {
                "n_candidates": {str(k): int(v) for k, v in
                                 zip(*np.unique(tf["n_dec"].to_numpy(), return_counts=True))},
                "primary_margin": {q: float(np.quantile(tf["primary_margin"].to_numpy(), f))
                                   for q, f in (("p10", .1), ("p25", .25), ("p50", .5),
                                                ("p75", .75), ("p90", .9))},
                "winner_dist_um": {q: float(np.quantile(tf["winner_dist_um"].to_numpy(), f))
                                   for q, f in (("p10", .1), ("p50", .5), ("p90", .9))},
                "max_src_out_degree": {str(k): int(v) for k, v in zip(
                    *np.unique(np.minimum(tf["max_src_out_degree"].to_numpy(), 6),
                               return_counts=True))},
                "division_involved_share": float(tf["division_involved"].mean()),
            },
            "feature_stats": stats,
        }
    s0 = per_fold["fold0"]["feature_stats"]
    s1 = per_fold["fold1"]["feature_stats"]
    signs = {}
    for c in cols:
        r0, r1 = s0[c]["corr_with_is_true_parent"], s1[c]["corr_with_is_true_parent"]
        signs[c] = {
            "corr_f0": r0, "corr_f1": r1,
            "sign_agrees": bool(np.sign(r0) == np.sign(r1)) if not (
                np.isnan(r0) or np.isnan(r1)) else None,
            "abs_gap": abs(r0 - r1) if not (np.isnan(r0) or np.isnan(r1)) else None,
            "mean_ratio_f1_over_f0": (s1[c]["mean"] / s0[c]["mean"]) if s0[c]["mean"] else None,
        }
    payload = {
        "schema_version": 1,
        "heartbeat": "ASSOC_FOLD_PATHOLOGY_CENSUS_COMPLETE",
        "packet": "PKT-0038", "lever": "LEVER-0041",
        "folds": per_fold,
        "e3_sign_test": signs,
        "n_features_with_opposite_sign": sum(1 for v in signs.values() if v["sign_agrees"] is False),
        "stage": "pre_ILP_candidate_ranking",
        "promotable": False,
        "elapsed_s": round(time.time() - t0, 1),
    }
    p = out_dir / "census.json"
    p.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(f"\nfeatures with OPPOSITE label-correlation sign across folds: "
          f"{payload['n_features_with_opposite_sign']}")
    for c, v in signs.items():
        print(f"  {c:22s} corr_f0={v['corr_f0']:+.4f} corr_f1={v['corr_f1']:+.4f} "
              f"agrees={v['sign_agrees']}")
    print(f"\nASSOC_FOLD_PATHOLOGY_CENSUS_COMPLETE -> {p}")
    return 0


# ==============================================================================================
# 5. DIRECT STANDARDISATION - how much of the fold difference is MIXTURE and how much is not
# ==============================================================================================
#
# The per-fold quartile bins of ``strata`` are fold-specific by construction, so they cannot be
# compared across folds. This command re-bins BOTH folds on ONE set of ABSOLUTE edges - declared
# here, not fitted - and then does the Kitagawa decomposition every mixture claim needs:
#
#   delta_f1 - delta_f0  =  SUM_s (w1_s - w0_s) * dbar_s      <- COMPOSITION, the E2 term
#                        +  SUM_s  wbar_s * (d1_s - d0_s)     <- WITHIN-STRATUM, everything else
#
# If the composition term carries the difference, E2 is confirmed and the folds simply meet
# different mixtures of the same problem. If the within-stratum term carries it, the SAME stratum
# behaves differently on the two folds and a mixture shift is not the explanation.
MARGIN_EDGES = [0.05, 0.15, 0.30, 0.50, 0.70]
MARGIN_LABELS = ["m<0.05", "0.05-0.15", "0.15-0.30", "0.30-0.50", "0.50-0.70", "m>=0.70"]


def cmd_standardise(args) -> int:
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold = {}
    for fold in (0, 1):
        surf = prepare_fold(fold)
        _bind_features(FULL_RECIPE)
        row = _outcome(surf, _score_oof(surf["dec"], FULL_RECIPE, "std.f%d" % fold),
                       "std.f%d" % fold)
        tf = target_frame(surf).filter(pl.col("n_dec") > 1)
        keys = [(c, int(x)) for c, x in zip(tf["crop"].to_list(), tf["target"].to_list())]
        b = np.array([surf["baseline_map"][k] for k in keys], dtype=np.float64)
        a = np.array([row["_per_target"][k] for k in keys], dtype=np.float64)
        lab = np.array(_bin(tf["primary_margin"].to_numpy(), MARGIN_EDGES, MARGIN_LABELS))
        cells = {}
        for level in MARGIN_LABELS:
            m = lab == level
            cells[level] = {
                "n": int(m.sum()),
                "w": float(m.mean()),
                "deployed_top1": float(b[m].mean()) if m.any() else None,
                "model_top1": float(a[m].mean()) if m.any() else None,
                "delta": float(a[m].mean() - b[m].mean()) if m.any() else None,
                "deployed_errors": int((1 - b[m]).sum()) if m.any() else 0,
                "gained": int(((b[m] == 0) & (a[m] == 1)).sum()),
                "lost": int(((b[m] == 1) & (a[m] == 0)).sum()),
                "error_conversion_rate": (float(((b[m] == 0) & (a[m] == 1)).sum()
                                                / max((1 - b[m]).sum(), 1)) if m.any() else None),
            }
        per_fold[fold] = {"summary": _strip(row), "cells": cells,
                          "overall_delta": float(a.mean() - b.mean()),
                          "n_contested": len(keys), "surface": surf["provenance"]}
        # persist the per-target outcome so a later instrument does not refit to re-read it
        pl.DataFrame({"crop": [k[0] for k in keys], "target": [k[1] for k in keys],
                      "deployed_correct": b.astype(np.int64),
                      "model_correct": a.astype(np.int64),
                      "primary_margin": tf["primary_margin"].to_numpy(),
                      "n_dec": tf["n_dec"].to_numpy(),
                      "winner_dist_um": tf["winner_dist_um"].to_numpy(),
                      "division_involved": tf["division_involved"].to_numpy(),
                      }).write_parquet(out_dir / ("contested_outcomes_f%d.parquet" % fold))

    comp = within = 0.0
    rows = []
    for level in MARGIN_LABELS:
        c0, c1 = per_fold[0]["cells"][level], per_fold[1]["cells"][level]
        if c0["n"] == 0 or c1["n"] == 0:
            rows.append({"level": level, "skipped": True, "n_f0": c0["n"], "n_f1": c1["n"]})
            continue
        dbar = 0.5 * (c0["delta"] + c1["delta"])
        wbar = 0.5 * (c0["w"] + c1["w"])
        comp += (c1["w"] - c0["w"]) * dbar
        within += wbar * (c1["delta"] - c0["delta"])
        rows.append({"level": level, "n_f0": c0["n"], "n_f1": c1["n"],
                     "w_f0": c0["w"], "w_f1": c1["w"],
                     "deployed_f0": c0["deployed_top1"], "deployed_f1": c1["deployed_top1"],
                     "delta_f0": c0["delta"], "delta_f1": c1["delta"],
                     "err_conv_f0": c0["error_conversion_rate"],
                     "err_conv_f1": c1["error_conversion_rate"]})
    observed = per_fold[1]["overall_delta"] - per_fold[0]["overall_delta"]
    payload = {
        "schema_version": 1,
        "heartbeat": "ASSOC_FOLD_PATHOLOGY_STANDARDISE_COMPLETE",
        "packet": "PKT-0038", "lever": "LEVER-0041",
        "margin_edges": MARGIN_EDGES, "margin_labels": MARGIN_LABELS,
        "per_fold": {("fold%d" % k): v for k, v in per_fold.items()},
        "kitagawa": {
            "observed_delta_f1_minus_f0": observed,
            "composition_term": comp,
            "within_stratum_term": within,
            "sum_check": comp + within,
            "residual": observed - (comp + within),
            "composition_share": comp / observed if observed else None,
            "within_share": within / observed if observed else None,
        },
        "cells": rows,
        "reading": "if the COMPOSITION term carries the fold difference, E2 (mixture shift) is "
                   "confirmed; if the WITHIN-STRATUM term carries it, the same stratum behaves "
                   "differently on the two folds and mixture is not the explanation.",
        "stage": "pre_ILP_candidate_ranking", "promotable": False,
        "elapsed_s": round(time.time() - t0, 1),
    }
    p = out_dir / "standardised_margin.json"
    p.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print("")
    print("%-12s %7s %7s %7s %7s %8s %8s %9s %9s %7s %7s"
          % ("margin bin", "n_f0", "n_f1", "w_f0", "w_f1", "dep_f0", "dep_f1",
             "d_f0", "d_f1", "ec_f0", "ec_f1"))
    for r in rows:
        if r.get("skipped"):
            print("%-12s SKIPPED n_f0=%d n_f1=%d" % (r["level"], r["n_f0"], r["n_f1"]))
            continue
        print("%-12s %7d %7d %7.4f %7.4f %8.4f %8.4f %+9.4f %+9.4f %7.4f %7.4f"
              % (r["level"], r["n_f0"], r["n_f1"], r["w_f0"], r["w_f1"], r["deployed_f0"],
                 r["deployed_f1"], r["delta_f0"], r["delta_f1"], r["err_conv_f0"],
                 r["err_conv_f1"]))
    k = payload["kitagawa"]
    print("\nobserved delta_f1 - delta_f0 = %+.6f" % k["observed_delta_f1_minus_f0"])
    print("  composition (E2 mixture)   = %+.6f (%.1f%%)"
          % (k["composition_term"], 100 * (k["composition_share"] or 0)))
    print("  within-stratum             = %+.6f (%.1f%%)"
          % (k["within_stratum_term"], 100 * (k["within_share"] or 0)))
    print("\nASSOC_FOLD_PATHOLOGY_STANDARDISE_COMPLETE -> %s" % p)
    return 0


# ==============================================================================================
# 6. THE APPARENT-FIT CEILING - is fold 0's null a POWER problem or a SEPARABILITY problem?
# ==============================================================================================
#
# An out-of-fold null has two readings and the packet cannot choose between them from the OOF
# number alone. (1) VARIANCE: fold 0 carries too few contested errors for a fitted head to
# generalise, so the signal exists and the fold cannot show it - that is FACT-0388's reading and
# it would make fold 0 an inadequate gate rather than a real refusal. (2) BIAS: the contested
# errors are not separable by this representation at all, in which case no amount of data helps
# and the refusal is about the representation.
#
# The APPARENT fit separates them. Fitting on all decidable rows and scoring the SAME rows is a
# memorisation ceiling: it is the most this representation can possibly express on this fold. The
# META identity columns are stripped before the fit (MinedRanker._split_meta asserts it), so the
# tree cannot memorise target identity and the ceiling is representational rather than trivial.
#
# THIS IS NOT A MODEL-SELECTION NUMBER AND IS NEVER QUOTED AS PERFORMANCE. It is in-fold by
# construction, it is an upper bound, and nothing is chosen by it.


def cmd_ceiling(args) -> int:
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for fold in (0, 1):
        surf = prepare_fold(fold)
        entry = {"deployed_contested_top1": surf["bar"],
                 "contested": len(surf["contested_keys"]),
                 "deployed_contested_errors": int(round(
                     (1 - surf["bar"]) * len(surf["contested_keys"])))}
        for name, feats in (("full_recipe", FULL_RECIPE),) + tuple(ABLATIONS.items()):
            model, sp = _fit_full(surf["dec"], feats, "ceiling.%s.f%d" % (name, fold))
            row = _outcome(surf, _score_frozen(model, surf["dec"], sp),
                           "ceiling.%s.f%d" % (name, fold))
            entry[name] = _strip(row)
        results["fold%d" % fold] = entry

    payload = {
        "schema_version": 1,
        "heartbeat": "ASSOC_FOLD_PATHOLOGY_CEILING_COMPLETE",
        "packet": "PKT-0038", "lever": "LEVER-0041",
        "what_this_is": "APPARENT (in-fold) fit. A memorisation ceiling on what each information "
                        "class can express on each fold, used ONLY to separate a variance null "
                        "from a separability null. Never a performance number and never a "
                        "selection criterion.",
        "reading": "if the apparent ceiling on fold 0 is also ~0, the fold-0 refusal is about the "
                   "REPRESENTATION and not about fold 0's statistical power; if it is large, the "
                   "signal exists on fold 0 and only the out-of-fold estimate cannot find it.",
        "folds": results,
        "stage": "pre_ILP_candidate_ranking", "promotable": False,
        "elapsed_s": round(time.time() - t0, 1),
    }
    p = out_dir / "apparent_ceiling.json"
    p.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    for f, e in results.items():
        print("\n%s  deployed contested top-1 %.6f  contested %d  errors %d"
              % (f, e["deployed_contested_top1"], e["contested"],
                 e["deployed_contested_errors"]))
        for name in ["full_recipe"] + list(ABLATIONS):
            r = e[name]
            print("  %-20s apparent_top1=%.6f delta=%+.6f gained=%5d lost=%5d net=%+6d"
                  % (name, r["contested_top1"], r["delta_vs_deployed"], r["gained"],
                     r["lost"], r["net"]))
    print("\nASSOC_FOLD_PATHOLOGY_CEILING_COMPLETE -> %s" % p)
    return 0


# ==============================================================================================
# 7. ARGMAX AGREEMENT - "zero churn" is a statement about CORRECTNESS, not about the choice
# ==============================================================================================
#
# ``parent_conversions`` compares per-target CORRECTNESS maps, so a target that swaps one WRONG
# parent for another wrong parent contributes nothing to gained, lost or churn. A stratum reported
# with churn 0 is therefore NOT the same claim as "the model chose what the deployed argmax chose",
# and reading it that way would overstate what the strata show. This command measures the stronger
# quantity directly: the SOURCE each rule picks per contested target, and how often they agree.


def _chosen_source(dec: pl.DataFrame, score: np.ndarray) -> dict:
    """The source each target's argmax picks, ties broken by LOWER SOURCE INDEX (contract rule 4)."""
    scored = dec.with_columns([pl.Series("_s", score)])
    out: dict[tuple[str, int], int] = {}
    for _key, group in scored.group_by("crop", "target"):
        s = group["_s"].to_numpy()
        src = group["source"].to_numpy()
        order = np.lexsort((src, -s))
        out[(str(group["crop"][0]), int(group["target"][0]))] = int(src[order[0]])
    return out


def cmd_argmax(args) -> int:
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold = {}
    for fold in (0, 1):
        surf = prepare_fold(fold)
        dec = surf["dec"]
        _bind_features(FULL_RECIPE)
        score = _score_oof(dec, FULL_RECIPE, "argmax.f%d" % fold)
        deployed = _chosen_source(dec, dec["prob"].to_numpy().astype(np.float64))
        model = _chosen_source(dec, score)
        tf = target_frame(surf).filter(pl.col("n_dec") > 1)
        keys = [(c, int(x)) for c, x in zip(tf["crop"].to_list(), tf["target"].to_list())]
        same = np.array([deployed[k] == model[k] for k in keys])
        b = np.array([surf["baseline_map"][k] for k in keys], dtype=np.int64)
        lab = np.array(_bin(tf["primary_margin"].to_numpy(),
                            [float(np.quantile(tf["primary_margin"].to_numpy(), q))
                             for q in (0.25, 0.5, 0.75)], ["q1", "q2", "q3", "q4"]))
        by_q = []
        for level in ("q1", "q2", "q3", "q4"):
            m = lab == level
            moved = ~same & m
            by_q.append({"level": level, "n": int(m.sum()),
                         "same_source": int(same[m].sum()),
                         "different_source": int((~same[m]).sum()),
                         "agreement_rate": float(same[m].mean()) if m.any() else None,
                         # the swaps CORRECTNESS cannot see: a wrong parent for another wrong one
                         "wrong_to_wrong_swaps": int((moved & (b == 0)).sum())})
        per_fold["fold%d" % fold] = {
            "n_contested": len(keys),
            "same_source": int(same.sum()),
            "different_source": int((~same).sum()),
            "agreement_rate": float(same.mean()),
            "wrong_to_wrong_swaps_total": int(((~same) & (b == 0)).sum()),
            "by_margin_quartile": by_q,
        }
        print("\nfold %d  contested %d  argmax agrees with deployed on %d (%.4f)  "
              "different on %d" % (fold, len(keys), int(same.sum()), float(same.mean()),
                                   int((~same).sum())))
        for r in by_q:
            print("  %-3s n=%6d same=%6d different=%5d agree=%.4f wrong_to_wrong=%4d"
                  % (r["level"], r["n"], r["same_source"], r["different_source"],
                     r["agreement_rate"], r["wrong_to_wrong_swaps"]))

    payload = {
        "schema_version": 1,
        "heartbeat": "ASSOC_FOLD_PATHOLOGY_ARGMAX_COMPLETE",
        "packet": "PKT-0038", "lever": "LEVER-0041",
        "what_this_is": "the SOURCE each rule picks per contested target, so that a stratum "
                        "reported with churn 0 can be read correctly: churn is a CORRECTNESS "
                        "statistic and is blind to a wrong-for-wrong parent swap.",
        "folds": per_fold,
        "stage": "pre_ILP_candidate_ranking", "promotable": False,
        "elapsed_s": round(time.time() - t0, 1),
    }
    p = out_dir / "argmax_agreement.json"
    p.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print("\nASSOC_FOLD_PATHOLOGY_ARGMAX_COMPLETE -> %s" % p)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("matrix", cmd_matrix), ("strata", cmd_strata),
                     ("ablate", cmd_ablate), ("census", cmd_census),
                     ("standardise", cmd_standardise), ("ceiling", cmd_ceiling),
                     ("argmax", cmd_argmax)):
        s = sub.add_parser(name)
        s.add_argument("--out-dir", type=Path, default=Path("C:/temp/assoc_pathology"))
        s.add_argument("--frozen", type=Path,
                       default=Path("C:/temp/assoc_tournament/frozen_selection.json"))
        s.set_defaults(func=fn)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
