r"""THE CROSS-FOLD DIVISION VERIFIER - route audit, calibrated baselines, operating points.

LEVER-0040 / PKT-0035. This is the model half of the lane. ``divverify_dataset.py`` builds the
rows and holds the leakage contract; this module fits the baselines, and
``divverify_adapter.py apply`` measures what their decisions do to the official metric.

WHAT THIS PACKET CHANGES ABOUT THE PREVIOUS ATTEMPT
---------------------------------------------------
PKT-0032 (FACT-0385) measured a cross-fitted verifier on 19 EMITTED TRUE FORKS and found the
mechanism works on fold 0 and inverts on fold 1. It also DIAGNOSED the reason pooling GT-derived
tuples inflated the number - ``out_degree`` and ``competing_parents`` read the CONSTRUCTION ROUTE,
not the physics - but it did not disarm it, so the larger positive population stayed unusable.

Three changes, each of which is a mechanism with a test rather than an assertion:

  1. ROUTE-NEUTRAL FEATURES. The route-reading columns are removed
     (``divverify_dataset.ROUTE_READING``) and ``route_audit`` then TRIES TO PREDICT THE ROUTE from
     what is left. A route AUC near 0.5 is what makes the enlarged positive class admissible; a
     high one says the fix failed and the arm must not be read. The audit is run and reported
     whatever it says.
  2. THE POSITIVE CLASS IS GENUINE MOTHER-DAUGHTER TUPLES, not only the forks the metric happened
     to credit. Say plainly what this does and does not do: it CANNOT increase the number of
     emitted true forks - that is a property of the pipeline, not of the dataset - so
     LEVER-0040's successor falsifier (2) is only partly served. What it increases is the number
     of genuine division tuples a model can learn the shape of.
  3. THE DEPLOYABLE RULE ACTS ON EVERY EMITTED FORK. FACT-0383: the metric charges 72 of fold 0's
     5,656 forks. A verifier at inference cannot know which 72, so its rejection set is drawn from
     all 5,656 and EVERY rejection deletes an edge. The oracle-restricted "charged forks only" view
     is still reported, labelled as an oracle, because it is the ceiling the honest rule is chasing.

PRESERVATION IS THE PRIMARY CONSTRAINT, NOT A TIE-BREAK
-------------------------------------------------------
At 5 true forks on fold 0 and 14 on fold 1 (FACT-0383), one lost true fork is 20% or 7% of the
recoverable signal. Every operating point therefore reports ``true_forks_lost`` first, and an
operating point that costs one is reported as a FAILURE OF THAT OPERATING POINT - not netted off
against the false positives it cut.

TWO THRESHOLDS, AND ONLY ONE OF THEM IS DEPLOYABLE
---------------------------------------------------
``oracle_threshold`` walks the judged fold's own labels to find how many false forks could be cut
before the first true one is lost. It is the FACT-0385-comparable diagnostic and it is an ORACLE -
it uses the answer to pick the cut. ``transferred_threshold`` is chosen on the TRAINING fold alone,
from out-of-fold predictions, as the largest threshold that loses no training positive, and is then
applied unchanged. Only the transferred one is a result.

NO HYPERPARAMETER IS SELECTED ON THE JUDGED FOLD
-------------------------------------------------
Library defaults, frozen in ``LINEAR_KW``/``TREE_KW``, no search. Arm selection for the expensive
end-to-end scoring uses ``SELECTION_RULE`` below, which reads out-of-fold performance INSIDE the
training fold and never touches the judged fold. Every arm is reported regardless, so nothing can
be chosen after the fact.

Usage
-----
  .venv\Scripts\python.exe scripts\win_bet\divverify_verifier.py route_audit ^
      --rows-dir C:/temp/divverify2 --out-dir C:/temp/divverify2
  .venv\Scripts\python.exe scripts\win_bet\divverify_verifier.py crossfold ^
      --rows-dir C:/temp/divverify2 --out-dir C:/temp/divverify2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(Path(__file__).resolve().parent))

import divverify_dataset as dv  # noqa: E402

SEED = 20260829
LINEAR_KW = dict(C=1.0, max_iter=5000, solver="lbfgs", class_weight="balanced")
TREE_KW = dict(max_iter=100, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=5,
               l2_regularization=0.0, early_stopping=False, random_state=SEED,
               class_weight="balanced")

# Pre-registered feature sets. `v1` reproduces PKT-0032's space so the two attempts are comparable.
FEATURE_SETS: dict[str, list[str]] = {
    "v1": list(dv.FEATURES_V1),
    "all": list(dv.FEATURES),
    "route_neutral": list(dv.ROUTE_NEUTRAL),
    # What survives after the route audit's second pass. Pure geometry, crowding and temporal
    # context - nothing that reads the linker's own decision. See dv.ROUTE_READING_STRICT for the
    # mechanism; this is the ONLY space in which the enlarged positive class can be honestly used,
    # and choosing it means giving up the association-margin and trajectory families entirely.
    "route_neutral_strict": list(dv.ROUTE_NEUTRAL_STRICT),
}

# Pre-registered training populations. Every one is reported; none is chosen after the fact.
#   emitted_charged  the forks the metric charges on the TRAINING fold - PKT-0032's best arm and
#                    the one its successor falsifier (1) names.
#   genuine_tuples   positives enlarged to every genuine mother-daughter tuple whose three members
#                    the official matcher paired, negatives still the charged false forks.
#   genuine_plus_cf  the same, plus all constructed counterfactuals.
#   genuine_plus_assoc  the same, plus ONLY the association-mistake counterfactuals, so the new
#                    class can be credited or blamed on its own.
POPULATIONS = ("emitted_charged", "genuine_tuples", "genuine_plus_cf", "genuine_plus_assoc")
MODELS = ("linear", "tree")
# Size of the training-fold-only feature selection. Fixed a priori; never tuned.
TOP_K = 8

SELECTION_RULE = (
    "Among all arms, keep those whose out-of-fold TRAINING-fold predictions preserve every "
    "training positive at the transferred threshold; of those, take the highest out-of-fold "
    "deployment AUC on the training fold. Ties break by population order then feature-set order "
    "then model order. The judged fold is never read. If no arm qualifies, none is scored "
    "end-to-end and that is reported as the outcome."
)


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


def _auc(y, s) -> float:
    return dv._auc(np.asarray(y), np.asarray(s, dtype=float))


def _clean(x: np.ndarray) -> np.ndarray:
    """Sentinels (-1) stay as they are - they are informative, not missing. Only non-finite values
    are repaired, and loudly enough to notice: a NaN reaching a tree silently becomes a split."""
    bad = ~np.isfinite(x)
    if bad.any():
        x = x.copy()
        x[bad] = -1.0
    return x


# ============================================================ population selection
def population_rows(table: pl.DataFrame, kind: str) -> pl.DataFrame:
    """The training rows for one pre-registered population. Deduped on the triple."""
    charged = pl.col("official_label").is_in(["tp_fork", "fp_fork"])
    gtpos = pl.col("official_label") == "gt_positive"
    cf = pl.col("source") == "counterfactual"
    assoc = pl.col("neg_kind") == "cf_assoc_mistake"
    if kind == "emitted_charged":
        sel = table.filter(charged)
    elif kind == "genuine_tuples":
        sel = table.filter(charged | gtpos)
    elif kind == "genuine_plus_cf":
        sel = table.filter(charged | gtpos | cf)
    elif kind == "genuine_plus_assoc":
        sel = table.filter(charged | gtpos | assoc)
    else:
        raise ValueError(f"unknown population {kind!r}")
    return sel.unique(subset=["crop", "mother", "d1", "d2", "label"],
                      keep="first", maintain_order=True)


def deployment_rows(table: pl.DataFrame, fold: int) -> pl.DataFrame:
    """The forks the metric CHARGES on the judged fold - tp_fork against fp_fork, nothing else."""
    return table.filter((pl.col("fold") == fold)
                        & pl.col("official_label").is_in(["tp_fork", "fp_fork"]))


def all_emitted_forks(table: pl.DataFrame, fold: int) -> pl.DataFrame:
    """Every fork the pipeline emitted on the judged fold - what a deployed verifier actually sees.

    FACT-0383: 98.7% of these are invisible to the metric, and a rejection still deletes an edge.
    """
    return table.filter((pl.col("fold") == fold) & (pl.col("source") == "pipeline_fork"))


# ============================================================ fitting
def fit_calibrated(kind: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Fit, and Platt-calibrate on OUT-OF-FOLD predictions grouped by crop.

    Grouping by crop is not decoration: frames of one crop are not independent, so a random split
    would let the calibrator see the same cell twice and read as better than it is.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold

    n_groups = len(set(groups.tolist()))
    n_pos = int(y.sum())
    n_splits = int(min(4, n_groups, max(2, n_pos)))
    oof = np.full(len(y), np.nan)
    if n_splits >= 2 and n_pos >= 2:
        sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        for tr, te in sgk.split(x, y, groups):
            if len(set(y[tr].tolist())) < 2:
                continue
            m = _make(kind)
            m.fit(_clean(x[tr]), y[tr])
            oof[te] = m.predict_proba(_clean(x[te]))[:, 1]
    full = _make(kind)
    full.fit(_clean(x), y)
    ok = np.isfinite(oof)
    if ok.sum() >= 8 and len(set(y[ok].tolist())) == 2:
        platt = LogisticRegression(max_iter=1000).fit(oof[ok].reshape(-1, 1), y[ok])
    else:
        platt = None
    return full, platt, oof


def apply_model(full, platt, x: np.ndarray) -> np.ndarray:
    p = full.predict_proba(_clean(x))[:, 1]
    if platt is not None:
        p = platt.predict_proba(p.reshape(-1, 1))[:, 1]
    return p


def calibrate_only(platt, p: np.ndarray) -> np.ndarray:
    return platt.predict_proba(np.asarray(p, dtype=float).reshape(-1, 1))[:, 1] if platt is not None else p


# ============================================================ operating points
def oracle_cut(y: np.ndarray, s: np.ndarray) -> dict:
    """How many false forks could be cut before the FIRST true one is lost, on this fold's labels.

    An ORACLE - it reads the answer to pick the cut. Reported because it is the quantity FACT-0385
    reports and the two attempts have to be comparable; it is never an operating point.
    """
    order = np.argsort(s, kind="stable")
    seen_fp, at_one_tp, cum_fp, lost = 0, 0, 0, 0
    first_done = False
    for j in order:
        if y[j] == 1:
            lost += 1
            if lost == 1:
                seen_fp, first_done = cum_fp, True
            if lost == 2:
                at_one_tp = cum_fp
                break
        else:
            cum_fp += 1
    if not first_done:
        seen_fp = cum_fp
    n_fp = int(len(y) - y.sum())
    n_tp = int(y.sum())
    expect = n_fp / (n_tp + 1.0) if n_tp else float("nan")
    out = {
        "false_forks": n_fp, "true_forks": n_tp,
        "fp_rejected_before_losing_any_tp": int(seen_fp),
        "fp_rejected_before_losing_a_second_tp": int(at_one_tp),
        "random_expectation": round(float(expect), 2),
        "above_chance": bool(n_tp and seen_fp > expect),
        "is_an_oracle": True,
    }
    out.update(permutation_null(n_tp, n_fp, int(seen_fp)))
    return out


def permutation_null(n_tp: int, n_fp: int, observed: int) -> dict:
    """The EXACT null for "how many false forks sort below the first true one". Closed form.

    Under a uniformly random ranking the statistic is the minimum rank of the n_tp true forks
    among n = n_tp + n_fp, and P(min >= k) = C(n-k, n_tp) / C(n, n_tp) exactly. No simulation, so
    no seed and no Monte-Carlo error.

    This is the packet's first-class treatment of the small positive population, not a footnote.
    At 5 true forks (fold 0) and 14 (fold 1) the statistic is coarse: comparing it to a single
    expected value hides how often a random ranking beats the observed count outright.
    """
    from math import comb

    if n_tp <= 0 or n_fp <= 0:
        return {"exact_null": {"skipped": "a class is empty"}}
    n = n_tp + n_fp
    denom = comb(n, n_tp)

    def surv(k: int) -> float:                       # P(min rank >= k)
        if k <= 0:
            return 1.0
        if n - k < n_tp:
            return 0.0
        return comb(n - k, n_tp) / denom

    p = surv(int(observed))
    median = next(k for k in range(0, n - n_tp + 2) if surv(k + 1) <= 0.5)
    p95 = next(k for k in range(0, n - n_tp + 2) if surv(k + 1) <= 0.05)
    return {"exact_null": {
        "median_under_null": int(median),
        "p95_under_null": int(p95),
        "one_sided_p": round(float(p), 4),
        "significant_at_0.05": bool(p < 0.05),
        "reading": ("P(a uniformly random ranking rejects at least this many false forks before "
                    "the first true one), computed exactly. With 5 and 14 positives this statistic "
                    "is coarse and a random ranking is often lucky - which is why it is measured"),
    }}


def threshold_from_training(y: np.ndarray, p_oof: np.ndarray) -> float:
    """The largest threshold that loses NO training positive, from out-of-fold predictions only."""
    ok = np.isfinite(p_oof) & (y == 1)
    if not ok.any():
        return 0.0
    return float(np.min(p_oof[ok]))


# The threshold family, declared before any run. `zero_loss` is the only one that satisfies the
# packet's primary constraint on the TRAINING fold; the quantile thresholds deliberately spend
# training positives so the shape of the trade is visible instead of guessed at. Each is reported
# with what it costs, and a judged-fold true fork lost is a FAILURE of that point, never a cost of
# doing business.
THRESHOLD_QUANTILES = (0.0, 0.05, 0.10, 0.25)


def threshold_family(y: np.ndarray, p_oof: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(p_oof) & (y == 1)
    if not ok.any():
        return {"zero_loss": 0.0}
    pos = np.sort(p_oof[ok])
    out = {"zero_loss": float(pos[0])}
    for q in THRESHOLD_QUANTILES[1:]:
        out[f"train_q{int(q * 100):02d}"] = float(np.quantile(pos, q))
    return out


def operating_point(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    rej = p < thr
    lost = int(((y == 1) & rej).sum())
    return {
        "threshold": round(float(thr), 6),
        "forks_rejected": int(rej.sum()),
        "true_forks_lost": lost,
        "false_forks_cut": int(((y == 0) & rej).sum()),
        "preserves_every_true_fork": bool(lost == 0),
        "verdict": ("OK" if lost == 0 else
                    f"FAILS the primary constraint - costs {lost} true fork(s)"),
    }


# ============================================================ commands
def _load(rows_dir: Path) -> pl.DataFrame:
    tabs = []
    for fold in (0, 1):
        p = rows_dir / f"rows_f{fold}.parquet"
        if not p.exists():
            raise RuntimeError(f"missing {p} - run `divverify_dataset.py census` for fold {fold}")
        tabs.append(pl.read_parquet(p))
    return pl.concat(tabs, how="vertical")


def cmd_route_audit(args) -> int:
    """Can the CONSTRUCTION ROUTE still be read off the features? The test, not the assertion."""
    table = _load(Path(args.rows_dir))
    out = {"schema_version": 1, "heartbeat": "DIVVERIFY_ROUTE_AUDIT_COMPLETE",
           "what_this_tests": (
               "FACT-0385 diagnosed that pooling GT-derived tuples with emitted forks inflated the "
               "unseen-fold AUC because out_degree and competing_parents read the construction "
               "route. Removing them is a mechanism; this is the test of whether it worked. A "
               "route AUC near 0.5 in the route_neutral space is what makes the enlarged positive "
               "class admissible. A high one says do not read the enlarged arms."),
           "route_discriminability": {}, "fold_discriminability": {}, "verdict": {}}

    # THE CONTRAST HAS TO HOLD THE LABEL FIXED. Comparing all emitted forks against the GT-derived
    # tuples answers a different question - those two groups differ in LABEL, and a feature that
    # separates them is doing its job. The route question is whether, AMONG ROWS OF THE SAME LABEL,
    # a model can still tell how the row was produced. If it can, then because every natural
    # negative arrives by the emitted route, "looks GT-derived" becomes a proxy for "positive" and
    # the enlarged class inflates exactly the way FACT-0385 measured.
    tp = table.filter(pl.col("official_label") == "tp_fork")
    gtpos = table.filter(pl.col("official_label") == "gt_positive")
    fp = table.filter(pl.col("official_label") == "fp_fork")
    cfneg = table.filter(pl.col("source") == "counterfactual")
    for fold in (0, 1):
        for tag, a_, b_, an, bn in (
                ("positives", tp.filter(pl.col("fold") == fold), gtpos.filter(pl.col("fold") == fold),
                 "emitted_true_forks", "gt_derived_tuples"),
                ("negatives", fp.filter(pl.col("fold") == fold), cfneg.filter(pl.col("fold") == fold),
                 "natural_false_forks", "constructed_negatives")):
            if not a_.height or not b_.height:
                continue
            y = np.r_[np.ones(a_.height), np.zeros(b_.height)]
            for fs, cols in FEATURE_SETS.items():
                x = np.vstack([a_.select(cols).to_numpy().astype(float),
                               b_.select(cols).to_numpy().astype(float)])
                per = {c: round(_auc(y, _clean(x[:, i])), 4) for i, c in enumerate(cols)}
                worst = max(per.items(), key=lambda kv: abs(kv[1] - 0.5))
                out["route_discriminability"][f"fold_{fold}_{tag}_{fs}"] = {
                    "label_held_fixed": tag,
                    an: int(a_.height), bn: int(b_.height),
                    "most_route_revealing_feature": {
                        "feature": worst[0], "auc": worst[1],
                        "distance_from_chance": round(abs(worst[1] - 0.5), 4)},
                    "features_beyond_0.75_or_below_0.25": sorted(
                        [k for k, v in per.items() if v > 0.75 or v < 0.25]),
                    "per_feature_auc": per,
                }

    # THE DISTRIBUTIONAL EVIDENCE, because an AUC on 5 rows is not a picture. If the emitted true
    # forks sit with the emitted FALSE forks and the GT-derived tuples sit apart, then enlarging
    # the positive class imports a shape the deployed generator cannot produce - which is a
    # population mismatch, not a feature-selection problem, and no route-neutral space fixes it.
    out["class_geometry"] = {}
    for fold in (0, 1):
        block = {}
        for tag, sel in (("emitted_true_forks", tp), ("emitted_false_forks", fp),
                         ("gt_derived_tuples", gtpos)):
            s = sel.filter(pl.col("fold") == fold)
            if not s.height:
                continue
            block[tag] = {"n": int(s.height)}
            for c in ("sister_um", "parent_midpoint_um", "pd2_um", "daughter_angle",
                      "sister_over_nn"):
                v = _clean(s[c].to_numpy().astype(float))
                block[tag][c] = {"median": round(float(np.median(v)), 3),
                                 "q25": round(float(np.percentile(v, 25)), 3),
                                 "q75": round(float(np.percentile(v, 75)), 3),
                                 "max": round(float(np.max(v)), 3)}
        out["class_geometry"][f"fold_{fold}"] = block

    # Fold discriminability - a DOMAIN-SHIFT measurement, not a leak. It is the standing candidate
    # mechanism for the FACT-0385 inversion, so it is measured rather than speculated about.
    for tag, sel in (("charged_forks", table.filter(pl.col("official_label").is_in(["tp_fork", "fp_fork"]))),
                     ("positives_only", table.filter(pl.col("label") == 1)),
                     ("emitted_forks", table.filter(pl.col("source") == "pipeline_fork"))):
        y = (sel["fold"].to_numpy() == 1).astype(int)
        if len(set(y.tolist())) < 2:
            continue
        per = {c: round(_auc(y, _clean(sel[c].to_numpy().astype(float))), 4)
               for c in dv.FEATURES}
        worst = max(per.items(), key=lambda kv: abs(kv[1] - 0.5))
        out["fold_discriminability"][tag] = {
            "rows": int(sel.height),
            "most_embryo_specific_feature": {"feature": worst[0], "auc": worst[1]},
            "features_beyond_0.75_or_below_0.25": sorted(
                [k for k, v in per.items() if v > 0.75 or v < 0.25]),
            "per_feature_auc": per,
        }

    def _worst(pred):
        return max((v["most_route_revealing_feature"]["distance_from_chance"]
                    for k, v in out["route_discriminability"].items() if pred(k)),
                   default=float("nan"))
    pos_rn = _worst(lambda k: "positives" in k and k.endswith("route_neutral"))
    pos_all = _worst(lambda k: "positives" in k and k.endswith("_all"))
    neg_rn = _worst(lambda k: "negatives" in k and k.endswith("route_neutral"))
    out["verdict"] = {
        "positives_route_signal_all_features": pos_all,
        "positives_route_signal_route_neutral": pos_rn,
        "negatives_route_signal_route_neutral": neg_rn,
        "enlarged_positive_class_admissible": bool(pos_rn < 0.25),
        "reading": ("distance_from_chance is |AUC-0.5| of the single most route-revealing feature "
                    "with the LABEL HELD FIXED. Below 0.25 (AUC inside [0.25, 0.75]) the enlarged "
                    "positive class is read as admissible; above it the pooled arms are reported "
                    "but not believed, and the reason is a population mismatch rather than a "
                    "feature that can be dropped."),
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "route_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["verdict"], indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_feature_auc"}
                      for k, v in out["route_discriminability"].items()}, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_feature_auc"}
                      for k, v in out["fold_discriminability"].items()}, indent=2))
    return 0


def cmd_crossfold(args) -> int:
    t_start = time.time()
    table = _load(Path(args.rows_dir))
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "heartbeat": "DIVVERIFY_CROSSFOLD_COMPLETE",
              "selection_rule": SELECTION_RULE,
              "external_rows_used": False,
              "why_no_external": (
                  "FACT-0385: Zebrahub is anti-aligned with our surface at pooled AUC 0.179 and "
                  "collapsed an in-domain fit from 0.80 to 0.21. LEVER-0040 forbids its use until "
                  "that is solved; this packet did not solve it, so it is not used."),
              "folds": {}}
    rejection_rows: list[dict] = []

    for judged in (0, 1):
        train_all = dv.training_rows_for(table, judged)
        dv.assert_no_leak(train_all, judged)

        # A FIFTH FEATURE SET, CHOSEN ON THE TRAINING FOLD AND NOWHERE ELSE. Rank every feature by
        # |AUC - 0.5| over the TRAINING fold's charged forks - the same decision the verifier makes,
        # measured on the embryo it is allowed to see - and keep the top TOP_K. k is fixed a priori;
        # only the identity of the features is learned, and it is learned from data the judged fold
        # never touches. The selected list is written into the report so it can be checked.
        fold_sets = dict(FEATURE_SETS)
        tr_charged = train_all.filter(pl.col("official_label").is_in(["tp_fork", "fp_fork"]))
        selected: list[str] = []
        if tr_charged.height and 0 < int(tr_charged["label"].sum()) < tr_charged.height:
            ytc = tr_charged["label"].to_numpy().astype(int)
            ranked = sorted(
                dv.FEATURES,
                key=lambda c: -abs(_auc(ytc, _clean(tr_charged[c].to_numpy().astype(float))) - 0.5))
            selected = ranked[:TOP_K]
            fold_sets[f"train_top{TOP_K}"] = selected

        dep = deployment_rows(table, judged)
        emitted = all_emitted_forks(table, judged)
        y_dep = dep["label"].to_numpy().astype(int)
        entry = {
            "trains_on_fold": sorted(set(train_all["fold"].to_list())),
            "trains_on_embryo": sorted(set(train_all["embryo"].to_list())),
            "judged_embryo": dv.FOLD_EMBRYO[judged],
            "charged_forks_judged": int(dep.height),
            "true_forks_judged": int(y_dep.sum()),
            "all_emitted_forks_judged": int(emitted.height),
            "train_selected_features": selected,
            "arms": {},
        }
        best = None
        for pop in POPULATIONS:
            tr = population_rows(train_all, pop)
            dv.assert_no_leak(tr, judged)
            ytr = tr["label"].to_numpy().astype(int)
            groups = np.asarray(tr["crop"].to_list())
            if ytr.sum() < 2 or ytr.sum() == len(ytr):
                entry["arms"][pop] = {"skipped": "fewer than two positives in the training rows"}
                continue
            tr_dep_mask = np.array([lab in ("tp_fork", "fp_fork")
                                    for lab in tr["official_label"].to_list()])
            for fs, cols in fold_sets.items():
                xtr = tr.select(cols).to_numpy().astype(float)
                for kind in MODELS:
                    full, platt, oof = fit_calibrated(kind, xtr, ytr, groups)
                    oof_cal = calibrate_only(platt, oof)
                    ok = np.isfinite(oof_cal)
                    inner_dep = tr_dep_mask & ok
                    inner_auc = (_auc(ytr[inner_dep], oof_cal[inner_dep])
                                 if inner_dep.sum() and 0 < ytr[inner_dep].sum() < inner_dep.sum()
                                 else float("nan"))
                    fam = threshold_family(ytr, oof_cal)
                    thr = fam["zero_loss"]

                    p_dep = apply_model(full, platt, dep.select(cols).to_numpy().astype(float))
                    p_all = apply_model(full, platt, emitted.select(cols).to_numpy().astype(float))
                    y_all = emitted["label"].to_numpy().astype(int)
                    name = f"{pop}|{fs}|{kind}"
                    arm = {
                        "train_rows": int(tr.height), "train_positives": int(ytr.sum()),
                        "n_features": len(cols),
                        "inner_oof_deployment_auc_training_fold": (
                            None if inner_auc != inner_auc else round(float(inner_auc), 4)),
                        "inner_oof_preserves_all_training_positives": bool(
                            np.isfinite(oof_cal[ytr == 1]).all() and thr > 0.0),
                        "auc_on_judged_fold_charged_forks": round(_auc(y_dep, p_dep), 4),
                        "oracle_threshold": oracle_cut(y_dep, p_dep),
                        "transferred_thresholds_charged_only_ORACLE_POPULATION": {
                            k: operating_point(y_dep, p_dep, v) for k, v in fam.items()},
                        "transferred_thresholds_all_emitted_forks": {
                            **{k: operating_point(y_all, p_all, v) for k, v in fam.items()},
                            "note": ("the DEPLOYABLE rule. Every rejection deletes an edge, "
                                     "including the ~98.7% of forks the metric ignores "
                                     "(FACT-0383), so the edge cost is measured by the adapter "
                                     "and not by this count"),
                        },
                        "threshold_note": (
                            "zero_loss is the only point that preserves every TRAINING positive; "
                            "the train_qNN points deliberately spend training positives to show "
                            "the shape of the trade and are not deployable as they stand"),
                    }
                    entry["arms"][name] = arm
                    # A POSITIVE HEARTBEAT, so its ABSENCE is the alarm. This grid is 80 arms of
                    # gradient boosting on a contended CPU; a run that prints only at the end is
                    # indistinguishable from a hung one, which cost this packet one killed run.
                    print(f"  judge f{judged} {name:44s} "
                          f"inner_auc={arm['inner_oof_deployment_auc_training_fold']} "
                          f"judged_auc={arm['auc_on_judged_fold_charged_forks']} "
                          f"oracle_cut={arm['oracle_threshold']['fp_rejected_before_losing_any_tp']}"
                          f"/{arm['oracle_threshold']['false_forks']} "
                          f"p={arm['oracle_threshold']['exact_null'].get('one_sided_p')} "
                          f"({time.time() - t_start:.0f}s)", flush=True)

                    qualifies = (arm["inner_oof_preserves_all_training_positives"]
                                 and inner_auc == inner_auc)
                    if qualifies and (best is None or inner_auc > best[0]):
                        best = (float(inner_auc), name, fam, p_all, p_dep)

        if best is None:
            entry["selected_arm"] = None
            entry["selected_note"] = ("no arm qualified under SELECTION_RULE - nothing is scored "
                                      "end-to-end for this fold")
        else:
            inner_auc, name, fam, p_all, p_dep = best
            entry["selected_arm"] = {"arm": name, "inner_oof_deployment_auc": round(inner_auc, 4),
                                     "thresholds": {k: round(float(v), 6) for k, v in fam.items()}}
            # Two rejection sets go to the adapter and no more, because each costs a full re-score
            # of the fold: the deployable zero-loss point, and one quantile point that deliberately
            # spends training positives so the trade has a measured shape rather than an argued one.
            emit = (("zero_loss", "selected_all_emitted_zero_loss"),
                    ("train_q10", "selected_all_emitted_q10"))
            crops = emitted["crop"].to_list()
            mothers = emitted["mother"].to_list()
            for key, arm_name in emit:
                if key not in fam:
                    continue
                thr_k = fam[key]
                for c, m, p in zip(crops, mothers, p_all):
                    rejection_rows.append({"fold": judged, "arm": arm_name,
                                           "crop": c, "mother": int(m), "score": float(p),
                                           "reject": bool(p < thr_k)})
            dcrops = dep["crop"].to_list()
            dmoth = dep["mother"].to_list()
            for c, m, p in zip(dcrops, dmoth, p_dep):
                rejection_rows.append({"fold": judged, "arm": "selected_charged_only_ORACLE",
                                       "crop": c, "mother": int(m), "score": float(p),
                                       "reject": bool(p < fam["zero_loss"])})
        report["folds"][f"judge_fold_{judged}"] = entry

    if rejection_rows:
        pl.DataFrame(rejection_rows).write_parquet(outdir / "rejections.parquet")
        report["rejections"] = str(outdir / "rejections.parquet")
    (outdir / "crossfold.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def cmd_leak_demo(args) -> int:
    """DEMONSTRATE the leakage contract instead of asserting it: measure what breaking it buys.

    Three arms per judged fold, on one fixed configuration so only the training set differs:

      honest    trained on the opposite embryo only - the contract as written.
      leaked    trained on the opposite embryo PLUS the judged fold's own rows. A DELIBERATE
                VIOLATION, run here and nowhere else, and never reported as a result.
      guard     the same leaked table handed to ``assert_no_leak``, which must RAISE.

    The honest-to-leaked gap is the contract's measured value. If it were near zero the contract
    would be costing nothing and buying nothing; if it is large, every unguarded number in this
    lane would have been optimistic by roughly that much. FACT-0378 is the standing warning that
    this is not hypothetical.
    """
    table = _load(Path(args.rows_dir))
    cols = FEATURE_SETS[args.feature_set]
    out = {"schema_version": 1, "heartbeat": "DIVVERIFY_LEAK_DEMO_COMPLETE",
           "configuration": {"population": args.population, "feature_set": args.feature_set,
                             "model": args.model},
           "what_this_is": ("a POSITIVE CONTROL for the leakage guard. The leaked arm is a "
                            "deliberate violation used only to price the contract; it is never a "
                            "result and never enters any other report"),
           "folds": {}}
    for judged in (0, 1):
        dep = deployment_rows(table, judged)
        y_dep = dep["label"].to_numpy().astype(int)
        xdep = dep.select(cols).to_numpy().astype(float)
        arms = {}
        honest_tr = population_rows(dv.training_rows_for(table, judged), args.population)
        dv.assert_no_leak(honest_tr, judged)
        leaked_tr = population_rows(table, args.population)   # the judged fold is NOT removed
        guard_fired = False
        try:
            dv.assert_no_leak(leaked_tr, judged)
        except RuntimeError as exc:
            guard_fired = True
            guard_msg = str(exc)[:200]
        for tag, tr in (("honest", honest_tr), ("leaked", leaked_tr)):
            ytr = tr["label"].to_numpy().astype(int)
            if ytr.sum() < 2:
                arms[tag] = {"skipped": "fewer than two positives"}
                continue
            full, platt, _oof = fit_calibrated(
                args.model, tr.select(cols).to_numpy().astype(float), ytr,
                np.asarray(tr["crop"].to_list()))
            p = apply_model(full, platt, xdep)
            arms[tag] = {
                "train_rows": int(tr.height), "train_positives": int(ytr.sum()),
                "auc_on_judged_fold_charged_forks": round(_auc(y_dep, p), 4),
                "fp_rejected_before_losing_any_tp":
                    oracle_cut(y_dep, p)["fp_rejected_before_losing_any_tp"],
            }
        gap = None
        if "auc_on_judged_fold_charged_forks" in arms.get("honest", {}) and \
                "auc_on_judged_fold_charged_forks" in arms.get("leaked", {}):
            gap = round(arms["leaked"]["auc_on_judged_fold_charged_forks"]
                        - arms["honest"]["auc_on_judged_fold_charged_forks"], 4)
        out["folds"][f"judge_fold_{judged}"] = {
            "judged_embryo": dv.FOLD_EMBRYO[judged],
            "arms": arms,
            "auc_bought_by_leaking": gap,
            "guard_raised_on_the_leaked_table": guard_fired,
            "guard_message": guard_msg if guard_fired else None,
        }
        if not guard_fired:
            raise RuntimeError(
                f"judge fold {judged}: assert_no_leak did NOT raise on a training table that "
                f"contains the judged fold. The leakage guard is a no-op and every number in this "
                f"lane is void until it is fixed")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "leak_demo.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("leak_demo")
    d.add_argument("--rows-dir", required=True)
    d.add_argument("--out-dir", required=True)
    d.add_argument("--population", default="genuine_tuples", choices=list(POPULATIONS))
    d.add_argument("--feature-set", default="route_neutral", choices=list(FEATURE_SETS))
    d.add_argument("--model", default="tree", choices=list(MODELS))
    d.set_defaults(func=cmd_leak_demo)
    r = sub.add_parser("route_audit")
    r.add_argument("--rows-dir", required=True)
    r.add_argument("--out-dir", required=True)
    r.set_defaults(func=cmd_route_audit)
    c = sub.add_parser("crossfold")
    c.add_argument("--rows-dir", required=True)
    c.add_argument("--out-dir", required=True)
    c.set_defaults(func=cmd_crossfold)
    args = ap.parse_args()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
