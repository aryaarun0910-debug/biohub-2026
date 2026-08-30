r"""THE ASSOCIATION MODEL TOURNAMENT - five scorer classes on ONE data contract (PKT-0038 / LEVER-0041).

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
It is a DRIVER. Every number it produces is computed by ``assoc_train_harness.run_harness`` over the
frozen surface of ``assoc_parent_dataset`` and decomposed by ``assoc_report``. This module owns
exactly three things the harness deliberately does not:

  1. the DERIVED BIDIRECTIONAL COLUMNS (label-free, deterministic, computed from the frozen table
     and never written back into it),
  2. the FITTING PROCEDURES that are themselves under test - an iterative hard-negative curriculum,
     an alternate-seed secondary, and a model frozen on one fold and applied unchanged to another,
  3. the FREEZE/APPLY discipline that makes the fold-0 result a transfer measurement rather than a
     second search.

It does not fork the harness, does not touch ``assoc_parent_dataset``'s surface or label contract,
and cannot promote anything: ``harness_verdict`` blocks on the missing full-chain arms by design.

FOLD 1 IS THE DISCOVERY SURFACE. FOLD 0 IS THE GATE. THEY ARE NEVER SWAPPED.
---------------------------------------------------------------------------
``FACT-0388`` is the whole reason: on the widened P34 surface fold 1 carries 26,601 contested
targets and 7,757 errors at deployed contested top-1 0.7084, against fold 0's 4,157 / 691 / 0.8338 -
11x the errors at a materially lower deployed accuracy. ``FACT-0386``'s best arm moved 12 of fold
0's 691 with churn 40 and an interval including zero, i.e. fold 0 is too thin to separate skill
from binning noise. An effect invisible on fold 0 can be measured on fold 1.

THE SELECTION RULE, DECLARED BEFORE ANY ARM WAS RUN
---------------------------------------------------
``PKT-0038`` falsifier (g) is the one most likely to bite: five model classes with hard-negative
mining over one discovery fold is a large search, and a winner picked after seeing fold 0 would be
selection wearing skill's clothes. So:

  * The winner is chosen on FOLD 1 ONLY, by the highest contested top-1 among arms whose crop-paired
    ci95 lower bound is strictly positive.
  * If NO arm is favourable on fold 1, the tournament HAS NO WINNER. The highest-scoring arm is
    still frozen and transferred, but the frozen record carries ``favourable_on_discovery: false``
    and every downstream payload inherits it, so a fold-0 number cannot later be read as a
    promotion candidate. A null discovery is a result, not a licence to keep searching.
  * ``freeze`` writes the architecture, the hyperparameters, the mining curriculum, the abstention
    threshold and a pickled fold-1 fit, then pins all of it by sha256. ``apply`` re-verifies those
    digests and REFUSES if any moved. There is no ``--force``.

FOLD 0 IS MEASURED TWICE, AND THE TWO ANSWER DIFFERENT QUESTIONS
-----------------------------------------------------------------
  GATE A - ARCHITECTURE TRANSFER. The identical recipe, refitted out of fold on fold 0's own
           crop-grouped splits. This is exactly the protocol that produced ``FACT-0386``, so it is
           directly comparable to the killed LEVER-0039 arms. It does NOT test falsifier (e).
  GATE B - WEIGHT TRANSFER, and it is the strict one. The fold-1 fit is applied to fold 0 with NO
           fold-0 fitting of any kind: ``FrozenTransferHead.fit`` is a no-op that refuses to learn,
           and the abstention policy is ``fixed`` at the threshold frozen on fold 1, so there is no
           refit, no threshold nudge and no per-fold calibration. This is the honest reading of
           "apply UNCHANGED", it is also a cross-EMBRYO measurement (fold 1 is all 6bba, fold 0 all
           44b6), and it is the number falsifiers (e) and (g) are settled on.

WHAT ITERATIVE MINING CAN AND CANNOT DO HERE - STATED BEFORE THE RESULT
-----------------------------------------------------------------------
The harness fits on the DECIDABLE rows only (``true_parent_is_candidate == 1``), which is the frozen
PKT-0034 protocol and the reason these arms are comparable to ``FACT-0386`` at all. On fold 1 that
is 122,820 rows for 91,410 targets: 91,410 positives and 31,410 negatives, at most three wrong
candidates per target. So mining here is REWEIGHTING WITHIN THE DECIDABLE CANDIDATE POOL, not the
acquisition of new negatives from the wider 3.7M-row surface. Four categories are mined, each round
recorded to a curriculum log so the schedule is reproducible rather than emergent:

  wrong_nearby_parent     a wrong candidate CLOSER to the target than the true parent is
  high_conf_substitution  a wrong candidate the current model scores at or above the true parent
  motion_consistent       a wrong candidate whose displacement is unremarkable (<= the training
                          positives' 75th percentile), i.e. one geometry cannot reject
  division_producing      a wrong candidate whose SOURCE already wins another target, so choosing
                          it makes that source a two-child fork - the FACT-0371 failure mode, mined
                          as a negative and NEVER read as division recovery

Mining outside the decidable pool is a PREPARED upgrade, not a silent default: it changes the
training population and would break comparability with the arms it must be judged against.

THE BLOCKED ARM, AND EXACTLY WHAT IT NEEDS
-------------------------------------------
Arm 3, the contextual transformer / HOCT-style head, is the only arm that cannot run on CPU. It is
written as a cache-gated spec by ``prepare`` and the harness refuses it until a Gate-1 licence
exists (``FACT-0387``); ``FACT-0392`` adds that HOCT's own head cannot rerank our candidate list at
all, so the contextual arm here is a head trained on OUR contract over the licensed cache's frozen
trunk features - not a port of theirs.

READING THE RESULT
------------------
``parent_conversions`` is a PRE-ILP quantity. ``FACT-0364`` measured that motion relink replaces the
solver's entire edge list at a median 99.9% coverage, so a better pre-ILP ranking has a documented
history of not surviving to the final graph; the genuine ``FACT-0376`` quantity is final-graph edge
TP/FP/FN and it is UNMEASURED here. ``FACT-0382`` adds the second condition: any contested gain
lives on the WIDENED surface, so a full-chain comparison must be (widened + scorer) against the
DEPLOYED control, never against a widened control. Both are why every verdict this module emits is
``promotable: false``.

COMMANDS
--------
    python scripts/win_bet/assoc_tournament.py discover --out-dir C:/temp/assoc_tournament
    python scripts/win_bet/assoc_tournament.py noisefloor --out-dir C:/temp/assoc_tournament
    python scripts/win_bet/assoc_tournament.py sideb    --out-dir C:/temp/assoc_tournament
    python scripts/win_bet/assoc_tournament.py freeze   --out-dir C:/temp/assoc_tournament
    python scripts/win_bet/assoc_tournament.py apply    --out-dir C:/temp/assoc_tournament
    python scripts/win_bet/assoc_tournament.py prepare  --out-dir C:/temp/assoc_tournament
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from assoc_baseline_rankers import LINEAR_KW, SEED, TREE_KW  # noqa: E402
from assoc_parent_dataset import FEATURES  # noqa: E402  the FROZEN feature contract
from assoc_train_harness import (  # noqa: E402  USED, NOT FORKED
    AbstainPolicy,
    HarnessRefusal,
    ModelSpec,
    decide,
    fit_out_of_fold,
    make_splitter,
    run_harness,
)

# ----------------------------------------------------------------------------------------------
# The substrate, bound by path so a run cannot silently read a different surface.
# ----------------------------------------------------------------------------------------------
#
# PROVENANCE, AND WHY THE PATH MOVED. The widened fold-1 target table was first materialised into
# ``C:/temp/audit_receipts/``, which is PKT-0036's AUDIT SCRATCH - another packet's private working
# area, and one a routine auditor clean-up would empty. Losing it costs a GPU acquisition to
# rebuild, not a re-run, because it is the surface FACT-0388 rests on. So it now lives in the
# ACQUISITION's own directory, which is the artifact of record, and this module reads nothing from
# the audit scratch. The relocated copy was verified byte-identical by sha256 before the move was
# adopted, and ``surface_provenance()`` re-records that digest into every payload so the table a
# number came from is explicit rather than inferred.
DISCOVERY_TABLE = Path("C:/temp/p34_f1/assoc_targets_split1.parquet")  # FACT-0388, P34 fold 1
DISCOVERY_SOURCE = {"preilp": "C:/temp/p34_f1/preilp_split1.parquet",
                    "ecb_dir": "C:/temp/p34_f1/ecb",
                    "receipt": "_evidence/audit/receipts/f1_p34.json",
                    "floor": 0.1, "rank_cap": 4}
GATE_TABLE = Path("C:/temp/assoc/f0.parquet")                     # FACT-0381/0386 fold 0
GATE_SOURCE = {"preilp": "C:/temp/p30_f0/preilp_split0.parquet",
               "ecb_dir": "C:/temp/p30_f0/ecb", "floor": 0.1, "rank_cap": 4}
DISCOVERY_FOLD, GATE_FOLD = 1, 0

META = ["group_uid", "src_uid"]        # identity carried through fit(); NEVER a model feature
BIDIR = ["fwd_prob_norm", "rev_prob_norm", "bidir_consensus",
         "rev_rank", "rev_margin_to_best", "is_mutual_best"]

MINING_ROUNDS = 3                      # fits; mining happens between them
HARD_NEG_BOOST = 4.0
SECONDARY_SEED = SEED + 104729         # a prime offset, declared, never searched
ABSTAIN_Q = 0.05

# Every ``declared`` estimator appends its per-round curriculum here. The harness constructs a
# fresh estimator per crop-grouped fold and then discards it, so a module-level collector is the
# only way the curriculum reaches disk - and the packet requires it recorded, not emergent.
MINING_LOG: list[dict] = []


def _module_copies() -> list:
    """Every live copy of THIS module, because a scripted run has two of them.

    MEASURED DEFECT, 2026-08-30, and it bit twice in one session.
    ``assoc_train_harness.resolve_callable`` resolves a declared head by
    ``importlib.import_module("assoc_tournament")``. When this file is run AS A SCRIPT the
    factories therefore execute inside ``sys.modules['assoc_tournament']`` while ``run_fold``,
    ``freeze`` and ``apply_to_gate`` execute inside ``sys.modules['__main__']`` - two module
    objects, two sets of module state.

      * FIRST BITE, SILENT. The mining curriculum was appended to one copy and serialised from the
        other, so ``discovery_fold1.json`` carried ``"mining_curriculum": []`` while claiming to
        record it. That is the silent no-op AGENTS.md names as worse than a crash.
      * SECOND BITE, LOUD. ``bind_frozen_model`` set ``FrozenTransferHead._PICKLE`` on the
        ``__main__`` class while the harness constructed the IMPORTED class, whose ``_PICKLE`` was
        still ``None`` - so ``apply`` refused with "constructed with no frozen model bound". The
        fail-closed constructor is why this surfaced as a refusal instead of a transfer arm that
        quietly refit itself.

    This is the "does the state cross a process, thread or replica boundary?" question from
    AGENTS.md, one boundary in from ``FACT-0060``. Every function that mutates module state routes
    through here, so a third instance of the same bug cannot be written by accident.
    """
    # The imported copy is created LAZILY, by resolve_callable, at the first declared-head fit -
    # which is AFTER `apply` binds the frozen model. Binding only what exists at bind time is
    # therefore binding nothing that will be used, so the copy is forced into existence here.
    try:
        importlib.import_module("assoc_tournament")
    except Exception:                      # pragma: no cover - a missing copy is not fatal
        pass
    seen, out = [], []
    for name in ("__main__", "assoc_tournament"):
        mod = sys.modules.get(name)
        # A same-named module that is not this file (pytest's __main__, say) must not be touched.
        if mod is not None and getattr(mod, "HEAD", None) == HEAD and id(mod) not in seen:
            seen.append(id(mod))
            out.append(mod)
    return out or [sys.modules[__name__]]


def _mining_logs() -> list[list[dict]]:
    """Every live copy of ``MINING_LOG``. ``run_fold`` REFUSES to write an empty one when a mining
    arm ran, so the first bite above becomes a crash rather than a plausible-looking payload."""
    seen, out = [], []
    for mod in _module_copies():
        log = getattr(mod, "MINING_LOG", None)
        if isinstance(log, list) and id(log) not in seen:
            seen.append(id(log))
            out.append(log)
    return out or [MINING_LOG]


def collected_curriculum() -> list[dict]:
    """The merged curriculum across every module copy, ordered as fitted."""
    merged: list[dict] = []
    for log in _mining_logs():
        merged.extend(log)
    return merged


def clear_curriculum() -> None:
    for log in _mining_logs():
        log.clear()


# ==============================================================================================
# 1. DERIVED BIDIRECTIONAL COLUMNS - label-free, deterministic, never written back to the surface
# ==============================================================================================

def bidirectional_columns(table: pl.DataFrame) -> pl.DataFrame:
    """Add the reverse-direction and consensus columns. Uses NO label and NO fitted quantity.

    ``FACT-0369`` verified at source that the deployed rule takes a softmax over the SOURCE axis and
    thresholds, so the pipeline only ever asks "which parent for this target". The mirror question -
    "which child for this parent" - is present in the same candidate table and is never asked. These
    six columns ask it. ``is_mutual_best`` is the classical bidirectional-consensus test and is the
    single column that makes a later consensus claim measurable at all.

    Determinism matters because the frozen fold-1 fit is applied to fold 0: both folds must get the
    columns from this one function, or Gate B compares a model to a different representation.
    """
    for col in ("crop", "target", "source", "prob"):
        if col not in table.columns:
            raise HarnessRefusal(f"bidirectional_columns: frozen surface is missing {col!r}")
    if any(c in table.columns for c in BIDIR + META):
        raise HarnessRefusal("derived columns already present - refusing to overwrite a surface")

    t = table.with_columns([
        (pl.col("crop") + "|" + pl.col("target").cast(pl.Utf8)).alias("_tkey"),
        (pl.col("crop") + "|" + pl.col("source").cast(pl.Utf8)).alias("_skey"),
    ])
    t = t.with_columns([
        pl.col("prob").sum().over("_tkey").alias("_tsum"),
        pl.col("prob").sum().over("_skey").alias("_ssum"),
        pl.col("prob").max().over("_tkey").alias("_tmax"),
        pl.col("prob").max().over("_skey").alias("_smax"),
        pl.col("prob").rank("ordinal", descending=True).over("_skey").alias("rev_rank"),
        pl.col("_tkey").rank("dense").alias("group_uid"),
        pl.col("_skey").rank("dense").alias("src_uid"),
    ])
    t = t.with_columns([
        (pl.col("prob") / pl.col("_tsum")).alias("fwd_prob_norm"),
        (pl.col("prob") / pl.col("_ssum")).alias("rev_prob_norm"),
        (pl.col("_smax") - pl.col("prob")).alias("rev_margin_to_best"),
        ((pl.col("prob") >= pl.col("_tmax")) & (pl.col("prob") >= pl.col("_smax")))
        .cast(pl.Int64).alias("is_mutual_best"),
    ])
    t = t.with_columns(
        ((pl.col("fwd_prob_norm") * pl.col("rev_prob_norm")).sqrt()).alias("bidir_consensus")
    )
    out = t.drop(["_tkey", "_skey", "_tsum", "_ssum", "_tmax", "_smax"])
    out = out.with_columns([pl.col(c).cast(pl.Float64) for c in BIDIR + META])
    missing = [c for c in BIDIR + META if c not in out.columns]
    if missing:
        raise HarnessRefusal(f"bidirectional_columns failed to produce {missing}")
    return out


# ==============================================================================================
# 2. THE FITTING PROCEDURES UNDER TEST
# ==============================================================================================

def _base_estimator(kind: str, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "tree":
        kw = dict(TREE_KW)
        kw["random_state"] = seed
        return HistGradientBoostingClassifier(**kw)
    if kind == "linear":
        return Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(**LINEAR_KW))])
    raise HarnessRefusal(f"unknown base estimator {kind!r}")


def _fit_weighted(model, x, y, w):
    from sklearn.pipeline import Pipeline

    if w is None:
        model.fit(x, y)
    elif isinstance(model, Pipeline):
        model.fit(x, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    else:
        model.fit(x, y, sample_weight=w)
    return model


def mine_hard_negatives(feat: np.ndarray, y: np.ndarray, groups: np.ndarray, srcs: np.ndarray,
                        scores: np.ndarray, names: list[str]) -> dict:
    """The four mined categories. Training rows only - this is called from inside ``fit``.

    Returns boolean masks plus counts. Every category is a WRONG candidate the current model has a
    concrete reason to keep confusing, which is what makes the curriculum a curriculum rather than
    a random upweighting of negatives.
    """
    idx = {n: i for i, n in enumerate(names)}
    for need in ("dist_um",):
        if need not in idx:
            raise HarnessRefusal(f"mining needs feature {need!r}; got {names}")
    dist = feat[:, idx["dist_um"]].astype(np.float64)
    neg = y == 0
    uniq, inv = np.unique(groups, return_inverse=True)
    n_groups = len(uniq)

    pos_score = np.full(n_groups, -np.inf)
    pos_dist = np.full(n_groups, np.inf)
    pos_rows = np.nonzero(y == 1)[0]
    pos_score[inv[pos_rows]] = scores[pos_rows]
    pos_dist[inv[pos_rows]] = dist[pos_rows]

    # Which row currently wins each group - ties by LOWER SOURCE INDEX, the frozen contract rule 4.
    order = np.lexsort((srcs, -scores))
    winner = np.full(n_groups, -1, dtype=np.int64)
    for r in order:
        g = inv[r]
        if winner[g] < 0:
            winner[g] = r
    won_src, won_cnt = np.unique(srcs[winner[winner >= 0]], return_counts=True)
    wins_by_src = dict(zip(won_src.tolist(), won_cnt.tolist()))
    # A source that already wins a group makes a two-child fork if it also takes this target.
    own_win = np.zeros(len(y), bool)
    own_win[winner[winner >= 0]] = True
    fork = np.array([wins_by_src.get(s, 0) for s in srcs.tolist()], dtype=np.int64) - own_win

    finite_pos = pos_dist[np.isfinite(pos_dist)]
    typical = float(np.percentile(finite_pos, 75)) if len(finite_pos) else np.inf

    cats = {
        "wrong_nearby_parent": neg & (dist < pos_dist[inv]),
        "high_conf_substitution": neg & (scores >= pos_score[inv]),
        "motion_consistent": neg & (dist <= typical),
        "division_producing": neg & (fork > 0),
    }
    union = np.zeros(len(y), bool)
    for m in cats.values():
        union |= m
    return {
        "masks": cats,
        "union": union,
        "counts": {k: int(m.sum()) for k, m in cats.items()},
        "union_count": int(union.sum()),
        "negatives": int(neg.sum()),
        "typical_displacement_um_p75": typical,
    }


class MinedRanker:
    """Iterative hard-negative curriculum. sklearn-compatible on the two methods the harness uses.

    The harness passes ``X`` with ``META`` in the leading columns; they are identity, not signal,
    and are stripped before any fit or predict. Stripping is asserted, not assumed: a mis-declared
    feature list would otherwise train on ``group_uid`` and score perfectly by memorising targets.

    Every fit is on TRAINING ROWS ONLY - the harness's crop-grouped split has already been applied
    and its integrity asserted - so nothing here can see a validation crop.
    """

    def __init__(self, base: str, feature_names: list[str], n_meta: int = len(META),
                 rounds: int = MINING_ROUNDS, boost: float = HARD_NEG_BOOST, seed: int = SEED,
                 row_bootstrap: float = 0.0, tag: str = "mined"):
        self.base, self.feature_names, self.n_meta = base, list(feature_names), int(n_meta)
        self.rounds, self.boost, self.seed = int(rounds), float(boost), int(seed)
        self.row_bootstrap, self.tag = float(row_bootstrap), str(tag)

    def _split_meta(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if x.shape[1] != self.n_meta + len(self.feature_names):
            raise HarnessRefusal(
                f"{self.tag}: expected {self.n_meta} meta + {len(self.feature_names)} feature "
                f"columns, got {x.shape[1]} - the declared feature list does not describe this X"
            )
        return x[:, : self.n_meta], x[:, self.n_meta:]

    def fit(self, x: np.ndarray, y: np.ndarray):
        meta, feat = self._split_meta(np.asarray(x, dtype=np.float64))
        y = np.asarray(y).astype(np.int64)
        groups = meta[:, 0].astype(np.int64)
        srcs = meta[:, 1].astype(np.int64)

        w = np.ones(len(y), dtype=np.float64)
        if self.row_bootstrap > 0:
            # The alternate-seed secondary differs from the primary by SEED and by a resampled
            # training distribution, so its errors are not the primary's by construction. That
            # independence is the whole point of arm 5 - a consensus of two identical fits is not
            # a consensus.
            rng = np.random.default_rng(self.seed)
            keep = rng.random(len(y)) < self.row_bootstrap
            keep |= y == 1                     # never resample away a target's only positive
            w = np.where(keep, 1.0, 0.0)
            if w.sum() == 0:
                raise HarnessRefusal(f"{self.tag}: row bootstrap kept nothing")

        hard_cum = np.zeros(len(y), bool)
        rounds: list[dict] = []
        model = None
        for r in range(max(1, self.rounds)):
            model = _fit_weighted(_base_estimator(self.base, self.seed), feat, y, w)
            if r == self.rounds - 1:
                rounds.append({"round": r, "fit_rows": int((w > 0).sum()), "mined_new": 0,
                               "hard_cumulative": int(hard_cum.sum()), "terminal": True})
                break
            scores = model.predict_proba(feat)[:, 1]
            mined = mine_hard_negatives(feat, y, groups, srcs, scores, self.feature_names)
            new = int((mined["union"] & ~hard_cum).sum())
            hard_cum |= mined["union"]
            base_w = np.ones(len(y)) if self.row_bootstrap <= 0 else (w > 0).astype(float)
            w = base_w * (1.0 + self.boost * hard_cum)
            rounds.append({"round": r, "fit_rows": int((w > 0).sum()), "mined_new": new,
                           "hard_cumulative": int(hard_cum.sum()),
                           "categories": mined["counts"], "negatives": mined["negatives"],
                           "typical_displacement_um_p75": mined["typical_displacement_um_p75"],
                           "terminal": False})
        MINING_LOG.append({"tag": self.tag, "base": self.base, "seed": self.seed,
                           "boost": self.boost, "rounds": rounds,
                           "n_train_rows": int(len(y)), "n_positives": int(y.sum())})
        self.model_ = model
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        _meta, feat = self._split_meta(np.asarray(x, dtype=np.float64))
        return self.model_.predict_proba(feat)


class FrozenTransferHead:
    """A model frozen on the DISCOVERY fold and applied to the GATE fold. ``fit`` REFUSES to learn.

    This is the mechanical guarantee behind "no refitting, no threshold nudging, no per-fold
    calibration". The harness still runs its crop-grouped splits and asserts their integrity, but
    every fold's model is the same frozen object, so the out-of-fold scores ARE the frozen model's
    direct predictions and no fold-0 information reaches the weights.
    """

    _PICKLE: Path | None = None
    _SHA: str | None = None

    def __init__(self):
        if FrozenTransferHead._PICKLE is None:
            raise HarnessRefusal(
                "FrozenTransferHead was constructed with no frozen model bound - call "
                "bind_frozen_model() first; a transfer arm with nothing to transfer is not an arm"
            )
        blob = Path(FrozenTransferHead._PICKLE).read_bytes()
        got = hashlib.sha256(blob).hexdigest()
        if got != FrozenTransferHead._SHA:
            raise HarnessRefusal(
                "frozen model bytes changed since the freeze was recorded - refusing to transfer"
            )
        self.model_ = pickle.loads(blob)
        self.n_fit_calls_ = 0

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.n_fit_calls_ += 1                 # counted so a silent refit is impossible to hide
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(np.asarray(x, dtype=np.float64))


def bind_frozen_model(path: Path, sha256: str) -> None:
    """Bind the frozen fit on EVERY module copy - see ``_module_copies`` for why that is not
    paranoia. ``apply`` binds from ``__main__`` and the harness constructs from the imported copy,
    so binding one of the two is a guaranteed refusal."""
    bound = 0
    for mod in _module_copies():
        cls = getattr(mod, "FrozenTransferHead", None)
        if cls is not None:
            cls._PICKLE = Path(path)
            cls._SHA = str(sha256)
            bound += 1
    if not bound:
        raise HarnessRefusal("bind_frozen_model found no FrozenTransferHead to bind")


# ==============================================================================================
# 3. THE ARMS - declared here in code, not in a spec, so the tournament is auditable at one place
# ==============================================================================================

NINE = list(FEATURES)
BIDIR_FEATURES = NINE + BIDIR


def make_mined_tree():
    return MinedRanker("tree", NINE, tag="A2m.tree.nine.mined")


def make_bidir_tree():
    return MinedRanker("tree", BIDIR_FEATURES, rounds=1, tag="A4.bidir.tree")


def make_bidir_tree_mined():
    return MinedRanker("tree", BIDIR_FEATURES, tag="A4m.bidir.tree.mined")


def make_secondary_seed():
    return MinedRanker("tree", BIDIR_FEATURES, rounds=1, seed=SECONDARY_SEED,
                       row_bootstrap=0.7, tag="A5.secondary.seed")


def make_frozen_transfer():
    return FrozenTransferHead()


HEAD = "assoc_tournament:{}"

ARMS: list[dict] = [
    # 1. calibrated linear / listwise control - the FACT-0386 protocol, unchanged
    {"tag": "A1.linear.nine", "class": "linear", "features": NINE,
     "abstain": {"kind": "none"}, "population": "alldec"},
    # 2. tree ranker - the arm that came closest on fold 0 and still failed its interval
    {"tag": "A2.tree.nine", "class": "tree", "features": NINE,
     "abstain": {"kind": "none"}, "population": "alldec"},
    # 2m. the same tree with the iterative hard-negative curriculum
    {"tag": "A2m.tree.nine.mined", "class": "declared", "head": HEAD.format("make_mined_tree"),
     "features": META + NINE, "abstain": {"kind": "none"}, "population": "alldec"},
    # 4. bidirectional head - forward and reverse competition plus their consensus
    {"tag": "A4.bidir.tree", "class": "declared", "head": HEAD.format("make_bidir_tree"),
     "features": META + BIDIR_FEATURES, "abstain": {"kind": "none"}, "population": "alldec"},
    # 4m. bidirectional head under the same curriculum
    {"tag": "A4m.bidir.tree.mined", "class": "declared",
     "head": HEAD.format("make_bidir_tree_mined"),
     "features": META + BIDIR_FEATURES, "abstain": {"kind": "none"}, "population": "alldec"},
    # 4a. the abstaining variant - this is what makes side (b) of the two-sided bar LIVE
    {"tag": "A4a.bidir.tree.abstain", "class": "declared", "head": HEAD.format("make_bidir_tree"),
     "features": META + BIDIR_FEATURES,
     "abstain": {"kind": "train_quantile", "q": ABSTAIN_Q}, "population": "alldec"},
    # 5. alternate-seed secondary head - exists so CONSENSUS becomes measurable (FACT-0378)
    {"tag": "A5.secondary.seed", "class": "declared", "head": HEAD.format("make_secondary_seed"),
     "features": META + BIDIR_FEATURES, "abstain": {"kind": "none"}, "population": "alldec"},
]

ARM_BY_TAG = {a["tag"]: a for a in ARMS}

# THE MEASURED NOISE FLOOR, not a model. FACT-0386 established that a HistGradientBoosting learner
# bins each feature into at most 255 quantile bins, so a tree given ONLY the deployed probability -
# which cannot in principle beat an argmax over that same scalar - still posted +0.0014 and net +6
# on fold 0 because candidates collided in a bin. Roughly HALF the best fold-0 arm's +12. Fold 1 is
# a different surface with 6.4x the contested targets, so its binning floor is NOT the fold-0 one
# and cannot be assumed; without it measured on this surface, a reader cannot tell how much of the
# tournament's fold-1 delta is skill. The LINEAR twin is the internal control that proves the
# instrument: a monotone transform of a single feature CANNOT move a within-target argmax, so it
# must return the deployed number with churn EXACTLY 0, and `noisefloor` refuses if it does not.
NOISE_FLOOR_ARMS: list[dict] = [
    {"tag": "N0.prob_only.linear", "class": "linear", "features": ["prob"],
     "abstain": {"kind": "none"}, "population": "alldec"},
    {"tag": "N0.prob_only.tree", "class": "tree", "features": ["prob"],
     "abstain": {"kind": "none"}, "population": "alldec"},
]

# Arm 3 is the only one that cannot run on CPU. Written as data so `prepare` and the report agree.
BLOCKED_ARM = {
    "tag": "A3.context.hoct_style",
    "class": "contextual",
    "contract": {"name": "frozen_trunk_node_features", "dim": "from_cache",
                 "pair_builder": "concat_diff", "extra_features": ["prob", "dist_um"],
                 "has_null_head": False},
    "abstain": {"kind": "none"},
    "population": "alldec",
    "blocked_on": [
        "a Gate-1 (EXP-0041) receipt whose band-B checked count is > 0 - FACT-0382 puts every "
        "contested error below the deployed 0.5 floor, so a band-A-only receipt validates exactly "
        "the band this task does not use and the harness refuses it",
        "a fetched per-crop feature cache carrying the CACHE_KEYS contract, written inside the "
        "predictor's own process by source rewriting and SERIALISED TO DISK then reloaded "
        "(FACT-0387 conditions i-iii)",
        "a licence written by `assoc_train_harness.py gate`, which pins the cache bytes by sha256 "
        "and is re-verified at training time",
        "PKT-0037 foundation certification and a PKT-0036 Gate-1 receipt before any GPU session",
        "a per-cache manifest that BINDS trunk, fold, embryo, normalisation, node ordering, "
        "candidate parameters and commit, verified by scripts/win_bet/audit_feature_cache.py "
        "`audit` - PKT-0038 forbids training on a cache whose manifest does not bind these",
        "a DUAL-TRUNK PAIR for the fold, passing audit_feature_cache.audit_dual_trunk: same node "
        "set, same fold, same crops, DIFFERENT checkpoints, DIFFERENT roles and DIFFERENT feature "
        "digests. FACT-0392's third risk is that trunk identity is a --weights argument recorded "
        "nowhere in the artifact, so without the pair a null result cannot be told apart from a "
        "wrong feature contract and the whole run is uninterpretable",
    ],
    "note": "FACT-0392: HOCT's own head cannot rerank our candidate list - its 15 um uncapped "
            "geometric ball discriminates among competitors our pipeline never offers, and its "
            "abstain mass makes absolute levels incomparable. This arm is therefore a head trained "
            "on OUR contract over the licensed cache's frozen-trunk node features, not a port.",
}


# ==============================================================================================
# 4. RUNNING A FOLD
# ==============================================================================================

def surface_provenance(path: Path, fold: int) -> dict:
    """Which table a number came from, by PATH and by sha256, recorded into every payload.

    A payload that names no table lets a later reader infer the surface, and this campaign has
    already paid for one inherited-artifact assumption (``FACT-0350``). The digest is recomputed
    here at read time, never copied from a receipt.
    """
    p = Path(path)
    src = DISCOVERY_SOURCE if fold == DISCOVERY_FOLD else GATE_SOURCE
    return {
        "table": str(p),
        "table_sha256": _sha(p) if p.is_file() else "ABSENT - synthetic or test surface",
        "built_from": src,
        "not_read_from": "C:/temp/audit_receipts/ - PKT-0036 audit scratch, another packet's "
                         "private working area; the fold-1 table was relocated into the "
                         "acquisition's own directory and verified byte-identical before use",
    }


def load_surface(path: Path, fold: int) -> pl.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise HarnessRefusal(f"fold-{fold} surface missing: {p}")
    if "audit_receipts" in str(p).replace("\\", "/"):
        raise HarnessRefusal(
            f"refusing to read {p}: C:/temp/audit_receipts/ is PKT-0036's audit scratch and may "
            "be cleaned at any time - read the acquisition directory instead"
        )
    return bidirectional_columns(pl.read_parquet(p))


def run_fold(table: pl.DataFrame, fold: int, arms: list[dict], out: Path, label: str,
             preserve_single: bool = True) -> dict:
    clear_curriculum()
    t0 = time.time()
    payload = run_harness(
        table=table,
        models=[ModelSpec.from_dict(a) for a in arms],
        fold=fold, cv_kind="GroupKFold", n_splits=5,
        preserve_single=preserve_single, cache_gate=None, role_index_by_crop=None,
        chain_arms=None, summarise=None, allow_degenerate=False,
    )
    # FAIL CLOSED on the curriculum. A mining arm that records nothing is indistinguishable in the
    # payload from an arm that mined nothing, and the packet requires the schedule reproducible
    # rather than emergent - so an empty log where one is owed is a refusal, not a warning.
    mining_arms = [a["tag"] for a in arms
                   if a.get("class") == "declared"
                   and a.get("head", "") != HEAD.format("make_frozen_transfer")]
    curriculum = collected_curriculum()
    if mining_arms and not curriculum:
        raise HarnessRefusal(
            f"mining arms {mining_arms} ran but the curriculum is EMPTY - the collector did not "
            "reach the estimators (see _mining_logs); refusing to write a payload that claims to "
            "record a curriculum it does not have"
        )
    payload["tournament"] = {
        "label": label,
        "packet": "PKT-0038",
        "lever": "LEVER-0041",
        "surface": surface_provenance(
            DISCOVERY_TABLE if fold == DISCOVERY_FOLD else GATE_TABLE, fold),
        "elapsed_s": round(time.time() - t0, 1),
        "arms_declared": [a["tag"] for a in arms],
        "blocked_arms": [BLOCKED_ARM["tag"]],
        "mining_curriculum": curriculum,
        "mining_curriculum_expected_from": mining_arms,
        "full_chain": {
            "measured": False,
            "why": "FACT-0364 - motion relink replaces the solver's whole edge list at a median "
                   "99.9% coverage, so parent_conversions is a PRE-ILP proxy and the genuine "
                   "FACT-0376 quantity (final-graph edge TP/FP/FN) is unmeasured. FACT-0382 adds "
                   "that the comparison must be (widened + scorer) vs the DEPLOYED control.",
        },
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return payload


def arm_row(model: dict, bar: float | None) -> dict:
    c = model["surface"]["contested"]
    k = model["conversions"]["contested"]
    b = k.get("crop_paired_bootstrap", {})
    s = model["single_candidate"]
    return {
        "tag": model["tag"], "contested_top1": c["top1"],
        "delta_vs_deployed": None if bar is None or c["top1"] is None else c["top1"] - bar,
        "gained": k["gained"], "lost": k["lost"], "net": k["net"], "churn": k["churn"],
        "ci95": b.get("ci95"), "favourable": bool(b.get("favourable")),
        "excludes_zero": bool(b.get("excludes_zero")),
        "abstentions": model["surface"]["abstentions"],
        "single_candidate_regressions": s["n_regressions"],
        "single_candidate_shadow": s["n_shadow_regressions"],
        "promotable": model["verdict"]["promotable"],
    }


def summarise_payload(payload: dict) -> dict:
    bar = payload["deployed_baseline"]["contested"]["top1"]
    return {"deployed_contested_top1": bar,
            "arms": [arm_row(m, bar) for m in payload["models"]]}


# ==============================================================================================
# 5. CONSENSUS - a DIAGNOSTIC, never a promotion channel
# ==============================================================================================

def consensus_diagnostic(table: pl.DataFrame, primary_tag: str, secondary_tag: str,
                         fold: int) -> dict:
    """Agreement between the primary and the alternate-seed secondary, per target.

    ``FACT-0378`` is why this exists and why it cannot be answered with the public configuration's
    secondary: that checkpoint saw all 199 movies, so it is legitimate against the hidden test set
    and un-validatable by any LOEO control we can build. A consensus claim needs FOLD-SPECIFIC
    secondary heads of our own, and this is the measurement they make possible.

    Reported as a diagnostic only. It re-derives the two arms' decisions through the harness's own
    ``fit_out_of_fold``/``decide`` at the same seed and splitter, and ASSERTS that the contested
    top-1 it recovers matches the run this run reports, so it cannot drift into a second protocol.
    """
    dec = table.filter(pl.col("true_parent_is_candidate") == 1)
    counts = dec.group_by(["crop", "target"]).agg(pl.len().alias("n_dec"))
    dec = dec.join(counts, on=["crop", "target"], how="left")
    contested = {(r[0], int(r[1])) for r in
                 counts.filter(pl.col("n_dec") > 1).select(["crop", "target"]).rows()}

    out: dict = {"fold": fold, "primary": primary_tag, "secondary": secondary_tag,
                 "promotable": False,
                 "note": "DIAGNOSTIC ONLY. No consensus rule is deployed, scored or promoted here; "
                         "this measures whether two independently-seeded heads of our own make "
                         "SEPARABLE errors, which is the precondition FACT-0378 denies the public "
                         "secondary."}
    per_target: dict[str, dict] = {}
    for tag in (primary_tag, secondary_tag):
        spec = ModelSpec.from_dict(ARM_BY_TAG[tag])
        x = dec.select(spec.features).to_numpy().astype(np.float64)
        score, null, _folds = fit_out_of_fold(dec, spec, x, make_splitter("GroupKFold", 5), tag)
        scored = dec.with_columns([pl.Series("_score", score), pl.Series("_null", null)])
        pt, _ledger, agg = decide(scored, "_score", None, preserve_single=True)
        per_target[tag] = pt
        out[tag] = {"contested_top1": agg["contested"]["top1"], "contested_n": agg["contested"]["n"]}

    p, s = per_target[primary_tag], per_target[secondary_tag]
    keys = sorted(set(p) & set(s) & contested)
    if not keys:
        raise HarnessRefusal("consensus diagnostic found no shared contested targets")
    pa = np.array([p[k] for k in keys])
    sa = np.array([s[k] for k in keys])
    agree = pa == sa
    out["contested"] = {
        "n": len(keys),
        "agreement_rate": float(agree.mean()),
        "both_correct": int((pa & sa).sum()),
        "both_wrong": int(((1 - pa) & (1 - sa)).sum()),
        "primary_only": int((pa & (1 - sa)).sum()),
        "secondary_only": int(((1 - pa) & sa).sum()),
        "top1_where_they_agree": float(pa[agree].mean()) if agree.any() else None,
        "oracle_of_the_two": float(np.maximum(pa, sa).mean()),
        "reading": "`oracle_of_the_two` is the CEILING a perfect consensus rule could reach and is "
                   "NOT a result - it is chosen with the label. The usable signal is whether "
                   "disagreement concentrates the errors: if top1_where_they_agree is far above "
                   "the pooled contested top-1, agreement is a usable abstention trigger.",
    }
    return out


# ==============================================================================================
# 6. FREEZE / APPLY
# ==============================================================================================

def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_winner(payload: dict) -> dict:
    """The DECLARED rule, applied to the discovery payload only. Never sees fold 0."""
    bar = payload["deployed_baseline"]["contested"]["top1"]
    rows = [arm_row(m, bar) for m in payload["models"]]
    favourable = [r for r in rows if r["favourable"] and r["contested_top1"] is not None]
    pool = favourable or [r for r in rows if r["contested_top1"] is not None]
    if not pool:
        raise HarnessRefusal("no arm produced a contested top-1 - nothing to freeze")
    best = max(pool, key=lambda r: r["contested_top1"])
    return {
        "winner": best["tag"],
        "favourable_on_discovery": bool(favourable),
        "selection_rule": "highest fold-1 contested top-1 among arms with a strictly positive "
                          "crop-paired ci95 lower bound; if none is favourable the tournament has "
                          "NO WINNER and the best arm is transferred for the record only",
        "n_arms_considered": len(rows),
        "n_arms_favourable": len(favourable),
        "discovery_row": best,
        "all_rows": rows,
    }


def freeze(payload_path: Path, table: pl.DataFrame, out_dir: Path) -> dict:
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    if int(payload["fold"]) != DISCOVERY_FOLD:
        raise HarnessRefusal(
            f"refusing to freeze on fold {payload['fold']} - fold {DISCOVERY_FOLD} is the "
            "discovery surface and the two are never swapped (FACT-0388)"
        )
    sel = select_winner(payload)
    arm = dict(ARM_BY_TAG[sel["winner"]])

    dec = table.filter(pl.col("true_parent_is_candidate") == 1)
    counts = dec.group_by(["crop", "target"]).agg(pl.len().alias("n_dec"))
    dec = dec.join(counts, on=["crop", "target"], how="left")
    spec = ModelSpec.from_dict(arm)

    clear_curriculum()
    from assoc_train_harness import make_estimator
    model = make_estimator(spec.model_class, spec.head)
    x = dec.select(spec.features).to_numpy().astype(np.float64)
    y = dec["is_true_parent"].to_numpy().astype(np.int64)
    model.fit(x, y)

    # The abstention threshold is frozen HERE, on fold-1 training targets, and is never refitted.
    train_scores = model.predict_proba(x)[:, 1]
    best: dict[tuple[str, int], float] = {}
    for c, t, s in zip(dec["crop"].to_numpy().tolist(), dec["target"].to_numpy().tolist(),
                       train_scores.tolist()):
        k = (c, int(t))
        if s > best.get(k, -np.inf):
            best[k] = s
    tau = AbstainPolicy(**arm["abstain"]).fit(np.asarray(list(best.values()), dtype=np.float64))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl = out_dir / "frozen_fold1_model.pkl"
    pkl.write_bytes(pickle.dumps(model))

    record = {
        "schema_version": 1,
        "heartbeat": "ASSOC_TOURNAMENT_FROZEN",
        "packet": "PKT-0038",
        "frozen_on_fold": DISCOVERY_FOLD,
        "applies_to_fold": GATE_FOLD,
        "discovery_payload_sha256": _sha(payload_path),
        "model_pickle": str(pkl),
        "model_pickle_sha256": _sha(pkl),
        "arm": arm,
        "abstain_frozen": {"kind": "none" if tau == -np.inf else "fixed",
                           "tau": None if tau == -np.inf else float(tau),
                           "fitted_on": "fold-1 training targets only; NEVER refitted on fold 0"},
        "seed": SEED,
        "secondary_seed": SECONDARY_SEED,
        "mining_curriculum_full_fit": collected_curriculum(),
        "selection": sel,
        "discipline": (
            "Architecture, hyperparameters, feature list, mining curriculum and abstention "
            "threshold are all fixed by this record. `apply` re-verifies both digests and refuses "
            "on any change. There is no --force and no per-fold recalibration."
        ),
    }
    rec_path = out_dir / "frozen_selection.json"
    rec_path.write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
    return record


def apply_to_gate(record_path: Path, table: pl.DataFrame, out_dir: Path) -> dict:
    rec = json.loads(Path(record_path).read_text(encoding="utf-8"))
    if rec.get("heartbeat") != "ASSOC_TOURNAMENT_FROZEN":
        raise HarnessRefusal(f"{record_path} is not a frozen-selection record")
    if int(rec["frozen_on_fold"]) != DISCOVERY_FOLD or int(rec["applies_to_fold"]) != GATE_FOLD:
        raise HarnessRefusal("frozen record does not describe a fold-1 -> fold-0 transfer")
    pkl = Path(rec["model_pickle"])
    if _sha(pkl) != rec["model_pickle_sha256"]:
        raise HarnessRefusal("frozen model bytes changed since the freeze - refusing to transfer")
    bind_frozen_model(pkl, rec["model_pickle_sha256"])

    arm = rec["arm"]
    gate_a = dict(arm)
    gate_a["tag"] = f"GATE_A.architecture_transfer.{arm['tag']}"
    gate_b = {
        "tag": f"GATE_B.weight_transfer.{arm['tag']}",
        "class": "declared",
        "head": HEAD.format("make_frozen_transfer"),
        "features": arm["features"],
        "abstain": ({"kind": "none"} if rec["abstain_frozen"]["kind"] == "none"
                    else {"kind": "fixed", "tau": rec["abstain_frozen"]["tau"]}),
        "population": arm.get("population", "alldec"),
    }
    out = run_fold(table, GATE_FOLD, [gate_a, gate_b],
                   Path(out_dir) / "gate_fold0.json", "gate-fold0")
    out["tournament"]["frozen_selection"] = {
        "record": str(record_path),
        "winner": arm["tag"],
        "favourable_on_discovery": rec["selection"]["favourable_on_discovery"],
        "the_transfer_measurement_is_gate_b_only": True,
        "gate_a": {
            "tag": gate_a["tag"],
            "is_transfer_measurement": False,
            "what_it_is": "architecture transfer - the identical recipe REFITTED out of fold on "
                          "fold 0's own crop-grouped splits, which is exactly the protocol that "
                          "produced FACT-0386 and is therefore directly comparable to the killed "
                          "LEVER-0039 arms",
            "what_it_is_not": "it is NOT the packet's transfer measurement and must never be "
                              "quoted as one: it refits weights (and, for an abstaining arm, its "
                              "threshold) on fold 0. It exists to separate 'the architecture does "
                              "not transfer' from 'fold 0 is too thin to show it' - the two "
                              "readings falsifier (g) has to choose between.",
        },
        "gate_b": {
            "tag": gate_b["tag"],
            "is_transfer_measurement": True,
            "what_it_is": "weight transfer - the fold-1 fit applied with NO fold-0 fitting of any "
                          "kind and the abstention threshold frozen on fold 1. No refit, no "
                          "threshold nudge, no per-fold calibration. This is also a cross-EMBRYO "
                          "measurement (6bba -> 44b6) and it is the number falsifiers (e) and (g) "
                          "are settled on.",
            "abstain_frozen": rec["abstain_frozen"],
        },
    }
    Path(Path(out_dir) / "gate_fold0.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    return out


# ==============================================================================================
# 7. CLI
# ==============================================================================================

def _print_rows(title: str, summary: dict) -> None:
    bar = summary["deployed_contested_top1"]
    print(f"\n{title}   deployed contested top-1 = {bar:.6f}")
    print(f"{'arm':42s} {'contested':>10s} {'delta':>9s} {'gain':>5s} {'lost':>5s} {'net':>5s} "
          f"{'churn':>6s} {'ci95_lo':>9s} {'ci95_hi':>9s} {'fav':>5s} {'sc.reg':>7s} "
          f"{'sc.shadow':>10s}")
    for r in summary["arms"]:
        lo, hi = (r["ci95"] or [float("nan"), float("nan")])
        print(f"{r['tag']:42s} {r['contested_top1']:10.6f} {r['delta_vs_deployed']:+9.6f} "
              f"{r['gained']:5d} {r['lost']:5d} {r['net']:+5d} {r['churn']:6d} "
              f"{lo:+9.5f} {hi:+9.5f} {str(r['favourable']):>5s} "
              f"{r['single_candidate_regressions']:7d} {r['single_candidate_shadow']:10d}")


def cmd_discover(args) -> int:
    table = load_surface(DISCOVERY_TABLE, DISCOVERY_FOLD)
    out = Path(args.out_dir) / "discovery_fold1.json"
    payload = run_fold(table, DISCOVERY_FOLD, ARMS, out, "discovery-fold1")
    summary = summarise_payload(payload)
    _print_rows("DISCOVERY (fold 1, the widened P34 surface - FACT-0388)", summary)
    cons = consensus_diagnostic(table, "A4.bidir.tree", "A5.secondary.seed", DISCOVERY_FOLD)
    (Path(args.out_dir) / "consensus_fold1.json").write_text(
        json.dumps(cons, indent=2, default=float), encoding="utf-8")
    c = cons["contested"]
    print(f"\nCONSENSUS DIAGNOSTIC (not promotable)  agree={c['agreement_rate']:.4f}  "
          f"both_correct={c['both_correct']}  both_wrong={c['both_wrong']}  "
          f"primary_only={c['primary_only']}  secondary_only={c['secondary_only']}  "
          f"top1|agree={c['top1_where_they_agree']:.4f}  oracle={c['oracle_of_the_two']:.4f}")
    print(f"\nASSOC_TOURNAMENT_DISCOVERY_COMPLETE -> {out}")
    return 0


def cmd_sideb(args) -> int:
    """Make side (b) of the two-sided bar LIVE, which preservation-on cannot.

    With ``preserve_single_candidate`` on, the null is structurally unavailable to single-candidate
    targets, so retention is 1.0 by construction and only the SHADOW count prices the constraint.
    ``PKT-0038`` falsifier (b) - "beats contested top-1 while REGRESSING single-candidate targets" -
    is therefore untestable in that mode. This run turns preservation OFF for the abstaining arms
    only, so every regression is listed by ``(crop, target, source)`` and the ledger refuses to emit
    a netted figure. It is a MEASUREMENT of the abstention risk, not a promotion arm.
    """
    table = load_surface(DISCOVERY_TABLE, DISCOVERY_FOLD)
    arms = [a for a in ARMS if a["abstain"]["kind"] != "none"]
    if not arms:
        raise HarnessRefusal("no abstaining arm declared - side (b) cannot be made live")
    out = Path(args.out_dir) / "sideb_fold1.json"
    payload = run_fold(table, DISCOVERY_FOLD, arms, out, "sideb-fold1", preserve_single=False)
    payload["tournament"]["side_b"] = {
        "preserve_single_candidate": False,
        "why": "PKT-0038 falsifier (b) is only testable when the null is available on "
               "single-candidate targets; with preservation on it is inert by construction "
               "(FACT-0386).",
    }
    Path(out).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    _print_rows("SIDE (b) LIVE (fold 1, preservation OFF)", summarise_payload(payload))
    print(f"\nASSOC_TOURNAMENT_SIDEB_COMPLETE -> {out}")
    return 0


def cmd_noisefloor(args) -> int:
    """Measure the binning floor on the DISCOVERY surface, through the same protocol as the arms.

    This is the control FACT-0386 was commissioned for and it is run on fold 1 because the floor is
    a property of the SURFACE - candidate collisions inside a quantile bin - not a constant carried
    over from fold 0. Two arms, both preregistered here in code:

      N0.prob_only.linear  the INSTRUMENT CHECK. A monotone transform of the single feature the
                           deployed argmax already uses cannot change any within-target ranking, so
                           churn must be exactly 0 and the contested top-1 exactly the deployed one.
                           Anything else means the protocol - not the model - is broken, and this
                           command REFUSES rather than reporting a floor it cannot trust.
      N0.prob_only.tree    the FLOOR itself. Its delta is binning, by construction, because it has
                           no information the argmax lacks.

    A tournament arm's delta is only skill to the extent it exceeds this.
    """
    table = load_surface(DISCOVERY_TABLE, DISCOVERY_FOLD)
    out = Path(args.out_dir) / "noisefloor_fold1.json"
    payload = run_fold(table, DISCOVERY_FOLD, NOISE_FLOOR_ARMS, out, "noisefloor-fold1")
    bar = payload["deployed_baseline"]["contested"]["top1"]
    by_tag = {m["tag"]: m for m in payload["models"]}

    lin = by_tag["N0.prob_only.linear"]
    lin_churn = int(lin["conversions"]["contested"]["churn"])
    lin_top1 = float(lin["surface"]["contested"]["top1"])
    if lin_churn != 0 or abs(lin_top1 - bar) > 1e-12:
        raise HarnessRefusal(
            "INSTRUMENT CHECK FAILED: a monotone transform of the deployed probability alone moved "
            f"the within-target argmax (churn {lin_churn}, top-1 {lin_top1!r} against the deployed "
            f"{bar!r}). That is arithmetically impossible for a correct protocol, so the floor "
            "measured beside it cannot be trusted and no arm's delta may be read against it."
        )

    tree = by_tag["N0.prob_only.tree"]
    floor_delta = float(tree["surface"]["contested"]["top1"]) - bar
    payload["tournament"]["noise_floor"] = {
        "instrument_check": {
            "arm": "N0.prob_only.linear", "churn": lin_churn, "contested_top1": lin_top1,
            "passed": True,
            "why": "a monotone transform of one feature cannot move a within-target argmax, so "
                   "churn 0 and an unchanged top-1 are the only correct outcome (FACT-0386)",
        },
        "floor": {
            "arm": "N0.prob_only.tree",
            "contested_top1": float(tree["surface"]["contested"]["top1"]),
            "delta": floor_delta,
            "gained": int(tree["conversions"]["contested"]["gained"]),
            "lost": int(tree["conversions"]["contested"]["lost"]),
            "net": int(tree["conversions"]["contested"]["net"]),
            "churn": int(tree["conversions"]["contested"]["churn"]),
            "ci95": tree["conversions"]["contested"].get("crop_paired_bootstrap", {}).get("ci95"),
        },
        "how_to_read_it": "This arm has NO information the deployed argmax lacks, so its entire "
                          "delta is quantile binning. Subtract it before calling any tournament "
                          "arm's delta skill; on fold 0 the floor was roughly HALF the best arm's "
                          "net (FACT-0386), which is why it is measured rather than assumed.",
    }
    Path(out).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    _print_rows("NOISE FLOOR (fold 1, prob-only controls)", summarise_payload(payload))
    print(f"\nINSTRUMENT CHECK PASSED - prob-only linear churn 0 at the deployed top-1")
    print(f"BINNING FLOOR on this surface: delta {floor_delta:+.6f}, "
          f"net {tree['conversions']['contested']['net']:+d}, "
          f"churn {tree['conversions']['contested']['churn']}")
    print(f"\nASSOC_TOURNAMENT_NOISEFLOOR_COMPLETE -> {out}")
    return 0


def cmd_freeze(args) -> int:
    table = load_surface(DISCOVERY_TABLE, DISCOVERY_FOLD)
    rec = freeze(Path(args.out_dir) / "discovery_fold1.json", table, Path(args.out_dir))
    print(f"\nFROZEN winner={rec['selection']['winner']} "
          f"favourable_on_discovery={rec['selection']['favourable_on_discovery']} "
          f"abstain={rec['abstain_frozen']['kind']} tau={rec['abstain_frozen']['tau']}")
    print(f"ASSOC_TOURNAMENT_FROZEN -> {Path(args.out_dir) / 'frozen_selection.json'}")
    return 0


def cmd_apply(args) -> int:
    table = load_surface(GATE_TABLE, GATE_FOLD)
    payload = apply_to_gate(Path(args.out_dir) / "frozen_selection.json", table, Path(args.out_dir))
    _print_rows("GATE (fold 0, frozen and applied unchanged)", summarise_payload(payload))
    print(f"\nASSOC_TOURNAMENT_GATE_COMPLETE -> {Path(args.out_dir) / 'gate_fold0.json'}")
    return 0


# Every trunk that could plausibly have produced a fold's node-feature cache. One spec is written
# per (fold, role) so that WHICHEVER cache lands first, an arm is already wired for it and training
# starts without editing a spec - host constraint (3), pipeline rather than batch.
#
#   pack_split0 / oof_split1   OUR OWN trunk, and the fold-legitimate one under the LOEO protocol:
#                              pack split_0 is the CLEAN fold-0 primary and OOF split_1 is fold 1's
#                              (FACT-0345, FACT-0378). This is the "contract we own" that host
#                              constraint (1) names, and it waits on no HOCT weight compatibility.
#   official / stabledet       the two HOCT-bundle trunks. FACT-0392's third risk is that which one
#                              produced a 32-dim cache is a `--weights` argument recorded NOWHERE in
#                              the checkpoint or the cache, so official-trunk features fed to a
#                              StableDet-trained head look exactly like a wrong contract. Host
#                              constraint (2): BOTH are cached in each fold acquisition and audited
#                              as a pair, so trunk provenance can never be the explanation offered
#                              for a null.
TRUNK_ROLES_BY_FOLD: dict[int, tuple[str, ...]] = {
    0: ("pack_split0", "official", "stabledet"),
    1: ("oof_split1", "official", "stabledet"),
}
DUAL_TRUNK_PAIR = ("official", "stabledet")


def context_spec(fold: int, role: str, table: str, preilp: str, ecb: str) -> dict:
    """The cache-gated spec for arm 3 on one (fold, trunk) pair.

    Written as data so ``prepare`` and the blocked-arm report cannot disagree about what the arm
    needs, and one spec per role so the arm is trainable the moment ANY licensed cache lands.
    """
    stem = f"f{fold}_{role}"
    return {
        "name": f"assoc_tournament_context_{stem}",
        "fold": fold,
        "no_claim": True,
        "note": "PREPARED, NOT LAUNCHED - PKT-0038 arm 3. " + BLOCKED_ARM["note"]
                + " BLOCKED ON: " + "; ".join(BLOCKED_ARM["blocked_on"]),
        "table": table,
        "trunk": {
            "role": role,
            "fold_legitimate": role in ("pack_split0", "oof_split1"),
            "manifest": f"C:/temp/assoc_tournament/cache_{stem}/cache_manifest.json",
            "audit": "scripts/win_bet/audit_feature_cache.py audit "
                     f"--cache-dir C:/temp/assoc_tournament/cache_{stem} --expect-fold {fold} "
                     f"--expect-role {role}",
            "dual_trunk_pair": list(DUAL_TRUNK_PAIR),
            "dual_trunk_audit": "audit_feature_cache.audit_dual_trunk("
                                f"C:/temp/assoc_tournament/cache_f{fold}_{DUAL_TRUNK_PAIR[0]}, "
                                f"C:/temp/assoc_tournament/cache_f{fold}_{DUAL_TRUNK_PAIR[1]})",
            "why": "FACT-0392 third risk - trunk identity is a --weights argument recorded nowhere "
                   "in the artifact, so a null measured on a single unaudited trunk cannot be told "
                   "apart from a wrong feature contract.",
        },
        "cache": {"dir": f"C:/temp/assoc_tournament/cache_{stem}",
                  "receipt": f"C:/temp/assoc_tournament/gate1_receipt_{stem}.json",
                  "licence": f"C:/temp/assoc_tournament/licence_{stem}.json",
                  "preilp": preilp, "ecb_dir": ecb, "reproducer": "receipt", "crops": None},
        "cv": {"kind": "GroupKFold", "n_splits": 5},
        "preserve_single_candidate": True,
        "allow_degenerate": False,
        "models": [
            {**{k: v for k, v in BLOCKED_ARM.items()
                if k in ("class", "contract", "abstain", "population")},
             "tag": f"{BLOCKED_ARM['tag']}.{role}"},
            # The abstaining twin. The packet requires an explicit null class THROUGHOUT, and
            # without it side (b) of the two-sided bar would be inert on the one arm that is
            # supposed to carry the richest representation.
            {**{k: v for k, v in BLOCKED_ARM.items()
                if k in ("class", "contract", "population")},
             "tag": f"{BLOCKED_ARM['tag']}.{role}.abstain",
             "abstain": {"kind": "train_quantile", "q": ABSTAIN_Q}},
        ],
        "out_dir": f"C:/temp/assoc_tournament/context_{stem}",
    }


def cmd_prepare(args) -> int:
    """Write the cache-gated specs for the one arm that cannot run on CPU. NOT launched."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_dir = Path(ROOT / "scripts" / "win_bet" / "assoc_specs")
    specs: dict[str, str] = {}
    for fold, table, pre, ecb in ((1, str(DISCOVERY_TABLE), "C:/temp/p34_f1/preilp_split1.parquet",
                                   "C:/temp/p34_f1/ecb"),
                                  (0, str(GATE_TABLE), "C:/temp/p30_f0/preilp_split0.parquet",
                                   "C:/temp/p30_f0/ecb")):
        # A single-trunk spec left on disk is exactly the trap harness_f1.json was: someone picks it
        # up and trains on a cache the pair audit never saw. Removed rather than left stale.
        stale = spec_dir / f"tournament_context_f{fold}.json"
        if stale.exists():
            stale.unlink()
            print(f"  removed the superseded single-trunk spec -> {stale}")
        for role in TRUNK_ROLES_BY_FOLD[fold]:
            spec = context_spec(fold, role, table, pre, ecb)
            p = spec_dir / f"tournament_context_f{fold}_{role}.json"
            p.write_text(json.dumps(spec, indent=2), encoding="utf-8")
            specs[f"f{fold}.{role}"] = str(p)
            print(f"  prepared (NOT launched) -> {p}")
    (out_dir / "blocked_arms.json").write_text(json.dumps({
        "blocked": [BLOCKED_ARM],
        "trunk_roles_by_fold": {str(k): list(v) for k, v in TRUNK_ROLES_BY_FOLD.items()},
        "dual_trunk_pair": list(DUAL_TRUNK_PAIR),
        "dual_trunk_rule": "Host constraint (2). Each fold acquisition caches BOTH HOCT-bundle "
                           "trunks and they are audited as a pair by "
                           "audit_feature_cache.audit_dual_trunk before any head is fitted, so "
                           "trunk provenance can never be offered as the explanation for a null "
                           "(FACT-0392 third risk).",
        "specs": specs,
    }, indent=2), encoding="utf-8")
    print("ASSOC_TOURNAMENT_PREPARE_COMPLETE")
    return 0


def _cap_threads() -> None:
    """Cap BLAS/OpenMP threads before sklearn is touched, unless the caller already chose.

    MEASURED on this machine 2026-08-30, and it is a 50x effect, not a tidy-up: with four agents
    sharing eight cores, every sklearn fit spawning eight OpenMP threads oversubscribes the box so
    badly that a 778-row HistGradientBoosting fit took 42.8 s. The same fit at one thread takes
    0.84 s, and a 98,000-row fit takes 5.4 s. A tournament of seven arms over five crop-grouped
    folds is unrunnable at the default and trivial at the cap, so the cap is applied here rather
    than left to whoever types the command.
    """
    import os

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "2")


def main(argv=None) -> int:
    _cap_threads()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("discover", cmd_discover), ("noisefloor", cmd_noisefloor),
                     ("sideb", cmd_sideb), ("freeze", cmd_freeze),
                     ("apply", cmd_apply), ("prepare", cmd_prepare)):
        s = sub.add_parser(name)
        s.add_argument("--out-dir", type=Path, default=Path("C:/temp/assoc_tournament"))
        s.set_defaults(func=fn)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
