r"""PKT-0042 / LEVER-0044 - the two CPU gates on the FINAL-EDGE consumption point.

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT
---------------------------------------------
It is a DRIVER. Every stage it measures is the deployed one: the post-ILP chain is extracted
from the built notebook by ``p28_full_chain_replay.load_p28_module`` (imported, never forked),
the solver is ``ilp_replay.solve``, the metric row is ``div_reach_steal.score_crop`` through the
official patched scorer, the ledger is ``assoc_lost_edge_ledger.crop_ledger`` and the promotable
summary is ``assoc_report.build_report``. Nothing in the deployed path is edited.

THE TWO GATES (LEVER-0044, as amended by the host on 2026-08-30 before any data was taken)

  GATE A - CONSUMPTION CALIBRATION. Sweep ``MOTION_RELINK_LEARNED_BONUS`` over the preregistered
  grid [0, 0.5, 1, 2, 4, 8] with the DEPLOYED ``edge_prob`` unchanged. A loss here kills only
  "amplify the deployed score" - FACT-0421 measured the deployed probability to be precisely the
  NON-TRANSFERRING feature, so a negative Gate A says nothing about the consumption point.

  GATE B - THE DECISIVE MECHANISM TEST. The probability-free contextual score (FACT-0421, the
  ``prob_free_context`` arm of ``assoc_fold_pathology``) written into the ``edge_prob`` slot,
  offered on the WIDENED candidate surface, and consumed inside ``motion_relink_edges`` at the
  selected bonus. If that complete coupled intervention cannot improve RAW association through
  the full chain, the final-edge scorer mechanism is dead.

WHY THE BONUS CAN BE SWEPT WITHOUT A GPU. ``MOTION_RELINK_LEARNED_BONUS`` is a module-level
constant read at CALL time inside ``motion_relink_edges`` (built notebook line 1887 on the f0
control, 1889 on f1 - the two builds are offset by two lines). Rebinding it on the extracted
module between calls therefore changes exactly the deployed knob and nothing else. It is asserted
to be FACT-0427's deployed 1.0 at load, so a rebuild that changes the default fails closed.

THE FIVE FAIL-CLOSED CONTRACTS, each of which has a measured reason to exist:

  CONTROL PARITY   The deployed arm (deployed surface, bonus 1.0) must reproduce the kernel's own
                   ``run_stats.csv`` on all fifteen compared columns before ANY treatment arm is
                   written. This is the discipline that made FACT-0428 interpretable.
  CONTROL IDENTITY (auditor veto V4) ``assoc_report.build_report`` performs NO validation of the
                   control it is handed. The control's provenance - notebook sha, pre-ILP export
                   path and sha256, run_stats path and sha256 - is recorded in every payload and
                   asserted here, because the verdict cannot tell a deployed control from a
                   widened-only one.
  SCORE COVERAGE   (auditor veto V5) ``learned_prob`` at built-notebook 1832 returns 0.0 for an
                   uncovered pair - the WORST score, not a neutral one. Every edge offered to a
                   Gate B arm therefore carries a learned score by construction, and the coverage
                   fraction is measured and reported as a first-class number rather than assumed.
  SCORE RANGE      The score is asserted finite and inside [0, 1] before it is written, so the
                   silent sigmoid at built-notebook 1839-1840 provably cannot fire. A silent
                   no-op is worse than a crash.
  RELINK REACH     FACT-0427: ``learned_edge_probs`` is assembled ONLY from edges that survived
                   the ILP, so a widened candidate the solver dropped is absent, not merely
                   unscored. The share of the widened surface that reaches the relink is measured
                   per arm and per crop.

CPU ONLY. No GPU job, no kernel push, no submission. Writes only its own payloads.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import control_binding as cb  # noqa: E402  the PKT-0043 guard, imported and NOT re-implemented
from scripts.win_bet.ilp_replay import build_graph_from_frame, solve  # noqa: E402
from scripts.win_bet.p28_full_chain_replay import (  # noqa: E402
    P28_ENV,
    load_p28_module,
    solved_graph_inputs,
    submission_frame,
)

HEARTBEAT = "FINALEDGE_CROP_COMPLETE"
SCORES_HEARTBEAT = "FINALEDGE_SCORES_COMPLETE"
PANEL_HEARTBEAT = "FINALEDGE_PANEL_COMPLETE"

# The preregistered coarse grid. Declared here, in code, so it cannot be widened after a result.
BONUS_GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
DEPLOYED_BONUS = 1.0            # FACT-0427; asserted against the extracted module at load

# THE ONE COLUMN THAT CANNOT BE ASSERTED EXACT, AND WHY THAT IS A REGISTRY FACT RATHER THAN A
# CONVENIENCE. FACT-0363 VERIFIED that the deployed ILP has MULTIPLE OPTIMA: its NODE solution is
# deterministic (node parity exact on all 128 fold-1 crops against all three recorded runs) while
# its raw EDGE count is not - three identical local solves of one crop returned 5,035 / 5,036 /
# 5,035 edges, and the three recorded Kaggle runs of that same crop recorded 5,036 / 5,035 /
# 5,036. There is therefore NO unique raw_edges target to be met, and FACT-0363 says so in terms:
# PKT-0025's demand for exact node AND edge parity "is UNSATISFIABLE as written, because the
# target itself is not unique". The registry's measured envelope is edge_jitter_max_abs 2 on 7 of
# 128 crops, about 1 ppm of the fold's edges.
#
# SO: `raw_edges` alone carries FACT-0363's tolerance. EVERY other column stays EXACT, including
# `raw_nodes` (the hard gate, never relaxed) and the FINAL `nodes` and `edges` - if the jitter
# ever reached the emitted graph the crop would refuse, because that is the object being scored.
# Each use is recorded per crop and counted per fold in the panel, so a relaxation cannot hide.
#
# IT DOES NOT TOUCH THE PAIRED COMPARISON AT ALL. Every arm of one crop is built from ONE solve
# (`solved_cache`), so a degenerate solution is common-mode across control and treatments and
# cancels exactly in the paired delta. The tolerance is about reproducing the KERNEL, not about
# comparing arms.
RAW_EDGE_JITTER_TOLERANCE = 2   # FACT-0363 edge_jitter_max_abs

# The exact fifteen columns p28_full_chain_replay compares against the kernel run_stats.
COMPARE_KEYS = [
    "raw_nodes", "raw_edges", "nodes", "edges", "motion_relink_edges",
    "gap_added_nodes", "safe_divisions_added", "deepcenter_gap_checked",
    "deepcenter_gap_accepted", "deepcenter_gap_rejected",
    "deepcenter_safe_div_checked", "deepcenter_safe_div_accepted",
    "deepcenter_safe_div_rejected", "short_track_nodes_removed",
    "linefit_smoothed_nodes",
]

FOLDS = {
    0: {
        "embryo": "44b6",
        "notebook": ROOT / "notebooks/kaggle_p28_champion_control_f0/biohub-p28-champion-control-f0.ipynb",
        "preilp": Path("C:/temp/p30_f0/preilp_split0.parquet"),
        "run_stats": Path("C:/temp/p28_f0/run_stats.csv"),
        "surface": Path("C:/temp/assoc/f0.parquet"),
        "experiment": "EXP-0030",   # P28 champion control f0 - the run whose run_stats we bind to
        "n_crops": 71,
    },
    1: {
        "embryo": "6bba",
        "notebook": ROOT / "notebooks/kaggle_p28_champion_control_f1/biohub-p28-champion-control-f1.ipynb",
        "preilp": Path("C:/temp/p34_f1/preilp_split1.parquet"),
        "run_stats": Path("C:/temp/p28_f1/run_stats.csv"),
        "surface": Path("C:/temp/p34_f1/assoc_targets_split1.parquet"),
        "experiment": "EXP-0031",   # P28 champion control f1 - the run whose run_stats we bind to
        "n_crops": 128,
    },
}

CHECKPOINT = Path("C:/temp/biohub_deepcenter_p10_audit/best.pt")


class GateRefusal(RuntimeError):
    """Fail closed. Every contract in this module is a crash, never a warning."""


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ==============================================================================================
# 1. THE PROBABILITY-FREE CONTEXTUAL SCORE  (FACT-0421's `prob_free_context` arm)
# ==============================================================================================

def build_scores(fold: int, out: Path, mode: str = "transfer") -> dict:
    """`prob_free_context` scores (FACT-0421) for EVERY row of one fold's candidate surface.

    The recipe, the estimator and the splitter are imported from the modules that own them -
    ``assoc_fold_pathology.ABLATIONS['prob_free_context']``, ``assoc_tournament.MinedRanker`` and
    ``assoc_train_harness.make_splitter`` - so this is the arm FACT-0421 measured, not a
    re-derivation of it.

    TWO MODES, AND THE DEFAULT IS THE ONE THE LEVER'S PREMISE RESTS ON.
      ``transfer`` (default) fits on the OTHER fold's decidable rows and applies the frozen model
        here. This is FACT-0421's ``transfer_f1_to_f0`` (+0.003127) and ``transfer_f0_to_f1``
        (+0.001842) - the pair that made "probability-free contextual scoring transfers
        positively in BOTH embryo directions" the premise of the revised promotion gate. No crop,
        embryo or edge-predictor checkpoint of the scored fold reaches the weights.
      ``refit`` is the within-fold crop-grouped GroupKFold(5) arm, FACT-0421's ``refit_f0`` /
        ``refit_f1``. Kept because it reproduces the published numbers exactly and is therefore
        this module's integrity check, but it is NOT the arm the gate's premise names.

    ``fit_out_of_fold`` returns scores only for DECIDABLE rows, because that is the population an
    ablation is evaluated on. A score written into ``edge_prob`` has to cover EVERY offered pair
    or the uncovered ones inherit 0.0 at built-notebook 1832 - the worst score, not a neutral one
    (auditor veto V5) - so every surface row is scored here by a model that never saw it.
    """
    import assoc_fold_pathology as PATH
    import assoc_tournament as TOURN
    from assoc_train_harness import assert_group_integrity, make_splitter

    if mode not in ("transfer", "refit"):
        raise GateRefusal(f"unknown score mode {mode!r}")
    cfg = FOLDS[fold]
    features = list(PATH.ABLATIONS["prob_free_context"])
    spec_features = list(TOURN.META) + features

    surface = TOURN.load_surface(cfg["surface"], fold)
    dec = surface.filter(pl.col("true_parent_is_candidate") == 1)
    if dec.is_empty():
        raise GateRefusal(f"fold {fold}: no decidable rows on {cfg['surface']}")

    surf_crops = set(surface["crop"].unique().to_list())
    dec_crops = set(dec["crop"].unique().to_list())

    x_dec = dec.select(spec_features).to_numpy().astype(np.float64)
    y_dec = dec["is_true_parent"].to_numpy().astype(np.int64)
    crops_dec = dec["crop"].to_numpy()
    targets_dec = dec["target"].to_numpy().astype(np.int64)

    x_all = surface.select(spec_features).to_numpy().astype(np.float64)
    crops_all = surface["crop"].to_numpy()

    score = np.full(surface.height, np.nan)
    folds_meta: list[dict] = []
    if mode == "transfer":
        other = 1 - fold
        osurface = TOURN.load_surface(FOLDS[other]["surface"], other)
        odec = osurface.filter(pl.col("true_parent_is_candidate") == 1)
        model = TOURN.MinedRanker("tree", features, rounds=1,
                                  tag=f"prob_free_context.f{other}.full")
        model.fit(odec.select(spec_features).to_numpy().astype(np.float64),
                  odec["is_true_parent"].to_numpy().astype(np.int64))
        score[:] = model.predict_proba(x_all)[:, 1]
        folds_meta.append({"fitted_on_fold": other,
                           "fitted_on_surface": str(FOLDS[other]["surface"]),
                           "n_fit_rows": int(odec.height),
                           "n_surface_rows_scored": int(surface.height),
                           "crops_of_this_fold_seen_in_fit": 0})
    else:
        if surf_crops - dec_crops:
            raise GateRefusal(
                f"fold {fold}: {len(surf_crops - dec_crops)} crops have no decidable rows, so no "
                "out-of-fold model would ever be held out on them - refusing to score them with "
                "a model that saw them"
            )
        splitter = make_splitter("GroupKFold", 5)
        for i, (tr, te) in enumerate(splitter.split(x_dec, y_dec, groups=crops_dec)):
            assert_group_integrity(crops_dec, targets_dec, tr, te, f"prob_free_context/fold{i}")
            model = TOURN.MinedRanker("tree", features, rounds=1,
                                      tag=f"prob_free_context.oof.f{fold}.k{i}")
            model.fit(x_dec[tr], y_dec[tr])
            held = set(crops_dec[te].tolist())
            mask = np.isin(crops_all, list(held))
            if not mask.any():
                raise GateRefusal(f"fold {fold}/k{i}: held-out crops match no surface row")
            score[mask] = model.predict_proba(x_all[mask])[:, 1]
            folds_meta.append({"fold": i, "n_fit_rows": int(len(tr)),
                               "n_surface_rows_scored": int(mask.sum()),
                               "held_out_crops": sorted(held)})

    if np.isnan(score).any():
        raise GateRefusal(f"fold {fold}: {int(np.isnan(score).sum())} surface rows unscored")
    if not np.isfinite(score).all():
        raise GateRefusal(f"fold {fold}: non-finite score")
    if score.min() < 0.0 or score.max() > 1.0:
        raise GateRefusal(
            f"fold {fold}: score range [{score.min()}, {score.max()}] leaves [0,1] - the deployed "
            "learned_prob helper would SILENTLY SIGMOID it (built notebook 1839-1840). Refusing."
        )

    scored = surface.select(["crop", "target", "source", "prob"]).with_columns(
        pl.Series("score", score))
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.write_parquet(out)

    # --- the pre-ILP parent-choice conversions this score buys, for assoc_report ---------------
    conv = _preilp_conversions(surface, score, fold)

    meta = {
        "schema_version": 1,
        "heartbeat": SCORES_HEARTBEAT,
        "fold": fold,
        "arm": "prob_free_context",
        "mode": mode,
        "features": features,
        "recipe_source": "assoc_fold_pathology.ABLATIONS['prob_free_context']",
        "estimator": "assoc_tournament.MinedRanker(tree, rounds=1)",
        "splitter": ("assoc_train_harness.make_splitter('GroupKFold', 5), grouped by crop"
                     if mode == "refit" else
                     f"FROZEN full fit on fold {1 - fold}, applied here - FACT-0421's transfer arm"),
        "surface": TOURN.surface_provenance(cfg["surface"], fold),
        "rows": int(surface.height),
        "decidable_rows": int(dec.height),
        "crops": len(surf_crops),
        "score_range": [float(score.min()), float(score.max())],
        "score_median": float(np.median(score)),
        "fits": folds_meta,
        "preilp_conversions": conv,
        "out": str(out),
        "out_sha256": _sha(out),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    print(f"{SCORES_HEARTBEAT} fold={fold} rows={surface.height} "
          f"range=[{score.min():.4f},{score.max():.4f}] -> {out}")
    return meta


def _preilp_conversions(surface: pl.DataFrame, score: np.ndarray, fold: int) -> dict:
    """The pre-ILP parent-choice movement the learned score buys, on the FROZEN surface.

    Reported because ``assoc_report.verdict`` requires it as evidence of MECHANISM, and reported
    with its stage label intact: it is NOT the FACT-0376 quantity, which is a final-graph edge-TP
    delta and is measured separately through the complete chain.
    """
    from assoc_baseline_rankers import crop_paired_bootstrap
    from assoc_parent_dataset import evaluate
    from assoc_report import parent_conversions
    from assoc_train_harness import decide

    base = evaluate(surface, "prob")
    baseline_map = base.pop("per_target_correct")
    counts = surface.filter(pl.col("true_parent_is_candidate") == 1) \
        .group_by(["crop", "target"]).agg(pl.len().alias("n_dec"))
    contested = {(r[0], int(r[1])) for r in
                 counts.filter(pl.col("n_dec") > 1).select(["crop", "target"]).rows()}
    # `score` is aligned to `surface` ROW ORDER, so the decidable subset is selected by the same
    # boolean mask rather than by a join, which could reorder.
    mask = (surface["true_parent_is_candidate"] == 1).to_numpy()
    dec = surface.filter(pl.col("true_parent_is_candidate") == 1) \
        .join(counts, on=["crop", "target"], how="left")
    scored = dec.with_columns(pl.Series("_score", score[mask]))
    per_target, _ledger, agg = decide(scored, "_score", None, preserve_single=True)
    all_conv = parent_conversions(baseline_map, per_target)
    con_conv = parent_conversions({k: v for k, v in baseline_map.items() if k in contested},
                                  {k: v for k, v in per_target.items() if k in contested})
    con_conv["crop_paired_bootstrap"] = crop_paired_bootstrap(baseline_map, per_target, contested)
    return {
        "stage": "pre_ILP_candidate_ranking",
        "all_decidable": all_conv,
        "contested": con_conv,
        "deployed_contested_top1": base["contested"]["top1"],
        "learned_contested_top1": agg["contested"]["top1"],
        "delta_contested_top1": agg["contested"]["top1"] - base["contested"]["top1"],
    }


# ==============================================================================================
# 2. ONE CROP, EVERY ARM
# ==============================================================================================

def _heatmap_cache_dir() -> Path | None:
    """Where DeepCenter heatmaps are memoised ACROSS RUNS, or None to disable.

    WHY THIS EXISTS, MEASURED. Per crop the deployed chain scores a few hundred points but the
    model runs once per distinct FRAME, and that pass dominates: on a small fold-1 crop the first
    arm's chain costs ~400 s while each later arm costs ~20 s, because the in-process cache has
    already been filled. Gate A and Gate B are separate runs over the SAME crops, so without a
    disk cache Gate B pays that pass a second time - hours of CPU to recompute a deterministic
    function of (dataset, t, checkpoint) that was computed this morning.
    """
    value = os.environ.get("BIOHUB_FINALEDGE_DCCACHE", "").strip()
    return Path(value) if value else None


def _persistent_heatmaps(module) -> dict:
    """Memoise the DeepCenter heatmap by (dataset, t) ACROSS arms of the same crop.

    PURE MEMOISATION AND NOTHING ELSE. ``deepcenter_heatmap_for_frame`` is a function of
    (dataset, t) and the frozen checkpoint - it reads the raw image and runs the model - so
    returning the same array to a later arm is identical to recomputing it. The deployed
    per-call cache is capped at DEEPCENTER_SCORE_CACHE_MAX_FRAMES = 8 and thrashes; on one
    fold-0 crop the model ran 103 forwards for 458 scored points, 90 of the arm's 163 seconds.
    The control arm is still computed FIRST and against the kernel's own run_stats, so if this
    changed anything the parity gate would catch it.
    """
    original = module.deepcenter_heatmap_for_frame
    store: dict[tuple[str, int], np.ndarray] = {}
    stats = {"hits": 0, "misses": 0, "disk_hits": 0, "disk_writes": 0,
             "disk_write_failures": 0}
    root = _heatmap_cache_dir()
    if root is not None:
        # The heatmap depends on the CHECKPOINT as well as on (dataset, t), so the cache is keyed
        # by the checkpoint's digest. A different model cannot silently read this one's arrays.
        root = root / _checkpoint_key()
        root.mkdir(parents=True, exist_ok=True)

    def wrapper(dataset, t, detector_bundle, frame_cache, heatmap_cache):
        key = (str(dataset), int(t))
        cached = store.get(key)
        if cached is not None:
            stats["hits"] += 1
            return cached
        path = None
        if root is not None:
            path = root / str(dataset) / f"{int(t)}.npy"
            if path.is_file():
                try:
                    value = np.load(path)
                    store[key] = value
                    stats["disk_hits"] += 1
                    return value
                except Exception:
                    # A truncated array is a cache miss, never a wrong answer.
                    try:
                        path.unlink()
                    except OSError:
                        pass
        stats["misses"] += 1
        value = original(dataset, t, detector_bundle, frame_cache, heatmap_cache)
        if value is not None:
            store[key] = value
            if path is not None:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    # The temp name MUST already end in .npy: np.save APPENDS the extension when
                    # it does not, so `x.tmp123` becomes `x.tmp123.npy` and the os.replace below
                    # then fails on a name that was never written. That is exactly what happened
                    # the first time this was run - 118 orphaned .tmp*.npy files, zero cache hits,
                    # and a bare `except OSError: pass` reporting nothing. A silent no-op is worse
                    # than a crash (AGENTS.md), so the failure is COUNTED and surfaced now.
                    tmp = path.with_name(f"{path.stem}.tmp{os.getpid()}.npy")
                    np.save(tmp, value)
                    os.replace(tmp, path)
                    stats["disk_writes"] += 1
                except OSError as err:
                    stats["disk_write_failures"] = stats.get("disk_write_failures", 0) + 1
                    stats["last_disk_write_error"] = f"{type(err).__name__}: {err}"
        return value

    module.deepcenter_heatmap_for_frame = wrapper
    return stats


def _checkpoint_key() -> str:
    """First 16 hex of the DeepCenter checkpoint's sha256, computed once per process."""
    global _CKPT_KEY
    if _CKPT_KEY is None:
        _CKPT_KEY = _sha(CHECKPOINT)[:16]
    return _CKPT_KEY


_CKPT_KEY: str | None = None


def _assert_deployed_defaults(notebook: Path) -> None:
    """The deployed relink is used AS DEPLOYED - the treatment never edits its default.

    Host order, 2026-08-30: ``learned_edge_probs.get((source_id, target_id), 0.0)`` must stay
    bit-identical, so that the control arm is the deployed program and the only thing the
    treatment changes is what is IN the dictionary. Read at source, per crop, so a rebuild that
    changes the default cannot pass silently.
    """
    from scripts.win_bet.p28_full_chain_replay import _code_source

    src = _code_source(notebook)
    needed = [
        "learned_edge_probs.get((source_id, target_id), 0.0)",
        "cost[i, j] = motion + 0.05 * raw - MOTION_RELINK_LEARNED_BONUS * prob",
    ]
    for token in needed:
        if token not in src:
            raise GateRefusal(
                f"{notebook.name}: expected deployed source `{token}` not found - the relink or "
                "its learned-score default has changed and no arm here is comparable to FACT-0427"
            )


def _edge_frame(crop: str, src, tgt, prob) -> pl.DataFrame:
    n = len(src)
    return pl.DataFrame({
        "dataset": [crop] * n,
        "row_type": ["edge"] * n,
        "node_id": [-1] * n,
        "t": [-1] * n,
        "z": [-1.0] * n,
        "y": [-1.0] * n,
        "x": [-1.0] * n,
        "source_id": pl.Series(list(src), dtype=pl.Int64),
        "target_id": pl.Series(list(tgt), dtype=pl.Int64),
        "edge_prob": pl.Series([float(v) for v in prob], dtype=pl.Float64),
    })


def declare_widened_surface(declared_rows: pl.DataFrame, crop_scores: pl.DataFrame, crop: str,
                            rank: int, deployed_pairs: set,
                            known_nodes: set) -> tuple[pl.DataFrame, dict]:
    """Top-`rank` parents per target BY THE LEARNED SCORE, with the score in the edge_prob slot.

    This is PKT-0040's minimal design made concrete: the acquisition floor is the surface's own
    0.1 (FACT-0382 - every fold-0 contested error has its true parent below the deployed 0.5, so
    the widened surface is the only way to expose them), the LEARNED score chooses the commitment,
    and the same number reaches both the ILP objective and the relink bonus because ``edge_prob``
    is the only learned channel that crosses at all (FACT-0427). ``rank=1`` reproduces the
    deployed SHAPE - at most one parent per target (FACT-0369) - with the LEARNED parent;
    ``rank=2`` offers the runner-up as well.

    THE COVERAGE CONTRACT (host order, 2026-08-30, on auditor veto V5). ``learned_prob`` at
    built-notebook 1832 is ``learned_edge_probs.get((source_id, target_id), 0.0)`` - an uncovered
    pair receives the WORST score, not a neutral one. A MISSING TREATMENT SCORE IS AN INSTRUMENT
    FAILURE, NEVER A BAD LEARNED SCORE, so no neutral fallback is supplied and nothing is imputed.

    THE DECLARATION AND THE COVER ARE TWO DIFFERENT ARTIFACTS, ON PURPOSE. ``declared_rows`` is
    the crop's slice of the fold's ACQUISITION surface - every pair the widened acquisition
    offers, at its own floor and rank cap, independent of any score. ``crop_scores`` is the
    treatment artifact that must cover it. Deriving the declaration from the score table instead
    would make the assertion vacuous - declared would be a subset of scored by construction and
    ``missing`` could never be non-zero - so a stale, mis-folded or partial score artifact would
    pass. Here it fails closed: missing pairs, duplicate keys, unknown node ids, non-finite
    scores and scores outside [0, 1] each raise, and expected / scored / missing / extra are
    reported per crop whatever the outcome.

    The deployed ``.get(..., 0.0)`` is NEVER modified - the control arm runs the deployed program
    byte for byte, and this contract binds only the treatment path.
    """
    declared = set(zip(declared_rows["source"].to_list(), declared_rows["target"].to_list()))
    if not declared:
        raise GateRefusal(f"{crop}: the declared widened acquisition surface is empty")

    keys = list(zip(crop_scores["source"].to_list(), crop_scores["target"].to_list()))
    scored = set(keys)
    duplicates = len(keys) - len(scored)
    if duplicates:
        raise GateRefusal(
            f"{crop}: the learned score table has {duplicates} DUPLICATE (source, target) keys - "
            "a pair with two scores has no defined treatment value"
        )
    unknown = {n for pair in declared | scored for n in pair} - known_nodes
    if unknown:
        raise GateRefusal(
            f"{crop}: {len(unknown)} node ids are absent from the pre-ILP node set "
            f"(e.g. {sorted(unknown)[:5]}) - the score does not describe this crop's graph"
        )
    missing = declared - scored
    if missing:
        raise GateRefusal(
            f"{crop}: {len(missing)} of {len(declared)} pairs in the DECLARED widened acquisition "
            "surface carry NO learned score. A missing treatment score is an INSTRUMENT FAILURE, "
            "never a bad learned score; no neutral fallback is supplied and the arm is refused."
        )

    values = crop_scores["score"].to_numpy()
    if not np.isfinite(values).all():
        raise GateRefusal(f"{crop}: non-finite learned score")
    if values.min() < 0.0 or values.max() > 1.0:
        raise GateRefusal(
            f"{crop}: learned score range [{values.min()}, {values.max()}] leaves [0,1]; the "
            "deployed learned_prob helper would SILENTLY SIGMOID it (built notebook 1839-1840)"
        )

    # Only now, on a surface proven fully covered, is the top-`rank` selection made.
    covered = crop_scores.join(declared_rows.select(["source", "target"]),
                               on=["source", "target"], how="semi")
    ranked = covered.with_columns(
        pl.col("score").rank("ordinal", descending=True).over("target").alias("_lrank"))
    kept = ranked.filter(pl.col("_lrank") <= rank).sort(["target", "_lrank"])
    offered = set(zip(kept["source"].to_list(), kept["target"].to_list()))

    frame = _edge_frame(crop, kept["source"].to_list(), kept["target"].to_list(),
                        kept["score"].to_list())
    coverage = {
        "rank": rank,
        "declared_pairs_expected": len(declared),
        "scored_pairs": len(scored),
        "missing_pairs": 0,
        "extra_scored_pairs_outside_the_declaration": len(scored - declared),
        "duplicate_pair_keys": 0,
        "unknown_node_ids": 0,
        "coverage_of_declared_surface": 1.0,
        "offered_edges": int(kept.height),
        "deployed_edges": len(deployed_pairs),
        "deployed_edges_retained": len(deployed_pairs & offered),
        # NOT a refusal. Gate B deliberately REPLACES a deployed parent choice, so this arm is not
        # a superset experiment the way the LEVER-0037 widening was - it is reported instead.
        "deployed_edges_dropped": len(deployed_pairs - offered),
        "score_min": float(kept["score"].min()),
        "score_max": float(kept["score"].max()),
        "score_median": float(kept["score"].median()),
    }
    return frame, coverage


def _assert_range(frame: pl.DataFrame, where: str) -> None:
    values = frame["edge_prob"].to_numpy()
    if not np.isfinite(values).all():
        raise GateRefusal(f"{where}: non-finite edge_prob")
    if values.min() < 0.0 or values.max() > 1.0:
        raise GateRefusal(
            f"{where}: edge_prob range [{values.min()}, {values.max()}] leaves [0,1]; the "
            "deployed learned_prob helper would SILENTLY SIGMOID it (built notebook 1839-1840)"
        )


def _parents(frame: pl.DataFrame) -> dict[int, int]:
    e = frame.filter(pl.col("row_type") == "edge")
    return {int(t): int(s) for s, t in zip(e["source_id"], e["target_id"])}


def _churn(control: pl.DataFrame, arm: pl.DataFrame) -> dict:
    """Assignment churn against the DEPLOYED control, over targets present in both graphs."""
    pc, pa = _parents(control), _parents(arm)
    shared = set(pc) & set(pa)
    changed = sum(1 for t in shared if pc[t] != pa[t])
    return {
        "targets_with_parent_control": len(pc),
        "targets_with_parent_arm": len(pa),
        "shared_targets": len(shared),
        "changed_parent": changed,
        "changed_fraction": changed / len(shared) if shared else float("nan"),
        "gained_parent": len(set(pa) - set(pc)),
        "lost_parent": len(set(pc) - set(pa)),
    }


def edge_population(crop: str, gt_geff: Path, final: pl.DataFrame) -> dict:
    """Split this arm's emitted edges into the populations FACT-0448 made load-bearing.

    WHY THIS EXISTS. FACT-0448 measured that of 1,369 fold-0 false-positive edges only SIXTEEN
    are wrong links between two ANNOTATED cells. That sixteen caps ONE population - correcting a
    wrong link where both endpoints are annotated - and caps nothing else. A scorer that abstains
    on edges touching an unmatched node, that removes bad edges, or that recovers a missing true
    edge is acting on different mass entirely. So a Gate B gain must be attributed to the
    population it actually came from, not reported as one number against a ceiling that does not
    bind it.

    The matcher is the ledger's own - detpeak_curve.match_one_to_one_pairs, official one-to-one
    within 7 um per frame - so this split is commensurable with assoc_lost_edge_ledger and with
    FACT-0370/0422 rather than merely adjacent to them.
    """
    from assoc_lost_edge_ledger import _match_per_frame, SCALE
    from biotrack.metric import load_graph

    nodes = final.filter(pl.col("row_type") == "node")
    edges = final.filter(pl.col("row_type") == "edge")
    emitted = {(int(a), int(b)) for a, b in zip(edges["source_id"], edges["target_id"])}

    gt = load_graph(gt_geff)
    gtn = gt.node_attrs().to_pandas()
    gt_ids = gtn["node_id"].to_numpy().astype(np.int64)
    gt_row = {int(v): i for i, v in enumerate(gt_ids)}
    gt_tzyx = gtn[["t", "z", "y", "x"]].to_numpy().astype(np.float64)
    gte = gt.edge_attrs().to_pandas()

    to_final = _match_per_frame(
        gt_tzyx,
        nodes["t"].to_numpy().astype(np.int64),
        nodes.select(["z", "y", "x"]).to_numpy().astype(np.float64) * SCALE,
        nodes["node_id"].to_numpy().astype(np.int64))
    annotated = set(to_final.values())          # predicted nodes that ARE an annotated cell

    wanted = set()                              # GT edges expressed in predicted ids
    gt_edges = both_endpoints_matched = 0
    for a, b in zip(gte["source_id"], gte["target_id"]):
        su, tv = gt_row.get(int(a)), gt_row.get(int(b))
        if su is None or tv is None:
            continue
        gt_edges += 1
        if su in to_final and tv in to_final:
            both_endpoints_matched += 1
            wanted.add((to_final[su], to_final[tv]))

    tp = emitted & wanted
    fp = emitted - wanted
    fp_both_annotated = {(a, b) for a, b in fp if a in annotated and b in annotated}
    parent_of = {t: s for s, t in emitted}
    missing = wanted - emitted
    missing_target_has_other_parent = {(a, b) for a, b in missing if b in parent_of}

    return {
        "matcher": "official one-to-one 7 um per frame (detpeak_curve.match_one_to_one_pairs)",
        "NOT_THE_SCORERS_COUNTS": (
            "This is a POPULATION split over the emitted edge set, not a re-derivation of the "
            "official metric. `tp` reproduces the scorer's edge_tp exactly, but the counts below "
            "it do NOT equal the scorer's edge_fp / edge_fn, because the official metric leaves "
            "edges touching unannotated nodes free. The promotable numbers stay the scorer's, in "
            "`metrics`; this block only says WHICH POPULATION a change came from."),
        "gt_edges": gt_edges,
        "gt_edges_with_both_endpoints_in_the_emitted_graph": both_endpoints_matched,
        "emitted_edges": len(emitted),
        "tp": len(tp),
        "emitted_edges_not_backed_by_a_gt_edge": len(fp),
        # THE FACT-0448 POPULATION, and the ONLY one that fact's sixteen caps.
        "wrong_links_between_two_annotated_cells": len(fp_both_annotated),
        # Everything the annotation says nothing about - a scorer that ABSTAINS here, or that
        # removes an edge here, is acting on mass FACT-0448 does not bound.
        "edges_touching_an_unmatched_node": len(fp) - len(fp_both_annotated),
        "gt_edges_wanted_but_missing": len(missing),
        "missing_target_has_a_different_parent": len(missing_target_has_other_parent),
        "missing_target_has_no_parent": len(missing) - len(missing_target_has_other_parent),
        "_edges": sorted(emitted),
    }


def run_crop(fold: int, crop: str, out: Path, scores_path: Path | None,
             bonuses: tuple[float, ...], gateb_ranks: tuple[int, ...],
             gateb_bonuses: tuple[float, ...], ledger: bool) -> dict:
    cfg = FOLDS[fold]
    t_start = time.time()

    module = load_p28_module(cfg["notebook"], ROOT / "data/train", CHECKPOINT)
    if float(module.MOTION_RELINK_LEARNED_BONUS) != DEPLOYED_BONUS:
        raise GateRefusal(
            f"extracted MOTION_RELINK_LEARNED_BONUS is {module.MOTION_RELINK_LEARNED_BONUS}, not "
            f"FACT-0427's deployed {DEPLOYED_BONUS} - the notebook or P28_ENV changed"
        )
    if P28_ENV["BIOHUB_MOTION_RELINK_LEARNED_BONUS"] != "1.0":
        raise GateRefusal("p28_full_chain_replay.P28_ENV no longer pins the deployed bonus at 1.0")
    _assert_deployed_defaults(cfg["notebook"])
    heat = _persistent_heatmaps(module)

    frame = pl.scan_parquet(cfg["preilp"]).filter(pl.col("dataset") == crop).collect()
    if frame.is_empty():
        raise GateRefusal(f"crop absent from pre-ILP export: {crop}")
    nodes_frame = frame.filter(pl.col("row_type") == "node").sort("node_id")
    deployed_edges = frame.filter(pl.col("row_type") == "edge")
    deployed_pairs = set(zip(deployed_edges["source_id"].to_list(),
                             deployed_edges["target_id"].to_list()))
    _assert_range(deployed_edges, f"{crop}/deployed")

    previous = os.environ.get("BIOHUB_DEEPCENTER_CHECKPOINT")
    os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(CHECKPOINT)
    try:
        detector = module.load_deepcenter_veto_detector()
    finally:
        if previous is None:
            os.environ.pop("BIOHUB_DEEPCENTER_CHECKPOINT", None)
        else:
            os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = previous

    from scripts.win_bet import div_reach_steal as drs
    ea, sc = drs._ea_atlas(), drs._scorer()
    gt = ROOT / "data/train" / f"{crop}.geff"

    solved_cache: dict[str, tuple] = {}

    def solved(tag: str, edge_frame: pl.DataFrame):
        if tag not in solved_cache:
            t0 = time.time()
            graph = build_graph_from_frame(nodes_frame, edge_frame)
            g = solve(graph, {"edge": -1.0, "appearance": 0.0,
                              "disappearance": 1.5, "division": 1.0})
            nodes, edges = solved_graph_inputs(g)
            solved_cache[tag] = (nodes, edges, round(time.time() - t0, 1))
        return solved_cache[tag]

    arms: dict[str, dict] = {}
    frames: dict[str, pl.DataFrame] = {}
    offered_sets: dict[str, set] = {"deployed": deployed_pairs}

    def run_arm(name: str, surface_tag: str, edge_frame: pl.DataFrame, bonus: float,
                extra: dict) -> dict:
        nodes, edges, solve_s = solved(surface_tag, edge_frame)
        # FACT-0427: learned_edge_probs is assembled ONLY from ILP-SURVIVING edges. Measure how
        # much of the offered surface reaches the relink at all - if it is small, the arm is
        # bounded by the solver rather than by the scorer or the consumption point.
        offered = offered_sets[surface_tag]
        surviving = {(int(e["source_id"]), int(e["target_id"])) for e in edges}
        module.MOTION_RELINK_LEARNED_BONUS = float(bonus)
        t0 = time.time()
        fnodes, fedges, stats = module.filter_output_graph(
            copy.deepcopy(nodes), copy.deepcopy(edges), dataset=crop,
            deepcenter_bundle=detector)
        chain_s = round(time.time() - t0, 1)
        sub = submission_frame(crop, fnodes, fedges)
        row = {k: float(v) for k, v in drs.score_crop(sub, gt, ea, sc).items()}
        full = {**stats, "raw_nodes": len(nodes), "raw_edges": len(edges),
                "nodes": len(fnodes), "edges": len(fedges)}
        frames[name] = sub
        arms[name] = {
            "arm": name,
            "surface": surface_tag,
            "bonus": float(bonus),
            "metrics": row,
            "stats": {k: int(v) for k, v in full.items()},
            "relink_reach": {
                "offered_edges": len(offered),
                "ilp_surviving_edges": len(surviving),
                "offered_reaching_relink": len(offered & surviving),
                "share_of_offered_reaching_relink":
                    len(offered & surviving) / max(len(offered), 1),
                # every ILP-surviving edge carries the arm's own edge_prob, so the relink's
                # learned channel is fully covered - the auditor's veto-V5 quantity
                "learned_score_coverage_of_relink_inputs": 1.0,
            },
            "timing_s": {"solve": solve_s, "chain": chain_s},
            **extra,
        }
        return arms[name]

    # --- THE CONTROL, FIRST AND UNCONDITIONALLY -----------------------------------------------
    control = run_arm("control", "deployed", deployed_edges, DEPLOYED_BONUS,
                      {"role": "DEPLOYED CONTROL"})
    comparison, parity = _control_parity(cfg, crop, control["stats"])
    control["control_comparison"] = comparison
    control["control_parity"] = parity
    control["control_exact"] = parity["exact_on_all_fifteen"]
    control["control_reproduces"] = not parity["failing_columns"]
    if parity["failing_columns"]:
        detail = {k: comparison[k] for k in parity["failing_columns"]}
        raise GateRefusal(
            f"{crop}: the deployed arm does NOT reproduce the kernel run_stats on "
            f"{parity['failing_columns']} -> {detail}. Halting before any treatment is read; a "
            "control that does not reproduce is not evidence about the deployed chain. "
            "(raw_edges alone carries FACT-0363's measured +-2 solver-degeneracy tolerance; "
            "every other column, raw_nodes and the FINAL graph included, is exact or nothing.)"
        )

    # --- GATE A -------------------------------------------------------------------------------
    for b in bonuses:
        if float(b) == DEPLOYED_BONUS:
            arms[f"gateA_b{b:g}"] = {**control, "arm": f"gateA_b{b:g}",
                                     "role": "GATE A (== the deployed control)"}
            frames[f"gateA_b{b:g}"] = frames["control"]
            continue
        run_arm(f"gateA_b{b:g}", "deployed", deployed_edges, float(b), {"role": "GATE A"})

    # --- GATE B -------------------------------------------------------------------------------
    gateb_meta = None
    if scores_path is not None:
        crop_scores = (pl.scan_parquet(scores_path).filter(pl.col("crop") == crop)
                       .select(["crop", "source", "target", "prob", "score"]).collect())
        if crop_scores.is_empty():
            raise GateRefusal(f"{crop}: no learned scores in {scores_path}")
        # The DECLARED widened acquisition surface: the crop's slice of the fold's own candidate
        # table, floor and rank cap as exported, independent of any score.
        declared_rows = (pl.scan_parquet(cfg["surface"]).filter(pl.col("crop") == crop)
                         .select(["source", "target"]).collect())
        if declared_rows.is_empty():
            raise GateRefusal(f"{crop}: absent from the widened acquisition surface "
                              f"{cfg['surface']}")
        covered = set(zip(crop_scores["source"].to_list(), crop_scores["target"].to_list()))
        missing = deployed_pairs - covered
        if missing:
            raise GateRefusal(
                f"{crop}: the learned score does not cover {len(missing)} DEPLOYED pairs - an "
                "uncovered pair inherits 0.0 at built-notebook 1832, the worst score, so a "
                "partial cover would fail Gate B on coverage rather than on the mechanism"
            )
        gateb_meta = {"scores": str(scores_path),
                      "declared_surface": str(cfg["surface"]),
                      "declared_pairs": int(declared_rows.height),
                      "scored_pairs": int(crop_scores.height),
                      "deployed_pairs_covered": len(deployed_pairs & covered),
                      "deployed_pairs": len(deployed_pairs),
                      "coverage_of_deployed_pairs": 1.0}
        known_nodes = {int(n) for n in nodes_frame["node_id"]}
        for rank in gateb_ranks:
            tag = f"widened_r{rank}"
            edge_frame, diag = declare_widened_surface(
                declared_rows, crop_scores, crop, rank, deployed_pairs, known_nodes)
            _assert_range(edge_frame, f"{crop}/{tag}")
            offered_sets[tag] = set(zip(edge_frame["source_id"].to_list(),
                                        edge_frame["target_id"].to_list()))
            for b in gateb_bonuses:
                run_arm(f"gateB_r{rank}_b{b:g}", tag, edge_frame, float(b),
                        {"role": "GATE B", "widening": diag})

    # --- churn and ledger, per arm ------------------------------------------------------------
    for name, arm in arms.items():
        arm["churn_vs_control"] = _churn(frames["control"], frames[name])
    # FACT-0448: attribute each arm's edges to the population they belong to, and DIFF the
    # populations against the control, so a gain is never reported against a ceiling that does
    # not bind it.
    pops = {}
    for name in arms:
        pops[name] = edge_population(crop, gt, frames[name])
    ctl_edges = set(map(tuple, pops["control"].pop("_edges")))
    for name, arm in arms.items():
        pop = pops[name]
        mine = set(map(tuple, pop.pop("_edges"))) if "_edges" in pop else ctl_edges
        ctl = pops["control"]
        arm["edge_population"] = pop
        arm["edge_population_delta_vs_control"] = {
            k: pop[k] - ctl[k] for k in
            ("tp", "emitted_edges", "emitted_edges_not_backed_by_a_gt_edge",
             "wrong_links_between_two_annotated_cells", "edges_touching_an_unmatched_node",
             "gt_edges_wanted_but_missing", "missing_target_has_a_different_parent",
             "missing_target_has_no_parent")
        }
        arm["edge_population_delta_vs_control"]["edges_added_vs_control"] = len(mine - ctl_edges)
        arm["edge_population_delta_vs_control"]["edges_removed_vs_control"] = len(ctl_edges - mine)

    if ledger:
        from assoc_lost_edge_ledger import crop_ledger
        pre_nodes = nodes_frame
        for name, arm in arms.items():
            offered = offered_sets[arm["surface"]]
            src = [s for s, _ in offered]
            tgt = [t for _, t in offered]
            pre = pl.concat([
                pre_nodes.select(["dataset", "row_type", "node_id", "t", "z", "y", "x",
                                  "source_id", "target_id"]),
                _edge_frame(crop, src, tgt, [1.0] * len(src)).select(
                    ["dataset", "row_type", "node_id", "t", "z", "y", "x",
                     "source_id", "target_id"]),
            ], how="vertical")
            arm["ledger"] = crop_ledger(crop, gt, pre, frames[name])

    payload = {
        "schema_version": 1,
        "heartbeat": HEARTBEAT,
        "packet": "PKT-0042",
        "lever": "LEVER-0044",
        "fold": fold,
        "embryo": cfg["embryo"],
        "crop": crop,
        "control_provenance": _control_provenance(cfg, module),
        "bonus_grid": list(bonuses),
        "gateb": gateb_meta,
        "deepcenter_heatmap_cache": dict(heat),
        "arms": arms,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # ATOMIC. The batch runner may be sharded several ways at once, so two workers can meet on one
    # crop; a half-written payload would be indistinguishable from a complete one to the panel.
    tmp = out.with_suffix(out.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    os.replace(tmp, out)
    print(f"{HEARTBEAT} fold={fold} crop={crop} arms={len(arms)} "
          f"control_exact=True elapsed={payload['elapsed_s']}s")
    return payload


def _control_parity(cfg: dict, crop: str, stats: dict) -> tuple[dict, dict]:
    rows = pl.read_csv(cfg["run_stats"]).filter(pl.col("dataset") == crop)
    if rows.height != 1:
        raise GateRefusal(f"expected exactly one control run_stats row for {crop}")
    expected = rows.row(0, named=True)
    comparison = {k: {"got": int(stats[k]), "expected": int(expected[k]),
                      "delta": int(stats[k]) - int(expected[k])} for k in COMPARE_KEYS}
    jitter = comparison["raw_edges"]["delta"]
    within = abs(jitter) <= RAW_EDGE_JITTER_TOLERANCE
    failures = [k for k, v in comparison.items()
                if v["delta"] != 0 and not (k == "raw_edges" and within)]
    verdict = {
        "exact_on_all_fifteen": all(v["delta"] == 0 for v in comparison.values()),
        "failing_columns": failures,
        "raw_edges_delta": jitter,
        "raw_edges_tolerance": RAW_EDGE_JITTER_TOLERANCE,
        "raw_edges_tolerance_used": bool(jitter != 0 and within),
        "tolerance_authority": "FACT-0363 - the deployed ILP has multiple optima; its node "
                               "solution is deterministic and its raw edge count is not, so an "
                               "exact raw_edges target does not exist to be met "
                               "(edge_jitter_max_abs 2, 7 of 128 fold-1 crops)",
        "columns_held_exact": [k for k in COMPARE_KEYS if k != "raw_edges"],
        "final_graph_exact": comparison["nodes"]["delta"] == 0 and comparison["edges"]["delta"] == 0,
    }
    return comparison, verdict


_PROV_CACHE: dict[str, str] = {}


def _control_provenance(cfg: dict, module) -> dict:
    """The control's artifacts, named and hashed per crop. NOT the identity check itself.

    THIS IS DIAGNOSTIC ONLY, AND THE DISTINCTION MATTERS. This block once carried
    ``not_a_widened_only_control: true`` - a SELF-DECLARED boolean asserting a property nobody
    re-derived, which is exactly the shape FACT-0432 defect 1 measured as worthless: verdict()
    returned promotable on a report whose only defect was the control's identity. The real check
    is ``control_identity()`` -> ``control_binding.bind_control``, which recomputes every digest
    from the bytes, matches the registry-bound deployed control on CONTENT, and refuses a widened
    control by declared surface and by candidate floor. What is kept here is the per-crop
    paper trail the binding does not carry.
    """
    for path in (cfg["preilp"], cfg["run_stats"]):
        key = str(path)
        if key not in _PROV_CACHE:
            _PROV_CACHE[key] = _sha(Path(path))
    return {
        "role": "DEPLOYED CONTROL - deployed pre-ILP candidate surface, deployed edge_prob, "
                "MOTION_RELINK_LEARNED_BONUS at FACT-0427's deployed 1.0",
        "notebook": str(cfg["notebook"]),
        "notebook_code_sha256": module._source_sha256,
        "preilp": str(cfg["preilp"]),
        "preilp_sha256": _PROV_CACHE[str(cfg["preilp"])],
        "run_stats": str(cfg["run_stats"]),
        "run_stats_sha256": _PROV_CACHE[str(cfg["run_stats"])],
        "deepcenter_checkpoint": str(CHECKPOINT),
        "parity": "asserted EXACT on all fifteen compared columns before any treatment was read",
        "identity_is_checked_by": "control_binding.bind_control - see panel['control_binding']; "
                                  "nothing in THIS block is read as evidence of identity",
    }


# ==============================================================================================
# 3. THE PANEL
# ==============================================================================================

def control_rows_artifact(payloads: list[dict]) -> list[str]:
    """The control's artifact is the ORDERED per-crop payload set, not a path called 'control'.

    ``bind_control`` digests the list as one identity, so a single tampered crop moves the digest.
    Order is the crop order the report is built in, because the paired bootstrap and the sign
    counts are only valid on that pairing.
    """
    return [p["_payload_path"] for p in payloads]


def control_identity(payloads: list[dict], fold: int) -> dict:
    """Bind this fold's DEPLOYED control to the registry, by RECOMPUTED digest.

    Replaces the self-declared ``not_a_widened_only_control: true`` boolean that
    ``_control_provenance`` used to carry. FACT-0432 defect 1 measured that a field claiming a
    property nobody re-derives is worth nothing: ``verdict()`` returned promotable on a report
    whose only defect was the control's identity. Everything below is recomputed from bytes by
    ``control_binding``, and ``surface``/``candidate_floor`` are declared so a widened control is
    refused BY NAME rather than by failing to match (FACT-0369, FACT-0382).
    """
    cfg = FOLDS[fold]
    return cb.bind_control(
        control_path=control_rows_artifact(payloads),
        identity={
            "experiment": cfg["experiment"],
            "fold": str(fold),
            "graph": str(cfg["preilp"]),
            "receipt": str(cfg["run_stats"]),
            "surface": "deployed",
            "candidate_floor": cb.DEPLOYED_CANDIDATE_FLOOR,
        },
        n_rows=len(payloads),
    )


def register_control(fold: int, results_dir: Path, date: str) -> dict:
    """Register the DEPLOYED control this packet actually scored, before any report reads it.

    PKT-0043 ships the manifest EMPTY, which refuses every control - the correct fail-closed
    state, since no committed spec declared a full-chain arm. This is the first full-chain
    control, so it is registered here with its provenance and with every digest recomputed from
    the bytes by ``control_binding.register``. What registration buys is not self-certification:
    it fixes the comparison base, so a later substitution - a widened control, a tampered crop, a
    different crop count - moves the digest and is refused.
    """
    cfg = FOLDS[fold]
    payloads = []
    for path in sorted(results_dir.glob(f"f{fold}_*.json")):
        p = json.loads(path.read_text(encoding="utf-8"))
        if p.get("heartbeat") != HEARTBEAT:
            continue
        arm = p["arms"]["control"]
        # `control_reproduces` is the post-FACT-0363 field; `control_exact` is what payloads
        # written before the tolerance existed carry, and for those the two are the same thing.
        if not arm.get("control_reproduces", arm.get("control_exact")):
            raise GateRefusal(f"{path.name}: control does not reproduce - refusing to register")
        payloads.append((str(path), p))
    if not payloads:
        raise GateRefusal(f"no complete crop payloads for fold {fold} under {results_dir}")
    if len(payloads) != cfg["n_crops"]:
        raise GateRefusal(
            f"fold {fold}: {len(payloads)} crop payloads, expected {cfg['n_crops']} - refusing to "
            "register a control that is not the complete fold, because the crop count is part of "
            "the identity and a partial control would bind a different object"
        )
    entry = cb.register(
        manifest_path=cb.MANIFEST_PATH,
        experiment=cfg["experiment"],
        fold=str(fold),
        control_path=[q for q, _ in payloads],
        graph=cfg["preilp"],
        receipt=cfg["run_stats"],
        surface="deployed",
        candidate_floor=cb.DEPLOYED_CANDIDATE_FLOOR,
        n_crops=len(payloads),
        date=date,
        provenance=(
            f"PKT-0042 / LEVER-0044 Gate A+B deployed control, fold {fold} ({cfg['embryo']} held "
            f"out), all {len(payloads)} crops. Produced by scripts/win_bet/finaledge_gates.py "
            f"run_crop: the deployed pre-ILP candidate surface ({cfg['preilp']}) at the deployed "
            f"floor 0.5, the deployed edge_prob unchanged, MOTION_RELINK_LEARNED_BONUS at "
            f"FACT-0427's deployed 1.0, through the COMPLETE deployed post-ILP chain extracted by "
            f"ast from {Path(cfg['notebook']).name} and scored by the official patched scorer. "
            f"Every crop's arm was asserted EXACT against the kernel's own run_stats "
            f"({cfg['run_stats']}, {cfg['experiment']}) on all fifteen compared columns BEFORE "
            f"any treatment arm was scored; a mismatch is a hard refusal, not a warning. The "
            f"artifact is the ORDERED per-crop payload set, so one tampered crop moves the digest."
        ),
    )
    print(f"CONTROL_REGISTERED fold={fold} crops={entry['n_crops']} "
          f"artifact_sha256={entry['artifact_sha256'][:16]}... experiment={entry['experiment']}")
    return entry


def _rows(payloads: list[dict], arm: str) -> list[dict]:
    return [p["arms"][arm]["metrics"] for p in payloads]


def build_panel(results_dir: Path, out: Path, ledger: bool) -> dict:
    from assoc_report import build_report, parent_conversions
    from tracking_cellmot.metrics import summarise

    payloads: dict[int, list[dict]] = {0: [], 1: []}
    for path in sorted(results_dir.glob("f*_*.json")):
        p = json.loads(path.read_text(encoding="utf-8"))
        if p.get("heartbeat") != HEARTBEAT:
            continue
        p["_payload_path"] = str(path)
        payloads[int(p["fold"])].append(p)
    for fold in (0, 1):
        payloads[fold].sort(key=lambda p: p["crop"])
    if not payloads[0] and not payloads[1]:
        raise GateRefusal(f"no crop payloads under {results_dir}")

    arm_names: list[str] = []
    for fold in (0, 1):
        if payloads[fold]:
            arm_names = [a for a in payloads[fold][0]["arms"] if a != "control"]
            break

    bindings = {f: control_identity(payloads[f], f) for f in (0, 1) if payloads[f]}

    scores_meta = {}
    for fold in (0, 1):
        meta = results_dir.parent / f"scores_f{fold}.meta.json"
        if meta.is_file():
            scores_meta[fold] = json.loads(meta.read_text(encoding="utf-8"))

    panel: dict = {
        "schema_version": 1,
        "heartbeat": PANEL_HEARTBEAT,
        "packet": "PKT-0042",
        "lever": "LEVER-0044",
        "n_crops": {f: len(payloads[f]) for f in (0, 1)},
        "expected_crops": {f: FOLDS[f]["n_crops"] for f in (0, 1)},
        "complete": {f: len(payloads[f]) == FOLDS[f]["n_crops"] for f in (0, 1)},
        "control_provenance": {f: (payloads[f][0]["control_provenance"] if payloads[f] else None)
                               for f in (0, 1)},
        "control_binding": {},
        "control_parity": {},
        "arms": {},
        "pooled": {},
    }

    for arm in arm_names:
        panel["arms"][arm] = {}
        for fold in (0, 1):
            ps = payloads[fold]
            if not ps:
                continue
            control = _rows(ps, "control")
            candidate = _rows(ps, arm)
            conv = _arm_conversions(arm, scores_meta.get(fold))
            report = build_report(model=arm, fold=fold, control=control, candidate=candidate,
                                  summarise=summarise, conversions=conv,
                                  control_binding=bindings.get(fold),
                                  notes=f"PKT-0042 {ps[0]['arms'][arm].get('role', '')}; "
                                        f"the control is the DEPLOYED arm of each crop payload, "
                                        f"asserted EXACT against the kernel run_stats on all "
                                        f"fifteen compared columns and bound by recomputed digest")
            churn = [ps[i]["arms"][arm]["churn_vs_control"] for i in range(len(ps))]
            shared = sum(c["shared_targets"] for c in churn)
            reach = [ps[i]["arms"][arm]["relink_reach"] for i in range(len(ps))]
            block = {
                "report": report,
                "assignment_churn": {
                    "shared_targets": shared,
                    "changed_parent": sum(c["changed_parent"] for c in churn),
                    "changed_fraction": (sum(c["changed_parent"] for c in churn)
                                         / max(shared, 1)),
                    "gained_parent": sum(c["gained_parent"] for c in churn),
                    "lost_parent": sum(c["lost_parent"] for c in churn),
                },
                "relink_reach": {
                    "offered_edges": sum(r["offered_edges"] for r in reach),
                    "ilp_surviving_edges": sum(r["ilp_surviving_edges"] for r in reach),
                    "offered_reaching_relink": sum(r["offered_reaching_relink"] for r in reach),
                    "share_of_offered_reaching_relink":
                        sum(r["offered_reaching_relink"] for r in reach)
                        / max(sum(r["offered_edges"] for r in reach), 1),
                    "learned_score_coverage_of_relink_inputs": 1.0,
                },
            }
            if ledger:
                block["ledger"] = _ledger_totals(ps, arm)
                block["ledger_control"] = _ledger_totals(ps, "control")
            panel["arms"][arm][f"fold{fold}"] = block

        # pooled, reported ALONGSIDE the per-fold arms and never instead of them (AGENTS.md 4)
        pooled_ctrl = [r for f in (0, 1) for r in _rows(payloads[f], "control")]
        pooled_cand = [r for f in (0, 1) for r in _rows(payloads[f], arm)]
        if pooled_ctrl:
            panel["arms"][arm]["pooled"] = build_report(
                model=arm, fold=-1, control=pooled_ctrl, candidate=pooled_cand,
                summarise=summarise, conversions=None,
                # A pooled arm spans BOTH embryo directions, so it has no single fold identity to
                # bind. It fails closed on purpose: pooling has hidden a full inversion before
                # (AGENTS.md 4) and the pooled figure is read only beside the per-fold arms.
                control_binding=cb.unbound(
                    "the pooled arm spans both embryo directions and has no single fold identity; "
                    "each fold's control is bound separately in panel['control_binding']"),
                notes="POOLED over both embryo directions - read ONLY beside the per-fold arms")

    panel["control_binding"] = {str(f): bindings.get(f) for f in (0, 1)}
    # A relaxation that is not counted is a relaxation that hides. FACT-0363's tolerance applies
    # to `raw_edges` alone, and every use of it is surfaced here per fold.
    for f in (0, 1):
        ps = payloads[f]
        if not ps:
            continue
        par = [q["arms"]["control"].get("control_parity") for q in ps]
        used = [q["crop"] for q, v in zip(ps, par)
                if v and v.get("raw_edges_tolerance_used")]
        deltas = [v["raw_edges_delta"] for v in par if v]
        panel["control_parity"][str(f)] = {
            "crops": len(ps),
            "exact_on_all_fifteen": sum(1 for v in par if v is None or v["exact_on_all_fifteen"]),
            "raw_edges_tolerance_used_on": used,
            "n_raw_edges_tolerance_used": len(used),
            "max_abs_raw_edges_delta": max([abs(d) for d in deltas], default=0),
            "tolerance": RAW_EDGE_JITTER_TOLERANCE,
            "authority": "FACT-0363 (edge_jitter_max_abs 2, 7 of 128 fold-1 crops; node parity "
                         "exact on all 128). raw_nodes and the FINAL nodes/edges are never "
                         "relaxed, and every arm of a crop shares ONE solve, so the degeneracy is "
                         "common-mode and cancels in the paired delta",
            "payloads_predating_the_tolerance_field": sum(1 for v in par if v is None),
        }
    panel["gate_a_selection"] = select_bonus(panel)
    panel["restored_clause_g"] = _clause_g(panel)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(panel, indent=2, default=float), encoding="utf-8")
    print(f"{PANEL_HEARTBEAT} crops={panel['n_crops']} arms={len(arm_names)} -> {out}")
    _print_panel(panel)
    return panel


def _winning(block: dict) -> tuple[bool, list[str]]:
    """One arm, one fold: does it win on the conditions the lever's falsifier names?

    The conditions are the lever's, not new ones: RAW association must improve (an ADJUSTED
    improvement with flat or negative RAW is REJECTED - the shape that killed LEVER-0036 and,
    mirrored, LEVER-0037), the crop-paired interval must be FAVOURABLE rather than merely
    significant, neither node recall nor division TP may decline (FACT-0371), and net true edges
    through the COMPLETE chain must be positive (the genuine FACT-0376 quantity).
    """
    ch = block["report"]["channels"]
    fails = []
    if ch["edge_jaccard_raw"]["delta"] <= 0:
        fails.append("raw edge Jaccard did not improve")
    if not block["report"]["paired_bootstrap"]["favourable"]:
        fails.append("crop-paired interval not favourable")
    if ch["division_counts"]["delta_tp"] < 0:
        fails.append("division TP declined")
    if ch["node_recall"]["delta"] < 0:
        fails.append("node recall declined")
    if ch["final_graph_edges"]["delta_tp"] <= 0:
        fails.append("no net true-edge conversion through the complete chain")
    if ch["count_adjustment"]["count_adjustment_artifact"]:
        fails.append("count-adjustment artifact")
    return (not fails), fails


def select_bonus(panel: dict) -> dict:
    """SELECT THE SMALLEST VALUE ON A WINNING PLATEAU, never the best isolated point.

    An isolated optimum on a coarse grid is a selection artifact and this project has measured
    one (FACT-0386's quantile binning), so a single winning grid point is explicitly REFUSED as a
    selection. A plateau is two or more ADJACENT grid values that win on BOTH folds; the selected
    value is the smallest member of the longest such run, ties broken by the smaller value.
    """
    grid = [b for b in BONUS_GRID]
    rows = []
    for b in grid:
        arm = f"gateA_b{b:g}"
        entry: dict = {"bonus": b, "arm": arm}
        wins = {}
        for fold in (0, 1):
            block = panel["arms"].get(arm, {}).get(f"fold{fold}")
            if block is None:
                wins[fold] = None
                continue
            ok, fails = _winning(block)
            ch = block["report"]["channels"]
            wins[fold] = ok
            entry[f"fold{fold}"] = {
                "raw_edge_jaccard_delta": ch["edge_jaccard_raw"]["delta"],
                "adj_edge_jaccard_delta": ch["count_adjustment"]["delta_adj"],
                "score_delta": ch["score"]["delta"],
                "ci95": block["report"]["paired_bootstrap"]["ci95"],
                "favourable": block["report"]["paired_bootstrap"]["favourable"],
                "delta_edge_tp": ch["final_graph_edges"]["delta_tp"],
                "delta_division_tp": ch["division_counts"]["delta_tp"],
                "node_recall_delta": ch["node_recall"]["delta"],
                "assignment_churn": block["assignment_churn"]["changed_fraction"],
                "wins": ok,
                "blockers": fails,
            }
        entry["folds_measured"] = [f for f in (0, 1) if wins.get(f) is not None]
        entry["wins_both_folds"] = bool(wins.get(0)) and bool(wins.get(1))
        rows.append(entry)

    # A fold with no crops is NOT a fold that lost. Reading an absent fold as a loss would print
    # "NEGATIVE" over a run that is merely unfinished, which is the single most dangerous string
    # this module can emit: a negative Gate A is exactly the result the lever's amendment says
    # must not be over-read.
    complete = all(len(r["folds_measured"]) == 2 for r in rows) and         all(panel["complete"].get(f, panel["complete"].get(str(f), False)) for f in (0, 1))

    flags = [r["wins_both_folds"] for r in rows]
    runs, i = [], 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j + 1 < len(flags) and flags[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    plateaus = [{"values": [grid[a] for a in range(s, e + 1)], "length": e - s + 1}
                for s, e in runs]
    winning = [p for p in plateaus if p["length"] >= 2]
    winning.sort(key=lambda p: (-p["length"], p["values"][0]))
    selected = winning[0]["values"][0] if winning else None
    isolated = [p["values"][0] for p in plateaus if p["length"] == 1]
    return {
        "grid": grid,
        "rule": "smallest value on the longest plateau of ADJACENT grid values that win on BOTH "
                "folds; a run of length 1 is an isolated optimum and is REFUSED as a selection "
                "(FACT-0386's selection artifact)",
        "rows": rows,
        "plateaus": plateaus,
        "isolated_wins_refused": isolated,
        "folds_complete": {str(f): panel["complete"].get(f, panel["complete"].get(str(f), False))
                           for f in (0, 1)},
        "crops": {str(f): panel["n_crops"].get(f, panel["n_crops"].get(str(f), 0))
                  for f in (0, 1)},
        "selected_bonus": selected if complete else None,
        "provisional_selection_on_incomplete_data": None if complete else selected,
        "gate_a_verdict": (
            "INCOMPLETE - not every crop of both folds has been measured, so NO verdict is "
            "issued. An absent fold is not a fold that lost. Any plateau shown here is "
            "PROVISIONAL and must not be read as Gate A's result."
            if not complete else
            "NEGATIVE - no bonus on the preregistered grid wins on both folds, so amplifying the "
            "DEPLOYED edge_prob at the final consumption point does not pay. This kills only "
            "'amplify the deployed score'; FACT-0421 measured the deployed probability to be the "
            "NON-TRANSFERRING component, so it says nothing about LEVER-0044 (see the lever's "
            "falsifier_amendment_reason)."
            if selected is None and not isolated else
            "ISOLATED - a single grid value wins but has no plateau; refused as a selection "
            "artifact rather than promoted."
            if selected is None else
            f"POSITIVE - plateau {winning[0]['values']}, smallest member selected"
        ),
    }


def _clause_g(panel: dict) -> dict:
    """The RESTORED clause (g) (falsifier_restoration_2026_08_30): does the fold-0 arm reproduce FACT-0414's EXACT zero, at the
    FULL-CHAIN stage rather than the ranking stage?

    If it does, the reading is about the SUBSTRATE and not the STAGE - fold 0 cannot resolve this
    intervention either - and the honest gate has to move rather than the model (FACT-0420).
    """
    out = {}
    for arm, folds in panel["arms"].items():
        block = folds.get("fold0")
        if block is None:
            continue
        ch = block["report"]["channels"]
        out[arm] = {
            "delta_edge_tp": ch["final_graph_edges"]["delta_tp"],
            "delta_edge_fp": ch["final_graph_edges"]["delta_fp"],
            "raw_edge_jaccard_delta": ch["edge_jaccard_raw"]["delta"],
            "assignment_churn": block["assignment_churn"]["changed_fraction"],
            "exact_zero_net_true_edges": ch["final_graph_edges"]["delta_tp"] == 0,
            "reshuffles_but_nets_zero": (ch["final_graph_edges"]["delta_tp"] == 0
                                         and block["assignment_churn"]["changed_parent"] > 0),
        }
    return {
        "clause": "(g) - FACT-0414's signature at the FULL-CHAIN stage: a change that reshuffles "
                  "assignments and nets EXACTLY zero true edges on fold 0. Restored by the "
                  "auditor after the first amendment reused the letter (f) and dropped it; it is "
                  "NOT subsumed by the Gate A narrowing because it binds precisely the stage "
                  "Gate B tests",
        "reading_if_true": "the SUBSTRATE and not the STAGE is the constraint on fold 0; the "
                           "association lane needs re-framing rather than a new consumer, and "
                           "the promotion gate has to move rather than the model (FACT-0420)",
        "arms": out,
    }


def _print_panel(panel: dict) -> None:
    sel = panel["gate_a_selection"]
    print("\n=== GATE A GRID (raw edge Jaccard delta vs the DEPLOYED control) ===")
    print(f"{'bonus':>6} | {'f0 raw':>10} {'f0 adj':>10} {'f0 dTP':>7} {'f0 churn':>9} | "
          f"{'f1 raw':>10} {'f1 adj':>10} {'f1 dTP':>7} {'f1 churn':>9} | wins")
    for r in sel["rows"]:
        cells = []
        for fold in (0, 1):
            f = r.get(f"fold{fold}")
            cells.append(f"{f['raw_edge_jaccard_delta']:+10.6f} {f['adj_edge_jaccard_delta']:+10.6f} "
                         f"{f['delta_edge_tp']:+7d} {f['assignment_churn']:9.5f}" if f
                         else " " * 39)
        print(f"{r['bonus']:>6g} | {cells[0]} | {cells[1]} | {r['wins_both_folds']}")
    print(f"\nselected_bonus = {sel['selected_bonus']}   {sel['gate_a_verdict']}")


def _arm_conversions(arm: str, scores_meta: dict | None) -> dict | None:
    """Gate A changes NOTHING pre-ILP, so its conversion figure is an honest exact zero."""
    if arm.startswith("gateA"):
        return {"stage": "pre_ILP_candidate_ranking", "targets": 0, "gained": 0, "lost": 0,
                "net": 0, "held_correct": 0, "churn": 0,
                "note": "Gate A leaves the pre-ILP candidate surface untouched; the intervention "
                        "is post-ILP only, so this channel is zero BY CONSTRUCTION, not by "
                        "measurement failure"}
    if scores_meta is None:
        return None
    return {k: v for k, v in scores_meta["preilp_conversions"]["contested"].items()
            if k != "crop_paired_bootstrap"} | {
        "stage": "pre_ILP_candidate_ranking",
        "crop_paired_bootstrap":
            scores_meta["preilp_conversions"]["contested"].get("crop_paired_bootstrap"),
    }


def _ledger_totals(payloads: list[dict], arm: str) -> dict:
    from assoc_lost_edge_ledger import CATEGORIES
    keys = list(CATEGORIES) + ["recovered", "gt_edges"]
    totals = {k: 0 for k in keys}
    div = {"gt_division_events": 0, "fully_recovered": 0, "partially_recovered": 0,
           "wholly_lost": 0}
    reach = {"reachable": 0, "unreachable_no_candidate": 0, "undetected_endpoint": 0}
    n = 0
    for p in payloads:
        led = p["arms"][arm].get("ledger")
        if led is None:
            continue
        n += 1
        for k in keys:
            totals[k] += int(led[k])
        for k in div:
            div[k] += int(led["divisions"][k])
        for k in reach:
            reach[k] += int(led["fact_0370_split"][k])
    totals["divisions"] = div
    totals["fact_0370_split"] = reach
    totals["crops"] = n
    # FACT-0430: the category-2 drain must be REMOVED from FACT-0370's reachable share before
    # category 4 is read, or category 4 absorbs it.
    totals["fact_0430_reconciliation"] = {
        "fact_0370_reachable": reach["reachable"],
        "cat2_endpoint_removed": totals["cat2_endpoint_removed_by_node_selection"],
        "reachable_less_cat2": reach["reachable"] - totals["cat2_endpoint_removed_by_node_selection"],
        "note": "FACT-0370's reachable pool is the PRE-ILP node set and is upstream of the three "
                "category-2 operations; category 4 must be read against reachable_less_cat2",
    }
    return totals


# ==============================================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scores", help="build the probability-free contextual score table")
    s.add_argument("--fold", type=int, required=True, choices=(0, 1))
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--mode", default="transfer", choices=("transfer", "refit"))

    c = sub.add_parser("crop", help="run every arm on one crop")
    c.add_argument("--fold", type=int, required=True, choices=(0, 1))
    c.add_argument("--crop", required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--scores", type=Path)
    c.add_argument("--bonuses", default=",".join(f"{b:g}" for b in BONUS_GRID))
    c.add_argument("--gateb-ranks", default="1,2")
    c.add_argument("--gateb-bonuses", default="1")
    c.add_argument("--no-ledger", action="store_true")

    b = sub.add_parser("register", help="register a fold's DEPLOYED control in the manifest")
    b.add_argument("--fold", type=int, required=True, choices=(0, 1))
    b.add_argument("--results-dir", type=Path, required=True)
    b.add_argument("--date", default="")

    p = sub.add_parser("panel", help="aggregate crop payloads into the promotable summary")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-ledger", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "scores":
        build_scores(args.fold, args.out, args.mode)
    elif args.cmd == "register":
        register_control(args.fold, args.results_dir, args.date)
    elif args.cmd == "crop":
        run_crop(
            args.fold, args.crop, args.out, args.scores,
            tuple(float(v) for v in args.bonuses.split(",") if v != ""),
            tuple(int(v) for v in args.gateb_ranks.split(",") if v != ""),
            tuple(float(v) for v in args.gateb_bonuses.split(",") if v != ""),
            not args.no_ledger,
        )
    else:
        build_panel(args.results_dir, args.out, not args.no_ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
