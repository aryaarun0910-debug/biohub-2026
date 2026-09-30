"""Is "annotated" predictable from anything other than being a cell?

E07 trains a scorer to rank candidates matched to annotated cells above the
rest, under a non-negative positive-unlabelled loss that assumes the labelled
positives are a uniform random sample of all positives ([[RL-0054]]). This
corpus is not that. [[F-0013]] records 44b6 annotated at 0.77 percent against
6bba's 9.71, and the movies E07 Stage 2 trained on carry one traced lineage each
([[R-0026]]). If the annotators' choice of which cells to follow is itself
predictable from where and when a candidate sits, how bright it is, how crowded
its neighbourhood is, or whether it persists between frames, then a scorer can
raise its objective by learning that choice instead of learning cellness, and
a held-out embryo with a different annotation regime would show it as a ranker
that transfers badly for a reason no oracle ceiling can see.

So the question is asked directly, on the training embryo, before any fold is
spent. Every candidate in the pool gets a small set of features that say
nothing about whether it is a cell beyond the DoG response the pool already
ranks by. A logistic model predicts "annotated" from the response alone, and
from the response plus each feature group; the gain in held-out AUC is the
propensity signal. Held-out means grouped by movie, never by candidate, so a
movie's own labels never inform the model that scores it. The gain is judged
against a permutation control that shuffles every non-response feature within
each movie and repeats the whole procedure, so a gain that unrelated features
could produce by overfitting is not a gain.

Nothing here reads the held-out embryo of any fold, and nothing here can promote
anything. The output is a decision input the controller applies mechanically.

Consumer: ``biohubx evaluate propensity``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from biohubx.contracts.coordinates import OFFICIAL_VOXEL_SCALE, VoxelScaleZYX
from biohubx.contracts.instances import InstanceSet
from biohubx.proposals import dog

FEATURE_NAMES: tuple[str, ...] = (
    "dog",
    "intensity",
    "z",
    "y",
    "x",
    "face_um",
    "t",
    "density",
    "persist_prev",
    "persist_next",
    "motion_um",
)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "control": ("dog",),
    "appearance": ("intensity",),
    "position": ("z", "y", "x", "face_um"),
    "time": ("t",),
    "density": ("density",),
    "temporal": ("persist_prev", "persist_next", "motion_um"),
}
"""Groups reported separately. ``control`` is what the pool already ranks by."""

NON_TEMPORAL_GROUPS: tuple[str, ...] = ("appearance", "position", "time", "density")
"""What A3 can see. ``temporal`` is what A4 additionally sees."""


class PropensityError(ValueError):
    """The probe cannot answer as asked, and says why."""


@dataclass(frozen=True)
class ProbeParameters:
    """Every constant the probe uses, declared in the config and recorded in the report."""

    density_radius_um: float
    persistence_radius_um: float
    motion_cap_um: float
    l2: float
    permutations: int
    seed: int
    null_quantile: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "density_radius_um": self.density_radius_um,
            "persistence_radius_um": self.persistence_radius_um,
            "motion_cap_um": self.motion_cap_um,
            "l2": self.l2,
            "permutations": self.permutations,
            "seed": self.seed,
            "null_quantile": self.null_quantile,
        }


@dataclass
class CandidateTable:
    """One movie window's candidates as rows: features, labels and where they came from."""

    dataset: str
    first_frame: int
    frames: int
    features: np.ndarray
    labels: np.ndarray
    estimate: float
    annotated_nodes: int

    @property
    def positives(self) -> int:
        return int(self.labels.sum())


def _nearest(
    trees: dict[int, cKDTree], frame: int, point: np.ndarray, parameters: ProbeParameters
) -> tuple[float, float]:
    """Persistence flag and capped distance to the nearest candidate in another frame."""
    if frame not in trees:
        return 0.0, parameters.motion_cap_um
    distance, _ = trees[frame].query(point, k=1)
    return (
        1.0 if float(distance) <= parameters.persistence_radius_um else 0.0,
        float(min(float(distance), parameters.motion_cap_um)),
    )


def candidate_table(
    *,
    dataset: str,
    volume: np.ndarray,
    first_frame: int,
    pool: InstanceSet,
    labels: np.ndarray,
    estimate: float,
    annotated_nodes: int,
    parameters: ProbeParameters,
    scale: VoxelScaleZYX = OFFICIAL_VOXEL_SCALE,
) -> CandidateTable:
    """Features that describe a candidate's circumstances rather than its cellness.

    ``dog`` is the response the pool already carries and is the control. The
    rest are position in the volume and distance to its nearest face, position
    in the window in time, normalised intensity at the centre, how many other
    candidates sit within a radius in the same frame, whether a candidate sits
    within a radius in the previous and next frames, and the distance to the
    nearest candidate in the next frame. None of them is a patch and none of
    them is anything a scorer is meant to learn; if they predict "annotated"
    beyond the response, that is propensity.
    """
    instances = pool.instances
    if len(instances) != len(labels):
        raise PropensityError(f"{len(instances)} candidates and {len(labels)} labels")
    frames = int(volume.shape[0])
    depth, height, width = (int(v) for v in volume.shape[1:])
    extent = (depth * scale.z_um, height * scale.y_um, width * scale.x_um)
    stride = dog.isotropic_plane_stride(scale)

    intensity_grids: dict[int, np.ndarray] = {}

    def intensity_at(frame: int, z: int, y: int, x: int) -> float:
        if frame not in intensity_grids:
            intensity_grids[frame] = dog.normalise_frame(volume[frame][:, ::stride, ::stride]).astype(
                np.float32
            )
        grid = intensity_grids[frame]
        zz = min(max(z, 0), grid.shape[0] - 1)
        yy = min(max(y // stride, 0), grid.shape[1] - 1)
        xx = min(max(x // stride, 0), grid.shape[2] - 1)
        return float(grid[zz, yy, xx])

    by_frame: dict[int, list[int]] = {}
    for index, instance in enumerate(instances):
        by_frame.setdefault(int(instance.frame), []).append(index)
    points = np.array(
        [(i.physical.z_um, i.physical.y_um, i.physical.x_um) for i in instances], dtype=np.float64
    )
    trees = {frame: cKDTree(points[rows]) for frame, rows in by_frame.items()}

    rows = np.zeros((len(instances), len(FEATURE_NAMES)), dtype=np.float64)
    for index, instance in enumerate(instances):
        frame = int(instance.frame)
        point = points[index]
        tree = trees[frame]
        density = len(tree.query_ball_point(point, parameters.density_radius_um)) - 1

        persist_prev, _ = _nearest(trees, frame - 1, point, parameters)
        persist_next, motion = _nearest(trees, frame + 1, point, parameters)
        face = min(
            point[0],
            extent[0] - point[0],
            point[1],
            extent[1] - point[1],
            point[2],
            extent[2] - point[2],
        )
        local = frame - first_frame
        rows[index] = (
            float(instance.confidence),
            intensity_at(frame, round(instance.voxel.z), round(instance.voxel.y), round(instance.voxel.x)),
            point[0] / extent[0],
            point[1] / extent[1],
            point[2] / extent[2],
            float(face),
            (local / (frames - 1)) if frames > 1 else 0.0,
            float(density),
            persist_prev,
            persist_next,
            motion,
        )
    return CandidateTable(
        dataset=dataset,
        first_frame=first_frame,
        frames=frames,
        features=rows,
        labels=np.asarray(labels, dtype=bool),
        estimate=float(estimate),
        annotated_nodes=int(annotated_nodes),
    )


def columns_for(groups: Sequence[str]) -> list[int]:
    names: list[str] = []
    for group in groups:
        for name in FEATURE_GROUPS[group]:
            if name not in names:
                names.append(name)
    return [FEATURE_NAMES.index(name) for name in names]


def fit_logistic(features: np.ndarray, labels: np.ndarray, *, l2: float) -> np.ndarray:
    """Class-balanced, L2-regularised logistic regression by L-BFGS with an analytic gradient.

    Features are standardised by the caller. Balanced, because 13 positives
    against 32,000 unlabelled rows otherwise fit the intercept and nothing
    else; the balance changes the intercept and not the ranking the AUC reads.
    Returns the weight vector with the intercept last.
    """
    n, d = features.shape
    design = np.hstack([features, np.ones((n, 1))])
    y = labels.astype(np.float64)
    positives = max(1.0, float(y.sum()))
    negatives = max(1.0, float(n - y.sum()))
    weight = np.where(y > 0.5, negatives / positives, 1.0)
    weight /= weight.sum()
    penalty = np.ones(d + 1) * l2
    penalty[-1] = 0.0

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ w
        # Stable log-loss: log(1 + exp(-y*z)) with y in {-1, +1}.
        signed = np.where(y > 0.5, 1.0, -1.0)
        margin = -signed * logits
        loss = np.logaddexp(0.0, margin)
        value = float((weight * loss).sum() + 0.5 * (penalty * w * w).sum())
        sigma = 1.0 / (1.0 + np.exp(-margin))
        gradient = design.T @ (weight * (-signed) * sigma) + penalty * w
        return value, gradient

    result = minimize(objective, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return np.asarray(result.x, dtype=np.float64)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC by ranks, ties averaged. Undefined without both classes."""
    positives = labels.astype(bool)
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        raise PropensityError("AUC needs at least one positive and one unlabelled row")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def held_out_auc(tables: Sequence[CandidateTable], columns: Sequence[int], *, l2: float) -> dict[str, Any]:
    """Leave-one-movie-out: fit on every other movie, score the held-out one, pool the scores.

    Standardisation statistics come from the training movies alone. A held-out
    movie with no positive contributes rows to the pooled AUC and no AUC of its
    own, which is recorded rather than skipped silently.
    """
    if len(tables) < 2:
        raise PropensityError("grouped held-out evaluation needs at least two movies")
    pooled_scores: list[np.ndarray] = []
    pooled_labels: list[np.ndarray] = []
    per_movie: dict[str, float | None] = {}
    cols = list(columns)
    for held in range(len(tables)):
        train = [t for i, t in enumerate(tables) if i != held]
        x_train = np.vstack([t.features[:, cols] for t in train])
        y_train = np.concatenate([t.labels for t in train])
        if y_train.sum() == 0:
            raise PropensityError(f"no positive outside {tables[held].dataset}; cannot fit a held-out model")
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0) + 1e-9
        weights = fit_logistic((x_train - mean) / std, y_train, l2=l2)
        x_test = (tables[held].features[:, cols] - mean) / std
        scores = np.hstack([x_test, np.ones((len(x_test), 1))]) @ weights
        pooled_scores.append(scores)
        pooled_labels.append(tables[held].labels)
        per_movie[tables[held].dataset] = (
            auc(scores, tables[held].labels) if tables[held].positives > 0 else None
        )
    scores_all = np.concatenate(pooled_scores)
    labels_all = np.concatenate(pooled_labels)
    return {"pooled": auc(scores_all, labels_all), "per_movie": per_movie}


def _permuted(tables: Sequence[CandidateTable], rng: np.random.Generator) -> list[CandidateTable]:
    """Every non-response feature row-shuffled within each movie; labels and the response untouched.

    The null has to answer the question the probe asks, which is whether the
    circumstances add anything beyond the response. Shuffling the labels would
    break the response too and compare a gain over a strong control with a gain
    over a random one. Shuffling the other features jointly, per movie, keeps
    the control model identical under the null and breaks only what is being
    tested, and keeps those features' correlations with each other.
    """
    control = columns_for(["control"])
    others = [index for index in range(len(FEATURE_NAMES)) if index not in control]
    out = []
    for table in tables:
        features = table.features.copy()
        order = rng.permutation(len(features))
        features[:, others] = features[order][:, others]
        out.append(
            CandidateTable(
                dataset=table.dataset,
                first_frame=table.first_frame,
                frames=table.frames,
                features=features,
                labels=table.labels,
                estimate=table.estimate,
                annotated_nodes=table.annotated_nodes,
            )
        )
    return out


def probe(tables: Sequence[CandidateTable], parameters: ProbeParameters) -> dict[str, Any]:
    """The whole measurement for one embryo: gains per group, a permutation null, a decision.

    Three gains are tested against their own nulls: every group together, the
    temporal group alone, and the non-temporal groups together. The first
    answers whether propensity is present; the other two answer which arm could
    see it. A gain counts only if it exceeds the declared quantile of the gain the
    same features produce, shuffled within movie, under the identical procedure.
    """
    if sum(t.positives for t in tables) == 0:
        raise PropensityError("no positives in any window; nothing to predict")
    l2 = parameters.l2
    control = held_out_auc(tables, columns_for(["control"]), l2=l2)
    groups: dict[str, dict[str, Any]] = {}
    for name in FEATURE_GROUPS:
        if name == "control":
            continue
        groups[name] = held_out_auc(tables, columns_for(["control", name]), l2=l2)
    tested = {
        "all": columns_for(["control", *[g for g in FEATURE_GROUPS if g != "control"]]),
        "temporal": columns_for(["control", "temporal"]),
        "non_temporal": columns_for(["control", *NON_TEMPORAL_GROUPS]),
    }
    observed = {name: held_out_auc(tables, cols, l2=l2) for name, cols in tested.items()}
    deltas = {name: observed[name]["pooled"] - control["pooled"] for name in tested}

    rng = np.random.default_rng(parameters.seed)
    null: dict[str, list[float]] = {name: [] for name in tested}
    for _ in range(parameters.permutations):
        shuffled = _permuted(tables, rng)
        base = held_out_auc(shuffled, columns_for(["control"]), l2=l2)["pooled"]
        for name, cols in tested.items():
            null[name].append(held_out_auc(shuffled, cols, l2=l2)["pooled"] - base)
    null_summary = {}
    for name, values in null.items():
        array = np.asarray(values, dtype=np.float64)
        threshold = float(np.quantile(array, parameters.null_quantile)) if len(array) else float("nan")
        null_summary[name] = {
            "count": len(array),
            "mean": float(array.mean()) if len(array) else None,
            "max": float(array.max()) if len(array) else None,
            "threshold": threshold,
            "exceeding_observed": int((array >= deltas[name]).sum()) if len(array) else None,
            "signal": bool(len(array)) and deltas[name] > threshold,
        }

    decision = decide(
        signal=bool(null_summary["all"]["signal"]),
        temporal=bool(null_summary["temporal"]["signal"]),
        non_temporal=bool(null_summary["non_temporal"]["signal"]),
    )
    return {
        "movies": [
            {
                "dataset": t.dataset,
                "first_frame": t.first_frame,
                "frames": t.frames,
                "candidates": len(t.labels),
                "positives": t.positives,
                "annotated_nodes": t.annotated_nodes,
                "estimate": t.estimate,
            }
            for t in tables
        ],
        "candidates_total": int(sum(len(t.labels) for t in tables)),
        "positives_total": int(sum(t.positives for t in tables)),
        "feature_names": list(FEATURE_NAMES),
        "control_auc": control,
        "group_auc": groups,
        "tested_auc": observed,
        "delta_auc": deltas,
        "null": null_summary,
        "parameters": parameters.to_dict(),
        "decision": decision,
    }


def decide(*, signal: bool, temporal: bool, non_temporal: bool) -> dict[str, Any]:
    """One embryo's reading, which by itself never kills the objective.

    Arya Arun's amended rule of 2026-09-04: a statistically present circumstance
    signal is weak, embryo-specific evidence unless it replicates across both
    embryos or demonstrably degrades a trained scorer's fixed-count held-out
    ceiling. Replication and degradation are judged by the controller over both
    embryos' readings and the Stage 2B results; this function only says what
    this embryo saw and which arm could see it.
    """
    if not signal:
        return {
            "verdict": "propensity_signal_absent",
            "next_action": "no signal on this embryo; continue",
            "A3": "eligible",
            "A4": "eligible on this embryo's evidence",
        }
    return {
        "verdict": "propensity_signal_present",
        "next_action": (
            "weak, embryo-specific evidence on its own; the objective is killed only if this "
            "replicates on the other embryo or the trained scorer's fixed-count held-out ceiling degrades"
        ),
        "A3": "eligible; the residual-propensity test on its trained scorer is mandatory",
        "A4": "hold; neighbouring frames can learn tracing propensity"
        if temporal
        else "hold pending the residual-propensity test on A3",
    }
