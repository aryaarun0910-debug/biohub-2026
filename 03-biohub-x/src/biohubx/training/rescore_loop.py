"""Train a candidate re-scorer on one embryo and score its kept candidates on the other: the E07 loop.

One function runs the whole thing so the CPU smoke, the Kaggle kernel and the
mini-fold share every line: window by window, the candidate pool, its labels
and its patches are built once and cached by typed digest so every arm sees
identical inputs; the scorer trains with the non-negative positive-unlabelled
risk and each movie's own class prior; the count-tied threshold is frozen on
the training movies; the held-out movies are scored through the oracle ceiling
beside A0 on the same windows. Nothing here reads the held-out embryo before
the threshold is frozen, and nothing tunes anything against it after.

Consumers: ``biohubx train rescore`` and the kernel entry
``biohubx.packaging.entry.run_rescore``.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from biohubx.contracts.instances import InstanceSet
from biohubx.data.competition import WindowSelection, load_ground_truth, load_window
from biohubx.evaluation.official_metric import EstimatedTotalNodes, metric_row, summarise_fold
from biohubx.evaluation.oracle import oracle_graph
from biohubx.proposals import dog, rescore
from biohubx.training.targets import positive_unlabelled_loss

Log = Callable[[str, str], None]


def _quiet(_stage: str, _detail: str) -> None:
    return None


@dataclass(frozen=True)
class LoopSpec:
    arm: str
    train_embryo: str
    evaluate_embryo: str
    train_movies: int
    evaluate_movies: int
    frames: int
    epochs: int
    batch: int
    learning_rate: float
    seed: int
    radii_um: tuple[float, ...]
    suppression_radius_um: float
    pool_quantile: float
    baseline_quantile: float
    config_digest: str
    count_ratio: float = 1.0
    train_datasets: tuple[str, ...] = ()
    """Explicit training movies, chosen at package time and shipped; empty means the first N in name order.

    Name order drew the sparsest tail of 44b6 for Stage 2 ([[R-0026]]). A
    selection made locally from the training embryo's own annotation counts
    travels with the package as a list, so the run reads what was chosen and
    the choice can be recomputed from the counts recorded beside it.
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "train_datasets": list(self.train_datasets),
            "train_embryo": self.train_embryo,
            "evaluate_embryo": self.evaluate_embryo,
            "train_movies": self.train_movies,
            "evaluate_movies": self.evaluate_movies,
            "frames": self.frames,
            "epochs": self.epochs,
            "batch": self.batch,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "radii_um": list(self.radii_um),
            "suppression_radius_um": self.suppression_radius_um,
            "pool_quantile": self.pool_quantile,
            "baseline_quantile": self.baseline_quantile,
            "config_digest": self.config_digest,
            "count_ratio": self.count_ratio,
        }


@dataclass
class MovieInputs:
    """Everything one movie's window contributes, built once and cached."""

    dataset: str
    first_frame: int
    estimate: float
    annotated_nodes: int
    pool_size: int
    baseline_size: int
    positives: int
    patches: torch.Tensor
    labels: torch.Tensor
    pool: InstanceSet
    baseline: InstanceSet
    annotated: Any
    cache_key: str


def movies_of(data_root: Path, embryo: str, count: int) -> list[str]:
    """The first ``count`` annotated movies of an embryo in sorted order: a fixed subset, never a choice."""
    ids = sorted(path.stem for path in (data_root / "train").glob(f"{embryo}*.geff"))
    return ids[:count] if count else ids


def annotated_node_counts(data_root: Path, embryo: str) -> dict[str, int]:
    """Annotated nodes per movie of one embryo, read from each geff's metadata and nodes."""
    counts: dict[str, int] = {}
    for dataset in movies_of(data_root, embryo, 0):
        counts[dataset] = len(load_ground_truth(data_root, dataset).lineage.nodes)
    return counts


STRATA = 4
"""Annotation-count quartiles. Four, so a selection of eight reads two per band."""

STRATUM_ORDER = (2, 1, 3, 0)
"""Which bands receive picks first when the count does not divide by four.

Median-first, then outward: the second and third quartiles before the densest
and the sparsest. E07 Stage 2 read the two sparsest-but-one movies of 44b6 by
taking the first two in name order and trained on 13 positives ([[R-0026]]);
Arya Arun's instruction of 2026-09-04 replaces name order with stratification
that neither repeats that nor selects only the densest.
"""


def stratified_movies(counts: dict[str, int], count: int) -> list[str]:
    """``count`` movies across annotation-count quartiles; deterministic; training-embryo labels only.

    Movies are ranked by annotated node count with the id as the tie-break, cut
    into four contiguous rank bands, and picks are taken band by band in
    ``STRATUM_ORDER``, each band yielding its median-rank movie first and then
    alternating outward. The same counts always give the same list, and nothing
    about the held-out embryo enters. The counts are what the selection used
    and travel with it, so a reader can recompute the choice.
    """
    if count <= 0:
        return []
    ranked = sorted(counts, key=lambda dataset: (counts[dataset], dataset))
    if count >= len(ranked):
        return ranked
    bands: list[list[str]] = [[] for _ in range(STRATA)]
    for index, dataset in enumerate(ranked):
        bands[min(STRATA - 1, index * STRATA // len(ranked))].append(dataset)

    def median_outward(band: list[str]) -> list[str]:
        order: list[str] = []
        middle = len(band) // 2
        for step in range(len(band)):
            offset = (step + 1) // 2 * (1 if step % 2 else -1)
            position = middle + offset
            if 0 <= position < len(band) and band[position] not in order:
                order.append(band[position])
        for dataset in band:
            if dataset not in order:
                order.append(dataset)
        return order

    queues = [median_outward(band) for band in bands]
    chosen: list[str] = []
    while len(chosen) < count:
        progressed = False
        for stratum in STRATUM_ORDER:
            if queues[stratum]:
                chosen.append(queues[stratum].pop(0))
                progressed = True
                if len(chosen) == count:
                    break
        if not progressed:
            break
    return chosen


def annotated_in_window(data_root: Path, dataset: str, frames: int) -> tuple[int, int]:
    """First frame the loop would read and the annotated nodes inside that window.

    The same window rule ``build_inputs`` applies, so what the preflight counts is
    what the run will see. Cheap: the geff only, no volume and no detector.
    """
    truth = load_ground_truth(data_root, dataset)
    annotated_frames = sorted({node.frame for node in truth.lineage.nodes})
    first = min(annotated_frames[0], max(0, truth.frames - frames)) if annotated_frames else 0
    inside = sum(1 for node in truth.lineage.nodes if first <= node.frame < first + frames)
    return first, inside


def cache_key(spec: LoopSpec, dataset: str, first_frame: int) -> str:
    payload = json.dumps(
        {
            "dataset": dataset,
            "first_frame": first_frame,
            "frames": spec.frames,
            "arm": spec.arm,
            "radii_um": list(spec.radii_um),
            "suppression_radius_um": spec.suppression_radius_um,
            "pool_quantile": spec.pool_quantile,
            "baseline_quantile": spec.baseline_quantile,
            "patch_half": rescore.PATCH_HALF,
            "standardise": True,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_inputs(
    data_root: Path, spec: LoopSpec, dataset: str, *, cache_dir: Path | None, log: Log = _quiet
) -> MovieInputs:
    truth = load_ground_truth(data_root, dataset)
    annotated_frames = sorted({node.frame for node in truth.lineage.nodes})
    first = min(annotated_frames[0], max(0, truth.frames - spec.frames)) if annotated_frames else 0
    key = cache_key(spec, dataset, first)
    cached = cache_dir / f"{key}.pt" if cache_dir is not None else None
    import zarr

    group: Any = zarr.open(str(data_root / "train" / f"{dataset}.zarr"), mode="r")
    depth, height, width = group["0"].shape[1:]
    window = load_window(
        data_root, WindowSelection(dataset, first, spec.frames, 0, depth, 0, height, 0, width), split="train"
    )
    baseline = dog.detect_instances(
        window.volume,
        dataset=window.annotated.dataset,
        radii_um=spec.radii_um,
        response_quantile=spec.baseline_quantile,
        suppression_radius_um=spec.suppression_radius_um,
        local_maxima_only=True,
        per_scale_union=True,
    )
    pool = rescore.extract_candidates(
        window.volume,
        dataset=window.annotated.dataset,
        radii_um=spec.radii_um,
        suppression_radius_um=spec.suppression_radius_um,
        response_quantile=spec.pool_quantile,
    )
    labels = rescore.candidate_labels(pool, window.annotated)
    if cached is not None and cached.is_file():
        patches = torch.load(cached, map_location="cpu", weights_only=True)
        if tuple(patches.shape[:1]) != (len(pool.instances),):
            patches = rescore.candidate_patches(window.volume, pool, arm=spec.arm, radii_um=spec.radii_um)
            torch.save(patches, cached)
        log("cache", f"{dataset} patches from {cached.name}")
    else:
        patches = rescore.candidate_patches(window.volume, pool, arm=spec.arm, radii_um=spec.radii_um)
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            staged = cached.with_suffix(".pt.partial")
            torch.save(patches, staged)
            staged.replace(cached)
    return MovieInputs(
        dataset=dataset,
        first_frame=first,
        estimate=float(window.window_estimated_total_nodes),
        annotated_nodes=len(window.annotated.nodes),
        pool_size=len(pool.instances),
        baseline_size=len(baseline.instances),
        positives=int(labels.sum()),
        patches=patches,
        labels=labels,
        pool=pool,
        baseline=baseline,
        annotated=window.annotated,
        cache_key=key,
    )


def ceiling(instances: InstanceSet, annotated: Any, estimate: float) -> dict[str, Any]:
    graph = oracle_graph(instances, annotated)
    row: dict[str, Any] = {
        "proposals": graph.proposals,
        "estimate": estimate,
        "node_ratio": (graph.proposals - estimate) / estimate if estimate else None,
        "matched_nodes": graph.matched_nodes,
        "annotated_nodes": graph.annotated_nodes,
        "match_fraction": graph.matched_nodes / graph.annotated_nodes if graph.annotated_nodes else None,
        "retained_edges": graph.retained_edges,
        "annotated_edges": graph.annotated_edges,
        "metric_row": None,
    }
    if graph.graph is not None:
        row["metric_row"] = metric_row(
            graph.graph, annotated, estimated_total_nodes=EstimatedTotalNodes.declared(estimate)
        )
    return row


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-direction totals and the official aggregate over the scored windows."""
    scored = [r["metric_row"] for r in rows if r["metric_row"] is not None]
    proposals = sum(int(r["proposals"]) for r in rows)
    estimate = sum(float(r["estimate"]) for r in rows)
    annotated = sum(int(r["annotated_nodes"]) for r in rows)
    matched = sum(int(r["matched_nodes"]) for r in rows)
    edges = sum(int(r["annotated_edges"]) for r in rows)
    retained = sum(int(r["retained_edges"]) for r in rows)
    out: dict[str, Any] = {
        "windows": len(rows),
        "windows_scored": len(scored),
        "proposals": proposals,
        "estimated_nodes": round(estimate, 1),
        "node_ratio": round((proposals - estimate) / estimate, 4) if estimate else None,
        "node_match_fraction": round(matched / annotated, 4) if annotated else None,
        "edge_retention": round(retained / edges, 4) if edges else None,
    }
    if scored:
        out.update(summarise_fold(scored).to_dict())
    return out


@dataclass
class LoopResult:
    spec: LoopSpec
    train_datasets: list[str]
    evaluate_datasets: list[str]
    epochs: list[dict[str, Any]] = field(default_factory=list)
    threshold: float = 1.0
    training_kept: int = 0
    training_estimate: float = 0.0
    held_out: dict[str, Any] = field(default_factory=dict)
    baseline_held_out: dict[str, Any] = field(default_factory=dict)
    pool_held_out: dict[str, Any] = field(default_factory=dict)
    per_movie: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_digest: str | None = None
    reload_identical: bool | None = None
    runtime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "train_datasets": self.train_datasets,
            "evaluate_datasets": self.evaluate_datasets,
            "epochs": self.epochs,
            "threshold": self.threshold,
            "training_kept": self.training_kept,
            "training_estimate": self.training_estimate,
            "held_out": {"arm": self.held_out, "A0": self.baseline_held_out, "pool": self.pool_held_out},
            "per_movie": self.per_movie,
            "checkpoint_digest": self.checkpoint_digest,
            "reload_identical": self.reload_identical,
            "runtime_seconds": self.runtime_seconds,
        }


def _forward(model: torch.nn.Module, patches: torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
    pieces = []
    for part in rescore.batches(patches, batch):
        pieces.append(model(part.to(device)))
    return torch.cat(pieces) if pieces else torch.zeros(0, device=device)


def run_loop(
    data_root: Path,
    spec: LoopSpec,
    *,
    device: torch.device,
    cache_dir: Path | None,
    checkpoint_path: Path | None,
    log: Log = _quiet,
) -> LoopResult:
    started = time.monotonic()
    torch.manual_seed(spec.seed)
    generator = torch.Generator().manual_seed(spec.seed)
    train_ids = list(spec.train_datasets) or movies_of(data_root, spec.train_embryo, spec.train_movies)
    foreign = [d for d in train_ids if d.split("_", 1)[0] != spec.train_embryo]
    if foreign:
        raise ValueError(f"shipped training movies are not all from {spec.train_embryo}: {foreign}")
    eval_ids = movies_of(data_root, spec.evaluate_embryo, spec.evaluate_movies)
    result = LoopResult(spec=spec, train_datasets=train_ids, evaluate_datasets=eval_ids)

    train_inputs = [
        build_inputs(data_root, spec, dataset, cache_dir=cache_dir, log=log) for dataset in train_ids
    ]
    log(
        "inputs",
        f"train movies={len(train_inputs)} pool={sum(m.pool_size for m in train_inputs)} "
        f"positives={sum(m.positives for m in train_inputs)} "
        f"baseline={sum(m.baseline_size for m in train_inputs)}",
    )

    model = rescore.build_scorer(spec.arm, seed=spec.seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)
    for epoch in range(spec.epochs):
        model.train()
        losses: list[float] = []
        clamped = 0
        batches_run = 0
        order = torch.randperm(len(train_inputs), generator=generator).tolist()
        for index in order:
            movie = train_inputs[index]
            prior = rescore.candidate_prior(movie.estimate, movie.pool_size)
            permutation = torch.randperm(movie.pool_size, generator=generator)
            for start in range(0, movie.pool_size, spec.batch):
                chosen = permutation[start : start + spec.batch]
                labels = movie.labels[chosen].to(device)
                if int(labels.sum()) == 0 or int((~labels).sum()) == 0:
                    continue
                logits = model(movie.patches[chosen].to(device))
                loss, terms = positive_unlabelled_loss(logits, labels, prior=prior)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
                losses.append(float(loss.item()))
                clamped += int(bool(terms["negative_risk_clamped"]))
                batches_run += 1
        record = {
            "epoch": epoch,
            "batches": batches_run,
            "loss_mean": float(np.mean(losses)) if losses else None,
            "loss_last": losses[-1] if losses else None,
            "negative_risk_clamped_batches": clamped,
        }
        result.epochs.append(record)
        log("epoch", json.dumps(record))
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            staged = checkpoint_path.with_suffix(".pt.partial")
            torch.save(model.state_dict(), staged)
            staged.replace(checkpoint_path)

    model.eval()
    with torch.no_grad():
        train_logits = torch.cat([_forward(model, m.patches, spec.batch, device) for m in train_inputs])
    result.training_estimate = sum(m.estimate for m in train_inputs)
    # The decode keeps, per movie, the top round(count_ratio * estimate) candidates by
    # score. The estimate is official metadata the node-ratio term itself uses, never
    # ground truth, so the same rule applies on the held-out embryo without anything
    # being selected there. The probability at the training cut is recorded, not used.
    result.training_kept = sum(min(m.pool_size, round(spec.count_ratio * m.estimate)) for m in train_inputs)
    result.threshold = rescore.threshold_for_count(
        train_logits, round(spec.count_ratio * result.training_estimate)
    )
    log(
        "decode",
        f"count_ratio={spec.count_ratio} keeps {result.training_kept} "
        f"of {train_logits.numel()} training candidates",
    )

    if checkpoint_path is not None and checkpoint_path.is_file():
        result.checkpoint_digest = (
            "raw_artifact_sha256:sha256:" + hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        )
        rebuilt = rescore.build_scorer(spec.arm, seed=spec.seed).to(device)
        rebuilt.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True), strict=True
        )
        rebuilt.eval()
        with torch.no_grad():
            again = torch.cat([_forward(rebuilt, m.patches, spec.batch, device) for m in train_inputs])
        result.reload_identical = bool(torch.equal(train_logits, again))
        model = rebuilt

    arm_rows: list[dict[str, Any]] = []
    a0_rows: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    for dataset in eval_ids:
        movie = build_inputs(data_root, spec, dataset, cache_dir=cache_dir, log=log)
        with torch.no_grad():
            logits = _forward(model, movie.patches, spec.batch, device).cpu()
        kept = rescore.keep_top(movie.pool, logits, round(spec.count_ratio * movie.estimate))
        arm_row = ceiling(kept, movie.annotated, movie.estimate)
        a0_row = ceiling(movie.baseline, movie.annotated, movie.estimate)
        pool_row = ceiling(movie.pool, movie.annotated, movie.estimate)
        arm_rows.append(arm_row)
        a0_rows.append(a0_row)
        pool_rows.append(pool_row)
        result.per_movie.append(
            {
                "dataset": dataset,
                "first_frame": movie.first_frame,
                "estimate": movie.estimate,
                "pool": movie.pool_size,
                "A0": a0_row["proposals"],
                "kept": arm_row["proposals"],
                "match_arm": arm_row["match_fraction"],
                "match_A0": a0_row["match_fraction"],
            }
        )
    result.held_out = aggregate(arm_rows)
    result.baseline_held_out = aggregate(a0_rows)
    result.pool_held_out = aggregate(pool_rows)
    result.runtime_seconds = round(time.monotonic() - started, 3)
    log(
        "held-out",
        json.dumps(
            {
                k: {
                    "n": v.get("proposals"),
                    "ratio": v.get("node_ratio"),
                    "match": v.get("node_match_fraction"),
                    "score": v.get("score"),
                }
                for k, v in (
                    ("arm", result.held_out),
                    ("A0", result.baseline_held_out),
                    ("pool", result.pool_held_out),
                )
            }
        ),
    )
    return result
