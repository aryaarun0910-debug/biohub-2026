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
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp.grad_scaler import GradScaler

from biohubx.contracts.instances import InstanceSet
from biohubx.data.competition import WindowSelection, load_ground_truth, load_window
from biohubx.evaluation import propensity
from biohubx.evaluation.official_metric import EstimatedTotalNodes, metric_row, summarise_fold
from biohubx.evaluation.oracle import oracle_graph
from biohubx.proposals import dog, rescore
from biohubx.training import runtime
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
    amp: bool = False
    batch_autotune: bool = False
    memory_ceiling_bytes: int = runtime.MEMORY_CEILING_BYTES
    sampling: str = "uniform"
    """``uniform`` (E07) or ``propensity_matched`` (E08): how a batch's unlabelled side is drawn."""
    residual_test: bool = False
    """Rerun E07-PROPENSITY-01 with the trained scorer's logit as control; mandatory for advancement."""
    worker_id: str = ""
    propensity_parameters: dict[str, float] | None = None
    """The probe's declared constants, shipped with the package, for the sampler and the residual test."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "train_datasets": list(self.train_datasets),
            "amp": self.amp,
            "batch_autotune": self.batch_autotune,
            "memory_ceiling_bytes": self.memory_ceiling_bytes,
            "sampling": self.sampling,
            "residual_test": self.residual_test,
            "worker_id": self.worker_id,
            "propensity_parameters": self.propensity_parameters,
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
    circumstances: Any = None
    """Per-candidate circumstance features (propensity.FEATURE_NAMES order) when the spec needs them."""


def _probe_parameters(spec: LoopSpec) -> propensity.ProbeParameters:
    if spec.propensity_parameters is None:
        raise ValueError("the spec carries no propensity parameters; the sampler and residual test need them")
    q = spec.propensity_parameters
    return propensity.ProbeParameters(
        density_radius_um=float(q["density_radius_um"]),
        persistence_radius_um=float(q["persistence_radius_um"]),
        motion_cap_um=float(q["motion_cap_um"]),
        l2=float(q["l2"]),
        permutations=int(q["permutations"]),
        seed=spec.seed,
        null_quantile=float(q["null_quantile"]),
    )


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


def staged_path(cached: Path) -> Path:
    """A partial-file name carrying the writer's pid, so two writers never race on one name.

    The rename onto the final name stays atomic; what changes is that a second
    process staging the same key cannot remove the first's partial file out
    from under it ([[R-0030]]).
    """
    return cached.with_name(f"{cached.name}.{os.getpid()}.partial")


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
    circumstances = None
    if spec.sampling == "propensity_matched" or spec.residual_test:
        circumstances = propensity.candidate_table(
            dataset=dataset,
            volume=window.volume,
            first_frame=first,
            pool=pool,
            labels=labels.numpy(),
            estimate=float(window.window_estimated_total_nodes),
            annotated_nodes=len(window.annotated.nodes),
            parameters=_probe_parameters(spec),
        ).features
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
            staged = staged_path(cached)
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
        circumstances=circumstances,
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
    amp: dict[str, Any] | None = None
    batch_ladder: dict[str, Any] | None = None
    batch_used: int = 0
    sampling: dict[str, Any] | None = None
    memory_ceiling: dict[str, Any] | None = None
    throughput: dict[str, Any] | None = None
    residual_propensity: dict[str, Any] | None = None
    ceiling_degraded: bool | None = None

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
            "amp": self.amp,
            "batch_ladder": self.batch_ladder,
            "batch_used": self.batch_used,
            "sampling": self.sampling,
            "memory_ceiling": self.memory_ceiling,
            "throughput": self.throughput,
            "residual_propensity": self.residual_propensity,
            "ceiling_degraded": self.ceiling_degraded,
        }


def _forward(model: torch.nn.Module, patches: torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
    pieces = []
    for part in rescore.batches(patches, batch):
        pieces.append(model(part.to(device)))
    return torch.cat(pieces) if pieces else torch.zeros(0, device=device)


def residual_propensity_test(
    train_inputs: Sequence[MovieInputs], train_logits: torch.Tensor, spec: LoopSpec
) -> dict[str, Any]:
    """E07-PROPENSITY-01 rerun with the trained scorer's logit as the control.

    Passes when the circumstance features add no more to the scorer's own
    held-out ranking than they add to the DoG response's, or no more than the
    permutation null, whichever is larger. A scorer that carried the input's
    weak propensity forward unchanged passes; one that learned to amplify it
    fails. Needs at least two training movies, and says so rather than
    imputing when it has one.
    """
    if any(m.circumstances is None for m in train_inputs):
        raise ValueError("the residual-propensity test needs circumstance features on every training movie")
    if len(train_inputs) < 2:
        return {"skipped": "needs at least two training movies", "passed": None}
    parameters = _probe_parameters(spec)
    column = propensity.FEATURE_NAMES.index("dog")
    logits = train_logits.detach().float().cpu().numpy()
    offset = 0
    scorer_tables: list[propensity.CandidateTable] = []
    dog_tables: list[propensity.CandidateTable] = []
    for movie in train_inputs:
        n = movie.pool_size
        features = np.asarray(movie.circumstances, dtype=np.float64).copy()
        with_scorer = features.copy()
        with_scorer[:, column] = logits[offset : offset + n]
        offset += n
        for table, rows in ((scorer_tables, with_scorer), (dog_tables, features)):
            table.append(
                propensity.CandidateTable(
                    dataset=movie.dataset,
                    first_frame=movie.first_frame,
                    frames=spec.frames,
                    features=rows,
                    labels=movie.labels.numpy(),
                    estimate=movie.estimate,
                    annotated_nodes=movie.annotated_nodes,
                )
            )
    scorer = propensity.probe(scorer_tables, parameters)
    baseline = propensity.probe(dog_tables, parameters)
    scorer_gain = float(scorer["delta_auc"]["all"])
    dog_gain = float(baseline["delta_auc"]["all"])
    threshold = float(scorer["null"]["all"]["threshold"])
    control = float(scorer["control_auc"]["pooled"])
    return {
        "passed": scorer_gain <= max(threshold, dog_gain),
        "scorer_control_auc": round(control, 4),
        "scorer_gain": round(scorer_gain, 4),
        "scorer_null_threshold": round(threshold, 4),
        "scorer_group_gains": {
            g: round(float(v["pooled"]) - control, 4) for g, v in scorer["group_auc"].items()
        },
        "dog_control_auc": round(float(baseline["control_auc"]["pooled"]), 4),
        "dog_gain": round(dog_gain, 4),
        "permutations": parameters.permutations,
        "rule": "pass if scorer_gain <= max(null threshold, dog_gain)",
    }


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
    result.memory_ceiling = runtime.memory_ceiling(device, spec.memory_ceiling_bytes)
    total_pool = sum(m.pool_size for m in train_inputs)
    total_positives = sum(m.positives for m in train_inputs)
    positive_rate = total_positives / total_pool if total_pool else 0.0
    first_movie = train_inputs[0]
    first_prior = rescore.candidate_prior(first_movie.estimate, first_movie.pool_size)
    probe_rows = min(first_movie.pool_size, 2048)

    # Mixed precision only after one fixed batch agrees with full precision (D-0045).
    amp_enabled = False
    if spec.amp:
        result.amp = runtime.verify_amp(
            model,
            first_movie.patches[:probe_rows],
            first_movie.labels[:probe_rows],
            prior=first_prior,
            device=device,
            loss_fn=positive_unlabelled_loss,
        )
        amp_enabled = bool(result.amp["enabled"])
        log("amp", json.dumps(result.amp))

    # The batch is a measured knee below the ceiling, not a guess and not the ceiling.
    batch = spec.batch
    if spec.batch_autotune:
        result.batch_ladder = runtime.autotune_batch(
            model,
            first_movie.patches,
            first_movie.labels,
            prior=first_prior,
            device=device,
            loss_fn=positive_unlabelled_loss,
            amp=amp_enabled,
            ceiling_bytes=spec.memory_ceiling_bytes,
            positive_rate=positive_rate,
        )
        batch = int(result.batch_ladder["chosen"])
        log("batch", json.dumps({"chosen": batch, "rungs": result.batch_ladder["rungs"]}))
    result.batch_used = batch

    # E08: the unlabelled risk importance-weighted onto the positives' circumstances.
    weights: list[torch.Tensor] | None = None
    if spec.sampling == "propensity_matched":
        if any(m.circumstances is None for m in train_inputs):
            raise ValueError(
                "propensity-matched sampling needs circumstance features on every training movie"
            )
        per_movie_weights, result.sampling = runtime.matched_sampling_weights(
            [m.circumstances for m in train_inputs], [m.labels.numpy() for m in train_inputs]
        )
        weights = [torch.from_numpy(w) for w in per_movie_weights]
        log("sampling", json.dumps(result.sampling))
    elif spec.sampling != "uniform":
        raise ValueError(f"unknown sampling {spec.sampling!r}")

    scaler = GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")
    meter = runtime.ThroughputMeter()
    for epoch in range(spec.epochs):
        model.train()
        losses: list[float] = []
        clamped = 0
        batches_run = 0
        epoch_positives = 0
        order = torch.randperm(len(train_inputs), generator=generator).tolist()
        for index in order:
            movie = train_inputs[index]
            prior = rescore.candidate_prior(movie.estimate, movie.pool_size)
            # Batch composition is the same for every arm. E08 differs only in
            # the loss, where the unlabelled risk is importance-weighted onto the
            # positives' circumstance distribution; ordering the permutation by
            # weight instead would reweight nothing over an epoch and would
            # cluster the positives against their least similar unlabelled
            # rows, which the residual-propensity test caught on the first run.
            permutation = torch.randperm(movie.pool_size, generator=generator)
            for start in range(0, movie.pool_size, batch):
                waited = time.perf_counter()
                chosen = permutation[start : start + batch]
                labels = movie.labels[chosen].to(device)
                if int(labels.sum()) == 0 or int((~labels).sum()) == 0:
                    meter.skipped_no_positive += 1
                    continue
                x = movie.patches[chosen].to(device, non_blocking=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                compute_started = time.perf_counter()
                meter.wait_seconds += compute_started - waited
                batch_weights = weights[index][chosen].to(device) if weights is not None else None
                with runtime.autocast_context(device, amp_enabled):
                    logits = model(x)
                    loss, terms = positive_unlabelled_loss(
                        logits.float(), labels, prior=prior, unlabelled_weights=batch_weights
                    )
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
                scaler.step(optimizer)
                scaler.update()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                meter.compute_seconds += time.perf_counter() - compute_started
                positives_here = int(labels.sum())
                meter.examples += int(labels.numel())
                meter.positives += positives_here
                meter.batches += 1
                fired = int(bool(terms["negative_risk_clamped"]))
                meter.clamped += fired
                clamped += fired
                losses.append(float(loss.item()))
                batches_run += 1
                epoch_positives += positives_here
                if meter.batches % 20 == 0:
                    meter.sample_utilisation(device)
        runtime.check_ceiling(device, spec.memory_ceiling_bytes, f"epoch {epoch}")
        record = {
            "epoch": epoch,
            "batches": batches_run,
            "batch": batch,
            "positives": epoch_positives,
            "loss_mean": float(np.mean(losses)) if losses else None,
            "loss_last": losses[-1] if losses else None,
            "negative_risk_clamped_batches": clamped,
            "peak_allocated_bytes": runtime.peak_allocated(device),
        }
        result.epochs.append(record)
        log("epoch", json.dumps(record))
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            staged = checkpoint_path.with_suffix(".pt.partial")
            torch.save(model.state_dict(), staged)
            staged.replace(checkpoint_path)
    result.throughput = meter.to_dict()
    log("throughput", json.dumps(result.throughput))

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
    arm_score = result.held_out.get("score")
    a0_score = result.baseline_held_out.get("score")
    if arm_score is not None and a0_score is not None:
        # The second way the objective is killed: the trained scorer's fixed-count
        # held-out ceiling fell below A0 on the same windows.
        result.ceiling_degraded = bool(float(arm_score) < float(a0_score))

    if spec.residual_test:
        result.residual_propensity = residual_propensity_test(train_inputs, train_logits, spec)
        log("residual-propensity", json.dumps(result.residual_propensity))
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
