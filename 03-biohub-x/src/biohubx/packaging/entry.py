"""The package's entry point: the one function a Kaggle notebook calls.

It runs the guards first and refuses before touching a tensor, then trains,
then writes an atomic result manifest carrying a typed digest of the checkpoint
it produced. Every stage prints a ``BIOHUBX_STAGE`` line before and after, so a
run that stalls is visible in the log rather than inferred from a missing
result at the end of an hour.

CPU smoke mode exists so this exact code path can be exercised locally before
any GPU is requested. The only difference is the device and the amount of data.

Consumers: the generated notebook, and ``biohubx package kaggle --smoke-local``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

STAGE = "BIOHUBX_STAGE"

RESCORE_EXPERIMENTS = frozenset({"E07", "E08"})
"""Families the candidate re-scoring loop serves. Anything else is the E03 detector path.

A B3 package labelled E08 once fell through to the detector: the gate's
stage subsequence still passed, because the detector announces the same
milestones, and the only tell was a two-hundred-second local exercise. The
routing is a named set now, and a test holds it.
"""


def stage(name: str, detail: str = "") -> None:
    """One heartbeat line. Printed unbuffered so a stall is visible immediately."""
    print(f"{STAGE} {name} {detail}".rstrip(), flush=True)


class EntryRefusal(RuntimeError):
    """A guard failed. The run stops before it costs anything."""


def _observe_accelerator() -> dict[str, Any]:
    """Every visible device, then the one device training will be pinned to.

    Observation comes first and selection second, deliberately. Restricting
    visibility up front (``CUDA_VISIBLE_DEVICES``) would make a two-card
    allocation look like a one-card allocation and destroy the very fact the
    guard was corrected to check. So the run counts what it was given, names and
    sizes each card, and only then pins itself to ``cuda:0``.

    Consumers: ``run_fold`` and ``run_rescore``, which pass this to the guard and
    copy it into their manifests.
    """
    import torch

    from biohubx.packaging.kaggle import TRAINING_DEVICE

    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(index) for index in range(count)]
    vram = [int(torch.cuda.get_device_properties(index).total_memory) for index in range(count)]
    if count:
        # Explicit, so the default device is the pinned one for any tensor
        # created without a device argument as well as for the ones that name it.
        torch.cuda.set_device(0)
    device = TRAINING_DEVICE if count else "cpu"
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "none",
        "gpu_count": count,
        "device_names": names,
        "device_vram_bytes": vram,
        "training_device": device,
        # The training card's size, which is what every headroom figure is
        # against. A second card's VRAM is recorded above and used by nothing.
        "total_vram_bytes": vram[0] if vram else 0,
        "capability": (
            f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
            if count
            else "none"
        ),
    }


def _stage_accelerator(hardware: dict[str, Any]) -> None:
    """Print what was observed, one line per card, before anything is trained."""
    stage(
        "environment",
        f"torch={hardware['torch']} cuda={hardware['cuda']} gpus={hardware['gpu_count']} "
        f"training_device={hardware['training_device']}",
    )
    for index, (name, size) in enumerate(
        zip(hardware["device_names"], hardware["device_vram_bytes"], strict=True)
    ):
        stage(
            "environment",
            f"visible cuda:{index} name={name!r} vram_total_bytes={size} "
            f"vram_total_gib={size / (1024**3):.2f}"
            + (" <- training" if index == 0 else " <- observed, idle"),
        )
    if not hardware["gpu_count"]:
        stage("environment", "vram_total_bytes=0 no accelerator present")


def _tree_digest(path: Path) -> str:
    """The registry's tree identity for a mounted dataset directory. Reads every byte."""
    from biohubx.hashing import tree_digest

    return tree_digest(path).digest.token


def _tree_shape(path: Path) -> tuple[int, int, int]:
    """File count, empty directory count and total bytes. Reads no file content."""
    from biohubx.hashing import tree_shape

    shape = tree_shape(path)
    return (shape.file_count, shape.empty_directory_count, shape.total_bytes)


def _spatial_shape(path: Path) -> tuple[int, int, int]:
    """The (Z, Y, X) extent of a competition volume, read from metadata only."""
    import zarr

    group: Any = zarr.open(str(path), mode="r")
    shape = group["0"].shape
    return int(shape[1]), int(shape[2]), int(shape[3])


def _discover(data_root: Path) -> tuple[list[str], list[str]]:
    """Mounted training datasets and every path the run can see."""
    train = data_root / "train"
    datasets = sorted(p.stem for p in train.glob("*.geff")) if train.is_dir() else []
    reachable = [str(p) for p in sorted(data_root.iterdir())] if data_root.is_dir() else []
    return datasets, reachable


INPUT_SEARCH_ROOT_ENV = "BIOHUBX_INPUT_ROOT"
INPUT_SEARCH_ROOT_DEFAULT = "/kaggle/input"
INPUT_SEARCH_DEPTH = 3


def _input_listing(search_root: Path, limit: int = 60) -> list[str]:
    """What is actually mounted, so that a refusal diagnoses itself.

    E07-SMOKE-01 attempt 2 refused with the path it wanted and not the paths it
    had, which leaves "the competition is not attached" and "the competition is
    attached somewhere else" indistinguishable from the log ([[R-0026]]).
    """
    found: list[str] = []
    frontier = [search_root]
    depth = 0
    while frontier and depth < INPUT_SEARCH_DEPTH and len(found) < limit:
        following: list[Path] = []
        for parent in frontier:
            try:
                children = sorted(parent.iterdir(), key=lambda item: item.name)
            except OSError:
                continue
            for child in children:
                found.append(str(child))
                if child.is_dir():
                    following.append(child)
                if len(found) >= limit:
                    return found
        frontier = following
        depth += 1
    return found


def _has_competition_layout(candidate: Path, expected_names: set[str]) -> bool:
    """Whether this directory is the competition root, judged by what must be in it.

    Named artifacts rather than a path, for the reason [[D-0033]] gives about the
    wheelhouse: a path is a guess, and the identities the package already carries
    are not. A hit here is only a candidate; the mounted trees are still verified
    against their registered digests before anything trains.
    """
    train = candidate / "train"
    if not train.is_dir():
        return False
    if not expected_names:
        return any(train.glob("*.geff"))
    return any((train / name).exists() for name in expected_names)


def _resolve_data_root(explicit: Path | None, expected_names: set[str]) -> tuple[Path, str]:
    """The competition root, found by the layout it must have rather than assumed.

    Kaggle does not promise where it mounts a source. [[R-0011]] recorded the
    wheelhouse mounting at two different paths in two kernels days apart, and the
    competition root moved the same way between E03-SMOKE-03 and E07-SMOKE-01
    ([[R-0026]]): the hardcoded default below was correct for one of those runs
    and wrong for the other. It is kept as a first candidate and no longer as the
    only one.
    """
    search_root = Path(os.environ.get(INPUT_SEARCH_ROOT_ENV, INPUT_SEARCH_ROOT_DEFAULT))
    candidates: list[tuple[Path, str]] = []
    if explicit is not None:
        candidates.append((explicit, "explicit argument"))
    configured = os.environ.get("BIOHUB_DATA_ROOT")
    if configured:
        candidates.append((Path(configured), "BIOHUB_DATA_ROOT"))
    candidates.append(
        (search_root / "competitions/biohub-cell-tracking-during-development", "the documented path")
    )
    candidates.append((search_root / "biohub-cell-tracking-during-development", "the slug directly"))

    seen: set[str] = set()
    for candidate, how in candidates:
        if str(candidate) in seen:
            continue
        seen.add(str(candidate))
        if _has_competition_layout(candidate, expected_names):
            return candidate, how

    # Nothing named worked, so search. Bounded in depth, and every hit is still
    # subject to the digest verification that follows.
    frontier = [search_root]
    for _ in range(INPUT_SEARCH_DEPTH):
        following: list[Path] = []
        for parent in frontier:
            try:
                children = sorted(parent.iterdir(), key=lambda item: item.name)
            except OSError:
                continue
            for child in children:
                if not child.is_dir():
                    continue
                if str(child) not in seen and _has_competition_layout(child, expected_names):
                    return child, f"found by layout under {search_root}"
                following.append(child)
        frontier = following

    for line in _input_listing(search_root):
        stage("inputs", f"mounted {line}")
    raise EntryRefusal(
        f"the competition training data is not mounted anywhere under {search_root}; "
        f"looked for a directory whose train/ holds {sorted(expected_names)[:3]}"
    )


def run_fold(
    spec_dict: dict[str, Any],
    *,
    data_root: Path | None = None,
    wheelhouse_verified: bool = False,
    wheelhouse_root: str = "",
) -> dict[str, Any]:
    """Guard, train, checkpoint, and write a result manifest. Refuses loudly.

    ``wheelhouse_verified`` is passed by the notebook that did the verifying. It
    defaults to False so that a caller who did not verify cannot pass the guard by
    saying nothing, which is how an absent check becomes a silent one.
    """
    if spec_dict.get("experiment") in RESCORE_EXPERIMENTS:
        return run_rescore(
            spec_dict,
            data_root=data_root,
            wheelhouse_verified=wheelhouse_verified,
            wheelhouse_root=wheelhouse_root,
        )
    from biohubx.packaging.kaggle import FoldSpec, PackageSpec, guard_report

    started = time.perf_counter()
    fold = FoldSpec(**spec_dict["fold"])
    spec = PackageSpec(
        commit=spec_dict["commit"],
        config_path=spec_dict["config_path"],
        config_digest=spec_dict["config_digest"],
        fold=fold,
        epochs=spec_dict["epochs"],
        batch_size=spec_dict["batch_size"],
        learning_rate=spec_dict["learning_rate"],
        accelerator=spec_dict["accelerator"],
        allowed_gpu_counts=tuple(int(n) for n in spec_dict["allowed_gpu_counts"]),
        smoke=spec_dict["smoke"],
        smoke_id=spec_dict["smoke_id"],
        wheelhouse_slug=spec_dict["wheelhouse_slug"],
        wheelhouse_tree=spec_dict["wheelhouse_tree"],
        expected_device_substring=spec_dict.get("expected_device_substring", "Tesla T4"),
        max_movies=spec_dict["max_movies"],
        runtime_ceiling_seconds=spec_dict["runtime_ceiling_seconds"],
    )
    # Set only by the pre-push gate, which runs this entry point on a machine with
    # no accelerator. It is an environment fact about the exercise, never a
    # property of the package, so a remote run cannot acquire it by accident.
    local_exercise = os.environ.get("BIOHUBX_LOCAL_EXERCISE") == "1"
    stage(
        "environment",
        f"{spec.smoke_id} fold={fold.fold_id} smoke={spec.smoke} local_exercise={local_exercise}",
    )

    import torch

    # The build and the card are separate facts and both were unmeasured. The
    # environment audit was CPU-only and measured torch 2.10.0+cpu; a GPU image
    # carries a CUDA build nobody here has seen.
    hardware = _observe_accelerator()
    _stage_accelerator(hardware)
    gpu_count = hardware["gpu_count"]
    device = hardware["training_device"]
    total_vram_bytes = hardware["total_vram_bytes"]

    root, how = _resolve_data_root(data_root, set(spec_dict.get("input_digests", {})))
    stage("inputs", f"root={root} resolved by {how}")
    datasets, reachable = _discover(root)
    if not datasets:
        raise EntryRefusal(f"no training datasets under {root}")

    from biohubx.data.competition import load_ground_truth

    fold_datasets = [d for d in datasets if d.split("_", 1)[0] == fold.train_embryo]
    if spec.max_movies:
        fold_datasets = fold_datasets[: spec.max_movies]
    # Exactly the paths this run will open. The guard checks these rather than
    # the mount listing, because Kaggle mounts the public-test split whether or
    # not a run wants it; reading it is the failure, not its existence.
    opened = [str(root / "train" / f"{d}{suffix}") for d in fold_datasets for suffix in (".zarr", ".geff")]
    stage(
        "fold",
        f"train_embryo={fold.train_embryo} movies={len(fold_datasets)} evaluate={fold.evaluate_embryo}",
    )

    # Identity before training. The digests the package shipped with are what the
    # mounted corpus must match; a run that trains on different bytes than were
    # registered is not the run that was approved.
    # Tiered exactly as `artifacts verify` is (D-0013). Every registered artifact
    # is checked by shape, which is cheap; the artifacts this run will actually
    # open are re-digested in full, which is not. Deep-checking all 142 movies
    # would read tens of gigabytes before a single gradient, and an artifact the
    # run never opens cannot affect what it learns.
    registered: dict[str, str] = spec_dict.get("input_digests", {})
    shapes: dict[str, list[int]] = spec_dict.get("input_shapes", {})
    opened_names = {f"{d}{suffix}" for d in fold_datasets for suffix in (".zarr", ".geff")}
    mounted: dict[str, str] = {}
    shape_failures: list[str] = []
    for name in registered:
        candidate = root / "train" / name
        if not candidate.is_dir():
            continue
        expected = shapes.get(name)
        if expected is not None:
            observed = _tree_shape(candidate)
            if list(observed) != list(expected):
                shape_failures.append(f"{name}: shape {observed} != registered {expected}")
                continue
        if name in opened_names:
            mounted[name] = _tree_digest(candidate)
        else:
            mounted[name] = registered[name]
    for line in shape_failures:
        stage("verify-failed", line)
    if shape_failures:
        raise EntryRefusal(f"mounted inputs do not match their registered shape: {shape_failures[:3]}")
    stage(
        "verify",
        f"registered={len(registered)} shape_checked={len(registered) - len(shape_failures)} "
        f"deep={len(opened_names & set(mounted))}",
    )

    report = guard_report(
        spec,
        mounted_digests=mounted,
        registered_digests=registered,
        dataset_ids=fold_datasets,
        reachable_paths=reachable,
        opened_paths=opened,
        gpu_count=gpu_count,
        device_names=hardware["device_names"],
        device_vram_bytes=hardware["device_vram_bytes"],
        training_device=device,
        dataset_sources=[spec.wheelhouse_slug],
        wheelhouse_verified=wheelhouse_verified,
        local_exercise=local_exercise,
    )
    for line in report["failures"]:
        stage("guard-failed", line)
    if not report["passed"]:
        raise EntryRefusal(f"guards failed: {report['failures']}")
    stage("guard", "passed")

    from biohubx.reference.architecture import REFERENCE_DOWNSAMPLE, ReferenceEdgeModel, ReferenceSpec
    from biohubx.training.targets import class_prior_from_estimate, positive_mask, positive_unlabelled_loss

    torch.manual_seed(fold.seed)
    model_spec = ReferenceSpec(
        unet_out_channels=32,
        unet_layers=(32, 64, 128),
        pos_feat_dim=32,
        window_size=2,
        downsample=REFERENCE_DOWNSAMPLE,
    )
    model = ReferenceEdgeModel(model_spec).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=spec.learning_rate)
    stage("model", f"seed={fold.seed} parameters={sum(p.numel() for p in model.parameters())}")

    from biohubx.data.competition import WindowSelection, load_window

    history: list[dict[str, float]] = []
    for epoch in range(spec.epochs):
        stage("epoch-start", f"{epoch + 1}/{spec.epochs}")
        epoch_loss, batches, clamped, positives_seen = 0.0, 0, 0, 0
        for dataset_id in fold_datasets:
            truth = load_ground_truth(root, dataset_id)
            frames = min(2 if spec.smoke else 8, truth.frames)
            # Start where the annotations are. On 44b6 a movie carries roughly one
            # annotated cell per frame and frames 0..n are often empty, so a fixed
            # start would spend most of a fold's compute on windows with nothing
            # to learn from. The choice uses only this training movie's own
            # annotations, so it leaks nothing from the evaluation embryo.
            annotated_frames = sorted({node.frame for node in truth.lineage.nodes})
            if not annotated_frames:
                stage("skip", f"{dataset_id}: no annotated frames at all")
                continue
            first = min(annotated_frames[0], max(0, truth.frames - frames))
            # Full spatial extent, not E01's crop. That crop was chosen for one
            # densely annotated 6bba movie and is empty on a sparse 44b6 one,
            # which would silently train on nothing.
            depth, height, width = _spatial_shape(root / "train" / f"{dataset_id}.zarr")
            try:
                window = load_window(
                    root,
                    WindowSelection(dataset_id, first, frames, 0, depth, 0, height, 0, width),
                    split="train",
                )
                dz, dy, dx = REFERENCE_DOWNSAMPLE
                grid_np = window.volume[:, ::dz, ::dy, ::dx]
                mask, placed = positive_mask(
                    tuple(grid_np.shape), window.annotated, downsample=REFERENCE_DOWNSAMPLE
                )
                prior = class_prior_from_estimate(window.window_estimated_total_nodes, int(grid_np.size))
            except ValueError as exc:
                stage("skip", f"{dataset_id}: {exc}")
                continue
            grid = torch.from_numpy(grid_np).unsqueeze(0).to(device)
            _, logits = model.detect(grid)
            loss, terms = positive_unlabelled_loss(logits[0, :, 0], mask.to(device), prior=prior)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
            clamped += int(terms["negative_risk_clamped"])
            positives_seen += placed
        if batches == 0:
            raise EntryRefusal(f"epoch {epoch + 1} trained on nothing")
        mean = epoch_loss / batches
        history.append({"epoch": epoch + 1, "loss": mean, "clamped_batches": float(clamped)})
        stage(
            "epoch-end",
            f"{epoch + 1}/{spec.epochs} loss={mean:.8f} batches={batches} "
            f"clamped={clamped} positives={positives_seen}",
        )
        if time.perf_counter() - started > spec.runtime_ceiling_seconds:
            raise EntryRefusal(f"runtime ceiling {spec.runtime_ceiling_seconds}s exceeded")

    out = Path(os.environ.get("BIOHUBX_OUTPUT", "/kaggle/working"))
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / f"detector-{fold.fold_id}.pt"
    staged = checkpoint.with_suffix(".pt.partial")
    torch.save(model.state_dict(), staged)
    staged.replace(checkpoint)
    stage("checkpoint", f"{checkpoint.name}")

    reloaded = ReferenceEdgeModel(model_spec)
    reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    stage("checkpoint", "strict reload ok")

    from biohubx.hashing import DigestKind, digest_file

    checkpoint_digest = digest_file(checkpoint, DigestKind.RAW_ARTIFACT).token
    # Peak rather than current: the number that decides whether a bigger window
    # or batch fits is the high-water mark, and reading it after training is the
    # only point at which it means anything.
    peak_allocated = int(torch.cuda.max_memory_allocated()) if gpu_count else 0
    peak_reserved = int(torch.cuda.max_memory_reserved()) if gpu_count else 0
    stage(
        "memory",
        f"peak_allocated_bytes={peak_allocated} peak_reserved_bytes={peak_reserved} "
        f"total_bytes={total_vram_bytes} "
        f"headroom_bytes={max(0, total_vram_bytes - peak_reserved)}",
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "smoke_id": spec.smoke_id,
        "spec": spec.to_dict(),
        "guards": report,
        "training_movies": fold_datasets,
        "history": history,
        "checkpoint": checkpoint.name,
        "checkpoint_digest": checkpoint_digest,
        "device": device,
        "hardware": {
            **hardware,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "wheelhouse": {
            "slug": spec.wheelhouse_slug,
            "tree": spec.wheelhouse_tree,
            "verified_before_install": wheelhouse_verified,
            "resolved_at": wheelhouse_root,
        },
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    manifest_path = out / f"result-{fold.fold_id}.json"
    partial = manifest_path.with_suffix(".json.partial")
    partial.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(manifest_path)
    stage("done", f"{spec.smoke_id} {manifest_path.name} digest={checkpoint_digest[:46]}")
    return manifest


def run_rescore(
    spec_dict: dict[str, Any],
    *,
    data_root: Path | None = None,
    wheelhouse_verified: bool = False,
    wheelhouse_root: str = "",
) -> dict[str, Any]:
    """Guard, train the E07 re-scorer, freeze the decode, score the held-out embryo, write a manifest.

    The same stage names as ``run_fold`` in the same order, because the pre-push
    gate and the retrieval reader expect that sequence. Under the local
    exercise the loop shrinks to one movie, two frames and one epoch; the
    package itself is unchanged by that, and the stage line says so.
    """
    from biohubx.packaging.kaggle import FoldSpec, PackageSpec, guard_report

    started = time.perf_counter()
    fold = FoldSpec(**spec_dict["fold"])
    spec = PackageSpec(
        commit=spec_dict["commit"],
        config_path=spec_dict["config_path"],
        config_digest=spec_dict["config_digest"],
        fold=fold,
        epochs=spec_dict["epochs"],
        batch_size=spec_dict["batch_size"],
        learning_rate=spec_dict["learning_rate"],
        accelerator=spec_dict["accelerator"],
        allowed_gpu_counts=tuple(int(n) for n in spec_dict["allowed_gpu_counts"]),
        smoke=spec_dict["smoke"],
        smoke_id=spec_dict["smoke_id"],
        wheelhouse_slug=spec_dict["wheelhouse_slug"],
        wheelhouse_tree=spec_dict["wheelhouse_tree"],
        expected_device_substring=spec_dict.get("expected_device_substring", "Tesla T4"),
        max_movies=spec_dict["max_movies"],
        runtime_ceiling_seconds=spec_dict["runtime_ceiling_seconds"],
    )
    loop = dict(spec_dict["loop"])
    local_exercise = os.environ.get("BIOHUBX_LOCAL_EXERCISE") == "1"
    if local_exercise:
        loop.update({"train_movies": 1, "evaluate_movies": 1, "frames": 2, "epochs": 1})
    stage(
        "environment",
        f"{spec.smoke_id} E07 arm={loop['arm']} fold={fold.fold_id} smoke={spec.smoke} "
        f"local_exercise={local_exercise}",
    )

    hardware = _observe_accelerator()
    _stage_accelerator(hardware)
    gpu_count = hardware["gpu_count"]
    device = hardware["training_device"]

    root, how = _resolve_data_root(data_root, set(spec_dict.get("input_digests", {})))
    stage("inputs", f"root={root} resolved by {how}")
    datasets, reachable = _discover(root)
    if not datasets:
        raise EntryRefusal(f"no training datasets under {root}")
    shipped_train = [str(d) for d in loop.get("train_datasets", [])]
    if shipped_train:
        # Chosen at package time from the training embryo's own annotation
        # counts and carried as a list; the run reads what was chosen. A shipped
        # movie that is not mounted is a refusal, not a silent fallback to name
        # order, because then the run would not be the one that was gated.
        absent = sorted(set(shipped_train) - set(datasets))
        if absent:
            raise EntryRefusal(f"shipped training movies are not mounted: {absent[:5]}")
        train_ids = shipped_train[:1] if local_exercise else shipped_train
    else:
        train_ids = [d for d in datasets if d.split("_", 1)[0] == fold.train_embryo][
            : int(loop["train_movies"]) or None
        ]
    eval_ids = [d for d in datasets if d.split("_", 1)[0] == fold.evaluate_embryo][
        : int(loop["evaluate_movies"]) or None
    ]
    opened_ids = train_ids + eval_ids
    opened = [str(root / "train" / f"{d}{suffix}") for d in opened_ids for suffix in (".zarr", ".geff")]
    stage(
        "fold",
        f"train_embryo={fold.train_embryo} movies={len(train_ids)} "
        f"evaluate={fold.evaluate_embryo} movies={len(eval_ids)}",
    )

    registered: dict[str, str] = spec_dict.get("input_digests", {})
    shapes: dict[str, list[int]] = spec_dict.get("input_shapes", {})
    opened_names = {f"{d}{suffix}" for d in opened_ids for suffix in (".zarr", ".geff")}
    mounted: dict[str, str] = {}
    shape_failures: list[str] = []
    for name in registered:
        candidate = root / "train" / name
        if not candidate.is_dir():
            continue
        expected = shapes.get(name)
        if expected is not None:
            observed = _tree_shape(candidate)
            if list(observed) != list(expected):
                shape_failures.append(f"{name}: shape {observed} != registered {expected}")
                continue
        mounted[name] = _tree_digest(candidate) if name in opened_names else registered[name]
    for line in shape_failures:
        stage("verify-failed", line)
    if shape_failures:
        raise EntryRefusal(f"mounted inputs do not match their registered shape: {shape_failures[:3]}")
    stage("verify", f"registered={len(registered)} deep={len(opened_names & set(mounted))}")

    report = guard_report(
        spec,
        mounted_digests=mounted,
        registered_digests=registered,
        # The guard's dataset_ids are the movies the run trains on. The held-out
        # movies are opened for evaluation only and travel in opened_paths, where
        # the public-test and registry checks still apply to them.
        dataset_ids=train_ids,
        reachable_paths=reachable,
        opened_paths=opened,
        gpu_count=gpu_count,
        device_names=hardware["device_names"],
        device_vram_bytes=hardware["device_vram_bytes"],
        training_device=device,
        dataset_sources=[spec.wheelhouse_slug],
        wheelhouse_verified=wheelhouse_verified,
        local_exercise=local_exercise,
    )
    for line in report["failures"]:
        stage("guard-failed", line)
    if not report["passed"]:
        raise EntryRefusal(f"guards failed: {report['failures']}")
    stage("guard", "passed")

    out = Path(os.environ.get("BIOHUBX_OUTPUT", "/kaggle/working"))
    out.mkdir(parents=True, exist_ok=True)
    workers = [dict(w) for w in loop.get("workers", [])] or [{"seed": int(fold.seed)}]
    if local_exercise:
        workers = workers[:1]
    parallel = gpu_count >= 2 and len(workers) >= 2 and not local_exercise
    stage(
        "workers",
        f"count={len(workers)} seeds={[w.get('seed') for w in workers]} "
        f"devices_visible={gpu_count} mode={'one worker per device' if parallel else 'serial on ' + device}",
    )

    if parallel:
        # One isolated process per visible device (D-0045): its own device, optimizer,
        # checkpoint, manifest and worker id. Nothing is shared but the read-only
        # inputs and the spec that was gated.
        import subprocess

        jobs: list[tuple[dict[str, Any], Path, subprocess.Popen[bytes]]] = []
        pending = list(enumerate(workers))
        manifests: list[dict[str, Any]] = []
        while pending:
            wave, pending = pending[:gpu_count], pending[gpu_count:]
            jobs = []
            for slot, (index, worker) in enumerate(wave):
                worker_device = f"cuda:{slot}"
                payload = {
                    "spec": spec_dict,
                    "loop": loop,
                    "worker": worker,
                    "worker_index": index,
                    "device": worker_device,
                    "hardware": hardware,
                    "guards": report,
                    "root": str(root),
                    "out": str(out),
                    "train_ids": train_ids,
                    "wheelhouse_verified": wheelhouse_verified,
                    "wheelhouse_root": wheelhouse_root,
                    "started": started,
                }
                job_path = out / f"worker-{index}.json"
                job_path.write_text(json.dumps(payload, default=str), encoding="utf-8")
                argv, env = worker_command(job_path, worker_device)
                process = subprocess.Popen(argv, env=env)
                jobs.append((worker, job_path, process))
                stage(
                    "worker-start",
                    f"index={index} seed={worker.get('seed')} device={worker_device} pid={process.pid}",
                )
            for worker, job_path, process in jobs:
                code = process.wait()
                result_path = job_path.with_suffix(".result.json")
                if code != 0 or not result_path.is_file():
                    raise EntryRefusal(f"worker seed={worker.get('seed')} exited {code} without a manifest")
                manifests.append(json.loads(result_path.read_text(encoding="utf-8")))
                stage("worker-done", f"seed={worker.get('seed')} manifest={manifests[-1]['manifest_name']}")
    else:
        manifests = []
        for index, worker in enumerate(workers):
            manifests.append(
                run_worker(
                    spec_dict,
                    loop,
                    worker,
                    index,
                    device,
                    root=root,
                    out=out,
                    train_ids=train_ids,
                    hardware=hardware,
                    guards=report,
                    wheelhouse_verified=wheelhouse_verified,
                    wheelhouse_root=wheelhouse_root,
                    started=started,
                    local_exercise=local_exercise,
                )
            )

    summary: dict[str, Any] = {
        "schema_version": 2,
        "experiment": spec_dict.get("experiment", "E07"),
        "smoke_id": spec.smoke_id,
        "arm": str(loop["arm"]),
        "fold": fold.fold_id,
        "spec": spec.to_dict(),
        "guards": report,
        "hardware": hardware,
        "local_exercise": local_exercise,
        "parallel_workers": parallel,
        "workers": [
            {
                "worker_id": m["worker_id"],
                "seed": m["seed"],
                "device": m["device"],
                "manifest": m["manifest_name"],
                "manifest_digest": m["manifest_digest"],
                "checkpoint_digest": m["loop"].get("checkpoint_digest"),
                "held_out_score": (m["loop"].get("held_out") or {}).get("arm", {}).get("score"),
                "A0_score": (m["loop"].get("held_out") or {}).get("A0", {}).get("score"),
                "ceiling_degraded": m["loop"].get("ceiling_degraded"),
                "residual_propensity_passed": (m["loop"].get("residual_propensity") or {}).get("passed"),
                "peak_allocated_bytes": m["hardware"].get("peak_allocated_bytes"),
                "throughput": m["loop"].get("throughput"),
            }
            for m in manifests
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    summary_path = out / f"e07-{fold.fold_id}-{loop['arm']}.json"
    staged = summary_path.with_suffix(".json.partial")
    staged.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    staged.replace(summary_path)
    stage(
        "done", f"manifest={summary_path.name} workers={len(manifests)} elapsed={summary['elapsed_seconds']}s"
    )
    return summary


def worker_command(job_path: Path, device: str) -> tuple[list[str], dict[str, str]]:
    """The child's argv and environment.

    A child process does not inherit ``sys.path``, and the payload is importable
    only through the directory the notebook inserted there, so both workers of
    wave-1 attempts 1 and 2 died on ``No module named biohubx`` before touching a
    tensor ([[R-0028]], [[R-0029]]). The package's ``src`` directory travels in
    ``PYTHONPATH`` explicitly; a test spawns a child that can find it no other way.
    """
    import sys

    package_src = str(Path(__file__).resolve().parents[2])
    inherited = os.environ.get("PYTHONPATH", "")
    env = {
        **os.environ,
        "BIOHUBX_WORKER_DEVICE": device,
        "PYTHONPATH": package_src + (os.pathsep + inherited if inherited else ""),
    }
    argv = [
        sys.executable,
        "-c",
        "import sys; from biohubx.packaging.entry import run_worker_file; run_worker_file(sys.argv[1])",
        str(job_path),
    ]
    return argv, env


def run_worker_file(path: str) -> None:
    """Subprocess entry: one worker, one device, from the job the parent wrote."""
    job = json.loads(Path(path).read_text(encoding="utf-8"))
    device = str(job["device"])
    import torch

    torch.cuda.set_device(int(device.split(":")[1]))
    manifest = run_worker(
        job["spec"],
        job["loop"],
        job["worker"],
        int(job["worker_index"]),
        device,
        root=Path(job["root"]),
        out=Path(job["out"]),
        train_ids=list(job["train_ids"]),
        hardware=dict(job["hardware"]),
        guards=dict(job["guards"]),
        wheelhouse_verified=bool(job["wheelhouse_verified"]),
        wheelhouse_root=str(job["wheelhouse_root"]),
        started=float(job["started"]),
        local_exercise=False,
    )
    Path(path).with_suffix(".result.json").write_text(json.dumps(manifest, default=str), encoding="utf-8")


def run_worker(
    spec_dict: dict[str, Any],
    loop: dict[str, Any],
    worker: dict[str, Any],
    index: int,
    device: str,
    *,
    root: Path,
    out: Path,
    train_ids: list[str],
    hardware: dict[str, Any],
    guards: dict[str, Any],
    wheelhouse_verified: bool,
    wheelhouse_root: str,
    started: float,
    local_exercise: bool,
) -> dict[str, Any]:
    """Train one seed on one device and write its own manifest. Shares nothing with another worker."""
    import torch

    from biohubx.packaging.kaggle import FoldSpec
    from biohubx.proposals import rescore
    from biohubx.training.rescore_loop import LoopSpec, run_loop

    fold = FoldSpec(**spec_dict["fold"])
    seed = int(worker.get("seed", fold.seed))
    arm = str(loop["arm"])
    worker_id = f"{spec_dict['smoke_id']}-{arm}-{fold.fold_id}-s{seed}"
    loop_spec = LoopSpec(
        arm=arm,
        train_embryo=fold.train_embryo,
        evaluate_embryo=fold.evaluate_embryo,
        train_movies=int(loop["train_movies"]),
        evaluate_movies=int(loop["evaluate_movies"]),
        frames=int(loop["frames"]),
        epochs=int(loop["epochs"]),
        batch=int(loop["batch"]),
        learning_rate=float(loop["learning_rate"]),
        seed=seed,
        radii_um=tuple(float(r) for r in loop["radii_um"]),
        suppression_radius_um=float(loop["suppression_radius_um"]),
        pool_quantile=float(loop["pool_quantile"]),
        baseline_quantile=float(loop["baseline_quantile"]),
        config_digest=str(spec_dict["config_digest"]),
        count_ratio=float(loop.get("count_ratio", 1.0)),
        train_datasets=tuple(train_ids),
        amp=bool(loop.get("amp", False)) and not local_exercise,
        batch_autotune=bool(loop.get("batch_autotune", False)) and not local_exercise,
        memory_ceiling_bytes=int(loop.get("memory_ceiling_bytes", 13 * 1024**3)),
        sampling=rescore.sampling_for(arm),
        residual_test=bool(loop.get("residual_test", False)),
        worker_id=worker_id,
        propensity_parameters=loop.get("propensity"),
    )
    stage(
        "model",
        f"worker={worker_id} device={device} count_ratio={loop_spec.count_ratio} "
        f"sampling={loop_spec.sampling}",
    )
    checkpoint = out / f"rescorer-{arm}-{fold.fold_id}-s{seed}.pt"
    ceiling_seconds = int(spec_dict["runtime_ceiling_seconds"])

    def log(name: str, detail: str) -> None:
        mapped = {
            "epoch": "epoch-end",
            "inputs": "inputs-built",
            "decode": "decode",
            "held-out": "held-out",
            "cache": "cache",
        }
        if name == "epoch":
            stage("epoch-start", f"{worker_id} " + detail[:40])
        stage(mapped.get(name, name), f"{worker_id} {detail}")
        if time.perf_counter() - started > ceiling_seconds:
            raise EntryRefusal(f"runtime ceiling {ceiling_seconds}s exceeded")

    torch_device = torch.device(device)
    result = run_loop(
        root,
        loop_spec,
        device=torch_device,
        cache_dir=out / "cache",
        checkpoint_path=checkpoint,
        log=log,
    )
    stage("checkpoint", f"{worker_id} {checkpoint.name} strict reload identical={result.reload_identical}")
    if result.reload_identical is False:
        raise EntryRefusal("the reloaded checkpoint did not reproduce its own output")
    peak_allocated = int(torch.cuda.max_memory_allocated(torch_device)) if torch_device.type == "cuda" else 0
    peak_reserved = int(torch.cuda.max_memory_reserved(torch_device)) if torch_device.type == "cuda" else 0
    stage("memory", f"{worker_id} peak_allocated_bytes={peak_allocated} peak_reserved_bytes={peak_reserved}")

    manifest_name = f"e07-{fold.fold_id}-{arm}-s{seed}.json"
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "experiment": spec_dict.get("experiment", "E07"),
        "smoke_id": spec_dict["smoke_id"],
        "worker_id": worker_id,
        "seed": seed,
        "device": device,
        "spec": {k: v for k, v in spec_dict.items() if k not in {"input_digests", "input_shapes"}},
        "loop": result.to_dict(),
        "guards": guards,
        "local_exercise": local_exercise,
        "hardware": {
            **hardware,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "wheelhouse": {
            "slug": spec_dict["wheelhouse_slug"],
            "tree": spec_dict["wheelhouse_tree"],
            "verified_before_install": wheelhouse_verified,
            "resolved_at": wheelhouse_root,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    manifest_path = out / manifest_name
    staged = manifest_path.with_suffix(".json.partial")
    text = json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    staged.write_text(text, encoding="utf-8")
    staged.replace(manifest_path)
    import hashlib

    manifest["manifest_name"] = manifest_name
    manifest["manifest_digest"] = (
        "raw_artifact_sha256:sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    stage("worker-manifest", f"{worker_id} {manifest_name} digest={manifest['manifest_digest'][:46]}")
    return manifest
