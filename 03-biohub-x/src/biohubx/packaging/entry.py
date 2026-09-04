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
    if spec_dict.get("experiment") == "E07":
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

    root = data_root or Path(
        os.environ.get(
            "BIOHUB_DATA_ROOT",
            "/kaggle/input/competitions/biohub-cell-tracking-during-development",
        )
    )
    stage("inputs", f"root={root}")
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
    from biohubx.training.rescore_loop import LoopSpec, run_loop

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

    import torch

    hardware = _observe_accelerator()
    _stage_accelerator(hardware)
    gpu_count = hardware["gpu_count"]
    device = hardware["training_device"]
    total_vram_bytes = hardware["total_vram_bytes"]

    root = data_root or Path(
        os.environ.get(
            "BIOHUB_DATA_ROOT", "/kaggle/input/competitions/biohub-cell-tracking-during-development"
        )
    )
    stage("inputs", f"root={root}")
    datasets, reachable = _discover(root)
    if not datasets:
        raise EntryRefusal(f"no training datasets under {root}")
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

    loop_spec = LoopSpec(
        arm=str(loop["arm"]),
        train_embryo=fold.train_embryo,
        evaluate_embryo=fold.evaluate_embryo,
        train_movies=int(loop["train_movies"]),
        evaluate_movies=int(loop["evaluate_movies"]),
        frames=int(loop["frames"]),
        epochs=int(loop["epochs"]),
        batch=int(loop["batch"]),
        learning_rate=float(loop["learning_rate"]),
        seed=int(fold.seed),
        radii_um=tuple(float(r) for r in loop["radii_um"]),
        suppression_radius_um=float(loop["suppression_radius_um"]),
        pool_quantile=float(loop["pool_quantile"]),
        baseline_quantile=float(loop["baseline_quantile"]),
        config_digest=spec.config_digest,
        count_ratio=float(loop.get("count_ratio", 1.0)),
    )
    stage("model", f"arm={loop_spec.arm} seed={fold.seed} count_ratio={loop_spec.count_ratio}")
    out = Path(os.environ.get("BIOHUBX_OUTPUT", "/kaggle/working"))
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / f"rescorer-{loop_spec.arm}-{fold.fold_id}.pt"

    def log(name: str, detail: str) -> None:
        mapped = {
            "epoch": "epoch-end",
            "inputs": "inputs-built",
            "decode": "decode",
            "held-out": "held-out",
            "cache": "cache",
        }
        if name == "epoch":
            stage("epoch-start", detail[:40])
        stage(mapped.get(name, name), detail)
        if time.perf_counter() - started > spec.runtime_ceiling_seconds:
            raise EntryRefusal(f"runtime ceiling {spec.runtime_ceiling_seconds}s exceeded")

    result = run_loop(
        root,
        loop_spec,
        device=torch.device(device),
        cache_dir=out / "cache",
        checkpoint_path=checkpoint,
        log=log,
    )
    stage("checkpoint", f"{checkpoint.name} strict reload identical={result.reload_identical}")
    if result.reload_identical is False:
        raise EntryRefusal("the reloaded checkpoint did not reproduce its own output")
    peak_allocated = int(torch.cuda.max_memory_allocated()) if gpu_count else 0
    peak_reserved = int(torch.cuda.max_memory_reserved()) if gpu_count else 0
    stage(
        "memory",
        f"peak_allocated_bytes={peak_allocated} peak_reserved_bytes={peak_reserved} "
        f"total_bytes={total_vram_bytes}",
    )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "E07",
        "smoke_id": spec.smoke_id,
        "spec": spec.to_dict(),
        "loop": result.to_dict(),
        "guards": report,
        "local_exercise": local_exercise,
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
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    manifest_path = out / f"e07-{fold.fold_id}-{loop_spec.arm}.json"
    staged = manifest_path.with_suffix(".json.partial")
    staged.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    staged.replace(manifest_path)
    stage("done", f"manifest={manifest_path.name} elapsed={manifest['elapsed_seconds']}s")
    return manifest
