"""Read-only validation for explicitly supplied competition-data roots.

The official layout nests datasets under split directories: ``train/`` holds a
``.zarr`` beside its ``.geff`` ground truth, and ``test/`` holds a ``.zarr``
alone. The absence of ground truth in the test split is the design, not a fault,
so a validator that reads it as an unpaired artifact is wrong about the world
rather than strict about it.

Both artifact kinds are Zarr-backed directories, not files. Presence is therefore
recorded as a kind plus a cheap top-level entry count, so that a present
directory can never be confused with a missing artifact. A single ``None`` size
would have conflated the two.

Nothing here writes to, moves, or opens the data. It inspects names and stats.

Consumer: ``biohubx data validate``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from biohubx.artifacts import atomic_write_text

ValidationReport = dict[str, object]
"""A validation report: schema_version, mode, status, and a list of dataset manifests."""

LAYOUT_SCHEMA = "official-split-layout/v1"
LEAF_LAYOUT_SCHEMA = "official-leaf-split/v1"
SYNTHETIC_LAYOUT_SCHEMA = "synthetic-leaf-split/v1"


class SplitRole(StrEnum):
    """Whether a split is expected to carry ground truth."""

    ANNOTATED = "annotated"
    """Every dataset pairs a volume with its sparse ground-truth graph."""

    UNANNOTATED = "unannotated"
    """Volumes only, by design. This is the submission split, not a broken one."""


class RoleProvenance(StrEnum):
    """How the role was determined, so a guess is never mistaken for a fact."""

    OFFICIAL_SPLIT_NAME = "official_split_name"
    """The directory is named by the official layout, so the role is declared."""

    INFERRED_FROM_CONTENTS = "inferred_from_contents"
    """A leaf directory whose role was read off what it happens to contain."""


class ArtifactKind(StrEnum):
    """What is actually on disk at an artifact path."""

    DIRECTORY = "directory"
    FILE = "file"
    ABSENT = "absent"


OFFICIAL_SPLIT_ROLES: dict[str, SplitRole] = {
    "train": SplitRole.ANNOTATED,
    "test": SplitRole.UNANNOTATED,
}
"""The official split names and what each one promises about annotation."""


@dataclass(frozen=True, slots=True)
class ArtifactPresence:
    """One artifact, and enough to tell present-directory from absent.

    ``size_bytes`` is populated only for a regular file and ``entry_count`` only
    for a directory, so the three states are distinguishable without reading a
    second field. The entry count is one directory listing, not a recursive walk:
    measuring bytes across a multi-gigabyte Zarr tree is a different job with a
    different cost, and it is not needed to answer "is this the right layout".
    """

    name: str
    kind: ArtifactKind
    size_bytes: int | None
    entry_count: int | None


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """One dataset, its split, and what that split promises about annotation."""

    dataset_id: str
    split: str
    split_role: SplitRole
    role_provenance: RoleProvenance
    relative_path: str
    schema: str
    artifacts: tuple[ArtifactPresence, ...]
    provenance: str
    split_identity: str


def _describe(path: Path) -> ArtifactPresence:
    if path.is_dir():
        return ArtifactPresence(
            name=path.name,
            kind=ArtifactKind.DIRECTORY,
            size_bytes=None,
            entry_count=sum(1 for _ in path.iterdir()),
        )
    if path.is_file():
        return ArtifactPresence(
            name=path.name,
            kind=ArtifactKind.FILE,
            size_bytes=path.stat().st_size,
            entry_count=None,
        )
    return ArtifactPresence(name=path.name, kind=ArtifactKind.ABSENT, size_bytes=None, entry_count=None)


def validate_data_root(root: Path) -> list[DatasetManifest]:
    """Validate an official competition root, or a single split directory.

    A root containing ``train/`` or ``test/`` is validated as the official split
    layout. A root containing datasets directly is validated as one leaf split,
    with its role inferred from the contents and that inference recorded. Any
    other shape is refused rather than guessed at.
    """
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError("explicit data root is missing or is not a directory")

    split_dirs = [name for name in sorted(OFFICIAL_SPLIT_ROLES) if (root / name).is_dir()]
    root_artifacts = sorted(root.glob("*.zarr")) + sorted(root.glob("*.geff"))

    if split_dirs and root_artifacts:
        raise ValueError(
            f"ambiguous data layout: {root} holds split directories {split_dirs} AND "
            f"{len(root_artifacts)} dataset artifacts of its own; it can be one or the other"
        )

    if split_dirs:
        manifests: list[DatasetManifest] = []
        for split in split_dirs:
            manifests.extend(
                _validate_split(
                    root / split,
                    split=split,
                    role=OFFICIAL_SPLIT_ROLES[split],
                    provenance=RoleProvenance.OFFICIAL_SPLIT_NAME,
                    schema=LAYOUT_SCHEMA,
                    relative_path=split,
                )
            )
        if not manifests:
            raise ValueError(f"split directories {split_dirs} contain no dataset artifacts")
        return manifests

    if not root_artifacts:
        raise ValueError(
            f"{root} contains no .zarr/.geff dataset artifacts and no train/ or test/ split directory"
        )

    role = _infer_leaf_role(root)
    return _validate_split(
        root,
        split=".",
        role=role,
        provenance=RoleProvenance.INFERRED_FROM_CONTENTS,
        schema=LEAF_LAYOUT_SCHEMA,
        relative_path=".",
    )


def _infer_leaf_role(root: Path) -> SplitRole:
    """Read a leaf directory's role off its contents, refusing a half-annotated one."""
    zarr_ids = {path.stem for path in root.glob("*.zarr")}
    geff_ids = {path.stem for path in root.glob("*.geff")}
    if not geff_ids:
        return SplitRole.UNANNOTATED
    if geff_ids == zarr_ids:
        return SplitRole.ANNOTATED
    raise ValueError(
        f"ambiguous data layout: {root} is partly annotated, so its role cannot be read off its "
        f"contents. Datasets with a volume but no ground truth: "
        f"{sorted(zarr_ids - geff_ids)}. Ground truth with no volume: {sorted(geff_ids - zarr_ids)}"
    )


def _validate_split(
    directory: Path,
    *,
    split: str,
    role: SplitRole,
    provenance: RoleProvenance,
    schema: str,
    relative_path: str,
) -> list[DatasetManifest]:
    """Validate one split directory against what its role promises."""
    zarr_paths = sorted(directory.glob("*.zarr"))
    geff_paths = sorted(directory.glob("*.geff"))
    zarr_ids = {path.stem for path in zarr_paths}
    geff_ids = {path.stem for path in geff_paths}

    nested = [path for path in directory.rglob("*.zarr") if path.parent != directory]
    if nested:
        raise ValueError(
            f"ambiguous data layout: {directory} nests dataset artifacts below its own level, "
            f"first at {nested[0].relative_to(directory)}"
        )

    if not zarr_ids:
        raise ValueError(f"split {split!r} at {directory} contains no .zarr volumes")

    if role is SplitRole.ANNOTATED:
        without_truth = sorted(zarr_ids - geff_ids)
        without_volume = sorted(geff_ids - zarr_ids)
        if without_truth or without_volume:
            raise ValueError(
                f"split {split!r} is annotated, so every volume needs its ground truth. "
                f"Volumes with no .geff: {without_truth}. "
                f"Ground truth with no .zarr: {without_volume}"
            )
    elif geff_ids:
        # Ground truth in the submission split would mean the layout is not what
        # it claims, and silently accepting it could leak labels into a split
        # that is supposed to have none.
        raise ValueError(
            f"split {split!r} is unannotated by design but contains ground truth for "
            f"{sorted(geff_ids)}; refusing rather than assuming the layout changed"
        )

    manifests: list[DatasetManifest] = []
    for dataset_id in sorted(zarr_ids):
        names = [f"{dataset_id}.zarr"]
        if role is SplitRole.ANNOTATED:
            names.append(f"{dataset_id}.geff")
        manifests.append(
            DatasetManifest(
                dataset_id=dataset_id,
                split=split,
                split_role=role,
                role_provenance=provenance,
                relative_path=relative_path,
                schema=schema,
                artifacts=tuple(_describe(directory / name) for name in names),
                provenance="host-supplied-existing-competition-data",
                split_identity=dataset_id,
            )
        )
    return manifests


def synthetic_validation_report() -> ValidationReport:
    """Return the dry fixture used when tests explicitly request synthetic mode."""
    return {
        "schema_version": 1,
        "mode": "synthetic",
        "status": "valid",
        "datasets": [
            asdict(
                DatasetManifest(
                    dataset_id="synthetic-dry",
                    split=".",
                    split_role=SplitRole.ANNOTATED,
                    role_provenance=RoleProvenance.INFERRED_FROM_CONTENTS,
                    relative_path=".",
                    schema=SYNTHETIC_LAYOUT_SCHEMA,
                    artifacts=(
                        ArtifactPresence(
                            name="synthetic-dry.zarr",
                            kind=ArtifactKind.ABSENT,
                            size_bytes=None,
                            entry_count=None,
                        ),
                        ArtifactPresence(
                            name="synthetic-dry.geff",
                            kind=ArtifactKind.ABSENT,
                            size_bytes=None,
                            entry_count=None,
                        ),
                    ),
                    provenance="generated-in-memory; no source data accessed",
                    split_identity="synthetic-dry",
                )
            )
        ],
    }


def write_validation_report(path: Path, report: ValidationReport) -> None:
    atomic_write_text(path, json.dumps(report, indent=2, sort_keys=True) + "\n")
