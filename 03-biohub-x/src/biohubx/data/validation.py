"""Read-only validation for explicitly supplied competition-data roots.

Consumer: ``biohubx data validate``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from biohubx.artifacts import atomic_write_text

ValidationReport = dict[str, object]
"""A validation report: schema_version, mode, status, and a list of dataset manifests."""


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    dataset_id: str
    relative_path: str
    schema: str
    expected_files: tuple[str, str]
    sizes_bytes: dict[str, int | None]
    provenance: str
    split_identity: str


def _path_size(path: Path) -> int | None:
    return path.stat().st_size if path.is_file() else None


def validate_data_root(root: Path) -> list[DatasetManifest]:
    """Validate the one unambiguous official sibling-pair layout without writes."""
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError("explicit data root is missing or is not a directory")

    root_zarr = sorted(root.glob("*.zarr"))
    root_geff = sorted(root.glob("*.geff"))
    nested = [path for path in root.rglob("*.zarr") if path.parent != root]
    if nested:
        raise ValueError("ambiguous data layout: nested and/or non-root dataset artifacts found")
    if not root_zarr and not root_geff:
        raise ValueError("data root contains no .zarr/.geff dataset artifacts")

    zarr_ids = {path.stem for path in root_zarr}
    geff_ids = {path.stem for path in root_geff}
    if zarr_ids != geff_ids:
        missing_geff = sorted(zarr_ids - geff_ids)
        missing_zarr = sorted(geff_ids - zarr_ids)
        raise ValueError(f"ambiguous data layout: unpaired artifacts geff={missing_geff} zarr={missing_zarr}")

    manifests: list[DatasetManifest] = []
    for dataset_id in sorted(zarr_ids):
        zarr = root / f"{dataset_id}.zarr"
        geff = root / f"{dataset_id}.geff"
        manifests.append(
            DatasetManifest(
                dataset_id=dataset_id,
                relative_path=".",
                schema="official-zarr-geff-sibling-pair/v1",
                expected_files=(zarr.name, geff.name),
                sizes_bytes={zarr.name: _path_size(zarr), geff.name: _path_size(geff)},
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
                    relative_path=".",
                    schema="synthetic-zarr-geff-sibling-pair/v1",
                    expected_files=("synthetic-dry.zarr", "synthetic-dry.geff"),
                    sizes_bytes={"synthetic-dry.zarr": None, "synthetic-dry.geff": None},
                    provenance="generated-in-memory; no source data accessed",
                    split_identity="synthetic-dry",
                )
            )
        ],
    }


def write_validation_report(path: Path, report: ValidationReport) -> None:
    atomic_write_text(path, json.dumps(report, indent=2, sort_keys=True) + "\n")
