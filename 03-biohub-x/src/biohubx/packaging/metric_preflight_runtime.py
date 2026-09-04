"""The five metric checks, run on the target inside the verified package, and their baseline computed here.

Strict imports, official-source integrity, the scorer's characterisation
fixtures, behavioural equivalence against this machine, and one tiny real-data
score. Each check records what it measured whether or not the previous one
passed, so a failed run says everything it learned and not only where it
stopped. Nothing here decides anything about cell tracking; it decides whether
the scorer on the target is the scorer here.

Consumers: the notebook ``biohubx package metric-preflight`` builds, through
:func:`run`; the command itself, through :func:`local_baseline`.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any

from biohubx.packaging.audit import RUNTIME_CLOSURE

EXPECTED_ABSENT: tuple[str, ...] = ("imagecodecs",)
"""Declared by tracksdata, imported by none of its modules, and left out of the
wheelhouse because the lock's build would drag numpy, numba and llvmlite onto
the image (R-0015). The preflight checks it is still absent, so a wheelhouse that
quietly grew it would be noticed."""

OFFICIAL_SOURCE_FILES: tuple[str, ...] = (
    "_vendor/official_competition/tracking_cellmot/__init__.py",
    "_vendor/official_competition/tracking_cellmot/metrics.py",
    "_vendor/official_competition/tracking_cellmot/division_metrics.py",
)
"""Relative to the biohubx package directory. The LICENSE beside them is not a
.py file and does not travel in the source archive; its digest is in
registry/official_source.yaml and is not re-checked on the target."""

FROZEN_PROPOSALS: dict[str, Any] = {
    "source": "biohubx.proposals.dog",
    "local_maxima_only": True,
    "radii_um": [2.0, 3.0],
    "response_quantile": 0.95,
    "suppression_radius_um": 4.0,
    "refine_centroids": False,
    "truncation": "none",
    "frozen_by": "D-0041",
}
COUNT_TOLERANCE = 0.001
"""Relative. Proposal counts on two builds of the same scipy can differ by a peak
that sits on a floating-point tie; a tenth of a percent is far above that and
far below anything a scoring difference could hide behind."""
SCORE_TOLERANCE = 1e-6
DEFAULT_COMPETITION_ROOT = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
REPORT_NAME = "metric-preflight.json"


def stage(name: str, detail: str = "") -> None:
    print(("BIOHUBX_STAGE " + name + " " + str(detail)).rstrip(), flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixtures_digest(fixtures: dict[str, Any]) -> str:
    """One digest over the whole fixture table, canonical JSON, so equality is one comparison."""
    canonical = json.dumps(fixtures, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def environment_summary() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for dist in (
        "numpy",
        "scipy",
        "polars",
        "tracksdata",
        "zarr",
        "numcodecs",
        "rustworkx",
        "geff",
        "pydantic",
        "numba",
    ):
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            versions[dist] = "absent"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": versions,
    }


def strict_imports(shipped_versions: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Check 1. Every closure name imports, the expected-absent one does not, shipped versions match."""
    names = [name for name in RUNTIME_CLOSURE if name not in EXPECTED_ABSENT]
    imports: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for name in names:
        entry: dict[str, Any] = {"importable": False}
        try:
            module = importlib.import_module(name)
            entry["importable"] = True
            with warnings.catch_warnings():
                # click deprecates __version__; the attribute is read only to record it
                warnings.simplefilter("ignore", DeprecationWarning)
                entry["version"] = str(getattr(module, "__version__", "unknown"))
            entry["file"] = str(getattr(module, "__file__", "unknown"))
        except BaseException as exc:
            entry["error"] = repr(exc)[:300]
            failures.append(f"{name}: {entry['error']}")
        imports[name] = entry
    absent: dict[str, Any] = {}
    for name in EXPECTED_ABSENT:
        try:
            importlib.import_module(name)
            absent[name] = {"absent": False}
            failures.append(f"{name} is importable but the wheelhouse expects it absent")
        except ImportError:
            absent[name] = {"absent": True}
    mismatches: list[str] = []
    installed: dict[str, str] = {}
    for dist, expected in shipped_versions.items():
        try:
            installed[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            installed[dist] = "absent"
        if installed[dist] != expected["version"]:
            mismatches.append(f"{dist}: installed {installed[dist]}, wheelhouse ships {expected['version']}")
    return {
        "probed": len(names),
        "imports": imports,
        "failures": failures,
        "expected_absent": absent,
        "shipped_installed_versions": installed,
        "version_mismatches": mismatches,
        "ok": not failures and not mismatches,
    }


def official_source_integrity(package_dir: Path, expected: dict[str, str]) -> dict[str, Any]:
    """Check 2. The vendored official files are what the registry pins, and the scorer imports them."""
    files: dict[str, Any] = {}
    failures: list[str] = []
    for relative, sha in expected.items():
        path = package_dir / relative
        if not path.is_file():
            files[relative] = {"present": False}
            failures.append(f"{relative} is missing from the package")
            continue
        observed = _sha256(path)
        files[relative] = {"present": True, "expected": sha, "observed": observed, "matches": observed == sha}
        if observed != sha:
            failures.append(f"{relative} does not match the pinned digest")
    module_file = "unknown"
    inside = False
    try:
        from biohubx._vendor.official_competition.tracking_cellmot import metrics as authoritative

        module_file = str(Path(str(authoritative.__file__)).resolve())
        inside = module_file.startswith(str(package_dir.resolve()))
        if not inside:
            failures.append(f"the authoritative module was imported from {module_file}, outside the package")
    except BaseException as exc:
        failures.append(f"the authoritative module does not import: {exc!r}"[:300])
    return {
        "files": files,
        "authoritative_module": module_file,
        "inside_package": inside,
        "failures": failures,
        "ok": not failures,
    }


def scorer_fixtures() -> dict[str, Any]:
    """Check 3. The fifteen characterisation fixtures through the pinned adapter."""
    from biohubx.evaluation.official_metric import evaluate_calibration_fixtures

    fixtures = evaluate_calibration_fixtures()
    return {"count": len(fixtures), "digest": fixtures_digest(fixtures), "fixtures": fixtures}


def tiny_real_data_score(
    data_root: Path, dataset_id: str, frames: int, frozen: dict[str, Any]
) -> dict[str, Any]:
    """Check 5. One movie, a few frames, frozen proposals, an oracle graph, the official scorer."""
    from biohubx.data.competition import WindowSelection, load_ground_truth, load_window
    from biohubx.evaluation.official_metric import EstimatedTotalNodes, metric_row, summarise_fold
    from biohubx.evaluation.oracle import oracle_graph
    from biohubx.proposals import dog

    started = time.monotonic()
    truth = load_ground_truth(data_root, dataset_id)
    annotated_frames = sorted({node.frame for node in truth.lineage.nodes})
    if not annotated_frames:
        raise ValueError(f"{dataset_id} has no annotated frame")
    first = min(annotated_frames[0], max(0, truth.frames - frames))
    import zarr

    group: Any = zarr.open(str(data_root / "train" / f"{dataset_id}.zarr"), mode="r")
    depth, height, width = group["0"].shape[1:]
    window = load_window(
        data_root, WindowSelection(dataset_id, first, frames, 0, depth, 0, height, 0, width), split="train"
    )
    instances = dog.detect_instances(
        window.volume,
        dataset=window.annotated.dataset,
        radii_um=tuple(float(r) for r in frozen["radii_um"]),
        response_quantile=float(frozen["response_quantile"]),
        suppression_radius_um=float(frozen["suppression_radius_um"]),
        local_maxima_only=bool(frozen["local_maxima_only"]),
    )
    ceiling = oracle_graph(instances, window.annotated)
    result: dict[str, Any] = {
        "dataset": dataset_id,
        "first_frame": first,
        "frames": frames,
        "shape_zyx": [int(depth), int(height), int(width)],
        "proposals": int(ceiling.proposals),
        "annotated_nodes": int(ceiling.annotated_nodes),
        "matched_nodes": int(ceiling.matched_nodes),
        "annotated_edges": int(ceiling.annotated_edges),
        "retained_edges": int(ceiling.retained_edges),
        "retained_divisions": int(ceiling.retained_divisions),
        "estimated_nodes": float(window.window_estimated_total_nodes),
        "scored": ceiling.graph is not None,
    }
    if ceiling.graph is not None:
        row = metric_row(
            ceiling.graph,
            window.annotated,
            estimated_total_nodes=EstimatedTotalNodes.declared(float(window.window_estimated_total_nodes)),
        )
        fold = summarise_fold([row])
        result.update(
            {
                "score": float(fold.score),
                "adjusted_edge_jaccard": float(fold.adjusted_edge_jaccard),
                "edge_jaccard": float(fold.edge_jaccard),
                "division_jaccard": None if fold.division_jaccard is None else float(fold.division_jaccard),
            }
        )
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result


COUNT_KEYS = (
    "proposals",
    "annotated_nodes",
    "matched_nodes",
    "annotated_edges",
    "retained_edges",
    "retained_divisions",
)
SCORE_KEYS = ("score", "adjusted_edge_jaccard", "edge_jaccard", "division_jaccard", "estimated_nodes")


def compare_real_data(expected: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    """Check 4, real-data half. Exact is the claim; within tolerance is the fallback that still passes."""
    per_key: dict[str, dict[str, Any]] = {}
    exact = True
    within = True
    for key in COUNT_KEYS:
        want, have = expected.get(key), observed.get(key)
        if want is None or have is None:
            per_key[key] = {"expected": want, "observed": have, "exact": False, "within_tolerance": False}
            exact = within = False
            continue
        difference = abs(int(want) - int(have))
        allowed = max(0, int(COUNT_TOLERANCE * max(int(want), 1)))
        per_key[key] = {
            "expected": want,
            "observed": have,
            "exact": difference == 0,
            "within_tolerance": difference <= allowed,
        }
        exact &= difference == 0
        within &= difference <= allowed
    for key in SCORE_KEYS:
        want, have = expected.get(key), observed.get(key)
        if want is None and have is None:
            per_key[key] = {"expected": None, "observed": None, "exact": True, "within_tolerance": True}
            continue
        if want is None or have is None:
            per_key[key] = {"expected": want, "observed": have, "exact": False, "within_tolerance": False}
            exact = within = False
            continue
        gap = abs(float(want) - float(have))
        per_key[key] = {
            "expected": want,
            "observed": have,
            "exact": gap == 0.0,
            "within_tolerance": gap <= SCORE_TOLERANCE,
        }
        exact &= gap == 0.0
        within &= gap <= SCORE_TOLERANCE
    if bool(expected.get("scored")) != bool(observed.get("scored")):
        per_key["scored"] = {
            "expected": expected.get("scored"),
            "observed": observed.get("scored"),
            "exact": False,
            "within_tolerance": False,
        }
        exact = within = False
    return {"per_key": per_key, "exact": exact, "within_tolerance": within}


def local_baseline(
    *,
    data_root: Path,
    dataset_id: str,
    frames: int,
    package_dir: Path,
    official_expected: dict[str, str],
    shipped_versions: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """What this machine measures, to travel in the spec and be reproduced on the target."""
    source = official_source_integrity(package_dir, official_expected)
    if not source["ok"]:
        raise ValueError(f"the local package does not carry the pinned official source: {source['failures']}")
    fixtures = scorer_fixtures()
    real = tiny_real_data_score(data_root, dataset_id, frames, FROZEN_PROPOSALS)
    return {
        "fixtures_digest": fixtures["digest"],
        "fixtures_count": fixtures["count"],
        "real_data": real,
        "official_source": official_expected,
        "shipped_versions": shipped_versions,
        "local_environment": environment_summary(),
    }


def run(
    spec: dict[str, Any],
    *,
    wheelhouse_root: str,
    package_dir: str,
    output_dir: str | None = None,
    data_root: str | None = None,
) -> dict[str, Any]:
    """The five checks on the target, in order, every one recorded, then a verdict."""
    started = time.time()
    baseline = spec["baseline"]
    out = Path(output_dir or os.environ.get("BIOHUBX_OUTPUT", "/kaggle/working"))
    package = Path(package_dir)
    report: dict[str, Any] = {
        "schema_version": 1,
        "preflight_id": spec["preflight_id"],
        "commit": spec["commit"],
        "wheelhouse_root": wheelhouse_root,
        "target_environment": environment_summary(),
        "local_environment": baseline.get("local_environment"),
        "checks": {},
    }

    def write() -> Path:
        out.mkdir(parents=True, exist_ok=True)
        manifest = out / REPORT_NAME
        partial = manifest.with_suffix(".json.partial")
        partial.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        partial.replace(manifest)
        return manifest

    stage("preflight-start", spec["preflight_id"] + " commit=" + spec["commit"])

    stage(
        "imports",
        f"strict, {len(RUNTIME_CLOSURE) - len(EXPECTED_ABSENT)} names, "
        f"{','.join(EXPECTED_ABSENT)} expected absent",
    )
    imports = strict_imports(baseline["shipped_versions"])
    report["checks"]["imports"] = imports
    for failure in imports["failures"][:20] + imports["version_mismatches"][:20]:
        stage("imports", "FAIL " + failure)
    stage("imports", f"ok={imports['ok']} probed={imports['probed']}")
    write()

    stage("official-source", "hashing the vendored files against the registry digests")
    source = official_source_integrity(package, baseline["official_source"])
    report["checks"]["official_source"] = source
    for failure in source["failures"][:20]:
        stage("official-source", "FAIL " + failure)
    stage("official-source", "ok={} inside_package={}".format(source["ok"], source["inside_package"]))
    write()

    stage("fixtures", "begin")
    fixtures_check: dict[str, Any]
    try:
        fixtures = scorer_fixtures()
        matches = fixtures["digest"] == baseline["fixtures_digest"]
        fixtures_check = {
            "count": fixtures["count"],
            "digest": fixtures["digest"],
            "expected_digest": baseline["fixtures_digest"],
            "matches_local": matches,
            "fixtures": fixtures["fixtures"],
            "ok": matches and fixtures["count"] == baseline["fixtures_count"],
        }
    except BaseException as exc:
        fixtures_check = {"error": repr(exc)[:600], "ok": False}
    report["checks"]["fixtures"] = fixtures_check
    stage(
        "fixtures",
        "ok={} digest={}".format(fixtures_check["ok"], str(fixtures_check.get("digest", "-"))[:16]),
    )
    write()

    stage("real-data", "begin")
    expected_real = baseline["real_data"]
    root = Path(data_root or os.environ.get("BIOHUB_DATA_ROOT", DEFAULT_COMPETITION_ROOT))
    real_check: dict[str, Any]
    try:
        observed_real = tiny_real_data_score(
            root, str(expected_real["dataset"]), int(expected_real["frames"]), spec["frozen_proposals"]
        )
        comparison = compare_real_data(expected_real, observed_real)
        real_check = {
            "root": str(root),
            "observed": observed_real,
            "expected": expected_real,
            "comparison": comparison,
            "ok": comparison["within_tolerance"],
        }
    except BaseException as exc:
        real_check = {"root": str(root), "error": repr(exc)[:600], "ok": False}
    report["checks"]["real_data"] = real_check
    if "observed" in real_check:
        stage(
            "real-data",
            "proposals={} score={} expected={}".format(
                real_check["observed"].get("proposals"),
                real_check["observed"].get("score"),
                expected_real.get("score"),
            ),
        )
    stage("real-data", "ok={}".format(real_check["ok"]))
    write()

    stage("equivalence", "fixtures exact, real data within tolerance, shipped versions equal")
    equivalence = {
        "fixtures_exact": bool(fixtures_check.get("matches_local")),
        "real_data_exact": bool(real_check.get("comparison", {}).get("exact")),
        "real_data_within_tolerance": bool(real_check.get("comparison", {}).get("within_tolerance")),
        "shipped_versions_equal": not imports["version_mismatches"],
    }
    equivalence["ok"] = bool(
        equivalence["fixtures_exact"]
        and equivalence["real_data_within_tolerance"]
        and equivalence["shipped_versions_equal"]
    )
    report["checks"]["equivalence"] = equivalence
    stage("equivalence", json.dumps(equivalence, sort_keys=True))

    report["ok"] = all(bool(check.get("ok")) for check in report["checks"].values())
    report["elapsed_seconds"] = round(time.time() - started, 3)
    manifest = write()
    stage("manifest", str(manifest) + " bytes=" + str(manifest.stat().st_size))
    stage("done", "ok={} elapsed={}s".format(report["ok"], report["elapsed_seconds"]))
    if not report["ok"]:
        raise SystemExit("the metric preflight failed; see " + str(manifest))
    return report
