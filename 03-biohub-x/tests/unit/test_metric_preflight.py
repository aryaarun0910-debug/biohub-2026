"""The metric preflight compares; it does not merely report.

A kernel that measured the target and printed numbers would leave the comparison
to whoever reads the log. These tests hold the builder to refusing a spec with no
baseline, the metadata to CPU with the metric wheelhouse alone, the comparison to
naming the key that differs, and the vendored official source in this tree to the
digests the registry pins, since that is the baseline the target is held to.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from biohubx.cli import app, repository_root
from biohubx.packaging.audit import RUNTIME_CLOSURE
from biohubx.packaging.kaggle import PackagingError
from biohubx.packaging.metric_preflight import (
    METRIC_WHEELHOUSE_SLUG,
    build_metric_preflight_notebook,
    metric_preflight_kernel_metadata,
)
from biohubx.packaging.metric_preflight_runtime import (
    EXPECTED_ABSENT,
    OFFICIAL_SOURCE_FILES,
    compare_real_data,
    fixtures_digest,
    official_source_integrity,
    strict_imports,
)

runner = CliRunner()

WHEELHOUSE_PAYLOAD = {
    "tree": "tree_sha256:sha256/v1:" + "a" * 64,
    "records": [["f", "b" * 64, "10", "wheels/x-1.0-py3-none-any.whl"]],
}
BASELINE = {
    "fixtures_digest": "c" * 64,
    "fixtures_count": 15,
    "real_data": {"dataset": "44b6_x", "frames": 2, "proposals": 100, "scored": True, "score": 0.5},
    "official_source": {name: "d" * 64 for name in OFFICIAL_SOURCE_FILES},
    "shipped_versions": {"zarr": {"version": "3.3.0", "sha256": "e" * 64}},
}


def _spec(**overrides: object) -> dict[str, object]:
    spec: dict[str, object] = {
        "preflight_id": "E06-METRIC-PREFLIGHT-01",
        "commit": "0" * 40,
        "kernel": "aryaarun07/biohub-x-metric-preflight",
        "runtime_ceiling_seconds": 900,
        "frozen_proposals": {},
        "baseline": BASELINE,
    }
    spec.update(overrides)
    return spec


def _payload() -> bytes:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("__init__.py", "")
    return buffer.getvalue()


def test_kernel_metadata_is_cpu_offline_with_the_metric_wheelhouse_alone() -> None:
    metadata = metric_preflight_kernel_metadata()
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["dataset_sources"] == [METRIC_WHEELHOUSE_SLUG]
    assert metadata["competition_sources"] == ["biohub-cell-tracking-during-development"]
    assert metadata["model_sources"] == [] and metadata["kernel_sources"] == []
    assert metadata["id"] == "aryaarun07/biohub-x-metric-preflight"


def test_builder_refuses_a_spec_without_a_baseline() -> None:
    incomplete = dict(BASELINE)
    del incomplete["real_data"]
    with pytest.raises(PackagingError, match="baseline 'real_data'"):
        build_metric_preflight_notebook(
            _spec(baseline=incomplete), payload=_payload(), wheelhouse_payload=WHEELHOUSE_PAYLOAD
        )


def test_builder_carries_payload_digest_identity_and_baseline() -> None:
    payload = _payload()
    notebook = build_metric_preflight_notebook(
        _spec(), payload=payload, wheelhouse_payload=WHEELHOUSE_PAYLOAD
    )
    source = "".join(notebook["cells"][0]["source"])
    assert hashlib.sha256(payload).hexdigest() in source
    assert str(WHEELHOUSE_PAYLOAD["tree"]) in source
    assert "c" * 64 in source, "the fixtures digest the target must reproduce travels in the cell"
    assert "metric_preflight_runtime import run" in source
    assert "/kaggle/input/biohubx-package" not in source


def test_builder_refuses_an_identity_that_is_not_a_tree_token() -> None:
    with pytest.raises(PackagingError, match="canonical tree token"):
        build_metric_preflight_notebook(
            _spec(), payload=_payload(), wheelhouse_payload={"tree": "sha256:abc", "records": [["f"]]}
        )


def test_fixtures_digest_is_order_independent_and_stable() -> None:
    one = fixtures_digest({"b": {"x": 1}, "a": [1, 2.5, "s"]})
    two = fixtures_digest({"a": [1, 2.5, "s"], "b": {"x": 1}})
    assert one == two
    assert one != fixtures_digest({"a": [1, 2.5, "s"], "b": {"x": 2}})


def test_compare_real_data_names_the_key_that_differs() -> None:
    expected = {
        "proposals": 1000,
        "matched_nodes": 90,
        "annotated_nodes": 100,
        "annotated_edges": 80,
        "retained_edges": 70,
        "retained_divisions": 1,
        "estimated_nodes": 950.0,
        "scored": True,
        "score": 0.9,
        "adjusted_edge_jaccard": 0.8,
        "edge_jaccard": 0.85,
        "division_jaccard": 1.0,
    }
    same = compare_real_data(expected, dict(expected))
    assert same["exact"] and same["within_tolerance"]

    near = dict(expected, proposals=1001, score=0.9 + 5e-7)
    result = compare_real_data(expected, near)
    assert not result["exact"] and result["within_tolerance"]
    assert result["per_key"]["proposals"]["exact"] is False
    assert result["per_key"]["proposals"]["within_tolerance"] is True

    far = dict(expected, retained_edges=60)
    result = compare_real_data(expected, far)
    assert not result["within_tolerance"]
    assert result["per_key"]["retained_edges"] == {
        "expected": 70,
        "observed": 60,
        "exact": False,
        "within_tolerance": False,
    }

    unscored = dict(expected, scored=False)
    assert compare_real_data(expected, unscored)["within_tolerance"] is False


def test_strict_imports_probe_the_closure_and_notice_a_wrong_shipped_version() -> None:
    result = strict_imports({"numpy": {"version": "0.0.0-not-installed", "sha256": "f" * 64}})
    assert result["probed"] == len(RUNTIME_CLOSURE) - len(EXPECTED_ABSENT)
    assert set(result["expected_absent"]) == set(EXPECTED_ABSENT)
    assert any(line.startswith("numpy: installed") for line in result["version_mismatches"])
    assert result["ok"] is False


def test_vendored_official_source_matches_the_registry_digests() -> None:
    """The baseline the target is held to is first held here."""
    root = repository_root()
    official = yaml.safe_load((root / "registry/official_source.yaml").read_text(encoding="utf-8"))
    expected = {}
    for item in official["sources"]:
        relative = str(item["vendored_path"]).removeprefix("src/biohubx/")
        if relative in OFFICIAL_SOURCE_FILES:
            expected[relative] = str(item["digests"]["raw"]).removeprefix("raw_artifact_sha256:sha256:")
    assert set(expected) == set(OFFICIAL_SOURCE_FILES)
    result = official_source_integrity(root / "src/biohubx", expected)
    assert result["ok"], result["failures"]
    assert result["inside_package"]


def _stage_built_wheelhouse(tmp_path: Path, *, stated_hash: str | None = None) -> tuple[Path, str]:
    house = tmp_path / "house"
    (house / "wheels").mkdir(parents=True)
    wheel = house / "wheels" / "builtpkg-1.0-py3-none-any.whl"
    wheel.write_bytes(b"not a real wheel, an identity")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    listed = stated_hash or digest
    (house / "requirements-offline.txt").write_text(
        f"builtpkg==1.0 \\\n    --hash=sha256:{listed}\n", encoding="utf-8"
    )
    (house / "dataset-metadata.json").write_text(json.dumps({"id": "o/x", "title": "x"}), encoding="utf-8")
    return house, digest


def test_wheelhouse_identity_accepts_a_built_wheel_only_with_its_registered_digest(tmp_path: Path) -> None:
    house, digest = _stage_built_wheelhouse(tmp_path)
    report = tmp_path / "identity.json"
    result = runner.invoke(
        app,
        [
            "package",
            "wheelhouse",
            "--path",
            str(house),
            "--out",
            str(report),
            "--built-wheel",
            f"builtpkg-1.0-py3-none-any.whl=sha256:{digest}",
        ],
    )
    assert result.exit_code == 0, result.output
    rows = json.loads(report.read_text(encoding="utf-8"))["wheels"]
    assert rows[0]["hash_authority"] == "biohubx_build"
    assert rows[0]["matches_registered_build"] is True
    assert rows[0]["matches_uv_lock"] is False

    wrong = runner.invoke(
        app,
        [
            "package",
            "wheelhouse",
            "--path",
            str(house),
            "--out",
            str(report),
            "--built-wheel",
            "builtpkg-1.0-py3-none-any.whl=sha256:" + "0" * 64,
        ],
    )
    assert wrong.exit_code == 2
    assert "does not match the registered build digest" in wrong.output

    unnamed = runner.invoke(app, ["package", "wheelhouse", "--path", str(house), "--out", str(report)])
    assert unnamed.exit_code == 2
    assert "appears in no uv.lock entry" in unnamed.output
