"""The diagnostic must be able to run on the machine it is diagnosing.

Which means it may not import Biohub-X, may not need the corpus, may not ask for
an accelerator, and may not stop at the first absent package. One traceback told
us zarr was missing; it told us nothing about the Python version or platform tag
a wheel would have to match.
"""

from __future__ import annotations

import ast
import json

import pytest

from biohubx.packaging.audit import (
    AUDIT_ID,
    AUDIT_KERNEL,
    AUDIT_RUNTIME_CEILING_SECONDS,
    EXPECTED_AUDIT_STAGES,
    RUNTIME_CLOSURE,
    AuditSpec,
    audit_kernel_metadata,
    blosc_fixture,
    build_audit_notebook,
)

SPEC = AuditSpec(
    audit_id=AUDIT_ID,
    commit="0" * 40,
    kernel=AUDIT_KERNEL,
    runtime_ceiling_seconds=AUDIT_RUNTIME_CEILING_SECONDS,
)


def source_of(spec: AuditSpec = SPEC) -> str:
    return "".join(build_audit_notebook(spec)["cells"][0]["source"])


def test_the_audit_asks_for_no_accelerator_no_internet_and_no_sources() -> None:
    metadata = audit_kernel_metadata()
    assert metadata["id"] == "aryaarun07/biohub-x-environment-audit"
    assert metadata["enable_gpu"] is False
    assert metadata["enable_internet"] is False
    assert metadata["dataset_sources"] == []
    assert metadata["model_sources"] == []
    assert metadata["competition_sources"] == [], "a diagnostic has no business touching the corpus"


def test_the_audit_does_not_import_the_thing_it_is_diagnosing() -> None:
    """A diagnostic that depends on Biohub-X cannot run when Biohub-X cannot."""
    source = source_of()
    assert "import biohubx" not in source
    assert "from biohubx" not in source
    assert "/kaggle/input" not in source
    assert "BIOHUB_DATA_ROOT" not in source, "the audit must not reach for the corpus"


def test_the_audit_cell_is_valid_python_and_carries_nbformat_fields() -> None:
    notebook = build_audit_notebook(SPEC)
    ast.parse("".join(notebook["cells"][0]["source"]))
    assert notebook["cells"][0]["id"]
    for required in ("display_name", "language", "name"):
        assert notebook["metadata"]["kernelspec"][required]


def test_every_probe_is_wrapped_so_one_absence_cannot_end_the_audit() -> None:
    source = source_of()
    # The import loop and every codec probe catch BaseException deliberately: a
    # missing compiled extension can raise things that are not ImportError.
    assert source.count("except BaseException") >= 5
    assert "missing_imports" in source
    assert 'stage("done"' in source


def test_the_probe_list_covers_the_stack_that_actually_failed() -> None:
    for name in ("zarr", "numcodecs", "blosc2", "tracksdata", "torch", "polars", "pydantic"):
        assert name in RUNTIME_CLOSURE, f"{name} is part of the runtime closure"
    assert len(RUNTIME_CLOSURE) == len(set(RUNTIME_CLOSURE)), "duplicate probes waste a run"


def test_the_codec_fixture_uses_the_competition_codec() -> None:
    """Importability is not the question; decoding blosc/zstd/bitshuffle is."""
    pytest.importorskip("numcodecs")
    import base64

    import numpy as np
    from numcodecs import Blosc

    payload, description = blosc_fixture()
    assert description == "uint16 arange(256)"
    decoded = np.frombuffer(
        Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE, blocksize=0).decode(
            base64.b64decode(payload)
        ),
        dtype="uint16",
    )
    assert np.array_equal(decoded, np.arange(256, dtype=np.uint16))
    assert payload in source_of(), "the fixture must travel inside the notebook"


def test_the_audit_writes_its_manifest_atomically() -> None:
    source = source_of()
    assert ".json.partial" in source
    assert "partial.replace(manifest)" in source


def test_the_expected_stage_sequence_ends_with_a_manifest() -> None:
    assert EXPECTED_AUDIT_STAGES[0] == "audit-start"
    assert EXPECTED_AUDIT_STAGES[-2:] == ("manifest", "done")
    for stage in EXPECTED_AUDIT_STAGES:
        assert f'stage("{stage}"' in source_of(), f"{stage} is never emitted"


def test_the_spec_travels_in_the_notebook() -> None:
    source = source_of()
    assert AUDIT_ID in source
    assert json.dumps(SPEC.to_dict(), sort_keys=True) in source
    assert str(AUDIT_RUNTIME_CEILING_SECONDS) in source
