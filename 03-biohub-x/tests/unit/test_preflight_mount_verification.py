"""The preflight must not trust the tree it mounts, and must not import to check it.

``requirements-offline.txt`` cannot authenticate itself. A substituted dataset
shipping its own matching requirements file satisfies ``pip --require-hashes``
exactly, so the hash check binds nothing unless the identity of the whole tree is
established first, from an anchor that arrived with the code.

The package may not import Biohub-X, because it has to run on an image where
Biohub-X cannot, so the authoritative walk cannot travel as code. It travels as
behaviour: the load-bearing test here executes the notebook's own verifier and
requires it to return exactly what ``biohubx.hashing.tree_digest`` returns for the
same tree. A second implementation of an identity is only safe while something
holds it to the first one.
"""

from __future__ import annotations

import hashlib
import pathlib
from pathlib import Path
from typing import Any

import pytest

from biohubx.hashing import tree_digest
from biohubx.packaging.audit import AuditError, AuditSpec
from biohubx.packaging.preflight import (
    EXPECTED_PREFLIGHT_STAGES,
    PREFLIGHT_ID,
    PREFLIGHT_KERNEL,
    PREFLIGHT_RUNTIME_CEILING_SECONDS,
    TREE_VERIFIER_SOURCE,
    build_preflight_notebook,
)

SPEC = AuditSpec(
    audit_id=PREFLIGHT_ID,
    commit="0" * 40,
    kernel=PREFLIGHT_KERNEL,
    runtime_ceiling_seconds=PREFLIGHT_RUNTIME_CEILING_SECONDS,
)

PAYLOAD = {
    "tree": "tree_sha256:sha256/v1:" + "a" * 64,
    "records": [["f", "b" * 64, "12", "requirements-offline.txt"]],
}


def verifier() -> dict[str, Any]:
    """Execute the verifier exactly as the notebook carries it, and nothing else."""
    namespace: dict[str, Any] = {"hashlib": hashlib, "pathlib": pathlib}
    exec(TREE_VERIFIER_SOURCE, namespace)
    return namespace


def source_of(payload: dict[str, Any] = PAYLOAD) -> str:
    return "".join(build_preflight_notebook(SPEC, published_payload=payload)["cells"][0]["source"])


def build_tree(root: Path) -> Path:
    (root / "wheels").mkdir(parents=True)
    (root / "wheels/zarr-3.3.0-py3-none-any.whl").write_bytes(b"zarr wheel bytes")
    (root / "wheels/numcodecs-0.15.1-cp312.whl").write_bytes(b"numcodecs wheel bytes")
    (root / "requirements-offline.txt").write_text("zarr==3.3.0\n", encoding="utf-8")
    (root / "WHEELHOUSE.json").write_text('{"schema_version": 1}\n', encoding="utf-8")
    (root / "README.md").write_text("what this is\n", encoding="utf-8")
    return root


def test_the_carried_verifier_agrees_with_the_authoritative_walk(tmp_path: Path) -> None:
    """The load-bearing test. Two implementations, one identity, or the gate fails here.

    The notebook cannot import ``biohubx.hashing``, so canonicalization v1 exists
    twice. That is only safe while something requires the copy to agree with the
    original, which is what this asserts, on a tree with nested directories and
    several files rather than on a single file where almost anything would agree.
    """
    root = build_tree(tmp_path / "mounted")
    authoritative = tree_digest(root)

    token, records = verifier()["biohubx_canonical_tree"](root)

    assert token == authoritative.digest.token
    assert [record[3] for record in records] == [record.relative_path for record in authoritative.records]
    assert len(records) == authoritative.file_count


def test_the_carried_verifier_agrees_on_an_empty_directory_and_a_zero_byte_file(
    tmp_path: Path,
) -> None:
    """Both cases have their own record shape, and both are easy to get subtly wrong.

    An empty directory is recorded in its own right under canonicalization v1, and
    a zero-byte file must serialise its size as ``0`` rather than as the ``-`` that
    marks an absent field.
    """
    root = build_tree(tmp_path / "mounted")
    (root / "empty").mkdir()
    (root / "wheels/zero.whl").write_bytes(b"")

    token, _ = verifier()["biohubx_canonical_tree"](root)

    assert token == tree_digest(root).digest.token


def test_the_carried_verifier_refuses_a_symlink_rather_than_following_it(tmp_path: Path) -> None:
    """Following one would record foreign bytes under a path that looks local."""
    root = build_tree(tmp_path / "mounted")
    outside = tmp_path / "outside.txt"
    outside.write_text("not part of the artifact\n", encoding="utf-8")
    try:
        (root / "link.txt").symlink_to(outside)
    except OSError:
        pytest.skip("creating symlinks needs privilege on Windows")

    with pytest.raises(RuntimeError, match="symlink inside the mounted tree"):
        verifier()["biohubx_canonical_tree"](root)


def test_the_carried_comparison_names_extra_missing_renamed_and_modified_files(
    tmp_path: Path,
) -> None:
    """A digest says the tree is wrong; the operator needs to know which file."""
    expected_root = build_tree(tmp_path / "expected")
    observed_root = build_tree(tmp_path / "observed")
    (observed_root / "README.md").unlink()
    (observed_root / "RENAMED.md").write_text("what this is\n", encoding="utf-8")
    (observed_root / "wheels/zarr-3.3.0-py3-none-any.whl").write_bytes(b"zarr wheel bytez")
    (observed_root / "EXTRA.txt").write_text("uninvited\n", encoding="utf-8")

    namespace = verifier()
    _, expected = namespace["biohubx_canonical_tree"](expected_root)
    _, observed = namespace["biohubx_canonical_tree"](observed_root)
    differences = namespace["biohubx_compare"](expected, observed)

    assert "missing README.md" in differences
    assert "extra RENAMED.md" in differences
    assert "extra EXTRA.txt" in differences
    assert any(item.startswith("content wheels/zarr-3.3.0-py3-none-any.whl") for item in differences)


def test_a_same_size_substitution_is_caught_by_the_carried_comparison(tmp_path: Path) -> None:
    """The negative control: a flipped byte moves no count and no total."""
    expected_root = build_tree(tmp_path / "expected")
    observed_root = build_tree(tmp_path / "observed")
    (observed_root / "requirements-offline.txt").write_text("zarr==3.3.1\n", encoding="utf-8")

    namespace = verifier()
    expected_token, expected = namespace["biohubx_canonical_tree"](expected_root)
    observed_token, observed = namespace["biohubx_canonical_tree"](observed_root)
    differences = namespace["biohubx_compare"](expected, observed)

    assert expected_token != observed_token
    assert [item.split()[0] for item in differences] == ["content"]


def test_verification_happens_before_pip_is_invoked() -> None:
    """Order is the whole point. Verifying after installing verifies nothing."""
    source = source_of()

    assert source.index('stage("verify"') < source.index('"pip", "install"')
    assert source.index("biohubx_canonical_tree(WHEELHOUSE)") < source.index('"pip", "install"')
    stages = EXPECTED_PREFLIGHT_STAGES
    assert stages.index("verify") < stages.index("install")


def test_the_expected_identity_travels_in_the_notebook_and_is_not_read_from_the_mount() -> None:
    """The anchor arrives with the code, or it anchors nothing.

    A substituted dataset can ship any WHEELHOUSE.json and any requirements file it
    likes, so an expectation sourced from the mount is an expectation the attacker
    writes.
    """
    payload = {
        "tree": "tree_sha256:sha256/v1:" + "c" * 64,
        "records": [["f", "d" * 64, "7", "requirements-offline.txt"]],
    }
    source = source_of(payload)

    assert "tree_sha256:sha256/v1:" + "c" * 64 in source
    verify_block = source[source.index('stage("verify"') : source.index('stage("install"')]
    assert "WHEELHOUSE.json" not in verify_block
    assert "EXPECTED" in verify_block


def test_the_verifier_the_notebook_carries_is_the_verifier_the_tests_exercise() -> None:
    """Otherwise these tests bind a copy nothing ships."""
    assert TREE_VERIFIER_SOURCE in source_of()


def test_the_preflight_still_imports_no_part_of_biohubx() -> None:
    source = source_of()

    assert "import biohubx" not in source
    assert "from biohubx" not in source


def test_a_package_cannot_be_built_without_an_identity_to_check() -> None:
    """A preflight that installs from whatever it mounted is the thing being removed."""
    with pytest.raises(AuditError, match="canonical tree identity"):
        build_preflight_notebook(SPEC, published_payload={"tree": "", "records": [["f", "x", "1", "a"]]})

    with pytest.raises(AuditError, match="carries no records"):
        build_preflight_notebook(SPEC, published_payload={"tree": PAYLOAD["tree"], "records": []})


def test_a_bare_hexadecimal_expectation_is_refused_because_it_states_no_kind() -> None:
    with pytest.raises(AuditError, match="canonical tree identity"):
        build_preflight_notebook(
            SPEC, published_payload={"tree": "a" * 64, "records": [["f", "x", "1", "a"]]}
        )
