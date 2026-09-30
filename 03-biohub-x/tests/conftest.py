"""Repository-wide test isolation fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "artifacts"


@pytest.fixture(scope="session", autouse=True)
def preserve_repository_artifacts() -> Iterator[None]:
    """Leave operator reports exactly as the test session found them.

    CLI tests intentionally exercise the stable production output paths. Their
    tiny fixture reports are useful inside the session, but they must not erase
    evidence from a real data run merely because somebody executed the gate.
    """
    before = {
        path.relative_to(ARTIFACT_ROOT): path.read_bytes()
        for path in ARTIFACT_ROOT.rglob("*")
        if path.is_file()
    }
    try:
        yield
    finally:
        current = [path for path in ARTIFACT_ROOT.rglob("*") if path.is_file()]
        for path in current:
            if path.relative_to(ARTIFACT_ROOT) not in before:
                path.unlink()
        for relative_path, content in before.items():
            target = ARTIFACT_ROOT / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
