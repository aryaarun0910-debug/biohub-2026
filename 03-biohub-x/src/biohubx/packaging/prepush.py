"""Everything that must pass before a package is allowed near the network.

E03-SMOKE was authorised, pushed once, and produced nothing: the generated
notebook failed ``nbformat`` validation at conversion, so no packaged code ran,
no guard fired and no heartbeat was printed. A whole GPU session bought a
stack trace (D-0026, E03-SMOKE recorded invalid).

So the checks Kaggle would have applied are applied here first, with the same
libraries, and the entry point is exercised locally until it proves it can
speak. Every check runs before any network call, and the first failure stops the
build rather than being reported alongside a push that already happened.

Consumer: ``biohubx package kaggle``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

REQUIRED_KERNELSPEC_FIELDS = ("display_name", "language", "name")
REQUIRED_CELL_FIELDS = ("id", "cell_type", "metadata", "source")
STAGE_MARKER = "BIOHUBX_STAGE"


class PrePushError(RuntimeError):
    """A pre-push check failed. Nothing is sent."""


@dataclass(frozen=True, slots=True)
class PrePushReport:
    """What was checked, so a passing build records more than that it passed."""

    nbformat_validated: bool
    nbconvert_converted: bool
    converted_bytes: int
    metadata_fields_present: bool
    entry_point_exercised: bool
    stage_lines_seen: int
    stage_names: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nbformat_validated": self.nbformat_validated,
            "nbconvert_converted": self.nbconvert_converted,
            "converted_bytes": self.converted_bytes,
            "metadata_fields_present": self.metadata_fields_present,
            "entry_point_exercised": self.entry_point_exercised,
            "stage_lines_seen": self.stage_lines_seen,
            "stage_names": list(self.stage_names),
            "all_passed": self.passed,
        }

    @property
    def passed(self) -> bool:
        return (
            self.nbformat_validated
            and self.nbconvert_converted
            and self.metadata_fields_present
            and self.entry_point_exercised
            and self.stage_lines_seen > 0
        )


def check_required_metadata(notebook: dict[str, Any]) -> None:
    """The fields whose absence killed the first run, checked by name.

    ``nbformat.validate`` covers these too. They are asserted separately anyway,
    because a validator that changes its mind between versions should not be the
    only thing standing between a build and a wasted GPU session.
    """
    kernelspec = notebook.get("metadata", {}).get("kernelspec", {})
    missing = [name for name in REQUIRED_KERNELSPEC_FIELDS if not kernelspec.get(name)]
    if missing:
        raise PrePushError(f"kernelspec is missing {missing}; this is what failed E03-SMOKE")
    cells = notebook.get("cells", [])
    if not cells:
        raise PrePushError("the notebook has no cells")
    for index, cell in enumerate(cells):
        absent = [name for name in REQUIRED_CELL_FIELDS if name not in cell or cell.get(name) is None]
        if absent:
            raise PrePushError(f"cell {index} is missing {absent}")


def validate_notebook(notebook: dict[str, Any]) -> None:
    """Run ``nbformat.validate`` exactly as the conversion step does."""
    try:
        import nbformat
    except ImportError as exc:  # pragma: no cover - the group is required to build
        raise PrePushError(
            "nbformat is not installed; `uv sync --group package-gate` before building a package"
        ) from exc

    node = nbformat.from_dict(notebook)  # type: ignore[no-untyped-call]
    try:
        nbformat.validate(node)
    except Exception as exc:
        raise PrePushError(f"nbformat rejected the generated notebook: {exc}") from exc


def convert_notebook(notebook: dict[str, Any]) -> int:
    """Convert through nbconvert's HTML exporter, the path Kaggle's error came from.

    Returns the size of the converted document. Conversion is what actually
    failed on Kaggle: validation happens inside it, and by the time it raises
    the session is already spent.
    """
    try:
        import nbformat
        from nbconvert import HTMLExporter
    except ImportError as exc:  # pragma: no cover - the group is required to build
        raise PrePushError(
            "nbconvert is not installed; `uv sync --group package-gate` before building a package"
        ) from exc

    written: str = nbformat.writes(nbformat.from_dict(notebook))  # type: ignore[no-untyped-call]
    node = nbformat.read(io.StringIO(written), as_version=4)  # type: ignore[no-untyped-call]
    try:
        body, _ = HTMLExporter().from_notebook_node(node)  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise PrePushError(f"nbconvert could not convert the generated notebook: {exc}") from exc
    return len(body)


def exercise_entry_point(spec: dict[str, Any], *, data_root: Any) -> tuple[int, tuple[str, ...]]:
    """Run the packaged entry point and require it to announce itself.

    A run that produces no heartbeat is indistinguishable from a run that never
    started, which is precisely the failure this gate exists to prevent. The
    entry point must emit at least one stage line before the build may proceed.
    """
    import contextlib

    from biohubx.packaging.entry import run_fold

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        run_fold(spec, data_root=data_root)
    lines = [line for line in captured.getvalue().splitlines() if line.startswith(STAGE_MARKER)]
    if not lines:
        raise PrePushError(
            "the packaged entry point produced no BIOHUBX_STAGE line, so a real run could fail "
            "silently and look identical to one that never started"
        )
    names = tuple(dict.fromkeys(line.split()[1] for line in lines if len(line.split()) > 1))
    return len(lines), names


def run_all(
    notebook: dict[str, Any],
    spec: dict[str, Any],
    *,
    data_root: Any,
) -> PrePushReport:
    """Every check, in the order that fails cheapest first. Raises on the first failure."""
    check_required_metadata(notebook)
    validate_notebook(notebook)
    converted = convert_notebook(notebook)
    stage_lines, stage_names = exercise_entry_point(spec, data_root=data_root)
    return PrePushReport(
        nbformat_validated=True,
        nbconvert_converted=True,
        converted_bytes=converted,
        metadata_fields_present=True,
        entry_point_exercised=True,
        stage_lines_seen=stage_lines,
        stage_names=stage_names,
    )
