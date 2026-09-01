from __future__ import annotations

import ast
from pathlib import Path


BIOHUBX_ROOT = Path("scripts/win_bet/biohubx")
FORBIDDEN_RUNTIME_PREFIXES = (
    "scripts.win_bet.assoc_train_harness",
    "scripts.win_bet.assoc_tournament",
    "scripts.win_bet.assoc_fold_pathology",
    "scripts.win_bet.finaledge_gates",
    "scripts.win_bet.p28_full_chain_replay",
    "scripts.win_bet.gpu_protection_contract",
    "scripts.win_bet.spec_restriction",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_biohubx_does_not_import_the_live_gate_b_or_provenance_surfaces():
    """Make the temporary red-suite exemption an import-boundary fact, not a judgement."""
    offenders: dict[str, list[str]] = {}
    for path in sorted(BIOHUBX_ROOT.glob("*.py")):
        blocked = sorted(
            name
            for name in _imports(path)
            if any(name == prefix or name.startswith(prefix + ".")
                   for prefix in FORBIDDEN_RUNTIME_PREFIXES)
        )
        if blocked:
            offenders[path.as_posix()] = blocked
    assert offenders == {}

