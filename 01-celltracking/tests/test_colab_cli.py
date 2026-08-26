"""Software contracts for scripts/core/colab_cli.py and colab_vm_bootstrap.py (official Colab CLI lane).

No WSL, no network: the `colab` wrapper is monkeypatched. What is tested is the plumbing that has
already bitten us once - session detection (a miss allocated a second billable L4), path mapping
across the Windows/WSL boundary, the poll snippet, and unit accounting.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "core"))
import colab_relay as relay  # noqa: E402

_spec = importlib.util.spec_from_file_location("colab_cli", ROOT / "scripts" / "core" / "colab_cli.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def test_to_wsl_maps_drive_letters():
    assert cli.to_wsl("C:/temp/colab_cli/x.sh") == "/mnt/c/temp/colab_cli/x.sh"
    assert cli.to_wsl(r"D:\data\a b\f.txt") == "/mnt/d/data/a b/f.txt"


def test_clean_strips_ansi_and_box_drawing():
    raw = "\x1b[1m[colab]\x1b[0m Session READY.\n\u2502 padded \u2502\n\n\u2500\u2500\u2500\n"
    assert cli.clean(raw).splitlines() == ["[colab] Session READY.", " padded"]


def test_session_exists_matches_owned_not_orphans(monkeypatch):
    listing = ("[biohub-l4] gpu-l4-s-kkb-ass1a1-4abvcxconwcy | Hardware: L4 | Variant: GPU\n"
               "[?] gpu-l4-s-kkb-usw4c0-2ntn6u0fuslmc | Hardware: L4 | Variant: GPU\n")
    monkeypatch.setattr(cli, "colab", lambda args, **kw: SimpleNamespace(stdout=listing, stderr="", returncode=0))
    assert cli.session_exists("biohub-l4")
    assert not cli.session_exists("biohub")          # prefix must not match
    assert not cli.session_exists("gpu-l4-s-kkb-usw4c0-2ntn6u0fuslmc")   # orphans are not adoptable


def test_poll_snippet_is_valid_python_and_parses(monkeypatch):
    code = cli.POLL_SNIPPET.format(job="p21-parity-f0-a6", n=12)
    ast.parse(code)
    payload = {"state": "running", "exit_code": None, "alive": True, "started": "x", "tail": "line"}
    monkeypatch.setattr(cli, "exec_py", lambda session, code, timeout_s: "noise\nPOLLJSON " + json.dumps(payload) + "\n")
    assert cli.poll_once("p21-parity-f0-a6", "biohub-l4") == payload


def test_spent_units_counts_open_sessions_by_wall_clock():
    l = {"spent_units": 1.0, "sessions": {"a": {"gpu": "NVIDIA L4", "started_ts": 1000.0, "open": True},
                                          "b": {"gpu": "T4", "started_ts": 0.0, "open": False}}}
    assert cli.spent_units(l, relay.DEFAULT_RATES, now=1000.0 + 3600.0) == pytest.approx(1.0 + 5.0)


def test_bootstrap_script_parses_and_targets_writable_root():
    src = (ROOT / "scripts" / "core" / "colab_vm_bootstrap.py").read_text(encoding="utf-8")
    ast.parse(src)
    assert "/content/kaggle/input" in src and "start_new_session=True" in src
    assert "PYTHONPATH" in src and "kernel_probe" in src
