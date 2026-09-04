"""The workstation manifest has to be readable by something other than a person.

[[D-0043]] makes ``tools/workstation/manifest.yaml`` the record consulted before
a tool is enabled: source, pin, licence, installed digest, capabilities,
consumer and status. It was assembled as text with its free-text fields emitted
bare, so every capability line carrying a colon and a space made the file
unparseable, and nothing had ever read it back. These tests are the thing that
would have noticed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "tools/workstation/manifest.yaml"
PROFILES = REPO_ROOT / "tools/workstation/profiles"


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    body: dict[str, Any] = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    return body


def test_the_manifest_parses_as_yaml(manifest: dict[str, Any]) -> None:
    """The regression. A record nothing can read is not a record."""
    assert manifest["schema_version"] == 2
    assert manifest["tools"]
    assert manifest["profiles"]


def test_every_tool_carries_what_a_tool_must_carry_before_it_is_enabled(
    manifest: dict[str, Any],
) -> None:
    for tool in manifest["tools"]:
        for field in ("id", "profile", "status", "source", "licence", "capabilities", "consumer"):
            assert field in tool, f"{tool.get('id')} has no {field}"
        assert tool["status"] in {"enabled", "disabled", "planned"}, tool["id"]


def test_every_declared_profile_has_a_profile_file_that_agrees_about_its_status(
    manifest: dict[str, Any],
) -> None:
    """A profile the manifest calls disabled and the JSON calls enabled is one
    `activate.py` would switch on while the record says it is off."""
    for name, status in manifest["profiles"].items():
        path = PROFILES / f"{name}.json"
        assert path.is_file(), f"{name} is declared and has no profile file"
        body = json.loads(path.read_text(encoding="utf-8"))
        assert body["profile"] == name
        if status == "enabled":
            assert body["status"] == "enabled", name
        else:
            assert body["status"] != "enabled", name


def test_every_tool_belongs_to_a_declared_profile(manifest: dict[str, Any]) -> None:
    declared = set(manifest["profiles"])
    for tool in manifest["tools"]:
        assert tool["profile"] in declared, f"{tool['id']} names undeclared profile {tool['profile']}"


def test_a_tool_in_a_disabled_profile_is_not_itself_enabled(manifest: dict[str, Any]) -> None:
    disabled = {name for name, status in manifest["profiles"].items() if status != "enabled"}
    for tool in manifest["tools"]:
        if tool["profile"] in disabled:
            assert tool["status"] != "enabled", f"{tool['id']} is enabled inside a disabled profile"


def test_the_colab_profile_is_disabled_and_pinned_to_an_exact_commit(manifest: dict[str, Any]) -> None:
    """Hosted exploration is off until Arya Arun turns it on, and when it is
    turned on it is a named commit rather than whatever main happens to be."""
    assert manifest["profiles"]["biohub-colab-explore"] == "disabled"
    tool = next(item for item in manifest["tools"] if item["id"] == "colab-mcp")
    assert tool["status"] == "disabled"
    assert tool["source"] == "https://github.com/googlecolab/colab-mcp"
    assert "b85ab6ec5206e06fdd289ff4ff3f9c4ee767b422" in tool["pin"]

    profile = json.loads((PROFILES / "biohub-colab-explore.json").read_text(encoding="utf-8"))
    assert profile["status"] == "disabled"
    pin = profile["mcpServers"]["colab"]["pin"]
    assert len(pin["commit"]) == 40
    assert pin["commit"] == "b85ab6ec5206e06fdd289ff4ff3f9c4ee767b422"
    assert pin["licence"] == "Apache-2.0"
