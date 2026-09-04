"""Write tools/workstation/manifest.yaml from what is actually installed.

Every installed tool carries a digest computed here rather than typed: a
canonicalization-v1 tree digest of its package directory for Python
distributions, the raw SHA-256 of the binary for the GitHub server, the SHA-256
of package-lock.json for the npm prefix. Planned and disabled tools carry their
pin and ledger entry and no installed digest, so a launcher that looks for one
refuses until the install has happened.

Run with the repository's own interpreter (it imports biohubx.hashing); it reads
the profile environments and writes nothing into them.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
from biohubx.hashing import tree_digest

TOOLS = Path(os.environ.get("BIOHUBX_TOOLS_ROOT") or REPO.parent / "Biohub-X-tools").resolve()
VISUAL_PY = TOOLS / "biohub-visual" / "Scripts" / "python.exe"
RESEARCH_PY = TOOLS / "biohub-research" / "py" / "Scripts" / "python.exe"
COMPUTE_PY = TOOLS / "biohub-compute" / "py" / "Scripts" / "python.exe"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def site_packages(python: Path) -> Path:
    out = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


def version_of(python: Path, dist: str) -> str:
    out = subprocess.run(
        [str(python), "-c", f"import importlib.metadata as m; print(m.version({dist!r}))"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def python_tool(python: Path, dist: str, package_dir: str) -> list[str]:
    """Version and tree digest lines for one installed distribution."""
    path = site_packages(python) / package_dir
    lines = [f"    version: {version_of(python, dist)}"]
    if path.is_dir():
        manifest = tree_digest(path)
        lines += [
            f"    installed_digest: {manifest.digest.token}",
            f"    installed_path: site-packages/{package_dir}",
            f"    installed_files: {manifest.file_count}",
            f"    installed_bytes: {manifest.total_bytes}",
        ]
    return lines


# (id, profile, distribution, package dir, source, licence, ledger, capabilities, consumer)
VISUAL = [
    (
        "napari",
        "napari",
        "napari",
        "https://github.com/napari/napari",
        "BSD-3-Clause",
        "RL-0024",
        "n-dimensional viewer: image, points, labels, shapes, tracks layers; orthogonal slices and 3D; screenshots",
        "viewer for E05/E06 error analysis",
    ),
    (
        "pyqt6",
        "PyQt6",
        "PyQt6",
        "https://www.riverbankcomputing.com/software/pyqt/",
        "GPL-3.0-only (workstation use, not distributed)",
        "RL-0024 (napari pyqt6 extra)",
        "Qt6 bindings napari renders through",
        "napari backend",
    ),
    (
        "napari-mcp",
        "napari-mcp",
        "napari_mcp",
        "https://github.com/royerlab/napari-mcp",
        "BSD-3-Clause",
        "RL-0025, RL-0039, RL-0044, RL-0045",
        "MCP server over a live viewer: init/close viewer, add and configure layers, camera, screenshot, session info; "
        "install_packages removed, execute_code gated (launch_napari_mcp.py)",
        "Claude Code visual sessions",
    ),
    (
        "napari-geff",
        "napari-geff",
        "napari_geff",
        "https://github.com/live-image-tracking-tools/napari-geff",
        "BSD-3-Clause",
        "RL-0026, RL-0040",
        "reads a GEFF store into napari tracks/points layers and writes tracks back to GEFF",
        "loading ground truth and predicted graphs as layers",
    ),
    (
        "motile-tracker",
        "motile-tracker",
        "motile_tracker",
        "https://github.com/funkelab/motile_tracker",
        "BSD-3-Clause",
        "RL-0027, RL-0041",
        "interactive track editing, lineage view, Motile ILP solving inside napari",
        "inspecting and hand-editing links to form probes",
    ),
    (
        "motile",
        "motile",
        "motile",
        "https://github.com/funkelab/motile",
        "MIT",
        "RL-0028",
        "ILP tracking on candidate graphs via ilpy",
        "motile-tracker's solver",
    ),
    (
        "geff",
        "geff",
        "geff",
        "https://github.com/live-image-tracking-tools/geff",
        "MIT",
        "RL-0029",
        "graph exchange file format reader and writer",
        "same pin as the repository",
    ),
    (
        "tracksdata",
        "tracksdata",
        "tracksdata",
        "https://github.com/royerlab/tracksdata",
        "BSD-3-Clause upstream; Biohub-X build (R-0014)",
        "RL-0023, R-0014",
        "graph representation used by the official scorer",
        "reading graphs the way the scorer does; installed from artifacts/wheelhouse-metric with hashes",
    ),
]

RESEARCH_PY_TOOLS = [
    (
        "semantic-scholar-mcp",
        "semantic-scholar-mcp",
        "semantic_scholar_mcp",
        "https://github.com/hy20191108/semantic-scholar-mcp",
        "MIT",
        "RL-0032",
        "Semantic Scholar search, paper details, citations and references over MCP; community server, unauthenticated by default",
        "citation expansion and convergence checks; every acquired source still enters the research ledger",
    ),
    (
        "mcp-client",
        "mcp",
        "mcp",
        "https://github.com/modelcontextprotocol/python-sdk",
        "MIT",
        "dependency of RL-0032",
        "MCP client used by validate_research.py to drive the profile's servers over stdio",
        "validation only",
    ),
]

COMPUTE_PY_TOOLS = [
    (
        "jupyter-mcp-server",
        "jupyter-mcp-server",
        "jupyter_mcp_server",
        "https://github.com/datalayer/jupyter-mcp-server",
        "BSD-3-Clause",
        "RL-0033",
        "notebook and kernel control over MCP against a local Jupyter server",
        "multi-call diagnostics and plots before a biohubx command exists",
    ),
]

DISABLED = [
    (
        "traccuracy",
        "biohub-crosscheck",
        "https://github.com/live-image-tracking-tools/traccuracy",
        "0.4.3",
        "BSD-3-Clause",
        "RL-0034",
        "CTC-style edge, division and track metrics with error localisation",
        "diagnostics after an official-score run; never the promotion authority",
    ),
    (
        "bioio",
        "biohub-crosscheck",
        "https://github.com/bioio-devs/bioio",
        "3.5.0",
        "BSD-3-Clause",
        "RL-0035",
        "microscopy format reading with explicit TCZYX normalisation",
        "an external dataset in a format the canonical reader does not support",
    ),
    (
        "ngff-zarr-mcp",
        "biohub-crosscheck",
        "https://github.com/thewtex/ngff-zarr",
        "0.14.0 over ngff-zarr 0.45.0",
        "MIT",
        "RL-0036, RL-0037",
        "OME-NGFF metadata inspection, conformance and conversion over MCP",
        "genuine OME-NGFF questions; not a replacement for the competition reader",
    ),
    (
        "fiji-mcp",
        "biohub-crosscheck",
        "not selected",
        "none; no Java runtime on this machine",
        "",
        "",
        "independent image-processing implementation",
        "an image-processing conclusion needing an implementation independent of the Python stack",
    ),
    (
        "bioimage-io",
        "biohub-crosscheck",
        "https://bioimage.io",
        "search only, no client installed",
        "",
        "",
        "public microscopy model zoo",
        "reusable public models; source, licences, training data, preprocessing, axes, scale and eligibility "
        "recorded before any download",
    ),
]


def record(tool_id: str, profile: str, status: str, fields: list[str]) -> list[str]:
    return [f"  - id: {tool_id}", f"    profile: {profile}", f"    status: {status}", *fields]


def main() -> None:
    visual_ok = VISUAL_PY.is_file()
    research_ok = RESEARCH_PY.is_file()
    compute_ok = COMPUTE_PY.is_file()
    research = TOOLS / "biohub-research"
    binary = research / "github-mcp-server" / "github-mcp-server.exe"
    npm_lock = research / "package-lock.json"

    lines = [
        "# Workstation tools, one record per tool, written by tools/workstation/write_manifest.py",
        "# from the installed environments before a tool is enabled (D-0043). installed_digest is",
        "# a canonicalization-v1 tree digest of the package directory for Python distributions,",
        "# the raw SHA-256 of the binary for the GitHub server, and the SHA-256 of package-lock.json",
        "# for the npm prefix. Planned and disabled tools carry none.",
        "schema_version: 2",
        "authorised_by: Arya Arun, 2026-09-04",
        "environments_root: BIOHUBX_TOOLS_ROOT, or a Biohub-X-tools directory beside the repository",
        "profiles:",
        f"  biohub-visual: {'enabled' if visual_ok else 'planned'}",
        f"  biohub-research: {'enabled' if research_ok and binary.is_file() and npm_lock.is_file() else 'planned'}",
        f"  biohub-compute: {'enabled' if compute_ok else 'planned'}",
        "  biohub-crosscheck: disabled",
        "tools:",
    ]

    for tool_id, dist, package_dir, source, licence, ledger, caps, consumer in VISUAL:
        fields = [
            f"    distribution: {dist}",
            f"    source: {source}",
            f"    licence: {licence}",
            f"    ledger: {ledger}",
        ]
        fields.append(
            "    lock: artifacts/wheelhouse-metric/requirements-offline.txt, tracksdata line, --require-hashes"
            if tool_id == "tracksdata"
            else "    lock: tools/workstation/requirements/biohub-visual.lock.txt"
        )
        if visual_ok:
            fields += python_tool(VISUAL_PY, dist, package_dir)
        fields += [f"    capabilities: {caps}", f"    consumer: {consumer}"]
        lines += record(tool_id, "biohub-visual", "enabled" if visual_ok else "planned", fields)

    fields = [
        "    source: https://github.com/github/github-mcp-server",
        "    release: v1.12.0, commit 9205304fedf10540c33ae41fcf6352fa97dcc9be, Windows x86_64 zip",
        "    release_zip_sha256: bc8782deda12dc1f36a182d0205bb4f11b05aee0eec7e2c8c109bb8f622fb4b4",
        "    licence: MIT",
        "    ledger: RL-0031, RL-0038, RL-0042, RL-0043, RL-0046, RL-0047",
    ]
    if binary.is_file():
        fields += [
            f"    installed_digest: raw_artifact_sha256:sha256:{sha256(binary)}",
            "    installed_path: biohub-research/github-mcp-server/github-mcp-server.exe",
            f"    installed_bytes: {binary.stat().st_size}",
        ]
    fields += [
        "    launch: tools/workstation/launch_github_mcp.py, always stdio --read-only, toolsets repos,issues; "
        "token from gh auth token at launch, never on disk",
        "    capabilities: GitHub API over MCP, read-only; repositories, file contents, commits, releases, issues",
        "    consumer: exact upstream commits, issues, releases, licences and history behind a claim",
    ]
    lines += record(
        "github-mcp-server", "biohub-research", "enabled" if binary.is_file() else "planned", fields
    )

    fields = [
        "    source: https://github.com/microsoft/playwright-mcp",
        "    pin: '@playwright/mcp 0.0.80 (npm), --save-exact under the biohub-research prefix'",
        "    licence: Apache-2.0",
        "    ledger: RL-0030",
    ]
    if npm_lock.is_file():
        pkg = json.loads(
            (research / "node_modules" / "@playwright" / "mcp" / "package.json").read_text(encoding="utf-8")
        )
        core = json.loads(
            (research / "node_modules" / "playwright-core" / "package.json").read_text(encoding="utf-8")
        )
        browsers = json.loads(
            (research / "node_modules" / "playwright-core" / "browsers.json").read_text(encoding="utf-8")
        )
        chromium = ", ".join(
            f"{b['name']} r{b['revision']} ({b.get('browserVersion', '?')})"
            for b in browsers["browsers"]
            if b["name"] in ("chromium", "chromium-headless-shell")
        )
        fields += [
            f"    version: {pkg['version']}",
            f"    playwright_core: {core['version']}",
            f"    installed_digest: raw_artifact_sha256:sha256:{sha256(npm_lock)}",
            "    installed_path: biohub-research/package-lock.json",
            f"    browsers: {chromium}, under biohub-research/browsers (PLAYWRIGHT_BROWSERS_PATH)",
        ]
    fields += [
        "    capabilities: browser automation over MCP; navigate, snapshot, screenshot, click, evaluate; isolated profile",
        "    consumer: dynamically rendered pages and browser state as evidence; CLI for bounded bulk collection",
    ]
    lines += record(
        "playwright-mcp", "biohub-research", "enabled" if npm_lock.is_file() else "planned", fields
    )

    for tool_id, dist, package_dir, source, licence, ledger, caps, consumer in RESEARCH_PY_TOOLS:
        fields = [
            f"    distribution: {dist}",
            f"    source: {source}",
            f"    licence: {licence}",
            f"    ledger: {ledger}",
            "    lock: tools/workstation/requirements/biohub-research.lock.txt",
        ]
        if research_ok:
            fields += python_tool(RESEARCH_PY, dist, package_dir)
        fields += [f"    capabilities: {caps}", f"    consumer: {consumer}"]
        lines += record(tool_id, "biohub-research", "enabled" if research_ok else "planned", fields)

    for tool_id, dist, package_dir, source, licence, ledger, caps, consumer in COMPUTE_PY_TOOLS:
        fields = [
            f"    distribution: {dist}",
            f"    source: {source}",
            f"    licence: {licence}",
            f"    ledger: {ledger}",
            "    lock: tools/workstation/requirements/biohub-compute.lock.txt",
        ]
        if compute_ok:
            fields += python_tool(COMPUTE_PY, dist, package_dir)
        fields += [f"    capabilities: {caps}", f"    consumer: {consumer}"]
        lines += record(tool_id, "biohub-compute", "enabled" if compute_ok else "planned", fields)

    for tool_id, profile, source, pin, licence, ledger, caps, consumer in DISABLED:
        lines += record(
            tool_id,
            profile,
            "disabled",
            [
                f"    source: {source}",
                f"    pin: {pin}",
                f"    licence: {licence}",
                f"    ledger: {ledger}",
                f"    capabilities: {caps}",
                f"    consumer: {consumer}",
            ],
        )

    (HERE / "manifest.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(
        f"manifest written: visual={'enabled' if visual_ok else 'planned'} "
        f"research={'enabled' if research_ok and binary.is_file() else 'planned'} "
        f"compute={'enabled' if compute_ok else 'planned'}"
    )


if __name__ == "__main__":
    main()
