"""Run the pinned GitHub MCP server binary read-only, with a token that never touches disk.

The binary is the release recorded in tools/workstation/manifest.yaml, verified
against its digest here before every launch. The token comes from the already
authenticated GitHub CLI (`gh auth token`) and is placed in the child's
environment only; nothing is written, printed or logged. `--read-only` is
always passed, whatever the caller asks for, because read-only mode is the
whole reason this profile exists.

Standard library only. Usage from a profile: python launch_github_mcp.py
[--toolsets repos,issues] [further github-mcp-server flags].
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.yaml"


def manifest_digest(tool_id: str) -> str:
    """Read one tool's installed digest from the manifest without a YAML library.

    The manifest is YAML written by our own script with one `installed_digest:`
    line per tool, so a line scan under the tool's id is enough and keeps this
    launcher dependency-free.
    """
    current = None
    for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("- id: "):
            current = line.split(":", 1)[1].strip()
        elif current == tool_id and line.startswith("installed_digest:"):
            return line.split(":", 1)[1].strip().removeprefix("raw_artifact_sha256:sha256:")
    raise SystemExit(f"{tool_id} has no installed_digest in {MANIFEST}")


def main(argv: list[str]) -> int:
    tools_root = Path(
        os.environ.get("BIOHUBX_TOOLS_ROOT") or HERE.parent.parent.parent / "Biohub-X-tools"
    ).resolve()
    binary = tools_root / "biohub-research" / "github-mcp-server" / "github-mcp-server.exe"
    if not binary.is_file():
        print(f"github-mcp-server is not installed at {binary}", file=sys.stderr)
        return 2
    expected = manifest_digest("github-mcp-server")
    observed = hashlib.sha256(binary.read_bytes()).hexdigest()
    if observed != expected:
        print("refusing to launch: the binary does not match the manifest digest", file=sys.stderr)
        return 2
    token = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=False)
    if token.returncode != 0 or not token.stdout.strip():
        print(
            "refusing to launch: `gh auth token` returned nothing; authenticate the GitHub CLI first",
            file=sys.stderr,
        )
        return 2
    env = dict(os.environ)
    env["GITHUB_PERSONAL_ACCESS_TOKEN"] = token.stdout.strip()
    env["GITHUB_READ_ONLY"] = "1"
    args = [str(binary), "stdio", "--read-only", *[a for a in argv if a != "--read-only"]]
    # stdio is inherited: the MCP client speaks to the binary directly.
    completed = subprocess.run(args, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
