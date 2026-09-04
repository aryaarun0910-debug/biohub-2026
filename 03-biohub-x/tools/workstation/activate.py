"""Write the repository's `.mcp.json` from one workstation profile, or remove it.

Profiles live in tools/workstation/profiles/<name>.json with two placeholders,
``${BIOHUBX_REPO}`` and ``${BIOHUBX_TOOLS}``, so no tracked file carries an
absolute path. The written `.mcp.json` is ignored by Git and read by the next
Claude Code session. One profile is active at a time; activating another
replaces the file rather than merging into it.

Standard library only. This is a workstation convenience, not part of biohubx.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PROFILES = HERE / "profiles"
TARGET = REPO / ".mcp.json"


def tools_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    env = os.environ.get("BIOHUBX_TOOLS_ROOT")
    if env:
        return Path(env).resolve()
    return (REPO.parent / "Biohub-X-tools").resolve()


def substitute(value, repo: Path, tools: Path):
    if isinstance(value, str):
        return value.replace("${BIOHUBX_REPO}", repo.as_posix()).replace("${BIOHUBX_TOOLS}", tools.as_posix())
    if isinstance(value, list):
        return [substitute(item, repo, tools) for item in value]
    if isinstance(value, dict):
        return {key: substitute(item, repo, tools) for key, item in value.items()}
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--profile", choices=sorted(p.stem for p in PROFILES.glob("*.json")))
    group.add_argument(
        "--none", action="store_true", help="Remove .mcp.json so no workstation server is active."
    )
    group.add_argument("--show", action="store_true", help="Print the active profile, if any.")
    parser.add_argument(
        "--tools-root", help="Where the environments live. Default: ../Biohub-X-tools beside the repo."
    )
    args = parser.parse_args()

    if args.show:
        if not TARGET.exists():
            print("no active profile")
            return 0
        doc = json.loads(TARGET.read_text(encoding="utf-8"))
        print(
            f"active profile: {doc.get('_biohubx_profile', '?')} servers: {sorted(doc.get('mcpServers', {}))}"
        )
        return 0
    if args.none:
        if TARGET.exists():
            TARGET.unlink()
            print(f"removed {TARGET}")
        else:
            print("nothing active")
        return 0

    profile = json.loads((PROFILES / f"{args.profile}.json").read_text(encoding="utf-8"))
    if profile.get("status") != "enabled":
        print(
            f"profile {args.profile} is {profile.get('status')!r}; enable it in the manifest first",
            file=sys.stderr,
        )
        return 2
    tools = tools_root(args.tools_root)
    servers = substitute(profile["mcpServers"], REPO, tools)
    missing = []
    for name, server in servers.items():
        command = server.get("command", "")
        if command and ("/" in command or "\\" in command) and not Path(command).exists():
            missing.append(f"{name}: {command}")
    if missing:
        print("refusing: a server command does not exist yet, install the profile first:", file=sys.stderr)
        for line in missing:
            print("  " + line, file=sys.stderr)
        return 2
    document = {"_biohubx_profile": args.profile, "mcpServers": servers}
    TARGET.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {TARGET} for profile {args.profile}: {sorted(servers)}")
    print("start a new Claude Code session for it to take effect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
