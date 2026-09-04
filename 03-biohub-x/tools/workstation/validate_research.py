"""Drive every server of the biohub-research profile over stdio and record what each exposes.

Objective: prove the three servers start from their pinned installations, answer
an MCP initialize and tools/list, and that the GitHub server exposes no write
tool. Falsifier: any server that fails to start or list tools, or a GitHub tool
whose name begins with a mutating verb.

Runs inside biohub-research/py, which carries the `mcp` client library as a
dependency of the Semantic Scholar server. Standard library plus `mcp`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from activate import REPO, substitute, tools_root

PROFILE = json.loads((HERE / "profiles" / "biohub-research.json").read_text(encoding="utf-8"))
TOOLS = tools_root(os.environ.get("BIOHUBX_TOOLS_ROOT"))
OUT = REPO / "artifacts" / "tool-sessions"
MUTATING = (
    "create_",
    "update_",
    "delete_",
    "merge_",
    "push_",
    "add_",
    "remove_",
    "dismiss_",
    "submit_",
    "request_",
    "assign_",
    "fork_",
    "star_",
    "unstar_",
    "close_",
    "reopen_",
    "edit_",
    "set_",
    "write_",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def probe(name: str, server: dict, call: tuple[str, dict] | None) -> dict:
    env = dict(os.environ)
    env.update(server.get("env") or {})
    params = StdioServerParameters(command=server["command"], args=list(server.get("args") or []), env=env)
    started = time.time()
    record: dict = {"command": server["command"], "args": list(server.get("args") or [])}
    try:
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            init = await session.initialize()
            record["server_info"] = {
                "name": getattr(init.serverInfo, "name", None),
                "version": getattr(init.serverInfo, "version", None),
                "protocol": str(init.protocolVersion),
            }
            listed = await session.list_tools()
            names = sorted(tool.name for tool in listed.tools)
            record["tools"] = names
            record["tool_count"] = len(names)
            if call is not None:
                tool, arguments = call
                if tool in names:
                    result = await session.call_tool(tool, arguments)
                    text = "".join(getattr(c, "text", "") for c in result.content)[:600]
                    record["call"] = {
                        "tool": tool,
                        "arguments": arguments,
                        "is_error": bool(result.isError),
                        "text_head": text,
                    }
                else:
                    record["call"] = {"tool": tool, "skipped": "not exposed"}
        record["ok"] = True
    except BaseException as exc:
        record["ok"] = False
        record["error"] = repr(exc)[:600]
    record["elapsed_seconds"] = round(time.time() - started, 3)
    return record


async def main() -> int:
    servers = substitute(PROFILE["mcpServers"], REPO, TOOLS)
    research = TOOLS / "biohub-research"
    calls = {
        "github": ("get_file_contents", {"owner": "github", "repo": "github-mcp-server", "path": "LICENSE"}),
        "playwright": ("browser_navigate", {"url": "about:blank"}),
        "semantic-scholar": None,
    }
    # Validation runs the browser headless; the profile itself leaves it visible.
    servers["playwright"]["args"] = [*servers["playwright"]["args"], "--headless"]
    record: dict = {
        "schema_version": 1,
        "session_id": "WS-RESEARCH-01",
        "profile": "biohub-research",
        "provenance_status": "integration_only",
        "objective": __doc__.split("\n\n")[1].strip(),
        "read_scope": "GitHub API (one LICENSE read), about:blank in a headless Chromium, no Semantic Scholar query",
        "write_scope": str(OUT),
        "host": {"python": sys.version.split()[0], "platform": platform.platform()},
        "installed": {},
        "servers": {},
    }
    binary = research / "github-mcp-server" / "github-mcp-server.exe"
    record["installed"]["github-mcp-server"] = {
        "path": str(binary),
        "sha256": sha256(binary),
        "bytes": binary.stat().st_size,
    }
    lock = research / "package-lock.json"
    pkg = json.loads(
        (research / "node_modules" / "@playwright" / "mcp" / "package.json").read_text(encoding="utf-8")
    )
    core = json.loads(
        (research / "node_modules" / "playwright-core" / "package.json").read_text(encoding="utf-8")
    )
    browsers = json.loads(
        (research / "node_modules" / "playwright-core" / "browsers.json").read_text(encoding="utf-8")
    )
    chromium = [b for b in browsers["browsers"] if b["name"] in ("chromium", "chromium-headless-shell")]
    record["installed"]["playwright-mcp"] = {
        "version": pkg["version"],
        "license": pkg.get("license"),
        "playwright_core": core["version"],
        "package_lock_sha256": sha256(lock),
        "chromium": [
            {"name": b["name"], "revision": b["revision"], "browserVersion": b.get("browserVersion")}
            for b in chromium
        ],
        "browsers_dir": sorted(p.name for p in (research / "browsers").iterdir())
        if (research / "browsers").is_dir()
        else [],
    }
    for name, server in servers.items():
        record["servers"][name] = await probe(name, server, calls.get(name))
    github_tools = record["servers"].get("github", {}).get("tools", [])
    mutating = [t for t in github_tools if t.startswith(MUTATING)]
    record["github_read_only"] = {
        "mutating_tools_exposed": mutating,
        "ok": not mutating and bool(github_tools),
    }
    record["ok"] = all(s.get("ok") for s in record["servers"].values()) and record["github_read_only"]["ok"]
    record["unresolved"] = (
        "Semantic Scholar was started and listed, not queried: a query is an acquisition and belongs in the "
        "research ledger with a note, not in a validation. The Playwright call opened about:blank only."
    )
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "WS-RESEARCH-01.json"
    target.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    for name, s in record["servers"].items():
        print(
            f"{name:17} ok={s.get('ok')} tools={s.get('tool_count')} info={s.get('server_info')} err={s.get('error', '')[:160]}"
        )
    print("github mutating tools exposed:", mutating)
    print("overall ok:", record["ok"], "->", target)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
