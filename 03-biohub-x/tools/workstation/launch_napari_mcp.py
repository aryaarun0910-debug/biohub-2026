"""Start napari-mcp with its package installer removed and code execution off by default.

napari-mcp 0.1.0 registers `install_packages` (pip inside the running server)
and `execute_code` (arbitrary Python in the server process). The workstation
authorisation forbids dynamic package installation outright and allows code
execution only for a reviewed, bounded snippet whose inputs and output path
are declared. So `install_packages` is removed unconditionally here, and
`execute_code` is removed unless BIOHUBX_NAPARI_EXECUTE=1 is set for that
declared session. Everything else napari-mcp offers (viewer, layers, camera,
screenshot, session information) is left as shipped.

`--list-tools` prints the tool names the server would expose and exits, which
is how the validation record proves the removals rather than asserting them.

Runs inside the biohub-visual environment only. Not part of biohubx.
"""

from __future__ import annotations

import asyncio
import os
import sys

from napari_mcp.server import ServerState, create_server

ALWAYS_REMOVED = ("install_packages",)
GATED = ("execute_code",)


def build():
    """Create the server, then unregister the tools this profile forbids.

    FastMCP 4 keeps registered tools on the server's local provider, and
    ``remove_tool`` there is the supported way to unregister one; the top-level
    ``FastMCP`` object has no such method. A name that is already absent raises
    KeyError, which is left to propagate: it means napari-mcp renamed the tool
    and the launcher can no longer prove what it removed.
    """
    server = create_server(ServerState())
    provider = server._local_provider
    removed = []
    for name in ALWAYS_REMOVED:
        provider.remove_tool(name)
        removed.append(name)
    if os.environ.get("BIOHUBX_NAPARI_EXECUTE") != "1":
        for name in GATED:
            provider.remove_tool(name)
            removed.append(name)
    return server, removed


async def tool_names(server) -> list[str]:
    return sorted(tool.name for tool in await server.list_tools())


def main(argv: list[str]) -> int:
    server, removed = build()
    if "--list-tools" in argv:
        names = asyncio.run(tool_names(server))
        print("removed:", ",".join(removed))
        print("exposed:", ",".join(names))
        present = [name for name in removed if name in names]
        if present:
            print("ERROR: still exposed:", ",".join(present), file=sys.stderr)
            return 1
        return 0
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
