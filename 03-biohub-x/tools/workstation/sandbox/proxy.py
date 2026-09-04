"""Forward one published localhost port into the internal sandbox network.

Docker's internal networks have no route out, which is exactly what the
compute sandbox wants, but such a network cannot publish a port either. This
forwarder runs in a second container that sits on both the internal network and
the default bridge, and copies bytes between a listening socket and the Jupyter
container. It forwards inbound connections only; nothing inside the sandbox can
use it to reach out, because it never opens a connection on the sandbox's
behalf except to the fixed upstream it was started with.

Standard library only. Usage: python proxy.py LISTEN_HOST LISTEN_PORT UPSTREAM_HOST UPSTREAM_PORT
"""

from __future__ import annotations

import asyncio
import contextlib
import sys


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        with contextlib.suppress(Exception):
            writer.close()


def make_handler(upstream_host: str, upstream_port: int):
    async def handle(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(upstream_host, upstream_port)
        except OSError:
            client_writer.close()
            return
        await asyncio.gather(pipe(client_reader, upstream_writer), pipe(upstream_reader, client_writer))

    return handle


async def main(argv: list[str]) -> None:
    listen_host, listen_port, upstream_host, upstream_port = argv[0], int(argv[1]), argv[2], int(argv[3])
    server = await asyncio.start_server(make_handler(upstream_host, upstream_port), listen_host, listen_port)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
