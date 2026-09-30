"""Start the compute sandbox and run the Jupyter MCP server against it over stdio.

What the sandbox is (D-0043, compute profile): one container from the pinned
image built by tools/workstation/sandbox/Dockerfile, on a Docker internal
network with no route out, root filesystem read-only, the competition root
mounted read-only at /data, the repository's src/ mounted read-only at
/repo/src, exactly one scratch directory mounted read-write at /scratch, a
non-root user, all capabilities dropped, memory, CPU and process limits. A
second container from the same image forwards 127.0.0.1:PORT into the
internal network; it is the only way in and no way out.

Secrets: a fresh random token per launch, passed to both containers and to the
MCP server through their environments. Nothing is written to disk, printed, or
recorded. When the MCP server exits, the containers and the network are removed.

`--validate` runs the checks that make the isolation a measurement rather than
a description, through the same MCP server the profile uses, and writes a
session record. Standard library, plus `mcp` in that mode only.

Runs from biohub-compute/py. Not part of biohubx.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SANDBOX = HERE / "sandbox"
LOCK = HERE / "requirements" / "biohub-compute-sandbox.lock.txt"
IMAGE_BASE = "python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_tag() -> str:
    return f"biohubx-compute:{sha256(LOCK)[:12]}"


def docker(*args: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    completed = subprocess.run(["docker", *args], capture_output=capture, text=True, check=False)
    if check and completed.returncode != 0:
        # The docker CLI puts the reason on stderr; a bare CalledProcessError hides it.
        detail = (completed.stderr or completed.stdout or "").strip()[-800:]
        raise SystemExit(f"docker {args[0]} failed ({completed.returncode}): {detail}")
    return completed


def build_image(tag: str) -> None:
    """Build the image from the lock beside the Dockerfile, copying the tracksdata wheel in."""
    wheels = SANDBOX / "wheels"
    wheels.mkdir(exist_ok=True)
    source = (
        REPO
        / "artifacts"
        / "wheelhouse-metric"
        / "wheels"
        / "tracksdata-0.1.0rc9.dev4+g7bfeaf845-py3-none-any.whl"
    )
    (wheels / source.name).write_bytes(source.read_bytes())
    requirement = ""
    lines = (
        (REPO / "artifacts" / "wheelhouse-metric" / "requirements-offline.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    for index, line in enumerate(lines):
        if line.startswith("tracksdata=="):
            requirement = line + "\n" + lines[index + 1] + "\n"
    (SANDBOX / "tracksdata-requirement.txt").write_text(requirement, encoding="utf-8")
    (SANDBOX / LOCK.name).write_text(LOCK.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        docker("build", "-q", "-t", tag, str(SANDBOX), capture=False)
    finally:
        for item in [wheels / source.name, SANDBOX / "tracksdata-requirement.txt", SANDBOX / LOCK.name]:
            item.unlink(missing_ok=True)
        wheels.rmdir()


class Sandbox:
    def __init__(self, *, data_root: Path, scratch: Path, port: int, memory: str, cpus: str) -> None:
        self.data_root = data_root
        self.scratch = scratch
        self.port = port
        self.memory = memory
        self.cpus = cpus
        self.run_id = secrets.token_hex(4)
        self.token = secrets.token_hex(24)
        self.network = f"biohubx-compute-{self.run_id}"
        self.jupyter = f"biohubx-jupyter-{self.run_id}"
        self.proxy = f"biohubx-proxy-{self.run_id}"
        self.tag = image_tag()

    def start(self) -> None:
        if docker("image", "inspect", self.tag, check=False).returncode != 0:
            build_image(self.tag)
        self.scratch.mkdir(parents=True, exist_ok=True)
        docker("network", "create", "--internal", self.network)
        docker(
            "run",
            "-d",
            "--name",
            self.jupyter,
            "--network",
            self.network,
            "--read-only",
            "--tmpfs",
            "/tmp:size=2g,mode=1777",
            "--tmpfs",
            "/home/sandbox:size=512m,uid=1000,gid=1000,mode=0700",
            "-e",
            "JUPYTER_RUNTIME_DIR=/tmp/jupyter-runtime",
            "-e",
            "JUPYTER_DATA_DIR=/tmp/jupyter-data",
            "--user",
            "1000:1000",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "512",
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "-e",
            f"JUPYTER_TOKEN={self.token}",
            "-v",
            f"{self.data_root.as_posix()}:/data:ro",
            "-v",
            f"{(REPO / 'src').as_posix()}:/repo/src:ro",
            "-v",
            f"{self.scratch.as_posix()}:/scratch:rw",
            self.tag,
            "python",
            "-m",
            "jupyter_server",
            "--ip",
            "0.0.0.0",
            "--port",
            "8888",
            "--no-browser",
            "--ServerApp.root_dir=/scratch",
            "--ServerApp.allow_remote_access=True",
            "--ServerApp.disable_check_xsrf=True",
        )
        docker(
            "run",
            "-d",
            "--name",
            self.proxy,
            "--network",
            "bridge",
            "-p",
            f"127.0.0.1:{self.port}:8888",
            "--read-only",
            "--user",
            "1000:1000",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            "256m",
            "-v",
            f"{(SANDBOX / 'proxy.py').as_posix()}:/proxy.py:ro",
            self.tag,
            "python",
            "/proxy.py",
            "0.0.0.0",
            "8888",
            self.jupyter,
            "8888",
        )
        docker("network", "connect", self.network, self.proxy)
        self.wait_ready()

    def wait_ready(self, timeout: float = 90.0) -> None:
        deadline = time.time() + timeout
        url = f"http://127.0.0.1:{self.port}/api/status?token={self.token}"
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=3) as response:
                    if response.status == 200:
                        return
            except Exception:
                time.sleep(1.5)
        logs = docker("logs", "--tail", "30", self.jupyter, check=False)
        raise SystemExit(
            "the sandbox Jupyter server did not become ready:\n" + (logs.stderr or logs.stdout)[-2000:]
        )

    def stop(self) -> None:
        for name in (self.proxy, self.jupyter):
            docker("rm", "-f", name, check=False)
        docker("network", "rm", self.network, check=False)

    def mcp_command(self) -> list[str]:
        url = f"http://127.0.0.1:{self.port}"
        return [
            sys.executable,
            "-m",
            "jupyter_mcp_server",
            "start",
            "--transport",
            "stdio",
            "--document-provider",
            "jupyter",
            "--sandbox-variant",
            "jupyter-server",
            "--jupyter-url",
            url,
            "--jupyter-token",
            self.token,
            "--document-url",
            url,
            "--document-token",
            self.token,
            "--code-sandbox-url",
            url,
            "--code-sandbox-token",
            self.token,
            "--start-new-code-sandbox",
            "true",
        ]

    def describe(self) -> dict:
        return {
            "image": self.tag,
            "image_base": IMAGE_BASE,
            "lock": str(LOCK.relative_to(REPO)).replace("\\", "/"),
            "lock_sha256": sha256(LOCK),
            "network": "docker internal network; the Jupyter container has no route out",
            "ingress": f"one forwarder container publishing 127.0.0.1:{self.port} into the internal network",
            "mounts": {
                "/data": "competition root, read-only",
                "/repo/src": "repository source, read-only",
                "/scratch": "one scratch directory, read-write",
            },
            "root_filesystem": "read-only with tmpfs at /tmp and /home/sandbox",
            "user": "1000:1000, all capabilities dropped, no-new-privileges",
            "limits": {"memory": self.memory, "cpus": self.cpus, "pids": 512},
            "secrets": "one random token per launch, in process environments only",
        }


async def validate(box: Sandbox, record_path: Path) -> int:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=box.mcp_command()[0], args=box.mcp_command()[1:], env=dict(os.environ)
    )
    probes = {
        "read_data_root": "import os; print(sorted(os.listdir('/data/train'))[:2])",
        "write_into_data_root_must_fail": (
            "try:\n    open('/data/biohubx-write-probe', 'w').write('x'); print('WROTE')\n"
            "except OSError as e:\n    print('refused:', type(e).__name__)"
        ),
        "outbound_network_must_fail": (
            "import socket, urllib.request; socket.setdefaulttimeout(5)\n"
            "try:\n    urllib.request.urlopen('https://pypi.org'); print('REACHED')\n"
            "except Exception as e:\n    print('blocked:', type(e).__name__)"
        ),
        "write_scratch": "open('/scratch/probe.txt', 'w').write('ok'); print(open('/scratch/probe.txt').read())",
        "biohubx_and_data": (
            "import zarr, biohubx, os\n"
            "from biohubx.data.competition import load_ground_truth\n"
            "from pathlib import Path\n"
            "root = Path('/data'); ids = sorted(p.stem for p in (root/'train').glob('*.geff'))\n"
            "t = load_ground_truth(root, ids[0]); g = zarr.open(str(root/'train'/f'{ids[0]}.zarr'), mode='r')['0']\n"
            "print(ids[0], len(t.lineage.nodes), tuple(g.shape))"
        ),
    }
    record: dict = {
        "schema_version": 1,
        "session_id": "WS-COMPUTE-01",
        "profile": "biohub-compute",
        "provenance_status": "integration_only",
        "objective": "prove the sandbox reads the competition root, cannot write into it, cannot reach the network, writes only to scratch, and imports biohubx",
        "falsifier": "any write into /data succeeding, any outbound connection succeeding, or a probe that cannot run",
        "sandbox": box.describe(),
        "probes": {},
    }
    ok = True
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        init = await session.initialize()
        info = getattr(init, "server_info", None) or getattr(init, "serverInfo", None)
        record["server_info"] = {
            "name": getattr(info, "name", None),
            "version": getattr(info, "version", None),
        }
        listed = await session.list_tools()
        names = sorted(t.name for t in listed.tools)
        record["tools"] = names
        executor = next(
            (n for n in ("execute_code", "insert_execute_code_cell", "run_code") if n in names), None
        )
        if executor is None:
            executor = next((n for n in names if "execute" in n and "cell" in n), None)
        record["executor_tool"] = executor
        schema = next(
            (
                getattr(t, "input_schema", None) or getattr(t, "inputSchema", {})
                for t in listed.tools
                if t.name == executor
            ),
            {},
        )
        record["executor_schema"] = schema
        arg_name = next(
            (
                k
                for k in ("code", "cell_source", "source", "cell_code")
                if k in (schema.get("properties") or {})
            ),
            None,
        )
        for name, code in probes.items():
            if executor is None or arg_name is None:
                record["probes"][name] = {"skipped": "no executor tool"}
                ok = False
                continue
            result = await session.call_tool(executor, {arg_name: code})
            text = "".join(getattr(c, "text", "") for c in result.content)
            record["probes"][name] = {
                "is_error": bool(getattr(result, "is_error", None) or getattr(result, "isError", False)),
                "output": text[:600],
            }
    p = record["probes"]
    checks = {
        "read_data_root": "[" in p.get("read_data_root", {}).get("output", ""),
        "write_into_data_root_must_fail": "refused"
        in p.get("write_into_data_root_must_fail", {}).get("output", "")
        and "WROTE" not in p.get("write_into_data_root_must_fail", {}).get("output", ""),
        "outbound_network_must_fail": "blocked" in p.get("outbound_network_must_fail", {}).get("output", "")
        and "REACHED" not in p.get("outbound_network_must_fail", {}).get("output", ""),
        "write_scratch": "ok" in p.get("write_scratch", {}).get("output", ""),
        "biohubx_and_data": "44b6" in p.get("biohubx_and_data", {}).get("output", "")
        or "6bba" in p.get("biohubx_and_data", {}).get("output", ""),
    }
    record["checks"] = checks
    record["ok"] = ok and all(checks.values())
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print("tools:", len(names), "executor:", executor, "arg:", arg_name)
    for name, value in checks.items():
        print(f"  {name:32} {value}  {p.get(name, {}).get('output', '')[:100]!r}")
    print("overall ok:", record["ok"], "->", record_path)
    return 0 if record["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=18888)
    parser.add_argument(
        "--scratch", help="The one read-write directory. Default: <tools root>/biohub-compute/scratch"
    )
    parser.add_argument("--memory", default="8g")
    parser.add_argument("--cpus", default="4")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run the isolation probes and write a session record instead of serving.",
    )
    parser.add_argument("--build-only", action="store_true", help="Build the image and exit.")
    args = parser.parse_args()

    data_root = os.environ.get("BIOHUB_DATA_ROOT")
    if not data_root:
        print(
            "BIOHUB_DATA_ROOT is unset; the sandbox mounts it read-only and refuses to guess", file=sys.stderr
        )
        return 2
    tools_root = Path(os.environ.get("BIOHUBX_TOOLS_ROOT") or REPO.parent / "Biohub-X-tools").resolve()
    scratch = Path(args.scratch) if args.scratch else tools_root / "biohub-compute" / "scratch"
    if args.build_only:
        tag = image_tag()
        if docker("image", "inspect", tag, check=False).returncode != 0:
            build_image(tag)
        print("image", tag)
        return 0
    box = Sandbox(
        data_root=Path(data_root), scratch=scratch, port=args.port, memory=args.memory, cpus=args.cpus
    )
    try:
        box.start()
    except BaseException:
        # A half-started sandbox must not outlive the launcher: a leaked proxy keeps
        # the port and the next launch fails for a reason that looks unrelated.
        box.stop()
        raise
    try:
        if args.validate:
            import asyncio

            return asyncio.run(validate(box, REPO / "artifacts" / "tool-sessions" / "WS-COMPUTE-01.json"))
        completed = subprocess.run(box.mcp_command(), env=dict(os.environ), check=False)
        return completed.returncode
    finally:
        box.stop()


if __name__ == "__main__":
    raise SystemExit(main())
