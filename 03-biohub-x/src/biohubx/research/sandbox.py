"""A disposable sandbox for third-party code, and nothing else runs it.

[[D-0039]] permits executing third-party code only in a sandbox with no secrets,
no general host mount, no external writes, bounded processes and resources, and
networking disabled unless separately declared. The ordinary workstation shell is
not such a sandbox. This module is the one place that satisfies those conditions,
so a build or a test of downloaded code has exactly one way to run and that way
is auditable.

The isolation is a container: a pinned base image, ``--network none``, a
read-only root filesystem with tmpfs scratch, one bind-mounted work directory and
nothing else from the host, a non-root user, CPU, memory and PID limits, and a
wall-clock timeout. Every run records the image digest, the command, the limits,
and the exit status.

Consumer: ``biohubx research sandbox``.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_IMAGE = "python:3.12-slim"
DEFAULT_CPUS = "2"
DEFAULT_MEMORY = "4g"
DEFAULT_PIDS = 256
DEFAULT_TIMEOUT_SECONDS = 1800


class SandboxError(RuntimeError):
    """The sandbox could not be established as specified."""


@dataclass(frozen=True, slots=True)
class SandboxRun:
    image: str
    image_digest: str
    command: list[str]
    workdir_host: str
    network: str
    cpus: str
    memory: str
    pids: int
    timeout_seconds: int
    returncode: int
    elapsed_seconds: float
    stdout_tail: str
    stderr_tail: str
    timed_out: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def docker_available() -> str:
    """The docker executable, or a refusal that says what is missing."""
    executable = shutil.which("docker")
    if executable is None:
        raise SandboxError("docker is not on PATH; there is no disposable sandbox on this machine")
    probe = subprocess.run(
        [executable, "info", "--format", "{{.OSType}}"], capture_output=True, text=True, check=False
    )
    if probe.returncode != 0:
        raise SandboxError(f"docker is installed but not running: {probe.stderr.strip()[:200]}")
    return executable


def image_digest(executable: str, image: str) -> str:
    """The pinned repo digest of a local image, refusing an image that has none.

    A tag is mutable; a digest is not. A run is recorded against the digest so
    that "python:3.12-slim" meaning something else next month cannot change what
    a recorded build was built with.
    """
    inspect = subprocess.run(
        [executable, "image", "inspect", image, "--format", "{{index .RepoDigests 0}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    digest = inspect.stdout.strip()
    if inspect.returncode != 0 or "@sha256:" not in digest:
        raise SandboxError(f"image {image!r} is not present locally with a repo digest; pull it first")
    return digest


def run_in_sandbox(
    command: list[str],
    *,
    workdir: Path,
    image: str = DEFAULT_IMAGE,
    cpus: str = DEFAULT_CPUS,
    memory: str = DEFAULT_MEMORY,
    pids: int = DEFAULT_PIDS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    allow_network: bool = False,
    env: dict[str, str] | None = None,
) -> SandboxRun:
    """Run one command inside the sandbox with ``workdir`` mounted at /work.

    ``workdir`` is the only host path the container can see, and it is the only
    place the container can write outside its own tmpfs. Nothing under the user's
    home, no credentials, no repository, no data root.
    """
    executable = docker_available()
    digest = image_digest(executable, image)
    workdir = workdir.resolve()
    if not workdir.is_dir():
        raise SandboxError(f"sandbox workdir does not exist: {workdir}")
    network = "bridge" if allow_network else "none"
    argv = [
        executable,
        "run",
        "--rm",
        "--network",
        network,
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=2g",
        "--tmpfs",
        "/home/sandbox:rw,size=1g",
        "--mount",
        f"type=bind,source={workdir.as_posix()},target=/work",
        "--workdir",
        "/work",
        "--user",
        "1000:1000",
        "--cpus",
        cpus,
        "--memory",
        memory,
        "--pids-limit",
        str(pids),
        "--security-opt",
        "no-new-privileges",
        "--cap-drop",
        "ALL",
        "-e",
        "HOME=/home/sandbox",
        "-e",
        "PIP_NO_CACHE_DIR=1",
    ]
    for key, value in sorted((env or {}).items()):
        argv += ["-e", f"{key}={value}"]
    argv += [digest, *command]

    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_seconds, check=False)
        returncode = completed.returncode
        stdout, stderr = completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = -1
        stdout = (
            (exc.stdout or b"").decode("utf-8", "replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            (exc.stderr or b"").decode("utf-8", "replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
    return SandboxRun(
        image=image,
        image_digest=digest,
        command=list(command),
        workdir_host=str(workdir),
        network=network,
        cpus=cpus,
        memory=memory,
        pids=pids,
        timeout_seconds=timeout_seconds,
        returncode=returncode,
        elapsed_seconds=round(time.monotonic() - started, 3),
        stdout_tail=stdout[-4000:],
        stderr_tail=stderr[-4000:],
        timed_out=timed_out,
    )
