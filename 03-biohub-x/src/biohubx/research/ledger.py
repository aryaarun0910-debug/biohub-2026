"""The ledger: one record per acquired public artifact, payload evictable, identity not.

Three properties come from [[D-0039]] and are enforced rather than hoped for.

Every record names where the bytes came from, when, how many, and their raw
digest, so a claim in the reference registry can say "RL-0007, section 3" and a
reader can re-fetch and check. Metadata survives payload eviction: deleting the
cached bytes to reclaim disk leaves the record, the digest and the source in
place, and marks the payload absent rather than pretending nothing was ever
there. And the ledger is consulted before a download, so the same URL is not
fetched twice by two branches that did not know about each other.

The budget is shared and counted here, because [[D-0039]] forbids evading a hard
limit by splitting work across branches. Requests and bytes are summed over the
whole ledger, not per session.

Payloads live under ``research/cache/``, which is ignored by Git. The ledger
records only repository-relative paths, so no machine-local absolute path enters a
tracked file.
"""

from __future__ import annotations

import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from biohubx.artifacts import atomic_write_text
from biohubx.hashing import DigestKind, digest_file

LEDGER_LOCK_PATH = Path("registry/.research-ledger.lock")
"""The one lock every canonical research-ledger write takes."""

LEDGER_LOCK_TIMEOUT_SECONDS = 120.0
"""Long enough to outlast an intake's bounded fetch, short enough to fail loudly."""

LEDGER_PATH = Path("registry/research-ledger.yaml")
CACHE_ROOT = Path("research/cache")
KINDS = ("paper", "repository", "notebook", "discussion", "page", "dataset_listing", "other")
DEFAULT_MAX_BYTES_PER_ITEM = 256 * 1024 * 1024
TOTAL_BYTE_BUDGET = 8 * 1024**3
TOTAL_REQUEST_BUDGET = 2000
USER_AGENT = "Biohub-X research intake (public sources only; contact via repository)"


class LedgerError(ValueError):
    """The intake cannot proceed as asked, and says why."""


@contextmanager
def ledger_lock(root: Path, *, timeout: float = LEDGER_LOCK_TIMEOUT_SECONDS) -> Iterator[Path]:
    """Exclusive-create a lock file, or refuse.

    ``O_CREAT | O_EXCL`` is atomic on every filesystem this repository runs on,
    including Windows, which is why it is used rather than a library. The holder
    writes its pid and the time it acquired, so a stale lock is diagnosable
    instead of anonymous. Nothing here breaks a lock automatically: a lock that
    outlives its holder is a fact worth seeing.
    """
    path = root / LEDGER_LOCK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    handle: int | None = None
    while True:
        try:
            handle = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                held = path.read_text(encoding="utf-8").strip() if path.exists() else "unknown holder"
                raise LedgerError(
                    f"the research ledger lock at {LEDGER_LOCK_PATH} was held for more than "
                    f"{timeout:.0f}s by {held}; no write was attempted"
                ) from None
            time.sleep(0.05)
    try:
        os.write(
            handle, f"pid={os.getpid()} acquired={datetime.now(UTC).isoformat(timespec='seconds')}\n".encode()
        )
        os.close(handle)
        handle = None
        yield path
    finally:
        if handle is not None:
            os.close(handle)
        path.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    id: str
    url: str
    kind: str
    campaign: str
    branch: str
    note: str
    recorded_utc: str
    fetched: bool
    fetched_utc: str | None
    http_status: int | None
    bytes: int | None
    raw_digest: str | None
    cache_path: str | None
    payload_present: bool
    status: str = "reference_only"
    claims: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_ledger(root: Path) -> list[LedgerEntry]:
    path = root / LEDGER_PATH
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [LedgerEntry(**item) for item in raw.get("entries", [])]


def save_ledger(root: Path, entries: list[LedgerEntry]) -> None:
    payload = {
        "schema_version": 1,
        "policy": "D-0039",
        "budget": {"total_bytes": TOTAL_BYTE_BUDGET, "total_requests": TOTAL_REQUEST_BUDGET},
        "entries": [entry.to_dict() for entry in entries],
    }
    atomic_write_text(root / LEDGER_PATH, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def next_id(entries: list[LedgerEntry]) -> str:
    highest = 0
    for entry in entries:
        match = re.fullmatch(r"RL-(\d{4})", entry.id)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"RL-{highest + 1:04d}"


def budget_used(entries: list[LedgerEntry]) -> tuple[int, int]:
    """Requests made and bytes fetched over the whole ledger, evicted or not."""
    requests = sum(1 for entry in entries if entry.fetched)
    total = sum(entry.bytes or 0 for entry in entries if entry.fetched)
    return requests, total


def find_by_url(entries: list[LedgerEntry], url: str) -> LedgerEntry | None:
    for entry in entries:
        if entry.url == url:
            return entry
    return None


def _is_public_http(url: str) -> bool:
    return url.startswith("https://") or url.startswith("http://")


def fetch(url: str, *, max_bytes: int) -> tuple[int, bytes]:
    """One bounded GET. Refuses anything that is not plain public HTTP."""
    if not _is_public_http(url):
        raise LedgerError(f"only public http(s) URLs are fetched, got {url!r}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = int(getattr(response, "status", 200))
            data = response.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:
        raise LedgerError(f"HTTP {exc.code} from {url}") from exc
    except urllib.error.URLError as exc:
        raise LedgerError(f"could not reach {url}: {exc.reason}") from exc
    if len(data) > max_bytes:
        raise LedgerError(
            f"{url} exceeds the per-item cap of {max_bytes} bytes; raise --max-bytes deliberately "
            "if this acquisition is intended"
        )
    return status, data


def _cache_name(url: str) -> str:
    tail = url.rstrip("/").rsplit("/", 1)[-1] or "index"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", tail)[:80]
    return safe or "payload"


def intake(
    root: Path,
    *,
    url: str,
    kind: str,
    campaign: str,
    branch: str,
    note: str,
    do_fetch: bool,
    max_bytes: int = DEFAULT_MAX_BYTES_PER_ITEM,
) -> tuple[LedgerEntry, bool]:
    """Record a source, optionally fetching it. Returns the entry and whether it is new.

    A URL already in the ledger is returned as it stands rather than fetched
    again, which is the dedupe [[D-0039]] asks for. The budget is checked before
    any request is made, against the whole ledger.
    """
    if kind not in KINDS:
        raise LedgerError(f"kind must be one of {KINDS}, got {kind!r}")
    if not _is_public_http(url):
        raise LedgerError(f"only public http(s) sources are recorded, got {url!r}")

    # The whole read, modify, write is inside the lock, fetch included. The
    # sequential id allocation below is exactly the operation that is correct for
    # one writer and lossy for two, and holding the lock across a bounded fetch
    # costs a queued caller seconds while dropping it would cost a record.
    with ledger_lock(root):
        return _intake_locked(
            root,
            url=url,
            kind=kind,
            campaign=campaign,
            branch=branch,
            note=note,
            do_fetch=do_fetch,
            max_bytes=max_bytes,
        )


def _intake_locked(
    root: Path,
    *,
    url: str,
    kind: str,
    campaign: str,
    branch: str,
    note: str,
    do_fetch: bool,
    max_bytes: int,
) -> tuple[LedgerEntry, bool]:
    entries = load_ledger(root)
    existing = find_by_url(entries, url)
    if existing is not None:
        return existing, False

    requests_used, bytes_used = budget_used(entries)
    if do_fetch and requests_used + 1 > TOTAL_REQUEST_BUDGET:
        raise LedgerError(f"request budget exhausted: {requests_used} of {TOTAL_REQUEST_BUDGET}")
    if do_fetch and bytes_used + max_bytes > TOTAL_BYTE_BUDGET:
        raise LedgerError(
            f"byte budget could be exceeded: {bytes_used} used of {TOTAL_BYTE_BUDGET}, "
            f"and this fetch may take up to {max_bytes}"
        )

    entry_id = next_id(entries)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    status: int | None = None
    size: int | None = None
    digest: str | None = None
    cache_rel: str | None = None
    fetched_at: str | None = None

    if do_fetch:
        status, data = fetch(url, max_bytes=max_bytes)
        cache_dir = root / CACHE_ROOT / entry_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / _cache_name(url)
        partial = target.with_suffix(target.suffix + ".partial")
        partial.write_bytes(data)
        partial.replace(target)
        size = len(data)
        digest = digest_file(target, DigestKind.RAW_ARTIFACT).token
        cache_rel = target.relative_to(root).as_posix()
        fetched_at = now

    entry = LedgerEntry(
        id=entry_id,
        url=url,
        kind=kind,
        campaign=campaign,
        branch=branch,
        note=note,
        recorded_utc=now,
        fetched=do_fetch,
        fetched_utc=fetched_at,
        http_status=status,
        bytes=size,
        raw_digest=digest,
        cache_path=cache_rel,
        payload_present=do_fetch,
    )
    entries.append(entry)
    save_ledger(root, entries)
    return entry, True


def evict(root: Path, entry_id: str) -> LedgerEntry:
    """Delete a cached payload and keep everything else about it."""
    with ledger_lock(root):
        return _evict_locked(root, entry_id)


def _evict_locked(root: Path, entry_id: str) -> LedgerEntry:
    entries = load_ledger(root)
    for index, entry in enumerate(entries):
        if entry.id != entry_id:
            continue
        if entry.cache_path:
            path = root / entry.cache_path
            if path.exists():
                path.unlink()
        updated = LedgerEntry(**{**entry.to_dict(), "payload_present": False})
        entries[index] = updated
        save_ledger(root, entries)
        return updated
    raise LedgerError(f"no ledger entry {entry_id}")
