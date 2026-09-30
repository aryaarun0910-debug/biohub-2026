"""The ledger must dedupe, count the shared budget, and outlive its payloads.

No network here. The fetch is replaced with a stub, because a unit test that
reaches the internet is a test of the internet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biohubx.research import ledger


def stub_fetch(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    monkeypatch.setattr(ledger, "fetch", lambda url, *, max_bytes: (200, payload))


def test_a_fetched_source_is_recorded_with_its_digest_and_a_relative_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_fetch(monkeypatch, b"public bytes")

    entry, created = ledger.intake(
        tmp_path,
        url="https://example.org/paper.pdf",
        kind="paper",
        campaign="RX-01",
        branch="proposals",
        note="scale-space detection",
        do_fetch=True,
    )

    assert created is True
    assert entry.id == "RL-0001"
    assert entry.raw_digest is not None and entry.raw_digest.startswith("raw_artifact_sha256:")
    assert entry.cache_path == "research/cache/RL-0001/paper.pdf"
    assert not Path(entry.cache_path).is_absolute()
    assert (tmp_path / entry.cache_path).read_bytes() == b"public bytes"


def test_the_same_url_is_not_fetched_twice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two branches that do not know about each other must not double a transfer."""
    calls: list[str] = []

    def counting(url: str, *, max_bytes: int) -> tuple[int, bytes]:
        calls.append(url)
        return 200, b"x"

    monkeypatch.setattr(ledger, "fetch", counting)
    first, created_first = ledger.intake(
        tmp_path,
        url="https://example.org/a",
        kind="page",
        campaign="RX-01",
        branch="a",
        note="",
        do_fetch=True,
    )
    second, created_second = ledger.intake(
        tmp_path,
        url="https://example.org/a",
        kind="page",
        campaign="RX-01",
        branch="b",
        note="",
        do_fetch=True,
    )

    assert (created_first, created_second) == (True, False)
    assert second.id == first.id
    assert calls == ["https://example.org/a"]


def test_eviction_deletes_the_bytes_and_keeps_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_fetch(monkeypatch, b"payload")
    entry, _ = ledger.intake(
        tmp_path,
        url="https://example.org/big.bin",
        kind="other",
        campaign="RX-01",
        branch="x",
        note="",
        do_fetch=True,
    )

    evicted = ledger.evict(tmp_path, entry.id)

    assert evicted.payload_present is False
    assert evicted.raw_digest == entry.raw_digest
    assert evicted.url == entry.url
    assert entry.cache_path is not None
    assert not (tmp_path / entry.cache_path).exists()
    # And the budget still counts it: eviction reclaims disk, not allowance.
    assert ledger.budget_used(ledger.load_ledger(tmp_path)) == (1, len(b"payload"))


def test_non_http_sources_are_refused() -> None:
    with pytest.raises(ledger.LedgerError, match="public http"):
        ledger.intake(
            Path(),
            url="file:///etc/passwd",
            kind="other",
            campaign="RX-01",
            branch="x",
            note="",
            do_fetch=False,
        )


def test_an_unknown_kind_is_refused() -> None:
    with pytest.raises(ledger.LedgerError, match="kind must be one of"):
        ledger.intake(
            Path(),
            url="https://example.org",
            kind="rumour",
            campaign="RX-01",
            branch="x",
            note="",
            do_fetch=False,
        )


def test_the_request_budget_is_counted_over_the_whole_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_fetch(monkeypatch, b"x")
    monkeypatch.setattr(ledger, "TOTAL_REQUEST_BUDGET", 2)
    for index in range(2):
        ledger.intake(
            tmp_path,
            url=f"https://example.org/{index}",
            kind="page",
            campaign="RX-01",
            branch="x",
            note="",
            do_fetch=True,
        )

    with pytest.raises(ledger.LedgerError, match="request budget exhausted"):
        ledger.intake(
            tmp_path,
            url="https://example.org/3",
            kind="page",
            campaign="RX-01",
            branch="y",
            note="",
            do_fetch=True,
        )
