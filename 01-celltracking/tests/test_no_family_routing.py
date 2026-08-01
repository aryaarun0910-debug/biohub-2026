"""No deployment path may route on crop or family identity.

CLAUDE.md forbids using family/crop identity as a deployment router, and the 2026-08-01
red-team audit found a LIVE mechanism that would do exactly that if anyone populated it:
`BIOHUB_SHORT_TRACK_MIN_LEN_BY_DATASET` is an environment-driven per-dataset override of the
short-track filter, keyed on the dataset stem. It currently defaults to `{}`, so it is inert --
but nothing stopped it being set, and a public notebook was found gating its division repair on
the literal prefix `6bba_`, so the failure mode is real and already observed in the wild.

These tests make the prohibition executable rather than aspirational.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_short_track_override_is_empty_by_default():
    """With no env var set, the per-dataset override must resolve to {}."""
    os.environ.pop("BIOHUB_SHORT_TRACK_MIN_LEN_BY_DATASET", None)
    from biotrack import wrapper

    parsed = wrapper._parse_short_track_min_len_by_dataset()
    assert parsed == {}, (
        "The per-dataset short-track override is populated. Keying the filter on a dataset stem "
        "is family/crop routing, which CLAUDE.md forbids in any deployment path."
    )


def test_short_track_override_is_not_set_in_this_environment():
    raw = os.environ.get("BIOHUB_SHORT_TRACK_MIN_LEN_BY_DATASET", "").strip()
    assert raw in ("", "{}"), (
        f"BIOHUB_SHORT_TRACK_MIN_LEN_BY_DATASET is set to {raw!r}. That routes the short-track "
        "filter on crop identity and must never reach a submission."
    )


def test_dataset_lookup_ignores_stem_when_override_absent():
    """short_track_min_len_for_dataset must return the global constant for every stem."""
    os.environ.pop("BIOHUB_SHORT_TRACK_MIN_LEN_BY_DATASET", None)
    from biotrack import wrapper

    baseline = int(wrapper.OUTPUT_MIN_TRACK_LEN)
    for stem in ("44b6_0113de3b", "6bba_05db0fb1", "44b6_d29c9ab2", None):
        assert wrapper.short_track_min_len_for_dataset(stem) == baseline, (
            f"stem {stem!r} resolved to a different min-track-len than the global default -- "
            "that is per-crop routing."
        )


@pytest.mark.parametrize("needle", ["44b6_", "6bba_"])
def test_no_family_literal_gates_control_flow_in_wrapper(needle):
    """The wrapper must not branch on an embryo-family literal.

    Matches the public-notebook failure mode found on 2026-07-31, where a shared kernel gated its
    division repair on the dataset prefix `6bba_` after fitting on four events.
    """
    src = (ROOT / "src" / "biotrack" / "wrapper.py").read_text(encoding="utf-8")
    hits = [
        ln.strip()
        for ln in src.splitlines()
        if needle in ln and not ln.strip().startswith("#")
    ]
    assert not hits, (
        f"wrapper.py branches on the family literal {needle!r}:\n  "
        + "\n  ".join(hits[:5])
        + "\nFamily identity must never reach a deployment decision."
    )
