"""Kaggle derives a kernel's slug from its TITLE, not from the declared id.

WHY THIS EXISTS
Pushing `p19_relink_sweep_f1` declared slug `biohub-p19-relink-sweep-f1`, but Kaggle created
`biohub-p19-relink-division-sweep-loeo-f1` from the title "Biohub P19 Relink Division Sweep
LOEO F1". `status` and `fetch` then address a kernel that does not exist. This was the SECOND
occurrence — commit 01f9fff was the same repair on a p4 fold-1 export.

Auditing all specs afterwards found TWO MORE that had already diverged and been left broken:
`p3_base_loeo_f0` and `p3_base_loeo_f1` declared `biohub-p3-base-loeo-f*` while the real
kernels sit at `biohub-p3-base-armb-off-loeo-f*` and were COMPLETE the whole time. Their
status and fetch had been silently addressing nothing.

The divergence is deterministic and therefore predictable at build time, which is where it
should be caught — not after a push has already created a kernel under the wrong name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPECS = sorted((REPO / "scripts" / "kaggle_specs").glob("*.json"))


def kaggle_slugify(title: str) -> str:
    """Reproduce Kaggle's title -> slug rule: lowercase, non-alphanumerics collapse to '-'."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", title.lower())).strip("-")


def test_slugify_matches_the_observed_divergence():
    """Anchor the rule on the case that actually happened, so the helper cannot drift."""
    assert kaggle_slugify("Biohub P19 Relink Division Sweep LOEO F1") == \
        "biohub-p19-relink-division-sweep-loeo-f1"
    assert kaggle_slugify("Biohub P9 Coupled Division") == "biohub-p9-coupled-division"


@pytest.mark.parametrize("spec_path", SPECS, ids=lambda p: p.stem)
def test_declared_slug_is_what_kaggle_will_create(spec_path: Path):
    """A spec whose title does not slugify to its declared slug will push to a DIFFERENT
    kernel than the one status/fetch/submit address."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    title, slug = spec.get("title"), spec.get("slug")
    if not title or not slug:
        pytest.skip("spec declares no title/slug pair")
    assert kaggle_slugify(title) == slug, (
        f"{spec_path.stem}: title {title!r} slugifies to "
        f"{kaggle_slugify(title)!r} but the spec declares {slug!r}. "
        "Kaggle would create the former; status/fetch would address the latter."
    )
