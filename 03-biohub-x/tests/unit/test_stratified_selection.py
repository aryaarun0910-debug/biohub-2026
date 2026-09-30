"""Which movies a fold reads, and why it is no longer the first ones in name order.

E07 Stage 2 took the first two 44b6 movies alphabetically, which are the second
and fourth sparsest of 71, and trained on 13 positives ([[R-0026]]). Selection
is now stratified across annotation-count quartiles of the training embryo,
median bands first, deterministic, and it must never collapse into "the densest
movies" either.
"""

from __future__ import annotations

from biohubx.training.rescore_loop import STRATA, stratified_movies


def _counts(n: int) -> dict[str, int]:
    # Ascending counts with ids that would sort the other way, so name order
    # and count order disagree and a mistaken sort shows.
    return {f"e_{n - i:03d}": 10 * (i + 1) for i in range(n)}


def test_selection_is_deterministic() -> None:
    counts = _counts(71)
    assert stratified_movies(counts, 8) == stratified_movies(dict(reversed(list(counts.items()))), 8)


def test_eight_picks_land_two_per_quartile() -> None:
    counts = _counts(72)
    chosen = stratified_movies(counts, 8)
    ranked = sorted(counts, key=lambda d: (counts[d], d))
    bands = [ranked[i * 18 : (i + 1) * 18] for i in range(STRATA)]
    per_band = [sum(1 for c in chosen if c in band) for band in bands]
    assert per_band == [2, 2, 2, 2]


def test_two_picks_come_from_the_middle_bands_not_the_sparsest_or_densest() -> None:
    counts = _counts(72)
    chosen = stratified_movies(counts, 2)
    ranked = sorted(counts, key=lambda d: (counts[d], d))
    sparsest, densest = set(ranked[:18]), set(ranked[-18:])
    assert not (set(chosen) & sparsest)
    assert not (set(chosen) & densest)


def test_the_selection_is_never_only_the_densest() -> None:
    counts = _counts(40)
    chosen = stratified_movies(counts, 4)
    ranked = sorted(counts, key=lambda d: (counts[d], d))
    assert set(chosen) != set(ranked[-4:])
    assert len(chosen) == 4 and len(set(chosen)) == 4


def test_asking_for_everything_returns_everything_once() -> None:
    counts = _counts(5)
    assert sorted(stratified_movies(counts, 5)) == sorted(counts)
    assert sorted(stratified_movies(counts, 50)) == sorted(counts)
    assert stratified_movies(counts, 0) == []


def test_the_pick_within_a_band_is_its_median_first() -> None:
    counts = _counts(8)  # two per band
    chosen = stratified_movies(counts, 4)
    ranked = sorted(counts, key=lambda d: (counts[d], d))
    # Bands are [0,1] [2,3] [4,5] [6,7]; median index of a two-element band is 1.
    assert chosen == [ranked[5], ranked[3], ranked[7], ranked[1]]
