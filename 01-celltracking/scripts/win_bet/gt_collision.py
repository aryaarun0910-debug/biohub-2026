"""Injective predicted<->GT id maps, with the mother-collision assertion.

The D0P/H0c/H0b/H1a/D0' lineage all built the reverse map with a bare

    gt_to_sub[int(mid)] = s

inside the match loop. Under a one-to-one bipartite `DistanceMatching` that cannot
collide, but nothing in those scripts enforced it: a second predicted node claiming the
same GT id would silently overwrite the first, and every downstream division statistic
(`reachable`, `retained`, `truth`, `gt_div`) would be computed against the survivor with
no trace in the output. Zero collisions were MEASURED on the current E0c cache, so the
defect is latent, not active -- but it is latent on THIS node population only, and the
division track keeps changing the node population.

`scripts/win_bet/phaseb_h0d_livefilter.py::build_gt_maps` introduced the guard. This
module is that guard extracted so the five earlier sites share one implementation instead
of five copies. Semantics are identical to H0d:

  * first writer wins (the loser is recorded, never silently clobbered);
  * a collision raises by default;
  * `allow_collisions=True` downgrades it to a counted, returned observation.

Because the guard RAISES rather than resolving, a run that completes proves it never
fired, which in turn proves the returned maps are bit-identical to what the unguarded
last-writer-wins loop produced. That is the whole point: it cannot change a number
without also refusing to produce one.
"""
from __future__ import annotations


def gt_maps_from_matches(
    sub_to_int: dict[int, int],
    int_to_gt: dict[int, object],
    *,
    context: str = "",
    allow_collisions: bool = False,
) -> tuple[dict[int, int], dict[int, int], int]:
    """Build ``gt_to_sub`` / ``sub_to_gt`` from a matched graph without clobbering.

    ``sub_to_int`` maps submission node id -> internal graph id; ``int_to_gt`` maps
    internal graph id -> matched GT node id (``None`` or ``-1`` when unmatched).
    Iteration follows ``sub_to_int`` insertion order, exactly as the sites it replaces.
    """
    gt_to_sub: dict[int, int] = {}
    sub_to_gt: dict[int, int] = {}
    collisions: list[tuple[int, int, int]] = []
    for s, iid in sub_to_int.items():
        mid = int_to_gt.get(iid)
        if mid in (None, -1):
            continue
        mid = int(mid)
        if mid in gt_to_sub:
            collisions.append((mid, gt_to_sub[mid], s))
            continue
        gt_to_sub[mid] = s
        sub_to_gt[s] = mid
    if collisions and not allow_collisions:
        where = f"{context}: " if context else ""
        raise AssertionError(
            f"{where}mother-collision: {len(collisions)} GT node(s) claimed by >1 predicted "
            f"node; first three {collisions[:3]}. The gt_to_sub map is not injective on this "
            f"node population -- pass allow_collisions=True to quantify instead of abort."
        )
    return gt_to_sub, sub_to_gt, len(collisions)
