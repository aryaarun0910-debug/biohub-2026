"""Injective predicted-to-GT id maps used by collision parity checks."""
from __future__ import annotations


def gt_maps_from_matches(
    sub_to_int: dict[int, int],
    int_to_gt: dict[int, object],
    *,
    context: str = "",
    allow_collisions: bool = False,
) -> tuple[dict[int, int], dict[int, int], int]:
    """Build injective GT/submission maps without silent last-writer clobbering."""
    gt_to_sub: dict[int, int] = {}
    sub_to_gt: dict[int, int] = {}
    collisions: list[tuple[int, int, int]] = []
    for submission_id, internal_id in sub_to_int.items():
        matched_id = int_to_gt.get(internal_id)
        if matched_id in (None, -1):
            continue
        matched_id = int(matched_id)
        if matched_id in gt_to_sub:
            collisions.append((matched_id, gt_to_sub[matched_id], submission_id))
            continue
        gt_to_sub[matched_id] = submission_id
        sub_to_gt[submission_id] = matched_id
    if collisions and not allow_collisions:
        where = f"{context}: " if context else ""
        raise AssertionError(
            f"{where}mother-collision: {len(collisions)} GT node(s) claimed by more than "
            f"one predicted node; first three {collisions[:3]}"
        )
    return gt_to_sub, sub_to_gt, len(collisions)
