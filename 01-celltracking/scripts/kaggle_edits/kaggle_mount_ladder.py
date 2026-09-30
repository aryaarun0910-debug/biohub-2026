# =====================================================================================
# THE CENTRAL BOUNDED DATASET RESOLVER.  One implementation, injected, never re-written.
#
# WHY THIS FILE EXISTS AS A SEPARATE, INJECTABLE SNIPPET
# -----------------------------------------------------
# Kaggle does not mount a dataset at one predictable root. Observed in this repository's own
# fetched logs: the competition lands at /kaggle/input/competitions/<slug>/ and user datasets at
# /kaggle/input/datasets/<owner>/<slug>/, while almost every patch written here has spelled
# /kaggle/input/<slug>/ by hand. That difference has now cost TWO fully-paid GPU sessions:
# 2026-08-01 a fold-1 kernel died at t=628 s, and 2026-08-30 Gate 1 attempt 2 died at t=238 s
# having compared zero crops (FACT-0397) - with the fix for the first sitting in
# scripts/kaggle_edits/loeo_retarget.py:86-101, in a file the second one LOADS.
#
# Writing the ladder down a third time in a third patch is how it will be got wrong a third time.
# So it is written ONCE, here, and every patch that needs a mount injects THIS text. The CPU-side
# preflight (scripts/win_bet/kaggle_mounts.py) does not re-implement it either - it execs this
# same file, so the resolver the preflight proves correct is byte-identical to the resolver the
# kernel runs. There is no second copy that can drift.
#
# BOUNDED, NEVER RECURSIVE. `**` under /kaggle/input would descend the 79 GB competition zarr
# tree. The ladder is a fixed number of single-level globs, tried shallowest first.
#
# FAIL CLOSED AND DIAGNOSE. A miss does not return None into somebody's `if path:`. It raises
# BiohubMountNotFound carrying a listing of what IS mounted, because attempt 2's failure message
# said only that a path was absent, which told the reader nothing about where to look instead.
#
# POSITIVE HEARTBEAT. Every successful resolution prints MOUNT_RESOLVED. Its ABSENCE is the alarm:
# a silent no-op is worse than a crash.
# =====================================================================================
import os as _bm_os
from pathlib import Path as _BmPath

BIOHUB_MOUNT_MAX_DEPTH = 4


class BiohubMountNotFound(FileNotFoundError):
    """Raised when the bounded ladder cannot resolve a declared input."""


def biohub_mount_diagnose(input_root="/kaggle/input", limit=60):
    """List what IS mounted, to depth 3. Turns 'not found' into a diagnosis."""
    root = _BmPath(input_root)
    seen = []
    if not root.exists():
        return ["<%s does not exist - trap 9: the kernel itself is broken>" % input_root]
    for pattern in ("*", "*/*", "*/*/*"):
        try:
            seen += [str(p) for p in sorted(root.glob(pattern))]
        except OSError:
            break
        if len(seen) >= limit:
            break
    return seen[:limit]


def biohub_mount_find(relative, slugs=(), owners=(), input_root="/kaggle/input",
                      max_depth=BIOHUB_MOUNT_MAX_DEPTH, require=True, label=""):
    """Resolve one dataset-relative path at whatever depth Kaggle mounted it.

    `relative`  path INSIDE the dataset, e.g. "meta/preilp_split0.parquet"; "" means the
                dataset directory itself.
    `slugs`     dataset slugs to try by name, e.g. ("biohub-identity-replay-f0",).
    `owners`    owners to try under the datasets/ convention, e.g. ("aryaarun07",).

    Order: the two mount conventions this repository has actually observed, then a bounded
    depth ladder on the remaining path. Never `**`.
    """
    root = _BmPath(input_root)
    rel = str(relative).strip("/")
    tried = []

    def _take(cand):
        tried.append(str(cand))
        return cand if cand.exists() else None

    # 1. named-slug conventions, in observed frequency order
    for slug in slugs:
        for base in (root / slug,
                     root / "datasets" / slug,
                     root / "competitions" / slug):
            hit = _take(base / rel if rel else base)
            if hit is not None:
                print("MOUNT_RESOLVED %s -> %s" % (label or rel or slug, hit), flush=True)
                return hit
        for owner in owners:
            for base in (root / "datasets" / owner / slug,
                         root / owner / slug):
                hit = _take(base / rel if rel else base)
                if hit is not None:
                    print("MOUNT_RESOLVED %s -> %s" % (label or rel or slug, hit), flush=True)
                    return hit

    # 2. bounded depth ladder - single-level globs only, shallowest first
    tail = ("/" + rel) if rel else ""
    for depth in range(1, int(max_depth) + 1):
        pattern = "/".join(["*"] * depth) + tail
        tried.append(str(root / pattern))
        try:
            hits = sorted(root.glob(pattern))
        except OSError:
            hits = []
        for cand in hits:
            if cand.exists():
                print("MOUNT_RESOLVED %s -> %s (depth %d)" % (label or rel or "root", cand, depth),
                      flush=True)
                return cand

    if not require:
        return None
    raise BiohubMountNotFound(
        "%s not found under %s. Tried %d locations: %r. Mounted now: %r"
        % (label or rel or "<dataset root>", input_root, len(tried), tried[:12],
           biohub_mount_diagnose(input_root))
    )


def biohub_mount_find_dir(relative, contains_glob="*", **kw):
    """Resolve a DIRECTORY and require it to be non-empty for `contains_glob`.

    'The directory exists' is not the question a caller ever actually has. Attempt 1 of Gate 1
    passed every existence check it had and compared nothing (FACT-0387); FACT-0394 is the same
    shape one level in. So a directory that matches nothing is a miss, not a hit.
    """
    kw = dict(kw)
    require = kw.pop("require", True)
    hit = biohub_mount_find(relative, require=False, **kw)
    if hit is not None and hit.is_dir() and any(hit.glob(contains_glob)):
        return hit
    if not require:
        return None
    raise BiohubMountNotFound(
        "%r resolved to %r, which is not a directory containing %r. Mounted now: %r"
        % (relative, str(hit) if hit else None, contains_glob,
           biohub_mount_diagnose(kw.get("input_root", "/kaggle/input")))
    )


_ = _bm_os  # keep the import meaningful for linters injected into a notebook cell
