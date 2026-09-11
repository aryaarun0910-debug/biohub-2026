# The authoritative scorer

`metrics.py` and `division_metrics.py` here are the **PATCHED** versions, copied from
`reference/royerlab-baseline/` at merge `075fc5f` (which contains `aa65e90`, the 2026-07-17
division-exploit patch).

## Why this directory had a landmine in it

The originally vendored copies were the **pre-patch** scorer — `_weakly_connected_components`,
450 lines, no `_is_strongly_connected_division`. They are byte-identical to commit `7396b7e`
(2026-07-08), nine days before the patch. They are preserved under `PRE-PATCH-DO-NOT-USE/`
for forensics only.

**Anything validated against the pre-patch scorer is calibrated to a closed exploit.** It scores
a far-away synthetic fork as a true positive; the live metric scores it as a false positive.

## How to tell them apart

    grep -c _is_strongly_connected_division division_metrics.py   # patched: 2, pre-patch: 0
    grep -c _weakly_connected             division_metrics.py     # patched: 0, pre-patch: 2

Also note the Kaggle **Evaluation page is itself stale** and still describes the pre-patch rule.
`metrics.md` in the upstream repo is correct; the competition page is not.
