# Submission candidate ledger

Every candidate must record source/config hash, output hash, structural audit, OOF or
public-source justification, graph delta from its anchor, and the causal question it asks.
No candidate may be submitted without all six.

**Submissions consumed to date: 6.** This cycle authorises up to 3 more; slots 4-5 require
explicit review before use.

## Standing prohibitions

Never submit: identical outputs; global threshold sweeps; family-routed systems; exploit
structures; anything tuned on the four visible placeholder movies.

## Quarantined public sources (structural exploits — never adopt)

At least ten highly-upvoted public notebooks inject a hub node at `t = -1000` with coordinates
`(-10000, -10000, -10000)` plus synthetic negative-time fork chains. One (`romanrozen/biohub-best-score`)
ships `BIOHUB_AUGMENT_HUB` defaulting **ON**. Reproducing "the top public notebook" walks directly
into the exploit. Full list in the research store.

## P0-A — exact clean 0.913 reproduction

| field | value |
|---|---|
| source | `saitejabandaruin/biohub-top-notebook-0-913` |
| source sha256 | `35681256f355bc98552d9cbdc4d208a9670657f18b7e7c4ce5637e6619e5e570` |
| size / cells | 148,047 bytes / 11 cells |
| identical copies | `nikitagajbhiye30/biohub-11`, `yiliu6111/biohub-v9-0913fork` (same sha256) |
| upstream author | `indarkarhana` (markdown stripped in the copies) |
| datasets | `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`, `biohub-temporal-unet3d-seed314159-v1`, `biohub-tracking-support-pack-50ep-v1`, `pilkwang-public-dataset-for-notebooks-figures`, `thtennant/taaf-kaggle-source-share-fork` |
| hardware | T4, internet OFF |
| reported public | `0.913` (self-report, grade C until we reproduce it) |
| novel delta | per-frame retention guard (revert blend to primary detector when the blend loses >10% of peaks) |

**Source structural audit — PASS (2026-07-31).**

| check | result |
|---|---|
| `AUGMENT_HUB` | 0 occurrences |
| negative-time literal (`t=-N`, `-1000`) | 0 |
| exploit coordinates (`-10000`, `-9999`) | 1 hit, **benign** |
| synthetic hub / fork construction | 0 |
| `synthetic_fork` / `fake_div` / `artificial` | 0 |
| skip edges (`t+2`, `frame_skip`) | 0 |

The single `-9999` match is `_pl.Series([-999999.0], dtype=Float64)` — a polars compiled-backend
probe, the same environment check documented in `reports/ENVIRONMENT_TRAPS.md`. Not an exploit.

**Still required before this may consume a slot:** reproduction on our account, output-CSV
structural audit (t>=0, in-degree<=1, out-degree<=2, consecutive-frame edges only, no
cross-dataset edges, coordinates in-volume), and the output sha256.

**Causal question:** does an independently reproduced clean public pipeline reach 0.913 on our
account, giving us a platform to build on instead of reasoning from another participant's score?

## P0-B — clean base + source-locked reverse-time

Source-locked mechanism: `reports/external/v19_reverse_time_block.py.txt`
(extract sha256 `eb94e2745f270649865b6b89f8927450`). Reproduction arm at the source-defined
`w=0.20`; no harmonic fusion, no blend sweep. Historical public effect `0.912 -> 0.914`.
**Causal question:** does reverse-time association reproduce its reported gain on a clean base?

## P0-C — v122 + reverse-time

Same mechanism applied to our own 0.908 baseline, detector/ILP/wrapper otherwise fixed.
**Causal question:** does reverse-time help *our* pipeline, or only theirs?

## Status

| candidate | kernel | output sha256 | audit | submitted | public |
|---|---|---|---|---|---|
| P0-A | `biohub-p0a-clean-913-repro` v1 | `8c1605b5944d25e4…` | **PASS 10/10** (re-verified independently) | **ref 55136759** | PENDING |
| P0-B | `biohub-p0b-clean-913-reverse-time` v1 | `4c285cae0c220a11…` | **PASS** (re-verified independently) | **ref 55136908** | PENDING |
| P0-C | `biohub-p0c-v122-revtime-run` v1 | `3370222f811fddc9…` | **FAIL** — 1 node out of volume | no | — |

**Slots consumed this cycle: 2 of 3.** Slot 3 held.

### P0-A result detail
237,298 rows = 120,797 nodes + 116,501 edges, 4 datasets, 314 divisions, t 0–99,
z 0–63, y 0–254, x 0–254. Kernel COMPLETE in 1522 s on T4x2. Pushed notebook asserted
byte-identical to the audited public source at build time; upstream re-pulled first and had
not drifted.

### P0-B result detail
Delta vs P0-A: nodes **+64** net (1,994 added / 1,930 removed after discounting 1,875 sub-2 µm
coordinate-smoothing shifts), edges **+103** net, divisions **−9** (314→305), **749 parent
reassignments** (0.66% of 112,993 targets with a parent in both arms). No harmonic fusion, no
blend sweep.

### P0-C — FAILS audit, withheld
`44b6_0b24845f` node 15274, t=43, **z=64** (valid 0–63). This is a latent **v122** defect, not a
property of reverse-time: v122 line-fit-smooths coordinates without clamping to the volume and
already parks 369 nodes exactly on z=63. The agent correctly did **not** clamp or drop the node,
which would have altered the pipeline mid-experiment. Delta vs v122 baseline: nodes +40, edges
+105, divisions **+10** (313→323), 850 parent reassignments.

### Causal read across P0-B and P0-C
Division counts move in **opposite directions** on the two bases — **−9** on the clean 0.913 base,
**+10** on v122 — and under 1% of parent assignments move either way. The reverse-time mechanism's
effect is **base-dependent**, so the public 0.912→0.914 claim does not transfer to our pipeline on
structural evidence alone. That is precisely what the two submitted arms are measuring.

### Licence status — UNRESOLVED
Neither the API, the SDK response, nor the `.ipynb` metadata carries a licence field. Kaggle's
notebook default is Apache 2.0; recorded as **UNVERIFIED-DEFAULT-APACHE-2.0**. The notebook
self-declares attribution to `pilkwang/biohub-cell-tracking-two-seeds-logit-blend` and flags
`metric_hack_used: false`, `public_output_used: false`. **A human should confirm on the notebook
page in a browser.**
