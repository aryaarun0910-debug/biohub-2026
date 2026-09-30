---
tags:
  - finding
  - key
---

# Measuring On The Films That Are Actually Scored

**The most important measurement in the project**, and it should have come first.

The submission covers exactly **4 films**. We hold their ILP graphs
(`artifacts/s05_output/.../unet_transformer/split_0/`, verified byte-identical
to the relink-ON run, so pure ILP output) and their ground truth is in
`data/.../train/`. So the actual scored films can be scored locally.

| change | validator (8 films) | **SCORED (4 films)** | verdict |
|---|---|---|---|
| [[s05]] over base | +0.01794 | **+0.02707** | holds, and is *larger* |
| [[s08]] over s05 | +0.00748 | **+0.00000** | neutral — validator-only |
| [[s09]] over s08 | +0.00487 | **−0.00086** | **sign flip** |
| s09 over base | +0.03030 | +0.02621 | worse than s05 alone |

**[[s05]] is the best configuration on the films that count.** It gains on all
four individually, including the heaviest:

| film | GT divisions | base | s05 | delta | GT edges |
|---|---|---|---|---|---|
| 44b6_0113de3b | 0 | 0.86847 | 0.96051 | **+0.09204** | 53 |
| 44b6_0b24845f | 0 | 0.97812 | 0.97925 | +0.00113 | 52 |
| 6bba_05b6850b | 0 | 0.96172 | 0.96689 | +0.00517 | 864 |
| 6bba_05db0fb1 | 3 | 0.84089 | 0.87974 | **+0.03885** | 1304 |

**Why [[s08]] evaporates:** the scored films have a division ledger of **0/6/3**,
so `divJ = 0` in every arm. s08's whole gain was one *division*
([[Division Recovery in 44b6_341df25f]]) and there is no division to recover here.

**Why [[s09]] flips:** it helps `6bba_05b6850b` (+0.00686, weight 861) and hurts
`6bba_05db0fb1` (−0.00585, weight 1258). The heavier film loses.

⚠ **Caveat:** labels on the scored films are sparse (0.16%–13.5%), only ~2,273 GT
edges and 3 divisions total. Noisier than the validator set — but it is the
actual target, and the s05 signal is broad and large.

Scripts: `102_on_real_scored_films.py`, `103_relink_on_scored_films.py`.

Related: [[Manifest Contamination]], [[Validator Films]], [[Test Films]]
