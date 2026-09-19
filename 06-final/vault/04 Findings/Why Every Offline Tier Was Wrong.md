---
tags:
  - finding
  - key
---

# Root Cause: We Were Scoring On The Model's Own Training Labels

[[s05]] scored **0.945** against a 0.947 baseline while every offline tier said
+0.022 to +0.027. This is why.

**Step 1 — it is not the chain.** `scripts/115_full_chain_local.py` extracts the
notebook's own `filter_output_graph` — the real chain, including single-parent
repair, single-child repair, the short-track rescue and the DeepCenter vetoes
that our ports omit — and runs it on the 4 scored films. It still says
**+0.02552**. The chain was never the explanation.

**Step 2 — it is the ground truth.**

| film | GT nodes | GT edges | n_est | labelled |
|---|---|---|---|---|
| 44b6_0113de3b | 52 | 50 | 25,755 | **0.20%** |
| 44b6_0b24845f | 51 | 49 | 32,795 | **0.16%** |
| 6bba_05b6850b | 861 | 845 | 6,362 | 13.53% |
| 6bba_05db0fb1 | 1,229 | 1,183 | 69,800 | 1.76% |
| **total** | **2,193** | **2,127** | **134,712** | **1.63%** |

The `.geff` we score against exists **only in `train/`**. `test/` holds `.zarr`
images and **no annotations** — Kaggle keeps the test ground truth private.

And the secondary manifest lists **train = all 199 films**, so those labelled
nodes were **training targets for the detector we are evaluating**.

> **We have been evaluating on the model's own training labels, on a 1.63% sample.**

**The mechanism this explains:** on cells the detector was fit to, the
[[ILP Linker]]'s raw assignment is already near-optimal, so
[[Motion Relink]]'s distance-based rewiring can only break correct edges —
removing it looks like a large win. On the other 98.4%, which is what Kaggle
actually scores, the ILP is less reliable and relink's rewiring is net helpful.

**[[Manifest Contamination]] was right and my rebuttal was wrong.** I argued
in-sample-ness was *symmetric* because the 4 scored films are in the 199-film
train list. True of the **films**; false of the **labels**. Kaggle scores against
test annotations the model never saw; we score against train annotations it was
fit to. The asymmetry is in the labels, not the films.

**What this invalidates:** [[Scored Films Measurement]], the validator tier,
[[The 32-Film Verdict]], and the 199-film tier from
[[Local Pipeline On The Mac]] — **all of them** share this flaw. More films does
not help; every film's annotations are training targets.

**What survives:** the board, at 5 submissions a day and ~8 h latency. And the
one honest local route, which is what the discussion post recommended: train a
detector with films **held out**, then evaluate post-processing on those films.
That measures generalisation — at the cost that conclusions from a
differently-trained detector need not transfer to the public stack
([[Transfer Lesson]] again, one level up).
