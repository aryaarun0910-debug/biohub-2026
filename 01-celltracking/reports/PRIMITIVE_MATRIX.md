# Primitive matrix — architectural cycle 2026-08-01

**Target:** 0.925 needs +0.011 from P0-B 0.914 · 0.950 needs +0.036.
**Objective:** exact pooled composite. **6bba carries 85.06% of edge mass — a 44b6-only figure is
never a headline.** Anchors every lane must reproduce before quoting a delta: pooled **0.665404**,
44b6 **0.759549**, 6bba **0.648965**.

Basis tags: `public` · `exact-pooled-OOF` · `cross-family-LOFO` · `in-family-CV` ·
`placeholder-proxy` · `GT-oracle`.

| lane | primitive | novelty | oracle ceiling | deployable Δ | family transfer | compute | decision |
|---|---|---|---|---|---|---|---|
| 1 | shape-aware localisation | — | — | — | — | — | RUNNING |
| 2 | acquisition-state inference | — | — | — | — | — | RUNNING |
| 3 | event-centric division state | — | — | — | — | — | RUNNING |
| 4 | CAP / track-as-point | real but insufficient | 35.45% of never-detected nodes @ w8 `[GT-oracle]` | **none — not measurable** | 85.14% / 32.23% | ~2 CPU-min, 0 GPU | **CLOSED — licence, Step 1** |
| 5 | dense self-supervised 3D motion | — | — | — | — | — | staged |
| 6 | counterfactual image critic | — | — | — | — | — | staged |
| 7 | pseudo-lineage substrate | — | — | — | — | — | staged |

---

## Lane 4 — CAP / track-as-point: CLOSED at Step 1 (licence)

**Four independent blocks, any one sufficient.**

1. **The CAP repository carries no licence at all.** GitHub API returns `license: null`, `/license`
   is HTTP 404, and none of its 104 blobs is a LICENSE/COPYING/NOTICE. Default copyright applies —
   all rights reserved. **This is stricter than the OrganoidTracker GPL-2 block: GPL-2 at least
   grants use; no licence grants nothing.**
2. **It is a CoTracker derivative and imports it at runtime** (`cap/models/core/cap/blocks.py:17`
   and both loss modules). CoTracker is **CC BY-NC 4.0**. A prize-bearing competition is not
   non-commercial, and the competition Winner Licence (MIT) cannot be granted over NC-derived code.
3. **The only obtainable weights are CC BY-NC 4.0** (`facebook/cotracker` on HuggingFace).
4. **It trains on CTC data**, already licence-blocked here, and the repo redistributes CTC
   evaluation binaries (~37 MB).

**Also: the advertised checkpoints do not exist.** The abstract claims "code and model checkpoints
are available"; there is no `.pth/.ckpt/.pt/.safetensors` in the tree, zero releases, and no
HuggingFace repo. Both the tracker weights and the required anchor-UNet weights are absent. The
released code also does not execute — `build_cap` is called with 4 args against a 1-arg signature,
`metrics.py` is imported but absent, and imports mix two package roots.

**The premise correction that matters more than the licence.** CAP is **not detection-free**.
`infer.py` seeds from `get_2D_anchor_points()`, which loads a separately-trained UNet segmenter and
takes mask centroids; mid-sequence "new cells" enter only through the division slots of
already-tracked cells. So CAP is **propagate-and-divide from detector-seeded points** — "no
detection" means "no *per-frame* detection", not "no detector". It therefore does not attack the
never-detected pool the way the lane premise assumed.

### The output worth keeping — a reusable prior for ANY propagate-and-divide proposer

`[GT-oracle, exact-pooled-OOF, 199 crops, E0c substrate, host matching rule: one-to-one bipartite,
7 µm, scale 1.625/0.40625/0.40625]` — artifact `inventory/propagator_reachability.json`.

| | pooled | 44b6 | 6bba |
|---|---:|---:|---:|
| GT nodes | 133,318 | 20,197 | 113,121 |
| never detected (no prediction within 7 µm) | **14,708 (11.03%)** | 895 (4.43%) | 13,813 (12.21%) |
| forward-reachable, unbounded | 45.46% | 88.60% | 42.66% |
| **forward-reachable, window 8** | **35.45%** | 85.14% | 32.23% |
| forward-reachable, window 16 | 40.37% | 88.60% | 37.24% |
| bidirectional, window 8 | 52.47% | 96.76% | 49.60% |
| bidirectional, window 16 | 58.87% | 97.65% | 56.36% |

**Any future lane proposing to attack the never-detected 43.80% must be priced against this table
before spending compute.** A seeded-propagation proposer has a 35.45% pooled ceiling at window 8 —
2.4× the 15% recall bar, so the idea is *not* dead on reachability — but the ceiling assumes
flawless propagation through frames where the detector saw nothing, says nothing about the 40.6%
precision arm, and is **2.6× better on the 15%-mass family** (85.14% vs 32.23%). The pooled number
is the 6bba number.

**Second finding, from an independent code path:** only **66 of 14,774** unmatched GT nodes had a
candidate within 7 µm and lost the bipartite assignment. The never-detected pool is a **genuine
detection miss, not arbitration** — which closes off "fix the matching" as a route to it, and
corroborates the never-detected/discarded split without reusing the attribution code.

**Third:** CAP is strictly 2D (10 × `Conv2d`, 0 × `Conv3d`, fixed 384×512). A tri-planar projection
over a 64-deep dense embryo volume would reintroduce cross-plane association — exactly the problem
the architecture is meant to bypass. Native 3D is a rewrite whose correlation neighbourhood grows
7²→7³ and which would have to train on our two families alone, i.e. the generic retraining CLAUDE.md
blocks, against the project's own finding that family-conditional shift is the binding constraint.

**One idea free to reimplement (ideas are not copyrightable):** the `P = 3` output head, where each
tracked point emits mother-continues plus two daughter branches with independent visibility flags —
making division a property of the *trajectory head* rather than a downstream fork classifier. Given
the division track's binding constraint is the selector, that framing is worth remembering. It needs
training data we do not have, so it is a note, not a queue item.

**Do not reopen** without a licence change by the authors *and* a CoTracker-free reimplementation.
