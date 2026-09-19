# What to change on the 0.947 pipeline, and why

> **SUPERSEDED, 2026-09-19.** The ranked gate-loosening plan below was measured
> on a weak local rebuild (adj 0.845, 9.3% orphan pool). On deployed-quality
> graphs the safe_div gates are at a LOCAL OPTIMUM in every direction: no single
> change beats them by more than one division event, loosening `tau`/`diverge`
> floods false positives, and `DIVERGE_UM = 0` was rejected by Kaggle's own
> validator (division FP 1 -> 15). Submissions s02/s03/s04 are retired.
> The real levers turned out to be STRUCTURAL: remove motion relink (+0.0295)
> and run gap2 after safe_div (+0.0076). Keep this file for the measurement
> method and the tried-and-failed log; do not act on the ranked plan.

Measured 2026-09-19 on all 199 training films and all 151 labelled divisions.
Everything here is reproducible from this repo: `scripts/17_safediv_gate_cost.py`,
`artifacts/safediv_gate_cost.csv`.

---

## The one-paragraph version

The public lineage's `safe_div` stage has six gates. **Four of them are
inert and two do all the blocking.** `SAFE_DIV_DIVERGE_UM` (2.25) and
`SAFE_DIV_SISTER_SYMMETRY_TAU` (0.6) between them reject **46 of the 61**
true divisions that reach the gates. Relaxing both takes proposable true
divisions from **15 to 46** — from 9.9% to 30.5% of every labelled division in
the training set. Those two values were set by sweeping **4 films**; nobody has
priced them against the 151 real events.

---

## The measurement

Deployed stage order is motion relink -> single-parent repair -> gap closing ->
gap2 -> **safe division** -> prune isolated -> short-track filter. So `safe_div`
sees the UNPRUNED post-relink graph. (Pricing the gates after pruning is the
wrong configuration and inflates the "daughter not detected" count from 7 to 27.)

For each of the 151 labelled divisions: match parent and both daughters to
predictions with the metric's own per-frame Hungarian at 7 um, then evaluate
each gate independently.

| gate | deployed | rejects | % of all 151 | recall if relaxed alone |
|---|---|---:|---:|---|
| existing_child | 10.0 um | 0 | 0.0% | 15 -> 15 |
| sister | 14.0 um | 2 | 1.3% | 15 -> 15 |
| parent | 9.0 um | 9 | 6.0% | 15 -> 15 |
| mutual_nn | on | 9 | 6.0% | 15 -> 15 |
| **symmetry** | **tau 0.6** | **29** | **19.2%** | **15 -> 25** |
| **diverge** | **2.25 um** | **32** | **21.2%** | **15 -> 29** |
| **symmetry + diverge** | | | | **15 -> 46** |

Four gates are individually inert: relaxing any one alone changes nothing,
because the two dominant gates still block.

### How far the gates are from what they reject

| gate | deployed | median of what it rejects | p90 | max |
|---|---|---|---|---|
| symmetry (asymmetry ratio) | 0.6 | **0.97** | 1.28 | 1.34 |
| parent `d(P,Q)` | 9.0 um | **10.53** | 14.82 | 17.04 |
| sister `d(C,Q)` | 14.0 um | **15.25** | 16.11 | 16.33 |

The symmetry gate sits at roughly **half the median** of the divisions it throws away.

---

## Ranked submission plan — ONE change per submission

| # | change | from -> to | true divisions unblocked |
|---|---|---|---|
| 1 | `BIOHUB_SAFE_DIV_DIVERGE_UM` | 2.25 -> 0 | 32 (21.2%) |
| 2 | `BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU` | 0.6 -> 1.35 | 29 (19.2%) |
| 3 | both together | | 15 -> 46 proposable |
| 4 | `BIOHUB_SAFE_DIV_MAX_UM` | 9 -> 12 | ~half of 9 |

Prior that these are the right *direction*: the only two safe-div loosenings
this lineage ever validated on the board -- sister 12 -> 14 and safe_div_max
7 -> 9 -- were each worth +0.001. Neither `diverge` nor `symmetry` has ever
been tested.

---

## What transfers, and what does not

**Recall cost transfers.** It is a property of real division geometry measured
against a detector the deployed pipeline also uses. A gate that rejects 21% of
real divisions here rejects roughly 21% there.

**Precision cost does NOT transfer.** It depends on the pool of unparented
candidate nodes, which is far cleaner in the deployed pipeline (adj_J 0.926)
than in the local reimplementation (0.845). Loosening these exact gates locally
made things *worse*: div_J fell 0.067 -> 0.027 because false positives rose 5x.

**Therefore the board is the only instrument that can price these.** One change
per submission. Expect the answer to be "some of the recall gain survives", not
all of it.

---

## The bigger prize, which is NOT a parameter

**59.6% of labelled divisions never reach the gates at all.** The dominant
cause is that the one-to-one Hungarian in motion-relink has already assigned
the true daughter to a different parent, and `safe_div` may only claim
unparented nodes. Of those blocked daughters, **53.5% are held by a parent with
no corresponding ground-truth edge** -- i.e. the incumbent edge is simply wrong.

No environment variable reaches this. It needs the assignment to be able to
contest a held target. A local attempt at that (`src/biohub/division2.py`)
passed its null-equivalence test but did not improve the score on a weak
baseline; it has not been tested on a graph of deployed quality.

---

## Other findings worth keeping

**Oracle ceilings** (`scripts/08_oracle.py`, all 199 films):
- 99.63% of GT nodes and **99.34% of GT edges have both endpoints matched** --
  detection is not the bottleneck; perfect detection buys only +0.0108 over
  perfect association.
- adj_J ceiling **0.9651** with current detections, **0.9713** with perfect ones.
- division ceiling **div_J 0.9536**.
- From the deployed 0.926 / ~0.20: association headroom **+0.039**, division
  headroom **+0.074**.

**The node-count term** (`scripts/09_nodecount.py`): ~+0.024 is genuinely
unclaimed (pruning captures only +0.004 of it), but `estimated_number_of_nodes`
is absent from test films and a predictor for it transfers 1-for-2 across
embryos (R2 0.937 one way, **0.494** the other, with a median 17% UNDERestimate
-- the direction that lands below n_est, where the published post-mortem
measured +0.013 offline and **-0.004 on the board**). Approved instead: a global
detection threshold 0.965 -> 0.99, +0.006 of ceiling, neither embryo crossing
parity.

**Ensemble components** (`scripts/13_ensemble_ablation.py`, 40 films):
detection TTA **+0.0217**, dual-seed **+0.0097**, and both land in `J` rather
than the multiplier. Bidirectional harmonic fusion and low-margin consensus are
worth **+0.0003 combined** -- they roughly double the transformer passes for
nothing, which matters against the 12-hour Kaggle limit.

**DeepCenter as a veto**: a real ranker (AUC **0.836**, **0.880** on the embryo
it never trained on, p=1.5e-5, beats a brightness control 0.836 vs 0.615) but
worth only **+0.0017**, because the candidate pool is ~180x larger than the
divisions in it. Useful band is exactly 0.20-0.25; above 0.30 true divisions
start dying. Also: `best.pt` is **epoch 2** and validation never improved over
the following 498 epochs.

---

## Tried and failed — do not repeat

| attempt | result |
|---|---|
| Learned division classifier on frozen UNet features | AUC **0.456** at the split frame -- chance. Nothing clears 0.70. |
| Anaphase hypothesis (signal precedes the split) | t-1 **0.510**, t-2 **0.484**. Both chance. Tested, not supported. |
| Consequence score as a ranker | Precision flat at ~0.10 from 12,507 proposals down to 196. Does not rank. |
| Cleaning the orphan pool first | Pool halved 9.33% -> 4.34%; division precision unchanged. |
| Loosening gates on recall evidence alone | TP 24 -> 53 but FP 5x; div_J **fell** 0.067 -> 0.027. |
| Fork-before-prune ordering | Slightly worse (0.0540 -> 0.0517): exposes the larger orphan pool. |
| Contested targets, tight gates | Fires on 2 of 6,063 proposals. No effect. |
| Synthetic division data (from the corpus) | Someone else's measurement: AP 0.98 on held-out synthetic, LB **0.910 -> 0.906**. |
