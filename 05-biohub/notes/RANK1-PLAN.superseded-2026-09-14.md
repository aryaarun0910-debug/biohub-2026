# Current campaign plan — fork the 0.947, beat it on divisions

Supersedes the validation-gate plan (kept as
[RANK1-PLAN.validation-gate-2026-09-12.md](RANK1-PLAN.validation-gate-2026-09-12.md)), whose
correctness findings stand but whose *sequencing* is overtaken: it assumed we would build a
detector. We do not need to.

## What changed

**The floor is the public 0.947, and it is free.** Every clean 0.947-family kernel attaches the
same three **CC0** datasets — a trained DeepCenter UNet3D, a trained temporal UNet3D, and the
tracking support pack. `detect` being a stub never required training anything.

**Reproduced and submitted.** `aryaarun07/biohub-repro-947` is byte-identical and T4-pinned;
submission 27 is pending. Two traps are now settled and both matter for the *final* submission,
which cannot be retried:

- Pushing with `enableGpu` alone lets Kaggle hand you a **P100 (sm_60)**, which the pinned
  PyTorch cannot execute at all. `machineShape="NvidiaTeslaT4"` is the field that binds;
  `acceleratorType` is accepted and silently ignored.
- A kernels-only entry must be submitted **explicitly, by kernel reference, via the CLI**. The
  REST endpoint is the file path and rejects it.

**Their pipeline runs locally against our ground truth**, scoring edgeJ **0.9261** against their
published 0.926, at `node_recall 1.0000`. Detection is not their bottleneck, which is consistent
with FOCUS-3D and the SSL routes having failed to beat 0.947 in public hands.

## Where it is beatable

The three 0.947-family kernels differ in **4 of 54** settings; fifty are consensus, so the
threshold surface is mined out. (Our own `fork_sister_um=18` was already tried at real density
and lost: 0.946 against 0.947 at 14.0.)

Nobody varies the **division gates**. EXP-21 prices them against ground truth:

| gate | rejects (Kaggle GT, n=151) | (Zebrahub, n=761,849) |
|---|---|---|
| `SISTER_SYMMETRY_TAU` 0.6 | **37.7%** | 54.9% |
| `DIVERGE_UM` 2.25 | **48.3%** | 85.6% |
| `MAX_UM` 9.0 | 19.9% | 14.6% |
| `SISTER_MAX_UM` 14.0 | 12.6% | 2.2% |
| **all combined** | **74.8%** | 87.0% |

They discard three quarters of real divisions **before ranking**. At perfect precision those
gates cap divJ at 0.252 — almost exactly the 0.231 they publish. That is the strongest evidence
we have that their division term is **recall-limited, not selection-limited**, and it corrects
our own earlier reading.

## The play, in order

1. **Loosen** `SAFE_DIV_DIVERGE_UM` and `SAFE_DIV_SISTER_SYMMETRY_TAU`. Both env-settable, both
   untouched by the entire family. Dropping divergence admits 48.7% instead of 25.2%: divJ 0.355
   at p=0.8, worth **+0.012** of score.
2. **Rank** the enlarged pool. Their selector is one line — `parent_dist + 0.15*sister_dist` —
   which measures *worse* than `parent_dist` alone (AUC 0.776 vs 0.806), because sister distance
   ranks at 0.551. Our learned ranker beats it by **+0.079 AUC held out** on 6bba.
3. **Let the cap enforce precision.** `GLOBAL_FRAC_CAP` 0.00375 sits just under the true
   biological rate of 0.00391 measured on Zebrahub, so it binds once recall is restored.

A ranker is worth most when there is a surplus of candidates to order — which their gates
currently prevent from existing. The steps compose; do them in that order.

## Discipline

- **Measure offline, not per submission.** `tools/div_sweep.py` reimplements their selector with
  the ranker pluggable; `kernels/gen-train-graphs` produces linked graphs for all 199 train
  datasets on a T4 so variants cost seconds instead of a 2-hour run each.
- **Leave-one-embryo-out, both directions, report the minimum.** Train and test are
  embryo-disjoint by the host's own statement.
- **Scoring takes up to 8 hours.** Never serialise experiments behind it; never block on polling.
- Oracle-detection results are conditional diagnostics at 6.7 cells/frame against a real 237.
  They are not a submittable ceiling.
