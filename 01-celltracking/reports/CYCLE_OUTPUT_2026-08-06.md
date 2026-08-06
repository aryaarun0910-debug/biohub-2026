# Cycle output — 2026-08-06 · halted deliberately

**Start:** `2c91bc6` · **End:** `6cdd016` (pushed, `0 0`) · **Tests:** 90 → **224**
**GPU spent:** 0 · **Kaggle pushes:** 0 · **Submissions:** 0 · **Score movement: 0.000**

Halted on instruction after Stage A. Nothing is half-merged; two branches carry
unmerged work with exact resume points recorded in §6.

---

## 1. The honest headline

**This cycle produced no candidate and no score movement. It was not supposed to.**

It began intending to spend ~11.5 T4-hours on a 199-crop D1/D1-F export. That export was
stopped, and the stopping is the deliverable: the instrument was wrong in **five independent
ways, four of which would have returned a plausible answer rather than an error.**

Public score remains **0.915**. Target 0.930 needs **+0.015 public**, which is **15× the
largest public gain this project has ever recorded** (P3, +0.001; arm B, +0.0085 OOF → 0.000
public). There is no calibrated OOF→public map and every attempt to assume one has failed.

---

## 2. Defects found, all verified independently before acceptance

| # | defect | would it have failed loudly? |
|---|---|---|
| **B1** | kernel emits no `d1_class`/`matched`; the partition was a missing CPU stage | no — audit simply unrunnable |
| **B2** | `d1f_probe.py`: H0 fitted not loaded; `pi_crop=None` hardcoded ⇒ **H1–H4 byte-identical** | **no** — four identical arms read as convergent evidence |
| **B3** | features identity-view, logits post-TTA | **no** — would have returned a verdict |
| **B4** | polars schema drop kills gt-only columns on **28/199 crops** | **no** — reports `COMPLETE` with correct `gt_rows` |
| **B5** | family and checkpoint perfectly confounded; the two 32-D bases are independent | **no** — returns "representation deficit" w.p. ≈1 |

Plus two caught while writing the fixes:

- **The v6 `unet_out` trap.** Every research lane proposed accumulating the TTA mean into
  `unet_out`; one wrote the code. `unet_out` is read at L442/L445 by `predict_edges`, *after*
  the TTA block, so that would have silently changed every edge feature and destroyed the 0.915
  association substrate — under cover of a bug fix.
- **`augpath_oracle.json` was the retracted post-wrapper run** (40 crops, not 199) sitting at
  the script's own default `--out`. Quarantined; `--out` is now required.

---

## 3. A defect in the deployed detector

The view written as the anti-transpose,

```python
torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)
```

**is exactly `imgs.flip(-1)`** — already view 1. Verified bit-exact, twice, independently.

So the deployed detector makes **8 encode calls over 7 distinct views**: `flip(-1)` carries
weight 2/8, the true anti-transpose 0/8, divisor 8. The measure is non-uniform on D4, and a
7-element subset of an order-8 group is never a subgroup — **the deployed average is not a
group average.** Against a true uniform D4 average: max |Δlogit| **1.33**, and **201 of 1,780
accepted peaks (~11%) change identity** on one frame.

**Decision: ship it unchanged.** v6 exists to describe the *deployed* detector. Correcting the
view set is a detector change — it moves the node population, forces a `predict_edges` rerun,
and voids the 0.889 anchor. Locked by a test asserting the collision is present and intentional.

---

## 4. Claims corrected — mine unless noted

| claim | correction |
|---|---|
| "TTA is exactly 4 views" | **8 encode calls, 7 distinct.** I read `vendor/`, which is replaced at build time. **The base file is not the run.** |
| "3-crop partition, GT 3,027" | GT **3,079**, M **1,521**. The total row summed GT/M over two crops but submission-matched over three. |
| the partition covers the smoke's families | **It does not.** `44b6_0113de3b` is 52/52 matched, so **every unmatched GT is 6bba** — `D = 0` and the class split are 2-crop, 6bba-only, zero 44b6 representation. |
| "M1 is CPU-exact" | **False.** Freezing peak *extraction* does not freeze the *accepted* set; a changed node population changes existing edge logits through cross-attention and source-softmax. |
| "`\|T ∪ L\|` < 3,400 at any precision" | **Wrong.** +0.015 needs **≈2,221** perfect recoveries. 3,400 silently embedded precision 0.653 and conflated admitted candidates with a TP floor. |
| "assert `Y == X`" | **Refuted.** All 8 inverses round-trip on non-square input; the output is mis-*weighted*, not corrupted. |
| "signed residual ⇒ abort catches wrong inverses" | **It does not** — injected inverse bugs are exactly zero-mean. Needs magnitude + accepted-peak-set equality. `1e-4` was 18–26× loose against a derived bound of ~7.3e-6. |
| "the TTA patch guard degrades silently" | **Overstated.** The next patch anchors on text containing `_nv`, so a failed patch hard-fails one step later. |
| "SOURCE never uses the checkpoint trained on its own family" | **Self-contradictory** (agent-caught). With two embryos and two LOEO checkpoints an out-of-sample source cell cannot exist. |
| "stratify by crop size" | **Zero variance** (agent-caught) — all 199 crops are exactly 100×64×256×256. Axis replaced by `pred_nodes`. |

---

## 5. Landed on master

| | |
|---|---|
| `scripts/d1_postprocess.py` | CPU M/C/T/L/D partition; six C6 defects repaired, each falsified by re-injection (**10/10 probes fail**) |
| `scripts/build_d1_factorial_manifests.py` + `data/d1_factorial/` | 2×2 checkpoint×family manifests, three tiers, byte-reproducible from one census |
| `scripts/kaggle_edits/d1_response_audit.py` | trap-21 fix: `infer_schema_length=None` + hard column contract |
| `scripts/augpath_oracle.py` | retracted artifact quarantined; `--out` required |
| `reports/D1_V6_SPEC.md` | v6 spec incl. §0b audit corrections |
| `reports/DECISION_PACKAGE_2026-08-06.md` | mechanism map, ranked candidates, kill rules, do-not-run list, §0 corrections |
| `.gitattributes` | pins factorial manifests to LF |
| tests | **90 → 224** |

**Verified measurements (unchanged by any repair):** partition GT 3,079 · M 1,521 · C 289 ·
T 849 · L 420 · **D 0**; `44b6_0113de3b` 52/52 pregraph vs 49/52 submission. Under submission
authority 44b6 gives M=49 **C=3** — those three GT have an *accepted* peak within 7 µm, so they
are wrapper assignment losses, not detection failures. Independent corroboration that pregraph
is the detection-honest substrate.

---

## 6. Unmerged work — exact resume points

**Not merged because untested. Nothing is lost; both branches are committed.**

### `worktree-agent-ab3310e9b5a4cd85d` @ `ffb2dd1` — v6 TTA export (Agent 2)
Worktree: `.claude/worktrees/agent-ab3310e9b5a4cd85d`. 5 commits, 11 files: both smoke
notebooks regenerated, `d1_response_audit.py`, `d1_inject.py`, both specs,
`assemble_p3_d1_smoke_spec.py`, build manifests, composition inventory,
`tests/test_d1_v6_export.py`.
**Base is `d540c38` — master has moved to `6cdd016`; rebase before finishing.**
Outstanding: was mid-way through replacing the refuted `Y == X` test with peak-set and guard
coverage. Requirements list is `reports/D1_V6_SPEC.md` incl. §0b.

### `worktree-agent-a6d67b0ce12e98f08` @ `68dd81d` — D1-F rewrite (Agent 6)
Worktree: `.claude/worktrees/agent-a6d67b0ce12e98f08`. 3 commits, `scripts/d1f_probe.py` only:
**2,291 lines, +2,258 vs master, syntax-valid, master already merged in.**
Verified: `REPRESENTATION DEFICIT` appears **only as a prohibition** (docstring L30,
`NOT_PERMITTED` L1812) — correction C2 implemented as intended.
**Outstanding and blocking: `tests/test_d1f_probe.py` does not exist.** 2,258 lines of
instrument code are unverified. Was running fixtures for three regimes when halted.

### Never started
Agents 7 (M1 ceiling), 8 (dense replay), 9 (CPU gates), 10 (red team). **Agent 9's Gate 2 — the
30-CPU-minute recall-ceiling re-audit at true nuclear density — is the highest-leverage
unstarted item in the project**, because it can re-price the +0.10332 detection oracle before
any GPU is spent.

---

## 7. Two decisions outstanding

1. **`.gitignore:2` ignores `data/` wholesale.** The factorial manifests are tracked only by
   force-add, so regenerations are invisible to `git status`. Clean fix is a
   `!data/d1_factorial/` negation. **Not done — `.gitignore` is under a standing do-not-touch
   instruction.**
2. **FULL does not fit: 22.44 T4-h against ~11.5 remaining.** Cannot be sharded away — every
   shard fits a session (largest 7.14 h = 79%) but ≥4 kernel runs are required and they cannot
   merge, because the LOEO retarget binds one fold at module level and rebinds `TEST_DIR`
   globally. Options: reduced (TARGET whole, SOURCE subsampled) **12.74 T4-h**, still over; or
   **split 1 only, 7.88 T4-h** — the primary experiment, ~85% of the objective, inside the
   existing allocation. **Recommended.**

---

## 8. Where 0.930 stands

Detection is the only pool large enough (oracle **+0.10332** vs association **+0.032931**).
+0.015 **OOF** needs ≈**2,221** perfect node recoveries of 15,296 missing — 14.5% recall at
perfect precision, or ~2,678 TP at 80%. It must then survive the surrendered node-ratio bonus
(−0.008633), assignment stealing (99.7% of FP cost), and an OOF→public transfer that has so far
been ~zero. Correction C5 compounds it: every measurement we own is **strict LOEO, which is not
the public P3 detector** (0.525 primary + 0.475 aligned secondary behind a per-frame retention
guard).

Two swarm findings make the ceiling *less* certain, not more: the 1.0000 recall ceiling was
measured on annotated centres 2.3× sparser than the real population (true spacing 9.68 µm vs
axial PSF ~10 µm), and the deployed TTA is a non-uniform 7-view average.

Ledger base rate: ~20 mechanisms decisive, **2 promoted, 1 cleared min-fold** — 5–10% each.

> **0.930 is not on an evidenced path.** Nothing on the shelf clears +0.020. The realistic
> options are to widen beyond head-only work — which means the encoder programme C2 forbids
> funding on a null probe — or to accept that remaining headroom here may be thousandths.

---

## 9. Standing rules unchanged

No submission from a probe or sampled-row metric. Promotion ≥ +0.015 OOF; submission ≥ +0.020
(**policy, not a calibrated transfer law**). The 199 crops come from **two embryos** — crop-block
bootstrap measures within-embryo variation and does not estimate private-embryo generalisation;
report both transfer directions separately. **The base file is not the run.**
