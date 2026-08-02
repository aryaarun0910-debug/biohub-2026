# Current handoff — next-cycle brief

**Updated:** 2026-08-02 (end of arm-B cycle)
**Branch:** `master`, clean and pushed
**Best public score:** **0.914** (P0-B) — **unchanged. Arm B did not move it.**
**Authoritative objective:** exact **pooled** composite. Min-fold is a robustness constraint.

## Read in order

1. This file
2. `reports/ARMB_RESIDUAL_CENSUS.md` — where arm B's remaining error actually is
3. `reports/PRIMITIVE_MATRIX.md` — every lane with oracle ceiling, deployable Δ, transfer, decision
4. `reports/CLAIMS.md` — generated from artifacts, never hand-edited (`scripts/claims_table.py`)
5. `reports/ENVIRONMENT_TRAPS.md` — 24 defects that each cost real time
6. `reports/submissions/CANDIDATE_LEDGER.md` · `reports/EXPERIMENT_LEDGER.md`

Verify state before doing anything:
```powershell
git status; .\.venv\Scripts\python.exe -m pytest -q            # expect 50
.\.venv\Scripts\python.exe scripts\claims_table.py --check     # expect 53
```

---

## 1. THE RESULT — arm B is publicly FLAT

| submission | what | public |
|---|---|---:|
| `55136908` | **P0-B** (deployment) | **0.914** |
| `55181562` | **arm B** solo, corrected invariant | **0.914** |

Expected ≈ +0.0045. Delivered **< 0.001** (both round to 0.914). Slots consumed: **10**.

Arm B was the one mechanism in this project's history to clear every gate and survive a
substrate transfer: P0-strict **+0.0079822** pooled, E0c **+0.0088059**, P(Δ>0) = 1.000
everywhere, min-fold clearing +0.005, 144/144 crops parity-exact. It moved **7.148%** of edges on
the test movies. The public score did not move at all.

**DECISION: retain P0-B as the deployment. Do NOT adopt arm B as the platform.**
**Do NOT discard it either — see §2, it is very likely unmeasurable rather than worthless.**

### The build itself was flawless — this is not a deployment defect

- Baseline arm re-run with the fix: sha256 `4c285cae0c220a11…`, **byte-identical to deployed
  P0-B**. The fix is provably inert on the P0-B path; the delta is attributable to the gate alone.
- Arm B artifact `83498f9e27d4453212f1d2e75b2b4999939733d1ce4fa183b0e4d670cc6ef6dd`.
- Independent audit: max out-degree **2**, max in-degree **1**, 0 dangling, 0 non-consecutive,
  0 duplicate nodes/edges, 0 negative coordinates. Degrees keyed on `(dataset, node_id)` (trap 24).
- Churn 7.148% vs the pre-registered 7.2%. Divisions 305 → 318. Edge length max 7.312 → 7.846 µm
  with **zero** edges over 10 µm or 14 µm — the "arm B can exceed the relaxed radius" watch item is
  **closed empirically**.

---

## 2. WHY — the public LB cannot resolve what we have been measuring

**This is the most important finding of the cycle. Read it before designing anything.**

GT annotation is **extremely sparse, and 13× denser in one family than the other**:

| family | crops | median annotated nodes | median annotated edges | median `estimated_number_of_nodes` | coverage |
|---|---:|---:|---:|---:|---:|
| `44b6` | 71 | 214 | 209 | 32,681 | **0.655%** |
| `6bba` | 128 | 826 | 800 | 9,691 | **8.529%** |

Total annotated edges across all 199 training crops: **128,883**.

The edge Jaccard is computed **only over annotated GT cells**. We predict ~24,500 edges per movie
and only a few hundred are ever scored.

**Consequences, and they are severe:**

- The public LB is 4 movies. Extrapolating train density, it is scored over roughly
  **~2,000 GT edges**. One edge ≈ **0.05%** of Jaccard. A **+0.008** effect requires about
  **+16 net correct edges** among the annotated subset.
- Arm B churned ~8,335 edges. Uniformly, ~143 of those touch annotated edges. Getting +0.008 needs
  ~56% of them correct versus 44% wrong. A genuinely small edge, easily invisible at this size.
- Our OOF instrument spans **128,883 annotated edges over 199 crops — roughly 64× more sensitive
  than the public leaderboard.**

**Therefore: arm B being publicly flat is NOT evidence that it is worthless.** With
P(Δ>0) = 1.000 over 144 crops it may still be worth ~+0.008 on the private set, which has many more
movies and correspondingly more resolution. The four public movies simply cannot see it.

**Rule for the next cycle: stop using the public LB as a measurement instrument.** It is a
confirmation device for large effects (≳0.005) only. Every slot spent to resolve something smaller
is a slot wasted. OOF is the instrument; the LB is the smoke test.

**Do not over-read this either.** It is an inference from *training* annotation density; we have no
GT for the test movies. It is consistent with every observation we have, but it is not proven.

---

## 3. Where arm B's remaining error actually is

`reports/ARMB_RESIDUAL_CENSUS.md`. 50 OOF crops, exact scorer, corrected arm-B wrapper, `prob = 0`
proxy. **No GPU, no slot.** Reproduce with `scripts/armb_census.py --stride 4 --workers 6 --gate B`.

```
score 0.734918 = adj_edge_J 0.734418 + 0.1 · div_J 0.005000
edge     TP/FP/FN : 28904 / 4193 / 6704
division TP/FP/FN : 1 / 151 / 48     precision 0.658%   recall 2.04%
total_node_ratio  : −0.147596
```

| oracle (a repaired FN becomes a TP) | Δ |
|---|---:|
| **edge recall perfect** | **+0.170959** |
| division perfect (both) | +0.099500 |
| edge precision perfect | +0.082648 |
| **division recall perfect** | **+0.024000** |
| **edge: 10% of FN recovered** | **+0.017096** |
| division precision perfect | +0.001541 |
| node ratio → 0 | **−0.008205** |

**Edge recall is worth ~7× everything division-related.** 6bba carries **95.2%** of missed edges
(6,380 vs 324) at score 0.7012 against 44b6's 0.9014.

**`total_node_ratio = −0.1476`: we under-predict nodes by 15% and the count adjustment pays us
+0.0082 for it.** Any primitive that adds nodes surrenders part of that bonus before earning
anything. Node count is not a free axis.

**Caveat that must travel with these numbers:** the detector is the public all-data 50-epoch pack,
so although our folds are true LOEO (fold 0 trains 6bba → tests 44b6; fold 1 the reverse), the
*detector* has seen both embryos. The absolute 0.7349 is optimistic on the detection dimension.
The ranking is unaffected — leakage hits every bucket alike — and true edge-recall headroom is
therefore **larger** than +0.171, not smaller.

---

## 4. Division sister-gate — designed, measured, CLOSED as a relaxation

The live gate is **`SAFE_DIV_SISTER_MAX_UM = 8.5`**, not the `DIV_SISTER_MAX_UM = 8.0` that
correction 6 killed (re-confirmed dead: `OUTPUT_DIVISION_GEOMETRY_FILTER` defaults `"0"` and is
never set in cell02). **Designing against the dead constant would repeat correction 6.**

The fault is real: across 199 crops and 151 true divisions, **GT sister p50 = 10.570 µm**. The gate
sits at 8.5 — *below the median* — and rejects **70.9%** of true divisions, versus 42.4%
(`SAFE_DIV_MAX_UM = 4.66`) and 40.4% (`SAFE_DIV_EXISTING_CHILD_MAX_UM = 7.65`). Only 26.5% of true
divisions clear all three.

**But relaxing it loses.** Break-even is 0.498% *on the division term alone*, except every safe
division also emits an edge that lands in edge FP:

| added k | div gain @ q=5% | edge cost @ q=5% | net |
|---:|---:|---:|---:|
| 100 | +0.001534 | −0.001714 | −0.000180 |
| 200 | +0.002321 | −0.003419 | **−0.001098** |
| 400 | +0.003121 | −0.006804 | **−0.003683** |

Added candidates need **≈20–25% precision**. Relaxing 8.5 → 12.0 µm unlocks **+23 of 151** true
divisions (1.58×) while candidate volume grows as (12/8.5)³ = **2.81×** → achievable precision
**≈2.7%** → net **≈ −0.0036. FALSIFIED on CPU, for free.**

Seventh instance of oracle-clears/selector-fails.

**Not falsified:** a **re-aim at constant admission count** (arm B's own move — change the gate
quantity, not the radius) incurs no extra edge cost, so any precision gain is pure. Ceiling is
**+0.024** even at perfect recall. My GT-density normalisation test of that idea is **WITHDRAWN as
invalid** — GT is ~0.65–8.5% as dense as the prediction graph, so it measured a variant nobody
would ship.

---

## 5. Everything else is closed

| mechanism | verdict |
|---|---|
| **arm B flow gate** | OOF +0.0080, **public +0.000** — retain on shelf, do not adopt, do not discard |
| suppress-all | +0.0016970 P0-strict but 44b6 −0.000619 — do not spend a slot alone |
| localisation / recentering | oracle +0.009, deployed −0.0088, sign-opposite — CLOSED |
| division: flat mother classification | needs AUC 0.983–0.999, measured 0.86–0.92 — CLOSED |
| division: branch-emergence proposer | GT-oracle 19.5× over budget — CLOSED |
| **division: sister-gate relaxation** | **net −0.0036 — CLOSED (this cycle)** |
| counterfactual image critic | foreclosed on arithmetic — never launched |
| bidirectional propagation | fails 56× (0.712% precision vs 39.83% break-even) — CLOSED |
| CAP / track-as-point | licence-blocked 4 ways — CLOSED |
| node budget · reverse-time · acquisition-state (arms C/C2/D/E/E2) | closed, negative or vacuous |
| dense registration as a motion vector | worse than assuming zero motion — DO NOT FUND |
| public frontier kernel `prvsiyan/…` | **our own 0.913 ancestor minus reverse-time. Nothing to take.** |

**D2 daughter-pair selector stays on the shelf** (+0.003541 paired, `pi_vis`-invariant), waiting on
a mother set no route can produce.

---

## 6. Corrections — eleven. Read before trusting older docs.

1. **The out-degree-3 diagnosis was wrong and its prescribed fix was a no-op** (this cycle).
   `motion_relink_edges` runs first, replaces the whole edge list, assigns one-to-one → out-degree
   ≤ 1 by construction. The real site is `add_safe_divisions_postlink`, which dedupes proposals by
   **target** only and never advances `out_by_source`. Locked by
   `tests/test_lineage_degree_invariants.py`.
2. **The fix is a one-edge SWAP under a binding cap**, not a pure removal — an earlier note in this
   cycle claiming "adds zero edges" is withdrawn. In production the cap was slack, so the net effect
   was exactly one edge removed.
3. **suppress-all sign**: −0.001728 was an edges-only artifact; **+0.002706** through the wrapper.
4. **node-budget mechanism**: not the node ratio, it is the **`d_tp/d_fp` composition** of deleted content.
5. **The 22/26 substrate inverted**: 44b6 22/26 but 6bba **66/125** — worst measured.
6. **Temporal NMS is NOT free** — destroys 38–48 of 92 true forks under other rankers.
7. **"Re-aim, not widen" was selectively quoted** — the tight-6 µm row was always +0.169%.
8. **The sister-gate lead was DEAD CODE** — `DIV_SISTER_MAX_UM = 8.0` sits in a filter no notebook
   enables. Re-confirmed this cycle. The live constant is `SAFE_DIV_SISTER_MAX_UM = 8.5`.
9. **The arm-B brief was ambiguous** — the arm gates on `|target − (source + kNN16_flow(source))|`,
   NOT the `motion` term.
10. **"P0-B has 8 forks"** was a units error — it has **305** (confirmed exactly on the artifact).
11. **Arm D falsified Lane 2's own bilateral core**: +0.003389 GT-oracle → **−0.0031111** exact.

---

## 7. The rules that generalise

**The instrument is the constraint, not the mechanism.** The public LB resolves ~0.0005 per
annotated edge over ~2,000 edges. OOF is ~64× more sensitive. Never spend a slot to resolve
something smaller than ~0.005.

**Substrate transfer is the dominant failure mode** — seven instances, **zero** confirmed successes
now that arm B has gone flat publicly. Assume every inherited constant is invalid until re-measured
on the substrate it is applied to.

**Oracle-clears / selector-fails** — localisation +0.009→−0.009, component retention
+0.0087→−0.0078, division +0.0646→+0.0001, propagation +0.059→−0.45, sister gate +0.024→−0.0036.
**Headroom has never been the constraint; selection has.** Do not fund another selector over an
existing candidate population without a base-rate argument first.

---

## 8. What to do next — in order

1. **Do not submit anything yet.** Nothing on the shelf is worth ≳0.005 on the public instrument.
2. **Settle the instrument question first.** Confirm the annotation-density inference on the test
   movies as far as possible, and decide explicitly whether the private set's larger movie count
   makes OOF-positive/public-flat mechanisms (arm B, suppress-all) worth carrying into the final
   selection. **This is the highest-value open question in the project** and it is free.
3. **Then attack 6bba edge recall** — +0.171 at oracle, 10% capture clears the +0.011 to 0.925,
   95% of the mass in one family. Before funding: a base-rate argument, because this is again a
   selector over an existing candidate population. Re-measure every constant on 6bba; inherit
   nothing from 44b6.
4. **Do not** fund divisions (caps at +0.024, and the only live relaxation is falsified), node-count
   growth (costs the −0.0082 adjustment bonus), or anything from the public frontier kernel.

---

## 9. Guardrails (unchanged)

No public-score exploitation, negative-time or out-of-volume nodes, synthetic hubs/forks. No
family/crop routing. No tuning on the four visible placeholder movies. External data needs URL,
licence, checksum, provenance — CTC, OrganoidTracker (GPL-2) and CAP (no licence) are blocked.
**Submissions come from NOTEBOOKS only.** Stage explicit paths when committing; agents run
concurrently and `git add -A` sweeps unreviewed work. `.claude/settings.json` and `.gitignore` are
deliberately dirty and must stay untouched.

**Known gap, unfixed by choice:** the deployed export clamps coordinates only at zero
(`max(0, round(v))`), with no upper bound. `OUTPUT_VOLUME_GUARD` exists in
`src/biotrack/wrapper.py` but defaults `"0"` and is **absent from the deployed notebook entirely**.
Measured impact: **2 nodes in 340,674** exceed the upper z bound by 1 voxel; the ledger's
independent P0-CR measurement found exactly 1 node in ~120,673. It is a rule-compliance gap, not a
score lever. Fold the ~6-line clamp into the next build, not into a validated artifact.
