# Current handoff

**Updated:** 2026-08-02
**Branch:** `master` (in sync with origin)
**Best public score:** **0.914** (P0-B) — unchanged
**Authoritative objective:** exact **pooled** composite. Min-fold is a robustness constraint.

## Read in order

1. This file
2. `reports/PRIMITIVE_MATRIX.md` — **the live document.** Every lane/workstream of the architectural
   cycle with oracle ceiling, deployable Δ, family transfer, basis tag, decision
3. `reports/CLAIMS.md` — **generated from artifacts, never hand-edited** (`scripts/claims_table.py`)
4. `reports/ENVIRONMENT_TRAPS.md` — **24 defects that each cost real time**
5. `reports/submissions/CANDIDATE_LEDGER.md` · `reports/EXPERIMENT_LEDGER.md` · `reports/NEXT_DECISION.md`

## 1. THE ONE THING TO DO NEXT

**Arm B is measured, validated, built, and blocked on a one-line bug. Fix it and submit.**

| kernel | status |
|---|---|
| `biohub-p2-armb-baseline` | **COMPLETE** — sha256 `4c285cae0c220a11…`, **byte-identical to live P0-B**. Harness has zero drift. |
| `biohub-p2-armb-flowgate` | **ERROR** — P0-B's own guard: `6bba_05db0fb1: invalid lineage degree` |

Diagnosed exactly from the failed session's artifact: **one node in 121,003 has out-degree 3**; zero
in-degree violations. The relink's assignment is one-to-one so it cannot make in-degree 2, but an
admitted relink edge can attach a **third** child to a source that already gained two from safe
divisions / single-parent repair / gap close. **The out-degree ≤ 2 invariant is assumed downstream
but never enforced inside the relink** — arm B admits more pairs and trips a latent bug.

**Repair:** enforce out-degree ≤ 2 at edge admission in `motion_relink_edges` (or drop the
lowest-probability third child). Then:
```powershell
$env:PYTHONUTF8=1
.\.venv\Scripts\python.exe scripts\kaggle_factory.py push --spec scripts\kaggle_specs\p2_armb_flowgate.json
# ~25 min, then: status -> fetch -> audit -> per-crop delta vs P0-B -> submit
```
**Do not re-run the baseline — it already passed.** Kaggle allows 2 concurrent GPU sessions.

## 2. What arm B is, and the number

**Change the gate QUANTITY in `motion_relink_edges` from raw source–target distance to the
kNN16 flow-compensated residual.** Same 6/10 µm radius, same cost, same nodes. **Routes on nothing.**

**CRITICAL — do not rebuild this from prose.** The measured arm gates on
`|target − (source + kNN16_flow(source))|`, where flow is a GT-free per-frame kNN16 median of raw
prediction-graph displacements. It is **NOT** the `motion` term at `wrapper.py:335` (that is a
per-node velocity extrapolation the relink populates itself). A previous brief said "flow-compensated
motion residual" and the literal reading is **the wrong arm**. The correct patch is
`scripts/kaggle_edits/armb_flow_gate.py`; its function text hashes to `c9ee69affc74a8d8…`.

| substrate | pooled | 44b6 | 6bba |
|---|---:|---:|---:|
| **P0-strict (deployment)** | **+0.0079822** | +0.0167567 | +0.0067055 |
| E0c | +0.0088059 | +0.0074359 | +0.0090668 |

P(Δ>0) = 1.000 everywhere; min-fold clears +0.005. **The only mechanism in this project's history to
clear every gate AND survive a substrate transfer.** 144/144 crops parity-exact; static regression
lock 13/13; CPU preflight reproduced the measurement on 11/11 crops.

Expected public effect **≈ +0.0045** (OOF→public slope ≈0.56 from two points — indicative only),
i.e. **~0.919**.

## 3. Everything else is closed

| mechanism | verdict |
|---|---|
| suppress-all | +0.0016970 P0-strict, but **44b6 −0.000619** — do not spend a slot alone |
| localisation / recentering | oracle +0.009, deployed −0.0088, sign-opposite — CLOSED |
| division: flat mother classification | needs AUC 0.983–0.999, measured 0.86–0.92 — CLOSED |
| division: branch-emergence proposer | GT-**oracle** still 19.5× over budget — **division route CLOSED** |
| counterfactual image critic | foreclosed on arithmetic (needs AUC 0.9386; appearance is 0.657/0.496) — never launched |
| bidirectional propagation | fails 56× (0.712% precision vs 39.83% break-even) — CLOSED |
| CAP / track-as-point | licence-blocked 4 ways; no weights exist; not detection-free — CLOSED |
| node budget · reverse-time · acquisition-state policies (arms C/C2/D/E/E2) | all closed, negative or vacuous |
| dense registration **as a motion vector** | worse than assuming zero motion — DO NOT FUND |

**D2 daughter-pair selector stays on the shelf** (+0.003541 paired, `pi_vis`-invariant), waiting on a
mother set no route can produce.

## 4. Corrections — nine this cycle. Read these before trusting older docs.

1. **suppress-all sign**: −0.001728 was an edges-only artifact; **+0.002706** through the complete wrapper (trap 14 in reverse).
2. **node-budget mechanism**: NOT the node ratio (count cost is `0.1·tp/N_est`, GT metadata, invariant). It is the **`d_tp/d_fp` composition** of deleted content.
3. **The 22/26 substrate inverted**: 44b6 22/26 but 6bba **66/125** — worst measured. 6bba is **85.06% of edge mass**.
4. **Temporal NMS is NOT free** — destroys 38–48 of 92 true forks under other rankers.
5. **"Re-aim, not widen" was selectively quoted** — the tight-6 µm row was always **+0.169%**. Branch A is still not live (B is not a superset of A, 7.2% churn), but the stronger claim was wrong.
6. **The sister-gate lead was DEAD CODE** — `DIV_SISTER_MAX_UM = 8.0` sits inside a filter no notebook enables. The "~71% of divisions excluded" claim is **withdrawn**.
7. **The arm-B brief was ambiguous** — see §2.
8. **"P0-B has 8 forks"** was a units error — it has **305** (8 was its metric division-FP count).
9. **Arm D falsified Lane 2's own bilateral core**: +0.003389 GT-oracle → **−0.0031111** exact.

## 5. The two rules that generalise

**Substrate transfer is the dominant failure mode** — six instances, one success (arm B). Assume every
inherited constant is invalid until re-measured on the substrate it is applied to.

**Oracle-clears / selector-fails is the second** — localisation +0.009→−0.009, component retention
+0.0087→−0.0078, division +0.0646→+0.0001, propagation +0.059→−0.45. **Headroom has never been the
constraint; selection has.** Do not fund another selector over an existing candidate population
without a base-rate argument.

## 6. Infrastructure built this cycle

- **`scripts/claims_table.py`** — 53 claims generated from `reports/inventory/*.json`, mandatory basis
  tags, enforced by tests. **Blocked wrong numbers three times.** Never hand-edit `CLAIMS.md`.
- **`src/biotrack/decision.py`** — shared expected-utility kernel, verified to 2.22e-16. Use it; do not
  roll your own arithmetic.
- **`tests/test_no_family_routing.py`** — locks the `BIOHUB_SHORT_TRACK_MIN_LEN_BY_DATASET` routing footgun.
- **Traps 14–24**, including trap 22 (our own `session_outputs()` truncated listings at 500 files) and
  trap 24 (`node_id` is not unique across movies — cost a false 15,471-hub reading).
- **`prob = zero` is an adequate proxy for a fixed gate** (≤2.26e-05) — this class of experiment no
  longer needs GPU.

## 7. State

43 commits this cycle, pushed. **42 tests pass.** 53 claims resolve. Submissions: **0 spent on
2026-08-01**; slots consumed to date 9. `.claude/settings.json` and `.gitignore` are deliberately
dirty and must stay untouched.

## 8. Guardrails (unchanged)

No public-score exploitation, negative-time or out-of-volume nodes, synthetic hubs/forks (measured
score-NEGATIVE). No family/crop routing. No tuning on the four visible placeholder movies. External
data needs URL, licence, checksum, provenance — **CTC, OrganoidTracker (GPL-2) and CAP (no licence)
are all blocked**. **Submissions come from NOTEBOOKS only.** Stage explicit paths when committing;
agents run concurrently and `git add -A` sweeps unreviewed work.
