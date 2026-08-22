# Is the linker structurally incapable of emitting divisions? — 2026-08-18

Labels: **[MEASURED]** = run this session against real data; **[CODE]** = read off source with
`file:line`; **[INFERENCE]** = derived, assumptions stated; **[UNVERIFIED]** = could not confirm,
with the test that would settle it.

Companion: `division_lane_2026-08-18.md` covers the *proposer* (`add_safe_divisions_postlink`)
and is not repeated here. This report covers the layer upstream of it — the **linker** — which
that report did not examine. Measurement discipline per `loeo_lb_gap_2026-08-18.md`: every
number below is a **count**, never a LOEO score.

---

## 0. VERDICT ON THE HYPOTHESIS

The hypothesis was: *"the linker is structurally incapable of emitting divisions, so
`add_safe_divisions_postlink` is a band-aid over an architectural defect."*

**CONFIRMED, but the mechanism is not the one proposed.** The seed observation — a perfect
one-to-one matching in the pre-wrapper graphs — is reproduced and extended (§2), but it is *not*
a bipartite-assignment artifact. The deployed linker is an **ILP**, and:

> **[MEASURED + CODE] The deployed ILP's cost table makes a division a strictly dominated
> action for every possible edge probability.** With the deployed weights
> (`--ilp-division-weight 1.0 --ilp-appearance-weight 0.0 --ilp-edge-weight -1.0`, read off the
> live fold-0 run log), opening a division changes the minimised objective by
> `+1.0 − p`, where `p ≤ 1.0` is the second daughter's edge probability. It is never negative.
> The solver therefore emits **zero** divisions no matter what the network predicts. Verified
> locally by running the exact `tracksdata.solvers.ILPSolver` at those weights on a toy lineage
> with `p = 0.99`: **0 divisions emitted** (§4).

And the defect is **layered** — three independent stages each suppress divisions, so fixing any
one alone changes nothing (§1, §6):

1. the hard-coded `0.5` candidate threshold on a **column**-softmax, which removes the second
   daughter from the candidate set entirely in 30 of the 61 well-posed cases (§5.2);
2. the ILP cost table, which discards any candidate that survives (§4);
3. `motion_relink_edges`, a `scipy.optimize.linear_sum_assignment` that **replaces** the ILP's
   edge set with a strict one-to-one matching on every crop (§1.3, measured).

Secondary finding, contradicting the seed hypothesis's premise: **the greedy edge selector is
not the constraint.** Under `use_ilp=True` it runs with *no* parent/child caps at all
(`predict_unet_transformer.py:85-93`), and `max_children_per_node = 2` — divisions permitted —
is the vendor default anyway. The one-to-one structure comes from the ILP, not the greedy.

Third finding: **the training objective does permit divisions and the inference stack forbids
them.** The mismatch the task asked about is real, and it is worse than "activation vs
assignment" — the training code even carries a division-upweighting hook that is hard-wired to
`1.0`, i.e. a no-op (§3).

Fourth finding, and the reason this matters operationally: **widening the proposer's geometry
gates does not substitute for fixing the linker.** A full CPU replay over all 199 crops (§5.4)
shows every re-gating raises true and false forks by the same factor — precision stays pinned
near 0.03 % — so the best setting tested is worth **Δscore ≤ +0.0004**, an order of magnitude
inside the ±0.003 noise floor. Geometry cannot separate divisions from orphan noise. The signal
that can is the network's own edge probability, which the ILP throws away and the proposer never
receives (it writes `"edge_prob": None` on every fork, `wrapper.py:854`).

---

## 1. WHAT THE ASSOCIATION STEP ACTUALLY IS [CODE]

Not Hungarian at the model, not greedy one-to-one, not threshold+dedupe. It is a **threshold →
unconstrained greedy pass-through → min-cost-flow ILP**, and then the wrapper runs **a second,
genuinely Hungarian assignment that discards the ILP's answer entirely.**

### 1.1 Stage A — edge candidate generation (`vendor/kaggle-cell-tracking/scripts/predict_unet_transformer.py`)

| what | where | value |
|---|---|---|
| activation | `:455-457` | `torch.softmax(raw, dim=0)` on the `(n_src, n_tgt)` logit block |
| threshold | `:73`, `:460-468` | `probs[i, j] > 0.5`, hard-coded — **not exposed on the CLI** |
| parent cap | `:82`, `:85-93`, `:478` | `None` when `use_ilp=True` → check skipped |
| child cap | `:83`, `:85-93`, `:476` | `None` when `use_ilp=True` → check skipped; `2` otherwise |

`raw` is `(n_src, n_tgt)`, so `softmax(dim=0)` normalises **down each target column**, over
candidate *parents*. Two consequences:

1. **A column sums to 1, so at most one entry per column can exceed 0.5.** In-degree ≤ 1 is a
   *theorem* of the activation + threshold pair, not a config choice. `max_parents_per_node` is
   dead code at this threshold.
2. **Rows are not normalised**, so a source exceeding 0.5 in two different columns is
   arithmetically fine. Out-degree 2 is permitted here.

Note the docstring at `:57-59` says the softmax is *"row-normalised over t+1 nodes"*. It is
column-normalised over t nodes. The docstring is wrong; the code matches training (§3).

### 1.2 Stage B — the ILP (`predict_unet_transformer.py:554-563`)

```
graph = build_graph(coords, edges)                 # :554  all prob>0.5 candidates
if cfg.use_ilp and graph.num_edges() > 0:          # :555
    solver = td.solvers.ILPSolver(                 # :556
        edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),   # :557
        appearance_weight=cfg.ilp_appearance_weight,                  # :558
        disappearance_weight=cfg.ilp_disappearance_weight,            # :559
        division_weight=cfg.ilp_division_weight)                      # :560
    graph = solver.solve(graph)                    # :563
save_graph(graph, output_dir / f"{name}.geff")     # :564  <- the exported pregraph
```

**[MEASURED — live run log]** `_evidence/kaggle_runs/p3_d1_pilot_f0_v1/biohub-p3-d1-pilot-f0.log:163`
shows the deployed command line:

```
--use-ilp --ilp-edge-weight -1.0 --ilp-appearance-weight 0.0
--ilp-disappearance-weight 1.5 --ilp-division-weight 1.0
```

`appearance_weight = 0.0` is **ours**, not the vendor's: the vendor default is `0.1`
(`predict_unet_transformer.py:78`), and `notebooks/kaggle_p3_armb/biohub-p3-armb.ipynb` sets
`os.environ["BIOHUB_ILP_APPEARANCE_WEIGHT"] = "0.0"` explicitly. That single override is what
kills divisions (§4).

The ILP formulation (`.venv/Lib/site-packages/tracksdata/solvers/_ilp_solver.py`):

| constraint | line | effect |
|---|---|---|
| `appear[v] + Σ in-edges == node[v]` | `:294-300` | **in-degree ≤ 1, structurally** |
| `disappear[v] + Σ out-edges == node[v] + division[v]` | `:309-314` | out-degree 2 **requires** `division[v] = 1` |
| `node[v] ≥ division[v]` | `:322-324` | division only on a selected node |
| objective sense | `ilpy/_components.py:145`, never overridden | **Minimise** |

So the objective delta of turning a single link into a division is exactly

```
Δ = division_weight − appearance_weight − p        [INFERENCE from the two constraints above,
                                                    MEASURED-confirmed in §4]
```

At the deployed weights that is `1.0 − 0.0 − p = 1.0 − p ≥ 0` for every `p ≤ 1`. **Divisions are
weakly dominated everywhere and strictly dominated for every `p < 1.0`.**

### 1.3 Stage C — the wrapper throws the ILP's graph away (`src/biotrack/wrapper.py`)

Even if stage B emitted a division, it would not survive:

| line | stage | effect on divisions |
|---|---|---|
| `:1145-1163` | `motion_relink_edges` result **replaces `edges` wholesale** | total |
| `:309-343` | that function is `scipy.optimize.linear_sum_assignment` per frame pair | **one-to-one by construction** |
| `:1165-1174` | `OUTPUT_SINGLE_PARENT_REPAIR` (default on, `:40`) | in-degree ≤ 1 again |
| `:1176-1184` | `OUTPUT_SINGLE_CHILD_REPAIR` (default **off**, `:41`) | would be out-degree ≤ 1 |
| `:1187` | `assert_degree_invariants` | asserts in ≤ 1, out ≤ 2 (`:1046-1066`) |
| `:1192` | `add_safe_divisions_postlink` | **the only division source in the deployed pipeline** |

`motion_relink_edges` is the true Hungarian in this system. It is a strict bipartite matching
(two passes, tight 6 µm then relaxed 10 µm), its cost ignores the learned probability except as
a `−1.0 × prob` bonus (`:340`), and its output *replaces* the learned edges rather than
refining them.

**[MEASURED]** `_evidence/kaggle_runs/p3_d1_pilot_f0_v1/run_stats.csv`, all 9 crops:
`motion_relink_replaced_raw_edges` equals the full incoming edge count (e.g. 27,480 of 27,488),
`motion_relink_fallback_raw = 0`, `motion_relink_skipped_large_frame = 0`,
`dropped_multi_parent_edges = 0`, `dropped_multi_child_edges = 0`. The relink runs on 100 % of
crops and replaces 100 % of the learned edges; the repair stages find nothing left to repair
because the Hungarian already guaranteed one-to-one.

---

## 2. THE PRE-WRAPPER GRAPHS [MEASURED]

Substrate: `_evidence/kaggle_runs/p3_d1_pilot_f0_v1/pregraphs_split0.parquet` (fold 0, 9 crops,
44b6) and `_evidence/agent_runs/ws_f_armB_p0b_2026-08-01/raw/pregraphs_split1.parquet`
(fold 1, **all 128 crops**, 6bba). These are the post-ILP, pre-wrapper geffs.

| | fold 0 (9 crops) | fold 1 (128 crops) | pooled |
|---|---|---|---|
| edges | 246,287 | 1,730,366 | **1,976,653** |
| out-degree histogram | `{1: 246287}` | `{1: 1730366}` | **`{1: 1976653}`** |
| in-degree histogram | `{1: 246287}` | `{1: 1730366}` | **`{1: 1976653}`** |
| min `edge_prob` | 0.5000027 | 0.5000001 | — |
| edges with `prob ≤ 0.5` | 0 | 0 | 0 |
| orphan targets (node at t>0 with no parent) | 23,571 (8.7 %) | 179,037 (9.4 %) | — |

**Zero out-degree-2 nodes in 1,976,653 edges.** The seed observation holds at full scale.

**The 0.5 floor is a hard threshold, not a softmax artifact.** The minimum observed probability
is 0.5000001 — one ULP-ish above the `> cfg.threshold` comparison at `:465`. A softmax artifact
would produce a smooth density down to 0; instead the distribution is truncated exactly at the
constant. (Quantiles, fold 0: p1 0.510, median 0.791, p95 0.974.)

Note this also means **in-degree ≤ 1 was never tested** by the ILP: the candidate set handed to
the solver already satisfies it (§1.1). The ILP's only *binding* decision is out-degree, and it
always decides "1".

---

## 3. TRAINING PERMITS DIVISIONS; INFERENCE FORBIDS THEM [CODE]

`vendor/kaggle-cell-tracking/scripts/train_unet_transformer.py:55-72`:

```python
probs = torch.softmax(logits, dim=0)  # dim=0 intentional: divisions allowed, merges aren't  :63
...
div_rows = target.sum(dim=1) > 1                                                            # :68
weight = torch.ones_like(loss)                                                              # :69
weight[div_rows] = 1.0                                                                      # :70
```

Two things:

1. **The activation matches inference exactly** (`softmax(dim=0)` in both), so there is *no*
   train/infer activation mismatch. The target matrix (`:40-52`) is built directly from the GT
   edge list, so a dividing mother is a row with two ones, and column-normalisation lets both
   reach 1.0 simultaneously. **The objective genuinely permits and rewards divisions.**
2. **The division upweighting hook is a no-op.** `weight[div_rows] = 1.0` sets the weight to the
   same value `torch.ones_like` already gave it. Divisions are trained at their natural
   frequency — 151 dividing rows against ~128,581 continuation rows measured from GT (§5), i.e.
   **0.12 %**. The focal term `(1 − p_t)²` gives them some emphasis; the explicit knob gives
   none.

**[INFERENCE]** Trackastra's `1 + Σ exp` denominator (an explicit "no parent" option) is absent
here: with a bare column softmax, every target column is forced to distribute a full unit of
mass across candidate parents whether or not a true parent exists. That inflates the
best-candidate probability for genuinely-unparented nodes and is a plausible contributor to the
measured 9.4 % orphan rate being as *low* as it is. Not tested; would need the pre-ILP
probability matrix.

So the reconciliation asked for is: **training allows divisions, the ILP forbids them
economically, and the wrapper's Hungarian forbids them structurally.** The mismatch is real.

---

## 4. LOCAL PROOF THAT THE DEPLOYED ILP CANNOT DIVIDE [MEASURED]

`tracksdata` and `ilpy` are installed in `.venv`, so this needs no GPU and no model. Toy
lineage: one track t=0..3, dividing at t=3, both daughters continuing to t=7, all edges above
the 0.5 candidate threshold. Solver constructed exactly as `predict_unet_transformer.py:556-560`.

| p(mother→daughter) | `division_weight` | `appearance_weight` | `disappearance_weight` | edges kept | **divisions emitted** |
|---|---|---|---|---|---|
| **0.99** | **1.0** | **0.0** | **1.5** | 10 | **0** ← deployed |
| 0.95 | 1.0 | 0.0 | 1.5 | 10 | **0** |
| 0.80 | 1.0 | 0.0 | 1.5 | 10 | **0** |
| 0.99 | 1.0 | 0.1 | 0.1 | 11 | 1 ← vendor default |
| 0.90 | 1.0 | 0.1 | 0.1 | 11 | 1 |
| 0.88 | 1.0 | 0.1 | 0.1 | 10 | **0** |
| 0.80 | 0.5 | 0.0 | 1.5 | 11 | 1 |
| 0.80 | 0.3 | 0.0 | 1.5 | 11 | 1 |
| 0.40 | 0.3 | 0.0 | 1.5 | 11 | 1 |

The break-even is exactly `division_weight < appearance_weight + p`, confirmed to the boundary:
at `appearance_weight = 0.1, division_weight = 1.0` the solver flips between p = 0.88 (no) and
p = 0.90 (yes). At `appearance_weight = 0.0` the condition becomes `1.0 < p`, which is
**unsatisfiable for a softmax output**.

> **[MEASURED, decisive] The deployed pipeline sets `BIOHUB_ILP_APPEARANCE_WEIGHT = "0.0"`,
> overriding the vendor's `0.1`. That one line makes starting a new track free, so the solver
> always prefers "the second daughter appears from nothing" over "the mother divides". Every
> division our model predicts is discarded at the solver, and `add_safe_divisions_postlink` then
> reinvents divisions from geometry alone.**

Reproduce: `scratchpad/ilp_division_probe.py` (§8).

---

## 5. THE PRIZE [MEASURED]

### 5.1 GT division geometry, model-free, all 199 crops

Measured directly from `data/train/*.geff`, no prediction involved. 151 GT divisions
(26 in 44b6, 125 in 6bba) — reproduces `division_lane_2026-08-18.md` §0 exactly.

| quantity | median | p10 | p90 | max |
|---|---|---|---|---|
| nearer daughter ← mother | 4.08 µm | — | 6.86 | — |
| farther daughter ← mother | 7.13 µm | 4.49 | 10.05 | 13.53 |
| sister ↔ sister | 10.57 µm | 6.36 | 14.36 | 20.30 |
| **ordinary continuation displacement** (128,581 non-dividing edges) | **1.82 µm** | — | 4.14 | p99 = 8.37 |

Divisions are **geometrically well separated** from ordinary motion: 7.13 µm vs 1.82 µm median.
The deployed proposer's `SAFE_DIV_MAX_UM = 4.66` sits *below* the division population and *above*
92.6 % of ordinary motion — the least discriminative place on the axis. A **band** rather than an
upper bound is the right shape: `parent ∈ [4.5, 12] ∧ sister ∈ [6, 18]` covers 126/151 (83.4 %)
of true divisions while containing only 7.9 % of ordinary continuation displacements — an 11.7×
background reduction versus the current gate.

Exact ceiling of the deployed gates (`BIOHUB_SAFE_DIV_MAX_UM=4.66`, `SISTER=8.5`,
`EXISTING_CHILD=7.65`, from `biohub-p3-armb.ipynb`), on GT geometry alone:

* **40 / 151** if the linker always keeps the *farther* daughter and orphans the nearer one (best case)
* **19 / 151** if it keeps the nearer one (worst case)
* raising only the candidate gate saturates at **42 / 151** — the *sister* gate is the binder
* `child ≤ 12 ∧ candidate ≤ 8 ∧ sister ≤ 18` → **140 / 151**

(Consistent with `division_lane_2026-08-18.md` §3.2's 22/151 figure, which uses the observed
rather than the best-case daughter assignment.)

### 5.2 What a division-capable assignment recovers, from the pre-wrapper graph

Fold 1 (128 crops, 125 GT divisions), against the post-ILP pre-wrapper graph. Each GT division
walked through: mother matched within 7 µm, both daughters matched, then the linker's actual behaviour.

| stage | count | of 125 |
|---|---|---|
| GT divisions | 125 | 100 % |
| mother detected | 110 | 88 % |
| both daughters detected | 80 | 64 % |
| daughters are **distinct** predicted nodes (not one merged blob) | **71** | 57 % |
| mother has out-degree 1 in the linker output | 65 | 52 % |
| the linked child **is** one of the true daughters | **61** | **49 %** |
| → other daughter is an **orphan** (no parent at all) | **31** | 25 % |
| → other daughter was **taken by a different parent** | **30** | 24 % |

**The assignment-only ceiling is 61 of 125 (49 %) in fold 1** — no new detection, no new model,
no retraining. Fold 0 has only 3 GT divisions in the 9 exported crops, too few to add.
Extrapolated to 199 crops that is roughly **61–70 of 151 (40–46 %)**.

Of the remaining 64: **60 need detection, not linking** — 15 undetected mothers, 30 undetected
daughters, 9 where the detector collapsed both daughters into a single node, 6 mothers with
out-degree 0. The other 4 are mothers whose one emitted edge goes to neither true daughter; they
are assignment failures too, so the strict assignment-reachable count is **65 of 125 (52 %)**.

The 61 split into two mechanically different populations:

**(a) 31 orphan cases — the ILP fix targets exactly these.** The second daughter has no parent
at all. Under the ILP that means either no candidate edge existed, *or* a candidate existed and
the solver declined it because accepting would have cost `+1.0` in division weight. **The second
case is recoverable by changing one number, with no new model output.** Which of the two it is
cannot be read from the post-ILP graph — see the falsification test in §6.1.

**(b) 30 stolen cases — the ILP fix cannot help these.** The thief edge has measured
`edge_prob` median 0.748 (min 0.510, max 0.987). Because the column softmax sums to 1, the
mother's probability in that same column is at most `1 − p_thief`, i.e. **< 0.5 in all 30**, so
the mother→daughter edge was never even a candidate. These need `cfg.threshold` lowered
(§6.2), and the mother is *farther* from the daughter than the thief in 27 of 30 — a distance
prior will not fix them either. The thief is a genuinely different cell, not a duplicate
detection of the mother: median thief↔mother separation 8.12 µm, 12 of 30 inside the 7 µm
scorer matching radius but **none inside 3.5 µm**.

For reference, the correct mother→daughter1 edge that *was* kept has median `edge_prob` 0.859
(min 0.502, n=61) — the model is confident about the daughter it keeps.

### 5.3 Score arithmetic [INFERENCE from CODE]

`division_jaccard = TP / (TP + FP + FN)`, and a wrong fork only becomes an FP when its **mother
matches an annotated GT node** (`src/biotrack/decision.py:155`). Current state: TP 5, FP 613,
FN 146, divJ 0.0065, contributing 0.00065 of the 0.915.

| scenario | TP | FP | FN | divJ | Δscore (0.1 × divJ) |
|---|---|---|---|---|---|
| today | 5 | 613 | 146 | 0.0065 | — |
| assignment fix, FP unchanged | 65 | 613 | 86 | 0.0851 | **+0.0079** |
| assignment fix **and** proposer re-gated to cut FP to 150 | 65 | 150 | 86 | 0.2158 | **+0.0209** |
| every reachable division, FP 100 | 89 | 100 | 62 | 0.3546 | **+0.0348** |

The gap to the leader is 0.951 − 0.915 = **0.036**. The division term alone can plausibly carry
a third to a half of it, but **only if precision moves too** — recall alone buys +0.008.

### 5.4 Re-gated proposer replay [MEASURED]

CPU replay of `add_safe_divisions_postlink` over the full 199-crop deployed post-wrapper
substrate (`c:/temp/subvoxel_f0`, `c:/temp/subvoxel_f1`), honouring the deployed caps
(`FRAME_FRAC_CAP = 0.0076`, `GLOBAL_FRAC_CAP = 0.00375`) and counting FP only at
annotated-mother forks:

| setting (`child` / `candidate` / `sister` µm, ranking, caps) | forks | TP | FP\* | FN | divJ | Δscore |
|---|---|---|---|---|---|---|
| deployed: ≤7.65 / ≤4.66 / ≤8.5, closest-first, 1× | 3,743 | 1 | 224 | 150 | 0.0027 | −0.00039 |
| ≤12 / ≤8 / ≤18, closest-first, 1× | 13,680 | 2 | 571 | 149 | 0.0028 | −0.00038 |
| ≤12 / ≤8 / ≤18, division-like score, 1× | 13,680 | 2 | 556 | 149 | 0.0028 | −0.00037 |
| ≤12 / **1.5–8** / **6–18**, division-like score, 1× | 13,663 | 4 | 495 | 147 | 0.0062 | −0.00003 |
| ≤12 / 1.5–8 / 6–18, division-like score, **4×** | 44,318 | 13 | 1,296 | 138 | 0.0090 | +0.00024 |
| ≤12 / **2.5–8** / **7–16**, division-like score, 4× | 36,602 | 11 | 921 | 140 | **0.0103** | **+0.00037** |
| ≤12 / 2.5–8 / 7–16, division-like score, 16× | 39,388 | 11 | 1,008 | 140 | 0.0095 | +0.00030 |

\* FP counted only where the fork's mother matches an annotated GT node (`decision.py:155`).
Δscore is the change in the `0.1 × division_jaccard` term only; edge-jaccard side effects are
not modelled.

**Caveat on absolute numbers.** This is a *re-application* of the proposer on top of the
already-processed deployed output, so mothers that already carry a safe division are excluded by
the `out-degree == 1` filter and true daughters already carrying an edge are excluded from the
orphan pool. It under-emits: 3,743 forks / TP 1 at the deployed gates, versus the true
11,582 / TP 5. **Treat the rows as relative, not absolute.** The precision it reports is
credible because it matches: 1/3,743 = 0.027 % here versus 5/11,582 = 0.043 % measured.

> **[MEASURED, and it is a negative result] Widening the geometry gates does not buy a division
> lane. Every setting tested raises TP and FP by the *same* factor — precision stays pinned near
> 0.03 % — so `division_jaccard` moves from 0.0027 to at best 0.0103, i.e. Δscore ≤ +0.0004, an
> order of magnitude inside the fold-0 noise floor of ±0.003.**

Two structural refinements do help *relatively* and should be kept in any re-gating: adding a
**lower bound** to the candidate and sister gates (TP 2 → 4 at constant fork count, the single
biggest ratio improvement in the table), and replacing the closest-first ranking (FP 571 → 556
at identical forks). Neither is close to sufficient on its own.

**Interpretation.** §5.1 showed a band gate cuts *ordinary continuation* background 11.7×, but
the pool the proposer actually draws from is **orphan** targets, not continuation targets.
Orphans are precisely the anomalous and spurious detections, and their geometry relative to a
nearby mother is division-shaped by accident. **Geometry alone does not separate divisions from
orphan noise.** The one signal that would — the network's own edge probability for the
mother→second-daughter link — is exactly what the ILP discards (§4) and what
`add_safe_divisions_postlink` never sees (it sets `"edge_prob": None` on every fork it adds,
`wrapper.py:854`). That is the argument for prioritising L1 (§6.1) over further gate work.

---

## 6. MINIMAL CHANGES, RANKED

All three are needed for the linker lane; **any one alone measurably changes nothing**, because
each downstream stage independently destroys divisions.

**Execution order.** Run the §6.1 falsification measurement *first* — it is a one-line kernel
edit, costs ~2 % wall-clock on a run that has to happen anyway, and it decides whether L1–L3 are
worth any budget at all. Do not write L2 or L3 before that number exists. If the number is
healthy, the cheapest complete lane is L1 (env var) + L3 (wrapper stage, default-off, CPU
testable) in one pilot; L2 only if L1+L3 lands and the residual is the 30 "stolen" cases.

### 6.1 L1 — make divisions economically reachable in the ILP (**cheapest, one number**)

**Patch.** `notebooks/kaggle_p3_armb/biohub-p3-armb.ipynb`, the env block that currently reads
`os.environ["BIOHUB_ILP_APPEARANCE_WEIGHT"] = "0.0"`. Either restore the vendor `0.1`, or —
preferred, because appearance weight also governs track-start economics everywhere — add
`os.environ["BIOHUB_ILP_DIVISION_WEIGHT"] = "0.55"`. At `appearance_weight = 0.0` this opens a
division whenever the second candidate edge has `p > 0.55`, i.e. only for edges the model
already believes more strongly than its own 0.5 candidate threshold.

**Why 0.55 and not lower:** every candidate edge is already `> 0.5`, so `division_weight ≤ 0.5`
would make *every* multi-candidate source divide. 0.55 keeps a margin.

**Falsification test — do this BEFORE changing anything (it is nearly free).** The number that
decides whether L1 is worth anything at all is *how many sources have ≥ 2 candidate edges in the
pre-ILP graph*. That graph already exists in memory at `predict_unet_transformer.py:554`, one
line before the solver. Add a kernel edit that writes it:

* new file `scripts/kaggle_edits/pre_ilp_export.py` — `save_graph(graph, out_dir / f"{name}.preilp.geff")` immediately after `:554`, plus a parquet roll-up matching the existing `loeo_pregraph_export.py` schema.
* **Cost:** ~+2 % kernel wall-clock, no extra GPU. Inert until pushed.
* **What to count:** (i) sources with out-degree ≥ 2 in the candidate graph — the raw supply; (ii) of those, how many are within 7 µm of a GT dividing mother with both its second-candidate target within 7 µm of the true second daughter — the recoverable supply. §5.2 caps (ii) at **31 in fold 1**.
* **Kill criterion:** if (ii) is below ~15 in fold 1 (i.e. under half the 31 orphan cases had a declined candidate), the ILP is not discarding real divisions and **L1 is dead** — the second daughter simply never scored above 0.5 and the lane collapses into L2 (threshold) or the training fix (§7).
* **Upper bound today:** ≤ 179,037 (fold-1 orphan targets), which is uninformatively loose; the export is the only way to tighten it.

### 6.2 L2 — lower the candidate threshold so the second daughter is a candidate at all

**Patch location.** `cfg.threshold` is hard-coded at `predict_unet_transformer.py:73` and has
**no CLI flag** — `main()` exposes `--det-threshold` and the four `--ilp-*` weights but not
`--threshold`. So this needs a kernel edit, not an env var.

**Minimal form — do NOT lower the global threshold.** A global drop to 0.2 multiplies the ILP's
variable count and risks the 9 h kernel budget. Instead, add a *division-candidate second pass*:
for each source that already has one candidate edge above 0.5, admit **at most one** additional
edge `(i, j)` where `j` is its next-best column, `prob[i, j] > t_div` (start `t_div = 0.25`), and
the geometry is division-shaped (`parent ∈ [4.5, 12] µm`, `sister ∈ [6, 18] µm` — §5.1). This
bounds candidate growth at +1 edge per source before geometry, far less after, and leaves the
ILP's in-degree constraint to arbitrate the 30 "stolen" cases correctly.

* new file `scripts/kaggle_edits/div_candidate_pass.py`, injected at `:468` (between the
  `candidates = sorted(...)` comprehension and the greedy loop).
* **Cost:** one GPU pilot; the network forward pass is unchanged, only the edge list grows.
* **Falsification:** with L1 already in place, count divisions in the pre-wrapper graph. If
  `t_div = 0.25` does not produce ≥ 60 out-degree-2 sources per embryo, the network simply does
  not represent these divisions and the lane closes — escalate to the training fix (§7).

### 6.3 L3 — stop the wrapper from deleting divisions

**Patch location.** `src/biotrack/wrapper.py:1158-1163`. `motion_relink_edges` returns a strict
Hungarian matching that replaces `edges` entirely, so any ILP division dies here regardless of
L1/L2.

**Minimal form.** A new env-gated stage `restore_learned_divisions(raw_edges, edges, ...)`
inserted between `:1163` and `:1165`, default **off**
(`BIOHUB_RESTORE_LEARNED_DIVISIONS`, matching the existing flag convention at `:40-56`). It
re-admits a raw edge `(s, t)` iff: `s` had out-degree 2 in `raw_edges`; `s` has exactly one
motion-relinked child; `t` has no parent after the relink; and `t` is at `frame(s) + 1`. This is
additive, cannot reduce the edge set, and keeps `assert_degree_invariants` satisfied
(out ≤ 2, in ≤ 1). `raw_edges` is already the function's parameter, so nothing new is threaded
through.

**CPU-replayable test on cached graphs.** `artifacts/kaggle/p0strict_cache/graphs` (199 crops,
71 + 128) is **not usable for this** — I checked it: it is a *post*-wrapper cache with no
`edge_prob` column and its degree histogram is `{1: 3576550, 2: 10724, 3: 2}`, i.e. an older,
more permissive wrapper configuration than the deployed one. Use instead the two pre-wrapper
parquets named in §2 (fold 0 9 crops + fold 1 **all 128**), replaying `filter_output_graph` on
CPU and counting out-degree-2 sources that (a) survive to export and (b) match a GT division
within 7 µm. **Counts only, no scorer.**

### 6.4 D1 — re-gate the proposer: **partially refuted, do not run it alone**

`division_lane_2026-08-18.md` §4/D1 proposes widening `SAFE_DIV_MAX_UM` 4.7 → 10 and
`SAFE_DIV_SISTER_MAX_UM` 7.2 → 15 on the grounds that this makes 81 of 89 metric-reachable
divisions *proposable*. §5.1 confirms the proposability claim exactly. **§5.4 measures what
happens when you actually do it, and the answer is Δscore ≤ +0.0004** — the fork count and the
TP count rise by the same factor, so `division_jaccard` barely moves. Proposability was never
the binding constraint; **precision** is.

Two pieces of D1 survive and should be folded into any future attempt, because they improve the
ratio rather than just the volume:

* **Give the candidate and sister gates a lower bound**, not just an upper one. §5.1: true
  divisions sit at 4.08 / 7.13 / 10.57 µm while ordinary motion sits at 1.82 µm, so
  `candidate ∈ [2.5, 8]` and `sister ∈ [7, 16]` is the discriminative shape. Measured effect at
  constant fork count: TP 2 → 4.
* **Replace the ranking score.** The deployed `score = parent_dist + 0.15 * sister_dist` sorted
  ascending (`wrapper.py:831,837`) ranks the *tightest* pairs first — backwards by ~4× in median
  distance. Measured effect: FP 571 → 556 at identical fork count. Small, but free.

Both are pure env/constant changes, no kernel edit, no GPU — but on the measured evidence they
are worth ~+0.0004 together, which is inside the ±0.003 fold-0 noise floor. **They are not a
lane on their own.** The lane is L1–L3, because only the linker carries the probability signal
that geometry lacks.

---

## 7. WHAT I COULD NOT VERIFY [UNVERIFIED]

1. **How many divisions the ILP is actually discarding.** The exported graphs are *post*-ILP, so
   a declined candidate is invisible. Bounded above by the orphan count (179,037 in fold 1),
   which is useless. §6.1's pre-ILP export settles it for ~2 % kernel time and is the single
   highest-information cheap measurement available in this lane.
2. **Whether the network represents divisions at all.** If the pre-ILP export shows almost no
   multi-candidate sources, the defect is in *training*, not in the assignment — and the
   candidate repair is `train_unet_transformer.py:70`, the no-op division weight (§3), plus a
   Trackastra-style `1 + Σ exp` background column. That is a retrain, not a kernel edit.
3. **Fold-0 division statistics from the pre-wrapper graph.** The exported fold-0 pregraph covers
   9 of 71 crops and contains only 3 GT divisions. All §5.2 numbers are fold-1 (6bba) only.
   6bba carries 125 of the 151 GT divisions, so this is the right embryo, but the 44b6 direction
   is unmeasured at this layer and must be reported separately per the operating contract.
4. **Whether L1+L2+L3 interact benignly with the motion gate.** The arm-B motion gate was
   promoted on LOEO and scored +0.000 on the LB (`loeo_lb_gap_2026-08-18.md`). Nothing here
   should be promoted on a LOEO delta; use counts, and prefer the pre-ILP export as the decision
   instrument.

---

## 8. FILES WRITTEN THIS SESSION

None inside the repo except this report. All analysis code is in the session scratchpad
(`…/1fbd2089-a749-4a48-bc21-72479893e469/scratchpad/`), deliberately outside Git per the
operating contract:

| file | what it does |
|---|---|
| `pregraph_div_audit.py` | degree histograms + per-GT-division walk over a pre-wrapper parquet (§2, §5.2) |
| `gt_div_geometry.py` | model-free GT division geometry and gate admissibility over 199 crops (§5.1) |
| `ilp_division_probe.py` | local `tracksdata` ILP proof that the deployed weights forbid divisions (§4) |
| `safe_div_gate_replay.py` | CPU replay of the proposer at alternative gates, FP counted at annotated mothers (§5.4) |

No commits, no pushes, no submissions, no GPU launches. `.claude/settings.json` untouched.
`scripts/kaggle_edits/pre_ilp_export.py` and `scripts/kaggle_edits/div_candidate_pass.py` are
**proposed** in §6 and have not been written — they are one-cell kernel edits and should be
written only when the host authorises the corresponding pilot.
