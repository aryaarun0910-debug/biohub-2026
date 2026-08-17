# Internal quick-wins / config archaeology — 2026-08-17

Scope: comb our OWN evidence (research.sqlite 102 findings, deployed wrapper, deployed
kernel env, cached OOF graphs) for un-exploited CPU-only or config-level wins toward
~0.930 that do NOT require the heavy external retrain. Deployed = **P3 harmonic, 0.915
public**. Ranked by EV x cheapness.

Two substrates matter and are constantly conflated in the ledger:
- **E0c** = the weak OOF substrate (LOEO 0.7595 / 0.6490). Cached graphs EXIST
  (`artifacts/kaggle/e0c_cache/graphs`, 199 crops).
- **P0-strict / P0-B** = the *deployment* substrate (public 0.914; P3 = +harmonic on it).
  Cached graphs EXIST (`artifacts/kaggle/p0strict_cache/graphs`,
  `artifacts/kaggle/p0strict_f0_cache/graphs`).
- **P3 harmonic** OOF graphs do **NOT** exist locally (finding D-10). This is the single
  gating fact for most CPU replays below.

## Deployed config, extracted (answers mandate item 1)

Full explicit env set by `notebooks/kaggle_p0b_clean913_revtime/*.ipynb` (35 overrides).
Highlights vs `src/biotrack/wrapper.py` defaults:

| Env var | Deployed | Wrapper default | Read |
|---|---|---|---|
| BIOHUB_DET_THRESHOLD | 0.96875 | 0.99 | public-swept (A-20), fine |
| BIOHUB_ILP_APPEARANCE_WEIGHT | **0.0** | 0.1 | A-11 **already shipped** |
| BIOHUB_ILP_DISAPPEARANCE_WEIGHT | **1.5** | 0.1 | A-11 **already shipped** |
| BIOHUB_GAP_CLOSE_MAX_GAP | 2 | 1 | (wrapper clamps to 1; kernel does not) |
| BIOHUB_GAP_CLOSE_UM | 5.8 | 6.0 | public-swept (beicicc), fine |
| BIOHUB_GAP_DENSITY_ADAPTIVE | **1** | 0 | density-adaptive gap is ON in prod |
| BIOHUB_GAP_DENSITY_GAIN | 0.040 | 0.040 | **round / untuned** |
| BIOHUB_OUTPUT_MIN_TRACK_LEN | 6 | 6 | **round**; public-swept 6/7/8 (A-20) |
| BIOHUB_SAFE_DIV_MAX_UM | 4.66 | 4.7 | 3-sig-fig, substrate unknown |
| BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM | 7.65 | 7.8 | 3-sig-fig, substrate unknown |
| BIOHUB_SAFE_DIV_FRAME_FRAC_CAP | 0.0076 | 0.008 | 2-sig-fig, substrate unknown |
| BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP | 0.00375 | 0.004 | 3-sig-fig, substrate unknown |
| BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE | 0 | (E-09) | shipped **disabled** |
| BIOHUB_OUTPUT_GAP2_RECOVERY | 0 | 0 | off |
| BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION | 0.90 | — | A-01 retention guard shipped |
| BIOHUB_USE_DEEPCENTER_VETO / GAP_VETO | 1 / 1 | — | DeepCenter veto ON |
| BIOHUB_ARMB_FLOW_GATE | **absent (=off in P3)** | (armb patch) | see #1 |

Net: the ILP birth/death asymmetry (A-11), det threshold, gap distance, and retention
guard are all already at public-swept values. The remaining round/untuned knobs
(GAP_DENSITY_GAIN, MIN_TRACK_LEN) and the 3-sig-fig safe-div caps are low-EV (below).

---

## CANDIDATE 1 (TOP) — Ship Arm B: flow-compensated motion-relink gate

- **Mechanism.** `scripts/kaggle_edits/armb_flow_gate.py`. In `motion_relink_edges`, the
  eligibility test changes from raw `|target - source| > gate_um` to flow-compensated
  `|target - (source + flow(source))| > gate_um`. Radius unchanged (tight 6.0 / relaxed
  10.0 um); relink cost expression unchanged. `flow(source)` is a GT-free, image-free,
  model-free kNN16 tissue-flow read off the crop's own pre-wrapper prediction graph.
  Config surface: `BIOHUB_ARMB_FLOW_GATE=1` + the wrapper patch.
- **Measured, on the DEPLOYMENT substrate (not E0c).** From
  `scripts/kaggle_edits/armb_flow_gate.py:23-26` and
  `_evidence/agent_runs/ws_f_armB_p0b_2026-08-01/`: P0-strict LOEO, complete wrapper,
  144/144 crops parity-exact: **pooled +0.00798, 44b6 +0.01676, 6bba +0.00671,
  P(delta>0)=1.000 on all three.** Also +0.0088 pooled on E0c (199 crops). This CLEARS
  the +0.005 bilateral promotion bar on BOTH families — the only lever in the corpus that
  does so on the deployment substrate.
- **Why un-shipped.** P3 shipped harmonic ALONE. The `p3_armb.json` interaction warning:
  arm B's relink COST consumes `prob`, which harmonic rewrites, "MUST be measured together
  and never assumed additive." The joint (harmonic x armB) has never been run. Notebooks
  `notebooks/kaggle_p3_armb_loeo_f0/`, `_f1/` are BUILT but carry no result JSON.
- **Why the interaction is probably small (new observation).** Arm B's GATE is *pure
  geometry* — the eligibility test `_gate_q > gate_um` does not read `prob` at all. `prob`
  only enters arm B's *cost* (`motion + 0.05*raw - LEARNED_BONUS*prob`), i.e. it affects
  which eligible target a source links to, not which are eligible. Harmonic perturbs the
  cost ranking within an unchanged eligible set — a second-order interaction.
- **Expected benefit.** +0.005 to +0.008 bilateral if it transfers through harmonic;
  plausibly 0.915 -> ~0.921-0.923.
- **Cheapest falsification (CPU, graphs exist).** On `p0strict_cache/graphs`, reproduce arm
  B's +0.008, then re-run its relink with the `prob` term ablated / perturbed by the
  harmonic re-scaling to bound the interaction: if arm B's delta stays >= +0.005 bilateral
  when the `LEARNED_BONUS*prob` cost term is zeroed, harmonic cannot break it and it ships.
  ~1 h CPU, zero GPU. Reuse the ws_f_armB replay driver.
- **Definitive test (GPU).** Run `p3_armb` LOEO f0+f1 (2 T4 sessions, notebooks already
  built) and require bilateral min-fold >= +0.005 under the patched pooled scorer.
- **Verdict: highest EV x cheapness on the board. CPU de-risk first, then 2 GPU sessions.**

## CANDIDATE 2 — Persist P3-harmonic OOF graphs ONCE (unlocks every CPU replay)

- **Mechanism / why.** D-10: no P3 OOF graphs exist, so *every* post-processing sweep
  below is either stuck on the weak E0c substrate or on pre-harmonic P0-strict. One GPU
  session that dumps P3 per-crop OOF graphs in the `e0c_cache/graphs` parquet schema turns
  candidates 1, 3, 5, 6 into pure-CPU replays and amortises across all of them.
- **Expected benefit.** Indirect but large: it is the enabling capital for the whole
  shotgun. D-10 calls it "the single highest-value unspent compute."
- **Cost.** One Kaggle T4 OOF session (~2-4 h), then `phaseb_h0c_replay.py`-style CPU.
- **Verdict: do this in the SAME session that runs Candidate 1's GPU arm, not separately.**

## CANDIDATE 3 — Node-count multiplier: HONEST DOWNGRADE for P3

- **Claim being checked.** RT-04/RT-05/RT-12/A-06: "the node-count multiplier is our whole
  deficit" (+0.02-0.03), oracle ceiling +9.9%/+9.5% on the edge term.
- **Archaeology correction.** That +0.02-0.03 is **E0c-specific**: E0c over-emits (pooled
  ratio +0.201, multiplier 0.980). The DEPLOYED P0-B/P3 already runs the public node
  calibration (det 0.96875, ILP 0.0/1.5, density-adaptive gap ON) and sits at the public
  multiplier ~1.010. So the recoverable public fraction of this channel (~+0.010) is
  **already captured in P3** — exactly as the mandate suspected. The residual oracle
  capacity (+0.08) is realizable ONLY by selective confidence-ranked deletion, which
  RT-05 proves is a detection-precision problem in disguise (random deletion is strictly
  negative). That IS the heavy retrain, not a cheap win.
- **Only cheap, decision-relevant probe left.** RT-05's decile test: sort predicted nodes
  by detector confidence, measure metric-visible fraction per decile on frozen OOF. If the
  bottom decile is NOT enriched for metric-invisible nodes vs the top, confidence pruning
  cannot beat random and the channel is formally closed for us. ~20 min CPU on cached
  graphs, zero GPU. Run it once to retire the "node budget" temptation with evidence.
- **Verdict: do the 20-min decile probe; do NOT invest in keep_frac sweeps expecting P3
  gain — the channel is mostly closed on the deployment substrate.**

## CANDIDATE 4 — E-02 border-conditioned retention (fixed node count reshuffle)

- **Mechanism.** `filter_short_track_components` deletes short boundary-explained and short
  interior components at the same rate (length-only rule, MIN_TRACK_LEN=6). Add a
  volume-boundary / t in {0,T} exemption so the SAME node budget is reallocated toward
  boundary-explained components. Fixed count => moves adj-edge-J via retention without
  touching the (closed) multiplier channel. Pure post-processing.
- **Why un-tried.** Our retention rule has only ever been length-based; the boundary flag
  is not computed.
- **Expected benefit.** Unquantified; plausibly small (+0.001-0.003). It is the one
  retention idea that is orthogonal to the closed multiplier channel.
- **Cheapest falsification (CPU).** On `p0strict_cache/graphs`, split the wrongly-deleted
  GT edges by whether their component touches the volume boundary / movie ends. If deleted-
  but-correct components are disproportionately boundary-touching, add the exemption and
  rescore with the patched scorer at IDENTICAL node count. < 1 h CPU, zero GPU.
- **Verdict: cheap, prototype-able on existing p0strict graphs today.**

## CANDIDATE 5 — Low-EV config sweeps (prototype on p0strict cache)

Grouped because each is CPU-cheap but low-EV; do only after 1-4.
- **MIN_TRACK_LEN {5,6,7,8}** (deployed 6, round). Public swept 6/7/8 and A-20 calls it
  noise; directly touches node count but the optimum is likely already 6. Confirm on the
  deployment substrate at fixed everything-else.
- **GAP_DENSITY_GAIN** (deployed 0.040, round) and **GAP_CLOSE_UM** micro-moves: the
  density-adaptive gap is ON; its gain is the one untuned constant. Small.
- **E-09 ADAPTIVE_SHORT_TRACK_RESCUE=1** (shipped disabled): re-admit sub-threshold
  components by mean edge prob under a node cap. Public shipped it DISABLED, which is
  evidence it does not clear the bar; test at most two parameterisations.
- **Safe-div caps (4.66 / 7.65 / 0.0076 / 0.00375):** tuned to 2-3 sig figs on an unknown
  (likely E0c/placeholder) substrate — a textbook "tuned on weak substrate, never re-tuned"
  smell. BUT division is a MINORITY of the gap and the safe-div layer already sits at/below
  the D-07 break-even (needs 4-7% precision among metric-visible, delivers ~0.1%). Re-tuning
  the caps cannot manufacture precision the candidates do not have. Low EV; only worth a
  sweep once P3 OOF graphs exist (free ride on Candidate 2).

---

## FLAGS — measurement artifacts / over-hasty items (do NOT re-propose)

- **"suppress-all division +0.0027" is an artifact (D-11).** The +0.0035/+0.0027 row is the
  **GT-INFORMED** retention arm (`phaseb_oracle_d0prime.py:99-108` picks the retained child
  by GT consistency). The GT-free control is **-0.00004 / -0.00204** (adjEdgeJ falls on
  6bba). Suppress-all is NOT a near-miss to re-measure on P0-B; it is a closed/negative
  lever. Re-proposing it = re-proposing a closed lever.
- **A-11 (asymmetric ILP birth/death) is already shipped**, not an open lever: deployed
  APPEARANCE=0.0 / DISAPPEARANCE=1.5. The findings-table entry is stale. Only a +/-0.1
  micro-sweep remains, which A-20 classifies as noise.
- **D-06: N_pred is NOT invariant in deployment.** `OUTPUT_KEEP_DIVISION_COMPONENTS=1`
  exempts division components from the short-track filter, so any fork edit silently moves
  node counts by ~+0.0005. Not a win, but every future division experiment on the true
  wrapper must diff `num_pred_nodes` (the phaseb harnesses copy node rows verbatim and
  cannot see this). A correctness caveat, flagged so it is not mistaken for a lever.
- **D-01 / D-07: global division operating point is dead.** Any "rank the flow-midpoint
  residual globally" proposal caps at +0.0011/-0.0010 and is negative on 6bba. Within-mother
  rank is informative; a global threshold is not. E-04 (temporal NMS = within-mother-time
  normalisation) is the only division re-frame that is not already falsified, and it is CPU-
  cheap, but it lives behind the same D-07 break-even wall.

## One-line ranking (EV x cheapness)

1. Arm B joint-with-harmonic — measured +0.008 bilateral on deployment substrate; CPU
   interaction-bound then 2 GPU LOEO sessions.
2. Persist P3 OOF graphs once — enabling capital; fold into #1's GPU session.
3. RT-05 decile probe (20 min CPU) — retire the node-budget channel with evidence.
4. E-02 border-conditioned retention — fixed-count CPU reshuffle on p0strict graphs.
5. Low-EV config sweeps (MIN_TRACK_LEN, GAP_DENSITY_GAIN, E-09, safe-div caps) — free ride
   once #2 lands; expect noise.
