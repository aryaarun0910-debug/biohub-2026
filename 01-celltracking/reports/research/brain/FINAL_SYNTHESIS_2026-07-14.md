# Final synthesis — the detect→link→repair family is exhausted at ~0.90–0.91

**Date:** 2026-07-14. Written after exhaustively, exactly measuring every lever against the
frozen, deployment-exact E0c baseline (44b6 0.7595 / 6bba 0.6490, public 0.889). This is the
honest strategic conclusion; it supersedes the optimistic framings in earlier docs.

## Executive verdict

Every incremental and "winning-scale" lever inside the shared field architecture (framewise
detect → link → graph-repair) has been measured to its exact ceiling. **All of them are
either saturated, capped small, or falsified.** The disciplined, honest system is the frozen
E0c wrapper at **~0.889 public / ~0.90–0.91 realistic ceiling.** There is **no measured lever
with a credible mechanism to reach 0.94+**. The only remaining route to materially exceed
~0.91 is a **genuinely different architecture** (below), which is expensive and uncertain.

## The exact evidence (every lever, measured)

| Lever | Exact result vs E0c | Verdict |
|---|---|---|
| Candidate **reranking** (geometry / Zebrahub-breadth), answerable-group top-1 | wrapper 0.9635/0.9319; breadth beats it on neither fold | **saturated** |
| **Suppress** existing forks (no-fork ablation) | −0.0000 / −0.0008 | **net-neutral** |
| **Divisions — reachability** oracle (optimistic) | +0.027 / +0.020 | overstated ~6× |
| **Divisions — exact** Oracle C, add-only (a classifier's ceiling) | +0.0045 / +0.0037 | small |
| **Divisions — exact** Oracle C, add-replace (conflict-resolved upper bound) | +0.0182 / +0.0138 | ceiling; needs a joint selector, not a classifier |
| **Divisions — learned** (leak-free, daughter-swap-invariant, T4) | recall@P0.9 mean **0.045** (~0 on 3/4 embryos) | **dead** — cannot operate at high precision cross-embryo |
| **Bracketed** gap-completion (safe) | ≤1.5% / 0.7% edge-recall ceiling; 44%/81% of misses isolated | small |
| **Isolated de-novo** detection (DAXI tracklets, Stages 2–4) | oracle recovery **0% / 8.8%** (≪20% gate); recovering few misses adds 3k–1.4M nodes → count collapse | **dead** |
| Selective association repair | expected <+0.005 (reranking saturated) | marginal (not separately run) |

**Net:** the largest *exact* deployable gain available is the division joint-selector at a
realistic fraction of +0.018/+0.014 — but the learned posterior that would drive it cannot
hit high precision cross-embryo, so even that is not currently deployable. **Nothing has
cleared the both-fold exact gate as a deployable improvement over 0.889.**

## Why each winning route died (specifically)

1. **Divisions.** The exact ceiling (+0.018/+0.014) lives almost entirely in *conflict
   resolution* (removing daughters' wrong parents / daughter competition), not classification
   (add-only is only +0.004). Realizing it needs near-perfect fork identification feeding a
   joint fork+edge selector. But the leak-free trajectory model gets **~0 recall at 90%
   precision on 3 of 4 held-out embryos** — it ranks divisions (PR-AUC 0.72) but cannot
   *identify* them at high precision across embryos, even on easy negatives. A high-recall-only
   posterior floods false forks; the deployable gain collapses toward the +0.004 add-only
   number (RED).
2. **Isolated-miss detection.** Isolated misses are reachable per-frame (Stage-1 100%) but the
   candidates that reach them are transient noise: they do **not** form coherent multi-frame
   tracklets (oracle recovery 0–9%). A lone recovered node restores **no edge** (its GT
   neighbours are also missed) yet still pays the node-count penalty, and low-threshold
   proposals add thousands-to-millions of spurious nodes. The count penalty is decisive.

## The disciplined floor (ship this)

The frozen **E0c wrapper = public 0.889** is the honest, deployment-exact system, already
submitted. No new configuration improves it on both folds exactly, so **no new submission is
justified** on current evidence. Private rank is unknowable; our only edge there is that the
board is a monoculture forking one baseline, so relative generalization *may* rank us
reasonably — but we have no measured lever to improve it.

## The only remaining route to >0.91 — a genuinely different architecture

The field's shared pipeline destroys weak temporal evidence *before* association. The
isolated-miss result is the key clue: single-frame detection + tracklet linking cannot
recover the weak cells that cap edge-J, and reranking existing candidates is saturated. The
one mechanism not yet tried is **learned motion-compensated temporal evidence integration**
(proper track-before-detect): warp a 3–7-frame window into candidate material coordinates and
**accumulate sub-threshold detector response along motion-compensated tubes BEFORE declaring
a node** — declaring a cell only when integrated evidence + motion consistency support it.
This directly attacks the isolated-miss population that the single-frame + tracklet approach
provably cannot (my DAXI experiment failed precisely because it thresholds each frame
independently instead of integrating weak evidence over motion).

**Honest risk assessment of this bet:**
- **Upside:** the only mechanism that could raise the detection ceiling on weak/isolated cells
  — the population large enough (44%/81% of misses) for a winning-scale gain.
- **Downside:** expensive (from-scratch 4D/temporal GPU model, breadth-pretrained), uncertain
  (if a cell is genuinely absent/occluded, no integration recovers it), must run offline ≤12h,
  and must survive the same count-penalty and cross-embryo-precision bars that killed the
  other levers. The evidence that isolated misses form *no coherent tracklet* is mildly
  discouraging — but tracklets from thresholded single frames are not the same as integrated
  sub-threshold evidence, so it is not decisive against the bet.

## Recommendation

1. **Ship nothing new** — 0.889 stands; no measured deployable improvement.
2. **Make a deliberate go/no-go on the temporal-integration moonshot** — it is the only route
   with a credible mechanism to exceed ~0.91, but it is a multi-week GPU program with real
   risk of also capping near the detection oracle. Before committing, run ONE cheap
   falsification: does motion-compensated accumulation of raw DAXI response along short tubes
   reveal a signal at isolated-miss locations that single-frame thresholding misses? (Extends
   the existing DAXI cache; no new model.) If yes → the bet has a mechanism; if no → the
   detection ceiling is real and ~0.90–0.91 is the honest end of this competition for us.
3. **Otherwise:** accept the disciplined ~0.90 floor and stop spending compute on saturated
   levers.

The value of this session is negative certainty: we now know, with exact measurements, that
the obvious levers cannot win — which prevents pouring weeks into any of them. The one honest
moonshot is temporal evidence integration, gated by the cheap accumulation falsification above.

## Addendum (2026-07-14, later same day) — the falsification came back negative

Ran the prescribed cheap falsification (`scripts/win_bet/daxi_accumulation_v2.py`, oracle GT-motion
accumulation vs static accumulation, paired hard controls from cached DAXI peaks). On the only
adequately-powered crop (6bba, 55 pairs), motion-compensated accumulation was **lower** than static
accumulation (0.1921 vs 0.2225 AUC, −0.0304); 44b6 (3 pairs) is underpowered. Full numbers and caveats
in `reports/inventory/daxi_accumulation_v2.txt` and the 2026-07-14 journal entry.

**Per this doc's own decision rule: this is a "no."** The oracle-motion mechanism this bet depends on
did not show an advantage on the powered test. A real confound (control matching used cached full-volume
response, evaluation used patch inference) keeps this from being a fully clean kill, so the moonshot is
marked **unsupported, not disproven** — but per the recommendation above ("if no → the detection ceiling
is real and ~0.90–0.91 is the honest end of this competition for us"), no multi-week temporal-integration
GPU program is justified on current evidence. Combined with the division posterior (dead) and isolated
detection (dead) results, **all three candidate winning-scale levers are now negative.**

**Current disciplined position:** ship nothing new; the frozen E0c wrapper (public 0.889, OOF 0.7595/0.6490)
stands as the floor. Any further architecture work should first clear its own cheap falsification gate
(e.g., a patch-response-matched rerun of this pilot with more 44b6 crops) before any GPU commitment.

## Post-patch execution note (2026-07-19)

Official scorer commit `075fc5f` leaves the substantive results stable. Full E0c remains
0.7595/0.6490; no-fork is neutral/-0.0002; exact Oracle C remains +0.0045/+0.0037
add-only and +0.0182/+0.0138 add-replace. The synthesis therefore still rules out a
plain division head, but the earlier temporal pilot's documented control-path confound is
now being closed with a pre-registered 20-crop, response-path-matched T4/CPU rerun. In
parallel, the clean-public-0.903 wrapper delta is receiving one isolated bilateral OOF gate.
