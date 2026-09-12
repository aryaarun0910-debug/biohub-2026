# Rank 1 — the plan

Rewritten 2026-09-11 after seven research agents swept all 102 discussions, 63 of the 203
scored notebooks, the organisers' source and its git history. The previous version is kept at
`RANK1-PLAN.superseded-2026-09-11.md`; most of its strategic reasoning was wrong and is
corrected below. Every number here is in the base — `python3 tools/ask.py facts <topic>`.

## The board, decomposed

The single most useful thing found tonight. A 0.947 notebook published its own kernel output:

    adj_edge_jaccard  0.9260
    division_jaccard  0.2308      (tp/fp/fn = 3/1/9)
    score             0.926 + 0.1 × 0.231 = 0.9491   ≈ its 0.947 on the board

So the plateau is **not** "edge 0.945, divisions at zero". It is **edge ~0.926, divisions
~0.231**. Headroom is near-equal: edge **+0.074**, division **+0.077**.

| | score | where it comes from |
|---|---|---|
| honest public ceiling | **0.9465** | ~400 teams, one codebase, different env vars |
| our best | 0.932 | 2026-08-30 and 08-31, 23 submissions, all COMPLETE |
| rank 1 | 0.970 | private work; needs edge ~0.947 at the plateau's divJ |
| north star | 0.980 | needs edge 0.940 **and** divJ 0.40 — neither term alone reaches it |

Everything displayed above 0.95 on the *notebook* rankings is a ghost: all 10 are
exploit-contaminated, and Kaggle froze their badges before the 2026-07-23 rescore. Their authors
now sit at 0.881–0.939. **The exploit is dead** — patched 2026-07-17, and a 2026-09-03 re-run at
2× scale scored 0.939, *below* the clean plateau.

## Where the score actually is

**Divisions, but not for the reason I first argued.** The plateau already scores divJ 0.231.
What makes it the better lever is that it is *frozen*: an eight-config post-process sweep leaves
divJ pinned at exactly 0.2308 in seven of eight configs. **Every knob the entire field tunes
moves only the edge term.**

The mechanism is structural, and it is the most actionable finding we have:

- With the pack's defaults (`division_weight=1.0`, `edge_weight=-1.0`) **a split earns exactly
  the same net ILP objective as a plain continuation.** The top tier's response is to price forks
  *above* continuation (1.2 against a disappearance of 1.5), which makes a fork never optimal.
- The public stack then **rebuilds its edge list from a two-pass Hungarian before `safe_div`
  runs**, destroying every fork the ILP produced. A frame-pair Hungarian cannot fork.
- Divisions are bolted back on by a filter cascade in which **every stage subtracts and none can
  propose**. Measured: 111 safe divisions inserted across 8 embryos yield **3 TP**.
- Someone measured the symptom directly: parent and both daughters detected **7/7**, both edges
  present **0/7**, parent out-degree 1 every time.

Divisions are not missed for want of a division model. They are **proposed, then destroyed**.

### What is already ruled out

Do not re-pay for these — all measured, all in the base:

- **Widening geometric gates.** On a dense lineage with 15,868 real divisions, 99.1% already have
  both arcs inside 7 µm. The public 9/14 µm gates are right. Recovering the rest is worth +0.002.
- **Distance gates generally.** Continuation:division is 203:1 at 7 µm and 875:1 at 9 µm. A long
  parent step is evidence *against* a division.
- **Appearance features.** Peak intensity gives AUC 0.73; elongation is chance. The one person who
  pushed hardest on it reports "I don't get good results".
- **Parent motion history.** Flat at 1.315 µm for eight frames before the split. Chance.
- **Weight-tuning the existing ILP.** FPs enter the denominator 1:1 against ~151 GT divisions, so
  divJ 0.22 needs a fork budget in the low thousands. One ILP emitted 1 TP against 3,260 FP.

### What is untried

- **Trackastra** (ECCV 2024, Cell Tracking Challenge winner) forks **natively**. The one notebook
  using it is bottlenecked by classical DoG detection at 0.616. It has never been paired with the
  support-pack detector.
- **FOCUS-3D pseudo-labels.** A competitor trained with *zero* Kaggle annotation — nodes from
  FOCUS-3D dense centroids, edges from synthetic augmented pairs — and got edge recall 0.84–0.99
  (mean 0.93) on held-out annotated samples, training **200 epochs without overfitting** against
  the loss-blowup-after-10 that three people report on sparse labels.
- **The positive-unlabelled detection loss already on our disk, unused** —
  `support-pack/source_scripts/train_full_frame_center_detector.py`: Gaussian target σ=1.0,
  `w_pos=12.0`, `w_bg=1.0` only below the 40th intensity percentile, `w_ignore=0.05` on bright
  unlabelled tissue. The deployed loss teaches every unannotated *real* cell to be a hard
  negative — the likely cause of that instability.
- **The division upweight hook is dead code**: `weight = torch.ones_like(loss); weight[div_rows] = 1.0`.

## The linking stack is settled (EXP-7 → EXP-10, 11–12 Sep)

Every threshold downstream of detection has now been swept **leave-one-embryo-out** against the
**full competition score** over all 199 datasets. 6bba carries 125 of the 151 divisions and the
test set is embryo-disjoint, so a pooled sweep only ever fits 6bba; the protocol is fit on one
embryo, report on the other, both directions, take the minimum.

| | before | after | Δ |
|---|---|---|---|
| score | 1.0920 | **1.1412** | +0.0492 |
| edge Jaccard | 0.9645 | **0.9955** | |
| division Jaccard | 0.3908 | **0.5431** | |
| division tp/fp/fn | 68/23/83 | **126/81/25** | |
| 44b6 / 6bba gap | 0.064 | **0.031** | narrowed |

**`motion_gate_um` 10 → 14 is the single biggest lever found** (+0.0396). It confirmed and then
fixed the recall diagnosis: 6bba's division FN fell 69 → 21, because those divisions were never
*proposed* as candidates. The candidate surface is one-dimensional in the effective radius
R = `motion_gate_um`·√(1−`edge_prob_min`), peaking at R ≈ 10.0 µm — tune R, never the two axes
separately.

**The fork discriminators are dead weight.** `fork_cos_max` is monotone loss on *both* embryos at
every tightening step, so it is off (1.0). Only divergence persistence earns its place, at the
loosest non-trivial setting: sisters must merely not *re-converge*. Re-tested after the candidate
set widened and the verdict held — low precision is not an argument for tightening, because
divJ = TP/(TP+FP+FN) penalises a lost true division exactly as hard as a false one.

**Three bugs, all found by sweeping rather than reading:**

1. The divergence check had **never once executed** — it indexed a frame-`t` parent map with a
   frame-`t+1` daughter node. Invisible except as identical results across every parameter value,
   including the disabled sentinel. Now a post-pass, since the t+1→t+2 assignment does not exist
   while frame `t` is being resolved.
2. `resolve` could give one parent **three or more children**. The official scorer silently
   *discards* the surplus rather than erroring, so this was corruption, not a crash. Out-degree is
   now capped at 2; zero scorer warnings across all 199.
3. The sweep harness itself optimised divJ alone — which would have reported edge-term losses as
   wins the moment it was pointed upstream of `resolve`. The objective is now the full score and
   all four stages run. Parity with `run_pipeline.py` verified to 4 dp before it was trusted.

### What this number is not

**1.1412 assumes oracle detection.** `detect` and `refine` are still stubs. It is the ceiling of
the linking stack given perfect inputs, not a submittable score, and not comparable to the 0.970
at rank 1. The whole remaining gap is detection.

One consequence is live and **must not be forgotten**: `repair` is disabled (`gap_max_frames=0`,
`min_track_len=1`) because under an oracle there are no detector misses to bridge and no spurious
tracks to prune. **Both premises reverse under a real detector.** The value carries an inline
warning in `contracts.py` and the decision is recorded **open**, not settled — re-sweep it the day
`detect` trains.

## The detector: dual-head, dense-supervised, with a refinement head

Added 2026-09-11 from the Obsidian vault. This is the single largest architectural change
against everything else on the board, and no public pipeline does any of it.

**Supervise detection densely, refinement sparsely.** The sparse annotation is the root of the
field's training instability — a BCE detection loss on single hard voxels teaches every
*unannotated real cell* to be a hard negative, which is why three separate people report loss
blowing up after ~10 epochs. The fix is not a better loss on sparse labels; it is to stop using
sparse labels for detection at all:

    node_logit  = node_head(x)      # supervised DENSE  — FOCUS-3D pseudo-centres
    refinement  = refine_head(x)    # supervised SPARSE — Kaggle, CONDITIONAL on detected peaks
    refine_coord = coord + dzyx ;   refine_logit = logit + dlogit

Sparse labels are fine for refinement, because refinement is only ever evaluated *at peaks that
already exist*. Measured by its author: **centroid error 1.18 → 0.90–0.97 voxels, an 18–24%
reduction, with 70–75% of points improved, 10–14% of peaks pruned, and Kaggle recall held at
99.5%.** Training runs 200 epochs without overfitting, against the ~10 that sparse-label
training survives.

Why this matters more than it looks: **centroid error has a cliff at σ≈2 µm** — 2.5 µm costs
16% of the edge Jaccard, 3 µm costs 41% — and separately, **a single duplicate detection 0.4 µm
away halves it**. A 20% error reduction plus a 10–14% peak prune lands squarely on the two most
expensive failure modes in the metric, and `refine_logit` gives the prune a learned ranker
rather than a threshold.

**Corollary that breaks our stage contract.** *"During linking, my zxy must change… you cannot
detect and fix a location."* Detection and localisation are not separable: many false positives
sit very close to a true node and are recoverable by moving them. So `detect → link` must not
freeze node positions — the graph carries mutable coordinates, and the linker may refine them.

**Do not augment.** Removing augmentation improved peak pruning (8% → 10–14%) *and* error
reduction (18% → 22–24%) while holding recall, and halved epoch time from 6.7 to 3.5 minutes.
Diagnosed as "too difficult or outside the real validation distribution".

**Train the linker on predicted points, not GT points.** Feed it `t0 = volume + predicted dense
points` and `t1 = augmented volume + predicted dense points`, copying edge links across by
linear assignment. This removes the train/inference mismatch we found — training currently picks
peaks at a raw logit of 0.3 (p≈0.574) while inference uses sigmoid > 0.99, so the linker has
never seen the candidate set it is scored on.

### Gate

Centroid error on held-out annotated nodes must fall ≥15% against the un-refined detector, with
Kaggle recall ≥99.0%. Below that, keep the plain detector and spend the time on the linker.

## Edge term: cheap wins first

- **NMS quality outranks everything.** A single duplicate detection 0.4 µm away **halves** the
  edge Jaccard, 1.000 → 0.500. Centroid error has a cliff at σ≈2 µm: 2.5 µm costs 16%, 3 µm 41%,
  4 µm 74% — driven by ~9–10 µm inter-cell spacing.
- **Per-movie motion gate.** The public pipelines inherit the metric's 7 µm *node-matching*
  tolerance as their *motion* gate, discarding ~2% of true links for free. The loss is per-movie —
  the worst loses one link in seven — so a per-movie p99 gate beats any constant.
- **Train/inference threshold mismatch.** Training picks peaks at `det_logits > 0.3` on the raw
  logit (p≈0.574); inference uses `sigmoid > 0.99`. The linker was never trained on the candidate
  set it is scored on. Training also matches at 5.0 µm against the metric's 7 µm.
- **Gap recovery means a NODE, not an edge.** All 7,998 GT edges span exactly one frame, so a
  t→t+2 edge can never match — the scorer drops it. Recover the intermediate node, emit two
  1-frame edges. Gaps are bounded: ~20 one-frame, ~3 two-frame, ~2 longer per sample.
- **+0.024 of edge recall is in RANKING, not detection.** Accumulating candidates to a 0.99
  cutoff reaches 0.966 edge recall; greedy top-1 gets 0.942. The right edges are already in the
  candidate set. A re-ranking module beats a better detector here.

## The traps that end runs

1. **Silent 0.0 from non-consecutive edges.** The scorer filters `dt≠1` edges *before* counting,
   so TP is structurally zero however good the detection is — and the CSV is accepted without
   error. A worked case scored exactly 0.0 against ~0.57 locally after striding frames to fit the
   time budget. **The pressure that causes this arrives in the final week**: the rerun takes 5–12 h
   against a hard 12 h ceiling and the visible clips run ~50× faster, telling you nothing.
   **Gate is built and verified: `python3 tools/validate_submission.py submission.csv`.**
   Stdlib-only, so it runs inside the internet-disabled notebook that writes the file. FATAL on
   non-consecutive edges, edgeless graphs, dangling endpoints, negative timepoints and
   out-of-volume coordinates; WARN on out-degree >2 and merges. Run it in the last cell, before
   the file is written, every single time.
2. **Never select on `data/test`** — four volumes, byte-identical to train, replaced at rerun.
3. **The Kaggle Evaluation page is stale.** It still describes the pre-patch rule. Design to
   `metrics.md` and the vendored source.
4. **`support/` shipped the PRE-PATCH scorer** until tonight. Anything validated against it was
   calibrated to the closed exploit. Fixed; originals in `PRE-PATCH-DO-NOT-USE/`.
5. **Frozen frames.** 947 byte-identical consecutive frame pairs across 114 of 128 `6bba` videos
   and **zero** in all 71 `44b6`. In one, GT moves 8.90 µm across identical pixels — beyond the
   7 µm radius. Copying a detection across a frozen frame is *guaranteed* to miss, and any fix
   tuned on train is tuned on one embryo while test is embryo-disjoint.
6. **Ties break on earliest submission.** 242 teams share 0.946, 178 share 0.947. A score-neutral
   resubmission at the plateau strictly loses rank.

## Runway: 12–29 September

Today is the 12th. The Mac lands on the 14th. The CPU-only work is **done** — see "The linking
stack is settled" above. Everything that remains needs the machine.

**Now → 13 Sep.** Kaggle identity verification is the only outstanding item, and it is the one
with a queue you do not control: it gates prizes. The `.geff` corpus is complete (4,179 files,
199/199 datasets) and every sweep above ran on it.

**14–15 Sep.** Machine up. Pin **torch ≥ 2.14.0** — conv3d on MPS was fixed there and is ~16×
faster with `bias=True`; build decoders from `Upsample`+`Conv3d`, never `ConvTranspose3d`. Port
the contracts, metric adapter, submission writer and digest system from Biohub-X. **Be
submittable.**

**16–18 Sep.** Reach the honest plateau. It is one codebase with `BIOHUB_*` env vars: 0.912→0.940
is a single edit, 0.940→0.947 is three flags. Weights are public. Gate: edge ≥0.92 held out.

**16–18 Sep (revised).** Dual-head detector first — it feeds everything downstream, and EXP-5
showed the linker fix is cheap once detection is good.

**19–24 Sep.** Divisions are **already past the gate offline** — divJ 0.5431 against a target of
0.30 — so this window is now about holding that under *real* detections rather than an oracle.
Re-sweep `repair` (the open decision) and re-check `motion_gate_um` the moment `detect` produces
noisy centroids, since both were tuned against perfect inputs. Do not re-sweep the fork
discriminators: EXP-9 already retested them against a wider candidate set and the verdict held.
Note 22 Sep is the entry and team-merge deadline.

**25–27 Sep.** Whichever term the numbers say is weaker. One change at a time.

**28–29 Sep.** Freeze, final two submissions. Treat the 28th as the deadline. Pick one
high-public and one low-variance run — public LB is 29% of test.

## Standing rules

- Five submissions a day. Keep two in hand on each of the last two days.
- **Never let a component train on data you need to validate the chain on.** The prior campaign's
  offline instrument was structurally broken because a model had seen all 199 labelled crops.
- Trust no offline proxy's *direction*: DeepCenter read +0.0009 offline against +0.003 on the LB;
  `icom` read +0.002 and scored −0.014.
- Numbers live in `db/biohub_base.db` once. Prose cites; prose never restates.
- Declare the falsifier before the run.
- `./tools/colab_vms.sh` before walking away. An idle A100 is ~15 units/hour.
