"""Generate the Obsidian vault in vault/ from a single source of truth.

The vault is the project knowledge base as a GRAPH: 80 interlinked notes, so
Obsidian's graph view shows how findings actually depend on each other rather
than a folder of files.

It is generated rather than hand-maintained so that two invariants hold:

  * EVERY [[wikilink]] resolves. The script exits non-zero and names the
    offender otherwise -- a dangling link is a node with no note, which is
    exactly the thing that makes a graph view lie.
  * NO note is orphaned. Anything nothing links to is reported, because an
    unreachable note is one nobody will find.

Regenerate with:  python scripts/build_vault.py
It deletes and rewrites every .md under vault/, so edit THIS file, not the
notes. vault/.obsidian/ is left alone (graph colours, app settings).

Folder layout is cosmetic; Obsidian builds the graph from links, not folders.
"""
from pathlib import Path
import re, sys

ROOT = Path("/Users/aryaarun/Developer/biohub26")
V = ROOT / "vault"
N = {}

def note(path, tags, body):
    N[path] = (tags, body.strip() + "\n")

# ─────────────────────────────── INDEX ───────────────────────────────
note("Home", ["moc"], """
# Biohub Cell Tracking — Vault

Everything known about this competition, as a graph. Start at a hub below.

## Hubs
- [[Competition]] — the task, the deadline, the money
- [[Tracking Pipeline]] — what actually runs
- [[The Metric]] — how score is computed, and how it can be gamed
- [[Submission Ledger]] — every submission and what it scored
- [[Local Harness]] — the thing that makes experiments cost seconds
- [[Operating Rules]] — the discipline that keeps results honest
- [[Scripts]] — every script and what it proves
- [[Failed Attempts]] — twelve dead ends, with the reason each died
- [[Kaggle Mechanics]] — the constraints that shape a day's work

## The three ideas that matter most
1. [[Transfer Lesson]] — a price measured on one topology does not hold on another. Three confirmed instances.
2. [[Ordering Bug Class]] — the recurring defect is *sequence*, not parameters.
3. [[Node Count Exploit]] — the metric pays for deleting nodes, without bound.

## What we are waiting for
- [[Board Predictions]] — pre-registered bands and what each outcome would mean

## Where the remaining headroom is
- [[Error Budget]] — 80% of the error is the linker; the detector ceiling is +0.0126
- [[Edge Pruning Is Dead]] — the largest class is real but not reachable by thresholding

## The decision layer — read before proposing any change
- [[Both-Sets Rule]] — what must be true before something ships
- [[Evidence Tiers]] — what each film set can and cannot support
- [[Harness Prediction Confirmed]] — the harness predicted [[s08]] before the run, to the mechanism

## Read this before trusting any local number
[[Scored Films Measurement]] — the 4 films the board scores can be measured
locally, and they disagree with the 8 validator films. [[Manifest Contamination]]
is the debate that prompted it.

## State as of 2026-09-19
- Board: **0.947**, stuck on the [[0.947 Plateau]] with 652 other teams.
- In flight: [[s08]] (neutral where scored) and [[s10]] (first change to pass the [[Both-Sets Rule]]).
- Queued: [[s11]], a data-generation run to quadruple the evidence base.
- Best known on the films that count: [[s05]], backed by [[Relink Across 199]] (176/199).
- [[Parametric Search Closed]] — every post-processing knob re-priced, nothing left.
""")

note("Competition", ["entity"], """
# Competition

**Biohub Cell Tracking During Development** (Kaggle, code competition).

| | |
|---|---|
| Closes | **2026-09-29 23:59 UTC** |
| Team | `aryaarun07` |
| Standing score | **0.947** |
| Rank | ~423 of 3,697 |
| Leader | 0.973 |
| 7th (last prize) | 0.964 |
| Gap to money | **+0.017** |

Scored by [[The Metric]]. Entries are notebooks, not files — see [[Kaggle Mechanics]].

The 0.947 score is not ours in any meaningful sense: it is the [[0.947 Plateau]],
a public-notebook result that 653 teams share exactly.

Related: [[Frozen Weights]], [[Validator Films]], [[Test Films]]
""")

note("0.947 Plateau", ["finding"], """
# The 0.947 Plateau

**653 teams sit at exactly 0.947** (ranks 186–838). That is not a coincidence —
it is the score of a widely-forked public notebook, reproduced verbatim.

Consequences:
- Any gain at all breaks a 653-way tie, so small real gains are worth more rank
  than their size suggests.
- Everything in the plateau shares the same defects, so the defects are
  *public*. See [[Motion Relink Is The Large Lever]] and [[Gap2 In The Wrong Position]].
- The base notebook is `public notebooks/biohub-0-947-lb-runnable-with-public-datasets.ipynb`.

Reproduced exactly as submission 56202098 → 0.947, confirming the fork is faithful.

Related: [[Competition]], [[Submission Ledger]], [[Tracking Pipeline]]
""")

# ─────────────────────────────── METRIC ───────────────────────────────
note("The Metric", ["concept"], """
# The Metric

```
score = adjusted_edge_jaccard + 0.1 * division_jaccard
adj   = max(0, J * (1 - 0.1 * (n_pred - n_est) / n_est))
```

Three parts, each with its own note:
- [[Adjusted Edge Jaccard]] — the bulk of the score
- [[Division Jaccard]] — worth 0.1×, but where the leverage is
- [[Node Count Multiplier]] — **uncapped above 1**, which is the trap

The multiplier being uncapped is the single most important property: deleting
nodes pays *without bound*. See [[Node Count Exploit]].

Locally computed by [[metric2]]. Do NOT use the other module — see [[Wrong Metric Module]].

Related: [[One Division Event Floor]], [[Proxy Score]]
""")

note("Proxy Score", ["concept"], """
# Proxy Score

The name for [[The Metric]] evaluated **offline**, on films where ground truth
is held — the 8 [[Validator Films]].

Two instruments compute it:
- Kaggle's **in-kernel validator**, inside the notebook run. This is the
  instrument that correctly rejected [[s01]].
- The [[Local Harness]], in seconds instead of 1.75 h.

They agree closely but not exactly: the harness omits [[Single Parent Repair]],
the short-track rescue and the [[DeepCenter]] vetoes, which costs it one extra
division false positive. See [[Harness Validated Against Kernel]].

**Proxy is not the board.** The board has spoken 3-for-3 on the division axis and
0-for-3 on the edge axis — see [[Axis Priors]].
""")

note("Adjusted Edge Jaccard", ["concept"], """
# Adjusted Edge Jaccard

`adj = max(0, J_edge * multiplier)` where `J_edge` is the Jaccard index over
predicted vs ground-truth *edges* (parent→child links between detections across
frames), and `multiplier` is the [[Node Count Multiplier]].

Because `adj` is a product, a gain can come from either factor — and a gain that
arrives through the multiplier while `J_edge` is flat is the [[Node Count Exploit]],
not tracking quality. [[Operating Rules]] requires reporting them **separately**.

Related: [[The Metric]], [[Division Jaccard]]
""")

note("Division Jaccard", ["concept"], """
# Division Jaccard

`divJ = TP / (TP + FP + FN)` over *division events* — a cell splitting into two.

Small in weight (0.1×) but large in leverage, because the denominator is tiny:
**12 ground-truth divisions across the 8 [[Validator Films]]**. One event moves
divJ by a lot. See [[One Division Event Floor]].

Ledger history on the validator films:
| configuration | TP/FP/FN | divJ |
|---|---|---|
| unmodified 0.947 | 3/1/9 | 0.2308 |
| [[s01]] `DIVERGE_UM=0` | 3/15/9 | 0.1111 |
| [[s05]] no relink | 4/1/8 | 0.3077 |
| [[s08]] + reorder (harness) | 5/2/7 | 0.3571 |

Divisions are produced by [[Safe Division]]. See [[Divisions Played Out]].

**This is the good axis** — see [[Axis Priors]].
""")

note("Node Count Multiplier", ["concept"], """
# Node Count Multiplier

```
mult = 1 - 0.1 * (n_pred - n_est) / n_est
```

`n_est` is an *estimate* of the true node count, supplied per film. `n_pred` is
what we output.

**It is uncapped above 1.** If `n_pred < n_est`, `mult > 1` and keeps rising as
you delete more. There is no ceiling. This is the [[Node Count Exploit]].

Two traps around it:
1. Deleting nodes raises it forever — [[Short Track Filter]] rides this monotonically.
2. The *aggregate* `mult` is not a node-count readout — see [[Aggregate Mult Re-weighting]].

The honest invariant is **`ratio = n_pred / n_est`**, not `mult`.

Related: [[Node Budget]], [[The Metric]]
""")

note("One Division Event Floor", ["concept"], """
# One Division Event Floor

The measurement resolution on the division axis.

```
one event = 0.1 / (TP + FP + FN)
```

- At a 12-division ledger: **0.0083** (the figure quoted in [[Operating Rules]])
- At the 5/2/7 ledger of the [[s08]] chain: **0.00714**

**Using the wrong denominator is a live error.** [[s08]] measures +0.00748; against
0.0083 that reads "below the floor", against the correct 0.00714 it reads
"exactly one division". The second is right.

Do not report division differences finer than one event as meaningful.

Related: [[Division Jaccard]], [[Operating Rules]]
""")

# ─────────────────────────────── PIPELINE ───────────────────────────────
note("Tracking Pipeline", ["moc"], """
# Tracking Pipeline

Nothing here trains a model. The whole entry is a public Kaggle notebook running
[[Frozen Weights]], all code in cell index 2 (~214k chars).

**Detection and linking:**
[[Temporal UNet3D Detector]] → [[Node Transformer Edge Scorer]] → [[ILP Linker]]

**Then a rule-based post-processing chain** — this is where every change we have
made lives, driven by `BIOHUB_*` environment variables in the config cell:

1. [[Motion Relink]] ← **removed in [[s05]]**
2. [[Single Parent Repair]]
3. [[Gap Closing]]
4. [[Gap2 Recovery]] ← **moved after safe division in [[s08]]**
5. [[Safe Division]]
6. [[Prune Isolated]]
7. [[Short Track Filter]]
8. [[Linefit Smoothing]] ← **retuned in [[s09]]**

[[DeepCenter]] supplies vetoes at two points.

The chain is now **[[Parametric Search Closed]]** — every knob re-priced, nothing left.
""")

note("Temporal UNet3D Detector", ["stage"], """
# Temporal UNet3D Detector

Produces candidate cell centres per frame. Runs from [[Frozen Weights]]
(`pilkwang/biohub-temporal-unet3d-seed314159-v1`).

Controlled by `BIOHUB_DET_THRESHOLD` (0.965), which is **upstream** — the
[[Local Harness]] cannot test it, because the harness starts from the
[[ILP Linker]] output. See [[Upstream Knobs]].

Raising the threshold 0.965→0.995 is ~33× cheaper per node removed than any
other route ([[Node Budget]]) — but that is the node-count axis, which
[[Operating Rules]] declares forbidden ground for *tuning*.

Related: [[Tracking Pipeline]], [[Drift Guard]]
""")

note("Node Transformer Edge Scorer", ["stage"], """
# Node Transformer Edge Scorer

Scores candidate parent→child edges between consecutive frames. Frozen.

Its knobs — `BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT` (0.15) and
`BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT` (0.75) — are [[Upstream Knobs]]:
untestable locally and all under the [[Drift Guard]].

A quirk with consequences: every `solution` flag in the prediction `.geff` is
True, so the file stores **only ILP-selected edges**. That is what cripples
[[Learned Bonus Is Crippled]].

Related: [[ILP Linker]], [[Local Harness]]
""")

note("ILP Linker", ["stage"], """
# ILP Linker

Integer linear program producing a global frame-to-frame assignment. Its output
`.geff` files are what the [[Local Harness]] reads — so everything downstream is
testable in seconds, and everything upstream is not.

Weights `BIOHUB_ILP_APPEARANCE_WEIGHT` (0.0) and
`BIOHUB_ILP_DISAPPEARANCE_WEIGHT` (2) are [[Upstream Knobs]].

The ILP already produces forks (divisions). [[Motion Relink]] then wipes them —
the first instance of the [[Ordering Bug Class]].

Graph shape note: no isolated nodes, minimum component size 4 in every film — the
ILP enforces this, which is why [[Prune Isolated]] is inert.
""")

note("Motion Relink", ["stage"], """
# Motion Relink

**The one large lever, and it was doing harm.** Removed in [[s05]].

Rewires edges using a two-pass distance gate: a tight pass at 6 µm, then a
relaxed pass at 10 µm.

**The mechanism of the defect:** the tight gate *forecloses* rather than defers.
A source matched to a nearer wrong target enters `used_i`, and the relaxed pass
can never revisit it. Pooling 5,540 correctly-linked GT edges, the break rate
jumps **171×** with a clean knee exactly at the 6 µm gate — 0.17% below, 28.6%
above.

Worth **+0.0224** on Kaggle's validator to delete (+0.0295 predicted locally).
An oracle per-film adaptive skip beats deletion by only 0.0014, so there is
nothing better to build than removal.

With relink off, five of the eight in-kernel sweep candidates become no-ops —
see [[In-Kernel Sweep Is Inert]].

Related: [[Ordering Bug Class]], [[Learned Bonus Is Crippled]]
""")

note("Single Parent Repair", ["stage"], """
# Single Parent Repair

Enforces that each node has at most one incoming edge. On by default
(`BIOHUB_OUTPUT_SINGLE_PARENT_REPAIR`).

**Not ported to the [[Local Harness]]** — one of the three known gaps between the
harness chain and the kernel chain, alongside the short-track rescue and the
[[DeepCenter]] vetoes. Held identical across arms in every A/B, so it cannot
manufacture a difference, but it means absolute harness proxy ≠ kernel proxy.

Related: [[Harness Validated Against Kernel]]
""")

note("Gap Closing", ["stage"], """
# Gap Closing

Rejoins a track end at *t* to a track start at *t+2*, Hungarian per frame.
Prefers reusing an existing unlinked node near the midpoint over synthesising one.

Deployed: `GAP_CLOSE_UM` 5.0, `GAP_CLOSE_REUSE_UM` 3.2, `GAP_CLOSE_MAX_GAP` 2.

It **eats the orphan pool** that [[Safe Division]] draws daughters from — the same
competition that makes [[Gap2 Recovery]]'s position matter. But measured, moving
gap closing after safe division is worth only ~0.00001. Not a lever.

`GAP_CLOSE_UM` 5→8 is **+0.00089 on raw graphs and −0.00015 on the [[s09]] chain** —
the third instance of the [[Transfer Lesson]].

`GAP_CLOSE_UM` is under the [[Drift Guard]].

Related: [[Parametric Search Closed]], [[Inert Variables]]
""")

note("Gap2 Recovery", ["stage"], """
# Gap2 Recovery

Two-frame recovery: joins a track end at *t* to a start at *t+3* via two
synthetic nodes. `GAP2_MAX_TOTAL_UM` 10.2, `GAP2_MAX_STEP_UM` 4.4.

**It was in the wrong position.** See [[Gap2 In The Wrong Position]] — the finding
behind [[s08]].

The start it joins to is an *orphan* (no incoming edge), which is exactly the pool
[[Safe Division]] draws its second daughters from. Run gap2 first and it eats a
daughter: one true division becomes a false positive plus a false negative.

Its own parameters are inert — `g2_step`, `g2_frac`, `g2_abs` all return exactly
0.00000. See [[Inert Variables]].

Related: [[Ordering Bug Class]], [[Gap Closing]]
""")

note("Safe Division", ["stage"], """
# Safe Division

Proposes a second child for a parent that has exactly one, drawing the candidate
from the **orphan pool** (nodes with no incoming edge). This is what produces
[[Division Jaccard]] true positives.

Gates (all deployed values):
| gate | value |
|---|---|
| `SAFE_DIV_MAX_UM` (parent→daughter) | 9.0 |
| `SAFE_DIV_SISTER_MAX_UM` | 14.0 |
| `SAFE_DIV_EXISTING_CHILD_MAX_UM` | 10.0 |
| `SAFE_DIV_SISTER_SYMMETRY_TAU` | 0.6 |
| `SAFE_DIV_DIVERGE_UM` | 2.25 |
| `SAFE_DIV_FRAME_FRAC_CAP` | 0.0076 |
| `SAFE_DIV_GLOBAL_FRAC_CAP` | 0.00375 |

**The gates are at a sharp local optimum** — re-confirmed on the [[s09]] chain, see
[[Safe Div Gates At Local Optimum]]. `tau` loses a division in *both* directions.

One never-touched boolean is load-bearing: `SAFE_DIV_REQUIRE_DIVERGENCE=0` costs
**−0.025** with division FP 2→20.

Because it competes for orphans, everything that consumes orphans must run
*after* it. See [[Ordering Bug Class]].

Related: [[s01]], [[Divisions Played Out]]
""")

note("Prune Isolated", ["stage"], """
# Prune Isolated

Drops nodes with no incoming and no outgoing edge.

**Completely inert on every chain tested** — `prune=False` returns exactly
0.00000. The [[ILP Linker]] already guarantees no isolated nodes and a minimum
component size of 4, so there is nothing for this stage to remove.

Related: [[Inert Variables]], [[Parametric Search Closed]]
""")

note("Short Track Filter", ["stage"], """
# Short Track Filter

Removes connected components shorter than `OUTPUT_MIN_TRACK_LEN` (deployed 6),
optionally keeping components that contain a fork.

**This stage is the [[Node Count Exploit]] in its purest form.** Raising the
threshold makes edge Jaccard *fall* while the [[Node Count Multiplier]] rises,
and the multiplier gain is **monotone all the way to L=40**. A genuine stage
would peak somewhere. This one never does.

On the [[s09]] chain the screen flags it automatically:
- `st_len=8` → proxy **+0.00103**, dJ **−0.00157**, dratio **−0.03218**
- `st_len=9` → proxy **+0.00041**, dJ **−0.00324**, dratio **−0.04598**

Proxy up, tracking quality down. The board reading is consistent: the historical
`OUTPUT_MIN_TRACK_LEN` 6→9 submission scored 0.947, no better than base.

Related: [[Operating Rules]], [[Failed Short Track Exploit]]
""")

note("Linefit Smoothing", ["stage"], """
# Linefit Smoothing

Moves each node toward a line fitted over ±`window` frames of its unique
predecessor/successor chain. **Topology and node count untouched** — it can move
[[Adjusted Edge Jaccard]] only through `J`, never through the multiplier.

Deployed `OUTPUT_LINEFIT_WEIGHT` 0.8, `OUTPUT_LINEFIT_WINDOW` 2. Neither key is
set in the notebook — both sit at their `os.environ.get` default, so changing one
*adds* a line.

**The textbook case of the [[Transfer Lesson]]:** w=0.4 is −0.00135 on the relinked
pipeline ([[s06]]) and **+0.00372** on the no-relink chain. Because it fits along
unique chains, and [[Motion Relink]] changes exactly that topology.

Retuned to 0.3 in [[s09]]. See [[Linefit Surface On s09]].

It must run **after** [[Safe Division]] — smoothing first collapses divisions from
5/2/7 to 3/1/9 or 2/3/10. In the deployed notebook it already is last.
""")

note("DeepCenter", ["stage"], """
# DeepCenter

A separate UNet3D centre-prior model (`pilkwang/biohub-deepcenter-unet3d-center-prior-v1`)
used as a **veto** at two points: gap confirmation and safe division.

`DEEPCENTER_GAP_VETO` 1 / threshold 0.25; `DEEPCENTER_SAFE_DIV_VETO` 1 /
threshold 0.20 (the latter under the [[Drift Guard]]).

As a division *ranker* it is real — AUC 0.836–0.857 — but it is already **at its
oracle ceiling**, worth +0.0029, and a plain brightness threshold matches it. The
whole gain is one fork in one film. See [[Failed DeepCenter Veto]].

Not ported to the [[Local Harness]], which is why the harness carries one extra
division false positive.

Related: [[Harness Validated Against Kernel]]
""")

# ─────────────────────────────── SUBMISSIONS ───────────────────────────────
note("Submission Ledger", ["moc"], """
# Submission Ledger

Recorded in `artifacts/ledger/runs.db` (SQLite: `runs`, `scores`,
`division_events`, `submissions`). **Predictions are written before results**, so
sign agreement stays honest.

## Ours, this lineage
| id | change | parent | local | board |
|---|---|---|---|---|
| [[s01]] | `SAFE_DIV_DIVERGE_UM` 2.25→0 | base | — | rejected by validator |
| s02/s03/s04 | safe-div symmetry variants | base | — | retired, never pushed |
| [[s05]] | `OUTPUT_MOTION_RELINK`=0 | base | +0.0224 | **PENDING** |
| [[s06]] | `OUTPUT_LINEFIT_WEIGHT` 0.8→0.4 | base | **−0.00135** | not pushed |
| [[s07]] | gap2 after safe_div | base | — | **superseded, wrong base** |
| [[s08]] | gap2 after safe_div | [[s05]] | +0.00748 | running |
| [[s09]] | `OUTPUT_LINEFIT_WEIGHT` 0.8→0.3 | [[s08]] | +0.00487 | running |

## Board history (earlier work)
0.947 — `MIN_TRACK_LEN` 6→9 · 0.947 EXP-19 · 0.947 verbatim repro ·
0.946 ppgrid · 0.946 EXP-33 · 0.932 learned-bonus 2.0 · 0.932 DeepCenter veto ·
0.931 public bridge · 0.928 p24 · 0.925 p9 · 0.925 p22 · 0.924 / 0.922 detection
threshold · 0.914 p27 · 0.912 P3 · **0.906 `SAFE_DIVISIONS=0`** · 0.496 untrained.

The 0.906 row is the useful one: switching divisions off costs 0.041, which sizes
the division axis.

Related: [[Axis Priors]], [[One Change Per Submission]]
""")

note("s01", ["submission", "failed"], """
# s01 — `SAFE_DIV_DIVERGE_UM` 2.25 → 0

**Rejected by Kaggle's in-kernel validator before it could reach the board.**

Reasoning was recall-based: the divergence gate rejects 32 true divisions, so
open it. What actually happened:

| | divJ | TP/FP/FN |
|---|---|---|
| unmodified | 0.2308 | 3/1/9 |
| s01 | **0.1111** | 3/15/9 |

False positives 1→15, true positives unchanged. The gate was not blocking true
divisions; it was the only thing holding back the flood.

**Why it matters beyond itself:** this is the event that proved the in-kernel
validator is a trustworthy instrument, which is why [[Proxy Score]] is treated as
a decision signal at all.

Related: [[Safe Division]], [[Failed Loosening Tau Diverge]]
""")

note("s05", ["submission"], """
# s05 — remove [[Motion Relink]]

One **added** line: `os.environ['BIOHUB_OUTPUT_MOTION_RELINK'] = '0'`.
4198 → 4199 source lines.

| | proxy | adj_edge | divJ | TP/FP/FN |
|---|---|---|---|---|
| unmodified 0.947 | 0.9491 | 0.9260 | 0.2308 | 3/1/9 |
| **s05** | **0.9715** | **0.9407** | **0.3077** | **4/1/8** |

**+0.0224** on Kaggle's own validator. The largest single move available.

Submitted 2026-09-19 12:25 UTC. **Still PENDING** — the board can take 8 hours,
which is why work did not wait on it.

Live kernel slug is `biohub-s05-no-motion-relink`, *not* the `id` in its
metadata — see [[Kaggle Slug From Title]].

Its outputs are the evidence base for [[In-Kernel Sweep Is Inert]] and
[[Harness Validated Against Kernel]].

Parent of [[s08]].
""")

note("s06", ["submission", "failed"], """
# s06 — `OUTPUT_LINEFIT_WEIGHT` 0.8 → 0.4 on the base

**Measured −0.00135. Not pushed.**

Predicted +0.0024 from raw-graph measurements. The sign flipped on the relinked
pipeline because [[Linefit Smoothing]] fits along unique predecessor/successor
chains and [[Motion Relink]] changes that topology.

**This is the origin of the [[Transfer Lesson]].** The error was asserting that
two changes were independent when they were not.

The same knob on the no-relink chain is **+0.00372** at w=0.4 and **+0.00487** at
w=0.3 — see [[s09]]. Same knob, opposite sign, different base.

Related: [[Linefit Surface On s09]]
""")

note("s07", ["submission", "superseded"], """
# s07 — gap2 after safe_div on the **unmodified base**

**Built, AST-proven, and deliberately never pushed.** Superseded by [[s08]].

The reorder itself is correct — see [[Gap2 In The Wrong Position]]. The base was
wrong. It applies the change to the notebook with [[Motion Relink]] still **on**,
but relink is what manufactures the orphan pool that gap2 and [[Safe Division]]
compete over. By the [[Transfer Lesson]], a delta measured with relink off need
not survive with it on.

Still buildable for comparison: `python scripts/96_reorder_variant.py s07`, which
reproduces its committed notebook byte-for-byte.

Related: [[Script 96 Reorder Variant]]
""")

note("s08", ["submission", "inflight"], """
# s08 — gap2 after safe_div, on [[s05]]

The rebuild [[s07]] should have been. **Pushed 2026-09-19 ~13:03 UTC**,
`biohub-s08-reorder-on-s05`.

Measured on the **full** s05 chain before building — the point of the
[[Transfer Lesson]]:

| | proxy | J | divJ | TP/FP/FN |
|---|---|---|---|---|
| A gap2 **before** safe_div (= s05) | 0.96682 | 0.93483 | 0.2857 | 4/2/8 |
| **B gap2 after (= s08)** | **0.97431** | 0.93516 | **0.3571** | **5/2/7** |
| C gap2 **off entirely** | 0.97297 | 0.93333 | 0.3571 | 5/2/7 |

**B − A = +0.00748.** The three-stage topology predicted +0.00922; the full chain
gave 19% less but **held its sign**.

Three things to keep straight:
- It **is one division** — see [[Division Recovery in 44b6_341df25f]]. Five of
  eight films move by exactly 0.00000.
- But it is **invariant**: positive in **9/9** perturbations of the surrounding
  stages, spread +0.00748 to +0.00750. That is the signature of a discrete
  contest over one orphan, not a geometric fit.
- **Row C is the surprise**: gap2 in its *deployed* position is worse than gap2
  switched off. It only earns its place after safe division.

Gain is on the **division axis** — see [[Axis Priors]].

⚠ **But it is worth +0.00000 on the films that are actually scored.** Their
division ledger is 0/6/3, so `divJ = 0` in every arm and there is no division to
recover. See [[Scored Films Measurement]]. Harmless, but not a gain.

**Kernel result, 2026-09-19:** in-kernel validator **0.9795** against [[s05]]'s
0.9715 — **+0.00803**, against a harness prediction of +0.00748 recorded before
the run. See [[Harness Prediction Confirmed]].

Proven by [[Script 96 Reorder Variant]], measured by [[Script 98 Reorder On Norelink]].
""")

note("s09", ["submission", "failed"], """
# s09 — CANCELLED

> **Cancelled 2026-09-19 mid-run, and superseded by [[s10]].** Kept because the
> mistake is the most instructive one in the project: it shipped on one film set.

# s09 — `OUTPUT_LINEFIT_WEIGHT` 0.8 → 0.3, on [[s08]]

**Pushed 2026-09-19 ~13:10 UTC**, `biohub-s09-linefit03-on-s08`. Local **+0.00487**.

The follow-on the handoff called for, and it lands as predicted: [[s06]] measured
this knob at −0.00135 on the relinked pipeline; here it is positive. See
[[Linefit Surface On s09]] for the full grid.

**w=0.3 was shipped, not the argmax.** The argmax is w=0.4/window=4 at +0.00620,
but that moves *two* variables onto a single cell the known ~0.002 wobble says is
untrustworthy. w=0.3 is the most **window-stable** weight — spread 0.00064 across
windows 2/3/4, against 0.00248 for w=0.4 — and leaves window at its deployed 2.

**Caveat, and a real one.** The gain is high-variance and on the wrong axis:
only 3/8 films improve at the argmax, two lose ~0.006, and the division ledger is
**unchanged at 5/2/7 in all 24 cells**. A pure edge-axis lever, where
[[Axis Priors]] is 0-for-3. Good local evidence, bad prior.

⚠ **On the films that are actually scored, s09 is −0.00086 — a sign flip.** It
helps one film and hurts the heavier one. See [[Scored Films Measurement]].
**Do not select s09.** [[s05]] alone is better where it counts.

This is exactly the failure the [[Manifest Contamination]] post predicted for
coordinate-localisation changes, and it converges with the two warnings already
on record: edge axis, and 3/8 films.

Built by [[make_env_variant]].
""")

# ─────────────────────────────── BIG IDEAS ───────────────────────────────
note("Transfer Lesson", ["concept", "key"], """
# The Transfer Lesson

> **A price measured on one topology does not hold on another.**

Learned expensively via [[s06]], and now confirmed **three independent times**:

| knob | on one base | on another | 
|---|---|---|
| [[Linefit Smoothing]] w=0.4 | **+0.0024** raw graphs | **−0.00135** relinked ([[s06]]) |
| [[Gap Closing]] `GAP_CLOSE_UM` 5→8 | **+0.00089** raw graphs | **−0.00015** [[s09]] chain |
| [[Gap2 Recovery]] reorder | +0.00922 three-stage | **+0.00748** full chain (sign held) |

The third is the instructive one: the price *shrank* 19% but kept its sign,
because the mechanism is discrete (a contest over one orphan node) rather than
continuous (a geometric fit whose neighbours can move it).

**Operational consequence:** re-price, never inherit. [[Script 100 Reprice On s09]]
and [[Script 101 Safediv Gates]] exist entirely because of this rule, and both
came back empty — see [[Parametric Search Closed]].

The corollary is why no [[Upstream Knobs]] variant has been pushed: changing the
edge scorer moves the ILP graph, which is the topology [[s08]] and [[s09]] were
measured on.
""")

note("Ordering Bug Class", ["concept", "key"], """
# Ordering Bug Class

> **The recurring defect in this pipeline is sequence, not parameters.**

Three instances found:

1. **[[Motion Relink]] wipes the [[ILP Linker]]'s forks.** Fixed by deleting the
   stage — [[s05]], +0.0224.
2. **[[Gap2 Recovery]] before [[Safe Division]] eats the orphan it needs.**
   5/2/7 → 4/3/8. Fixed by reordering — [[s08]], +0.00748.
3. **[[Linefit Smoothing]] before [[Safe Division]]** collapses divisions to
   3/1/9 or 2/3/10. Already correct in the deployed notebook.

The unifying rule: **divisions must settle before anything perturbs geometry or
consumes orphans.**

A fourth candidate was checked and rejected on evidence: moving [[Gap Closing]]
after safe division is worth ~0.00001. Measured, not assumed.

Related: [[Safe Division]], [[Parametric Search Closed]]
""")

note("Node Count Exploit", ["concept", "key"], """
# Node Count Exploit

The [[Node Count Multiplier]] is **uncapped above 1**, so deleting nodes pays
without bound. Any stage that removes nodes can raise the score while making
tracking *worse*.

**Signature:** proxy rises, `J_edge` flat or falling, `ratio` moves.

Known offender: [[Short Track Filter]], whose multiplier gain is monotone to L=40.
A real stage would peak.

Every screening script now flags it automatically rather than leaving it to be
noticed — on the [[s09]] chain all three positive rows were exploits:
`st_len=8`, `st_len=9`, `gc_um=4.0`.

**Two subtleties:**
- The honest invariant is `ratio = n_pred/n_est`, **not** aggregate `mult` — see
  [[Aggregate Mult Re-weighting]].
- [[Operating Rules]] declares the node-count term forbidden ground for *tuning*.
  Measuring it is fine. See [[Node Budget]].
""")

note("Parametric Search Closed", ["finding", "key"], """
# The Parametric Search — Closed On One Set, Reopened On Another

⚠ **This note previously said "nothing survives, do not re-tune". That was
wrong, and the reason is instructive.**

The sweeps that produced it — [[Script 100 Reprice On s09]] (21 settings) and
[[Script 101 Safediv Gates]] (7 gates) — both ran on the **VALIDATOR** films. See
[[Evidence Tiers]]. Re-running the same 35 configs on the **SCORED** films
(`scripts/106_reprice_on_scored.py`, 19 s on 16 cores) found a candidate the
validator films structurally could not see: `OUTPUT_LINEFIT_WEIGHT` 0.6,
**+0.00471 with `ratio` exactly unchanged**. That became [[s10]].

**What is actually closed:**
- Every knob, on the **validator** films — nothing survives there.
- Every knob, on the **scored** films — except linefit, now shipped as [[s10]].
- `OUTPUT_MIN_TRACK_LEN`, killed by the [[Both-Sets Rule]] — see [[MIN_TRACK_LEN Is Dead]].

**What this episode teaches**, beyond the knobs: a negative result inherits the
scope of the set it was measured on. "Nothing left to tune" meant "nothing left
to tune *on the 8 validator films*", and the wider claim was never licensed.

Two durable facts still stand:
- A **third** confirmation of the [[Transfer Lesson]] (`GAP_CLOSE_UM` 5→8:
  +0.00089 raw, −0.00015 on the s09 chain).
- [[Inert Variables]] re-confirmed after three topology changes.
- **[[Safe Division]] earns its keep on EDGES** even with zero division credit on
  the scored films: switching it off takes the ledger 0/6/3 → 0/0/3 and still
  costs **−0.00179**.

What remains is [[Upstream Knobs]], untestable locally, and better measurement —
see [[s11]].
""")

note("Gap2 In The Wrong Position", ["finding"], """
# Gap2 Is In The Wrong Position

The finding behind [[s07]] and [[s08]].

[[Gap2 Recovery]] joins a track end at *t* to a start at *t+3*. That start is an
**orphan** — no incoming edge — which is precisely the pool
[[Safe Division]] draws its second daughters from
(`candidate_ids = [... if node_id not in incoming]`).

Run gap2 first and it eats a daughter: one true division becomes a false positive
plus a false negative. Run safe division first and gap2 still collects its edge
gain, because the daughter it gives up is one node out of ~750.

**Measured on raw + safe_div:**
| | proxy | divJ | TP/FP/FN |
|---|---|---|---|
| anchor | 0.96964 | 0.3571 | 5/2/7 |
| + gap2 **before** | 0.96163 (−0.00802) | 0.2667 | 4/3/8 |
| + safe_div **then** gap2 | 0.97085 (+0.00121) | 0.3571 | 5/2/7 |

Proven behaviourally on the notebook's *own* stage functions: a contested orphan
goes to safe division under the new order, and uncontested gap2 pairs are
bit-identical in both orders — including synthetic node ids, because safe
division adds **edges only, never nodes**.

Related: [[Ordering Bug Class]], [[Script 96 Reorder Variant]]
""")

note("Motion Relink Is The Large Lever", ["finding"], """
# Motion Relink Is The Large Lever

See [[Motion Relink]] for the mechanism. The summary:

- Removing it is worth **+0.0224** on Kaggle's validator, **+0.0295** predicted locally.
- It is *systematic*, not a few unlucky films: break rate jumps **171×** across
  5,540 GT edges with a clean knee at the 6 µm gate.
- An **oracle** per-film adaptive skip beats plain deletion by only **0.0014** —
  so there is no cleverer version worth building.

This is the single biggest number in the project and the basis of [[s05]],
[[s08]] and [[s09]], all of which sit on the no-relink base.

**Unresolved:** it has not yet been confirmed on the board. [[s05]] is still
pending, and [[Axis Priors]] says edge-axis changes are 0-for-3.
""")

note("Divisions Played Out", ["finding"], """
# Divisions Are Played Out

Of the **12** ground-truth divisions on the [[Validator Films]], the chain now
recovers **5** (up from 3 deployed).

Autopsy of the remaining 7:
| cause | count | addressable? |
|---|---|---|
| daughter held by another parent | 4 | 1 — three are *genuinely* nearer a neighbour |
| detection / matching failure | 3 | no — [[Frozen Weights]] |
| blocked by a gate | 1 | missed by 0.35 µm |

**Realistically addressable: ~2.**

And the gate one may already be gone: loosening `parent_max` by 33% on the
[[s09]] chain catches **nothing**, which suggests [[s08]]'s reorder took it. See
[[Safe Div Gates At Local Optimum]].

The contested-daughter route was tried and failed — see [[Failed Fork Before Prune]].

Related: [[Division Jaccard]], [[Safe Division]]
""")

note("Safe Div Gates At Local Optimum", ["finding"], """
# Safe Div Gates Are At A Local Optimum

Confirmed twice: originally on clean graphs, and again on the [[s09]] chain by
[[Script 101 Safediv Gates]] — which was **not** redundant, because safe
division's *input* changed when [[s08]] moved [[Gap2 Recovery]] behind it.

No gate in any direction gains a division. The ledger holds at 5/2/7.

| gate | swept | result |
|---|---|---|
| `parent_max` | 8.5 → 12.0 | nothing, even at +33% |
| `sister_max` | 13 → 16 | inert |
| `child_max` | 9 → 12 | inert |
| `frame_cap`, `glob_cap` | ×2 | inert |
| `tau` | 0.5 → 1.0 | **sharp optimum at 0.6** — loses a division in *both* directions |
| `diverge` | 1.5 → 3.0 | 1.75 and 1.5 gain a TP but FP goes 2→5 and 2→8, divJ **falls** |

The `diverge` rows reproduce [[Failed Loosening Tau Diverge]] exactly: TP rises,
divJ falls, because FP rises faster.

Related: [[Parametric Search Closed]], [[Divisions Played Out]]
""")

note("Inert Variables", ["finding"], """
# Inert Variables

72 post-processing variables exist; **38 have never been set by anyone** in the
fork lineage. All were priced against real ground truth and **none beats one
division event** ([[One Division Event Floor]]).

**Six are fully inert** — they return exactly 0.00000:
`g2_frac`, `g2_frame`, `g2_step`, `gc_frac`, `gc_reuse`, `sd_req_nn`

Re-confirmed on the [[s09]] chain after three topology changes, along with
`prune=False` and `st_forks=False` ([[Prune Isolated]] has nothing to remove).

**One never-touched boolean is load-bearing**, and in the dangerous direction:
`SAFE_DIV_REQUIRE_DIVERGENCE=0` costs **−0.025**, division FP 2→20.

Related: [[Parametric Search Closed]], [[Safe Division]]
""")

note("Node Budget", ["finding"], """
# Node Budget

5,429,739 detections against 4,725,117 estimated — **+14.9%**, costing ~0.0124
through the [[Node Count Multiplier]].

The split is the interesting part: **44b6 +1.1%, 6bba +32.1%**. The entire
penalty is a 6bba problem — and 6bba holds 125 of the 151 divisions.

Cheapest route to fix it would be raising `BIOHUB_DET_THRESHOLD` 0.965→0.995,
about **33× cheaper per node removed** than any alternative.

**But it is forbidden ground.** [[Operating Rules]] permits measuring the
node-count term and forbids *tuning* it, because that is the [[Node Count Exploit]].

Note the [[Validator Films]] are not representative here: on those 8 films
`ratio` is **0.8977** (under-estimate, multiplier above 1), while the test films
run over. The derivative is the same sign, so conclusions hold, but the level
does not transfer.

Related: [[Temporal UNet3D Detector]], [[Upstream Knobs]]
""")

note("Learned Bonus Is Crippled", ["finding"], """
# `MOTION_RELINK_LEARNED_BONUS` Is Structurally Crippled

Every `solution` flag in the prediction `.geff` is True, so the file stores
**only ILP-selected edges**. Therefore `prob.get(pair, 0.0)` returns 0 for every
alternative edge.

The `−β·prob` term is not a likelihood at all — it is a flat ~0.9 µm
**incumbency discount** applied to whichever edge the [[ILP Linker]] already chose.

A board submission moved this knob 1.0→2.0 and scored 0.932.

Moot in the current lineage anyway, since [[Motion Relink]] is off from [[s05]]
onward.

Related: [[Node Transformer Edge Scorer]]
""")

note("Division Recovery in 44b6_341df25f", ["finding"], """
# The Division [[s08]] Recovers

The entire +0.00748 of [[s08]] is **one division event in one film**.

`44b6_341df25f` goes **0/0/1 → 1/0/0** (TP/FP/FN) when [[Gap2 Recovery]] moves
behind [[Safe Division]].

**Independently corroborated by Kaggle**, not just the harness:
`artifacts/s05_output/validator_results.csv` shows this film with `div_tp=0,
div_fn=1` under [[s05]] as deployed — exactly the division predicted to be
recoverable. That file is the kernel's own validator output, not ours.

Five of the eight [[Validator Films]] move by exactly 0.00000.

This is why [[s08]] is described as sitting **at** the
[[One Division Event Floor]] rather than above it — real and discrete, but n=1.

Related: [[Harness Validated Against Kernel]]
""")

note("Linefit Surface On s09", ["finding"], """
# The Linefit (weight, window) Surface

Measured by [[Script 99 Linefit On s08]] on the full [[s08]] chain. Deployed
w=0.8/window=2 scores 0.97431; deltas against it:

| w \\\\ window | 2 | 3 | 4 |
|---|---|---|---|
| 0.2 | +0.00440 | +0.00278 | +0.00278 |
| **0.3** | **+0.00487** | +0.00505 | +0.00441 |
| 0.4 | +0.00372 | +0.00570 | **+0.00620** |
| 0.5 | +0.00291 | +0.00570 | +0.00166 |
| 0.6 | +0.00338 | +0.00261 | +0.00278 |
| 0.8 *(deployed)* | 0 | +0.00032 | −0.00241 |
| 1.0 | −0.00354 | −0.00821 | −0.00981 |

**w=0.3 is the robust pick**, not the argmax — its spread across windows is
**0.00064**, against 0.00248 for w=0.4 and 0.00404 for w=0.5.

Two invariants confirmed across all 24 cells:
- `ratio` **identical** at 0.8977 — linefit moves coordinates only.
- Division ledger **unchanged** at 5/2/7 — a pure edge-axis lever.

Aggregate `mult` drifts up to 8.85e-05, which is *not* a node-count change — see
[[Aggregate Mult Re-weighting]].

**Superseded as a shipping decision by [[Scored Films Measurement]]:** w=0.3 is
−0.00086 where it counts. Three film sets agree the plateau is **0.4–0.6** —
scored peaks at 0.6, validator at 0.4, the 199-film rebuild at 0.5 — and all
three put 0.3 outside it. s10 ships **0.6**.

⚠ The gain is concentrated: only 73/199 films move at all, median delta 0.00000.

Related: [[s09]], [[Linefit Smoothing]], [[Transfer Lesson]]
""")

# ─────────────────────────────── FAILED ───────────────────────────────
note("Failed Attempts", ["moc"], """
# Failed Attempts — do not repeat

Each has its own note with the reason it failed, because the reason is usually
more useful than the failure.

**Learning approaches**
- [[Failed Division Classifier]] — AUC 0.456, chance
- [[Failed Anaphase Hypothesis]] — chance at t−1 and t−2
- [[Failed Synthetic Division Data]] — AP 0.98 offline, board 0.910→0.906
- [[Failed Node Count Predictor]] — transfers 1-for-2 across embryos

**Gate and threshold tuning**
- [[s01]] — `DIVERGE_UM=0`, FP 1→15
- [[Failed Loosening Tau Diverge]] — TP up, divJ down
- [[Failed DeepCenter Veto]] — real ranker, already at its oracle ceiling

**Structural**
- [[Failed Fork Before Prune]] — contest fires on 2 of 6,063 proposals
- [[Failed Division Ceiling]] — the premise was arithmetically wrong
- [[Failed Short Track Exploit]] — the metric, not the tracking
- [[s06]] — right knob, wrong base

**Search**
- [[Parametric Search Closed]] — 28 settings re-priced, nothing survives
""")

note("Failed Division Classifier", ["failed"], """
# Failed: learned division classifier

Trained on frozen [[Temporal UNet3D Detector]] features to predict whether a node
is about to divide.

**AUC 0.456 at the split frame — worse than chance.** Nothing tried clears 0.70.

The features simply do not carry the signal. This is the strongest evidence that
[[Divisions Played Out]] is a *detection* limit, not a ranking one.

Related: [[Failed Anaphase Hypothesis]]
""")

note("Failed Anaphase Hypothesis", ["failed"], """
# Failed: the anaphase hypothesis

Hypothesis: a dividing cell shows morphological signal *before* the split, so
look one or two frames earlier.

**t−1: AUC 0.510. t−2: 0.484.** Both chance.

Tested properly and not supported. Recorded because it is an attractive idea that
will occur to anyone who reads [[Failed Division Classifier]] and assumes the
timing was the problem.
""")

note("Failed Loosening Tau Diverge", ["failed"], """
# Failed: loosening `tau` / `diverge` on recall evidence

The trap that [[s01]] fell into, in general form. Loosening [[Safe Division]]
gates raises true positives and raises false positives **faster**, so
[[Division Jaccard]] falls.

Original measurement: TP up, FP ×5, divJ **fell** 0.067 → 0.027.

Reproduced exactly on the [[s09]] chain by [[Script 101 Safediv Gates]]:
- `diverge=1.75` → 6 TP but FP 2→5, divJ −0.0042
- `diverge=1.50` → 6 TP but FP 2→8, divJ −0.0571

**The screening rule that comes from this:** a gate change only counts if
**divJ rises**. Never TP.

Related: [[Safe Div Gates At Local Optimum]]
""")

note("Failed Fork Before Prune", ["failed"], """
# Failed: fork-before-prune ordering and contested targets

An attempt to resolve the "daughter held by another parent" cases in
[[Divisions Played Out]] by reordering forking against pruning, and by handling
contested targets explicitly.

**Neutral to worse.** The contest fires on **2 of 6,063 proposals** — the
situation is simply too rare to pay for.

Worth knowing because it is the obvious next idea after [[s08]] succeeds by
resolving a *different* orphan contest. That one worked because it was
systematic; this one is not.
""")

note("Failed DeepCenter Veto", ["failed"], """
# Failed: [[DeepCenter]] as a division veto

Unusually, this failed *despite working*.

As a ranker it is genuine — **AUC 0.836–0.857**. But it is already **at its oracle
ceiling**: even a perfect threshold on it is worth only **+0.0029**, and a plain
brightness threshold matches its performance. The whole gain is one fork in one
film.

The lesson is about ceilings rather than accuracy: a good classifier applied to a
problem with almost no headroom returns almost nothing.

Related: [[Divisions Played Out]]
""")

note("Failed Division Ceiling", ["failed"], """
# Failed: the division detection ceiling

**Refuted — the premise was arithmetically wrong.**

Hypothesis: the detector's pooling kernel merges sister cells that are too close,
imposing a hard floor on division recall.

What is actually true: `pool_kernel_um` **quantises**. Values 3.0, 4.0 and 5.0 all
produce the identical (3,3,3) kernel, and the true floor is **3.25 µm Chebyshev**.
Only **1 of 151** divisions sits below it, and **zero** are actually lost to it.

The instance that motivated the whole investigation turned out to be a *linker*
failure with all three nodes correctly detected.

A good reminder to check the arithmetic of a mechanism before building on it.
""")

note("Failed Short Track Exploit", ["failed"], """
# Failed: short-track filter L≥6 as a "gain"

Raising `OUTPUT_MIN_TRACK_LEN` raises proxy. It is **not** a tracking improvement
— it is the [[Node Count Exploit]].

The diagnostic that settles it: the multiplier gain is **monotone all the way to
L=40**, where the graph is obliterated. A real stage would peak at some sensible
track length and then degrade.

Board evidence agrees: the `MIN_TRACK_LEN` 6→9 submission scored **0.947**, the
same as base.

See [[Short Track Filter]] for the numbers on the current chain.
""")

note("Failed Node Count Predictor", ["failed"], """
# Failed: learned `n_est` / node-count predictor

An attempt to predict the node-count estimate that drives the
[[Node Count Multiplier]].

**Transfers 1-for-2 across embryos**: R² 0.937 on one, **0.494** on the other,
with a median **17% underestimate** — and underestimating is the *dangerous*
direction, because it inflates the apparent multiplier.

Related: [[Node Budget]]
""")

note("Failed Synthetic Division Data", ["failed"], """
# Failed: synthetic division training data

Someone else's published result, recorded so it is not re-attempted.

Synthetic data gave **AP 0.98 on held-out synthetic** divisions. On the board:
**0.910 → 0.906**.

A textbook case of a held-out set that shares the generator's biases measuring
nothing about the real distribution.

Related: [[Failed Division Classifier]]
""")

# ─────────────────────────────── INFRASTRUCTURE ───────────────────────────────
note("Local Harness", ["infra", "key"], """
# Local Harness

**The single most valuable asset in the project.**

The Kaggle kernel keeps its prediction `.geff` files, and **8 of them are TRAIN
films** — so we hold *deployed-quality graphs with ground truth*. Post-processing
experiments run in **seconds** instead of ~1.75 h.

```
artifacts/s01_output/tracking_repo/predictions/unknown/unet_transformer_val/split_0/*.geff
```

**Validated three times against the board or the kernel:** it reproduced the
[[s01]] failure direction, it reproduces the deployed division ledger exactly
(3/1/9), and it agrees with Kaggle's own validator film-by-film on 7 of 8 — see
[[Harness Validated Against Kernel]].

Use, in this order: [[metric2]], [[Script 24 Keep ILP Edges]],
[[Script 91 Other Stages]], `scripts/25_relink_control.py`.

**Its hard limit:** it starts from the [[ILP Linker]] *output*, so it cannot test
[[Upstream Knobs]] at all.

⚠ Do not use the wrong modules — see [[Wrong Metric Module]].
""")

note("Harness Prediction Confirmed", ["finding", "key"], """
# The Harness Predicted s08 Before The Run, And Was Right

The first closed prediction loop: harness number recorded in the [[Ledger]]
**before** the kernel ran, then compared.

| | kernel (8 films, full deployed chain) | harness (same films, our ports) |
|---|---|---|
| divisions | 4/1/8 → **5/1/7** | 4/2/8 → **5/2/7** |
| divJ | 0.3077 → 0.3846 | 0.2857 → 0.3571 |
| edge term | **+0.00035** | **+0.00034** |
| proxy delta | **+0.00803** | **+0.00748** |

Absolute in-kernel validator proxy: [[s05]] **0.9715**, [[s08]] **0.9795**,
[[s10]] **0.9736**.

**A second loop closed, with a different answer.** [[s10]] was predicted at
+0.00338 and measured **+0.00207** — sign right, magnitude over-predicted by 39%.
The mechanism was still confirmed exactly: divJ unchanged at 4/1/8 (pure edge
axis, as predicted), `edges_fragmented` 111 → 108 and `edges_recovered`
5565 → 5568, so the gain is precisely **3 recovered fragmented edges**, and
`edges_lost_to_detection` held at 75 as a coordinate-only stage requires.

**Why s08 was near-exact and s10 was not:** s08's mechanism is **discrete** — one
orphan, one division — and a missing stage cannot blur it. s10's is
**continuous**: [[Linefit Smoothing]] fits along chains, and the harness omits
[[Single Parent Repair]], the short-track rescue and the [[DeepCenter]] vetoes,
all of which change those chains. The [[Transfer Lesson]] operating one level up,
between harness and kernel.

**Rule of thumb earned here:** trust harness magnitudes for discrete mechanisms,
trust only their sign for continuous ones.

**Mechanism confirmed exactly:** one division recovered, which is what
[[Division Recovery in 44b6_341df25f]] predicted.

**The entire 0.00055 gap is the denominator.** The harness carries one extra
division false positive — the un-ported [[DeepCenter]] safe-div veto, documented
*before* this run — so the division term is 0.1×(1/13) = +0.00769 for the kernel
and 0.1×(1/14) = +0.00714 for the harness. Nothing else differs.

⚠ **This validates the INSTRUMENT on the VALIDATOR tier. It does not make [[s08]]
worth anything on the board** — [[Scored Films Measurement]] puts it at +0.00000
there, because the scored films have no recoverable division. Both are true: the
harness is accurate, and the validator films are not the target. That distinction
is the whole point of [[Evidence Tiers]].

Footnote: s08's sweep kept `base` although `dcgap035` led by 0.0000147 —
`PPSWEEP_SELECT_MARGIN` is 0.001, so the guard held. See [[In-Kernel Sweep Is Inert]].
""")

note("Harness Validated Against Kernel", ["finding"], """
# Harness Validated Against The Kernel

On the **no-relink topology specifically** — which matters, because that is the
base [[s05]], [[s08]] and [[s09]] all sit on.

Source: `artifacts/s05_output/validator_results.csv`, Kaggle's own in-kernel
validator output, per film.

| film | kernel | harness |
|---|---|---|
| 44b6_12dfb391 | 0/0/1 | 0/0/1 |
| 44b6_267148e4 | 1/0/0 | 1/0/0 |
| 44b6_2a2eff9f | 1/1/0 | 1/**2**/0 |
| 44b6_341df25f | 0/0/1 | 0/0/1 |
| 6bba_062c8d37 | 1/0/0 | 1/0/0 |
| 6bba_07e24132 | 0/0/2 | 0/0/2 |
| 6bba_085bf656 | 0/0/1 | 0/0/1 |
| 6bba_09961292 | 1/0/3 | 1/0/3 |
| **total** | **4/1/8** | **4/2/8** |

**Agreement on 7 of 8**, one extra false positive — attributable to the
un-ported [[DeepCenter]] safe-division veto.

The kernel total 4/1/8 matches the figure recorded for [[s05]] independently.

Related: [[Local Harness]], [[Division Recovery in 44b6_341df25f]]
""")

note("metric2", ["infra"], """
# `src/biohub/metric2.py`

**THE** metric implementation for the [[Local Harness]] graphs.

`SCALE = (1.625, 0.40625, 0.40625)` — **anisotropic**, original-voxel coordinates.

Two implementation details that have bitten:
- `aggregate()` averages with `weight = tp + fp + fn` (line 91), which depends on
  the *predictions*. See [[Aggregate Mult Re-weighting]].
- `score()` returns `J_edge`, `multiplier`, `adj`, `n_pred`, `n_est` and the
  division ledger separately, which is what makes [[Operating Rules]]'
  report-them-separately rule enforceable.

⚠ [[Wrong Metric Module]] — `metric.py` and `postprocess.py` are 4× wrong here.
""")

note("Ledger", ["infra"], """
# Ledger — `artifacts/ledger/runs.db`

SQLite. Tables: `runs`, `scores`, `division_events`, `submissions`.

**Predictions are recorded before results**, so sign agreement between offline
and board stays honest rather than reconstructed afterwards.

Every row for [[s08]] and [[s09]] carries an explicit **falsifier** — for s08,
"if [[s05]] does not beat 0.947 this build is void, because its base is void".

Related: [[Submission Ledger]], [[Operating Rules]]
""")

# ─────────────────────────────── SCRIPTS ───────────────────────────────
note("Scripts", ["moc"], """
# Scripts

**Harness core**
- [[Script 24 Keep ILP Edges]] — loaders and `safe_div`
- [[Script 91 Other Stages]] — ports of every post-processing stage
- [[metric2]] — the metric

**Builders** (each proves its own output)
- [[Script 96 Reorder Variant]] — the statement reorder, 7 proofs
- [[make_variant]] — substitute an existing env line
- [[make_env_variant]] — add or substitute on any parent, with [[Drift Guard]] rewrite

**Experiments on the current chain**
- [[Script 98 Reorder On Norelink]] — the [[s08]] measurement
- [[Script 99 Linefit On s08]] — the [[s09]] measurement
- [[Script 100 Reprice On s09]] — 21 settings
- [[Script 101 Safediv Gates]] — 7 gates

A convention worth keeping: every experiment script prints `J` and the
multiplier separately and flags the [[Node Count Exploit]] itself, so the guard
is executable rather than remembered.
""")

note("Script 24 Keep ILP Edges", ["infra"], """
# `scripts/24_keep_ilp_edges.py`

Supplies `load_pred()`, `load_gt()` and **`safe_div()`** — the deployed
[[Safe Division]] rule reimplemented on original-voxel coordinates.

`safe_div` is fully parameterised (`parent_max`, `sister_max`, `child_max`,
`tau`, `diverge`, `frame_cap`, `glob_cap`), which is what made
[[Script 101 Safediv Gates]] possible.

Other scripts import it by `exec`-ing the module prefix — a pattern with a
sharp edge, see [[Docstring Shadowing]].
""")

note("Script 91 Other Stages", ["infra"], """
# `scripts/91_other_stages.py`

Ports of [[Gap Closing]], [[Gap2 Recovery]], [[Short Track Filter]],
[[Linefit Smoothing]] and [[Prune Isolated]] onto original-voxel coordinates
with the anisotropic `SCALE`.

Also supplies the `run()` / `show()` reporting harness that every later
experiment reuses, including the `<<FALSE GAIN: all multiplier` flag.

**That flag has a known false alarm**: it only inspects `J_edge` and cannot see
`divJ`, so a row whose gain is a *division* gets labelled as an exploit. This
happens on row C of the [[s08]] measurement.

Measures each stage **one at a time** against a raw + safe_div anchor, with each
stage at its deployed position — which is why its numbers are a *three-stage*
topology and needed re-measuring on the full chain. See [[Transfer Lesson]].
""")

note("Script 96 Reorder Variant", ["infra"], """
# `scripts/96_reorder_variant.py`

Builds the [[Gap2 Recovery]] reorder and **proves** it seven ways. Targets:
`s07` (wrong base, kept for comparison) and `s08`.

1. **line diff** — exactly 4 lines, 2 statements moved
2. **AST** — 558 top-level statements, exactly 1 differs; within
   `filter_output_graph`, `sorted(old) == sorted(new)`, i.e. a strict permutation
3. **compile**
4. **[[Drift Guard]]** — the diff sets no `os.environ` value and touches no guarded key
5. **behavioural** — runs the notebook's *own* stage functions on two toy graphs,
   confirming a contested orphan goes to safe division and uncontested gap2
   pairs are bit-identical, synthetic node ids included
6. **env insertion** — the [[s05]] line adds exactly one top-level statement
7. **diff vs the shipped [[s05]] notebook** — exactly the two moved statements

Proof 7 is what makes "[[One Change Per Submission]]" *demonstrated* rather than
asserted.
""")

note("Script 98 Reorder On Norelink", ["infra"], """
# `scripts/98_reorder_on_norelink.py`

The measurement behind [[s08]]: the **full** [[s05]] chain at deployed
parameters, [[Gap2 Recovery]] on either side of [[Safe Division]], nothing else
differing.

Produced +0.00748 against the +0.00922 the three-stage topology predicted —
the [[Transfer Lesson]] applied *before* building rather than discovered after.

Includes a 9-case sensitivity sweep over the stages either side, all positive,
spread +0.00748 to +0.00750.

Also the site of [[Docstring Shadowing]], caught and fixed.
""")

note("Script 99 Linefit On s08", ["infra"], """
# `scripts/99_linefit_on_s08.py`

Sweeps [[Linefit Smoothing]] `(weight, window)` over the full [[s08]] chain —
24 cells. Produced [[Linefit Surface On s09]] and the choice behind [[s09]].

Asserts the right invariant: **`ratio`**, not `mult`. The first version asserted
on `mult` and reported FAIL; the assertion was wrong, not the chain. See
[[Aggregate Mult Re-weighting]].

Reports a **plateau** rather than an argmax, because the surface wobbles ~0.002
between neighbouring cells on only 8 films.
""")

note("Script 100 Reprice On s09", ["infra"], """
# `scripts/100_repice_on_s09.py`

Re-prices 21 settings across [[Gap Closing]], [[Gap2 Recovery]],
[[Prune Isolated]] and [[Short Track Filter]] on the [[s09]] chain.

**Result: nothing.** Every positive row is the [[Node Count Exploit]].

Two design choices worth copying:
- It computes the [[One Division Event Floor]] from the **live ledger**
  (`0.1/(tp+fp+fn)` = 0.00714) rather than using the 0.0083 figure, which assumed
  a different denominator.
- It prints `ratio` on every row and emits the verdict itself, so the exploit is
  *flagged* rather than left to be spotted.

Related: [[Parametric Search Closed]]
""")

note("Script 101 Safediv Gates", ["infra"], """
# `scripts/101_safediv_gates_on_s09.py`

Re-prices all seven [[Safe Division]] gates on the [[s09]] chain — the family
[[Script 100 Reprice On s09]] missed, and the one that matters most because
divisions are the good axis ([[Axis Priors]]).

Deliberately **fine steps** near the deployed values, because the one known
gate-blocked division misses by 0.35 µm and a coarse sweep would step over it.

Screening rule: a row only counts if **divJ rises** — not TP. That is what makes
[[Failed Loosening Tau Diverge]] label itself instead of looking like a win.

**Result: nothing.** See [[Safe Div Gates At Local Optimum]].
""")

note("make_variant", ["infra"], """
# `scripts/make_variant.py`

Builds one-change notebook variants by **substituting** an existing
`os.environ[...]` line. Refuses to write unless the changed-line count matches
what was requested.

Its limitation is why [[make_env_variant]] exists: most interesting knobs are
never set in the notebook at all, so changing one means *adding* a line.

Warns when a key is covered by the [[Drift Guard]].
""")

note("make_env_variant", ["infra"], """
# `scripts/make_env_variant.py`

Adds or substitutes **one** env line on **any parent notebook**, so variants
chain: [[s08]] is [[s05]] + a reorder, [[s09]] is [[s08]] + one line.

Auto-detects its mode:
- **ADD** — key unset in the parent (sits at its `os.environ.get` default)
- **SUBSTITUTE** — key already set, rewritten in place

When the key is under the [[Drift Guard]], it rewrites the guard entry **in the
same edit** and proves the result is exactly those two lines with no other
guarded key disturbed. That is what unlocks [[Upstream Knobs]].

Five proofs, including two that catch silent no-ops: the key must actually be
**read** somewhere, and the value must **differ** from the current one.

Also enforces `slug == slugify(title)` — see [[Kaggle Slug From Title]].
""")

# ─────────────────────────────── RULES & GOTCHAS ───────────────────────────────
note("Operating Rules", ["moc", "key"], """
# Operating Rules

`ABORT_RULES.md` is frozen and carries pre-registered gates, standing
prohibitions and a measurement-error table. The parts that matter:

- **[[One Change Per Submission]].** Bundling has already cost one
  uninterpretable sweep.
- **Report `J` and the multiplier separately, every time.** A gain arriving
  through the multiplier while `J` is flat is the [[Node Count Exploit]].
- **The node-count term is forbidden ground for *tuning*.** Measuring is fine —
  see [[Node Budget]].
- **[[One Division Event Floor]].** Do not report division differences finer
  than one event as meaningful.
- **[[Axis Priors]].** Edge-axis changes have a bad track record here.

Recorded in the [[Ledger]], predictions before results.
""")

note("One Change Per Submission", ["rule"], """
# One Change Per Submission

The discipline that keeps the [[Submission Ledger]] interpretable.

**The subtlety that matters:** "one change" means one change relative to **the
thing you are comparing against**, not relative to the 0.947 base. [[s09]] is
three changes from base but **one** from [[s08]], and that is what makes it a
legitimate probe.

This is *demonstrated*, not asserted: [[Script 96 Reorder Variant]] diffs the
built notebook against the shipped [[s05]] notebook, and [[make_env_variant]]
diffs against its stated parent.

A guard rewrite forced by a guarded key is bookkeeping, not a second change —
see [[Drift Guard]].
""")

note("Axis Priors", ["rule", "key"], """
# Axis Priors

The one published offline-vs-board ledger in this competition:

| axis | record |
|---|---|
| **division** | **3 for 3** |
| **edge** | **0 for 3** |

This is the single best prior available for deciding what to believe.

Applied to what is in flight:
- [[s08]]'s gain **is a division** ([[Division Recovery in 44b6_341df25f]]) →
  good axis.
- [[s09]]'s gain leaves the division ledger untouched at 5/2/7 in all 24 cells →
  **pure edge axis**, bad prior, despite better local evidence.
- [[s05]] is an edge-axis change with much better evidence than the three that
  failed — and [[Scored Films Measurement]] now puts it at **+0.02707 on the
  scored films**, larger than on the validator and positive on all four.

**Update:** the axis prior may partly *be* a composition effect. Division gains
are measured where there are 12 divisions and scored where there are 3.

Related: [[Operating Rules]], [[Proxy Score]]
""")

note("Kaggle Mechanics", ["moc"], """
# Kaggle Mechanics

Practical constraints that shape what is possible in a day.

- **Maximum 2 concurrent GPU sessions.** A third push is refused outright.
- **A commit run is ~1.75 h**, of which a large share is the validator and its
  candidate sweep — which is **provably inert** once relink is off, see
  [[In-Kernel Sweep Is Inert]].
- **The board can take 8 hours** to return a score. The in-kernel validator
  ([[Proxy Score]]) returns in ~1.75 h and is the real decision signal.
- **Slugs come from the title**, not the metadata id — [[Kaggle Slug From Title]].
- Auth: token at `~/.kaggle/access_token`. Submit with
  `kaggle competitions submit -c ... -k <owner>/<slug> -v <n> -f submission.csv -m "..."`.
- Machine note: **GPU batching gives literally zero gain** on this hardware
  (50.4 ms/window at B=1 vs 51.5 ms at B=23). Measured. The real headroom is 18
  CPU cores; most scripts are single-threaded.
""")

note("Kaggle Slug From Title", ["gotcha"], """
# Gotcha: Kaggle slugs come from the TITLE

Kaggle derives the live kernel slug by **slugifying the title**, ignoring the
`id` field in `kernel-metadata.json`.

Evidence:
- [[s05]] shipped `"id": "aryaarun07/biohub-s05-no-relink"` and went live at
  **`biohub-s05-no-motion-relink`** = slugify("Biohub S05 no motion relink").
- [[s06]]'s log file is `biohub-s06-linefit-weight-0-4.log`, from
  "Biohub S06 linefit weight 0.4".

**Worse, the failure is misleading:** querying the metadata id returns
*"Permission 'kernels.get' was denied"*, which reads like a private-notebook or
auth problem, not a wrong name.

Both builders now **abort** unless `slug == slugify(title)`.

Related: [[make_env_variant]], [[Script 96 Reorder Variant]]
""")

note("In-Kernel Sweep Is Inert", ["gotcha", "finding"], """
# Gotcha: the in-kernel sweep is inert once relink is off

The notebook runs an 8-candidate post-processing sweep on held-out train films
before producing its submission. **Five of the eight candidates are
[[Motion Relink]] knobs**, which cannot do anything with the stage switched off.

From `artifacts/s05_output/ppsweep_results.csv` — identical to the last digit:

| candidate | proxy |
|---|---|
| `base` | 0.9715059814960677 |
| `tight55` | 0.9715059814960677 |
| `relaxed9` | 0.9715059814960677 |
| `bonus125` | 0.9715059814960677 |
| `gap2step40` | 0.9715059814960677 |
| `reuse28` | 0.9715059814960677 |

`ppsweep_selected.json` picks `base` with `{}` overrides. The two that *do* move
(`dcgap035`, `gap45`) differ by ~1e-5.

**Cost: ~8 × 250 s ≈ 33 minutes** of every relink-off run, for nothing.

This upgrades a previously "untested but low risk" runtime cut to **evidenced** —
for relink-off variants only. `gap2step40` and `reuse28` tying base also
re-confirms [[Inert Variables]].

Related: [[Kaggle Mechanics]]
""")

note("Aggregate Mult Re-weighting", ["gotcha"], """
# Gotcha: aggregate `mult` is not a node-count readout

[[metric2]]'s `aggregate()` averages per-film values with
**`weight = tp + fp + fn`** — the edge-confusion union, which depends on the
**predictions**.

So any stage that moves coordinates changes edge matching, which changes each
film's weight in the average, which drifts the aggregate multiplier by ~1e-5
**with node counts completely untouched**.

Measured: across 24 [[Linefit Smoothing]] cells, aggregate `mult` drifts up to
**8.85e-05** while `ratio` is **identical** at 0.8977.

**Consequence:** the [[Node Count Exploit]] shows up in **`ratio`**, not in a
small `mult` wobble. Assert on `ratio`.

This caused a real false alarm — [[Script 99 Linefit On s08]] initially reported
FAIL on a correct chain because the assertion was wrong.
""")

note("Wrong Metric Module", ["gotcha"], """
# Gotcha: the wrong metric module fails silently

`src/biohub/metric.py` and `src/biohub/postprocess.py` contain lookalikes of
[[metric2]] and the stage ports — but they assume the **downsampled isotropic
64³ grid** with `GRID_UM = 1.625`.

On the [[Local Harness]] `.geff` graphs, which are original-voxel and
**anisotropic** (`SCALE = (1.625, 0.40625, 0.40625)`), they are **4× wrong in y
and x**.

**They do not raise.** They return plausible numbers.

Both now carry warning comments. Use [[metric2]] and
[[Script 91 Other Stages]] for anything touching those graphs.
""")

note("Drift Guard", ["gotcha"], """
# The `_EXPECTED_NUMERIC` drift guard

A single dict in cell 2 asserting nine env values at import time. If a set value
disagrees with the guard, **the kernel aborts at cell 3** — roughly 20 minutes
into a 1.75 h run.

Guarded keys:
`DET_THRESHOLD` 0.965 · `ILP_APPEARANCE_WEIGHT` 0.0 ·
`ILP_DISAPPEARANCE_WEIGHT` 2 · `GAP_CLOSE_UM` 5.0 · `OUTPUT_MIN_TRACK_LEN` 6.0 ·
`SAFE_DIV_MAX_UM` 9.0 · `DEEPCENTER_SAFE_DIV_THRESHOLD` 0.20 ·
`BIDIRECTIONAL_EDGE_WEIGHT` 0.15 · `SECONDARY_EDGE_FEATURE_TTA_WEIGHT` 0.75

Note the overlap with [[Upstream Knobs]] — **five of the nine are upstream**,
which is why the guard was the blocker on that whole axis.

[[make_env_variant]] now rewrites the guard entry in the same edit and proves it.

Related: [[One Change Per Submission]]
""")

note("Docstring Shadowing", ["gotcha"], """
# Gotcha: `exec()` overwrites `__doc__`

Several scripts reuse [[Script 91 Other Stages]] by `exec`-ing its source prefix.
CPython compiles a source string in `"exec"` mode, and a **leading string literal
becomes the module docstring**, assigned into the target globals.

So `exec(other_module_source)` **silently replaces the calling module's
`__doc__`**.

Effect when it bit: [[Script 98 Reorder On Norelink]] printed *91's* docstring as
its log header, so a committed artifact described the wrong experiment. The
numbers were right; the label was not.

Fix: capture `_DOC = __doc__` **before** the `exec`.

A small bug, recorded because it produces a confidently-wrong artifact rather
than an error.
""")

note("Upstream Knobs", ["concept"], """
# Upstream Knobs

Everything before the [[ILP Linker]]'s output. The [[Local Harness]] **cannot
test any of it**, because it starts from that output.

| knob | deployed | stage |
|---|---|---|
| `DET_THRESHOLD` | 0.965 | [[Temporal UNet3D Detector]] |
| `ILP_APPEARANCE_WEIGHT` | 0.0 | [[ILP Linker]] |
| `ILP_DISAPPEARANCE_WEIGHT` | 2 | [[ILP Linker]] |
| `BIDIRECTIONAL_EDGE_WEIGHT` | 0.15 | [[Node Transformer Edge Scorer]] |
| `SECONDARY_EDGE_FEATURE_TTA_WEIGHT` | 0.75 | [[Node Transformer Edge Scorer]] |

All five sit under the [[Drift Guard]]. [[make_env_variant]] can now build them.

**Untested, and deliberately not yet pushed.** Changing the edge scorer moves the
ILP graph itself — the topology [[s08]] and [[s09]] were measured on — so by the
[[Transfer Lesson]] it would invalidate the post-processing gains and require
re-validating them on top. Each probe also costs a full kernel run with no local
preview.

This is the **only axis left** after [[Parametric Search Closed]].
""")

# ─────────────────────────────── DATA ───────────────────────────────
note("Validator Films", ["data"], """
# Validator Films

The 8 held-out **train** films used by both Kaggle's in-kernel validator and the
[[Local Harness]] — which is why the two are comparable.

`44b6_12dfb391` · `44b6_267148e4` · `44b6_2a2eff9f` · `44b6_341df25f` ·
`6bba_062c8d37` · `6bba_07e24132` · `6bba_085bf656` · `6bba_09961292`

They hold **12 ground-truth divisions** between them, which sets the
[[One Division Event Floor]].

**Two known non-representativeness problems:**
- On the node-count axis they run *under* `n_est` (ratio 0.8977) while the test
  films run over — see [[Node Budget]].
- Two embryo prefixes behave very differently; 6bba carries most divisions and
  all of the node-count penalty.

`44b6_341df25f` is where [[s08]]'s whole gain lives.

Related: [[Test Films]], [[Proxy Score]]
""")

note("Test Films", ["data"], """
# Test Films

The 4 films the submission is actually scored on:

`44b6_0113de3b` · `44b6_0b24845f` · `6bba_05b6850b` · `6bba_05db0fb1`

Node counts are heavily skewed — `6bba_05db0fb1` alone has ~70k nodes against
`6bba_05b6850b`'s ~6k, so the real score is dominated by one film. A historical
submission note records the score as **95.4% 6bba**.

That skew is why per-film reporting matters and why a gain on the
[[Validator Films]], which are weighted evenly, need not transfer.

Related: [[Node Budget]], [[Axis Priors]]
""")

note("Frozen Weights", ["data"], """
# Frozen Weights

Nothing in this project trains a model. Three public datasets by `pilkwang`,
SHA256-verified against values pinned in the notebook:

- `biohub-tracking-support-pack-50ep-v1`
- `biohub-temporal-unet3d-seed314159-v1` → [[Temporal UNet3D Detector]]
- `biohub-deepcenter-unet3d-center-prior-v1` → [[DeepCenter]]

Already downloaded to `weights/`.

**This is the binding constraint on the whole project.** Every detection and
matching failure in [[Divisions Played Out]] is unfixable, and
[[Failed Division Classifier]] showed the frozen features do not carry the
division signal.

The base notebook is a fork of Reyhan Ksatria's 0.947 notebook with three input
paths changed and nothing else. See [[0.947 Plateau]].
""")

note("Scored Films Measurement", ["finding", "key"], """
# Measuring On The Films That Are Actually Scored

**The most important measurement in the project**, and it should have come first.

The submission covers exactly **4 films**. We hold their ILP graphs
(`artifacts/s05_output/.../unet_transformer/split_0/`, verified byte-identical
to the relink-ON run, so pure ILP output) and their ground truth is in
`data/.../train/`. So the actual scored films can be scored locally.

| change | validator (8 films) | **SCORED (4 films)** | verdict |
|---|---|---|---|
| [[s05]] over base | +0.01794 | **+0.02707** | holds, and is *larger* |
| [[s08]] over s05 | +0.00748 | **+0.00000** | neutral — validator-only |
| [[s09]] over s08 | +0.00487 | **−0.00086** | **sign flip** |
| s09 over base | +0.03030 | +0.02621 | worse than s05 alone |

**[[s05]] is the best configuration on the films that count.** It gains on all
four individually, including the heaviest:

| film | GT divisions | base | s05 | delta | GT edges |
|---|---|---|---|---|---|
| 44b6_0113de3b | 0 | 0.86847 | 0.96051 | **+0.09204** | 53 |
| 44b6_0b24845f | 0 | 0.97812 | 0.97925 | +0.00113 | 52 |
| 6bba_05b6850b | 0 | 0.96172 | 0.96689 | +0.00517 | 864 |
| 6bba_05db0fb1 | 3 | 0.84089 | 0.87974 | **+0.03885** | 1304 |

**Why [[s08]] evaporates:** the scored films have a division ledger of **0/6/3**,
so `divJ = 0` in every arm. s08's whole gain was one *division*
([[Division Recovery in 44b6_341df25f]]) and there is no division to recover here.

**Why [[s09]] flips:** it helps `6bba_05b6850b` (+0.00686, weight 861) and hurts
`6bba_05db0fb1` (−0.00585, weight 1258). The heavier film loses.

⚠ **Caveat:** labels on the scored films are sparse (0.16%–13.5%), only ~2,273 GT
edges and 3 divisions total. Noisier than the validator set — but it is the
actual target, and the s05 signal is broad and large.

Scripts: `102_on_real_scored_films.py`, `103_relink_on_scored_films.py`.

Related: [[Manifest Contamination]], [[Validator Films]], [[Test Films]]
""")

note("Manifest Contamination", ["finding", "gotcha", "key"], """
# The All-Train Manifest Claim

A discussion post argued the public stack's detectors saw all 199 train videos,
so any local hold-out is in-sample and local measurement cannot be trusted.

**Verified true**, from the manifests in `weights/`:

```
unet_transformer/split_0/split_manifest.json
  method: unet_transformer_alltrain_seed314159_v1
  train:  199        test: 40 -- a strict SUBSET of train (40/40 overlap)
```

All 8 [[Validator Films]] are in that 199.

**But the inference does not follow.** All **4** [[Test Films]] — the films the
submission is actually scored on — are in the *same* 199 train list. In-sample-ness
is **symmetric**, so it cannot by itself explain a local-vs-board sign flip.

What differs is **composition**:
| | validator | scored |
|---|---|---|
| films | 8 | 4 |
| GT divisions | **12** | **3** |
| weighting | roughly even | dominated by one 70k-node film |

**A sharper fact nobody mentioned:** the [[DeepCenter]] split is not a hold-out at
all — it is **entirely embryo-level**. Train = all 71 `44b6` films, val = all 128
`6bba` films. DeepCenter has never seen a single 6bba frame, and 6bba carries 125
of 151 divisions and most of the scored weight.

**Their counter-example was right about the mechanism, though.** A sub-voxel peak
refinement gained locally and lost on the board — a coordinate-localisation
change, the same family as [[s09]], which [[Scored Films Measurement]]
independently shows flipping sign.

**Verdict:** the fact is real, the causal story is wrong, the warning lands.
Fixed by [[Scored Films Measurement]], which removes the need to argue about it.
""")

note("Edge Pruning Is Dead", ["failed", "key"], """
# Edge Pruning — a real discriminator beaten by the base rate

[[Error Budget]] found false-positive edges are the largest error class
(+0.03000 oracle ceiling) and that **nothing in the pipeline removes an edge** —
[[Gap Closing]], [[Gap2 Recovery]] and [[Safe Division]] all add. The `.geff`
carries a probability per ILP-selected edge, so pruning is buildable.

**It does not work, and the reason is principled rather than empirical.**

**The discriminator is real.** Pooled **AUC 0.788** (validator) and **0.780**
(scored); FP median probability 0.83 against TP median 0.94.

**The base rate kills it.** 5,540 TP against 194 FP — **29:1**. Removing an FP
gains 1; removing a TP costs ~2, because that GT edge becomes unmatched
(`tp−1` *and* `fn+1`). So a threshold must cut FP at more than twice the TP rate,
and it never does:

| threshold | TP cut | FP cut | TP per FP | net |
|---|---|---|---|---|
| 0.55 | 60 | 16 | 3.8 | −104 |
| 0.75 | 440 | 67 | 6.6 | −813 |
| 0.83 *(FP median)* | 870 | 99 | 8.8 | **−1641** |
| 0.90 | 1840 | 147 | 12.5 | −3533 |

Empirically confirmed: every threshold from 0.50 to 0.85 loses on **both** tiers,
worsening as it rises.

**What would be needed:** a *relative* rule — among a node's candidate
successors, drop the weaker — which does not pay the base rate. That is
structurally impossible here: every `solution` flag in the `.geff` is True, so
the file stores **only ILP-selected edges** and there are no alternatives to
compare against. Same blocker as [[Learned Bonus Is Crippled]].

⚠ **A methodological catch worth keeping.** The first diagnostic required BOTH
endpoints of an edge to match a GT node, found **5** FPs instead of 184, reported
AUC **0.496**, and would have killed this direction as "no signal". The metric
(metric2.py:51-57) counts an edge as FP when **either** endpoint matches a GT
node that ought to have an edge in that direction — typically a labelled cell
linked to an *unlabelled* node. Use the metric's own definition or the diagnostic
measures nothing.

So the [[Error Budget]]'s +0.030 FP ceiling is **real but not reachable by
thresholding**. It remains the largest untouched class.
""")

note("Board Predictions", ["rule", "key"], """
# Board Predictions — pre-registered, before any result

Baseline **0.947** (verbatim repro, submission 56202098). Board resolution is
**0.001**; anything finer is invisible. 7th/last prize **0.964**, leader **0.973**.

| submission | band | central | basis |
|---|---|---|---|
| **[[s05]]** | 0.960 – 0.970 | **0.969** | in-kernel +0.0224 (right chain, wrong films); scored-film +0.02707 (right films, wrong chain) |
| **[[s10]]** | s05 +0.002…+0.005 | s05 +0.003 | scored +0.00471, in-kernel +0.00207 |
| **[[s08]]** | **equal to s05** ±0.001 | = s05 | scored films +0.00000 — no recoverable division there |

**Why a real gain is expected:** the two estimates agree at ~+0.022…+0.027
*despite differing in both chain and films*. **Why to discount it:** our chain
sits 0.0564 below the board in absolute terms, and the stages accounting for that
gap — [[Single Parent Repair]] especially — repair the same damage
[[Motion Relink]] causes. Partially redundant, so the board gain should land
**at or below** the in-kernel figure, not above.

## What each outcome means

**[[s05]]** — `≥0.964` prize territory · `>0.947` lane confirmed · **`=0.947` is
the most damaging result**, worse than a small loss: it would mean +0.0224
in-kernel transferred to *nothing*, refuting the validator that rejected [[s01]]
· `<0.947` lane dead, everything today void.

**[[s10]]** — the [[Both-Sets Rule]] on trial. `>s05` validates it. `=s05` is
uninformative (below resolution), not a refutation. `<s05` refutes it: three
tiers agreed and were wrong, which would mean 12 films cannot settle an
edge-axis question and [[s11]]'s 32 films become the whole strategy.

**[[s08]] — the instrument test, and the important one.** `=s05` validates
[[Scored Films Measurement]]: it predicted +0.00000 while the validator films
said +0.00803, and was right. `≠s05` refutes it — and with it the basis for
cancelling [[s09]] and shipping [[s10]].

> **The asymmetry worth naming:** s08 is predicted to be worth *nothing* and is
> the most informative of the three. s05 carries the score; s08 carries the
> epistemics.
""")

note("Error Budget", ["finding", "key"], """
# Error Budget — where the remaining error actually lives

Answers one question: **is a better detector worth building?** No.

From the kernel's own validator output for [[s05]] (not our ports), 8 films:
`edge TP 5565, FP 184, FN 186`, micro edge Jaccard **0.93766**. The kernel
classifies every missed GT edge by cause. Oracle ablation — set one class to
zero, hold the rest fixed:

| oracle fix | edges | ceiling | owner |
|---|---|---|---|
| perfect DETECTION | 75 | **+0.01264** | detector — *what SSL would target* |
| zero fragmentation | 111 | +0.01870 | linker |
| zero wrong-association | 1 | +0.00017 | linker |
| **zero FALSE-POSITIVE edges** | 184 | **+0.03000** | linker |

- **detector ceiling: +0.01264**
- **linker ceiling: +0.04887 — 3.9× the detector**
- **296 of 371 errors (80%) are linker-owned**

The detector already recovers **96.75%** of GT edges. Perfect detection — not a
better model, *perfect* — is worth +0.0126, and three learned attempts already
died at chance: [[Failed Division Classifier]], [[Failed Anaphase Hypothesis]],
[[Failed Synthetic Division Data]]. With [[Frozen Weights]] and ten days, SSL is
the wrong target.

**The blind spot this exposes:** false-positive edges are the single largest
class (+0.03000) and **nothing in the pipeline targets them**. [[Gap Closing]],
[[Gap2 Recovery]] and [[Safe Division]] all *add* edges. Not one stage removes a
wrong one. Every change in the [[Submission Ledger]] has pushed in the
add-edges direction.

⚠ **Ceilings are oracles and are NOT additive.** FP and FN trade off directly —
linking more conservatively cuts FP and raises FN. The honest read is the
*asymmetry of attention*, not the arithmetic: 80% of the error is linking, and
the largest class has never been attacked.

⚠ The kernel's own breakdown sums to 187 against a reported FN of 186 — its
classifier and its metric disagree by one edge. Treat ceilings as ±1 edge.

Related: [[Evidence Tiers]], [[Upstream Knobs]], [[ILP Linker]]
""")

note("Evidence Tiers", ["rule", "key"], """
# Evidence Tiers — what each film set can and cannot support

Every number in this vault was measured somewhere. **Where** decides what it can
carry. Quote the tier whenever you quote a delta.

| tier | films | GT edges | GT divisions | graphs | what it supports |
|---|---|---|---|---|---|
| **SCORED** | 4 | ~2,273 | **3** | deployed ILP | the only set that *is* the target. Small and noisy, but final. |
| **VALIDATOR** | 8 | ~5,700 | 12 | deployed ILP | mechanism and divisions. **Does not predict the board** — see [[Scored Films Measurement]]. |
| **REBUILD-199** | 199 | large | 151 | *weaker rebuild* (greedy, not ILP) | **direction and consistency only.** Absolute values do not carry. Must be calibrated on the 4 scored films first. |
| **KERNEL VALIDATOR** | 8 | — | 12 | deployed, in-kernel | the instrument that rejected [[s01]]. Same 8 films as VALIDATOR. |

**The three failure modes this table exists to prevent:**
1. Quoting a VALIDATOR delta as if it were a board prediction — that is [[s09]].
2. Quoting a REBUILD-199 absolute — it has none; only its *direction* is meaningful.
3. Reporting a division difference finer than [[One Division Event Floor]], which
   is **0.00714** at a 14-division ledger and **0.0111** on the scored films.

The 199-film tier is only admissible because it is **calibrated**: it reproduced
the deployed direction on 4/4 for relink ([[Relink Across 199]]) and 3/4 for
linefit. An uncalibrated rebuild claim is not evidence.

Related: [[Both-Sets Rule]], [[Transfer Lesson]]
""")

note("Both-Sets Rule", ["rule", "key"], """
# The Both-Sets Rule

> **A change ships only if it is positive on BOTH the scored films and the
> validator films — and does not gain by dropping `J` on either.**

Written after [[s09]], which shipped on the validator films alone (+0.00487) and
is **−0.00086** where it counts.

**It has already earned its keep twice:**

- **[[s10]] passes.** Linefit weight on the s05 chain: 0.2 and 0.3 are
  validator-only, 0.7 is scored-only, **0.4/0.5/0.6 are positive on both**, and
  0.6 maximises the worst case. A third tier later agreed — see
  [[Relink Across 199]] and the 0.4–0.6 plateau.
- **`OUTPUT_MIN_TRACK_LEN` fails.** See [[MIN_TRACK_LEN Is Dead]]. It gains proxy
  on both sets at L=7..10 but `J` **falls on the validator set every time**. The
  two sets disagree about the *mechanism*, which the rule catches and a proxy
  comparison would not.

**The second clause matters as much as the first.** Proxy can rise on both sets
while the gain is the [[Node Count Exploit]] on one of them. Require `J` up, or
`ratio` unchanged, not just proxy up.

Related: [[Evidence Tiers]], [[Operating Rules]]
""")

note("MIN_TRACK_LEN Is Dead", ["failed"], """
# `OUTPUT_MIN_TRACK_LEN` Is Dead

The scored-film re-price flagged 6→9 as the next probe: **+0.00772 with J
RISING** (+0.00293), which is *not* the [[Node Count Exploit]] signature and is
why it looked real.

Put through the [[Both-Sets Rule]] before spending a GPU slot
(`scripts/108_shorttrack_both_sets.py`), it fails at **every** setting:

| L | scored dJ | validator dJ | dratio |
|---|---|---|---|
| 7 | +0.00000 | −0.00019 | −0.020 |
| 8 | −0.00217 | −0.00076 | −0.041 |
| 9 | **+0.00293** | **−0.00243** | −0.056 |
| 10 | **+0.00293** | **−0.00379** | −0.071 |

Proxy rises on both sets at L=7..10, but `J` falls on the validator set every
time and the gain tracks `ratio` dropping 2–7%. The two sets disagree about the
**mechanism**, not the size — the exploit wearing a disguise on one film set.

Consistent with [[Failed Short Track Exploit]]: monotone to L=40. **No slot spent.**
""")

note("Relink Across 199", ["finding", "key"], """
# Relink Removal, Across All 199 Films

The largest-sample result in the project, and the strongest support for [[s05]].

`artifacts/graphs/*.npz` stores **both arms** of exactly the s05 contrast:
`raw_edges` (greedy, forks allowed → relink OFF) and `linked_edges` (1:1
Hungarian tight-then-relaxed → relink ON).

**Calibrated first**, per [[Evidence Tiers]] — the 4 scored films are in this set
and their deployed answer is known from [[Scored Films Measurement]]. The rebuild
agrees in direction on **4/4**: +0.09382, +0.03218, +0.01333, +0.05210.

Only then the other 195:

- removing relink **helps 176/199 films (88.4%)**
- mean **+0.03858**, median +0.02456, p05 −0.00653, p95 +0.12539
- `44b6`: 52/71 (73.2%) · **`6bba`: 124/128 (96.9%)**
- weighted aggregate over 199 films: **+0.03764**

**6bba is where removal is near-universal**, and 6bba dominates the scored weight.

`scripts/105_relink_across_199.py`, 199 films × 2 arms in **1.5 s** on 16 workers.

⚠ One bug worth remembering: `gt_edges` in the npz stores **node IDs**, not
indices. Passing them straight to `score_film` gives `J = 0.00000` on every film,
**silently** — the first run said "relink helps 0/199" and that was the loader.
""")

note("s10", ["submission", "inflight"], """
# s10 — `OUTPUT_LINEFIT_WEIGHT` 0.8 → 0.6, on [[s05]]

Pushed 2026-09-19 ~13:52 UTC, `biohub-s10-linefit06-on-s05`. **The first change
selected under the [[Both-Sets Rule]].**

| set | delta |
|---|---|
| SCORED (4) | **+0.00471** |
| VALIDATOR (8) | **+0.00338** |
| REBUILD-199 | +0.00136 (optimum 0.5 at +0.00148) |

Three tiers, three different optima — 0.6, 0.4, 0.5 — but **one shared plateau at
0.4–0.6**, and all three put [[s09]]'s 0.3 outside it.

**Pure edge axis.** `ratio` is identical across all ten weights on both sets,
asserted in the script rather than assumed, so there is no node-count component.
Division ledger unchanged.

⚠ **The gain is concentrated AND two-sided.** Across the 199-film tier at w=0.6
vs deployed 0.8: **119/199 films change, of which 73 improve and 46 GET WORSE**;
80 are untouched because linefit only acts where chains are long enough to fit.
The median delta is exactly 0.00000 at every weight. So s10 is net positive but
carries real downside spread — do not read the mean as a per-film expectation,
and do not describe it as "73 films move".

Supersedes [[s09]] as the linefit decision.
""")

note("s11", ["submission", "infra"], """
# s11 — a data-generation run, not a scoring change

**The problem it solves:** every Kaggle run regenerates the same **12**
deployed-quality graphs (8 [[Validator Films]] + 4 [[Test Films]]). Twelve films
is what killed [[s06]] and [[s09]].

`BIOHUB_VALIDATOR_N_PER_TYPE` controls films per embryo type and the kernel emits
a `.geff` for every one. s11 raises it **4 → 16 = 32 validator films**.

Cost, from [[s05]]'s own `validator_prediction_state.json` (~115 s/film):

| films | projected run |
|---|---|
| 8 (current) | 1.0 h |
| **32 (s11)** | **3.4 h** |
| 40 | 4.2 h |

**Harvest its graphs; do not submit it.** It quadruples the deployed-quality
evidence base permanently.

⚠ **Known limitation, stated up front:** `val_stems` takes `ranked[:N]` per type,
so films 5–16 are *lower ranked* than the current 8 — sparser labels, fewer
divisions. It buys many more **edges** and only some more **divisions**. That is
the right trade: the scored films hold 3 divisions total and `divJ` is 0 in every
arm there.

Related: [[Evidence Tiers]], [[Kaggle Mechanics]]
""")

# ───────────────────────── FACT VERIFICATION ─────────────────────────
# The atlas drifted once already: "Parametric Search Closed" said "nothing
# survives, do not re-tune" after s10 had already been found. A knowledge base
# that lies is worse than none, so every load-bearing number is RECOMPUTED from
# its source artifact at build time and the build FAILS if the note disagrees.
#
# Each check is (label, note, recompute) where recompute returns a string that
# must appear verbatim in that note.

def _csv(path):
    import csv as _c
    with open(ROOT / path) as fh:
        return list(_c.DictReader(fh))


def _relink_helps():
    rows = _csv("artifacts/relink_across_199.csv")
    n = sum(1 for r in rows if float(r["delta_off_minus_on"]) > 0)
    return f"{n}/{len(rows)}"


def _linefit_movers():
    rows = _csv("artifacts/linefit_across_199.csv")
    n = sum(1 for r in rows if float(r["w0.6"]) != float(r["w0.8"]))
    return f"{n}/{len(rows)}"


def _scored_delta(change):
    """The SCORED column is the SECOND signed number on the row."""
    import re as _re
    txt = (ROOT / "artifacts/relink_on_scored.log").read_text()
    for line in txt.splitlines():
        if line.strip().startswith(change):
            nums = _re.findall(r"[+-]\d+\.\d+", line)
            if len(nums) < 2:
                raise AssertionError(f"{change!r}: expected 2 numbers, got {nums}")
            return nums[1]
    raise AssertionError(f"{change!r} not found in relink_on_scored.log")


def _reprice(label):
    for r in _csv("artifacts/reprice_on_scored.csv"):
        if r["label"] == label:
            return f"{float(r['dproxy']):+.5f}"
    raise AssertionError(f"{label!r} not in reprice_on_scored.csv")


def _manifest_train(path, key):
    import json as _j
    return str(len(_j.loads((ROOT / path).read_text())[key]))


def _scored_divisions():
    """Total GT divisions across the 4 scored films -- the atlas claims 3."""
    want = ["44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"]
    d = {r["film"]: int(r["divisions"]) for r in _csv("artifacts/films.csv")}
    return f"{sum(d[f] for f in want)} divisions total"


CHECKS = [
    ("relink helps N/199", "Relink Across 199", _relink_helps),
    ("linefit movers", "s10", _linefit_movers),
    ("s05 on scored films", "Scored Films Measurement", lambda: _scored_delta("s05 over BASE")),
    ("s09 on scored films", "Scored Films Measurement", lambda: _scored_delta("s09 over s08")),
    ("s10 linefit on scored", "s10", lambda: _reprice("lf_w=0.6")),
    ("safe_div off on scored", "Parametric Search Closed", lambda: _reprice("STAGE OFF: safe_div")),
    ("secondary manifest train",  "Manifest Contamination",
     lambda: _manifest_train("weights/biohub-temporal-unet3d-seed314159-v1/weights/"
                             "unet_transformer/split_0/split_manifest.json", "train")),
    ("deepcenter train", "Manifest Contamination",
     lambda: _manifest_train("weights/biohub-deepcenter-unet3d-center-prior-v1/"
                             "weights/full_frame_center/split_manifest.json", "train")),
    ("deepcenter val", "Manifest Contamination",
     lambda: _manifest_train("weights/biohub-deepcenter-unet3d-center-prior-v1/"
                             "weights/full_frame_center/split_manifest.json", "val")),
    ("scored film divisions", "s11", _scored_divisions),
    ("s08 kernel proxy", "Harness Prediction Confirmed",
     lambda: __import__("json").loads((ROOT / "artifacts/s08_output/ppsweep_selected.json")
             .read_text())["base_proxy"].__format__(".4f")),
]


def verify():
    bad, skipped = [], []
    for label, target, fn in CHECKS:
        if target not in N:
            bad.append(f"{label}: note {target!r} does not exist")
            continue
        try:
            want = fn()
        except FileNotFoundError as e:
            skipped.append(f"{label}: source artifact missing ({e.filename})")
            continue
        except Exception as e:
            bad.append(f"{label}: recompute failed -- {e}")
            continue
        body = (N[target][1].replace("\u2212", "-").replace("\u2013", "-")
                .replace("\u2014", "-"))
        if want not in body:
            bad.append(f"{label}: artifact says {want!r}, note {target!r} does not contain it")
    for m in skipped:
        print(f"  SKIP  {m}")
    if bad:
        print("FACT CHECK FAILED -- the atlas disagrees with its own artifacts:")
        for m in bad:
            print(f"  {m}")
        sys.exit(1)
    print(f"fact check: {len(CHECKS) - len(skipped)}/{len(CHECKS)} load-bearing "
          f"numbers recomputed from artifacts and matched")


# ─────────────────────────────── WRITE ───────────────────────────────
FOLDER = {
    "moc": "00 Maps", "concept": "01 Concepts", "stage": "02 Pipeline",
    "submission": "03 Submissions", "finding": "04 Findings",
    "failed": "05 Failed", "infra": "06 Infrastructure",
    "rule": "07 Rules", "gotcha": "08 Gotchas", "data": "09 Data",
    "entity": "09 Data",
}

def folder_for(tags):
    for t in tags:
        if t in FOLDER:
            return FOLDER[t]
    return "99 Misc"

verify()

names = set(N)
dangling, written = {}, 0

if V.exists():
    for p in sorted(V.rglob("*.md")):
        p.unlink()

for name, (tags, body) in N.items():
    for link in re.findall(r"\[\[([^\]|#]+)", body):
        if link.strip() not in names:
            dangling.setdefault(link.strip(), []).append(name)

if dangling:
    print("DANGLING LINKS — every one is a node with no note:")
    for link, srcs in sorted(dangling.items()):
        print(f"  [[{link}]]  <- {', '.join(srcs)}")
    sys.exit(1)

for name, (tags, body) in N.items():
    d = V / folder_for(tags)
    d.mkdir(parents=True, exist_ok=True)
    fm = "---\ntags:\n" + "".join(f"  - {t}\n" for t in tags) + "---\n\n"
    (d / f"{name}.md").write_text(fm + body)
    written += 1

# inbound-link census, so the graph is known to be connected
inbound = {n: 0 for n in names}
for name, (tags, body) in N.items():
    for link in set(re.findall(r"\[\[([^\]|#]+)", body)):
        inbound[link.strip()] += 1
orphans = [n for n, c in inbound.items() if c == 0 and n != "Home"]
total_links = sum(len(re.findall(r"\[\[([^\]|#]+)", b)) for _, b in N.values())

print(f"wrote {written} notes to {V}")
print(f"{total_links} wikilinks, 0 dangling")
print(f"orphan notes (nothing links in): {orphans if orphans else 'none'}")
print("\nmost-linked-to notes:")
for n, c in sorted(inbound.items(), key=lambda x: -x[1])[:12]:
    print(f"  {c:3d}  {n}")
