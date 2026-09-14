# Where we actually are

Last updated 2026-09-14. This file is the single honest answer to "what is the
state of this campaign". It supersedes nothing -- the review documents keep
their dated snapshots -- but if you read one file, read this one.

## Position

| | |
|---|---|
| Public LB | **0.947** |
| Rank | ~100-200 of 3,425 |
| Reproducible | yes -- submission #27 reproduced the public 0.947 kernel exactly |
| Calibration points | #27 (0.947), #28 (0.947, null), #29 (pending) |
| Deadline | 2026-09-29 23:59 UTC; entry/merge cutoff 2026-09-22 |

## The metric, exactly

    score = adj_edge_jaccard + 0.1 * division_jaccard
    adj    = max(0, J * (1.1 - 0.1 * N_pred / n_total))

The multiplier is bounded at **1.1** (as N_pred -> 0) and is exactly **1.0** at
full density. Theoretical max 1.1972. The leaderboard resolves **3 decimals**,
so anything under 0.001 is unobservable and must not be chased.

Exchange rates, measured:

- division break-even precision **24%** (one true division is worth 3.2 false ones)
- edge break-even precision **48.1%** (a false edge and a true edge move J nearly symmetrically)

## The structural law

Every candidate region in this problem has a base rate near **0.1%** against a
metric that needs 24-48% precision. That is a required lift of **150-320x**.
The features available give about **2x**. This is arithmetic, and it is why the
division and edge routes are closed rather than merely difficult.

## Sparse annotation -- the fact that governs everything

Only **2.82%** of cells are annotated (133,318 of 4,725,117). The asymmetry
between embryos is **7x** (44b6 = 0.77%, 6bba = 5.37%). Annotated density is
6.7 cells/frame; real density is **203-237** cells/frame -- a factor of **35**.

Two consequences that have each cost us real time:

1. **The sparse-annotation trap.** Counting unannotated-but-real items as false
   positives. It bit three separate times (a 98.86% "swap" base rate; edgeJ of
   0.03 against a true 0.85; a competitor's own `precision_sparse` reading 0.065).
   Any precision number computed against GT must be restricted to *judgeable*
   items.
2. **Wrong-regime tuning.** Everything tuned before EXP-29 was tuned at oracle
   density, 35x away from production. EXP-17 was overturned on exactly this.

## The four levers, and their status

| Lever | Pool | Status |
|---|---|---|
| edge_jaccard numerator (find missed true edges) | +0.05 | **closed** -- 0.53% detection / 5.3% linking base rate; 316x lift needed |
| edge_jaccard denominator (drop false edges) | +0.03 | **closed** -- EXP-31 cleared break-even, EXP-32 showed appearance adds nothing, #28 returned null on LB |
| 0.1 * division_jaccard | +0.04 | **closed** -- 2.0% detection / 32.5% selection / 65.6% contention; all three fail the 24% bar |
| N_pred multiplier | **+0.068** | **open** -- never directly attacked with a trained classifier |

### Why N_pred is hard rather than easy

97.3% of predicted nodes match no GT node. So **chance alone is 97.3% "safe"**
at removal. A classifier does not need to be good, it needs to beat 97.3%
precision. Removing 25% of unmatched nodes gives multiplier 0.9727 (enough to
win); removing all of them gives 1.0376. Their current multiplier is 1.0038 of
a possible 1.1000.

Every rule tried so far loses edge Jaccard faster than it gains multiplier:
spatial targeting (EXP-22), track-length pruning (EXP-34), their own output
scorer (v-outgrid). **EXP-36** is the first attempt to ask the question as a
supervised problem with the right target: not "is this node spurious?" but
"did a human annotator label this cell?".

## The false ceiling -- stated plainly

Everything in this repository is **post-hoc surgery on DeepCenter's output**.
The detector has never been changed. The post-processing ceiling is real and
has been mapped thoroughly across ~35 experiments. The **detection** ceiling is
completely unmeasured.

The leader at ~0.96 is very unlikely to be post-processing better. The public
forum evidence (dense finetune, elastic augmentation, dzyx bias,
velocity extrapolation) is a *training* story. The open technical problem is:

> how do you finetune a 3D U-Net centre-detector when only 2.8% of instances
> are labelled and unlabelled does not mean negative?

That is partial-label / positive-unlabelled learning for dense detection, and
it is the thing most likely to be worth more than 0.001.

## Reverse-engineering the 0.947 kernel

- Reproduced byte-identically (`kernels/repro-947/`).
- Their `ppsweep` tunes **in-kernel** on **8 held-out train stems containing 12
  GT divisions**, then applies its winner *over* the environment -- so env vars
  for swept parameters are silently ignored. This cost us two runs before it
  was understood.
- Their proxy over-reads the LB by **+0.0041**. This is a **constant offset**
  and therefore **cancels in a difference**. An earlier rule of mine ("a variant
  must beat baseline by >0.004") was wrong and would have discarded the only
  positive result in 38.
- Their sweep builds combinations only from *positive singles*, so it
  structurally cannot find a combination whose parts individually score badly.
  Submission #29 is exactly such a combination (`tight55+vel025+tight50relax9`,
  proxy +0.00132, where `vel025` alone is -0.00061). This is the one genuine
  structural edge we have over the public kernel.

## What remains

1. **EXP-36** -- matched-vs-unmatched node classifier. The only untested idea.
2. **v-fine1 / v-fine2** -- 34 configs around the winning combination. ~+0.001.
3. **Repair re-test** -- `gap_max_frames=0`, `min_track_len=1` were set at
   oracle density and never re-tested at real density. Expected ~0.000.

There is no fourth item. If EXP-36 returns AUC near 0.6, the post-processing
route is finished and the only remaining path is retraining the detector.

## Discipline that must not be relaxed

- **LOEO.** Train and test embryos are disjoint by host statement. Fit on one
  embryo, report on the other, take the **minimum**.
- **Paired bootstrap over datasets**, not over divisions. Unpaired, 151
  divisions resolve only +/-0.0068 of score -- wider than every effect we have
  ever measured.
- **Never `sorted(files)[:N]`.** That slice is entirely 44b6. It bit three
  times (EXP-18, EXP-26, EXP-32). Balance embryos explicitly.
- **Pre-check every submission against the in-kernel proxy.** This has already
  prevented two bad submissions.
- **A kernel run is not a submission.** Submit explicitly via the CLI.
- **`machineShape="NvidiaTeslaT4"`** selects the GPU. `acceleratorType` is
  accepted and silently ignored, and a P100 (sm_60) is fatal -- the pinned torch
  needs sm_70+.

## EXP-36 result -- the N_pred lever is closed (2026-09-14)

827,797 nodes, 30 embryo-balanced datasets, 14 features, LOEO GradientBoosting,
target = "did a human annotator label this cell".

    AUC 0.641 (held out 44b6) / 0.557 (held out 6bba)
    best single feature: comp_len, AUC 0.726  <- this is EXP-34's track length, already null

Removal precision does beat chance, but chance is the wrong bar. Removing
fraction f of nodes gains multiplier 0.0962*f and costs edge Jaccard about
2*f*(1-p)/m, so net-positive requires

    (1 - p) < 0.0479 * m

| embryo | matched m | error needed | error achieved | short by |
|---|---|---|---|---|
| 44b6 | 0.55% | 0.026% | 0.258% | 9.8x |
| 6bba | 4.21% | 0.202% | 3.633% | 18x |

At 25% removal on 6bba the rule discards 2,815 of 13,050 matched nodes -- 21.6%
of the entire matched set -- to buy a 2.4% multiplier gain. Net about -0.15.

Why it fails, stated causally: annotators chose **lineages to follow**. That is
a decision made once at the root of a track and then propagated down it. It is
not a property visible at an individual node, so no per-node feature set can
recover it.

**All four levers are now closed.** Post-processing on DeepCenter's output is
finished. The only remaining path to a score above ~0.950 is changing the
detector: partial-label / positive-unlabelled finetuning of the 3D U-Net on the
2.8% annotated cells, where "unlabelled" must not be treated as negative.

## EXP-37 -- the detector is NOT the bottleneck (2026-09-14)

Measured before spending GPU hours on a detector retrain, because the retrain is
only worth doing if detection is what binds. It is not.

    GT nodes 15,934   detected 15,901   NODE RECALL  0.9979
    GT edges 15,369   reachable 15,316  EDGE CEILING 0.9966
    structurally unrecoverable edges: 0.34%
    44b6 node recall 0.9986 / edge ceiling 0.9971
    6bba node recall 0.9978 / edge ceiling 0.9964

The edge ceiling is the maximum edge recall ANY linker can reach on this
detector's output -- an edge missing an endpoint cannot be produced. At 0.9966
against a measured edgeJ of 0.9116, detection accounts for 0.34 points of an
8.5-point gap. **This retracts the pivot recommended earlier the same day.**

There is a second and sharper reason not to retrain. n_total in the metric is
`estimated_number_of_nodes` -- the estimated count of ALL real cells, not the
annotated ones. A better detector finds more real-but-unannotated cells, which
raises N_pred and *lowers* the multiplier. **The metric actively penalises
better detection.** That is the most likely reason the public leaderboard piles
up between 0.940 and 0.948 regardless of model.

## The measured decomposition of 0.947

From work/repro_out/validator_results.csv (their pipeline, our GT):

| component | value | max |
|---|---|---|
| edge_jaccard | 0.9116 | 1.0 |
| multiplier | ~1.005 | 1.1 |
| adj_edge_jaccard | 0.9114 | — |
| div_jaccard | 0.2308 micro (0.3125 per-sample mean) | 1.0 |

Edge failures split as: edges_fragmented 17.1/dataset (linking) against
edges_lost_to_detection 9.9/dataset (detection), with wrong_association_edges
exactly 0. Fragmentation is the single largest identified failure mode.

## EXP-38 / 38b -- division relaxation, and a claim I had to retract

I read a division precision of 72.7% off their validator and argued we were
running 6x more conservatively than break-even (12.4% at that operating point,
since Jaccard moves +0.609 per true division and -0.086 per false one).

**That rate is 3 true divisions against 1 false one.** Their whole validator is
8 stems carrying 12 GT divisions. A sample of four carries no information, and
the claim is withdrawn. This is the same resolution problem already recorded for
the paired bootstrap, where even 151 divisions resolve only +/-0.0068.

EXP-38's own hand-rolled division metric also failed validation (divJ 0.0172 vs
0.3125), so it was discarded and EXP-38b calls the competition's `evaluate()`
and `per_sample_metrics()` directly. The cause of that mismatch turned out to be
the substrate, not the metric: work/train_graphs carry ~187 native linker forks
per dataset, where their pipeline's output carries almost none.

Real scorer, 40 embryo-balanced datasets:

    config                    divTP  divFP  divFN    divJ    edgeJ    SCORE    delta
    baseline (as-is)             11    481     15  0.0217   0.8899   0.8722  +0.0000
    theirs (9,14,.6,2.25)        11    502     15  0.0208   0.8889   0.8710  -0.0012
    max_um 13                    11    502     15  0.0208   0.8889   0.8710  -0.0012
    tau .25                      11    485     15  0.0215   0.8896   0.8718  -0.0004

**divTP never moves.** Every gate relaxation adds false divisions and not one
true one, and our own linker's native forking already runs at 2.2% division
precision -- below the 12.4% break-even, not above it. Divisions are closed.

## EXP-39 and what the test set actually is (2026-09-14)

**The test set is four datasets.** 44b6_0113de3b, 44b6_0b24845f, 6bba_05b6850b,
6bba_05db0fb1 -- and `estimated_number_of_nodes` ships for every one of them in
the organisers' released GEFF metadata: 25,755 / 32,795 / 6,362 / 69,800. The
images are shared with train; the leaderboard annotations are held-out cells in
the same movies (our local GT for those four stems holds only 52 / 51 / 861 /
1,229 annotated nodes).

This makes the multiplier term **exactly computable offline**. It is applied per
dataset, adj is weight-averaged by each sample's annotated-edge count
(metrics.summarise), and node rows per dataset are countable in any
submission.csv. `tools/multiplier.py` does it. The multiplier stops being a
quantity we infer from leaderboard deltas.

Their 0.947 pipeline on the real test set:

    44b6_0113de3b  N_pred 25,637  n_total 25,755  ratio -0.005  mult 1.0005
    44b6_0b24845f  N_pred 20,721  n_total 32,795  ratio -0.368  mult 1.0368
    6bba_05b6850b  N_pred  6,152  n_total  6,362  ratio -0.033  mult 1.0033
    6bba_05db0fb1  N_pred 70,284  n_total 69,800  ratio +0.007  mult 0.9993
    unweighted mean 1.0100 of a possible 1.1000

### The experiment

Where N_pred EXCEEDS n_total the excess cannot be real-but-unannotated cells --
there are not that many cells to be unannotated. That excess is *provably*
spurious, which is exactly the property EXP-36 searched for and could not find.
So trim only datasets above a ratio threshold, only down to a target ratio,
smallest connected components first.

On 50 train datasets (median ratio +0.109, 39 of 50 over-predicting):

    config                    edgeJ    divJ      adj    SCORE    delta
    no trim                  0.8933  0.0223   0.8829   0.8852  +0.0000
    ratio>0.00 -> 0.00       0.8950  0.0218   0.8969   0.8991  +0.0140
    ratio>0.00 -> -0.05      0.8919  0.0220   0.8975   0.8998  +0.0146
    ALL -> -0.20             0.8577  0.0242   0.8754   0.8778  -0.0074

    LOEO: 44b6 +0.0025   6bba +0.0177   -> minimum +0.0025

Edge Jaccard *improves* on 6bba under the trim (0.9029 -> 0.9062). Over-detection
is causally upstream of fragmentation, which is our largest edge failure mode.

**It does not transfer.** Only one test dataset over-predicts, by 0.7%, worth
+0.0007 -- under the leaderboard's 0.001 resolution. Null for submission.

The general lesson, which has now cost three experiments: our cached train
graphs are NOT the 0.947 pipeline's output. They over-predict (median +0.109)
and carry ~187 native linker forks per dataset where their pipeline carries
almost none. Anything measured on train_graphs must be re-checked against their
actual output before it is believed to be a submittable gain.

## EXP-40 to 43 -- the fragmentation route, closed (2026-09-14)

Run against their validator output graphs (predictions/.../unet_transformer_val/split_0), NOT
work/train_graphs, after the substrate lesson cost three experiments earlier the same day.

### EXP-40 -- flow-compensated stitching works, and reaches almost nothing

Build a motion field from the accepted edges (v = x_j - x_i, interpolated by Gaussian-weighted
kNN), predict each orphan sink's successor at x_i + v(x_i), require mutual best, solve Hungarian.

    r4  mutual   500 proposed    9 judgeable   8 correct   88.9%
    r6  mutual   745 proposed   12 judgeable  10 correct   83.3%
    r9  mutual 1,068 proposed   15 judgeable  11 correct   73.3%
    r12 mutual 1,526 proposed   17 judgeable  11 correct   64.7%

Precision is far above the 48.1% break-even. Reach is the problem: worth about +0.0013.

### EXP-41 -- why. Only 5.5% of the pool is free-endpoint

    GT edges 5,751   present 5,552   MISSING 199
      FREE          11     5.5%   <- all EXP-40 can reach
      I_BUSY        36    18.1%
      J_BUSY        41    20.6%
      BOTH_BUSY     47    23.6%
      NO_NODE       64    32.2%

**"wrong_association_edges = 0" is an artefact of sparse annotation.** Their diagnostic counts a
wrong link only when BOTH endpoints are annotated. In 62.3% of missing edges an endpoint is
already linked to something else -- a wrong association whose wrong partner is an unannotated
cell, so the diagnostic reads zero. Any plan resting on "the existing links are perfectly
trustworthy, never touch them" is resting on a measurement artefact.

### EXP-42 -- the true successor is reachable, and flow does not help

    rank of TRUE successor      raw dist   flow-comp
      top-1                        37.0%       37.0%
      top-2                        80.0%       80.7%
      top-3                        92.6%       91.9%
    flow beats raw on 11, loses on 12, ties 112

All 135 judgeable cases have the true successor within 15 um. So this is a two-way
discrimination, not a search. And **flow compensation is a null** -- the embryo's coherent
motion is small relative to cell spacing and never reorders candidates.

### EXP-43 -- the linker already wins the reranking comparison

    judgeable single-successor nodes: 5,635
    linker AGREES with nearest : 5,575   correct 5,499  wrong 76
    linker DIFFERS from nearest:    60
        linker right, nearest wrong :  53
        nearest right, linker wrong :   7
        both wrong                  :   0
    head-to-head precision of 'nearest': 11.7%  (break-even 48.1%)

The linker beats nearest-neighbour 53 to 7. Snapping all disagreements to nearest loses 46
edges. Every proposed rerank -- LAP stitching, Kalman gating, mutual NN, a learned rejoin
classifier, Trackastra, HOCT -- reranks candidates by geometry, and the global linker already
beats that decisively because it has strictly more information.

**The residual: 76 of 83 errors are cases where linker and nearest AGREED and both were wrong.**
The true successor simply is not the nearest cell. Identifying those 76 among 5,575 agreements
is a 1.4% base rate against a 48.1% bar -- a 34x lift. The structural law, third instance.

Fragmentation is closed. It was the last route flagged as open.

## EXP-44 / 45 -- why the last route is closed: the data is at the aliasing limit

EXP-43 localised every remaining edge error to one shape: the true successor exists, another
cell is closer, the tracker takes the closer one. EXP-44 tests, pairwise on that HARD set only,
which geometric signal prefers the true successor over the near impostor.

    signal          prefers TRUE    n
    euclid                  0.0%  138   control, 0% by construction
    flow_knn               39.9%  138
    flow_affine            38.4%  138
    accel                  15.4%  130
    neigh_topo             39.1%  138
    back_cycle             39.9%  138

**Every signal is below 50%** -- they all side with the impostor, clustered near 39%. That
uniformity is the tell: each feature is dominated by the same term, distance.

EXP-45 measures why, and the answer is structural rather than a matter of feature design:

    |mu|  local displacement field        1.40 um   (p90 6.17)
    gap   d(true) - d(nearest)            3.26 um   (p90 7.49)
    d(i, true successor)                  6.89 um
    d(i, nearest impostor)                3.63 um
    nearest-neighbour spacing at t+1      6.70 um
    coherence |mu| / spread               0.77

    |mu| exceeds the gap it must overturn in 43.5% of cases
    local motion is coherent (|mu| > spread) in 40.6% of cases

Two independent reasons the premise fails. The correction available (1.40 um of flow) is
**smaller than the ordering error it must overturn** (3.26 um gap), so no flow-based feature of
any sophistication -- affine, deformation field, transformer -- can reorder these candidates.
And coherence is 0.77: the spread of neighbour displacements EXCEEDS the collective component,
so the tissue is not moving collectively at this frame interval. "Neighbours predict each other"
is the assumption these methods need, and the data does not satisfy it.

The decisive number is the last pair. On the hard set the true successor sits **6.89 um** away
while nearest-neighbour spacing is **6.70 um**: the cell moves roughly ONE FULL INTER-CELL
SPACING per frame. That is the aliasing limit. At that ratio the nearest cell at t+1 is not
expected to be the same cell, and the correspondence is not recoverable from geometry because
the information is not present in the geometry.

**CORRECTED 2026-09-14 (later).** The paragraph below overstated this. These statistics are
conditioned on the HARD RESIDUAL SET -- cases where the true successor is not the nearest cell --
not on all edges. Displacement comparable to spacing demonstrates ambiguity for PROXIMITY-BASED
matching; it is not a Nyquist limit on cell identity, and longer temporal context, appearance or
global constraints may retain information proximity does not. Nor does it show the ~0.96 leaders
are exploiting the scorer: that claim needs a score decomposition or an ablation, not a
displacement-to-spacing ratio. Acquisition timing is also unresolved -- the competition zarr
metadata says 1 second (a placeholder), while the published Ultrack whole-embryo protocol used
90 s and DaXi ~60 s. At 90 s, 6.89 um implies 4.59 um/min against published lateral-mesoderm
averages of 2.2-2.8 um/min, so these are roughly 2x typical speed, consistent with being the
hard tail rather than the norm.

With that scope, it is still a good explanation of:
  - why appearance fails (EXP-32) -- the cells genuinely look alike
  - why flow fails (EXP-42, 44) -- motion is incoherent and too small
  - why the linker's residual errors are irreducible (EXP-43)
  - why the public leaderboard saturates at 0.940-0.948 regardless of model

The 97.6% of edges the linker gets right are the ones sampled below the aliasing limit. The
1.5% it misses are the tail where displacement reaches inter-cell spacing. That tail is not a
modelling failure. It is a property of the acquisition.

## The weighting discovery, and the first real gain (2026-09-14)

### 6bba is 95.4% of the score

`tools/testproxy.py` scores a prediction on THE ACTUAL FOUR TEST MOVIES using the released
annotations. Their 0.947 pipeline, per stem:

    stem               N_pred  n_total    mult   edgeJ     adj   weight  share
    44b6_0113de3b      25,822   25,755  0.9997  1.0000  0.9997       50   2.3%
    44b6_0b24845f      22,491   32,795  1.0314  0.9600  0.9902       50   2.3%
    6bba_05b6850b       6,305    6,362  1.0009  0.9697  0.9705      857  38.6%
    6bba_05db0fb1      70,687   69,800  0.9987  0.8688  0.8677     1265  56.9%

adj is weight-averaged by each sample's ANNOTATED edge count (metrics.summarise), so **6bba
carries 95.4% of the score and 44b6 carries 4.6%**. Every leave-one-embryo-out MINIMUM this
campaign reported was dominated by an embryo worth one twentieth of the result. EXP-39 gained
+0.0177 on 6bba and +0.0025 on 44b6, and was reported as +0.0025.

`6bba_05db0fb1` alone is 56.9% of the score and holds nearly all the loss (edgeJ 0.8688 against
1.0000 / 0.9600 / 0.9697). On it, 27 of 84 missing edges are lost to detection -- 2.3% of its GT
edges, double the validator rate.

### EXP-47 -- OUTPUT_MIN_TRACK_LEN, judged on the right substrate

    config              adj      SCORE    delta   per-stem adj
    theirs (6, none)  0.91281  0.91281  +0.00000  0.961 1.000 0.975 0.866
    minlen 8          0.91790  0.91790  +0.00510  0.962 1.010 0.979 0.871
    minlen 9          0.92061  0.92061  +0.00780  0.963 1.013 0.984 0.872
    minlen 10         0.92228  0.92228  +0.00947  0.963 1.017 0.986 0.874
    minlen 11         0.90875  0.90875  -0.00406  0.964 1.020 0.987 0.849   <- cliff
    minlen10+emax11   0.92269  0.92269  +0.00988

EXP-34 read this null on 199 train graphs. Wrong substrate twice: a different pipeline, and
even embryo weighting.

### Why it is robust: the gain is ENTIRELY multiplier

    stem             minlen   N_pred    mult   edgeJ     adj
    6bba_05db0fb1         6   68,441  1.0019  0.8639  0.8656
    6bba_05db0fb1        10   62,931  1.0098  0.8653  0.8738
    6bba_05b6850b         6    6,044  1.0050  0.9697  0.9745
    6bba_05b6850b        10    5,447  1.0144  0.9719  0.9859

**Edge Jaccard never falls; it rises slightly on both heavy stems.** The multiplier depends only
on N_pred and n_total, both known exactly with no annotation dependence, so this transfers to the
hidden annotations essentially unchanged. Short components are tracking fragments: pruning them
drops ~8% of nodes and costs no annotated edges. This is the provably-spurious pool EXP-39
sought, reached by another route.

**Submitted minlen 9 (+0.0078), not the peak at 10**, because the cliff at 11 is one step away
and the hidden annotations are a different sample of the same movies.

Caveat: their filter has OUTPUT_KEEP_DIV and SHORT_TRACK_RESCUE_MIN_LEN, so the kernel prunes
more gently than the plain component-size model used offline. Direction should hold; magnitude
may differ.

## EXP-48 -- per-dataset ratio targeting is WORSE than a global minlen

The multiplier is applied per dataset and n_total is known for all four test movies, so targeting
a node ratio per dataset looked strictly better than one global OUTPUT_MIN_TRACK_LEN. It is not.

    order    target      adj    SCORE    delta
    none          -  0.91307  0.91307 +0.00000
    size      +0.00  0.91236  0.91236 -0.00071
    size      -0.05  0.91416  0.91416 +0.00109
    size      -0.10  0.91497  0.91497 +0.00190   <- best, vs minlen 9 at +0.00780
    size      -0.15  0.90998  0.90998 -0.00309
    size      -0.20  0.89104  0.89104 -0.02203

Peak +0.0019 against the global minlen's +0.0078. The reason is worth keeping: a ratio target
STOPS pruning once a dataset reaches it, but edge Jaccard stays flat far below n_total, so
pruning keeps paying. 44b6_0b24845f already sits at ratio -0.368, so a -0.10 target prunes it not
at all, while minlen 10 takes it to 13,479 nodes and multiplier 1.0589. The global threshold wins
precisely because it is unbounded.

`size` and `span` orderings give identical output, which is legitimate rather than a bug: a track
is a chain, so a 10-node component spans 10 frames. (An earlier version of this script grouped by
rank VALUE instead of by component and was discarded.)

## Instrument calibration -- and a correction (2026-09-14)

`tools/score_submission_local.py` scores a real submission.csv on the four test movies against
the released annotations. Run over every variant we hold:

    variant      local      their_proxy
    repro       0.89319      0.94904
    ns          0.89319      0.94904   <- both LB 0.947, EXACT null
    tight45     0.89319      0.94726   <- byte-identical test submission
    tight50     0.89319      0.94648   <- byte-identical test submission
    outgrid     0.89319      0.94904   <- byte-identical test submission
    ppgrid      0.89629      0.94904   +0.0031
    minlen9     0.89638      0.94900   +0.0032
    vr          0.89051      0.92803   both negative
    vs          0.89237      0.94621   both negative

**The one hard test passes.** repro and noSister have identical leaderboard scores (0.947 both),
and the local proxy scores them identically to five decimals. The noSister change produced no
difference on the test set whatsoever.

**CORRECTION.** An earlier note here said their in-kernel proxy reads minlen9 at -0.0021 and so
conflicted with our +0.0032. That used 0.9511 as the baseline; repro's actual base_proxy is
0.94904, against which minlen9's 0.94900 is -0.00004. **The two instruments do not conflict.**
Theirs is insensitive where ours resolves, which is expected: 8 TRAIN stems with EVEN embryo
weighting against four TEST movies weighted 95.4% 6bba.

**tight45, tight50 and outgrid produce byte-identical test submissions.** Those knobs change
nothing on the test data even though their proxy reads differences on train stems. A large part
of their ppsweep is inert where it counts.

**ppgrid independently gives +0.0031** by a different mechanism than minlen9's multiplier gain.
The two have never been combined -- that is the obvious next run.

## EXP-49 -- does the Ultrack annotation protocol reopen the N_pred pool? Weakly, and not enough

The published Ultrack ground-truth protocol (Nature Methods 2025) introduces sparse RANDOM red
nuclear labelling by early microinjection, the marker propagates to daughters, and annotators
select "long, green-overlapping, high-quality lineages" -- 152 tracklets spanning 85-521 frames.
Competition provenance for 44b6 / 6bba is UNCONFIRMED, so this measured the claim rather than
assuming it.

Two things follow from the protocol even if it holds. Membership depends partly on a SEPARATE
FLUORESCENCE CHANNEL not present in the supplied images, so it is not a deterministic selector we
could compute -- which is consistent with EXP-36's failure. And the selection is on lineage
LENGTH, which is measurable.

On the four real test movies, predicted-component length of matched vs unmatched nodes:

    ALL   matched 2,172 (median component 74 nodes)   unmatched 120,622 (median 43)

     minlen  matched KEPT  unmatched KEPT  nodes pruned
          9        99.6%           96.0%          3.9%
         14        97.4%           89.4%         10.5%
         20        93.6%           81.2%         18.6%
         50        73.7%           42.5%         57.0%

Annotated cells do sit in longer components, but by a factor of **1.7**, not the order of
magnitude "85-521 frame lineages" implies. FRAGMENTATION destroys the signal: a real 300-frame
lineage arrives as several ~70-node pieces, so predicted length is a weak proxy for annotated
lineage length.

Applying the break-even condition  eps/J < delta/(m+delta)  at m = 1.0138:

    step        extra pruning   delta/(m+delta)   matched lost   eps/J ~ 2f   verdict
    minlen 9        3.9%            0.38%            0.4%          ~0.8%      marginal
    minlen 14      10.5%            0.64%            2.2%          ~4.4%      negative
    minlen 20      18.6%            1.42%            6.4%         ~12.8%      clearly negative

**Prediction recorded before the kernel landed: biohub-ml14 will score worse than minlen9.**

Aggressive pruning is not available. minlen 9-10 is at or near the optimum, and the 0.0862
multiplier headroom is not independently attainable -- the 1.1 ceiling sits at zero predicted
nodes, which cannot preserve edge recall.

## ml9-norescue -- a null, and a wrong attribution corrected (2026-09-14)

    variant        local test score
    repro           0.89319
    minlen9         0.89638
    ml9-norescue    0.89639   <- exact null vs minlen9

Disabling ADAPTIVE_SHORT_TRACK_RESCUE changed essentially nothing: N_pred on the heavy stem is
68,565 either way, and only 44b6_0b24845f moved at all (18,328 -> 18,209).

**Correction.** I attributed minlen9's gain being a third of my offline model to the short-track
rescue pulling components back. That was wrong. The rescue has
BIOHUB_SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC = 0.10 and only fires when more than 10% of nodes
are removed; minlen 9 removes about 4%, so it never triggered and turning it off was a no-op.

The real cap was already in EXP-49's table: at minlen 9 only **4.0% of nodes** sit in components
shorter than 9. Their pipeline already filters at minlen 6 and gap-closes fragments, so the final
graph has few short components left to remove. My offline model ran on PRE-FILTER geffs that
still contained them -- the same substrate error as EXP-34 and EXP-38, in a new place.

So OUTPUT_MIN_TRACK_LEN is near its ceiling at about +0.003, not because pruning stops paying but
because there is little left to prune. Nothing to submit from this run.

OUTPUT_KEEP_DIVISION_COMPONENTS=0 would prune division-bearing short components, but the local
proxy reads divJ 0.0000 (the released annotations on these four movies carry almost no
divisions), so we would be flying blind on a term worth 0.1 * divJ. Not worth it.

## tools/fastpp.py -- a local post-processing loop (2026-09-14)

Iteration was bottlenecked at ~3 hours per variant because every knob we tune sits AFTER the
U-Net but the only way to run it was a full Kaggle kernel. run_stats.csv shows the split:

    44b6_0113de3b   raw_nodes 25,822  ->  nodes 25,637

The cached .geff files are the RAW linker output. Everything between them and submission.csv --
gap closing, motion relink, gap2 recovery, safe divisions, DeepCenter gating, short-track
filtering, linefit smoothing -- is CPU Python in notebook cell 5. Cell 4 is the only GPU stage
and is skippable when raw predictions exist.

fastpp.py execs cells 0-3 and 5 locally. Six blockers had to clear: missing pandas / IPython /
deprecated wheels; BIOHUB_MODEL_ARTIFACTS; two repo integrity checksums; the DeepCenter and
secondary-seed checkpoints (both need env vars RE-ASSERTED after each cell, since the notebook
assigns os.environ itself); and the subtle one -- the support pack ships its OWN repo/predictions
holding unrelated sample stems, which shadowed ours so write_test_submission globbed the wrong
graphs. Cell 3 re-materialises the repo, so the symlink must be restored AFTER it runs.

Also: our local "50ep" support pack is actually the 400ep snapshot (Kaggle re-versioned the
dataset). Per-file checksums showed ONLY scripts/evaluate.py differs; every
src/biohub_tracking/*.py matched exactly.

### Fidelity: NOT byte-exact, bias +0.00046

                      local    kernel   diff
    44b6_0113de3b    25,622    25,637    -15
    44b6_0b24845f    20,709    20,721    -12
    6bba_05b6850b     6,152     6,152      0
    6bba_05db0fb1    70,262    70,284    -22
    score           0.89365   0.89319   +0.00046

22 nodes in 70,000 (0.03%), one dataset exact -- consistent with CPU-vs-GPU float differences
moving a few DeepCenter gate candidates across a threshold.

**Usage rule.** This is a SCREENING instrument, not a substitute for a kernel run. The bias is
below the leaderboard's 0.001 resolution and 7x smaller than the effects we chase, so it can rank
configs and reject bad ones; any winner is still confirmed with a real kernel run before being
submitted. Runtime ~34 min per config on CPU, dominated by DeepCenter heatmaps.

## ppgrid + minlen9: super-additive, +0.0082 (2026-09-14)

                        local test score   delta
    repro (baseline)         0.89319      +0.0000
    minlen9                  0.89638      +0.0032
    ppgrid alone             0.89629      +0.0031
    ppgrid + minlen9         0.90143      +0.0082   <- vs +0.0063 from summing

    stem               N_pred    mult   edgeJ     adj
    44b6_0113de3b      25,325  1.0017  0.8679  0.8694
    44b6_0b24845f      18,353  1.0440  0.9800  1.0232   (edgeJ was 0.9412 at baseline)
    6bba_05b6850b       5,875  1.0077  0.9685  0.9759
    6bba_05db0fb1      68,577  1.0018  0.8472  0.8487

Every stem improved. The mechanism for the super-additivity is visible in the sweep's own choice:
it re-selected **combo(tight55+vel025)** rather than the combo(tight55+vel025+tight50relax9) it
picked without minlen9. Pruning short components changes the landscape the in-kernel sweep
optimises over, so the two levers are not independent -- the pruning lets a different relink
configuration win.

Submitted as #31. Two submissions remaining today.

Note the multiplier is 1.0138, identical to minlen9 alone, so this gain is NOT multiplier -- it
is edge Jaccard, from the relink configuration. That makes it annotation-dependent and therefore
less certain to transfer than minlen9's was. The leaderboard decides.

### fastpp VALIDATION FAILED -- deltas do not transfer (2026-09-14)

                fastpp local     real kernel
    base          0.89365          0.89319
    minlen9       0.89466          0.89638
    delta        +0.00101         +0.00319     <- 3x under-read

Per-stem on 6bba_05db0fb1 (56.9% of the score):

    kernel  base 0.8455 -> minlen9 0.8417   edgeJ loss 0.0038
    fastpp  base 0.8463 -> minlen9 0.8386   edgeJ loss 0.0077

Node pruning is comparable (kernel -1,719, fastpp -1,770), so the divergence is in WHICH
components get pruned: the small CPU-vs-GPU differences in DeepCenter gating change which nodes
exist, and minlen then removes different components. The +0.00046 absolute bias was never the
issue -- the error COMPOUNDS through the pruning stage.

**fastpp is NOT fit for screening at the 0.002-0.003 scale.** A 3x error on effect size would
have ranked minlen9 as marginal when it is the best measured lever we have. It is shelved for
config screening. The GPU kernel remains the only trustworthy arbiter, and
tools/score_submission_local.py on a REAL kernel submission remains the right instrument.

Process note: this tool was described as capable three times in one session -- byte-fidelity,
then parallelism, then delta accuracy -- and measurement walked back each claim. The tool cost
roughly two hours and returned a negative result. The discipline that caught it (validate the
instrument against a case whose answer is already known) is the only reason it did not
contaminate a submission decision.

## Live state at end of 2026-09-14

**Submissions**

    #31  ppgrid + minlen9      PENDING   local +0.0082  (best measured)
    #30  minlen9               PENDING   local +0.0032
    #29  ppgrid combo          PENDING   local +0.0031
    #28  noSister              0.947     (null, byte-identical test submission to #27)
    #27  repro of the 0.947    0.947     (exact reproduction)

Two submissions remain for the day, deliberately held. #29/#30/#31 are a natural experiment:
they test whether `tools/score_submission_local.py` SIZES a gain correctly or only detects nulls
(it has passed the null test exactly). Spending the last two before that answer would be guessing.

**Kernels**

    biohub-ppgrid-ml12   RUNNING   joint minlen optimum above 9
    biohub-ppgrid-ml7    RUNNING   joint minlen optimum below 9
    biohub-rw-ml9        BUILT     embryo-reweighted sweep objective, waiting on a slot

**Shelved**

    tools/fastpp.py      3x delta under-read; GPU kernels remain the only arbiter
    biohub-det88         det94 evidence argued against lowering the detection threshold

**Experiments today:** EXP-36 through EXP-49. Every one closed a route or corrected a method
error; none opened a new pool. The gains came from two places neither of which was an experiment
in the usual sense -- discovering the test set is four movies we hold annotations for, and
discovering the score weights 6bba at 95.4%.

## EXP-50 -- divisions were closed on the WRONG mechanism (2026-09-14)

EXP-38b concluded divisions were closed: eight gate relaxations, divTP never moved, only false
positives added. That conclusion was right about the observation and wrong about the cause.

In `add_safe_divisions`:

    cands = [i for i in kids_frame if i not in incoming and i not in used_t]

**Only daughters with NO INCOMING EDGE are ever proposed.** A daughter already linked to some
other parent is excluded before any gate is evaluated, so no relaxation of max_um, symmetry_tau
or diverge_um could possibly reach it. The eight relaxations were testing a door already bricked
up.

Measured on their validator output (12 divisions) and on all 199 train graphs (151 divisions):

    their validator, 12 divisions        199 train graphs, 151 divisions
      PRESENT        0   0.0%              PRESENT       60  39.7%
      FREE           0   0.0%              FREE           9   6.0%
      CONTENDED     11  91.7%              CONTENDED     79  52.3%
      NO_DAUGHTER    1   8.3%              NO_DAUGHTER    3   2.0%

    of the 91 missed divisions on train:
      reachable by the gates (FREE)        9.9%
      STRUCTURALLY EXCLUDED (CONTENDED)   86.8%
      undetectable (NO_DAUGHTER)           3.3%

**The division route is open.** It needs a different mechanism from anything tried: allow
contended daughters as candidates and decide whether to STEAL them from their current parent.

The economics of a steal are unusually good. If the truth is P->Q and the prediction has X->Q,
then X->Q is already a false positive when Q matched a GT node. Replacing it wins three ways at
once -- edge TP+1, FP-1, FN-1 -- and adds a division on top. Recovering even half the contended
pool would take div_tp from 3 to ~7 of 12 on the validator, divJ 0.23 -> ~0.6, worth about
+0.0038 on the score before the edge gains.

The deciding signal is the open question. Geometry is the cheap first test. The public notebook
"A dividing nucleus gets smaller, not dimmer" (zhincez) supplies an independent one worth trying:
nucleus VOLUME drops ~0.27 around a division while PEAK brightness holds, and the drop begins
1-2 frames BEFORE the split, so it is predictive rather than descriptive. Their key methodological
point is that mean intensity in a fixed-radius ball is an ARTEFACT -- a fixed probe around a
smaller object contains more background, so the mean falls with nothing dimming. Measure peak and
half-max volume in the same box instead.

## EXP-51 -- contended steals are NOT decidable on geometry; divisions close properly

EXP-50 showed 86.8% of missed divisions are excluded from the candidate set because the daughter
already has a parent. This tests whether the steal can be decided geometrically.

    HEAD-TO-HEAD on true contended divisions
      true parent closer than the incumbent: 0/20 = 0.0%

    ALL steal candidates: 7,918,981   true 17   BASE RATE 0.0002%
      rule                            fires   true  precision
      d_PQ < d_XQ                    42,377      0     0.00%
      sym < 0.6                     384,186      4     0.00%
      sym < 0.3                     117,502      2     0.00%
      cos < 0.0                   3,192,314     13     0.00%
      diverge > 0                 2,919,617     13     0.00%
      ALL FOUR                        6,017      0     0.00%

**The true parent is never closer than the incumbent -- 0 of 20.** That is exactly why the linker
chose the incumbent. And a 0.0002% base rate against a 12-24% break-even needs ~60,000x lift, the
most extreme instance of the structural law measured in this campaign.

The "ALL FOUR" row is decisive: tightening to 6,017 candidates catches ZERO of the 17 true
divisions. Contended divisions do not look like divisions geometrically. The volume signal from
the public notebook would have to supply all 60,000x by itself, from an effect size of 0.27 with
overlapping distributions. It cannot.

**Divisions are closed** -- now on the correct mechanism and with a far better-supported number
than EXP-38b's gate sweeps provided.

## Audit: which closures were wrong?

| originally | actually | cause |
|---|---|---|
| EXP-34: OUTPUT_MIN_TRACK_LEN null on 199 train graphs | **+0.0032**, our best single lever | wrong substrate |
| EXP-38b: divisions closed by gate relaxation | conclusion right, **mechanism wrong** | gates were never the constraint |
| tools/fastpp.py: working screening instrument | under-reads deltas 3x | not validated before use |
| EXP-50: the division route is open | closed by EXP-51 within the hour | base rate 0.0002% |

**One genuine false negative recovered** -- min_track_len, which produced today's gains. Every
other closure survived re-examination. The single recurring cause is SUBSTRATE: measuring on the
wrong artifact (a different pipeline, a pre-filter stage, or an even embryo weighting). That is
now a standing warning at the top of this file and in RANK1-PLAN.

## Substrate audit part 2 -- EXP-31 revisited under the correct weighting

Systematic check of which experiments used which artifact:

    EXP-23..EXP-39   train_graphs   (WRONG pipeline -- over-predicts, ~187 native forks/dataset)
    EXP-40..EXP-45   validator      (their pipeline, 8 train stems)
    EXP-46..EXP-48   test geffs     (RAW stage, not the submission)
    EXP-49           submission     (correct artifact)

Fifteen experiments ran on train_graphs, the substrate that produced the one confirmed false
negative (EXP-34). EXP-31 is the one whose conclusion turned on a weighting rule we have since
shown to be wrong.

EXP-31 found the SWAP pool is structurally different from everything else: 2,887 GT edges have
their target assigned to a different parent, out of ~124k predicted edges -- a **2.3% base rate**,
15x richer than any other pool, needing only 8.1x lift rather than 150-320x. It was rejected
because "the min-across-embryos margin is nil: -0.0003 at 49%".

Re-run, per embryo:

                              top50   top200  top1000    (break-even 48.1%)
    logistic   held-out 44b6    44%     48%*     49%*
    logistic   held-out 6bba    46%     53%*     52%*
    grad-boost held-out 44b6    62%*    60%*     50%*
    grad-boost held-out 6bba    28%     48%      54%*

The MINIMUM is 49% (44b6). Weighted at the real 95.4% 6bba it is **52-53%**, clearing break-even
by 4-5 points. **The rejection was an artefact of the min-across-embryos rule**, which optimises
for the embryo worth 4.6% of the score.

But the value is small, and the original note oversold it. A correct swap fix moves the Jaccard
numerator +1 and the denominator -1; a wrong one does the reverse. At 52% precision on the top
1,000 of 2,887 that is +40 net edges across 199 graphs, about **+0.0011** -- at leaderboard
resolution, not the "+0.0128" the note quoted, which assumed fixing ALL 2,887 at 65%.

So: real signal, rejected for the wrong reason, worth ~+0.001 rather than ~+0.013. Worth a
submission only if cheaper levers run out, and it would need re-measuring on the real artifact
first since it was computed on train_graphs.

## Verification: is the visible test set the scored test set? (2026-09-14)

Prompted by the public EDA notebook (zhincez/label-eda), which states the hidden test films come
from an embryo appearing nowhere in training. That would invalidate everything measured today,
since our four test films carry the train embryo prefixes.

The host said on record (competition discussion, frozen in evidence/):

> "Hi, indeed there are two unique embryo_ids in the training set. You can assume the test sit is
> roughly similar in size, with no overlap in embryo_ids between train and test sets."

**That statement does not match the shipped data.** Checks run:

- `sample_submission.csv` requires exactly four datasets: 44b6_0113de3b, 44b6_0b24845f,
  6bba_05b6850b, 6bba_05db0fb1 -- both train embryo prefixes.
- All four exist in `data/images/train/` AND `data/images/test/` AND `data/train_geff/`.
- The test copies are FULL SIZE, not truncated placeholders: 436M and 516M, matching their
  train/ counterparts byte-for-byte in size. So they are not a sample that gets swapped at
  scoring time, which was the main risk.

What reconciles it: the released annotations on the four scored films are unusually sparse.
Expected at typical embryo density against released counts:

    44b6_0113de3b   0.8% x 25,755 =   206      released    52
    44b6_0b24845f   0.8% x 32,795 =   262      released    51
    6bba_05b6850b   9.7% x  6,362 =   617      released   861
    6bba_05db0fb1   9.7% x 69,800 = 6,771      released 1,229

Roughly 18-25% of expected. The organisers withheld most annotations on the scored films, which
is exactly why our local proxy reads 0.893 against a leaderboard 0.947.

**The embryo weighting is robust to this.** At typical density the mix is 44b6 6.0% / 6bba 94.0%,
against 4.6% / 95.4% measured from released counts. Same conclusion either way, so today's
decisions stand.

Unresolved: the host's "no overlap in embryo_ids" is contradicted by the shipped data and by
sample_submission. Treat it as inaccurate rather than as evidence of a hidden swap -- but if a
private rerun ever produces scores wildly out of line with public, this is the first thing to
re-examine.
