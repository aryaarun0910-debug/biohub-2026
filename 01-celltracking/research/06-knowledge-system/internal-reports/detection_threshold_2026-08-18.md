# Detection threshold: the T-class prize is 0 on 44b6 and mostly background on 6bba — 2026-08-18

Labels: **[MEASURED]** = computed this session from in-repo artifacts against the official
scorer. **[CODE]** = read off source, `file:line` given. **[INFERENCE]** = derived, assumptions
stated. Companion: `research/06-knowledge-system/internal-reports/operating_point_2026-08-18.md`
(L3, the lever this report adjudicates).

---

## 0. HEADLINE — the recoverable-node census, per family

The question the whole lever rests on: **of the GT nodes we miss, how many are class T** —
unmatched, with an *unaccepted* local maximum inside the scorer's 7 µm radius, i.e. recoverable
by lowering `det_threshold` alone (`src/biotrack/d1_partition.py:11-14, 56-62`)?

| family | GT nodes (pilot) | unmatched | **T** | **T as % of GT** | **T as % of misses** | C | L | D |
|---|---|---|---|---|---|---|---|---|
| **44b6** (fold 0, split-0 ckpt) | 2,366 | 263 | **0** | **0.00 %** | **0.00 %** | 168 (63.9 %) | 95 (36.1 %) | 0 |
| **6bba** (fold 1, split-1 ckpt) | 9,604 | 2,431 | **1,098** | **11.43 %** | **45.17 %** | 736 (30.3 %) | 597 (24.6 %) | 0 |

**[MEASURED]** — `_evidence/derived/p3_d1_pilot_f0_v1/d1_derived_split0.parquet` and
`.../p3_d1_pilot_f1_v1/d1_derived_split1.parquet`, deployment routing only (each family read
under the checkpoint that held it out; `d1f_probe.py:145`).

On **44b6 the prize is exactly zero, not merely small.** Not one of its 263 missed GT nodes has
an unaccepted local maximum within 7 µm. Stronger: across all 2,366 GT rows the mean count of
*unaccepted* maxima within 7 µm is **0.001** (2 rows out of 2,366). Every local maximum near a
44b6 GT node is already accepted at 0.96875. There is no threshold in `[0, 0.96875)` that returns
a single 44b6 node.

On **6bba the raw T count looks large — and most of it is background.** See §1.2: under a
crop-matched null, over half of the "recoverable" T rows at any threshold are chance background
maxima, and the T mass is concentrated in one pathological crop.

### 0.1 The two numbers that decide the lever

**[MEASURED]** Break-even exchange rate, from the exact pooled identity (§2.1), and the measured
rate the detector actually offers:

| fold | **nodes per recovery to break even** | **nodes per genuine recovery, measured at T=0.5** | verdict |
|---|---|---|---|
| 0 (44b6) | 2,853 | **∞** (0 recoveries at any threshold) | strictly negative |
| 1 (6bba) | 493 | 474 on the pilot; **811** after correcting the pilot's 1.71× miss-rate bias | negative |

The lever is under water on both embryo directions, and on fold 0 it is under water by an
unbounded margin.

---

## 1. Substrate, coverage, and what the T-class does and does not mean

### 1.1 Coverage and limitations — stated honestly **[MEASURED]**

The D1 pilot is a **crop subset, not a frame subset**. Every audited crop is audited on **all
100 of its frames** (`grid_zyx = [64,64,64]`, `n_frames = n_frames_total = 100` for all 19 crops,
read from `_evidence/derived/p3_d1_pilot_factorial_v1/basis_*/d1_manifest.json`). So within a
crop there is no temporal sampling gap.

| | fold 0 (44b6) | fold 1 (6bba) |
|---|---|---|
| crops in pilot / in fold | **9 / 71** | **10 / 128** |
| share of fold `N_est` held by pilot crops | 10.7 % | 9.5 % |
| share of scoring weight `W` held by pilot crops | 11.7 % | 8.6 % |
| scorer `node_recall`, pilot crops | 0.9835 | **0.7698** |
| scorer `node_recall`, non-pilot crops | 0.9843 | **0.8737** |
| **pilot miss-rate bias** | **1.04× (representative)** | **1.71× (pilot is much harder)** |

**The fold-0 pilot is representative; the fold-1 pilot is not.** Its crops carry 1.71× the fold's
miss rate, so every T-class count on 6bba is an **over-estimate** of what the fold offers. This
cuts in the lever's favour when the answer is "zero" (fold 0) and against it when the answer is
"some" (fold 1) — i.e. the bias runs the wrong way for the hypothesis in both cases.

Note the pilot's own `d1_class` uses **pregraph** matching (`match_authority: "pregraph"`), which
is the correct authority for a *detection* question: it isolates the detector from graph
construction. Pooled pipeline `node_recall` (`metrics.py:532`, an unweighted crop mean) is
0.9842 / 0.8656, consistent with the host's 0.9871 / 0.8695.

### 1.2 The T-class is a NECESSARY but very weak condition — and this is the crux **[MEASURED]**

`_islm = (_lgc == _plc)` (`scripts/kaggle_edits/d1_response_audit.py:578`) is an **unthresholded**
local-max mask, so T-class membership is the full-range upper bound on threshold recoverability.
It is also nearly free to satisfy by chance, because local maxima are dense:

| | local maxima / frame | accepted / frame | subthreshold / frame | **null E[subthr in a 7 µm sphere]** |
|---|---|---|---|---|
| 44b6 | 1,270.3 | 406.4 | 863.9 | **1.103** |
| 6bba | 1,112.6 | 263.2 | 849.4 | **1.085** |

(A 7 µm sphere is 334.8 downsampled voxels — level-0 scale `(1.625, 0.40625, 0.40625)` ×
`downsample=[1,4,4]` — of a 64³ = 262,144-voxel frame, i.e. 0.128 %.)

Observed mean unaccepted-maxima count within 7 µm of a **T-class** GT node on 6bba is **1.513**,
against a uniform null of **1.085** — only **1.39×** chance. A T-class GT node's neighbourhood
looks very nearly like background. By contrast M and C nodes are strongly *depleted* of
unaccepted maxima (0.100 and 0.124, i.e. 0.09–0.11× the null): where the detector has committed,
the 3 µm max-pool leaves only accepted peaks.

**Crop-matched null test.** For each T row, draw its own number of unaccepted maxima at random
from *its own crop's* measured subthreshold-probability sample and take the max — what a GT node
with no detector response would look like. 300 replicates:

| threshold | observed T rows recoverable | null mean | null p95 | **genuine excess** | excess as % of GT |
|---|---|---|---|---|---|
| 0.95 | 24 | 17.4 | 25.0 | **6.6** (inside the null band — **not significant**) | 0.07 % |
| 0.90 | 66 | 44.4 | 54.0 | 21.6 | 0.23 % |
| 0.85 | 90 | 55.9 | 68.0 | 34.1 | 0.35 % |
| 0.80 | 110 | 63.6 | 77.0 | 46.4 | 0.48 % |
| 0.70 | 144 | 75.7 | 89.0 | 68.3 | 0.71 % |
| 0.60 | 168 | 87.9 | 101.0 | 80.1 | 0.83 % |
| **0.50** | 181 | 96.6 | 112.0 | **84.4** | **0.88 %** |
| 0.30 | 218 | 113.8 | 131.0 | 104.2 | 1.08 % |
| 0.01 | 437 | 198.3 | 218.0 | 238.7 | 2.49 % |

**Reading: "T-class" over-states the prize by roughly 2× at every threshold**, and near the
deployed operating point (0.95) the entire effect is inside the null band. The honest recoverable
population on the (already 1.71×-inflated) 6bba pilot is **84 GT nodes out of 9,604 at T = 0.5**.

### 1.3 The T mass is one crop, and that crop is not threshold-limited **[MEASURED]**

| crop | n_T | median `best7_prob` of the T rows | frac > 0.5 | scorer node_recall |
|---|---|---|---|---|
| **6bba_6feb10f0** | **810 (73.8 % of all T)** | **6.08e-04** | 0.090 | 0.1133 |
| 6bba_fc83837d | 95 | 7.54e-02 | 0.326 | 0.7367 |
| 6bba_3db54e20 | 70 | 2.72e-02 | 0.143 | 0.6733 |
| 6bba_57b7cc1e | 39 | 5.36e-01 | 0.538 | 0.7920 |
| all others (6) | 84 | 0.24–0.77 | 0.39–0.72 | 0.90–0.97 |

Three-quarters of the 6bba T-class comes from one crop whose T-row maxima sit at a median
probability of **6e-4** — logit ≈ −7.4, against the deployed logit 3.434
(`scripts/d1/d1f_probe.py:133-134`). That is not a mis-set operating point; the detector emits
essentially nothing there. And it is **not globally cold**: `6bba_6feb10f0` accepts 138.3
peaks/frame, mid-pack among its 10 siblings (range 110.6–1028.6). It produces plenty of peaks —
they are in the wrong places. **[INFERENCE]** Its 810 T rows are functionally class **D**
(response-poor) wearing a T label, because a background local maximum happened to fall inside
7 µm. No threshold recovers them; only a different representation would.

### 1.4 Subthreshold maxima are strongly depleted near annotated cells **[MEASURED]**

Directly, from the per-frame uniform-random subthreshold sample
(`d1_response_audit.py:724-727`, `choice(..., replace=False)`, so unbiased):

| family | subthr rows sampled | **expected within 7 µm of an annotated GT (uniform null)** | **observed** | depletion |
|---|---|---|---|---|
| 44b6 | 28,800 | 97 | **0** | **≥ 97×** |
| 6bba | 32,000 | 393 | **93** | **4.2×** |

Whatever else lowering the threshold admits, it is admitting it **away from the cells the metric
scores**. On 44b6, of 28,800 sampled subthreshold local maxima, **zero** lie within 7 µm of any
annotated GT node. This is the independent confirmation of the T = 0 result in §0, arrived at
from the peak side rather than the GT side.

---

## 2. The end-to-end trade-off model

### 2.1 An exact pooled identity, not an approximation **[CODE] + [MEASURED]**

From `metrics.py:444-449` and `:499-504`, with `w_i = TP_i + FP_i + FN_i` and
`m_i = 1 − 0.1·(N_i − N_est,i)/N_est,i`:

```
pooled adj_edge_jaccard  =  Σ_i TP_i · m_i  /  Σ_i w_i
```

because `w_i · adjJ_i = w_i · (TP_i/w_i) · m_i = TP_i · m_i`. Verified against the official
scorer on both folds (`scripts/core/score_loeo_submission.py`, per-crop rows re-derived this
session from `c:/temp/subvoxel_f{0,1}/loeo_split{0,1}_strict.csv.gz`; my fold-0 summary is
byte-identical to the sibling's):

| fold | `Σ TP·m / W` | scorer `adj_edge_jaccard` |
|---|---|---|
| 0 | **0.901802** | **0.901802** |
| 1 | **0.703756** | **0.703756** |

Fold aggregates **[MEASURED]**:

| fold | W | TP | FP | FN | N_pred | N_est | N_pred/N_est |
|---|---|---|---|---|---|---|---|
| 0 | 21,210 | 18,746 | 1,384 | 1,080 | 1,842,800 | 2,618,970 | 0.7036 |
| 1 | 123,413 | 86,194 | 14,356 | 22,863 | 1,957,978 | 2,106,147 | 0.9296 |

Note `W` = 21,210 against **1.77 M predicted edges**: only ~1.2 % of our fold-0 edges are
scoreable at all. `pred_valid = out_valid(source) OR in_valid(target)`
(`metrics.py:193-194`) — an edge counts only if an endpoint **matched an annotated GT node**.
This is decisive for the cost side and is why the naive "40,000 junk nodes ⇒ 80,000 FP edges"
accounting is wrong.

### 2.2 Marginal prices **[MEASURED]**

Differentiating the identity:

| event | fold 0 price (pooled adjJ) | fold 1 price |
|---|---|---|
| +1 predicted node, anywhere (budget only) | −3.375e−08 | −3.316e−08 |
| +1 recovered GT node that links both ways (+2 TP, −2 FN, `W` unchanged) | +9.628e−05 | +1.633e−05 |
| **break-even ratio** | **2,853 nodes per recovery** | **493 nodes per recovery** |

The budget charge is **not** waived by our sitting under `N_est`: `m` is linear in `N_pred`, so
`∂m/∂N_pred = −0.1/N_est` is constant. Under-producing raises the *level* of `m` (we collect a
bonus, worth +0.018 / +0.005 per `operating_point_2026-08-18` §3.3) but never makes a marginal
node free. Lowering the threshold hands part of that bonus back at a fixed rate.

### 2.3 Steal risk is real but negligible **[MEASURED]**

The host flagged that a new FP node can steal a GT match under the scorer's one-to-one
per-frame matching. Measured directly from the per-GT k-list (`k0..k7` dist/prob/accepted): for
each already-matched GT node, does lowering the threshold introduce an unaccepted maximum inside
7 µm that is **nearer** than the current incumbent?

| threshold | 44b6 thefts / 2,103 matched | 6bba thefts / 7,173 matched |
|---|---|---|
| 0.90 | **0** | 9 (0.13 %) |
| 0.50 | **0** | 15 (0.21 %) |
| 0.10 | **0** | 16 (0.22 %) |

At 15 events against 84 recoveries on fold 1, and exactly zero on fold 0, stealing is a
second-order term. **This partially exonerates the lever** — the cost is not theft, it is the
node budget, and the shortfall is on the gain side.

### 2.4 Predicted score-versus-threshold curve **[MEASURED] + [INFERENCE]**

Two scalings from pilot to fold: **volume-like** (new peaks scale with `N_est` share) and
**miss-like** (recoveries scale with the fold's miss rate, correcting the fold-1 pilot's 1.71×
bias). "Optimistic" assumes the pilot's elevated miss rate holds fold-wide; "honest" applies the
correction. **Both columns assume `λ_link = 1`** — every recovered node links correctly in both
temporal directions. That is an upper bound and cannot be improved on.

**Fold 0 (44b6)** — zero recoveries at every threshold, so gain is identically zero:

| t | new nodes (fold) | Δ budget | Δ gain | **NET** |
|---|---|---|---|---|
| 0.95 | 189,694 | −0.00640 | +0.00000 | **−0.0064** |
| 0.90 | 339,732 | −0.01147 | +0.00000 | **−0.0115** |
| 0.80 | 421,571 | −0.01423 | +0.00000 | **−0.0142** |
| 0.50 | 581,460 | −0.01962 | +0.00000 | **−0.0196** |

**Fold 1 (6bba)**:

| t | new nodes (fold) | recoveries (opt / honest) | Δ budget | Δ gain (opt / honest) | **NET opt** | **NET honest** |
|---|---|---|---|---|---|---|
| 0.95 | 77,801 | 68 / 40 | −0.00258 | +0.00111 / +0.00065 | −0.0015 | **−0.0019** |
| 0.90 | 175,123 | 230 / 134 | −0.00581 | +0.00376 / +0.00219 | −0.0021 | **−0.0036** |
| 0.80 | 265,473 | 488 / 285 | −0.00880 | +0.00798 / +0.00466 | −0.0008 | **−0.0042** |
| 0.70 | 325,706 | 721 / 421 | −0.01080 | +0.01177 / +0.00687 | +0.0010 | **−0.0039** |
| 0.60 | 375,064 | 846 / 494 | −0.01244 | +0.01381 / +0.00806 | +0.0014 | **−0.0044** |
| 0.50 | 420,518 | 891 / 520 | −0.01394 | +0.01455 / +0.00849 | +0.0006 | **−0.0055** |

**The curve has no useful peak.** Fold 0 is monotone decreasing over the whole range. Fold 1's
*optimistic* branch peaks at **+0.0014 at t ≈ 0.60** — a quarter of its own ±0.004 noise floor,
i.e. unmeasurable even if real — and the honest branch is negative everywhere, minimum −0.0019 at
t = 0.95. Every entry in both tables is either negative or below the noise floor.

**Reconciliation with the sibling's `p* ≈ 0.50` [INFERENCE].** The break-even derivation prices a
node *conditional on landing in annotated territory*. Measured conditional precision at T = 0.5 on
6bba: **429** new peaks land within 7 µm of an annotated GT node, of which **84** are genuine
recoveries — **19.6 %**, against the required `p* = 0.41` for fold 1. The two framings agree, and
they agree on the sign. `p* ≈ 0.5` correctly killed "the node budget is why we sit at 0.97"; it
was never a prediction that 0.5 scores better, and it does not.

### 2.5 The mirror arm: **raising** the threshold **[MEASURED]**

The same superset makes `t > 0.96875` replayable too, and §1.4 implies it is the more natural
direction on 44b6 (nodes there are nearly pure budget cost). Measured:

| t | 44b6: GT matches lost / nodes removed / **NET** | 6bba: GT matches lost / nodes removed / **NET** |
|---|---|---|
| 0.98 | 0 / 23,162 / **+0.00078** | 27 (0.38 %) / 19,202 / **−0.0040** |
| 0.99 | 0 / 46,324 / **+0.00156** | 107 (1.49 %) / 38,404 / **−0.0171** |
| 0.995 | 2 / 75,277 / +0.00074 | 205 (2.86 %) / 99,851 / −0.0319 |
| 0.999 | 223 (10.6 %) / 555,894 / −0.1821 | 684 (9.54 %) / 303,394 / −0.1073 |

Matched-GT peak probabilities are saturated: 44b6 q01 = 0.99754, 6bba q01 = 0.98492. Raising to
0.99 costs 44b6 nothing and buys **+0.0016** — **half its ±0.003 noise floor, so unmeasurable** —
while costing 6bba **−0.0171**, far above its floor. A global raise is clearly net negative; a
per-family raise buys an unmeasurable amount on the fold that carries only 14.7 % of the edge
weight. Both arms are dead.

### 2.6 What genuinely cannot be predicted without the export **[INFERENCE]**

Stated plainly, so this is not read as more certain than it is:

1. **`λ_link`** — whether a recovered node actually acquires two correct edges. Not measurable
   from these artifacts. I set it to **1** (maximally favourable). The true value is below 1 and
   probably well below it for marginal nodes, whose temporal neighbours are likely also missing.
   Lowering it only worsens every number above.
2. **Second-order edge effects** of a denser candidate set on the transformer's `n_src × n_tgt`
   assignment (`predict_unet_transformer.py:449-465`) — the linker sees a different candidate
   population and may re-route existing good edges. Sign unknown, magnitude unknown.
3. **Division term.** `SCORE = adjJ + 0.1·div_jaccard`; fold 0 already sits at
   `div_tp/fp/fn = 2/106/24`, `div_jaccard = 0.0152`. More candidate nodes plausibly add division
   FPs on annotated cells. Second-order and **negative**; not modelled.
4. **Fold-1 pilot extrapolation.** The 1.71× miss-rate correction is a first-order adjustment on
   9.5 % of the fold, not a measurement of the other 118 crops.

None of these can rescue the result, because **the optimistic bound — `λ_link = 1`, no theft, no
division penalty, pilot miss rates assumed fold-wide — is already ≤ +0.0014 on fold 1 and
≤ −0.0064 on fold 0.**

---

## 3. Export design review — the patch is correct, and cheaper than advertised

**Build only. Nothing was pushed, nothing was submitted, no GPU was launched.**

### 3.1 Build verification **[MEASURED]**

Both specs build clean through `scripts/core/kaggle_factory.py build`, which asserts exact match
counts per edit; exit 0 on both:

| spec | built artifact | result |
|---|---|---|
| `scripts/kaggle_specs/p4_detsweep_export_f0.json` | `notebooks/kaggle_p4_detsweep_export_f0/biohub-p4-detsweep-export-f0.ipynb` | **builds, all `expect: 1` satisfied** |
| `scripts/kaggle_specs/p4_detsweep_export_f1.json` | `notebooks/kaggle_p4_detsweep_export_f1/biohub-p4-detsweep-export-f1.ipynb` (sha256 `0a57bc8d…`) | **builds, all `expect: 1` satisfied** |

### 3.2 The graph really is bit-identical to `p3_base` **[CODE] + [MEASURED]**

Diffing the specs against their paired baselines `p3_base_loeo_f{0,1}.json`:

- identical `base_notebook` and `base_sha256` (`01408a17…`);
- **identical env block** — zero differing variables, so `BIOHUB_DET_THRESHOLD` stays `"0.96875"`
  (confirmed present in the built notebook);
- 18 → 20 edits: exactly **two additive** edits, the `detpeak_export.py` cell insert and widening
  `_LOEO_KEEP` to retain `detpeaks` / `detpeak_manifest.json`.

At code level the patch replaces
`is_peak = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)` with
`is_peak = _local_max & (_sig > det_threshold)` where `_local_max = (logits == pooled)` and
`_sig = torch.sigmoid(logits)` — algebraically the same expression, same operands, same order.
The export block is a `try/except`-wrapped side effect on a separate tensor that never touches
`is_peak`. **The emitted graph is bit-identical and the export is purely observational.** The
monotonicity argument it relies on is sound: the `logits == pooled` test at
`predict_unet_transformer.py:285` is threshold-free, so the T = 0.5 peak set is a strict superset
of every higher threshold.

### 3.3 Size — the sibling's estimate is ~6× too high **[MEASURED]**

The sibling assumed a "3–6× superset" and projected 80–155 MB raw / 30–60 MB gzipped per fold.
I can now measure the ratio rather than guess it, from accepted and subthreshold peak densities:

| fold | accepted/frame | subthr/frame | frac subthr > 0.5 | **superset / deployed** | peaks | raw | gzipped |
|---|---|---|---|---|---|---|---|
| 0 (44b6) | 406.4 | 863.9 | 0.0799 | **1.170×** | ≈ 2.16 M | 25.9 MB | **≈ 12–16 MB** |
| 1 (6bba) | 263.2 | 849.4 | 0.0471 | **1.152×** | ≈ 2.26 M | 27.1 MB | **≈ 12–16 MB** |

The local-max constraint caps growth far harder than assumed: **the whole `[0.5, 1.0]` superset is
only ~15–17 % larger than what we already emit.** That is comfortably inside the artifact budget
(33 MB gzipped CSVs today), and it also means the flood available to the lever is bounded — which
is itself part of why the lever cannot pay.

### 3.4 What the CPU replay would buy

Filter the peak table at any `t ∈ [0.5, 1.0]`, rebuild the graph, re-link, re-score — no further
GPU, unlimited thresholds, both directions, with `edge_tp` and `N_pred/N_est` recorded jointly at
each point (which also settles L5 for free). That is a genuinely well-built instrument. My
finding is not that the instrument is wrong; it is that **the artifacts we already hold answer its
question**, and the answer is negative.

---

## 4. Recommendation

**Do not spend the GPU export, and do not spend a submission slot. Ship `det_threshold = 0.96875`
unchanged.**

The lever's premise — that it changes the detection surface rather than permuting edges, and so
belongs to the class that should transfer where arm B did not — is **correct**. It fails on
magnitude and sign, not on class:

1. **44b6 offers literally nothing.** T = 0 of 263 misses; 0 of 28,800 sampled subthreshold
   maxima within 7 µm of an annotated cell; 0 thefts. Every threshold below 0.96875 is pure
   budget loss, −0.0064 to −0.0196.
2. **6bba's prize is roughly half background** (crop-matched null), **three-quarters concentrated
   in one crop whose maxima sit at p ≈ 6e-4**, and measured on a pilot with **1.71× the fold's
   miss rate**. Corrected, the exchange rate is 811 nodes per recovery against a break-even
   of 493.
3. **The optimistic bound is already below the noise floor.** Best case anywhere in the sweep is
   **+0.0014 on fold 1 at t ≈ 0.60**, against a fold-1 floor of ±0.004 — and that branch assumes
   `λ_link = 1`, ignores the division penalty, and grants the pilot's inflated miss rate to the
   whole fold. Per the measurement warning: **this lands below ~0.005 on a single fold and is
   therefore unmeasurable**, quite apart from being negative under honest scaling.
4. The mirror arm (raising the threshold) is **+0.0016 on fold 0 — half its own noise floor — and
   −0.0171 on fold 1**, so it is dead too.

**If the export is run anyway** (it is cheap, ~12–16 MB per fold, graph provably unchanged, and it
would convert every number in §2 from inference to measurement), then run it as an *audit of this
report*, not as a search for a shipping threshold — and pre-register the criteria below rather
than sweeping for a maximum.

### Falsification test

Replay the exported table at `t ∈ {0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.94, 0.96875, 0.98, 0.99}` on
**both** folds, recording `edge_tp`, `N_pred/N_est` and `node_recall` jointly at each point.

**This report is falsified iff, on the fold-1 *full* 128-crop substrate, some `t < 0.96875`
delivers a paired Δ(score) ≥ +0.005 while fold 0 is not worse than −0.002.** That requires the
measured exchange rate to beat 493 nodes per recovery fold-wide — §2.4 predicts 811, so the
prediction is sharp and it is refutable by a single replay.

Two auxiliary predictions that the same replay tests, and which would each independently break the
analysis if they failed:

- **P1**: fold-0 score is **monotone decreasing** as `t` falls, with no local maximum anywhere in
  `[0.5, 0.96875)`. (T = 0 leaves no other possibility.)
- **P2**: fold-1 `edge_tp` rises by **less than 1 per 200 additional predicted nodes** across
  `t ∈ [0.5, 0.96875)`. (§2.4 predicts ~1 per 400 counting both edges of a recovery.)

### Kill criterion

**Already met — the lever is CLOSED as of this report**, on the artifacts in hand, with no GPU
spend. Formally: kill if the honest-scaled predicted gain is below the fold's noise floor on both
folds, which it is (fold 0 strictly negative at every threshold; fold 1 maximum +0.0014
optimistic / negative everywhere honest, against ±0.004).

Re-opening requires a **new mechanism**, not a new threshold. Per
`CLAUDE.md` and `research/06-knowledge-system/failed-experiments.md`, the two mechanisms this
analysis leaves genuinely open — and neither is a thresholding lever — are:

- **The L-class**, which this report did not chase: 95 nodes on 44b6 (36.1 % of its misses) and
  597 on 6bba (24.6 %) have **no local maximum at all within 7 µm but one within 15 µm**. That is
  a *localisation* deficit, not an acceptance deficit, and it is larger on 44b6 than anything
  thresholding touches. It is unreachable by any operating point.
- **`6bba_6feb10f0`-class crops**, where the detector emits ~138 peaks/frame but matches 11 % of
  GT. That is a representation/domain failure and points at the retrain lane, not at
  `det_threshold`.

---

## Appendix — reproduction

- T-class census + admission rates: `<scratchpad>/tclass.py` over
  `_evidence/derived/p3_d1_pilot_f{0,1}_v1/d1_derived_split{0,1}.parquet` joined to
  `_evidence/derived/p3_d1_pilot_factorial_v1/basis_{0,1}/*__rows.parquet` on `(dataset, row_id)`.
  Deployment routing only: 44b6 ← basis_0, 6bba ← basis_1 (`d1f_probe.py:145`).
- Crop-matched background null: `<scratchpad>/null.py`, 300–400 replicates, per-crop
  subthreshold-probability pools, seed 20260818.
- Steal risk from the per-GT k-list: `<scratchpad>/steal.py`.
- Pooled identity, marginal prices, pilot-representativeness: `<scratchpad>/model.py`;
  score curve both directions: `<scratchpad>/curve.py`.
- Per-crop scorer rows: `.venv/Scripts/python.exe scripts/core/score_loeo_submission.py --csv
  c:/temp/subvoxel_f{0,1}/loeo_split{0,1}_strict.csv.gz --gt-dir data/train --json-out …`
  → fold-0 summary reproduces the sibling's `f0_base.json` **byte-identically**
  (`score = 0.9033175059980829`), confirming identical substrate and scorer.
- Builds (build only, never pushed): `scripts/core/kaggle_factory.py build --spec
  scripts/kaggle_specs/p4_detsweep_export_f{0,1}.json`.
