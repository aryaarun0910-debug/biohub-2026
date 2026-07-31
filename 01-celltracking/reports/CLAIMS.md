# Claims table (generated -- do not edit by hand)

**Generated:** 2026-07-31 by `scripts/claims_table.py`

Every number below is read from an artifact at generation time. If an artifact moves or a
path stops resolving, this file fails to build rather than asserting a stale value. Hand-
editing it defeats the entire point -- change the artifact, then regenerate.

**Basis tags.** `public` = a leaderboard score we received - `exact-pooled-OOF` = exact
patched scorer, 199-crop corpus, pooled - `cross-family-LOFO` = fitted on one family,
evaluated on the other - `in-family-CV` = within one family (WEAKEST, inflates) -
`placeholder-proxy` = the four visible movies (in-sample, biased) - `GT-oracle` = uses
ground truth in the decision, so it is a CEILING and never a candidate.


## Deployment

| claim | value | basis | note |
|---|---:|---|---|
| P0-B public (deployment base) | *(see note)* | `public` | NOT ARTIFACT-BACKED - 0.914 - recorded in reports/submissions/CANDIDATE_LEDGER.md, ref 55136908 |
| P0-A public (clean 0.913 reproduction) | *(see note)* | `public` | NOT ARTIFACT-BACKED - 0.913 - ref 55136759 |
| v122 public | *(see note)* | `public` | NOT ARTIFACT-BACKED - 0.908 - ref 54854143 |
| E0c public (scientific anchor) | *(see note)* | `public` | NOT ARTIFACT-BACKED - 0.889 |

## P0 substrate (fold 0 / 44b6)

| claim | value | basis | note |
|---|---:|---|---|
| adj_edge_jaccard | `0.89859` | `exact-pooled-OOF` | 71 crops, arm strict. NOT comparable to E0c 0.7595 until reconciled in one pass. |
| node_recall | `0.98457` | `exact-pooled-OOF` |  |
| division_jaccard | `0.01587` | `exact-pooled-OOF` | TP 2 / FP 100 / FN 24 - its own division layer is near-worthless here |
| reachable GT divisions | `22` | `exact-pooled-OOF` | of 26. Beats E0c 20/26, clean903 20/26, v122 15/26. 6bba UNMEASURED on this substrate. |

## Node budget (CLOSED)

| claim | value | basis | note |
|---|---:|---|---|
| arm D (v122) delta @ keep_frac 0.975 | `-0.00088` | `exact-pooled-OOF` | 199 crops. Pooled optimum is keep_frac 1.00 = no pruning. |
| arm D node ratio @ 1.0 | `-0.1598` | `exact-pooled-OOF` | under-predicts; parity-exact against the published v122 anchor |
| P0-B delta @ keep_frac 0.975 | `-0.0000103` | `placeholder-proxy` | proxy is BIASED IN FAVOUR of the mechanism and it still fails |
| P0-B node ratio | `-0.10282` | `placeholder-proxy` |  |
| E0c arm A delta @ keep_frac 0.975 (same 4 movies) | `+0.006339` | `placeholder-proxy` | the sign flips with the sign of the node ratio |

## ssl x geometry gate (CLOSED)

| claim | value | basis | note |
|---|---:|---|---|
| pooled delta | `+0.001042` | `exact-pooled-OOF` | headline was +0.00141, a projection; exact scorer says this |
| 44b6 delta | `+0.003153` | `exact-pooled-OOF` |  |
| 6bba delta (min-fold) | `+0.000097` | `exact-pooled-OOF` | P(gain) = 0.52, a coin flip, against a +0.005 bilateral gate |
| baseline parity: 44b6 | `0.759549` | `exact-pooled-OOF` | reproduces the published 0.759549 anchor - this is what makes the delta trustworthy |
| baseline parity: 6bba | `0.648965` | `exact-pooled-OOF` | reproduces the published 0.648965 anchor |

## Association FN attribution

| claim | value | basis | note |
|---|---:|---|---|
| baseline pooled | `0.6654043` | `exact-pooled-OOF` |  |
| total edge FN | `27705` | `exact-pooled-OOF` | 43.80% never detected; 43.32% detected then discarded |

## Division economics

| claim | value | basis | note |
|---|---:|---|---|
| H0c full oracle pooled delta | `+0.06012` | `GT-oracle` | CEILING, not a candidate - fork choice uses ground truth |
| suppress-all alone | `-0.001728` | `exact-pooled-OOF` | negative alone; the gain is entirely in reconstruction |
| delta at 1 FP per true fork | `+0.03618` | `GT-oracle` | 92 true / 92 false |
| delta at 5 FP per true fork | `+0.01127` | `GT-oracle` | 88 true / 460 false |
| delta at 10 FP per true fork | `+0.00313` | `GT-oracle` | 88 true / 912 false |
| delta at 25 FP per true fork | `-0.007762` | `GT-oracle` | NEGATIVE - this is where a weak selector lands |
| precision required for +0.005 pooled | `0.10153` | `exact-pooled-OOF` | the 10.15% break-even, AT FULL RECALL; the marginal threshold is what a selector needs |
| precision required for +0.002 pooled | `0.07967` | `exact-pooled-OOF` |  |

---

28 claims from 6 artifacts under `reports/inventory/`.
