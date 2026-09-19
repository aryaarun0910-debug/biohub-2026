---
tags:
  - data
---

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
