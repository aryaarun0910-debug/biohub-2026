---
tags:
  - finding
---

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
