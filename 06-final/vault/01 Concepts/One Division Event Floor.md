---
tags:
  - concept
---

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
