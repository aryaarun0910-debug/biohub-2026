---
tags:
  - failed
---

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
