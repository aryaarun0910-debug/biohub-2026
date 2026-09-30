---
tags:
  - failed
---

# Failed: learned division classifier

Trained on frozen [[Temporal UNet3D Detector]] features to predict whether a node
is about to divide.

**AUC 0.456 at the split frame — worse than chance.** Nothing tried clears 0.70.

The features simply do not carry the signal. This is the strongest evidence that
[[Divisions Played Out]] is a *detection* limit, not a ranking one.

Related: [[Failed Anaphase Hypothesis]]
