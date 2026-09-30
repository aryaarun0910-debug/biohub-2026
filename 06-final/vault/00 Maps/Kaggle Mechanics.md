---
tags:
  - moc
---

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
