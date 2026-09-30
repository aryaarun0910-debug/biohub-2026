# Final results

Pulled from the Kaggle API on 2026-09-30, the day after the competition closed.

| | |
|---|---|
| Final (private) rank | **1488 / 4017** teams |
| Public rank at close | 1224 / 4017 (0.947) |
| Best private score, any submission | **0.914** |
| Best public score | 0.947 |
| Winning private score | 0.977 |
| Submissions | 34, all complete ([submissions.csv](submissions.csv)) |

The Kaggle API does not say which two submissions were selected for final
scoring, so the rank above is Kaggle's own figure rather than one derived here.

## By campaign

| submissions | campaign | best public | best private |
|---|---|---|---|
| 1 – 25 | [01-celltracking](../01-celltracking/) | 0.932 | 0.908 |
| 26 | [03-biohub-x](../03-biohub-x/) | 0.496 | 0.477 |
| 27 – 31 | [05-biohub](../05-biohub/) | 0.947 | 0.914 |
| 32 – 34 | [06-final](../06-final/) | 0.946 | 0.913 |

The 0.947 / 0.914 base (submission 27) is a byte-identical reproduction of the
public notebook `sjlee101/biohub-lf-dctta`. None of the seven changes submitted on
top of it improved the private score: they landed between 0.910 and 0.914.

## Public vs private

The gap between public and private widened as the work went on. It was about
0.007 in mid-July (submissions 3 – 5) and 0.032 – 0.034 for every submission from
13 September onwards, so public-leaderboard gains in the last month mostly did not
carry over to the private set.

The final sprint's pre-registered instrument test held up. Submission 34 (s08)
was predicted to score the same as s05, because the four scored films have no
recoverable division. It did: 0.946 / 0.913 against 0.945 / 0.913.
