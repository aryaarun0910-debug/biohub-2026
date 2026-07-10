# Rules gate — automated test-time adaptation

**Retrieved authenticated through Kaggle API:** 2026-07-10.
**Competition:** `biohub-cell-tracking-during-development`.

## Relevant clauses

Competition-specific Data Access and Use states that participants may use Competition Data “for any purpose, whether commercial or non-commercial,” subject to the Rules.

General Competition Entry Rule 3.4.b prohibits using information from “hand labeling or human prediction” of validation or test records.

Code Requirements state:

- CPU or GPU notebook runtime ≤12 hours;
- internet disabled for submission;
- freely/publicly available external data and pretrained models allowed;
- output must be `submission.csv`.

The Rules define Competition Data to include private and public test sets.

## Operational interpretation

Automated, self-supervised adaptation performed by the submitted notebook on the provided unlabeled test images is permitted on the face of these clauses. It uses Competition Data algorithmically and does not introduce hand labels or human predictions.

Allowed implementation scope:

- per-embryo normalization/calibration;
- cycle/path-consistency losses;
- pseudo-labels generated only by the model/algorithm;
- per-embryo scene-flow or motion fitting;
- resetting source weights between embryos.

Still prohibited or quarantined:

- manual annotation or human correction of test records;
- identifying a hidden crop and transferring its public-source trajectory labels;
- reconstructing private labels through submission scores;
- the known unmatched-fork division evaluator exploit;
- internet access during the submitted notebook rerun.

Keep this rules snapshot and the adaptation procedure in the winner provenance bundle. Organizer confirmation would reduce residual adjudication risk but is not a technical blocker under the current text.
