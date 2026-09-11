# Review validation

The maintenance review is covered by `tests/test_knowledge.py` and the
`Knowledge integrity` GitHub Actions workflow. Run:

```bash
python3 -m unittest discover -s tests -v
python3 tools/sync_source_rows.py check
bash -n tools/harvest_kaggle.sh tools/colab_guard.sh
```

The offline tests restore the actual tracked snapshot into a new database and replay frozen
Kaggle evidence without changing knowledge. Focused fixtures cover source and supersession
links, experiment-to-artifact references, repeated imports, refused drift, malformed snapshots,
failed atomic exports, digest tampering, retained manual annotations, acquisition timestamps,
repository fetch errors, commit-stat attribution and the explicit CPU smoke-report contract.

Harvest tests replace curl with local fixtures: a failed request leaves old evidence intact;
a complete successful batch publishes an index that verifies before ingestion. The harvester
now stages all responses and rejects HTTP errors, invalid JSON and missing response fields
before replacing existing evidence. Normal publish exceptions restore the previous batch.
A killed process or power failure during publication can require manual recovery from its
staging directory; the directory lock must then be inspected and removed before retrying.

Clean restore initially exposed a deferred-foreign-key transaction bug in the new snapshot
loader. An explicit staging transaction fixes forward supersession references; connection
and file cleanup now also passes the test run without resource warnings.

The local suite and shell syntax checks passed. These checks establish knowledge-tool behavior,
not model accuracy or Kaggle acceptance. No training, inference benchmark, paid GPU experiment
or live Kaggle submission was run as part of this review. The reference model/scorer suite
requires its separate heavy environment and was not run. Platform parity observations added
by the concurrent session remain separately sourced in the database.

Most approach and persistence changes were included by the concurrent shared-checkout commit
`6a70879` while this review was in progress. This follow-up completes the regression fixes,
transaction cleanup, staged harvesting and CI coverage against that combined state.
