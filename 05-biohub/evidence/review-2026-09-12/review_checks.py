"""Run from repository root using the scoring environment. Diagnostic failures are findings."""
import ast
import contextlib
import csv
import hashlib
import io
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
import numpy as np
sys.path[:0] = ['src', 'tools', 'reference/royerlab-baseline/src', 'reference/royerlab-baseline']
from biohub.contracts import Graph, Config
from biohub.detect import local_maxima
from biohub.submit import check, write, HEADER
from biohub.resolve import resolve
from _eval_common import to_td, score_one
from validate_submission import main as validate
from scripts.csv_to_geffs import build_graph_from_rows
from tracking_cellmot.metrics import evaluate, per_sample_metrics
import polars as pl

results = []
def record(name, expected, actual):
    results.append(dict(name=name, expected=expected, actual=actual, passed=expected == actual))
    print(name, file=sys.stderr, flush=True)

record('flat_heatmap_has_no_unique_peaks', 0, len(local_maxima(np.ones((3, 3, 3)), Config())))
heat = np.zeros((9, 25, 25)); heat[4, 12, 5] = 1; heat[4, 12, 10] = .99
record('stem_grid_peaks_8um_apart_survive_5um_nms', 2, len(local_maxima(heat, Config())))
g = Graph(t=np.array([0, 1]), zyx=np.array([[2., 2., 2.], [2., 2., 2.]]), edges=np.array([[0, 1]]), dataset='synthetic')
g.zyx[0, 0] = np.nan
record('writer_rejects_nan', True, bool(check(g)))
g.zyx[0, 0] = 2
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / 'submission.csv'
    p.write_text(','.join(HEADER) + '\n')
    with contextlib.redirect_stdout(io.StringIO()): status = validate(str(p))
    record('validator_rejects_header_only', 1, status)
    g.edges = np.array([[-2, 1]])
    record('writer_rejects_negative_endpoint', True, bool(check(g)))
    g.edges = np.array([[0, 1]])
    write([g], str(p), expect_datasets=['synthetic'])
    with contextlib.redirect_stdout(io.StringIO()): status = validate(str(p), ['synthetic'])
    df = pl.read_csv(p)
    pred = build_graph_from_rows(df.filter(pl.col('row_type') == 'node'), df.filter(pl.col('row_type') == 'edge'))
    er = evaluate(pred, to_td(g), scale=(1.625, .40625, .40625))
    record('graph_csv_reload_official_scorer_roundtrip', [0, 1, 0, 0], [status, er.edge_tp, er.edge_fp, er.edge_fn])

# Equal frame counts with a dying distractor: Hungarian consumes both daughters before forks.
h = Graph(t=np.array([0, 0, 1, 1]), zyx=np.array([[5, 10, 10], [5, 20, 10], [5, 9, 10], [5, 11, 10]], float),
          edges=np.array([[0, 2], [0, 3], [1, 2], [1, 3]]), edge_prob=np.array([.99, .99, .49, .49]))
record('fork_can_compete_with_two_continuations', 1, resolve(h, Config(fork_accept_p=-1)).forks())
# Execute only the inflation function, without running EXP-18's top-level sweep.
p = Path(__file__).with_name('exp18_reviewed.py')
if p.exists():
    tree = ast.parse(p.read_text()); fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'inflate')
    ns = {'np': np, 'SCALE': np.array([1.625, .40625, .40625])}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(p), 'exec'), ns)
    tt, zz = ns['inflate'](np.arange(100), np.tile([32, 128, 128], (100, 1)), 2, np.random.default_rng(0))
    record('inflation_preserves_one_extra_track_node_per_frame', True, bool(np.all(np.bincount(tt[100:], minlength=100) == 1)))
files = sorted(Path('data/train_geff').glob('*.geff'))
metadata = {'first_60_embryos': dict(Counter(p.stem[:4] for p in files[:60]))}
from geff import GeffMetadata
from _eval_common import load_gt
counts = Counter(); estimated = Counter()
for p in files:
    # GEFF node count from array metadata, no graph materialisation.
    a = json.loads((p / 'nodes/ids/zarr.json').read_text())
    counts[p.stem[:4]] += a['shape'][0]
    estimated[p.stem[:4]] += json.loads((p / 'zarr.json').read_text())['attributes']['geff']['extra']['estimated_number_of_nodes']
metadata.update(annotated=dict(counts), estimated=dict(estimated))
metadata['multiplier_at_estimated_density'] = 1.0
metadata['multiplier_at_annotated_density_pooled_illustration'] = 1.1 - .1 * sum(counts.values()) / sum(estimated.values())
report = dict(checks=results, metadata=metadata,
              hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path('src/biohub/fork_model.json'), Path('reference/royerlab-baseline/src/tracking_cellmot/metrics.py'), Path('reference/royerlab-baseline/src/tracking_cellmot/division_metrics.py')]})
print(json.dumps(report, indent=2))
