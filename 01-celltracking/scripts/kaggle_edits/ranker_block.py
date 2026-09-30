r"""Vendored public local-association ranker loader (definitions only).

SOURCE: extracted verbatim from the public Kaggle notebook
`dalloliogm/biohub-exp227-divergence-mutualnn-wide` (lines 322-703 of the flattened
notebook), which in turn loads the public CC dataset
`pilkwang/biohub-local-association-ranker-unet300-v1` (18.5 KB).

WHAT IT IS: a 22-feature MLP (22 -> 64 -> 32 -> 1) that scores a candidate association
edge. Ships its own median/mean/std normalisation inside the checkpoint. Manifest records
`best_epoch 7`, `best_score 0.9777`, 385,648 training rows over 108,876 groups.

WHY WE WANT IT: our measured bottleneck is the EDGE term, not divisions (oracle headroom
FN->0 is +0.167 vs +0.0993 for divisions; and mikelou1 reached divJ 0.30 yet scores only
0.928). This ranker is the rerank half of the retrieve-then-rerank primitive that Ultrack
and Linajea both implement and our pipeline lacks.

HOW IT IS USED (exp227's integration, verbatim):
    evidence = 0.85 * ranker_prob + 0.15 * primary_prob
    cost[i, col] = motion_dist + 0.05 * raw_dist - MOTION_RELINK_LEARNED_BONUS * evidence
i.e. it REPLACES the raw edge probability in the relink cost with a blend. Ours currently
uses the bare `prob`.

*** TWO CONSTRAINTS THAT MUST NOT BE LOST ***

1. AUTHOR'S OWN LIMIT, verbatim from ASSOCIATION_RANKER_MANIFEST.json:
   "Use only as a constrained local association tie-breaker, not as a global edge veto."

2. LOCAL EVALUATION IS CONTAMINATED. The manifest records `datasets_seen: 199`,
   `split_column: "dataset"`, `val_fraction: 0.15` -- so it was trained on ~169 of the 199
   competition TRAINING movies. It has therefore seen most of the crops we score against in
   `data/train`. Any local score using this ranker is inflated and MUST NOT be used as a
   promotion gate. It is legitimate for the leaderboard (the hidden set is not among the
   199), so the only honest instrument for it is a submission slot. This is the same leak
   pattern `leevvin` documented for the public support pack.

VERIFIED 2026-08-25 (local, CPU): loads from a directory containing the dataset, reports
input_dim 22 and 22 feature names, and returns finite probabilities in [0, 1].

The trailing Kaggle-root discovery and instantiation from the source notebook are
DELIBERATELY OMITTED -- this file is definitions only, so it can be exercised locally.
Instantiate with `_PublicLocalAssociationRanker(<dir containing the .pt>)`.
Requires `os` in the executing namespace.
"""

import json as _ranker_json
import re as _ranker_re
from pathlib import Path as _RankerPath

import numpy as _ranker_np
import torch as _ranker_torch

_RANKER_FALLBACK_FEATURES = ['edge_prob', 'source_in_degree', 'source_out_degree', 'target_in_degree', 'target_out_degree', 'source_density_7um', 'target_density_7um', 'raw_distance_um', 'motion_distance_um', 'motion_gain_um', 'candidate_rank', 'candidate_count', 'dz_um', 'dy_um', 'dx_um', 'abs_dz_um', 'abs_dy_um', 'abs_dx_um', 'velocity_um', 'source_frame_size_norm', 'target_frame_size_norm', 't_norm']


def _ranker_normalize_feature_name(value: str) -> str:
    return _ranker_re.sub(r'[^a-z0-9]+', '_', str(value).strip().lower()).strip('_')


def _ranker_feature_aliases_from_semantics(
    *,
    edge_prob: float,
    has_learned_edge: float,
    source_in_degree: float,
    source_out_degree: float,
    target_in_degree: float,
    target_out_degree: float,
    source_frame_count: float,
    target_frame_count: float,
    source_density_7um: float,
    target_density_7um: float,
    candidate_rank_dist: float,
    candidate_count: float,
    edge_dz_um: float,
    edge_dy_um: float,
    edge_dx_um: float,
    edge_dist_um: float,
    edge_xy_um: float,
    edge_abs_z_um: float,
    motion_dist_um: float,
    motion_gain_um: float,
    source_has_prev: float,
    target_has_next: float,
    target_best_next_prob: float,
    t_norm: float,
    velocity_um: float = 0.0,
) -> dict[str, float]:
    # Exact 22-feature contract from the attached public artifact, plus aliases
    # used by earlier public snapshots. All values are in physical microns where named *_um.
    values = {
        'edge_prob': edge_prob,
        'has_learned_edge': has_learned_edge,
        'source_in_degree': source_in_degree,
        'source_out_degree': source_out_degree,
        'target_in_degree': target_in_degree,
        'target_out_degree': target_out_degree,
        'source_frame_count': source_frame_count,
        'target_frame_count': target_frame_count,
        'source_density_7um': source_density_7um,
        'target_density_7um': target_density_7um,
        'candidate_rank_dist': candidate_rank_dist,
        'candidate_rank': candidate_rank_dist,
        'candidate_count': candidate_count,
        'edge_dz_um': edge_dz_um,
        'edge_dy_um': edge_dy_um,
        'edge_dx_um': edge_dx_um,
        'edge_dist_um': edge_dist_um,
        'edge_xy_um': edge_xy_um,
        'edge_abs_z_um': edge_abs_z_um,
        'motion_dist_um': motion_dist_um,
        'motion_gain_um': motion_gain_um,
        'source_has_prev': source_has_prev,
        'target_has_next': target_has_next,
        'target_best_next_prob': target_best_next_prob,
        't_norm': t_norm,
        # Backward-compatible aliases.
        'learned_edge_prob': edge_prob,
        'primary_prob': edge_prob,
        'prob': edge_prob,
        'src_in_degree': source_in_degree,
        'src_out_degree': source_out_degree,
        'dst_in_degree': target_in_degree,
        'dst_out_degree': target_out_degree,
        'source_frame_size': source_frame_count,
        'target_frame_size': target_frame_count,
        'src_density_7um': source_density_7um,
        'dst_density_7um': target_density_7um,
        'raw_distance_um': edge_dist_um,
        'distance_um': edge_dist_um,
        'dist_um': edge_dist_um,
        'motion_distance_um': motion_dist_um,
        'motion_dist': motion_dist_um,
        'motion_gain': motion_gain_um,
        'dz_um': edge_dz_um,
        'dy_um': edge_dy_um,
        'dx_um': edge_dx_um,
        'abs_dz_um': edge_abs_z_um,
        'abs_dy_um': abs(edge_dy_um),
        'abs_dx_um': abs(edge_dx_um),
        'velocity_um': velocity_um,
        'speed_um': velocity_um,
        'time_norm': t_norm,
        'bias': 1.0,
    }
    return {_ranker_normalize_feature_name(key): float(value) for key, value in values.items()}


def _ranker_matrix_from_alias_rows(alias_rows: list[dict[str, float]], feature_names: list[str]) -> _ranker_np.ndarray:
    missing: set[str] = set()
    rows: list[list[float]] = []
    for aliases in alias_rows:
        row: list[float] = []
        for feature_name in feature_names:
            key = _ranker_normalize_feature_name(feature_name)
            if key not in aliases:
                missing.add(str(feature_name))
                row.append(0.0)
            else:
                row.append(float(aliases[key]))
        rows.append(row)
    if missing:
        raise RuntimeError(
            'The attached ranker requests unsupported feature names before inference: '
            + ', '.join(sorted(missing))
        )
    matrix = _ranker_np.asarray(rows, dtype=_ranker_np.float32)
    if matrix.ndim != 2 or not _ranker_np.isfinite(matrix).all():
        raise RuntimeError('Invalid public-ranker semantic preflight matrix.')
    return matrix


def _ranker_natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in _ranker_re.split(r"(\d+)", value)]


def _ranker_deep_values(obj, accepted_keys: set[str]):
    out = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_norm = str(key).lower().replace('-', '_').replace(' ', '_')
            if key_norm in accepted_keys:
                out.append(value)
            out.extend(_ranker_deep_values(value, accepted_keys))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            out.extend(_ranker_deep_values(value, accepted_keys))
    return out


def _ranker_to_1d(value):
    if value is None:
        return None
    if isinstance(value, _ranker_torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        arr = _ranker_np.asarray(value)
    except Exception:
        return None
    if arr.ndim != 1:
        return None
    return arr


def _ranker_candidate_roots() -> list[_RankerPath]:
    roots: list[_RankerPath] = []
    explicit = os.environ.get('BIOHUB_LOCAL_RANKER_ROOT', '').strip()
    if explicit:
        roots.append(_RankerPath(explicit))
    roots.extend([
        _RankerPath('/kaggle/input/datasets/pilkwang/biohub-local-association-ranker-unet300-v1'),
        _RankerPath('/kaggle/input/biohub-local-association-ranker-unet300-v1'),
    ])
    input_root = _RankerPath('/kaggle/input')
    if input_root.exists():
        roots.extend(sorted(input_root.glob('**/biohub-local-association-ranker-unet300-v1')))
        roots.extend(sorted(input_root.glob('**/*local*association*ranker*')))
    unique = []
    seen = set()
    for root in roots:
        try:
            key = root.resolve() if root.exists() else root
        except Exception:
            key = root
        if key in seen:
            continue
        seen.add(key)
        if root.is_dir():
            unique.append(root)
    return unique


def _ranker_checkpoint_candidates(root: _RankerPath) -> list[_RankerPath]:
    candidates = []
    for pattern in ('**/*.pt', '**/*.pth', '**/*.jit', '**/*.torchscript'):
        candidates.extend(root.glob(pattern))
    def priority(path: _RankerPath):
        name = path.name.lower()
        return (
            0 if 'local_association_ranker' in name else 1 if 'ranker' in name else 2,
            0 if 'best' in name else 1,
            len(path.parts),
            str(path),
        )
    return sorted({path for path in candidates if path.is_file()}, key=priority)


def _ranker_metadata_objects(root: _RankerPath) -> list[object]:
    objects: list[object] = []
    for path in sorted(root.glob('**/*.json')):
        try:
            if path.stat().st_size <= 4_000_000:
                objects.append(_ranker_json.loads(path.read_text()))
        except Exception:
            pass
    for path in sorted(root.glob('**/*.npz')):
        try:
            with _ranker_np.load(path, allow_pickle=True) as data:
                objects.append({key: data[key].tolist() for key in data.files})
        except Exception:
            pass
    return objects


def _ranker_extract_state(payload):
    if isinstance(payload, _ranker_torch.nn.Module):
        return payload, None
    if isinstance(payload, dict):
        for key in ('model', 'ranker', 'network', 'module'):
            value = payload.get(key)
            if isinstance(value, _ranker_torch.nn.Module):
                return value, None
        for key in ('model_state_dict', 'state_dict', 'ranker_state_dict', 'net_state_dict', 'weights'):
            value = payload.get(key)
            if isinstance(value, dict) and any(isinstance(v, _ranker_torch.Tensor) for v in value.values()):
                return None, value
        if any(isinstance(v, _ranker_torch.Tensor) for v in payload.values()):
            return None, payload
    raise RuntimeError('Unsupported local-ranker checkpoint payload. Expected a torch module or state_dict.')


class _InferredRankerMLP(_ranker_torch.nn.Module):
    def __init__(self, state: dict[str, _ranker_torch.Tensor], activation: str = 'relu'):
        super().__init__()
        cleaned = {}
        for key, value in state.items():
            key2 = str(key)
            for prefix in ('module.', 'model.', 'ranker.', 'network.'):
                if key2.startswith(prefix):
                    key2 = key2[len(prefix):]
            cleaned[key2] = value.detach().cpu()
        weight_items = [(key, value) for key, value in cleaned.items() if key.endswith('.weight') and value.ndim == 2]
        weight_items.sort(key=lambda item: _ranker_natural_key(item[0]))
        if not weight_items:
            raise RuntimeError('No 2D linear weights were found in the ranker checkpoint.')
        self.layers = _ranker_torch.nn.ModuleList()
        for weight_key, weight in weight_items:
            prefix = weight_key[:-len('.weight')]
            bias = cleaned.get(prefix + '.bias')
            layer = _ranker_torch.nn.Linear(int(weight.shape[1]), int(weight.shape[0]), bias=bias is not None)
            with _ranker_torch.no_grad():
                layer.weight.copy_(weight.to(dtype=_ranker_torch.float32))
                if bias is not None:
                    layer.bias.copy_(bias.to(dtype=_ranker_torch.float32))
            self.layers.append(layer)
        self.activation = activation.lower()

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index + 1 < len(self.layers):
                if self.activation == 'gelu':
                    x = _ranker_torch.nn.functional.gelu(x)
                elif self.activation in {'silu', 'swish'}:
                    x = _ranker_torch.nn.functional.silu(x)
                elif self.activation == 'tanh':
                    x = _ranker_torch.tanh(x)
                else:
                    x = _ranker_torch.relu(x)
        return x


class _PublicLocalAssociationRanker:
    def __init__(self, root: _RankerPath):
        self.root = root
        checkpoints = _ranker_checkpoint_candidates(root)
        if not checkpoints:
            raise FileNotFoundError(f'No .pt/.pth ranker checkpoint found under {root}')
        self.checkpoint = checkpoints[0]
        try:
            payload = _ranker_torch.load(self.checkpoint, map_location='cpu', weights_only=False)
        except TypeError:
            payload = _ranker_torch.load(self.checkpoint, map_location='cpu')
        metadata_objects = [payload] + _ranker_metadata_objects(root)

        module, state = _ranker_extract_state(payload)
        activation_values = []
        for obj in metadata_objects:
            activation_values.extend(_ranker_deep_values(obj, {'activation', 'hidden_activation'}))
        activation = str(activation_values[0]) if activation_values else 'relu'
        self.model = module if module is not None else _InferredRankerMLP(state, activation=activation)
        self.model.eval().cpu()

        if module is not None:
            linear_layers = [layer for layer in module.modules() if isinstance(layer, _ranker_torch.nn.Linear)]
            if not linear_layers:
                raise RuntimeError('Loaded ranker module contains no torch.nn.Linear input layer.')
            input_dim = int(linear_layers[0].in_features)
        else:
            input_dim = int(self.model.layers[0].in_features)
        self.input_dim = input_dim

        feature_candidates = []
        for obj in metadata_objects:
            feature_candidates.extend(_ranker_deep_values(obj, {
                'feature_names', 'features', 'input_features', 'columns', 'feature_columns',
            }))
        feature_names = None
        for candidate in feature_candidates:
            if isinstance(candidate, (list, tuple)) and candidate and all(isinstance(v, str) for v in candidate):
                if len(candidate) == input_dim:
                    feature_names = list(candidate)
                    break
        if feature_names is None:
            if input_dim != len(_RANKER_FALLBACK_FEATURES):
                raise RuntimeError(
                    f'Ranker input_dim={input_dim}, but no matching feature_names metadata was found. '
                    'The public fallback is defined only for 22 features.'
                )
            feature_names = list(_RANKER_FALLBACK_FEATURES)
            self.feature_source = 'public_22_feature_fallback'
        else:
            self.feature_source = 'artifact_metadata'
        self.feature_names = feature_names

        mean_candidates = []
        std_candidates = []
        for obj in metadata_objects:
            mean_candidates.extend(_ranker_deep_values(obj, {
                'feature_mean', 'feature_means', 'x_mean', 'mean', 'scaler_mean', 'means',
            }))
            std_candidates.extend(_ranker_deep_values(obj, {
                'feature_std', 'feature_stds', 'x_std', 'std', 'scale', 'scaler_scale', 'stds',
            }))
        self.mean = _ranker_np.zeros(input_dim, dtype=_ranker_np.float32)
        self.std = _ranker_np.ones(input_dim, dtype=_ranker_np.float32)
        for candidate in mean_candidates:
            arr = _ranker_to_1d(candidate)
            if arr is not None and len(arr) == input_dim and _ranker_np.isfinite(arr).all():
                self.mean = arr.astype(_ranker_np.float32)
                break
        for candidate in std_candidates:
            arr = _ranker_to_1d(candidate)
            if arr is not None and len(arr) == input_dim and _ranker_np.isfinite(arr).all():
                self.std = _ranker_np.maximum(arr.astype(_ranker_np.float32), 1e-6)
                break

        positive_values = []
        for obj in metadata_objects:
            positive_values.extend(_ranker_deep_values(obj, {'positive_class', 'positive_class_index', 'pos_class'}))
        self.positive_class_index = int(positive_values[0]) if positive_values else 1

    @_ranker_torch.inference_mode()
    def predict_proba(self, matrix: _ranker_np.ndarray) -> _ranker_np.ndarray:
        matrix = _ranker_np.asarray(matrix, dtype=_ranker_np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.input_dim:
            raise ValueError(f'Bad ranker feature matrix shape: {matrix.shape}; expected (*, {self.input_dim})')
        if not _ranker_np.isfinite(matrix).all():
            raise ValueError('Ranker feature matrix contains non-finite values.')
        x = (matrix - self.mean[None, :]) / self.std[None, :]
        out = self.model(_ranker_torch.from_numpy(x)).detach().cpu()
        if out.ndim == 1:
            out = out[:, None]
        if out.shape[1] == 1:
            values = out[:, 0]
            if bool(_ranker_torch.all((values >= 0.0) & (values <= 1.0))):
                probs = values
            else:
                probs = _ranker_torch.sigmoid(values)
        elif out.shape[1] == 2:
            probs = _ranker_torch.softmax(out, dim=1)[:, self.positive_class_index]
        else:
            raise RuntimeError(f'Unexpected ranker output shape: {tuple(out.shape)}')
        probs_np = probs.numpy().astype(_ranker_np.float64)
        if not _ranker_np.isfinite(probs_np).all():
            raise RuntimeError('Ranker returned non-finite probabilities.')
        return _ranker_np.clip(probs_np, 0.0, 1.0)