"""Expected-Jaccard edge weights and small lineage-constrained ILP backends.

For selected edges ``x``, the approximation used here is::

    expected_J(x) = sum(p_tp[e] * x[e]) / (G + sum(p_fp[e] * x[e]))

where ``G`` is the number of ground-truth edges.  A fixed Dinkelbach parameter
``lambda`` gives the additive objective ``p_tp - lambda * p_fp``.  Importantly,
``p_fp`` is a separately calibrated probability: sparse annotation means that
an edge can be neither a metric TP nor a metric-counted FP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np


ArrayCalibrator = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CandidateEdges:
    source_id: np.ndarray
    target_id: np.ndarray
    p_tp: np.ndarray
    p_fp: np.ndarray

    def __post_init__(self) -> None:
        source = np.asarray(self.source_id, dtype=np.int64)
        target = np.asarray(self.target_id, dtype=np.int64)
        p_tp = np.asarray(self.p_tp, dtype=np.float64)
        p_fp = np.asarray(self.p_fp, dtype=np.float64)
        n = len(source)
        if any(a.ndim != 1 or len(a) != n for a in (target, p_tp, p_fp)):
            raise ValueError("all candidate fields must be one-dimensional and equally sized")
        if np.any(source == target):
            raise ValueError("self edges are not valid lineage candidates")
        if len(set(zip(source.tolist(), target.tolist()))) != n:
            raise ValueError("duplicate source_id,target_id candidate")
        for name, values in (("p_tp", p_tp), ("p_fp", p_fp)):
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError(f"{name} must contain finite probabilities in [0, 1]")
        object.__setattr__(self, "source_id", source)
        object.__setattr__(self, "target_id", target)
        object.__setattr__(self, "p_tp", p_tp)
        object.__setattr__(self, "p_fp", p_fp)

    def __len__(self) -> int:
        return len(self.source_id)


@dataclass(frozen=True)
class LogitCalibration:
    """Independent Platt/temperature-style calibration hook.

    The transform is ``sigmoid(logit(p) / temperature + bias)``.  Fit TP and
    metric-counted-FP parameters independently on embryo-held-out data.
    """

    temperature: float = 1.0
    bias: float = 0.0
    epsilon: float = 1e-6

    def __call__(self, values: np.ndarray) -> np.ndarray:
        if self.temperature <= 0:
            raise ValueError("calibration temperature must be positive")
        p = np.clip(np.asarray(values, dtype=np.float64), self.epsilon, 1 - self.epsilon)
        logits = np.log(p) - np.log1p(-p)
        z = logits / self.temperature + self.bias
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def calibrate_candidates(
    candidates: CandidateEdges,
    *,
    tp_calibrator: ArrayCalibrator | None = None,
    fp_calibrator: ArrayCalibrator | None = None,
) -> CandidateEdges:
    """Apply separate, caller-supplied calibration transforms."""

    tp = candidates.p_tp if tp_calibrator is None else tp_calibrator(candidates.p_tp)
    fp = candidates.p_fp if fp_calibrator is None else fp_calibrator(candidates.p_fp)
    return CandidateEdges(candidates.source_id, candidates.target_id, tp, fp)


def reweight_candidates(candidates: CandidateEdges, lambda_: float) -> np.ndarray:
    if not np.isfinite(lambda_) or lambda_ < 0:
        raise ValueError("lambda must be finite and non-negative")
    return candidates.p_tp - float(lambda_) * candidates.p_fp


@dataclass(frozen=True)
class SolveResult:
    selected: np.ndarray
    weights: np.ndarray
    backend: str
    objective: float
    expected_tp: float
    expected_fp: float

    def ratio(self, ground_truth_edges: float) -> float:
        denominator = float(ground_truth_edges) + self.expected_fp
        return self.expected_tp / denominator if denominator > 0 else 0.0


def _capacities_ok(candidates: CandidateEdges, selected: np.ndarray) -> bool:
    sources = candidates.source_id[selected]
    targets = candidates.target_id[selected]
    return (
        all(count <= 2 for count in np.unique(sources, return_counts=True)[1])
        and all(count <= 1 for count in np.unique(targets, return_counts=True)[1])
    )


def _solve_greedy(candidates: CandidateEdges, weights: np.ndarray, tolerance: float) -> np.ndarray:
    selected = np.zeros(len(candidates), dtype=bool)
    out_count: dict[int, int] = {}
    in_count: dict[int, int] = {}
    # Stable index tie-break makes the fallback reproducible.
    for i in sorted(range(len(weights)), key=lambda j: (-weights[j], j)):
        if weights[i] <= tolerance:
            continue
        source, target = int(candidates.source_id[i]), int(candidates.target_id[i])
        if out_count.get(source, 0) < 2 and in_count.get(target, 0) < 1:
            selected[i] = True
            out_count[source] = out_count.get(source, 0) + 1
            in_count[target] = in_count.get(target, 0) + 1
    return selected


def _solve_scip(candidates: CandidateEdges, weights: np.ndarray, tolerance: float) -> np.ndarray:
    from pyscipopt import Model, quicksum

    model = Model("metric_aligned_edges")
    model.hideOutput()
    variables = [model.addVar(vtype="B", name=f"edge_{i}") for i in range(len(weights))]
    for source in np.unique(candidates.source_id):
        ids = np.flatnonzero(candidates.source_id == source)
        model.addCons(quicksum(variables[i] for i in ids) <= 2)
    for target in np.unique(candidates.target_id):
        ids = np.flatnonzero(candidates.target_id == target)
        model.addCons(quicksum(variables[i] for i in ids) <= 1)
    for i, weight in enumerate(weights):
        if weight <= tolerance:
            model.addCons(variables[i] == 0)
    model.setObjective(quicksum(float(weights[i]) * variables[i] for i in range(len(weights))), "maximize")
    model.optimize()
    status = str(model.getStatus()).lower()
    if status not in {"optimal", "bestsollimit"}:
        raise RuntimeError(f"SCIP did not return a usable solution: {status}")
    return np.asarray([model.getVal(var) > 0.5 for var in variables], dtype=bool)


def _solve_scipy(candidates: CandidateEdges, weights: np.ndarray, tolerance: float) -> np.ndarray:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    source_values = np.unique(candidates.source_id)
    target_values = np.unique(candidates.target_id)
    matrix = lil_matrix((len(source_values) + len(target_values), len(weights)), dtype=np.float64)
    upper = np.r_[np.full(len(source_values), 2.0), np.ones(len(target_values))]
    for row, source in enumerate(source_values):
        matrix[row, np.flatnonzero(candidates.source_id == source)] = 1.0
    for offset, target in enumerate(target_values, start=len(source_values)):
        matrix[offset, np.flatnonzero(candidates.target_id == target)] = 1.0
    ub = np.where(weights > tolerance, 1.0, 0.0)
    result = milp(
        c=-weights,
        integrality=np.ones(len(weights), dtype=np.int8),
        bounds=Bounds(np.zeros(len(weights)), ub),
        constraints=LinearConstraint(matrix.tocsr(), -np.inf, upper),
        options={"disp": False},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"SciPy MILP failed: {result.message}")
    return np.asarray(result.x > 0.5, dtype=bool)


def _solve_tracksdata(candidates: CandidateEdges, weights: np.ndarray, tolerance: float) -> np.ndarray:
    """Use the repository's division-native tracksdata/ilpy/SCIP flow model."""

    import polars as pl
    import tracksdata as td

    graph = td.graph.InMemoryGraph()
    graph.add_edge_attr_key("metric_cost", pl.Float64, 0.0)
    graph.add_edge_attr_key("candidate_index", pl.Int64, -1)
    node_ids = sorted(set(candidates.source_id) | set(candidates.target_id))
    internal = graph.bulk_add_nodes([{"t": 0} for _ in node_ids])
    lookup = dict(zip(node_ids, internal, strict=True))
    allowed = np.flatnonzero(weights > tolerance)
    if len(allowed) == 0:
        return np.zeros(len(candidates), dtype=bool)
    graph.bulk_add_edges(
        [
            {
                "source_id": lookup[int(candidates.source_id[i])],
                "target_id": lookup[int(candidates.target_id[i])],
                "metric_cost": -float(weights[i]),
                "candidate_index": int(i),
            }
            for i in allowed
        ]
    )
    solution = td.solvers.ILPSolver(
        edge_weight="metric_cost",
        node_weight=0.0,
        appearance_weight=0.0,
        disappearance_weight=0.0,
        division_weight=0.0,
        merge_weight=None,
        return_solution=True,
    ).solve(graph)
    selected = np.zeros(len(candidates), dtype=bool)
    if solution is not None and solution.num_edges() > 0:
        indices = solution.edge_attrs(attr_keys=["candidate_index"])["candidate_index"].to_list()
        selected[np.asarray(indices, dtype=np.int64)] = True
    return selected


def solve_lineage(
    candidates: CandidateEdges,
    *,
    lambda_: float,
    backend: str = "auto",
    score_tolerance: float = 1e-12,
) -> SolveResult:
    """Maximize fixed-lambda weights under parent<=1 and child<=2 constraints.

    ``auto`` tries direct PySCIPOpt, then SciPy MILP, then greedy.  ``tracksdata``
    invokes the repository's fuller flow formulation explicitly.  Greedy is a
    reproducible fallback, not guaranteed optimal.
    """

    weights = reweight_candidates(candidates, lambda_)
    if len(candidates) == 0:
        selected = np.zeros(0, dtype=bool)
        used_backend = "empty"
    else:
        solvers = {
            "scip": _solve_scip,
            "scipy": _solve_scipy,
            "greedy": _solve_greedy,
            "tracksdata": _solve_tracksdata,
        }
        if backend == "auto":
            failures = []
            for name in ("scip", "scipy", "greedy"):
                try:
                    selected = solvers[name](candidates, weights, score_tolerance)
                    used_backend = name
                    break
                except (ImportError, RuntimeError) as exc:
                    failures.append(f"{name}: {exc}")
            else:  # pragma: no cover - greedy has no optional dependency
                raise RuntimeError("all solver backends failed: " + "; ".join(failures))
        else:
            if backend not in solvers:
                raise ValueError(f"unknown backend {backend!r}")
            selected = solvers[backend](candidates, weights, score_tolerance)
            used_backend = backend
    if not _capacities_ok(candidates, selected):
        raise RuntimeError(f"{used_backend} returned a lineage-infeasible edge set")
    return SolveResult(
        selected=selected,
        weights=weights,
        backend=used_backend,
        objective=float(weights[selected].sum()),
        expected_tp=float(candidates.p_tp[selected].sum()),
        expected_fp=float(candidates.p_fp[selected].sum()),
    )


@dataclass(frozen=True)
class DinkelbachStep:
    iteration: int
    lambda_: float
    ratio: float
    residual: float
    selected_count: int


@dataclass(frozen=True)
class DinkelbachResult:
    solution: SolveResult
    lambda_: float
    converged: bool
    history: tuple[DinkelbachStep, ...]


def dinkelbach_solve(
    candidates: CandidateEdges,
    *,
    ground_truth_edges: float,
    initial_lambda: float = 0.0,
    backend: str = "auto",
    tolerance: float = 1e-9,
    max_iterations: int = 50,
) -> DinkelbachResult:
    """Optimize the discrete expected-J approximation when OOF ``G`` is known."""

    if not np.isfinite(ground_truth_edges) or ground_truth_edges <= 0:
        raise ValueError("ground_truth_edges must be finite and positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    lambda_ = float(initial_lambda)
    history: list[DinkelbachStep] = []
    converged = False
    solution: SolveResult | None = None
    for iteration in range(max_iterations):
        solution = solve_lineage(candidates, lambda_=lambda_, backend=backend)
        denominator = float(ground_truth_edges) + solution.expected_fp
        ratio = solution.expected_tp / denominator
        residual = solution.expected_tp - lambda_ * denominator
        history.append(DinkelbachStep(iteration, lambda_, ratio, residual, int(solution.selected.sum())))
        if abs(residual) <= tolerance:
            converged = True
            lambda_ = ratio
            break
        lambda_ = ratio
    assert solution is not None
    # Return a solution carrying weights for the final lambda, even if the last
    # update happened on the convergence boundary.
    if not np.isclose(lambda_, history[-1].lambda_):
        solution = solve_lineage(candidates, lambda_=lambda_, backend=backend)
    return DinkelbachResult(solution, lambda_, converged, tuple(history))
