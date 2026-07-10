"""Command-line entry point for metric-aligned candidate selection."""

from __future__ import annotations

import argparse
import json

from .core import LogitCalibration, calibrate_candidates, dinkelbach_solve, solve_lineage
from .io import load_nodes, read_candidates, validate_candidates, write_solution


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="CSV: source_id,target_id,p_tp[,p_fp]")
    parser.add_argument("--nodes", required=True, help="GEFF or node_id,t CSV")
    parser.add_argument("--output", required=True, help="reweighted candidate CSV")
    parser.add_argument("--backend", choices=["auto", "scip", "scipy", "tracksdata", "greedy"], default="auto")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lambda", dest="lambda_", type=float, help="fixed OOF-calibrated lambda")
    group.add_argument("--ground-truth-edges", type=float, help="run Dinkelbach on labelled OOF data")
    parser.add_argument(
        "--missing-p-fp",
        choices=["error", "one-minus-p-tp", "constant"],
        default="error",
        help="explicit ablation policy; default refuses a missing p_fp",
    )
    parser.add_argument("--p-fp-constant", type=float)
    parser.add_argument("--tp-temperature", type=float, default=1.0)
    parser.add_argument("--tp-bias", type=float, default=0.0)
    parser.add_argument("--fp-temperature", type=float, default=1.0)
    parser.add_argument("--fp-bias", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidates = read_candidates(
        args.candidates,
        missing_p_fp=args.missing_p_fp,
        p_fp_constant=args.p_fp_constant,
    )
    nodes = load_nodes(args.nodes)
    validate_candidates(candidates, nodes)
    tp_calibrator = (
        None
        if (args.tp_temperature, args.tp_bias) == (1.0, 0.0)
        else LogitCalibration(args.tp_temperature, args.tp_bias)
    )
    fp_calibrator = (
        None
        if (args.fp_temperature, args.fp_bias) == (1.0, 0.0)
        else LogitCalibration(args.fp_temperature, args.fp_bias)
    )
    candidates = calibrate_candidates(
        candidates,
        tp_calibrator=tp_calibrator,
        fp_calibrator=fp_calibrator,
    )
    if args.lambda_ is not None:
        lambda_ = args.lambda_
        result = solve_lineage(candidates, lambda_=lambda_, backend=args.backend)
        converged = None
        iterations = None
    else:
        solved = dinkelbach_solve(
            candidates,
            ground_truth_edges=args.ground_truth_edges,
            backend=args.backend,
        )
        lambda_, result = solved.lambda_, solved.solution
        converged, iterations = solved.converged, len(solved.history)
    write_solution(args.output, candidates, result, lambda_=lambda_)
    print(
        json.dumps(
            {
                "backend": result.backend,
                "lambda": lambda_,
                "selected": int(result.selected.sum()),
                "candidates": len(candidates),
                "objective": result.objective,
                "expected_tp": result.expected_tp,
                "expected_fp": result.expected_fp,
                "converged": converged,
                "iterations": iterations,
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
