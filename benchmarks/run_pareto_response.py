"""CLI for the experimental stochastic Pareto perturbation-response study."""

from __future__ import annotations

import argparse
from pathlib import Path

from misda.benchmark import write_json
from misda.benchmarks.pareto_response import run_pareto_response_validation
from misda.benchmarks.validation import (
    NOISE_REPLICATE_SEEDS,
    NOISE_SIGMAS,
    NOISY_ROBUSTNESS_PROBLEM_IDS,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--draws", type=int)
    parser.add_argument("--problem-id", action="append", dest="problem_ids")
    parser.add_argument("--sigma", action="append", type=float, dest="sigmas")
    parser.add_argument("--replicate-seed", action="append", type=int, dest="replicate_seeds")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifact = run_pareto_response_validation(
        n=48 if args.quick else 300,
        draws=(8 if args.quick and args.draws is None else args.draws),
        problem_ids=(
            tuple(args.problem_ids)
            if args.problem_ids
            else (("independence", "monotonic_redundancy") if args.quick else NOISY_ROBUSTNESS_PROBLEM_IDS)
        ),
        sigmas=(
            tuple(args.sigmas)
            if args.sigmas
            else ((0.0, 0.10) if args.quick else NOISE_SIGMAS)
        ),
        replicate_seeds=(
            tuple(args.replicate_seeds)
            if args.replicate_seeds
            else ((101,) if args.quick else NOISE_REPLICATE_SEEDS)
        ),
    )
    write_json(artifact, args.output)


if __name__ == "__main__":
    main()
