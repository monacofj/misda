"""CLI front end for the observation-noise robustness study."""

from __future__ import annotations

import argparse
from pathlib import Path

from misda.benchmark import DEFAULT_SEED, write_json
from misda.benchmarks.validation import (
    NOISE_REPLICATE_SEEDS,
    NOISE_SIGMAS,
    NOISY_ROBUSTNESS_PROBLEM_IDS,
    run_noisy_robustness,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--problem-id", action="append", dest="problem_ids")
    parser.add_argument("--sigma", action="append", type=float, dest="sigmas")
    parser.add_argument("--replicate-seed", action="append", type=int, dest="replicate_seeds")
    parser.add_argument("--misda-seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifact = run_noisy_robustness(
        n=48 if args.quick else 300,
        misda_seed=args.misda_seed,
        problem_ids=(
            tuple(args.problem_ids)
            if args.problem_ids
            else (("independence", "total_redundancy") if args.quick else NOISY_ROBUSTNESS_PROBLEM_IDS)
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
