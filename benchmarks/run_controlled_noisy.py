"""CLI front end for the fixed-noise controlled benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from misda.benchmark import DEFAULT_SEED, write_json
from misda.benchmarks.validation import (
    CONTROLLED_PROBLEM_IDS,
    DEFAULT_OBSERVATION_SEED,
    DEFAULT_SIGMA,
    run_controlled_noisy,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="Use N=64 for smoke testing.")
    parser.add_argument("--problem-id", action="append", dest="problem_ids")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--observation-seed", type=int, default=DEFAULT_OBSERVATION_SEED)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifact = run_controlled_noisy(
        n=64 if args.quick else 300,
        seed=args.seed,
        observation_seed=args.observation_seed,
        sigma=args.sigma,
        problem_ids=(
            tuple(args.problem_ids) if args.problem_ids else CONTROLLED_PROBLEM_IDS
        ),
    )
    write_json(artifact, args.output)


if __name__ == "__main__":
    main()
