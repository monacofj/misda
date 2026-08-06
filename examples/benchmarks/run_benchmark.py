"""Run the canonical and synthetic-MOP baseline batteries as a JSON CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import misda
from misda.benchmark import (
    BenchmarkCase,
    DEFAULT_SEED,
    FORMAT_VERSION,
    METHOD,
    serialize_benchmark_result,
    software_versions,
    write_json,
)
from examples.benchmarks.cases import CANONICAL_CASES
from examples.mop_definitions import (
    mopA_monotonic_redundancy,
    mopB_tradeoff_with_redundancies,
    mopC_latent_blocks_4x5,
    mopD_pure_conflict_groups,
    mopE_partial_redundancy_noisy,
    mopF_regime_switching,
)


MOP_CASES = (
    ("mop_a", mopA_monotonic_redundancy),
    ("mop_b", mopB_tradeoff_with_redundancies),
    ("mop_c", mopC_latent_blocks_4x5),
    ("mop_d", mopD_pure_conflict_groups),
    ("mop_e", mopE_partial_redundancy_noisy),
    ("mop_f", mopF_regime_switching),
)

BENCHMARK_CASES = tuple(
    (f"case_{number:02d}", generator)
    for number, (_, generator) in enumerate(CANONICAL_CASES, start=1)
) + MOP_CASES


def run_benchmark(
    *,
    n: int = 1000,
    seed: int = DEFAULT_SEED,
    case_ids: set[str] | None = None,
    serializer=None,
) -> dict:
    cases = []
    for case_id, generator in BENCHMARK_CASES:
        if case_ids is not None and case_id not in case_ids:
            continue
        frame, truth = generator(N=n, seed=seed)
        if serializer is not None:
            case = serializer(case_id, frame, truth, seed=seed)
        else:
            declaration = BenchmarkCase.from_truth(
                case_id,
                truth,
                adversarial=case_id == "case_05",
            )
            result = misda.analyze(
                frame,
                method=METHOD,
                name=truth["name"],
                seed=seed,
                max_evaluated_mis=1,
            )
            case = serialize_benchmark_result(
                declaration,
                result,
                frame,
                seed=seed,
            )
        cases.append(case)
    if case_ids is not None:
        found = {case["case_id"] for case in cases}
        unknown = sorted(case_ids - found)
        if unknown:
            raise ValueError(f"Unknown case id(s): {', '.join(unknown)}")

    return {
        "format_version": 1 if serializer is not None else FORMAT_VERSION,
        "suite": "benchmark",
        "method": METHOD,
        "parameters": {"n": int(n), "seed": int(seed)},
        "software": software_versions(),
        "cases": cases,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use N=64 for smoke testing; never use this for scientific baselines.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Run only this case id; may be supplied more than once.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifact = run_benchmark(
        n=64 if args.quick else 1000,
        case_ids=set(args.case_ids) if args.case_ids else None,
    )
    write_json(artifact, args.output)


if __name__ == "__main__":
    main()
