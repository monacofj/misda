"""Run the canonical clean controlled-diagnostic battery as a JSON CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import misda
from misda.benchmark import (
    BenchmarkCase,
    DECLARATION_MISMATCH,
    DEFAULT_SEED,
    FORMAT_VERSION,
    METHOD,
    serialize_benchmark_result,
    software_versions,
    write_json,
)
from misda.benchmarks import PROBLEM_BY_ID, diagnostic_truth


# Stable CLI ids are preserved across the R5 migration even though the
# scientific catalogue itself is now a single unified diagnostic suite.
BENCHMARK_CASES = (
    ("case_01", PROBLEM_BY_ID["independence"]),
    ("case_02", PROBLEM_BY_ID["total_redundancy"]),
    ("case_03", PROBLEM_BY_ID["blocks_4x5"]),
    ("case_04", PROBLEM_BY_ID["blocks_2x10"]),
    ("case_05", PROBLEM_BY_ID["transitive_chain"]),
    ("case_06", PROBLEM_BY_ID["mixed_independent_and_blocks"]),
    ("case_07", PROBLEM_BY_ID["antagonistic_linear_groups"]),
    ("mop_a", PROBLEM_BY_ID["monotonic_redundancy"]),
    ("mop_b", PROBLEM_BY_ID["tradeoff_redundancies"]),
    ("mop_c", PROBLEM_BY_ID["nonlinear_blocks_4x5"]),
    ("mop_d", PROBLEM_BY_ID["antagonistic_nonlinear_groups"]),
    ("mop_e", PROBLEM_BY_ID["overlapping_factors"]),
    ("mop_f", PROBLEM_BY_ID["regime_switching"]),
)


def run_benchmark(
    *,
    n: int = 300,
    seed: int = DEFAULT_SEED,
    case_ids: set[str] | None = None,
    serializer=None,
) -> dict:
    cases = []
    for case_id, problem in BENCHMARK_CASES:
        if case_ids is not None and case_id not in case_ids:
            continue

        dataset = problem.generate(N=n, seed=seed, sigma=0.0)
        frame = dataset.Y
        truth = diagnostic_truth(problem, dataset.Z)

        if serializer is not None:
            case = serializer(case_id, frame, truth, seed=seed)
        else:
            declaration = BenchmarkCase.from_truth(case_id, truth)
            mis_set = misda.discover(
                frame,
                name=truth["name"],
                seed=seed,
            )
            misda.evaluate(
                mis_set,
                metrics=("linear", "pareto"),
            )
            case = serialize_benchmark_result(
                declaration,
                mis_set,
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
        "suite": "diagnostic_clean",
        "method": METHOD,
        "parameters": {"n": int(n), "seed": int(seed), "sigma": 0.0},
        "software": software_versions(),
        "cases": cases,
    }


def unexpected_mismatch_case_ids(artifact) -> tuple[str, ...]:
    return tuple(case_id for case_id, _checks in unexpected_mismatch_details(artifact))


def unexpected_mismatch_details(artifact):
    details = []
    for case in artifact.get("cases", ()):
        assessment = case.get("assessment") or {}
        if assessment.get("status") != DECLARATION_MISMATCH:
            continue
        checks = tuple(
            {
                "field": check.get("field"),
                "observed": check.get("observed"),
                "expected": check.get("expected"),
                "reason": check.get("reason"),
            }
            for check in assessment.get("checks", ())
            if check.get("status") == DECLARATION_MISMATCH
        )
        details.append((case["case_id"], checks))
    return tuple(details)


def enforce_scientific_assessment(artifact) -> None:
    details = unexpected_mismatch_details(artifact)
    if not details:
        return
    lines = ["Unexpected benchmark declaration mismatch:"]
    for case_id, checks in details:
        lines.append(f"  {case_id}")
        for check in checks:
            lines.append(
                "    "
                f"{check['field']}: observed={check['observed']}, "
                f"expected={check['expected']}, reason={check['reason']}"
            )
    raise SystemExit("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use N=64 for smoke testing; never use this for scientific baselines.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail the scientific acceptance command on unexpected declaration mismatches.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Run only this stable diagnostic case id; may be supplied more than once.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifact = run_benchmark(
        n=64 if args.quick else 300,
        case_ids=set(args.case_ids) if args.case_ids else None,
    )
    write_json(artifact, args.output)
    if args.strict and not args.quick:
        enforce_scientific_assessment(artifact)


if __name__ == "__main__":
    main()
