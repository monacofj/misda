"""Run the classical DTLZ reference MOP examples as a JSON CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import misda
import misda.benchmarks as bench
from misda.benchmark import (
    BenchmarkCase,
    DEFAULT_SEED,
    FORMAT_VERSION,
    serialize_benchmark_result,
    software_versions,
    write_json,
)


DEFAULT_M = 10
DEFAULT_N_VARS = 19


def run_classical_mops(
    *,
    n: int = 300,
    m: int = DEFAULT_M,
    n_vars: int = DEFAULT_N_VARS,
    seed: int = DEFAULT_SEED,
    problem_ids: set[str] | None = None,
    on_front: bool = True,
) -> dict:
    """Run MISDA on reproducible classical DTLZ reference samples.

    No MISDA-specific dimensional truth is declared. Known Pareto-front geometry
    is serialized separately as reference context.
    """
    requested = set(bench.CLASSICAL_MOPS) if problem_ids is None else set(problem_ids)
    unknown = requested - set(bench.CLASSICAL_MOPS)
    if unknown:
        raise ValueError(f"Unknown classical problem id(s): {', '.join(sorted(unknown))}")

    cases = []
    for problem_id, spec in bench.CLASSICAL_MOPS.items():
        if problem_id not in requested:
            continue
        F, X = spec["generator"](
            N=n,
            M=m,
            n_vars=n_vars,
            on_front=on_front,
            seed=seed,
        )
        frame = pd.DataFrame(F, columns=[f"f{i}" for i in range(1, m + 1)])
        mis_set = misda.discover(frame, name=spec["name"], seed=seed)
        ranking = misda.rank(mis_set)
        misda.evaluate(
            mis_set,
            metrics=("linear", "pareto"),
            candidates=ranking[:1],
        )

        case = serialize_benchmark_result(
            BenchmarkCase(case_id=problem_id, name=spec["name"]),
            mis_set,
            frame,
            seed=seed,
        )
        case["problem_id"] = problem_id
        case["decision_variables"] = int(n_vars)
        case["sample_on_pareto_front"] = bool(on_front)
        case["reference_geometry"] = {
            "pareto_geometry": spec["pareto_geometry"],
            "pareto_manifold_dimension": int(spec["pareto_manifold_dimension"](m)),
            "misda_dimension_truth": None,
            "note": (
                "Known Pareto-front geometry is reference context, not a declaration "
                "of MISDA latent or structural dimension."
            ),
        }
        cases.append(case)

    return {
        "format_version": FORMAT_VERSION,
        "suite": "classical_mops",
        "method": "static",
        "parameters": {
            "n": int(n),
            "m": int(m),
            "n_vars": int(n_vars),
            "seed": int(seed),
            "on_front": bool(on_front),
        },
        "software": software_versions(),
        "cases": cases,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller smoke configuration (N=64, M=5, n_vars=14).",
    )
    parser.add_argument(
        "--problem-id",
        action="append",
        dest="problem_ids",
        help="Run only this classical problem id; may be supplied more than once.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifact = run_classical_mops(
        n=64 if args.quick else 300,
        m=5 if args.quick else DEFAULT_M,
        n_vars=14 if args.quick else DEFAULT_N_VARS,
        problem_ids=set(args.problem_ids) if args.problem_ids else None,
    )
    write_json(artifact, args.output)


if __name__ == "__main__":
    main()
