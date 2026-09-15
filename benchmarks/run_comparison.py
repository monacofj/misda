"""Run the diagnostic MISDA/PCA comparison battery as a JSON CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import misda
import misda.benchmarks as bench
from misda.benchmark import (
    BenchmarkCase,
    DEFAULT_SEED,
    FORMAT_VERSION,
    METHOD,
    serialize_benchmark_result,
    software_versions,
    write_json,
)


COMPARISON_PROBLEM_IDS = (
    "total_redundancy",
    "blocks_4x5",
    "antagonistic_linear_groups",
    "nonlinear_blocks_4x5",
    "transitive_chain",
)


def _curve_value(curve, dimension):
    return next(
        point[bench.COMMON_RECONSTRUCTION_METRIC]
        for point in curve
        if point["dimension"] == int(dimension)
    )


def run_comparison(
    *,
    n: int = 300,
    seed: int = DEFAULT_SEED,
    problem_ids: set[str] | None = None,
) -> dict:
    """Compare MISDA and PCA on clean diagnostics with explicit truth.

    PCA is represented by reconstruction curves. No component-selection rule is
    imposed here, because an arbitrary explained-variance cutoff would create a
    dimension estimator that is not intrinsic to PCA and would conflate PCA's
    component axis with MISDA's latent/structural estimands.
    """
    cases = []
    requested = set(COMPARISON_PROBLEM_IDS) if problem_ids is None else set(problem_ids)
    unknown = requested - set(COMPARISON_PROBLEM_IDS)
    if unknown:
        raise ValueError(f"Unknown problem id(s): {', '.join(sorted(unknown))}")

    for problem_id in COMPARISON_PROBLEM_IDS:
        if problem_id not in requested:
            continue
        problem = bench.PROBLEM_BY_ID[problem_id]
        dataset = problem.generate(N=n, seed=seed, sigma=0.0)
        truth = bench.diagnostic_truth(problem, dataset.Z)
        declaration = BenchmarkCase.from_truth(problem_id, truth)

        mis_set = misda.discover(dataset.Y, name=truth["name"], seed=seed)
        misda.evaluate(mis_set, metrics=("linear",), candidates=1)
        case = serialize_benchmark_result(
            declaration,
            mis_set,
            dataset.Y,
            seed=seed,
        )

        selected_dimension = int(mis_set.structural_ranking.selected_dimension)
        latent_truth = int(truth["latent_expected"])
        structural_truth = int(truth["structural_expected"])
        pca_external_curve = bench.pca_external_reconstruction_curve(
            dataset.Y,
            max_components=dataset.Y.shape[1],
        )
        pca_native_curve = bench.pca_in_sample_reconstruction_curve(
            dataset.Y,
            max_components=dataset.Y.shape[1],
        )
        misda_common = bench.misda_global_standardized_external_r2(
            dataset.Y,
            mis_set,
        )

        reference_dimensions = {
            "latent_truth": latent_truth,
            "structural_truth": structural_truth,
            "misda_selected": selected_dimension,
        }
        pca_at_reference = {
            name: {
                "dimension": int(dimension),
                bench.COMMON_RECONSTRUCTION_METRIC: _curve_value(
                    pca_external_curve,
                    dimension,
                ),
            }
            for name, dimension in reference_dimensions.items()
        }

        case["problem_id"] = problem_id
        case["observation"] = {"sigma": 0.0}
        case["pca"] = {
            "native_metric": "global_standardized_r2",
            "native_protocol": "in_sample",
            "native_curve": pca_native_curve,
            "external_metric": bench.COMMON_RECONSTRUCTION_METRIC,
            "external_protocol": "leave_one_out",
            "external_curve": pca_external_curve,
            "component_selection": None,
            "component_selection_reason": (
                "No arbitrary explained-variance cutoff is imposed; PCA remains "
                "a reconstruction curve in this comparison."
            ),
            "at_reference_dimensions": pca_at_reference,
        }
        case["comparison"] = {
            "metric": bench.COMMON_RECONSTRUCTION_METRIC,
            "protocol": "leave_one_out",
            "objective_weighting": "equal_after_variance_standardization",
            "truth": {
                "latent_dimension": latent_truth,
                "structural_dimension": structural_truth,
            },
            "misda": {
                "latent_dimension": int(mis_set.analysis.latent_dimension),
                "structural_dimension": int(mis_set.analysis.structural_dimension),
                "selected_dimension": selected_dimension,
                "latent_error": int(abs(mis_set.analysis.latent_dimension - latent_truth)),
                "structural_error": int(abs(mis_set.analysis.structural_dimension - structural_truth)),
                bench.COMMON_RECONSTRUCTION_METRIC: misda_common,
            },
            "pca_at_misda_selected_dimension": pca_at_reference["misda_selected"],
        }
        cases.append(case)

    return {
        "format_version": FORMAT_VERSION,
        "suite": "diagnostic_comparison",
        "methods": [METHOD, "pca"],
        "parameters": {"n": int(n), "seed": int(seed), "sigma": 0.0},
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
        "--problem-id",
        action="append",
        dest="problem_ids",
        help="Run only this diagnostic problem id; may be supplied more than once.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifact = run_comparison(
        n=64 if args.quick else 300,
        problem_ids=set(args.problem_ids) if args.problem_ids else None,
    )
    write_json(artifact, args.output)


if __name__ == "__main__":
    main()
