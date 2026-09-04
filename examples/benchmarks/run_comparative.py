"""Run the static MISDA and PCA comparison battery as a JSON CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

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
from examples.mop_definitions import (
    mopA_monotonic_redundancy,
    mopC_latent_blocks_4x5,
    mopD_pure_conflict_groups,
)


COMPARATIVE_CASES = (
    ("exp_01", mopA_monotonic_redundancy),
    ("exp_02", mopC_latent_blocks_4x5),
    ("exp_03", mopD_pure_conflict_groups),
)


def _serialized_misda_common_score(frame, case) -> float:
    matrix = (
        frame.to_numpy(dtype=float)
        if hasattr(frame, "to_numpy")
        else np.asarray(frame, dtype=float)
    )
    centered = matrix - np.mean(matrix, axis=0)
    totals = np.sum(centered * centered, axis=0)
    selected = set(case.get("selected_indices", ()))
    reconstruction = case.get("linear_reconstruction") or {}
    per_objective = reconstruction.get("r2_by_objective") or {}
    labels = (
        list(frame.columns)
        if hasattr(frame, "columns")
        else [f"f{i+1}" for i in range(matrix.shape[1])]
    )

    scores = []
    for index, label in enumerate(labels):
        if totals[index] <= np.finfo(float).eps:
            continue
        if index in selected:
            scores.append(1.0)
            continue
        value = per_objective.get(label)
        if value is None:
            raise ValueError(
                f"serialized MISDA artifact lacks reconstruction R² for objective {label!r}."
            )
        scores.append(float(value))
    if not scores:
        raise ValueError("global reconstruction R² is undefined for all-constant data.")
    return float(np.mean(scores))


def run_comparative(
    *,
    n: int = 500,
    seed: int = DEFAULT_SEED,
    case_ids: set[str] | None = None,
    serializer=None,
) -> dict:
    cases = []
    for case_id, generator in COMPARATIVE_CASES:
        if case_ids is not None and case_id not in case_ids:
            continue
        frame, truth = generator(N=n, seed=seed)
        mis_set = None
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
                candidates=1,
            )
            case = serialize_benchmark_result(
                declaration,
                mis_set,
                frame,
                seed=seed,
            )

        case["pca"] = {
            "metric": "global_standardized_r2",
            "protocol": "in_sample",
            "curve": bench.pca_in_sample_reconstruction_curve(
                frame, max_components=10
            ),
        }

        misda_common = (
            bench.misda_global_standardized_external_r2(frame, mis_set)
            if mis_set is not None
            else _serialized_misda_common_score(frame, case)
        )
        selected_dimension = int(case["estimated"]["selected_dimension"])
        pca_external_curve = bench.pca_external_reconstruction_curve(
            frame,
            max_components=frame.shape[1],
        )
        pca_same_dimension = next(
            point[bench.COMMON_RECONSTRUCTION_METRIC]
            for point in pca_external_curve
            if point["dimension"] == selected_dimension
        )
        case["comparison"] = {
            "metric": bench.COMMON_RECONSTRUCTION_METRIC,
            "protocol": "leave_one_out",
            "objective_weighting": "equal_after_variance_standardization",
            "misda": {
                "dimension": selected_dimension,
                bench.COMMON_RECONSTRUCTION_METRIC: misda_common,
            },
            "pca": {
                "at_misda_dimension": pca_same_dimension,
                "curve": pca_external_curve,
            },
        }
        cases.append(case)
    if case_ids is not None:
        found = {case["case_id"] for case in cases}
        unknown = sorted(case_ids - found)
        if unknown:
            raise ValueError(f"Unknown case id(s): {', '.join(unknown)}")

    return {
        "format_version": 1 if serializer is not None else FORMAT_VERSION,
        "suite": "comparative",
        "methods": [METHOD, "pca"],
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
    artifact = run_comparative(
        n=64 if args.quick else 500,
        case_ids=set(args.case_ids) if args.case_ids else None,
    )
    write_json(artifact, args.output)


if __name__ == "__main__":
    main()
