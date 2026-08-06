"""Run the static MISDA and PCA comparison battery as a JSON CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

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


def _pca_curve(frame, max_components: int = 10) -> list[dict]:
    scaled = StandardScaler().fit_transform(frame)
    curve = []
    for dimension in range(1, min(max_components, frame.shape[1]) + 1):
        pca = PCA(n_components=dimension)
        scores = pca.fit_transform(scaled)
        reconstructed = pca.inverse_transform(scores)
        curve.append(
            {
                "dimension": dimension,
                "global_standardized_r2": float(
                    r2_score(scaled, reconstructed)
                ),
            }
        )
    return curve


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
        if serializer is not None:
            case = serializer(case_id, frame, truth, seed=seed)
        else:
            declaration = BenchmarkCase.from_truth(case_id, truth)
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
        case["pca"] = {
            "metric": "global_standardized_r2",
            "curve": _pca_curve(frame),
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
