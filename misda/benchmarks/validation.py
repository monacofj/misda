"""Reusable benchmark-validation runners shared by notebooks and CLI front ends."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

import misda
from misda.benchmark import (
    BenchmarkCase,
    DEFAULT_SEED,
    FORMAT_VERSION,
    METHOD,
    matrix_sha256,
    serialize_benchmark_result,
    software_versions,
)
from misda.benchmarks.problems import PROBLEM_BY_ID
from misda.benchmarks.truth import diagnostic_truth

DEFAULT_OBSERVATION_SEED = 456
DEFAULT_SIGMA = 0.10
SAMPLING_REPLICATE_SEEDS = tuple(range(1001, 1021))
NOISE_REPLICATE_SEEDS = (101, 202, 303, 404, 505)
NOISE_SIGMAS = (0.00, 0.05, 0.10, 0.20, 0.40)

CONTROLLED_PROBLEM_IDS = (
    "independence",
    "total_redundancy",
    "blocks_4x5",
    "blocks_2x10",
    "mixed_independent_and_blocks",
    "monotonic_redundancy",
    "antagonistic_linear_groups",
    "tradeoff_redundancies",
    "nonlinear_blocks_4x5",
    "antagonistic_nonlinear_groups",
    "overlapping_factors",
    "transitive_chain",
    "regime_switching",
)

NOISY_ROBUSTNESS_PROBLEM_IDS = (
    "independence",
    "total_redundancy",
    "blocks_4x5",
    "monotonic_redundancy",
    "antagonistic_linear_groups",
    "nonlinear_blocks_4x5",
    "antagonistic_nonlinear_groups",
    "transitive_chain",
)

# Historical JSON ids remain stable even though the public notebook presentation
# now uses a single 1..13 sequence.
SERIALIZATION_CASE_ID = {
    "independence": "case_01",
    "total_redundancy": "case_02",
    "blocks_4x5": "case_03",
    "blocks_2x10": "case_04",
    "transitive_chain": "case_05",
    "mixed_independent_and_blocks": "case_06",
    "antagonistic_linear_groups": "case_07",
    "monotonic_redundancy": "mop_a",
    "tradeoff_redundancies": "mop_b",
    "nonlinear_blocks_4x5": "mop_c",
    "antagonistic_nonlinear_groups": "mop_d",
    "overlapping_factors": "mop_e",
    "regime_switching": "mop_f",
}
PRESENTATION_CASE = {
    problem_id: index
    for index, problem_id in enumerate(CONTROLLED_PROBLEM_IDS, start=1)
}


def _validate_problem_ids(problem_ids: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(problem_ids)
    unknown = set(normalized) - set(PROBLEM_BY_ID)
    if unknown:
        raise ValueError(f"Unknown problem id(s): {', '.join(sorted(unknown))}")
    return normalized


def _safe_float(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _support_reasons(mis_set) -> list[str]:
    return sorted(
        {
            reason
            for candidate_support in mis_set.support.results
            for reason in candidate_support.reasons
        }
    )


def _mean(records, key):
    values = [float(record[key]) for record in records if record[key] is not None]
    return float(sum(values) / len(values)) if values else None


def _pareto_decomposition(mis_set, benchmark_result):
    ranking = misda.rank(mis_set)
    selected = ranking.mis() if len(ranking) else None
    selected_index = ranking.indices[0] if ranking.indices else None
    reduction = selected.pareto if selected is not None else None
    stability = mis_set.pareto_stability
    return {
        "truth_size": (
            len(benchmark_result.pareto_expected)
            if benchmark_result.pareto_expected is not None
            else None
        ),
        "observed_size": (
            len(benchmark_result.observed_pareto_indices)
            if benchmark_result.observed_pareto_indices is not None
            else None
        ),
        "reduced_size": reduction.reduced_front_size if reduction is not None else None,
        "observation_jaccard": _safe_float(benchmark_result.observation_pareto_jaccard),
        "reduction_jaccard": _safe_float(reduction.jaccard if reduction is not None else None),
        "end_to_end_jaccard": _safe_float(benchmark_result.pareto_jaccard),
        "observed_fraction": _safe_float(stability.observed_front_fraction),
        "additive_epsilon": _safe_float(
            stability.epsilon_for_candidate(selected_index)
            if selected_index is not None
            else None
        ),
        "dominance_margin_min": _safe_float(stability.dominance_margin_min),
        "dominance_margin_median": _safe_float(stability.dominance_margin_median),
        "dominance_margin_max": _safe_float(stability.dominance_margin_max),
    }


def analyze_controlled_noisy_problem(
    problem_id: str,
    *,
    n: int = 300,
    seed: int = DEFAULT_SEED,
    observation_seed: int = DEFAULT_OBSERVATION_SEED,
    sigma: float = DEFAULT_SIGMA,
):
    """Run one fixed-noise controlled problem and retain runtime objects."""
    _validate_problem_ids((problem_id,))
    problem = PROBLEM_BY_ID[problem_id]
    dataset = problem.generate(
        N=n,
        seed=seed,
        sigma=sigma,
        observation_seed=observation_seed,
    )
    truth = diagnostic_truth(problem, dataset.Z)
    mis_set = misda.discover(dataset.Y, name=truth["name"], seed=seed)
    mis_set.evaluate(metrics=("linear", "pareto"))
    ranking = misda.rank(mis_set)
    benchmark_result = misda.benchmark(mis_set, truth)
    return {
        "problem": problem,
        "dataset": dataset,
        "truth": truth,
        "result_obj": mis_set,
        "ranking_obj": ranking,
        "benchmark_obj": benchmark_result,
    }


def run_controlled_noisy(
    *,
    n: int = 300,
    seed: int = DEFAULT_SEED,
    observation_seed: int = DEFAULT_OBSERVATION_SEED,
    sigma: float = DEFAULT_SIGMA,
    problem_ids: Iterable[str] = CONTROLLED_PROBLEM_IDS,
) -> dict:
    problem_ids = _validate_problem_ids(problem_ids)
    cases = []
    for problem_id in problem_ids:
        runtime = analyze_controlled_noisy_problem(
            problem_id,
            n=n,
            seed=seed,
            observation_seed=observation_seed,
            sigma=sigma,
        )
        declaration = BenchmarkCase.from_truth(
            SERIALIZATION_CASE_ID[problem_id], runtime["truth"]
        )
        case = serialize_benchmark_result(
            declaration,
            runtime["result_obj"],
            runtime["dataset"].Y,
            seed=seed,
        )
        case["problem_id"] = problem_id
        case["presentation_case"] = PRESENTATION_CASE[problem_id]
        case["observation"] = {
            "sigma": float(sigma),
            "observation_seed": int(observation_seed),
        }
        case["clean_input_sha256"] = matrix_sha256(runtime["dataset"].Z)
        case["pareto_decomposition"] = _pareto_decomposition(
            runtime["result_obj"], runtime["benchmark_obj"]
        )
        cases.append(case)

    return {
        "format_version": FORMAT_VERSION,
        "suite": "controlled_noisy",
        "method": METHOD,
        "parameters": {
            "n": int(n),
            "seed": int(seed),
            "observation_seed": int(observation_seed),
            "sigma": float(sigma),
        },
        "software": software_versions(),
        "cases": cases,
    }


def run_sampling_robustness(
    *,
    n: int = 300,
    misda_seed: int = DEFAULT_SEED,
    replicate_seeds: Iterable[int] = SAMPLING_REPLICATE_SEEDS,
    problem_ids: Iterable[str] = CONTROLLED_PROBLEM_IDS,
) -> dict:
    problem_ids = _validate_problem_ids(problem_ids)
    replicate_seeds = tuple(int(seed) for seed in replicate_seeds)
    if not replicate_seeds:
        raise ValueError("At least one replicate seed is required.")

    records = []
    for problem_id in problem_ids:
        problem = PROBLEM_BY_ID[problem_id]
        for sample_seed in replicate_seeds:
            dataset = problem.generate(N=n, seed=sample_seed, sigma=0.0)
            truth = diagnostic_truth(problem, dataset.Z)
            mis_set = misda.discover(dataset.Y, name=truth["name"], seed=misda_seed)
            ranking = misda.rank(mis_set)
            mis_set.evaluate(
                metrics=("pareto",),
                candidates=ranking.mis(),
            )
            benchmark_result = misda.benchmark(mis_set, truth)
            reasons = _support_reasons(mis_set)
            records.append(
                {
                    "presentation_case": PRESENTATION_CASE[problem_id],
                    "problem_id": problem_id,
                    "sample_seed": sample_seed,
                    "latent_expected": benchmark_result.latent_expected,
                    "latent_observed": int(mis_set.analysis.latent_dimension),
                    "latent_exact": bool(benchmark_result.latent_exact),
                    "structural_expected": benchmark_result.structural_expected,
                    "structural_observed": int(mis_set.analysis.structural_dimension),
                    "structural_exact": bool(benchmark_result.structural_dimension_exact),
                    "selected_dimension": int(ranking.selected_dimension),
                    "selected_unit_adequacy": benchmark_result.assessment.get(
                        "selected_unit_adequacy"
                    ),
                    "support_status": mis_set.support.status,
                    "support_reasons": reasons,
                    "supported": mis_set.support.status == "SUPPORTED",
                    "transitive_chaining": "TRANSITIVE_CHAINING" in reasons,
                    "hidden_spectral_structure": "HIDDEN_SPECTRAL_STRUCTURE" in reasons,
                }
            )

    summary = []
    for problem_id in problem_ids:
        group = [record for record in records if record["problem_id"] == problem_id]
        summary.append(
            {
                "presentation_case": PRESENTATION_CASE[problem_id],
                "problem_id": problem_id,
                "replicates": len(group),
                "latent_recovery": _mean(group, "latent_exact"),
                "structural_recovery": _mean(group, "structural_exact"),
                "selected_unit_recovery": _mean(group, "selected_unit_adequacy"),
                "supported_rate": _mean(group, "supported"),
                "transitive_chaining_rate": _mean(group, "transitive_chaining"),
                "hidden_spectral_structure_rate": _mean(
                    group, "hidden_spectral_structure"
                ),
            }
        )

    return {
        "format_version": FORMAT_VERSION,
        "suite": "sampling_robustness",
        "method": METHOD,
        "parameters": {
            "n": int(n),
            "misda_seed": int(misda_seed),
            "sigma": 0.0,
            "replicate_seeds": list(replicate_seeds),
        },
        "software": software_versions(),
        "records": records,
        "summary": summary,
    }


def run_noisy_robustness(
    *,
    n: int = 300,
    misda_seed: int = DEFAULT_SEED,
    problem_ids: Iterable[str] = NOISY_ROBUSTNESS_PROBLEM_IDS,
    sigmas: Iterable[float] = NOISE_SIGMAS,
    replicate_seeds: Iterable[int] = NOISE_REPLICATE_SEEDS,
) -> dict:
    problem_ids = _validate_problem_ids(problem_ids)
    sigmas = tuple(float(value) for value in sigmas)
    replicate_seeds = tuple(int(seed) for seed in replicate_seeds)
    if not sigmas:
        raise ValueError("At least one sigma is required.")
    if not replicate_seeds:
        raise ValueError("At least one replicate seed is required.")

    records = []
    for problem_position, problem_id in enumerate(problem_ids):
        problem = PROBLEM_BY_ID[problem_id]
        for replicate_seed in replicate_seeds:
            sample_seed = int(
                np.random.SeedSequence([replicate_seed, problem_position, 1])
                .generate_state(1)[0]
            )
            observation_sequence = np.random.SeedSequence(
                [replicate_seed, problem_position, 2]
            )
            X = problem.sample(N=n, seed=sample_seed)
            Z = problem.evaluate(X)
            truth = diagnostic_truth(problem, Z)
            epsilon = np.random.default_rng(observation_sequence).normal(size=Z.shape)

            for sigma in sigmas:
                Y = problem.observe(Z, sigma=sigma, standard_noise=epsilon)
                mis_set = misda.discover(Y, name=truth["name"], seed=misda_seed)
                ranking = misda.rank(mis_set)
                mis_set.evaluate(
                    metrics=("pareto",),
                    candidates=ranking.mis(),
                )
                benchmark_result = misda.benchmark(mis_set, truth)
                selected = ranking.mis()
                selected_index = ranking.indices[0] if ranking.indices else None
                pareto_stability = mis_set.pareto_stability
                reasons = _support_reasons(mis_set)
                records.append(
                    {
                        "problem_id": problem_id,
                        "replicate_seed": replicate_seed,
                        "sample_seed": sample_seed,
                        "sigma": float(sigma),
                        "latent_expected": benchmark_result.latent_expected,
                        "latent_observed": int(mis_set.analysis.latent_dimension),
                        "latent_exact": bool(benchmark_result.latent_exact),
                        "structural_expected": benchmark_result.structural_expected,
                        "structural_observed": int(mis_set.analysis.structural_dimension),
                        "structural_exact": bool(
                            benchmark_result.structural_dimension_exact
                        ),
                        "selected_dimension": int(
                            misda.rank(mis_set).selected_dimension
                        ),
                        "selected_unit_adequacy": benchmark_result.assessment.get(
                            "selected_unit_adequacy"
                        ),
                        "pareto_end_to_end_jaccard": _safe_float(
                            benchmark_result.pareto_jaccard
                        ),
                        "pareto_observation_jaccard": _safe_float(
                            benchmark_result.observation_pareto_jaccard
                        ),
                        "pareto_reduction_jaccard": _safe_float(
                            selected.pareto.jaccard
                            if selected is not None and selected.pareto is not None
                            else None
                        ),
                        "pareto_observed_fraction": _safe_float(
                            pareto_stability.observed_front_fraction
                        ),
                        "pareto_additive_epsilon": _safe_float(
                            pareto_stability.epsilon_for_candidate(selected_index)
                            if selected_index is not None
                            else None
                        ),
                        "pareto_dominance_margin_min": _safe_float(
                            pareto_stability.dominance_margin_min
                        ),
                        "pareto_dominance_margin_median": _safe_float(
                            pareto_stability.dominance_margin_median
                        ),
                        "pareto_dominance_margin_max": _safe_float(
                            pareto_stability.dominance_margin_max
                        ),
                        "support_status": mis_set.support.status,
                        "support_reasons": reasons,
                        "supported": mis_set.support.status == "SUPPORTED",
                        "unsupported": mis_set.support.status == "UNSUPPORTED",
                        "transitive_chaining": "TRANSITIVE_CHAINING" in reasons,
                    }
                )

    summary = []
    for problem_id in problem_ids:
        for sigma in sigmas:
            group = [
                record
                for record in records
                if record["problem_id"] == problem_id and record["sigma"] == sigma
            ]
            summary.append(
                {
                    "problem_id": problem_id,
                    "sigma": float(sigma),
                    "replicates": len(group),
                    "latent_recovery": _mean(group, "latent_exact"),
                    "structural_recovery": _mean(group, "structural_exact"),
                    "selected_unit_recovery": _mean(
                        group, "selected_unit_adequacy"
                    ),
                    "pareto_observation_jaccard": _mean(
                        group, "pareto_observation_jaccard"
                    ),
                    "pareto_reduction_jaccard": _mean(
                        group, "pareto_reduction_jaccard"
                    ),
                    "pareto_end_to_end_jaccard": _mean(
                        group, "pareto_end_to_end_jaccard"
                    ),
                    "pareto_observed_fraction": _mean(
                        group, "pareto_observed_fraction"
                    ),
                    "pareto_additive_epsilon": _mean(
                        group, "pareto_additive_epsilon"
                    ),
                    "pareto_dominance_margin_min": _mean(
                        group, "pareto_dominance_margin_min"
                    ),
                    "pareto_dominance_margin_median": _mean(
                        group, "pareto_dominance_margin_median"
                    ),
                    "pareto_dominance_margin_max": _mean(
                        group, "pareto_dominance_margin_max"
                    ),
                    "supported_rate": _mean(group, "supported"),
                    "unsupported_rate": _mean(group, "unsupported"),
                    "transitive_chaining_rate": _mean(
                        group, "transitive_chaining"
                    ),
                }
            )

    return {
        "format_version": FORMAT_VERSION,
        "suite": "noisy_robustness",
        "method": METHOD,
        "parameters": {
            "n": int(n),
            "misda_seed": int(misda_seed),
            "problem_ids": list(problem_ids),
            "sigmas": list(sigmas),
            "replicate_seeds": list(replicate_seeds),
        },
        "software": software_versions(),
        "records": records,
        "summary": summary,
    }
