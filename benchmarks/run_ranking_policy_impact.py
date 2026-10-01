"""Audit size_span versus dominance_preservation on existing MISDA batteries.

The audit is deliberately observational: it never changes discovery or the public
ranking default.  Each generated objective matrix is discovered exactly once,
then both ranking views are compared over the same immutable MIS universe.
Benchmark truth is consulted only after both selections have been made.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np

import misda
import misda.benchmarks as bench
from misda.benchmark import DEFAULT_SEED, matrix_sha256, software_versions, write_json
from misda.benchmarks import PROBLEM_BY_ID, diagnostic_truth
from misda.benchmarks.validation import (
    CONTROLLED_PROBLEM_IDS,
    DEFAULT_OBSERVATION_SEED,
    DEFAULT_SIGMA,
    NOISE_REPLICATE_SEEDS,
    NOISE_SIGMAS,
    NOISY_ROBUSTNESS_PROBLEM_IDS,
    SAMPLING_REPLICATE_SEEDS,
)


FORMAT_VERSION = 1
BASELINE_POLICY = misda.SIZE_SPAN
CANDIDATE_POLICY = misda.DOMINANCE_PRESERVATION
DEFAULT_SUITES = (
    "controlled",
    "controlled_noisy",
    "sampling_robustness",
    "noisy_robustness",
    "comparison",
)
COMPARISON_PROBLEM_IDS = (
    "total_redundancy",
    "blocks_4x5",
    "antagonistic_linear_groups",
    "nonlinear_blocks_4x5",
    "transitive_chain",
)


def _safe_float(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _graph_edges(graph):
    return tuple(
        sorted(tuple(sorted((int(left), int(right)))) for left, right in graph.edges())
    )


def _discovery_snapshot(mis_set):
    analysis = mis_set.analysis
    return {
        "original_dimension": int(analysis.original_dimension),
        "latent_dimension": int(analysis.latent_dimension),
        "structural_dimension": int(analysis.structural_dimension),
        "alpha_onset": _safe_float(analysis.alpha_onset),
        "alpha_null": _safe_float(analysis.alpha_null),
        "alpha": _safe_float(analysis.alpha),
        "aggressiveness": float(analysis.aggressiveness),
        "separation_status": analysis.separation_status.value,
        "structural_edges": _graph_edges(analysis.structural_graph),
        "dependence_edges": _graph_edges(analysis.dependence_graph),
        "candidate_universe": tuple(candidate.indices for candidate in mis_set),
    }


def _candidate_index(mis_set, candidate):
    return next(
        index for index, observed in enumerate(mis_set) if observed is candidate
    )


def _truth_unit_evidence(mis_set, candidate, truth):
    blocks = truth.get("blocks_expected")
    expected_dimension = truth.get("structural_expected")
    if blocks is None or expected_dimension is None:
        return {"adequate": None, "counts": None, "reason": "NO_DECLARATION"}

    blocks = tuple(tuple(block) for block in blocks)
    labels = tuple(
        mis_set.analysis.structural_graph.nodes[index]["label"]
        for index in range(mis_set.analysis.original_dimension)
    )
    flattened = [label for block in blocks for label in block]
    unambiguous = bool(
        len(blocks) == int(expected_dimension)
        and len(flattened) == len(set(flattened)) == len(labels)
        and set(flattened) == set(labels)
    )
    if not unambiguous:
        return {
            "adequate": None,
            "counts": None,
            "reason": "DECLARATION_NOT_UNAMBIGUOUS",
        }

    selected = set(candidate.objectives)
    counts = tuple(len(selected.intersection(block)) for block in blocks)
    adequate = bool(
        all(count == 1 for count in counts) and candidate.size == len(blocks)
    )
    return {"adequate": adequate, "counts": list(counts), "reason": None}


def _truth_pareto_evidence(candidate, truth):
    expected = truth.get("pareto_expected")
    if expected is None or candidate.pareto is None:
        return {
            "jaccard": None,
            "precision": None,
            "recall": None,
            "reason": "NO_DECLARATION" if expected is None else "NOT_EVALUATED",
        }
    expected = set(int(index) for index in expected)
    observed = set(int(index) for index in candidate.pareto.reduced_front_indices)
    intersection = len(expected & observed)
    union = len(expected | observed)
    return {
        "jaccard": float(intersection / union) if union else 1.0,
        "precision": float(intersection / len(observed)) if observed else 0.0,
        "recall": float(intersection / len(expected)) if expected else 1.0,
        "reason": None,
    }


def _candidate_common_external_r2(data, mis_set, candidate):
    """Return the comparison-suite common score for an arbitrary selected MIS."""

    if candidate.linear is None:
        return None
    matrix = data.to_numpy(dtype=float) if hasattr(data, "to_numpy") else np.asarray(data)
    matrix = np.asarray(matrix, dtype=float)
    labels = tuple(
        mis_set.analysis.structural_graph.nodes[index]["label"]
        for index in range(mis_set.analysis.original_dimension)
    )
    selected = set(candidate.indices)
    eliminated_scores = candidate.linear.r2_by_objective or {}
    centered = matrix - np.mean(matrix, axis=0)
    totals = np.sum(centered * centered, axis=0)
    scores = []
    for index, label in enumerate(labels):
        if totals[index] <= np.finfo(float).eps:
            continue
        if index in selected:
            scores.append(1.0)
            continue
        value = eliminated_scores.get(label)
        if value is None:
            return None
        scores.append(float(value))
    return float(np.mean(scores)) if scores else None


def _policy_evidence(mis_set, ranking, truth, *, data, pca_by_dimension=None):
    candidate = ranking.selected
    if candidate is None:
        raise AssertionError(f"{ranking.policy}: ranking has no selected candidate")
    support = mis_set.support_for(candidate)
    assessment = ranking.assessment
    dominance = candidate.dominance
    linear = candidate.linear
    pareto = candidate.pareto
    evidence = {
        "policy": ranking.policy,
        "candidate_index": _candidate_index(mis_set, candidate),
        "selected_indices": list(candidate.indices),
        "selected_labels": [str(label) for label in candidate.objectives],
        "selected_dimension": int(candidate.size),
        "support": {
            "status": support.status,
            "reasons": list(support.reasons),
            "transitivity_excess": _safe_float(support.transitivity_excess),
            "spectral_excess": _safe_float(support.spectral_excess),
        },
        "reduction_assessment": (
            {
                "status": assessment.status,
                "trustworthy": bool(assessment.trustworthy),
            }
            if assessment is not None
            else None
        ),
        "dominance": (
            {
                "new_dominance_rate": float(dominance.new_dominance_rate),
                "new_dominance_pairs": int(dominance.new_dominance_pairs),
                "original_no_dominance_pairs": int(
                    dominance.original_no_dominance_pairs
                ),
                "exact_preservation": bool(dominance.exact_preservation),
            }
            if dominance is not None
            else None
        ),
        "linear": (
            {
                "mean_r2": _safe_float(linear.mean_r2),
                "worst_r2": _safe_float(linear.worst_r2),
            }
            if linear is not None
            else None
        ),
        "pareto": (
            {
                "retention": _safe_float(pareto.retention),
                "validity": _safe_float(pareto.validity),
                "jaccard": _safe_float(pareto.jaccard),
                "exact_preservation": bool(pareto.exact_preservation),
            }
            if pareto is not None
            else None
        ),
        "truth": {
            "structural_units": _truth_unit_evidence(mis_set, candidate, truth),
            "pareto": _truth_pareto_evidence(candidate, truth),
        },
    }
    if pca_by_dimension is not None:
        evidence["comparison"] = {
            "misda_global_standardized_external_r2": _safe_float(
                _candidate_common_external_r2(data, mis_set, candidate)
            ),
            "pca_at_selected_dimension": _safe_float(
                pca_by_dimension.get(candidate.size)
            ),
        }
    return evidence


def _truth_outcome(baseline, candidate):
    improvements = 0
    regressions = 0
    comparisons = 0

    before_units = baseline["truth"]["structural_units"]["adequate"]
    after_units = candidate["truth"]["structural_units"]["adequate"]
    if before_units is not None and after_units is not None:
        comparisons += 1
        if bool(after_units) > bool(before_units):
            improvements += 1
        elif bool(after_units) < bool(before_units):
            regressions += 1

    before_pareto = baseline["truth"]["pareto"]["jaccard"]
    after_pareto = candidate["truth"]["pareto"]["jaccard"]
    if before_pareto is not None and after_pareto is not None:
        comparisons += 1
        if not np.isclose(before_pareto, after_pareto, rtol=0.0, atol=1e-12):
            if after_pareto > before_pareto:
                improvements += 1
            else:
                regressions += 1

    if comparisons == 0:
        return "not_declared"
    if improvements and regressions:
        return "mixed"
    if improvements:
        return "improved"
    if regressions:
        return "regressed"
    return "neutral"


def _audit_case(*, suite, problem_id, data, truth, misda_seed, metadata):
    mis_set = misda.discover(data, name=truth.get("name"), seed=misda_seed)
    before = _discovery_snapshot(mis_set)

    # The candidate policy needs this metric for the complete MIS universe.
    mis_set.evaluate(metrics=("dominance",), candidates="all")
    baseline_ranking = misda.rank(mis_set, policy=BASELINE_POLICY)
    candidate_ranking = misda.rank(mis_set, policy=CANDIDATE_POLICY)

    selected_candidates = []
    for ranking in (baseline_ranking, candidate_ranking):
        if ranking.selected is not None and all(
            observed is not ranking.selected for observed in selected_candidates
        ):
            selected_candidates.append(ranking.selected)
    mis_set.evaluate(
        metrics=("linear", "pareto"),
        candidates=selected_candidates,
    )

    after = _discovery_snapshot(mis_set)
    if before != after:
        raise AssertionError(
            f"{suite}/{problem_id}: candidate evaluation or ranking mutated discovery"
        )

    pca_by_dimension = None
    if suite == "comparison":
        curve = bench.pca_external_reconstruction_curve(
            data,
            max_components=mis_set.analysis.original_dimension,
        )
        pca_by_dimension = {
            int(item["dimension"]): float(item[bench.COMMON_RECONSTRUCTION_METRIC])
            for item in curve
        }

    baseline = _policy_evidence(
        mis_set,
        baseline_ranking,
        truth,
        data=data,
        pca_by_dimension=pca_by_dimension,
    )
    candidate = _policy_evidence(
        mis_set,
        candidate_ranking,
        truth,
        data=data,
        pca_by_dimension=pca_by_dimension,
    )

    baseline_rate = baseline["dominance"]["new_dominance_rate"]
    candidate_rate = candidate["dominance"]["new_dominance_rate"]
    if candidate_rate > baseline_rate + 1e-15:
        raise AssertionError(
            f"{suite}/{problem_id}: dominance policy selected a worse dominance score"
        )

    comparison = {
        "same_selected_mis": (
            baseline["selected_indices"] == candidate["selected_indices"]
        ),
        "same_selected_dimension": (
            baseline["selected_dimension"] == candidate["selected_dimension"]
        ),
        "dominance_delta": float(candidate_rate - baseline_rate),
        "dominance_improved": bool(candidate_rate < baseline_rate - 1e-15),
        "truth_outcome": _truth_outcome(baseline, candidate),
        "trust_changed": (
            baseline["reduction_assessment"]["status"]
            != candidate["reduction_assessment"]["status"]
        ),
    }
    return {
        "suite": suite,
        "problem_id": problem_id,
        "input_sha256": matrix_sha256(data),
        "metadata": dict(metadata),
        "invariants_preserved": True,
        "analysis": {
            "original_dimension": before["original_dimension"],
            "latent_dimension": before["latent_dimension"],
            "structural_dimension": before["structural_dimension"],
            "alpha_onset": before["alpha_onset"],
            "alpha_null": before["alpha_null"],
            "alpha": before["alpha"],
            "aggressiveness": before["aggressiveness"],
            "separation_status": before["separation_status"],
            "n_structural_edges": len(before["structural_edges"]),
            "n_dependence_edges": len(before["dependence_edges"]),
            "n_mis": len(before["candidate_universe"]),
        },
        "policies": {
            BASELINE_POLICY: baseline,
            CANDIDATE_POLICY: candidate,
        },
        "comparison": comparison,
    }


def _filtered_problem_ids(allowed, requested):
    if requested is None:
        return tuple(allowed)
    return tuple(problem_id for problem_id in allowed if problem_id in requested)


def _iter_controlled(*, n, seed, problem_ids):
    for problem_id in _filtered_problem_ids(CONTROLLED_PROBLEM_IDS, problem_ids):
        problem = PROBLEM_BY_ID[problem_id]
        dataset = problem.generate(N=n, seed=seed, sigma=0.0)
        yield {
            "suite": "controlled",
            "problem_id": problem_id,
            "data": dataset.Y,
            "truth": diagnostic_truth(problem, dataset.Z),
            "misda_seed": seed,
            "metadata": {"sample_seed": seed, "sigma": 0.0},
        }


def _iter_controlled_noisy(*, n, seed, observation_seed, sigma, problem_ids):
    for problem_id in _filtered_problem_ids(CONTROLLED_PROBLEM_IDS, problem_ids):
        problem = PROBLEM_BY_ID[problem_id]
        dataset = problem.generate(
            N=n,
            seed=seed,
            sigma=sigma,
            observation_seed=observation_seed,
        )
        yield {
            "suite": "controlled_noisy",
            "problem_id": problem_id,
            "data": dataset.Y,
            "truth": diagnostic_truth(problem, dataset.Z),
            "misda_seed": seed,
            "metadata": {
                "sample_seed": seed,
                "observation_seed": observation_seed,
                "sigma": float(sigma),
            },
        }


def _iter_sampling_robustness(*, n, misda_seed, replicate_seeds, problem_ids):
    for problem_id in _filtered_problem_ids(CONTROLLED_PROBLEM_IDS, problem_ids):
        problem = PROBLEM_BY_ID[problem_id]
        for sample_seed in replicate_seeds:
            dataset = problem.generate(N=n, seed=int(sample_seed), sigma=0.0)
            yield {
                "suite": "sampling_robustness",
                "problem_id": problem_id,
                "data": dataset.Y,
                "truth": diagnostic_truth(problem, dataset.Z),
                "misda_seed": misda_seed,
                "metadata": {"sample_seed": int(sample_seed), "sigma": 0.0},
            }


def _iter_noisy_robustness(
    *, n, misda_seed, replicate_seeds, sigmas, problem_ids
):
    allowed = _filtered_problem_ids(NOISY_ROBUSTNESS_PROBLEM_IDS, problem_ids)
    canonical_positions = {
        problem_id: position
        for position, problem_id in enumerate(NOISY_ROBUSTNESS_PROBLEM_IDS)
    }
    for problem_id in allowed:
        problem_position = canonical_positions[problem_id]
        problem = PROBLEM_BY_ID[problem_id]
        for replicate_seed in replicate_seeds:
            sample_seed = int(
                np.random.SeedSequence([int(replicate_seed), problem_position, 1])
                .generate_state(1)[0]
            )
            observation_sequence = np.random.SeedSequence(
                [int(replicate_seed), problem_position, 2]
            )
            X = problem.sample(N=n, seed=sample_seed)
            Z = problem.evaluate(X)
            truth = diagnostic_truth(problem, Z)
            epsilon = np.random.default_rng(observation_sequence).normal(size=Z.shape)
            for sigma in sigmas:
                Y = problem.observe(Z, sigma=float(sigma), standard_noise=epsilon)
                yield {
                    "suite": "noisy_robustness",
                    "problem_id": problem_id,
                    "data": Y,
                    "truth": truth,
                    "misda_seed": misda_seed,
                    "metadata": {
                        "replicate_seed": int(replicate_seed),
                        "sample_seed": sample_seed,
                        "sigma": float(sigma),
                    },
                }


def _iter_comparison(*, n, seed, problem_ids):
    for problem_id in _filtered_problem_ids(COMPARISON_PROBLEM_IDS, problem_ids):
        problem = PROBLEM_BY_ID[problem_id]
        dataset = problem.generate(N=n, seed=seed, sigma=0.0)
        yield {
            "suite": "comparison",
            "problem_id": problem_id,
            "data": dataset.Y,
            "truth": diagnostic_truth(problem, dataset.Z),
            "misda_seed": seed,
            "metadata": {"sample_seed": seed, "sigma": 0.0},
        }


def _summarize(records):
    def one(group):
        outcomes = Counter(record["comparison"]["truth_outcome"] for record in group)
        return {
            "runs": len(group),
            "same_selected_mis": sum(
                record["comparison"]["same_selected_mis"] for record in group
            ),
            "same_selected_dimension": sum(
                record["comparison"]["same_selected_dimension"] for record in group
            ),
            "policy_divergences": sum(
                not record["comparison"]["same_selected_mis"] for record in group
            ),
            "dominance_improved": sum(
                record["comparison"]["dominance_improved"] for record in group
            ),
            "truth_improved": outcomes["improved"],
            "truth_regressed": outcomes["regressed"],
            "truth_mixed": outcomes["mixed"],
            "truth_neutral": outcomes["neutral"],
            "truth_not_declared": outcomes["not_declared"],
            "trust_changed": sum(
                record["comparison"]["trust_changed"] for record in group
            ),
        }

    by_suite = {}
    for suite in DEFAULT_SUITES:
        group = [record for record in records if record["suite"] == suite]
        if group:
            by_suite[suite] = one(group)
    return {"overall": one(records), "by_suite": by_suite}


def run_policy_impact_audit(
    *,
    suites: Iterable[str] = DEFAULT_SUITES,
    n: int = 300,
    seed: int = DEFAULT_SEED,
    observation_seed: int = DEFAULT_OBSERVATION_SEED,
    controlled_sigma: float = DEFAULT_SIGMA,
    sampling_replicate_seeds: Iterable[int] = SAMPLING_REPLICATE_SEEDS,
    noise_replicate_seeds: Iterable[int] = NOISE_REPLICATE_SEEDS,
    noise_sigmas: Iterable[float] = NOISE_SIGMAS,
    problem_ids: Iterable[str] | None = None,
):
    suites = tuple(dict.fromkeys(str(suite) for suite in suites))
    unknown_suites = sorted(set(suites) - set(DEFAULT_SUITES))
    if unknown_suites:
        raise ValueError(f"Unknown suite(s): {', '.join(unknown_suites)}")

    requested = None if problem_ids is None else set(problem_ids)
    if requested is not None:
        unknown = sorted(requested - set(PROBLEM_BY_ID))
        if unknown:
            raise ValueError(f"Unknown problem id(s): {', '.join(unknown)}")

    iterators = {
        "controlled": lambda: _iter_controlled(
            n=n, seed=seed, problem_ids=requested
        ),
        "controlled_noisy": lambda: _iter_controlled_noisy(
            n=n,
            seed=seed,
            observation_seed=observation_seed,
            sigma=controlled_sigma,
            problem_ids=requested,
        ),
        "sampling_robustness": lambda: _iter_sampling_robustness(
            n=n,
            misda_seed=seed,
            replicate_seeds=tuple(int(value) for value in sampling_replicate_seeds),
            problem_ids=requested,
        ),
        "noisy_robustness": lambda: _iter_noisy_robustness(
            n=n,
            misda_seed=seed,
            replicate_seeds=tuple(int(value) for value in noise_replicate_seeds),
            sigmas=tuple(float(value) for value in noise_sigmas),
            problem_ids=requested,
        ),
        "comparison": lambda: _iter_comparison(
            n=n, seed=seed, problem_ids=requested
        ),
    }

    records = []
    for suite in suites:
        for case in iterators[suite]():
            records.append(_audit_case(**case))

    if not records:
        raise ValueError("The selected suites/problem filters produced no audit runs.")

    artifact = {
        "format_version": FORMAT_VERSION,
        "suite": "ranking_policy_impact",
        "policies": {
            "baseline": BASELINE_POLICY,
            "candidate": CANDIDATE_POLICY,
        },
        "parameters": {
            "n": int(n),
            "seed": int(seed),
            "observation_seed": int(observation_seed),
            "controlled_sigma": float(controlled_sigma),
            "sampling_replicate_seeds": [
                int(value) for value in sampling_replicate_seeds
            ],
            "noise_replicate_seeds": [int(value) for value in noise_replicate_seeds],
            "noise_sigmas": [float(value) for value in noise_sigmas],
            "suites": list(suites),
            "problem_ids": sorted(requested) if requested is not None else None,
        },
        "software": software_versions(),
        "summary": _summarize(records),
        "records": records,
    }
    return artifact


def _print_summary(artifact):
    print("Ranking policy impact audit")
    print("===========================")
    print(f"baseline : {artifact['policies']['baseline']}")
    print(f"candidate: {artifact['policies']['candidate']}")
    for suite, summary in artifact["summary"]["by_suite"].items():
        print(
            f"{suite}: runs={summary['runs']}, "
            f"same_mis={summary['same_selected_mis']}, "
            f"divergences={summary['policy_divergences']}, "
            f"truth(+/-/mixed)={summary['truth_improved']}/"
            f"{summary['truth_regressed']}/{summary['truth_mixed']}, "
            f"dominance_improved={summary['dominance_improved']}"
        )
    overall = artifact["summary"]["overall"]
    print(
        "overall: "
        f"runs={overall['runs']}, divergences={overall['policy_divergences']}, "
        f"truth_improved={overall['truth_improved']}, "
        f"truth_regressed={overall['truth_regressed']}, "
        f"truth_mixed={overall['truth_mixed']}"
    )
    regressions = [
        record
        for record in artifact["records"]
        if record["comparison"]["truth_outcome"] in {"regressed", "mixed"}
    ]
    for record in regressions:
        baseline = record["policies"][BASELINE_POLICY]
        candidate = record["policies"][CANDIDATE_POLICY]
        print(
            "TRUTH REGRESSION: "
            f"{record['suite']}/{record['problem_id']} {record['metadata']} "
            f"{baseline['selected_labels']} -> {candidate['selected_labels']} "
            f"outcome={record['comparison']['truth_outcome']}"
        )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suite",
        action="append",
        choices=DEFAULT_SUITES,
        dest="suites",
        help="Audit only this suite; may be supplied more than once.",
    )
    parser.add_argument(
        "--problem-id",
        action="append",
        dest="problem_ids",
        help="Restrict to this benchmark problem id; may be supplied more than once.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use N=64 and abbreviated robustness replicate sets for smoke testing.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    quick = bool(args.quick)
    artifact = run_policy_impact_audit(
        suites=tuple(args.suites) if args.suites else DEFAULT_SUITES,
        n=64 if quick else 300,
        sampling_replicate_seeds=(SAMPLING_REPLICATE_SEEDS[:1] if quick else SAMPLING_REPLICATE_SEEDS),
        noise_replicate_seeds=(NOISE_REPLICATE_SEEDS[:1] if quick else NOISE_REPLICATE_SEEDS),
        noise_sigmas=((0.0, DEFAULT_SIGMA) if quick else NOISE_SIGMAS),
        problem_ids=args.problem_ids,
    )
    write_json(artifact, args.output)
    _print_summary(artifact)


if __name__ == "__main__":
    main()
