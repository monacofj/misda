"""Probe uncertainty in dominance-preservation ranking evidence by row resampling.

The primary experiment deliberately holds the two candidate objective subsets
fixed after discovery on the original observed Y. Bootstrap row resamples then
recompute only the dominance-preservation evidence. This isolates uncertainty
in the ranking signal from uncertainty in threshold inference, graph discovery,
or MIS enumeration.

Benchmark truth is attached only after the Y-only ranking/resampling statistic
has been computed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

import misda
from benchmarks import run_ranking_policy_impact as impact
from misda._dominance import evaluate_dominance_preservation, prepare_dominance_pairs
from misda.benchmark import DEFAULT_SEED, matrix_sha256, software_versions, write_json
from misda.benchmarks import PROBLEM_BY_ID, diagnostic_truth
from misda.benchmarks.validation import NOISE_REPLICATE_SEEDS, NOISE_SIGMAS


FORMAT_VERSION = 1
DEFAULT_BOOTSTRAP_REPLICATES = 128
DEFAULT_BOOTSTRAP_SEED = 20261001
CONTROL_SAMPLE_SEEDS = (101, 202, 303, 404)
CONTROL_SCREEN_POWER = 8
TARGET_NOISY_PROBLEMS = (
    "blocks_4x5",
    "antagonistic_linear_groups",
    "nonlinear_blocks_4x5",
    "antagonistic_nonlinear_groups",
)
DEFAULT_NOISY_DIVERGENCES_PER_PROBLEM = 3
TOL = 1e-15


def _as_array(data):
    if hasattr(data, "to_numpy"):
        data = data.to_numpy(dtype=float)
    return np.asarray(data, dtype=float)


def _dominance_rate(data, selected_indices):
    prepared = prepare_dominance_pairs(_as_array(data))
    return float(
        evaluate_dominance_preservation(prepared, selected_indices)[
            "new_dominance_rate"
        ]
    )


def bootstrap_dominance_delta(
    data,
    dominance_indices,
    structural_indices,
    *,
    replicates=DEFAULT_BOOTSTRAP_REPLICATES,
    seed=DEFAULT_BOOTSTRAP_SEED,
):
    """Estimate uncertainty of the dominance-policy advantage by paired bootstrap.

    DeltaD is always defined as

        D(S_dom) - D(S_struct)

    so negative values favor the candidate selected by
    ``dominance_preservation``. The candidate subsets are fixed; only rows are
    resampled with replacement.
    """

    matrix = _as_array(data)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("data must contain at least two rows.")
    replicates = int(replicates)
    if replicates < 1:
        raise ValueError("replicates must be at least one.")

    dominance_indices = tuple(int(index) for index in dominance_indices)
    structural_indices = tuple(int(index) for index in structural_indices)
    original_prepared = prepare_dominance_pairs(matrix)
    original_dom = evaluate_dominance_preservation(
        original_prepared, dominance_indices
    )["new_dominance_rate"]
    original_struct = evaluate_dominance_preservation(
        original_prepared, structural_indices
    )["new_dominance_rate"]
    original_delta = float(original_dom - original_struct)

    rng = np.random.default_rng(int(seed))
    n_rows = matrix.shape[0]
    deltas = np.empty(replicates, dtype=float)
    dom_rates = np.empty(replicates, dtype=float)
    struct_rates = np.empty(replicates, dtype=float)

    for replicate in range(replicates):
        row_index = rng.integers(0, n_rows, size=n_rows)
        prepared = prepare_dominance_pairs(matrix[row_index])
        dom_rate = evaluate_dominance_preservation(
            prepared, dominance_indices
        )["new_dominance_rate"]
        struct_rate = evaluate_dominance_preservation(
            prepared, structural_indices
        )["new_dominance_rate"]
        dom_rates[replicate] = float(dom_rate)
        struct_rates[replicate] = float(struct_rate)
        deltas[replicate] = float(dom_rate - struct_rate)

    negative = deltas < -TOL
    positive = deltas > TOL
    tied = ~(negative | positive)
    quantiles = np.quantile(deltas, [0.05, 0.25, 0.50, 0.75, 0.95])

    return {
        "replicates": replicates,
        "bootstrap_seed": int(seed),
        "dominance_indices": list(dominance_indices),
        "structural_indices": list(structural_indices),
        "original_dominance_rate": float(original_dom),
        "original_structural_rate": float(original_struct),
        "original_delta": original_delta,
        "negative_rate": float(np.mean(negative)),
        "tie_rate": float(np.mean(tied)),
        "positive_rate": float(np.mean(positive)),
        "sign_reversal_rate": float(np.mean(positive)) if original_delta < -TOL else (
            float(np.mean(negative)) if original_delta > TOL else 0.0
        ),
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas, ddof=1)) if replicates > 1 else 0.0,
        "q05_delta": float(quantiles[0]),
        "q25_delta": float(quantiles[1]),
        "median_delta": float(quantiles[2]),
        "q75_delta": float(quantiles[3]),
        "q95_delta": float(quantiles[4]),
        "mean_dominance_rate": float(np.mean(dom_rates)),
        "mean_structural_rate": float(np.mean(struct_rates)),
    }


def _candidate_pair(data, *, misda_seed):
    mis_set = misda.discover(data, seed=int(misda_seed))
    mis_set.evaluate(metrics=("dominance",), candidates="all")
    structural = misda.rank(mis_set, policy=misda.SIZE_SPAN)
    dominance = misda.rank(mis_set, policy=misda.DOMINANCE_PRESERVATION)
    return mis_set, structural, dominance


def _truth_pareto_jaccard(candidate, truth):
    expected = truth.get("pareto_expected")
    if expected is None or candidate.pareto is None:
        return None
    expected = set(int(index) for index in expected)
    observed = set(int(index) for index in candidate.pareto.reduced_front_indices)
    union = expected | observed
    return float(len(expected & observed) / len(union)) if union else 1.0


def _truth_outcome(structural_jaccard, dominance_jaccard):
    if structural_jaccard is None or dominance_jaccard is None:
        return "not_declared"
    if np.isclose(structural_jaccard, dominance_jaccard, rtol=0.0, atol=1e-12):
        return "neutral"
    return "improved" if dominance_jaccard > structural_jaccard else "regressed"


def _record_from_discovery(
    *,
    family,
    problem_id,
    data,
    mis_set,
    structural,
    dominance,
    metadata,
    bootstrap_replicates,
    bootstrap_seed,
    truth=None,
    known_safe=None,
):
    structural_candidate = structural.mis()
    dominance_candidate = dominance.mis()
    resampling = bootstrap_dominance_delta(
        data,
        dominance_candidate.indices,
        structural_candidate.indices,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    structural_support = mis_set.support_for(structural_candidate)
    dominance_support = mis_set.support_for(dominance_candidate)

    truth_block = None
    if truth is not None:
        selected = [structural_candidate]
        if dominance_candidate is not structural_candidate:
            selected.append(dominance_candidate)
        mis_set.evaluate(metrics=("pareto",), candidates=selected)
        structural_jaccard = _truth_pareto_jaccard(structural_candidate, truth)
        dominance_jaccard = _truth_pareto_jaccard(dominance_candidate, truth)
        truth_block = {
            "structural_pareto_jaccard": structural_jaccard,
            "dominance_pareto_jaccard": dominance_jaccard,
            "pareto_delta": (
                None
                if structural_jaccard is None or dominance_jaccard is None
                else float(dominance_jaccard - structural_jaccard)
            ),
            "outcome": _truth_outcome(structural_jaccard, dominance_jaccard),
        }

    control_block = None
    if known_safe is not None:
        safe = tuple(int(index) for index in known_safe)
        control_block = {
            "known_safe_indices": list(safe),
            "structural_matches_known_safe": tuple(structural_candidate.indices) == safe,
            "dominance_matches_known_safe": tuple(dominance_candidate.indices) == safe,
        }

    return {
        "family": family,
        "problem_id": problem_id,
        "input_sha256": matrix_sha256(data),
        "metadata": dict(metadata),
        "analysis": {
            "original_dimension": int(mis_set.analysis.original_dimension),
            "latent_dimension": int(mis_set.analysis.latent_dimension),
            "structural_dimension": int(mis_set.analysis.structural_dimension),
            "n_mis": int(len(mis_set)),
        },
        "same_selected_mis": tuple(structural_candidate.indices)
        == tuple(dominance_candidate.indices),
        "same_selected_dimension": structural_candidate.size
        == dominance_candidate.size,
        "structural": {
            "indices": list(structural_candidate.indices),
            "labels": [str(label) for label in structural_candidate.objectives],
            "support_status": structural_support.status,
            "support_reasons": list(structural_support.reasons),
        },
        "dominance": {
            "indices": list(dominance_candidate.indices),
            "labels": [str(label) for label in dominance_candidate.objectives],
            "support_status": dominance_support.status,
            "support_reasons": list(dominance_support.reasons),
        },
        "resampling": resampling,
        "truth": truth_block,
        "analytical_control": control_block,
    }


def _optimization_control_records(
    *,
    sample_seeds=CONTROL_SAMPLE_SEEDS,
    screen_power=CONTROL_SCREEN_POWER,
    misda_seed=DEFAULT_SEED,
    bootstrap_replicates=DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
):
    # Imported lazily so the core package/tests do not require the optional
    # optimization benchmark dependency merely to import this module.
    import moeabench as mb

    problems = {
        "DTLZ5": (mb.mops.DTLZ5(M=10), (8, 9)),
        "DPF1": (mb.mops.DPF1(M=10, D=2, K=5), (0, 1)),
    }
    records = []
    for problem_position, (name, (mop, known_safe)) in enumerate(problems.items()):
        for sample_seed in sample_seeds:
            sampler = qmc.Sobol(d=mop.N, scramble=True, seed=int(sample_seed))
            X = qmc.scale(
                sampler.random_base2(m=int(screen_power)),
                np.asarray(mop.xl, dtype=float),
                np.asarray(mop.xu, dtype=float),
            )
            Y = np.asarray(mop.evaluation(X)["F"], dtype=float)
            frame = pd.DataFrame(Y, columns=[f"f{i + 1}" for i in range(Y.shape[1])])
            mis_set, structural, dominance = _candidate_pair(
                frame, misda_seed=misda_seed
            )
            records.append(
                _record_from_discovery(
                    family="optimization_control",
                    problem_id=name,
                    data=frame,
                    mis_set=mis_set,
                    structural=structural,
                    dominance=dominance,
                    metadata={
                        "sample_seed": int(sample_seed),
                        "screen_power": int(screen_power),
                        "n": int(Y.shape[0]),
                    },
                    bootstrap_replicates=bootstrap_replicates,
                    bootstrap_seed=int(
                        np.random.SeedSequence(
                            [bootstrap_seed, problem_position, int(sample_seed)]
                        ).generate_state(1)[0]
                    ),
                    known_safe=known_safe,
                )
            )
    return records


def _noisy_divergence_records(
    *,
    n=300,
    misda_seed=DEFAULT_SEED,
    problem_ids=TARGET_NOISY_PROBLEMS,
    replicate_seeds=NOISE_REPLICATE_SEEDS,
    sigmas=tuple(value for value in NOISE_SIGMAS if value > 0),
    divergences_per_problem=DEFAULT_NOISY_DIVERGENCES_PER_PROBLEM,
    bootstrap_replicates=DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
):
    wanted = set(problem_ids)
    counts = defaultdict(int)
    records = []
    iterator = impact._iter_noisy_robustness(
        n=int(n),
        misda_seed=int(misda_seed),
        replicate_seeds=tuple(int(value) for value in replicate_seeds),
        sigmas=tuple(float(value) for value in sigmas),
        problem_ids=wanted,
    )
    for case_position, case in enumerate(iterator):
        problem_id = case["problem_id"]
        if counts[problem_id] >= int(divergences_per_problem):
            continue
        mis_set, structural, dominance = _candidate_pair(
            case["data"], misda_seed=case["misda_seed"]
        )
        if tuple(structural.mis().indices) == tuple(dominance.mis().indices):
            continue
        counts[problem_id] += 1
        records.append(
            _record_from_discovery(
                family="noisy_controlled_divergence",
                problem_id=problem_id,
                data=case["data"],
                mis_set=mis_set,
                structural=structural,
                dominance=dominance,
                metadata=case["metadata"],
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=int(
                    np.random.SeedSequence(
                        [bootstrap_seed, case_position, counts[problem_id]]
                    ).generate_state(1)[0]
                ),
                truth=case["truth"],
            )
        )
        if all(
            counts[problem_id] >= int(divergences_per_problem)
            for problem_id in wanted
        ):
            break
    return records, {problem_id: int(counts[problem_id]) for problem_id in problem_ids}


def _summary(records):
    def summarize(group):
        if not group:
            return {"runs": 0}
        negatives = [record["resampling"]["negative_rate"] for record in group]
        reversals = [record["resampling"]["sign_reversal_rate"] for record in group]
        original = [record["resampling"]["original_delta"] for record in group]
        return {
            "runs": int(len(group)),
            "same_selected_mis": int(sum(record["same_selected_mis"] for record in group)),
            "median_original_delta": float(np.median(original)),
            "median_negative_rate": float(np.median(negatives)),
            "min_negative_rate": float(np.min(negatives)),
            "max_negative_rate": float(np.max(negatives)),
            "median_sign_reversal_rate": float(np.median(reversals)),
        }

    by_family = {}
    for family in dict.fromkeys(record["family"] for record in records):
        group = [record for record in records if record["family"] == family]
        by_family[family] = summarize(group)

    by_problem = {}
    for problem_id in dict.fromkeys(record["problem_id"] for record in records):
        group = [record for record in records if record["problem_id"] == problem_id]
        entry = summarize(group)
        outcomes = defaultdict(int)
        for record in group:
            if record["truth"] is not None:
                outcomes[record["truth"]["outcome"]] += 1
        if outcomes:
            entry["truth_outcomes"] = dict(sorted(outcomes.items()))
        controls = [
            record["analytical_control"]
            for record in group
            if record["analytical_control"] is not None
        ]
        if controls:
            entry["dominance_matches_known_safe"] = int(
                sum(control["dominance_matches_known_safe"] for control in controls)
            )
            entry["structural_matches_known_safe"] = int(
                sum(control["structural_matches_known_safe"] for control in controls)
            )
        by_problem[problem_id] = entry

    return {
        "overall": summarize(records),
        "by_family": by_family,
        "by_problem": by_problem,
    }


def run_probe(
    *,
    include_optimization_controls=True,
    include_noisy_controls=True,
    bootstrap_replicates=DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
    control_sample_seeds=CONTROL_SAMPLE_SEEDS,
    control_screen_power=CONTROL_SCREEN_POWER,
    n=300,
    noisy_problem_ids=TARGET_NOISY_PROBLEMS,
    noisy_divergences_per_problem=DEFAULT_NOISY_DIVERGENCES_PER_PROBLEM,
    noise_replicate_seeds=NOISE_REPLICATE_SEEDS,
    noise_sigmas=tuple(value for value in NOISE_SIGMAS if value > 0),
):
    records = []
    noisy_found = {}
    if include_optimization_controls:
        records.extend(
            _optimization_control_records(
                sample_seeds=tuple(int(value) for value in control_sample_seeds),
                screen_power=int(control_screen_power),
                bootstrap_replicates=int(bootstrap_replicates),
                bootstrap_seed=int(bootstrap_seed),
            )
        )
    if include_noisy_controls:
        noisy_records, noisy_found = _noisy_divergence_records(
            n=int(n),
            problem_ids=tuple(noisy_problem_ids),
            replicate_seeds=tuple(int(value) for value in noise_replicate_seeds),
            sigmas=tuple(float(value) for value in noise_sigmas),
            divergences_per_problem=int(noisy_divergences_per_problem),
            bootstrap_replicates=int(bootstrap_replicates),
            bootstrap_seed=int(bootstrap_seed),
        )
        records.extend(noisy_records)

    if not records:
        raise ValueError("The selected probe sections produced no records.")

    return {
        "format_version": FORMAT_VERSION,
        "suite": "ranking_resampling_probe",
        "definition": "DeltaD = D(S_dom) - D(S_struct); negative favors dominance_preservation",
        "parameters": {
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_seed": int(bootstrap_seed),
            "control_sample_seeds": [int(value) for value in control_sample_seeds],
            "control_screen_power": int(control_screen_power),
            "n": int(n),
            "noisy_problem_ids": list(noisy_problem_ids),
            "noisy_divergences_per_problem": int(noisy_divergences_per_problem),
            "noise_replicate_seeds": [int(value) for value in noise_replicate_seeds],
            "noise_sigmas": [float(value) for value in noise_sigmas],
        },
        "software": software_versions(),
        "noisy_divergences_found": noisy_found,
        "summary": _summary(records),
        "records": records,
    }


def _print_summary(artifact):
    print("Ranking evidence resampling probe")
    print("=================================")
    print(artifact["definition"])
    for problem_id, summary in artifact["summary"]["by_problem"].items():
        print(
            f"{problem_id}: runs={summary['runs']}, "
            f"median P(DeltaD<0)={summary['median_negative_rate']:.3f}, "
            f"median reversal={summary['median_sign_reversal_rate']:.3f}, "
            f"median DeltaD={summary['median_original_delta']:.6f}"
        )
        if "truth_outcomes" in summary:
            print(f"  truth outcomes: {summary['truth_outcomes']}")
        if "dominance_matches_known_safe" in summary:
            print(
                "  known-safe matches: "
                f"dominance={summary['dominance_matches_known_safe']}/"
                f"{summary['runs']}, structural={summary['structural_matches_known_safe']}/"
                f"{summary['runs']}"
            )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--section",
        action="append",
        choices=("optimization_controls", "noisy_controls"),
        help="Run only this section; may be supplied more than once.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small deterministic configuration for smoke testing.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    sections = set(args.section or ("optimization_controls", "noisy_controls"))
    quick = bool(args.quick)
    artifact = run_probe(
        include_optimization_controls="optimization_controls" in sections,
        include_noisy_controls="noisy_controls" in sections,
        bootstrap_replicates=8 if quick else int(args.bootstrap_replicates),
        control_sample_seeds=(CONTROL_SAMPLE_SEEDS[:1] if quick else CONTROL_SAMPLE_SEEDS),
        control_screen_power=6 if quick else CONTROL_SCREEN_POWER,
        n=64 if quick else 300,
        noisy_problem_ids=(TARGET_NOISY_PROBLEMS[:1] if quick else TARGET_NOISY_PROBLEMS),
        noisy_divergences_per_problem=1 if quick else DEFAULT_NOISY_DIVERGENCES_PER_PROBLEM,
        noise_replicate_seeds=(NOISE_REPLICATE_SEEDS[:2] if quick else NOISE_REPLICATE_SEEDS),
        noise_sigmas=((0.10, 0.20) if quick else tuple(value for value in NOISE_SIGMAS if value > 0)),
    )
    write_json(artifact, args.output)
    _print_summary(artifact)


if __name__ == "__main__":
    main()
