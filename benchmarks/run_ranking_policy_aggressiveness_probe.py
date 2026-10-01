"""Probe whether ranking-policy divergences are explained by aggressiveness.

This experiment is deliberately benchmark-only.  It first reproduces the
canonical noisy-robustness cases at aggressiveness=1.0 and keeps only cases in
which size_span and dominance_preservation choose different MISs.  Those same
observed matrices are then rediscovered over a deterministic aggressiveness
grid.  Benchmark truth is attached only after each policy has selected a
candidate.

The probe is intended to answer whether a future profile(Y) would expose the
noisy ranking problem as threshold sensitivity, or whether the preference of
`dominance_preservation` persists across the calibrated aggressiveness range.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import misda
from misda.benchmark import DEFAULT_SEED, matrix_sha256, software_versions, write_json
from misda.benchmarks.validation import (
    NOISE_REPLICATE_SEEDS,
    NOISE_SIGMAS,
)

from benchmarks import run_ranking_policy_impact as impact


FORMAT_VERSION = 1
DEFAULT_AGGRESSIVENESS_GRID = tuple(round(value, 2) for value in np.linspace(0.0, 1.0, 21))


def _state(case, aggressiveness):
    data = case["data"]
    truth = case["truth"]
    mis_set = misda.discover(
        data,
        name=truth.get("name"),
        seed=case["misda_seed"],
        aggressiveness=float(aggressiveness),
    )
    mis_set.evaluate(metrics=("dominance",), candidates="all")
    size_span = misda.rank(mis_set, policy=misda.SIZE_SPAN)
    dominance = misda.rank(mis_set, policy=misda.DOMINANCE_PRESERVATION)

    selected = []
    for ranking in (size_span, dominance):
        if ranking.selected is not None and all(
            observed is not ranking.selected for observed in selected
        ):
            selected.append(ranking.selected)
    mis_set.evaluate(metrics=("pareto",), candidates=selected)

    baseline = impact._policy_evidence(
        mis_set,
        size_span,
        truth,
        data=data,
    )
    candidate = impact._policy_evidence(
        mis_set,
        dominance,
        truth,
        data=data,
    )
    return {
        "aggressiveness": float(aggressiveness),
        "alpha": float(mis_set.analysis.alpha),
        "structural_dimension": int(mis_set.analysis.structural_dimension),
        "latent_dimension": int(mis_set.analysis.latent_dimension),
        "n_mis": len(mis_set),
        "policies": {
            misda.SIZE_SPAN: baseline,
            misda.DOMINANCE_PRESERVATION: candidate,
        },
        "comparison": {
            "same_selected_mis": (
                baseline["selected_indices"] == candidate["selected_indices"]
            ),
            "same_selected_dimension": (
                baseline["selected_dimension"] == candidate["selected_dimension"]
            ),
            "truth_outcome": impact._truth_outcome(baseline, candidate),
            "dominance_delta": float(
                candidate["dominance"]["new_dominance_rate"]
                - baseline["dominance"]["new_dominance_rate"]
            ),
            "trust_changed_between_policies": (
                baseline["reduction_assessment"]["status"]
                != candidate["reduction_assessment"]["status"]
            ),
        },
    }


def _record(case, grid):
    state_by_aggressiveness = {}
    state_one = _state(case, 1.0)
    if state_one["comparison"]["same_selected_mis"]:
        return None
    state_by_aggressiveness[1.0] = state_one

    for aggressiveness in grid:
        value = float(aggressiveness)
        if np.isclose(value, 1.0, rtol=0.0, atol=1e-15):
            continue
        state_by_aggressiveness[value] = _state(case, value)

    states = [state_by_aggressiveness[value] for value in sorted(state_by_aggressiveness)]
    at_one = state_by_aggressiveness[1.0]
    dominance_at_one = at_one["policies"][misda.DOMINANCE_PRESERVATION]
    dimension_at_one = dominance_at_one["selected_dimension"]
    trust_at_one = dominance_at_one["reduction_assessment"]["status"]

    dominance_selections = {
        tuple(state["policies"][misda.DOMINANCE_PRESERVATION]["selected_indices"])
        for state in states
    }
    size_span_selections = {
        tuple(state["policies"][misda.SIZE_SPAN]["selected_indices"])
        for state in states
    }
    structural_dimensions = {state["structural_dimension"] for state in states}
    selected_dimensions = {
        state["policies"][misda.DOMINANCE_PRESERVATION]["selected_dimension"]
        for state in states
    }
    trust_statuses = {
        state["policies"][misda.DOMINANCE_PRESERVATION]["reduction_assessment"]["status"]
        for state in states
    }

    lower_states = [state for state in states if state["aggressiveness"] < 1.0]
    regression_at_one = at_one["comparison"]["truth_outcome"] == "regressed"
    lower_non_regressed = [
        state
        for state in lower_states
        if state["comparison"]["truth_outcome"] != "regressed"
    ]
    lower_non_regressed_same_dimension = [
        state
        for state in lower_non_regressed
        if state["policies"][misda.DOMINANCE_PRESERVATION]["selected_dimension"]
        == dimension_at_one
    ]

    profile_signal = bool(
        len(dominance_selections) > 1
        or len(structural_dimensions) > 1
        or len(selected_dimensions) > 1
        or len(trust_statuses) > 1
    )
    return {
        "problem_id": case["problem_id"],
        "input_sha256": matrix_sha256(case["data"]),
        "metadata": dict(case["metadata"]),
        "at_aggressiveness_one": {
            "truth_outcome": at_one["comparison"]["truth_outcome"],
            "size_span_selected": at_one["policies"][misda.SIZE_SPAN]["selected_labels"],
            "dominance_selected": dominance_at_one["selected_labels"],
            "selected_dimension": int(dimension_at_one),
            "trust_status": trust_at_one,
        },
        "profile_summary": {
            "profile_signal": profile_signal,
            "dominance_selection_changes": len(dominance_selections) > 1,
            "size_span_selection_changes": len(size_span_selections) > 1,
            "structural_dimension_changes": len(structural_dimensions) > 1,
            "dominance_selected_dimension_changes": len(selected_dimensions) > 1,
            "dominance_trust_changes": len(trust_statuses) > 1,
            "policies_reconverge_at_lower_aggressiveness": any(
                state["comparison"]["same_selected_mis"] for state in lower_states
            ),
            "regression_at_one": regression_at_one,
            "regression_persists_at_all_tested_points": bool(
                regression_at_one
                and all(
                    state["comparison"]["truth_outcome"] == "regressed"
                    for state in states
                )
            ),
            "lower_aggressiveness_avoids_regression": bool(
                regression_at_one and lower_non_regressed
            ),
            "lower_aggressiveness_avoids_regression_same_dimension": bool(
                regression_at_one and lower_non_regressed_same_dimension
            ),
            "lowest_non_regressed_aggressiveness": (
                min(state["aggressiveness"] for state in lower_non_regressed)
                if lower_non_regressed
                else None
            ),
            "highest_non_regressed_aggressiveness": (
                max(state["aggressiveness"] for state in lower_non_regressed)
                if lower_non_regressed
                else None
            ),
        },
        "states": states,
    }


def _summarize(records):
    def one(group):
        at_one = Counter(record["at_aggressiveness_one"]["truth_outcome"] for record in group)
        regressions = [
            record
            for record in group
            if record["profile_summary"]["regression_at_one"]
        ]
        return {
            "divergences_at_one": len(group),
            "truth_improved_at_one": at_one["improved"],
            "truth_regressed_at_one": at_one["regressed"],
            "truth_neutral_at_one": at_one["neutral"],
            "truth_mixed_at_one": at_one["mixed"],
            "dominance_selection_changes": sum(
                record["profile_summary"]["dominance_selection_changes"]
                for record in group
            ),
            "structural_dimension_changes": sum(
                record["profile_summary"]["structural_dimension_changes"]
                for record in group
            ),
            "trust_changes": sum(
                record["profile_summary"]["dominance_trust_changes"]
                for record in group
            ),
            "policies_reconverge_lower": sum(
                record["profile_summary"]["policies_reconverge_at_lower_aggressiveness"]
                for record in group
            ),
            "regressions_with_profile_signal": sum(
                record["profile_summary"]["profile_signal"] for record in regressions
            ),
            "regressions_without_profile_signal": sum(
                not record["profile_summary"]["profile_signal"] for record in regressions
            ),
            "regressions_persist_all_tested_points": sum(
                record["profile_summary"]["regression_persists_at_all_tested_points"]
                for record in regressions
            ),
            "regressions_avoided_lower": sum(
                record["profile_summary"]["lower_aggressiveness_avoids_regression"]
                for record in regressions
            ),
            "regressions_avoided_lower_same_dimension": sum(
                record["profile_summary"]["lower_aggressiveness_avoids_regression_same_dimension"]
                for record in regressions
            ),
        }

    by_problem = {}
    for problem_id in sorted({record["problem_id"] for record in records}):
        by_problem[problem_id] = one(
            [record for record in records if record["problem_id"] == problem_id]
        )
    return {"overall": one(records), "by_problem": by_problem}


def run_aggressiveness_probe(
    *,
    n=300,
    seed=DEFAULT_SEED,
    replicate_seeds=NOISE_REPLICATE_SEEDS,
    sigmas=NOISE_SIGMAS,
    aggressiveness_grid=DEFAULT_AGGRESSIVENESS_GRID,
    problem_ids=None,
):
    grid = tuple(sorted({float(value) for value in aggressiveness_grid}))
    if not grid or grid[0] < 0.0 or grid[-1] > 1.0:
        raise ValueError("aggressiveness_grid must contain values in [0, 1].")
    if not any(np.isclose(value, 1.0, rtol=0.0, atol=1e-15) for value in grid):
        grid = tuple(sorted((*grid, 1.0)))

    requested = None if problem_ids is None else set(problem_ids)
    cases = impact._iter_noisy_robustness(
        n=n,
        misda_seed=seed,
        replicate_seeds=tuple(int(value) for value in replicate_seeds),
        sigmas=tuple(float(value) for value in sigmas),
        problem_ids=requested,
    )

    records = []
    total_cases = 0
    for case in cases:
        total_cases += 1
        observed = _record(case, grid)
        if observed is not None:
            records.append(observed)

    return {
        "format_version": FORMAT_VERSION,
        "suite": "ranking_policy_aggressiveness_probe",
        "parameters": {
            "n": int(n),
            "seed": int(seed),
            "replicate_seeds": [int(value) for value in replicate_seeds],
            "sigmas": [float(value) for value in sigmas],
            "aggressiveness_grid": list(grid),
            "problem_ids": sorted(requested) if requested is not None else None,
        },
        "software": software_versions(),
        "screened_noisy_cases": int(total_cases),
        "summary": _summarize(records),
        "records": records,
    }


def _print_summary(artifact):
    summary = artifact["summary"]["overall"]
    print("Ranking policy × aggressiveness probe")
    print("=====================================")
    print(f"screened noisy cases : {artifact['screened_noisy_cases']}")
    print(f"divergences at a=1   : {summary['divergences_at_one']}")
    print(
        "truth at a=1         : "
        f"+{summary['truth_improved_at_one']} / "
        f"-{summary['truth_regressed_at_one']} / "
        f"neutral {summary['truth_neutral_at_one']}"
    )
    print(
        "a=1 regressions      : "
        f"profile-signal={summary['regressions_with_profile_signal']}, "
        f"no-signal={summary['regressions_without_profile_signal']}, "
        f"persist-all={summary['regressions_persist_all_tested_points']}, "
        f"avoided-lower={summary['regressions_avoided_lower']}, "
        f"avoided-same-dim={summary['regressions_avoided_lower_same_dimension']}"
    )
    print(f"policies reconverge lower: {summary['policies_reconverge_lower']}")
    for problem_id, values in artifact["summary"]["by_problem"].items():
        print(
            f"{problem_id}: divergences={values['divergences_at_one']}, "
            f"regressions={values['truth_regressed_at_one']}, "
            f"profile-signal={values['regressions_with_profile_signal']}, "
            f"persist-all={values['regressions_persist_all_tested_points']}, "
            f"avoided-lower={values['regressions_avoided_lower']}"
        )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--problem-id",
        action="append",
        dest="problem_ids",
        help="Restrict to this noisy-robustness problem id; may be repeated.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use N=64, one replicate, two sigma values and a five-point grid.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    quick = bool(args.quick)
    artifact = run_aggressiveness_probe(
        n=64 if quick else 300,
        replicate_seeds=(NOISE_REPLICATE_SEEDS[:1] if quick else NOISE_REPLICATE_SEEDS),
        sigmas=((0.0, 0.10) if quick else NOISE_SIGMAS),
        aggressiveness_grid=(
            (0.0, 0.25, 0.5, 0.75, 1.0)
            if quick
            else DEFAULT_AGGRESSIVENESS_GRID
        ),
        problem_ids=args.problem_ids,
    )
    write_json(artifact, args.output)
    _print_summary(artifact)


if __name__ == "__main__":
    main()
