# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Calibrate the Full NSGA-III treatment before ranking-policy comparison.

This probe varies population/reference-direction resolution while keeping the
optimizer implementation, GT, run seed, and original objective formulation
fixed. It is intentionally Full-only: no ranking policy is involved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks import run_ranking_policy_optimization as audit


POPULATIONS = (60, 120, 240)
GENERATIONS = 800
CHECKPOINTS = (100, 200, 400, 800)
MOEA_SEEDS = (321,)
REF_DIRS_SEED = audit.REF_DIRS_SEED


def _run_full(
    name,
    mop,
    gt,
    *,
    population,
    generations,
    checkpoints,
    moea_seed,
    ref_dirs_seed,
):
    X0 = audit._paired_initial_population(mop, size=population, seed=moea_seed)
    experiment = audit._make_experiment(
        mop,
        X0=X0,
        generations=generations,
        population=population,
        seed=moea_seed,
        ref_dirs_seed=ref_dirs_seed,
        name=(
            f"{name} — Full calibration — pop {population} — seed {moea_seed}"
        ),
    )
    trajectory = []
    for generation in checkpoints:
        front = audit._front_at_generation(experiment, mop, generation)
        trajectory.append(
            {
                "generation": int(generation),
                **audit._front_metrics(front, gt),
            }
        )
    return {
        "population": int(population),
        "moea_seed": int(moea_seed),
        "evaluations_at_final_checkpoint": int(population * checkpoints[-1]),
        "trajectory": trajectory,
        "final": trajectory[-1],
    }


def run_case(
    name,
    mop,
    *,
    populations=POPULATIONS,
    generations=GENERATIONS,
    checkpoints=CHECKPOINTS,
    moea_seeds=MOEA_SEEDS,
    ref_dirs_seed=REF_DIRS_SEED,
):
    checkpoints = audit._validate_checkpoints(checkpoints, generations)
    gt, calibration_sidecar = audit._calibrated_ground_truth(mop)
    runs = []
    for population in populations:
        if int(population) < 2:
            raise ValueError("population must be at least 2")
        for moea_seed in moea_seeds:
            runs.append(
                _run_full(
                    name,
                    mop,
                    gt,
                    population=int(population),
                    generations=generations,
                    checkpoints=checkpoints,
                    moea_seed=int(moea_seed),
                    ref_dirs_seed=ref_dirs_seed,
                )
            )
    return {
        "problem": name,
        "M": int(mop.M),
        "N": int(mop.N),
        "gt_size": int(len(gt)),
        "calibration_sidecar": calibration_sidecar,
        "generations": int(generations),
        "checkpoints": [int(value) for value in checkpoints],
        "ref_dirs_seed": int(ref_dirs_seed),
        "runs": runs,
    }


def run_calibration(
    *,
    problem_names=("DTLZ5", "DPF1"),
    populations=POPULATIONS,
    generations=GENERATIONS,
    checkpoints=CHECKPOINTS,
    moea_seeds=MOEA_SEEDS,
    ref_dirs_seed=REF_DIRS_SEED,
):
    suite = audit.problem_suite()
    unknown = sorted(set(problem_names) - set(suite))
    if unknown:
        raise ValueError(f"unknown problem(s): {', '.join(unknown)}")
    return {
        "format_version": 1,
        "purpose": (
            "Full-only NSGA-III population/reference-direction and budget calibration"
        ),
        "results": [
            run_case(
                name,
                suite[name],
                populations=populations,
                generations=generations,
                checkpoints=checkpoints,
                moea_seeds=moea_seeds,
                ref_dirs_seed=ref_dirs_seed,
            )
            for name in problem_names
        ],
    }


def _print_result(payload):
    for case in payload["results"]:
        print(f"\n=== {case['problem']}: Full-only calibration ===")
        for run in case["runs"]:
            print(
                f"population={run['population']}, seed={run['moea_seed']}, "
                f"final_evals={run['evaluations_at_final_checkpoint']}"
            )
            for row in run["trajectory"]:
                print(
                    f"  g={row['generation']:4d}: "
                    f"GD+={row['gdplus']:.6g}, "
                    f"IGD+={row['igdplus']:.6g}, "
                    f"HV={row['relative_hv']:.6g}, ND={row['nd_size']}"
                )
        print("No automatic convergence threshold is applied.")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem",
        action="append",
        choices=("DTLZ5", "DPF1"),
        dest="problems",
    )
    parser.add_argument(
        "--population",
        action="append",
        type=int,
        dest="populations",
    )
    parser.add_argument("--generations", type=int, default=GENERATIONS)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=int,
        dest="checkpoints",
    )
    parser.add_argument(
        "--moea-seed",
        action="append",
        type=int,
        dest="moea_seeds",
    )
    parser.add_argument("--ref-dirs-seed", type=int, default=REF_DIRS_SEED)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    payload = run_calibration(
        problem_names=tuple(args.problems or ("DTLZ5", "DPF1")),
        populations=tuple(args.populations or POPULATIONS),
        generations=args.generations,
        checkpoints=tuple(args.checkpoints or CHECKPOINTS),
        moea_seeds=tuple(args.moea_seeds or MOEA_SEEDS),
        ref_dirs_seed=args.ref_dirs_seed,
    )
    _print_result(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote {args.output}")
    return payload


if __name__ == "__main__":
    main()
