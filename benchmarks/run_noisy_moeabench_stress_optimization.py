# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""End-to-end stress audit for noisy DTLZ5/DPF1 discovery conditions.

This runner is intentionally separate from ``run_noisy_ranking_optimization``:
that module reproduces the controlled ``noisy_robustness`` battery, whereas
this module probes the MoeaBench DTLZ5/DPF1 controls from issue #83.  Observation
noise affects only the discovery matrix; optimization and evaluation use the
clean underlying MOP and a common calibrated GT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import misda
from benchmarks.run_noisy_ranking_optimization import _mean_degradation
from benchmarks.run_noisy_ranking_safety import _screen_frame, observe
from benchmarks.run_ranking_policy_optimization import (
    MISDA_SEED,
    POLICIES,
    REF_DIRS_SEED,
    SCREEN_POWER,
    _calibrated_ground_truth,
    _sample_screening,
    problem_suite,
    run_seed,
    summarize_runs,
)


FORMAT_VERSION = 1
DEFAULT_POPULATION = 240
DEFAULT_GENERATIONS = 800
DEFAULT_CHECKPOINTS = (100, 200, 400, 800)
DEFAULT_MOEA_SEEDS = (321, 654, 987, 135, 246)


def screen_condition(problem, *, sigma, observation_seed, screen_power=SCREEN_POWER):
    suite = problem_suite()
    if problem not in suite:
        raise ValueError(f"unknown problem {problem!r}; expected one of {sorted(suite)}")
    mop = suite[problem]
    _, clean = _sample_screening(mop, power=int(screen_power), seed=MISDA_SEED)
    noisy = observe(
        clean,
        sigma=float(sigma),
        observation_seed=int(observation_seed),
    )
    screening = _screen_frame(
        noisy,
        name=f"{problem} noisy stress sigma={sigma} seed={observation_seed}",
        seed=MISDA_SEED,
    )
    return mop, screening


def run_condition(
    problem,
    *,
    sigma,
    observation_seed,
    population=DEFAULT_POPULATION,
    generations=DEFAULT_GENERATIONS,
    checkpoints=DEFAULT_CHECKPOINTS,
    moea_seeds=DEFAULT_MOEA_SEEDS,
    ref_dirs_seed=REF_DIRS_SEED,
):
    mop, screening = screen_condition(
        problem,
        sigma=float(sigma),
        observation_seed=int(observation_seed),
    )
    gt, calibration_sidecar = _calibrated_ground_truth(mop)
    checkpoints = tuple(sorted(set(int(value) for value in checkpoints)))
    runs = [
        run_seed(
            f"{problem}-noisy-stress",
            mop,
            screening,
            gt,
            generations=int(generations),
            checkpoints=checkpoints,
            population=int(population),
            moea_seed=int(seed),
            ref_dirs_seed=int(ref_dirs_seed),
        )
        for seed in moea_seeds
    ]
    return {
        "problem": problem,
        "stress_only": True,
        "sigma": float(sigma),
        "observation_seed": int(observation_seed),
        "screening": screening,
        "optimization": {
            "population": int(population),
            "generations": int(generations),
            "checkpoints": list(checkpoints),
            "moea_seeds": [int(seed) for seed in moea_seeds],
            "ref_dirs_seed": int(ref_dirs_seed),
            "gt_size": int(len(gt)),
            "calibration_sidecar": calibration_sidecar,
            "runs": runs,
            "mean_final": summarize_runs(runs),
            "mean_degradation_vs_full": {
                policy: _mean_degradation(runs, policy) for policy in POLICIES
            },
        },
    }


def _parse_condition(value):
    try:
        problem, sigma, observation_seed = value.split(":", 2)
        return problem, float(sigma), int(observation_seed)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "condition must be PROBLEM:SIGMA:OBSERVATION_SEED"
        ) from exc


def _print_result(result):
    print(
        f"\n=== {result['problem']} stress sigma={result['sigma']} "
        f"observation_seed={result['observation_seed']} ==="
    )
    print(
        f"structural dimension={result['screening']['analysis']['structural_dimension']}; "
        f"same selected MIS={result['screening']['same_selected_mis']}"
    )
    for policy in POLICIES:
        selected = result["screening"]["policies"][policy]
        metrics = result["optimization"]["mean_final"][policy]
        loss = result["optimization"]["mean_degradation_vs_full"][policy]
        print(
            f"{policy}: {selected['selected_labels']}, "
            f"trustworthy={selected['assessment']['trustworthy']}, "
            f"IGD+={metrics['igdplus']:.6g}, HV={metrics['relative_hv']:.6g}, "
            f"IGD loss vs Full={loss['igdplus_loss']:+.6g}, "
            f"HV loss vs Full={loss['relative_hv_loss']:+.6g}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        action="append",
        type=_parse_condition,
        required=True,
        help="PROBLEM:SIGMA:OBSERVATION_SEED; may be repeated",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--checkpoint", action="append", type=int)
    parser.add_argument("--moea-seed", action="append", type=int)
    parser.add_argument("--ref-dirs-seed", type=int, default=REF_DIRS_SEED)
    args = parser.parse_args(argv)

    checkpoints = tuple(args.checkpoint) if args.checkpoint else DEFAULT_CHECKPOINTS
    moea_seeds = tuple(args.moea_seed) if args.moea_seed else DEFAULT_MOEA_SEEDS
    results = [
        run_condition(
            problem,
            sigma=sigma,
            observation_seed=observation_seed,
            population=args.population,
            generations=args.generations,
            checkpoints=checkpoints,
            moea_seeds=moea_seeds,
            ref_dirs_seed=args.ref_dirs_seed,
        )
        for problem, sigma, observation_seed in args.condition
    ]
    payload = {
        "format_version": FORMAT_VERSION,
        "question": (
            "When both ranking choices remain supported under strongly noisy "
            "MoeaBench discovery, which Reduced treatment degrades less relative "
            "to Full in the clean original objective space?"
        ),
        "results": results,
    }
    for result in results:
        _print_result(result)
    if args.output is not None:
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"\nWrote {args.output}")
    return payload


if __name__ == "__main__":
    main()
