# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Boundary stress test for noisy DTLZ5 ranking/support semantics.

This is deliberately *not* a realistic-noise benchmark.  It probes a high-noise
condition discovered by the #84 screening where candidate support and ranking
point in different directions, then evaluates the downstream consequences in
the clean original MOP.
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
    _calibrated_ground_truth,
    _sample_screening,
    problem_suite,
    run_seed,
    summarize_runs,
)


FORMAT_VERSION = 1
DEFAULT_SIGMA = 2.0
DEFAULT_OBSERVATION_SEED = 101
DEFAULT_POPULATION = 240
DEFAULT_GENERATIONS = 800
DEFAULT_CHECKPOINTS = (100, 200, 400, 800)
DEFAULT_MOEA_SEEDS = (321, 654, 987, 135, 246)


def screen_condition(*, sigma=DEFAULT_SIGMA, observation_seed=DEFAULT_OBSERVATION_SEED):
    mop = problem_suite()["DTLZ5"]
    _, clean = _sample_screening(mop, power=9, seed=MISDA_SEED)
    noisy = observe(clean, sigma=float(sigma), observation_seed=int(observation_seed))
    return mop, _screen_frame(
        noisy,
        name=f"DTLZ5 stress sigma={sigma} observation_seed={observation_seed}",
        seed=MISDA_SEED,
    )


def run_audit(
    *,
    sigma=DEFAULT_SIGMA,
    observation_seed=DEFAULT_OBSERVATION_SEED,
    population=DEFAULT_POPULATION,
    generations=DEFAULT_GENERATIONS,
    checkpoints=DEFAULT_CHECKPOINTS,
    moea_seeds=DEFAULT_MOEA_SEEDS,
    ref_dirs_seed=REF_DIRS_SEED,
):
    mop, screening = screen_condition(
        sigma=float(sigma), observation_seed=int(observation_seed)
    )
    gt, calibration_sidecar = _calibrated_ground_truth(mop)
    checkpoints = tuple(sorted(set(int(value) for value in checkpoints)))
    runs = [
        run_seed(
            "DTLZ5-stress",
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
        "format_version": FORMAT_VERSION,
        "problem": "DTLZ5",
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


def _print(result):
    print(
        f"DTLZ5 high-noise stress: sigma={result['sigma']}, "
        f"observation_seed={result['observation_seed']}"
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
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--observation-seed", type=int, default=DEFAULT_OBSERVATION_SEED)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_audit(sigma=args.sigma, observation_seed=args.observation_seed)
    _print(result)
    if args.output is not None:
        # Drop live MISSet/ranking objects retained by the screening helper.
        serializable = dict(result)
        serializable["screening"] = {
            key: value
            for key, value in result["screening"].items()
            if key not in {"mis_set", "rankings", "X", "F"}
        }
        args.output.write_text(
            json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"Wrote {args.output}")
    return result


if __name__ == "__main__":
    main()
