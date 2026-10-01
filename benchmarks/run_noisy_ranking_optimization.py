# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""End-to-end Full-vs-Reduced audit for selected noisy controlled conditions.

Discovery reproduces the repository's ``noisy_robustness`` protocol exactly:
a clean controlled MOP is sampled once for a replicate, observation noise is
added only to Y, and both ranking policies select from that same noisy Y.
Optimization is then performed on the underlying clean MOP.  The primary safety
quantity is oriented degradation relative to Full in the original objective
space; positive loss always means that Reduced is worse than Full.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import moeabench as mb

import misda
from benchmarks.run_ranking_policy_impact import _audit_case
from benchmarks.run_ranking_policy_optimization import (
    MISDA_SEED,
    POLICIES,
    REF_DIRS_SEED,
    _nd_front,
    run_seed,
    summarize_runs,
)
from misda.benchmarks import PROBLEM_BY_ID, diagnostic_truth
from misda.benchmarks.validation import NOISY_ROBUSTNESS_PROBLEM_IDS


FORMAT_VERSION = 2
DEFAULT_POPULATION = 240
DEFAULT_GENERATIONS = 400
DEFAULT_CHECKPOINTS = (50, 100, 200, 400)
DEFAULT_MOEA_SEEDS = (321, 654, 987, 135, 246)
DEFAULT_DISCOVERY_N = 300
DEFAULT_GT_SIZE = 2000

# The first end-to-end noisy-discovery control uses a genuine bounded trade-off
# MOP.  More controlled problems can be added here only when their optimization
# domains are explicit rather than inferred from a sampling distribution.
CONTROLLED_DOMAINS = {
    "antagonistic_nonlinear_groups": {
        "columns": ("x",),
        "xl": (0.0,),
        "xu": (1.0,),
    },
}


class ControlledDiagnosticMOP(mb.mops.BaseMop):
    """Adapt an executable MISDA diagnostic problem to MoeaBench."""

    def __init__(self, problem, *, columns, xl, xu):
        self.problem = problem
        self.columns = tuple(columns)
        probe = problem.evaluate(
            pd.DataFrame([np.asarray(xl, dtype=float)], columns=self.columns)
        )
        super().__init__(
            name=f"MISDA-{problem.id}",
            M=int(probe.shape[1]),
            N=len(self.columns),
            xl=np.asarray(xl, dtype=float),
            xu=np.asarray(xu, dtype=float),
        )

    def evaluation(self, X, n_ieq_constr=0):
        frame = pd.DataFrame(np.asarray(X, dtype=float), columns=self.columns)
        F = self.problem.evaluate(frame).to_numpy(dtype=float)
        return {"F": F}

    def ps(self, n_points=100):
        if self.N != 1:
            raise NotImplementedError("analytical PS is implemented only for 1-D controls")
        return np.linspace(self.xl[0], self.xu[0], int(n_points))[:, None]


def _controlled_mop(problem_id):
    if problem_id not in CONTROLLED_DOMAINS:
        raise ValueError(
            f"{problem_id!r} has no explicit optimization domain in this audit; "
            f"expected one of {sorted(CONTROLLED_DOMAINS)}"
        )
    spec = CONTROLLED_DOMAINS[problem_id]
    return ControlledDiagnosticMOP(
        PROBLEM_BY_ID[problem_id],
        columns=spec["columns"],
        xl=spec["xl"],
        xu=spec["xu"],
    )


def _noisy_discovery_case(problem_id, *, sigma, replicate_seed, n, misda_seed):
    """Reproduce one established noisy_robustness discovery condition."""

    if problem_id not in NOISY_ROBUSTNESS_PROBLEM_IDS:
        raise ValueError(f"{problem_id!r} is not in the noisy-robustness battery")
    problem_position = NOISY_ROBUSTNESS_PROBLEM_IDS.index(problem_id)
    problem = PROBLEM_BY_ID[problem_id]
    sample_seed = int(
        np.random.SeedSequence([int(replicate_seed), problem_position, 1])
        .generate_state(1)[0]
    )
    observation_sequence = np.random.SeedSequence(
        [int(replicate_seed), problem_position, 2]
    )
    X = problem.sample(N=int(n), seed=sample_seed)
    Z = problem.evaluate(X)
    epsilon = np.random.default_rng(observation_sequence).normal(size=Z.shape)
    Y = problem.observe(Z, sigma=float(sigma), standard_noise=epsilon)
    truth = diagnostic_truth(problem, Z)

    audit = _audit_case(
        suite="noisy_optimization",
        problem_id=problem_id,
        data=Y,
        truth=truth,
        misda_seed=int(misda_seed),
        metadata={
            "replicate_seed": int(replicate_seed),
            "sample_seed": sample_seed,
            "sigma": float(sigma),
        },
    )
    return audit


def _analytic_gt(mop, *, size=DEFAULT_GT_SIZE):
    """Dense clean Pareto reference for the 1-D antagonistic control."""

    if mop.N != 1:
        raise NotImplementedError("dense analytic GT is implemented only for 1-D controls")
    X = np.linspace(mop.xl[0], mop.xu[0], int(size))[:, None]
    return _nd_front(mop.evaluation(X)["F"])


def _degradation_vs_full(reduced, full):
    """Return oriented loss: positive values always mean Reduced is worse."""

    return {
        "gdplus_loss": float(reduced["gdplus"] - full["gdplus"]),
        "igdplus_loss": float(reduced["igdplus"] - full["igdplus"]),
        "relative_hv_loss": float(full["relative_hv"] - reduced["relative_hv"]),
    }


def _mean_degradation(runs, policy):
    losses = [
        _degradation_vs_full(
            row["final"]["arms"][policy],
            row["final"]["arms"]["full"],
        )
        for row in runs
    ]
    return {
        key: float(np.mean([loss[key] for loss in losses]))
        for key in ("gdplus_loss", "igdplus_loss", "relative_hv_loss")
    }


def run_condition(
    *,
    problem_id: str,
    sigma: float,
    replicate_seed: int,
    discovery_n: int = DEFAULT_DISCOVERY_N,
    misda_seed: int = MISDA_SEED,
    population: int = DEFAULT_POPULATION,
    generations: int = DEFAULT_GENERATIONS,
    checkpoints=DEFAULT_CHECKPOINTS,
    moea_seeds=DEFAULT_MOEA_SEEDS,
    ref_dirs_seed: int = REF_DIRS_SEED,
    gt_size: int = DEFAULT_GT_SIZE,
):
    audit = _noisy_discovery_case(
        problem_id,
        sigma=float(sigma),
        replicate_seed=int(replicate_seed),
        n=int(discovery_n),
        misda_seed=int(misda_seed),
    )
    mop = _controlled_mop(problem_id)
    gt = _analytic_gt(mop, size=int(gt_size))
    checkpoints = tuple(sorted(set(int(value) for value in checkpoints)))

    # run_seed needs only selected_indices from the policy snapshots; passing the
    # audit policy records keeps the exact Revisão-09 selections intact.
    screening = {"policies": audit["policies"]}
    runs = [
        run_seed(
            problem_id,
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
        "problem_id": problem_id,
        "sigma": float(sigma),
        "replicate_seed": int(replicate_seed),
        "static_audit": audit,
        "optimization": {
            "population": int(population),
            "generations": int(generations),
            "checkpoints": list(checkpoints),
            "moea_seeds": [int(seed) for seed in moea_seeds],
            "ref_dirs_seed": int(ref_dirs_seed),
            "gt_size": int(len(gt)),
            "runs": runs,
            "mean_final": summarize_runs(runs),
            "mean_degradation_vs_full": {
                policy: _mean_degradation(runs, policy) for policy in POLICIES
            },
        },
    }


def _parse_condition(value):
    try:
        problem_id, sigma, replicate_seed = value.split(":", 2)
        return problem_id, float(sigma), int(replicate_seed)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "condition must be PROBLEM_ID:SIGMA:REPLICATE_SEED"
        ) from exc


def _print_result(result):
    comparison = result["static_audit"]["comparison"]
    print(
        f"\n=== {result['problem_id']} sigma={result['sigma']} "
        f"replicate={result['replicate_seed']} ==="
    )
    print(
        f"static truth outcome: {comparison['truth_outcome']}; "
        f"observed dominance delta={comparison['dominance_delta']:.6g}"
    )
    for policy in POLICIES:
        selected = result["static_audit"]["policies"][policy]["selected_labels"]
        trusted = result["static_audit"]["policies"][policy]["reduction_assessment"]["trustworthy"]
        metrics = result["optimization"]["mean_final"][policy]
        loss = result["optimization"]["mean_degradation_vs_full"][policy]
        print(
            f"{policy}: {selected}, trustworthy={trusted}, "
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
        help="PROBLEM_ID:SIGMA:REPLICATE_SEED; may be repeated",
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
            problem_id=problem_id,
            sigma=sigma,
            replicate_seed=replicate_seed,
            population=args.population,
            generations=args.generations,
            checkpoints=checkpoints,
            moea_seeds=moea_seeds,
            ref_dirs_seed=args.ref_dirs_seed,
        )
        for problem_id, sigma, replicate_seed in args.condition
    ]
    payload = {
        "format_version": FORMAT_VERSION,
        "question": (
            "When noisy discovery changes the representative MIS, is the resulting "
            "Reduced optimization materially worse than Full in the clean original "
            "objective space?"
        ),
        "results": results,
    }
    for result in results:
        _print_result(result)
    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote {args.output}")
    return payload


if __name__ == "__main__":
    main()
