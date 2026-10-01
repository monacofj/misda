# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Screen ranking-policy behavior when only discovery Y is noisy.

The underlying MOP and downstream optimization problem remain clean. This first
stage of issue #84 does not run the MOEA; it maps where observation noise changes
MIS selection, support assessment, and policy disagreement so the expensive
end-to-end audit can focus on informative conditions.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import misda

from benchmarks.run_ranking_policy_optimization import (
    MISDA_SEED,
    POLICIES,
    SCREEN_POWER,
    _candidate_snapshot,
    _sample_screening,
    problem_suite,
)
from misda.benchmarks.validation import NOISE_REPLICATE_SEEDS, NOISE_SIGMAS


FORMAT_VERSION = 1


def observe(frame: pd.DataFrame, *, sigma: float, observation_seed: int) -> pd.DataFrame:
    """Apply the repository's scale-relative additive Gaussian observation model."""

    sigma = float(sigma)
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0.0:
        return frame.copy()

    values = frame.to_numpy(dtype=float)
    scales = (
        np.std(values, axis=0, ddof=1)
        if len(frame) > 1
        else np.zeros(values.shape[1], dtype=float)
    )
    epsilon = np.random.default_rng(int(observation_seed)).normal(size=values.shape)
    observed = values + sigma * scales[np.newaxis, :] * epsilon
    return pd.DataFrame(observed, columns=frame.columns, index=frame.index)


def _screen_frame(frame: pd.DataFrame, *, name: str, seed: int):
    mis_set = misda.discover(frame, name=name, seed=int(seed))
    mis_set.evaluate(metrics=("dominance",), candidates="all")
    rankings = {policy: misda.rank(mis_set, policy=policy) for policy in POLICIES}

    selected = []
    for ranking in rankings.values():
        candidate = ranking.mis()
        if all(candidate is not observed for observed in selected):
            selected.append(candidate)
    mis_set.evaluate(metrics=("linear", "pareto"), candidates=selected)

    snapshots = {
        policy: _candidate_snapshot(mis_set, ranking)
        for policy, ranking in rankings.items()
    }
    analysis = mis_set.analysis
    return {
        "policies": snapshots,
        "same_selected_mis": (
            snapshots[misda.SIZE_SPAN]["selected_indices"]
            == snapshots[misda.DOMINANCE_PRESERVATION]["selected_indices"]
        ),
        "dominance_delta": float(
            snapshots[misda.DOMINANCE_PRESERVATION]["dominance"]["new_dominance_rate"]
            - snapshots[misda.SIZE_SPAN]["dominance"]["new_dominance_rate"]
        ),
        "analysis": {
            "original_dimension": int(analysis.original_dimension),
            "latent_dimension": int(analysis.latent_dimension),
            "structural_dimension": int(analysis.structural_dimension),
            "aggressiveness": float(analysis.aggressiveness),
            "separation_status": analysis.separation_status.value,
            "n_mis": int(len(mis_set)),
        },
    }


def _selection_key(snapshot):
    return tuple(int(i) for i in snapshot["selected_indices"])


def _summarize(rows):
    by_problem = {}
    for problem in sorted({row["problem"] for row in rows}):
        problem_rows = [row for row in rows if row["problem"] == problem]
        sigma_summary = {}
        for sigma in sorted({row["sigma"] for row in problem_rows}):
            group = [row for row in problem_rows if row["sigma"] == sigma]
            sigma_summary[str(sigma)] = {
                "n": len(group),
                "policy_disagreement": sum(not row["same_selected_mis"] for row in group),
                "size_span_changed_from_clean": sum(
                    not row["policies"][misda.SIZE_SPAN]["matches_clean_selection"]
                    for row in group
                ),
                "dominance_changed_from_clean": sum(
                    not row["policies"][misda.DOMINANCE_PRESERVATION]["matches_clean_selection"]
                    for row in group
                ),
                "size_span_trustworthy": sum(
                    row["policies"][misda.SIZE_SPAN]["assessment"]["trustworthy"]
                    for row in group
                ),
                "dominance_trustworthy": sum(
                    row["policies"][misda.DOMINANCE_PRESERVATION]["assessment"]["trustworthy"]
                    for row in group
                ),
            }
        by_problem[problem] = sigma_summary

    transitions = Counter()
    for row in rows:
        for policy in POLICIES:
            before = tuple(row["clean"][policy])
            after = tuple(row["policies"][policy]["selected_indices"])
            if before != after:
                transitions[(row["problem"], policy, before, after)] += 1

    return {
        "by_problem_and_sigma": by_problem,
        "selection_transitions": [
            {
                "problem": key[0],
                "policy": key[1],
                "from": list(key[2]),
                "to": list(key[3]),
                "count": count,
            }
            for key, count in sorted(transitions.items(), key=lambda item: (-item[1], item[0]))
        ],
    }


def run_audit(
    *,
    screen_power: int = SCREEN_POWER,
    misda_seed: int = MISDA_SEED,
    sigmas=NOISE_SIGMAS,
    observation_seeds=NOISE_REPLICATE_SEEDS,
):
    rows = []
    clean_reference = {}

    for name, mop in problem_suite().items():
        _, clean_frame = _sample_screening(mop, power=screen_power, seed=misda_seed)
        clean = _screen_frame(
            clean_frame,
            name=f"{name} clean ranking-policy screening",
            seed=misda_seed,
        )
        clean_reference[name] = {
            policy: _selection_key(clean["policies"][policy]) for policy in POLICIES
        }

        for observation_seed in tuple(int(value) for value in observation_seeds):
            for sigma in tuple(float(value) for value in sigmas):
                observed = observe(
                    clean_frame,
                    sigma=sigma,
                    observation_seed=observation_seed,
                )
                screened = _screen_frame(
                    observed,
                    name=f"{name} noisy ranking-policy screening",
                    seed=misda_seed,
                )
                policies = screened["policies"]
                for policy in POLICIES:
                    policies[policy]["matches_clean_selection"] = bool(
                        _selection_key(policies[policy]) == clean_reference[name][policy]
                    )

                rows.append(
                    {
                        "problem": name,
                        "sigma": sigma,
                        "observation_seed": observation_seed,
                        "clean": {
                            policy: list(clean_reference[name][policy]) for policy in POLICIES
                        },
                        "same_selected_mis": bool(screened["same_selected_mis"]),
                        "dominance_delta": float(screened["dominance_delta"]),
                        "analysis": screened["analysis"],
                        "policies": policies,
                    }
                )

    return {
        "format_version": FORMAT_VERSION,
        "question": (
            "How does observation noise in discovery Y change size_span and "
            "dominance_preservation before the end-to-end MOEA audit?"
        ),
        "noise_model": (
            "Y_j = Z_j + sigma * std(Z_j) * epsilon_j, epsilon_j ~ N(0,1)"
        ),
        "screen_power": int(screen_power),
        "misda_seed": int(misda_seed),
        "sigmas": [float(value) for value in sigmas],
        "observation_seeds": [int(value) for value in observation_seeds],
        "clean_reference": {
            problem: {policy: list(indices) for policy, indices in reference.items()}
            for problem, reference in clean_reference.items()
        },
        "rows": rows,
        "summary": _summarize(rows),
    }


def _print_summary(result):
    print("Noisy ranking-policy screening audit")
    for problem, sigma_rows in result["summary"]["by_problem_and_sigma"].items():
        print(f"\n=== {problem} ===")
        for sigma, row in sigma_rows.items():
            print(
                f"sigma={sigma}: disagreement={row['policy_disagreement']}/{row['n']}, "
                f"size changed={row['size_span_changed_from_clean']}/{row['n']}, "
                f"dominance changed={row['dominance_changed_from_clean']}/{row['n']}, "
                f"trustworthy(size/dom)={row['size_span_trustworthy']}/"
                f"{row['dominance_trustworthy']}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--screen-power", type=int, default=SCREEN_POWER)
    parser.add_argument("--misda-seed", type=int, default=MISDA_SEED)
    parser.add_argument("--sigma", action="append", type=float)
    parser.add_argument("--observation-seed", action="append", type=int)
    args = parser.parse_args(argv)

    result = run_audit(
        screen_power=args.screen_power,
        misda_seed=args.misda_seed,
        sigmas=tuple(args.sigma) if args.sigma else NOISE_SIGMAS,
        observation_seeds=(
            tuple(args.observation_seed)
            if args.observation_seed
            else NOISE_REPLICATE_SEEDS
        ),
    )
    _print_summary(result)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote {args.output}")
    return result


if __name__ == "__main__":
    main()
