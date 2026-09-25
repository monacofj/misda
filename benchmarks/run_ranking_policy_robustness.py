# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Resampling robustness for the experimental Pareto-retention ranking.

This experiment isolates sampling variability.  Each replicate draws a new
scrambled Sobol sample, while MISDA's permutation seed is kept fixed.  Truth
controls are attached only after discovery/evaluation/ranking.

Two questions are reported separately for each known control:

1. discovery: was the control subset present among the discovered MISs?
2. ranking: conditional on being present, was it in the first scientific
   Pareto-retention tie group?

The canonical size-span first group is recorded alongside the experimental
Pareto-retention first group for comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

import misda
import moeabench as mb


M = 10
SCREEN_POWER = 9
MISDA_SEED = 123
SAMPLE_SEEDS = (101, 202, 303, 404, 505, 606, 707, 808)


def _problem_suite():
    return {
        "DTLZ2": mb.mops.DTLZ2(M=M),
        "DTLZ5": mb.mops.DTLZ5(M=M),
        "DPF1": mb.mops.DPF1(M=M, D=2, K=5),
    }


def _controls(name):
    if name == "DTLZ5":
        return {
            "safe": (8, 9),
            "unsafe_witness": (0, 9),
        }
    if name == "DPF1":
        return {
            "safe_base": (0, 1),
            "previous_size_span": (1, 7),
        }
    return {}


def _sample_objectives(mop, *, power, sample_seed):
    sampler = qmc.Sobol(d=mop.N, scramble=True, seed=int(sample_seed))
    X = qmc.scale(
        sampler.random_base2(m=int(power)),
        np.asarray(mop.xl, dtype=float),
        np.asarray(mop.xu, dtype=float),
    )
    return np.asarray(mop.evaluation(X)["F"], dtype=float)


def _candidate_index(mis_set, indices):
    target = tuple(int(index) for index in indices)
    for index, candidate in enumerate(mis_set):
        if tuple(candidate.indices) == target:
            return index
    return None


def _top_group(ranking):
    return set(ranking.groups[0]) if ranking.groups else set()


def _evaluate_replicate(name, mop, *, power, sample_seed, misda_seed):
    F = _sample_objectives(mop, power=power, sample_seed=sample_seed)
    frame = pd.DataFrame(F, columns=[f"f{i + 1}" for i in range(mop.M)])

    mis_set = misda.discover(
        frame,
        name=f"{name} Pareto-retention resampling probe",
        seed=int(misda_seed),
    )
    mis_set.evaluate(metrics=("pareto",), candidates="all")

    structural = misda.rank(mis_set, policy=misda.SIZE_SPAN)
    pareto = misda.rank(mis_set, policy=misda.PARETO_RETENTION)
    structural_top = _top_group(structural)
    pareto_top = _top_group(pareto)

    selected = pareto.mis()
    row = {
        "problem": name,
        "sample_seed": int(sample_seed),
        "misda_seed": int(misda_seed),
        "n_screen": int(F.shape[0]),
        "n_mis": int(len(mis_set)),
        "latent_dimension": int(mis_set.analysis.latent_dimension),
        "structural_dimension": int(mis_set.analysis.structural_dimension),
        "pareto_selected": ",".join(map(str, selected.objectives)),
        "pareto_selected_size": int(selected.size),
        "pareto_selected_retention": float(selected.pareto.retention),
        "pareto_selected_exact": bool(selected.pareto.exact_preservation),
        "pareto_top_group_size": int(len(pareto.groups[0])),
        "size_span_top_group_size": int(len(structural.groups[0])),
    }

    # DTLZ2 is a negative control: no proper subset is globally safe.
    if name == "DTLZ2":
        row.update(
            {
                "any_observed_exact": bool(
                    any(
                        candidate.pareto.exact_preservation
                        for candidate in mis_set
                    )
                ),
                "max_observed_retention": float(
                    max(candidate.pareto.retention for candidate in mis_set)
                ),
            }
        )

    for control_name, control_indices in _controls(name).items():
        index = _candidate_index(mis_set, control_indices)
        present = index is not None
        row[f"{control_name}_present"] = present
        row[f"{control_name}_size_span_top"] = (
            bool(index in structural_top) if present else False
        )
        row[f"{control_name}_pareto_top"] = (
            bool(index in pareto_top) if present else False
        )
        if present:
            candidate = mis_set[index]
            row[f"{control_name}_retention"] = float(
                candidate.pareto.retention
            )
            row[f"{control_name}_exact"] = bool(
                candidate.pareto.exact_preservation
            )
        else:
            row[f"{control_name}_retention"] = None
            row[f"{control_name}_exact"] = None

    return row


def run_robustness(
    *,
    sample_seeds=SAMPLE_SEEDS,
    screen_power=SCREEN_POWER,
    misda_seed=MISDA_SEED,
):
    rows = []
    for name, mop in _problem_suite().items():
        for sample_seed in sample_seeds:
            rows.append(
                _evaluate_replicate(
                    name,
                    mop,
                    power=screen_power,
                    sample_seed=sample_seed,
                    misda_seed=misda_seed,
                )
            )
    return pd.DataFrame(rows)


def _rate(series):
    values = pd.Series(series, dtype=float)
    return float(values.mean()) if len(values) else None


def summarize_robustness(results):
    summaries = []

    for name, group in results.groupby("problem", sort=False):
        summary = {
            "problem": name,
            "replicates": int(len(group)),
            "n_screen": int(group["n_screen"].iloc[0]),
            "median_n_mis": float(group["n_mis"].median()),
            "median_selected_retention": float(
                group["pareto_selected_retention"].median()
            ),
        }

        if name == "DTLZ2":
            summary.update(
                {
                    "observed_exact_rate": _rate(group["any_observed_exact"]),
                    "median_max_observed_retention": float(
                        group["max_observed_retention"].median()
                    ),
                    "min_max_observed_retention": float(
                        group["max_observed_retention"].min()
                    ),
                    "max_max_observed_retention": float(
                        group["max_observed_retention"].max()
                    ),
                }
            )

        for control_name in _controls(name):
            present = group[f"{control_name}_present"].astype(bool)
            n_present = int(present.sum())
            summary[f"{control_name}_present_rate"] = _rate(present)
            summary[f"{control_name}_size_span_top_rate"] = _rate(
                group[f"{control_name}_size_span_top"]
            )
            summary[f"{control_name}_pareto_top_rate"] = _rate(
                group[f"{control_name}_pareto_top"]
            )
            if n_present:
                summary[
                    f"{control_name}_pareto_top_given_present_rate"
                ] = _rate(
                    group.loc[present, f"{control_name}_pareto_top"]
                )
                summary[f"{control_name}_median_retention"] = float(
                    group.loc[present, f"{control_name}_retention"].median()
                )
            else:
                summary[
                    f"{control_name}_pareto_top_given_present_rate"
                ] = None
                summary[f"{control_name}_median_retention"] = None

        summaries.append(summary)

    return pd.DataFrame(summaries)


def _json_value(value):
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    return value


def _records(frame):
    return [
        {key: _json_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-power", type=int, default=SCREEN_POWER)
    parser.add_argument("--misda-seed", type=int, default=MISDA_SEED)
    parser.add_argument(
        "--sample-seeds",
        type=int,
        nargs="+",
        default=list(SAMPLE_SEEDS),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    results = run_robustness(
        sample_seeds=tuple(args.sample_seeds),
        screen_power=args.screen_power,
        misda_seed=args.misda_seed,
    )
    summary = summarize_robustness(results)

    print("\n=== Pareto-retention resampling robustness ===")
    print(summary.to_string(index=False))

    if args.output is not None:
        payload = {
            "configuration": {
                "M": M,
                "screen_power": int(args.screen_power),
                "n_screen": int(2 ** args.screen_power),
                "misda_seed": int(args.misda_seed),
                "sample_seeds": [int(seed) for seed in args.sample_seeds],
            },
            "summary": _records(summary),
            "replicates": _records(results),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
