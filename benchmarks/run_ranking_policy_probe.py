# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare ranking hypotheses on optimization-control MOPs.

The script reproduces the optimization notebook independent Sobol screening
for DTLZ2, DTLZ5, and DPF1. All candidate scores are computed from observed Y
only. Analytical truth labels are attached afterwards for interpretation and
never feed discovery, evaluation, or ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import qmc

import misda
import moeabench as mb


M = 10
SCREEN_POWER = 9
MISDA_SEED = 123


def _screen(name, mop):
    sampler = qmc.Sobol(d=mop.N, scramble=True, seed=MISDA_SEED)
    X = qmc.scale(
        sampler.random_base2(m=SCREEN_POWER),
        np.asarray(mop.xl, dtype=float),
        np.asarray(mop.xu, dtype=float),
    )
    F = np.asarray(mop.evaluation(X)["F"], dtype=float)
    frame = pd.DataFrame(F, columns=[f"f{i + 1}" for i in range(mop.M)])
    result = misda.discover(
        frame,
        name=f"{name} ranking-policy probe",
        seed=MISDA_SEED,
    )
    misda.evaluate(result, metrics=("linear", "pareto"), candidates="all")
    return result, F


def _positive_correlation_coverage(F, indices):
    corr = np.corrcoef(np.asarray(F, dtype=float), rowvar=False)
    selected = tuple(int(index) for index in indices)
    selected_set = set(selected)
    outside = [
        index for index in range(F.shape[1])
        if index not in selected_set
    ]
    if not outside:
        return 1.0, 1.0

    best = [
        max(
            0.0,
            max(float(corr[source, target]) for source in selected),
        )
        for target in outside
    ]
    return float(np.min(best)), float(np.mean(best))


def _rank_positions(rows, key):
    ordered = sorted(range(len(rows)), key=lambda index: key(rows[index]))
    return {index: position + 1 for position, index in enumerate(ordered)}


def _truth_label(name, indices):
    objectives = tuple(index + 1 for index in indices)
    if name == "DTLZ2":
        return "UNSAFE: every proper subset changes PF dominance"
    if name == "DTLZ5":
        if objectives == (9, 10):
            return "SAFE control"
        if objectives == (1, 10):
            return "UNSAFE witness"
        return ""
    if name == "DPF1":
        if objectives == (1, 2):
            return "SAFE base pair"
        if objectives == (2, 8):
            return "PF-only evidence; global safety uncertified"
        return ""
    return ""


def _probe(name, mop):
    mis_set, F = _screen(name, mop)
    pareto_ranking = misda.rank(mis_set, policy=misda.PARETO_RETENTION)
    pareto_position = {
        index: position + 1
        for position, index in enumerate(pareto_ranking.indices)
    }

    rows = []
    for candidate_index, candidate in enumerate(mis_set):
        worst_corr, mean_corr = _positive_correlation_coverage(
            F,
            candidate.indices,
        )
        rows.append(
            {
                "candidate_index": candidate_index,
                "objectives": ",".join(map(str, candidate.objectives)),
                "size": candidate.size,
                "span": candidate.structural.span,
                "size_span_position": candidate_index + 1,
                "worst_corr": worst_corr,
                "mean_corr": mean_corr,
                "worst_r2": candidate.linear.worst_r2,
                "mean_r2": candidate.linear.mean_r2,
                "pareto_retention": candidate.pareto.retention,
                "pareto_validity": candidate.pareto.validity,
                "pareto_jaccard": candidate.pareto.jaccard,
                "pareto_retention_position": pareto_position[candidate_index],
                "truth": _truth_label(name, candidate.indices),
            }
        )

    for row in rows:
        if not np.isclose(row["pareto_validity"], 1.0):
            raise AssertionError(
                f"{name}: objective projection produced Pareto validity != 1"
            )
        if not np.isclose(row["pareto_jaccard"], row["pareto_retention"]):
            raise AssertionError(
                f"{name}: Jaccard != retention under objective projection"
            )

    correlation = _rank_positions(
        rows,
        lambda row: (
            -row["size"],
            -row["worst_corr"],
            -row["mean_corr"],
            row["objectives"],
        ),
    )
    reconstruction = _rank_positions(
        rows,
        lambda row: (
            -row["size"],
            -row["worst_r2"],
            -row["mean_r2"],
            row["objectives"],
        ),
    )
    for index, row in enumerate(rows):
        row["correlation_position"] = correlation[index]
        row["reconstruction_position"] = reconstruction[index]

    table = pd.DataFrame(rows).sort_values(
        ["pareto_retention_position", "candidate_index"]
    )
    print(f"\n=== {name} ===")
    print(
        table[
            [
                "candidate_index",
                "objectives",
                "size",
                "span",
                "size_span_position",
                "correlation_position",
                "reconstruction_position",
                "pareto_retention",
                "pareto_validity",
                "pareto_jaccard",
                "pareto_retention_position",
                "truth",
            ]
        ].to_string(index=False)
    )
    print(
        "Projection identity verified: validity=1 and Jaccard=retention "
        "for every candidate."
    )
    return table


def main():
    results = {
        "DTLZ2": _probe("DTLZ2", mb.mops.DTLZ2(M=M)),
        "DTLZ5": _probe("DTLZ5", mb.mops.DTLZ5(M=M)),
        "DPF1": _probe("DPF1", mb.mops.DPF1(M=M, D=2, K=5)),
    }

    print("\n=== External-truth controls (interpretation only) ===")
    for name, table in results.items():
        controls = table.loc[table["truth"] != ""]
        if controls.empty:
            continue
        print(f"\n{name}")
        print(
            controls[
                [
                    "objectives",
                    "size_span_position",
                    "correlation_position",
                    "reconstruction_position",
                    "pareto_retention",
                    "pareto_validity",
                    "pareto_jaccard",
                    "pareto_retention_position",
                    "truth",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
