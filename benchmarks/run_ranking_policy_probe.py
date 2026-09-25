# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Probe candidate-ranking heuristics on the DTLZ5 optimization control.

This is a diagnostic script, not a benchmark acceptance test. It reproduces
the optimization notebook screening protocol, evaluates every discovered MIS,
and compares candidate-ordering signals without feeding analytical truth back
into MISDA:

* canonical size_span ordering;
* positive Pearson-correlation coverage of eliminated objectives;
* external linear reconstruction (PRESS/LOO R2);
* observed Pareto retention on the same Sobol screening sample.

The analytical DTLZ5 labels are added only after all data-driven quantities are
computed: for M=10, f9,f10 is the known optimization-safe control and f1,f10
has an explicit unsafe counterexample in optimization.ipynb.
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


def _screen(mop):
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
        name="DTLZ5 ranking-policy probe",
        seed=MISDA_SEED,
    )
    misda.evaluate(result, metrics=("linear", "pareto"), candidates="all")
    return result, F


def _positive_correlation_coverage(F, indices):
    """Return worst/mean best positive pairwise correlation to retained columns."""

    corr = np.corrcoef(np.asarray(F, dtype=float), rowvar=False)
    selected = tuple(int(index) for index in indices)
    selected_set = set(selected)
    outside = [
        index for index in range(F.shape[1])
        if index not in selected_set
    ]
    if not outside:
        return 1.0, 1.0

    best = []
    for target in outside:
        best.append(
            max(
                0.0,
                max(float(corr[source, target]) for source in selected),
            )
        )
    return float(np.min(best)), float(np.mean(best))


def _rank_positions(rows, key):
    ordered = sorted(range(len(rows)), key=lambda index: key(rows[index]))
    return {index: position + 1 for position, index in enumerate(ordered)}


def main():
    mop = mb.mops.DTLZ5(M=M)
    mis_set, F = _screen(mop)

    rows = []
    for candidate_index, candidate in enumerate(mis_set):
        worst_corr, mean_corr = _positive_correlation_coverage(
            F,
            candidate.indices,
        )
        objective_numbers = tuple(index + 1 for index in candidate.indices)
        truth = ""
        if objective_numbers == (M - 1, M):
            truth = "SAFE control"
        elif objective_numbers == (1, M):
            truth = "UNSAFE witness"

        rows.append(
            {
                "candidate_index": candidate_index,
                "objectives": ",".join(candidate.objectives),
                "size": candidate.size,
                "span": candidate.structural.span,
                "worst_corr": worst_corr,
                "mean_corr": mean_corr,
                "worst_r2": candidate.linear.worst_r2,
                "mean_r2": candidate.linear.mean_r2,
                "pareto_retention": candidate.pareto.retention,
                "pareto_validity": candidate.pareto.validity,
                "truth": truth,
            }
        )

    correlation_rank = _rank_positions(
        rows,
        lambda row: (
            -row["size"],
            -row["worst_corr"],
            -row["mean_corr"],
            row["objectives"],
        ),
    )
    reconstruction_rank = _rank_positions(
        rows,
        lambda row: (
            -row["size"],
            -row["worst_r2"],
            -row["mean_r2"],
            row["objectives"],
        ),
    )
    pareto_rank = _rank_positions(
        rows,
        lambda row: (
            -row["pareto_retention"],
            -row["pareto_validity"],
            row["objectives"],
        ),
    )

    for index, row in enumerate(rows):
        row["size_span_rank"] = index + 1
        row["correlation_rank"] = correlation_rank[index]
        row["reconstruction_rank"] = reconstruction_rank[index]
        row["pareto_rank"] = pareto_rank[index]

    table = pd.DataFrame(rows).sort_values(
        ["correlation_rank", "reconstruction_rank", "candidate_index"]
    )
    columns = [
        "candidate_index",
        "objectives",
        "size",
        "span",
        "size_span_rank",
        "worst_corr",
        "mean_corr",
        "correlation_rank",
        "worst_r2",
        "mean_r2",
        "reconstruction_rank",
        "pareto_retention",
        "pareto_validity",
        "pareto_rank",
        "truth",
    ]
    print(table[columns].to_string(index=False))

    safe = table.loc[table["truth"] == "SAFE control"].iloc[0]
    unsafe = table.loc[table["truth"] == "UNSAFE witness"].iloc[0]
    print()
    print(
        "SAFE f9,f10: "
        f"correlation rank={int(safe['correlation_rank'])}, "
        f"reconstruction rank={int(safe['reconstruction_rank'])}, "
        f"Pareto-retention rank={int(safe['pareto_rank'])}"
    )
    print(
        "UNSAFE f1,f10: "
        f"correlation rank={int(unsafe['correlation_rank'])}, "
        f"reconstruction rank={int(unsafe['reconstruction_rank'])}, "
        f"Pareto-retention rank={int(unsafe['pareto_rank'])}"
    )


if __name__ == "__main__":
    main()
