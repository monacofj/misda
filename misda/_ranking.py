# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Structural metrics used by MISDA ranking policies."""

import numpy as np


def compute_mis_metrics(mis_list, adjacency, labels):
    """Compute structural metrics for every maximal independent set."""

    adjacency = np.asarray(adjacency, dtype=int)
    n_objectives = adjacency.shape[0]
    results = []

    for candidate in mis_list:
        selected = sorted(candidate)
        selected_set = set(selected)
        outside = [
            index for index in range(n_objectives) if index not in selected_set
        ]

        internal_degrees = [
            sum(adjacency[source, target] for target in selected)
            for source in selected
        ]
        external_degrees = [
            sum(adjacency[source, target] for target in outside)
            for source in selected
        ]

        external_neighbors = set()
        for source in selected:
            for target in outside:
                if adjacency[source, target] == 1:
                    external_neighbors.add(target)

        neighborhood = len(external_neighbors)
        remainder = max(1, len(outside))
        results.append(
            {
                "mis_indices": selected,
                "mis_labels": [labels[index] for index in selected],
                "size": len(selected),
                "neighborhood": neighborhood,
                "neighborhood_ratio": neighborhood / remainder,
                "span": int(sum(external_degrees)),
                "avg_external_degree": (
                    float(np.mean(external_degrees))
                    if external_degrees
                    else 0.0
                ),
                "avg_internal_degree": (
                    float(np.mean(internal_degrees))
                    if internal_degrees
                    else 0.0
                ),
            }
        )
    return results
