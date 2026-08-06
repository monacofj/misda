# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Legacy metrics and ordering operations used to rank MIS candidates."""

import numpy as np


def compute_mis_metrics(mis_list, adjacency, labels):
    """
    Computes various metrics for each Maximal Independent Set (MIS).

    Args:
        mis_list (list): A list of MIS, where each MIS is a list of node indices.
        adjacency (np.ndarray): The adjacency matrix of the graph.
        labels (list): A list of labels for the nodes.

    Returns:
        list: A list of dictionaries, each containing metrics for an MIS.
    """
    A = np.array(adjacency, dtype=int)
    n = A.shape[0]
    results = []

    for S in mis_list:
        S = sorted(S)
        S_set = set(S)
        notS = [i for i in range(n) if i not in S_set]

        internal_deg = [sum(A[u, v] for v in S) for u in S]
        avg_internal = float(np.mean(internal_deg)) if internal_deg else 0.0

        ext_deg = [sum(A[u, v] for v in notS) for u in S]
        avg_ext = float(np.mean(ext_deg)) if ext_deg else 0.0

        ext_nodes = set()
        for u in S:
            for v in notS:
                if A[u, v] == 1:
                    ext_nodes.add(v)
        neighborhood = len(ext_nodes)

        remainder = max(1, len(notS))
        neighborhood_ratio = neighborhood / remainder
        span = int(sum(ext_deg))

        results.append({
            "mis_indices": S,
            "mis_labels": [labels[i] for i in S],
            "size": len(S),
            "neighborhood": neighborhood,
            "neighborhood_ratio": neighborhood_ratio,
            "span": span,
            "avg_external_degree": avg_ext,
            "avg_internal_degree": avg_internal,
        })
    return results


def sort_mis_metrics(mis_metrics):
    """
    Sorts a list of MIS metrics dictionaries based on a predefined ranking criteria.
    The primary sorting keys are: size (desc), neighborhood (desc), avg_external_degree (desc),
    span (desc), and mis_labels (asc) for tie-breaking.

    Args:
        mis_metrics (list): A list of dictionaries, each containing metrics for an MIS.

    Returns:
        list: The sorted list of MIS metrics dictionaries.
    """
    return sorted(
        mis_metrics,
        key=lambda x: (
            -x["size"],
            -x["neighborhood"],
            -x["avg_external_degree"],
            -x["span"],
            tuple(x["mis_labels"]),
        )
    )
