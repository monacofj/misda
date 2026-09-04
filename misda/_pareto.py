# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pareto-front preservation operations for static MISDA evaluation."""

import numpy as np


def get_nondominated_mask(Y):
    """Return the nondominated-row mask for a minimization matrix."""

    data = np.asarray(Y, dtype=float)
    if data.ndim != 2:
        raise ValueError("Y must be a two-dimensional matrix.")
    n_rows = data.shape[0]
    if n_rows <= 1:
        return np.ones(n_rows, dtype=bool)

    efficient = np.ones(n_rows, dtype=bool)
    order = np.lexsort(data.T[::-1])
    for position in range(1, n_rows):
        index = order[position]
        prior_indices = order[:position]
        efficient_priors = prior_indices[efficient[prior_indices]]
        if len(efficient_priors) == 0:
            continue
        prior_values = data[efficient_priors]
        current = data[index]
        dominated = (
            (prior_values <= current).all(axis=1)
            & (prior_values < current).any(axis=1)
        )
        if dominated.any():
            efficient[index] = False
    return efficient


def get_nondominated_mask_minimize(Y):
    """Return a minimization front mask after exact vector deduplication."""

    data = np.asarray(Y, dtype=float)
    if data.ndim != 2:
        raise ValueError("Y must be a two-dimensional matrix.")
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("Y must not be empty.")
    unique, inverse = np.unique(data, axis=0, return_inverse=True)
    unique_mask = get_nondominated_mask(unique)
    return unique_mask[inverse]


def evaluate_pareto_preservation(
    Y,
    selected_indices,
    *,
    full_front=None,
    directions=None,
):
    """Evaluate retention and validity of a reduced minimization front."""

    if directions is not None:
        raise ValueError("Mixed objective directions are not supported.")
    data = np.asarray(Y, dtype=float)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("Y must be a non-empty two-dimensional matrix.")
    selected = tuple(sorted(set(int(index) for index in selected_indices)))
    if not selected:
        raise ValueError("selected_indices must not be empty.")
    if any(index < 0 or index >= data.shape[1] for index in selected):
        raise IndexError("selected_indices contains an out-of-range objective.")

    if full_front is None:
        full = get_nondominated_mask_minimize(data)
    else:
        full = np.asarray(full_front, dtype=bool)
        if full.shape != (data.shape[0],):
            raise ValueError("full_front must contain one flag per observation.")
    reduced = get_nondominated_mask_minimize(data[:, selected])
    intersection = full & reduced
    union = full | reduced
    n_full = int(np.sum(full))
    n_reduced = int(np.sum(reduced))
    n_intersection = int(np.sum(intersection))
    n_union = int(np.sum(union))
    return {
        "pareto_retention": (
            float(n_intersection / n_full) if n_full else None
        ),
        "pareto_validity": (
            float(n_intersection / n_reduced) if n_reduced else None
        ),
        "pareto_jaccard": (
            float(n_intersection / n_union) if n_union else None
        ),
        "full_front_size": n_full,
        "reduced_front_size": n_reduced,
        "intersection_size": n_intersection,
        "union_size": n_union,
        "exact_preservation": bool(np.array_equal(full, reduced)),
        "reduced_front_indices": tuple(
            int(index) for index in np.flatnonzero(reduced)
        ),
    }
