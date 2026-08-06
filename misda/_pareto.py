# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Legacy Pareto-front operations used by MISDA validation."""

import numpy as np


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
    }


def get_nondominated_mask(Y):
    """
    Returns boolean mask of non-dominated solutions (Minimization) for a dataset Y.
    Complexity: O(N^2)
    Args:
        Y (np.ndarray): shape (N, M)
    Returns:
        np.array(bool): shape (N,), True if non-dominated.
    """
    # Ensure numpy
    Y = np.asarray(Y)
    N, M = Y.shape
    is_efficient = np.ones(N, dtype=bool)
    for i in range(N):
        # i is dominated by j if:
        # all(Y[j] <= Y[i]) AND any(Y[j] < Y[i])
        better_or_equal = (Y <= Y[i]).all(axis=1)
        better = (Y < Y[i]).any(axis=1)
        dominators = better_or_equal & better
        if dominators.any():
            is_efficient[i] = False
    return is_efficient


def evaluate_pareto_consistency(result_obj, df_original=None):
    """
    Compares the True Pareto Front (Full M) vs Surrogate Pareto Front (Reduced k).
    Calculates Precision (Safety) and Recall (Coverage).

    Args:
        result_obj (MISDAResult): The result object from misda.analyze()
        df_original (pd.DataFrame or np.ndarray): Original data. If None, tries to use result_obj.Y

    Returns:
        (precision, recall):
            Precision = P(True Optimum | Surrogate Optimum) -> Safety
            Recall    = P(Surrogate Optimum | True Optimum) -> Coverage
    """
    Y_full = df_original if df_original is not None else result_obj.Y
    if hasattr(Y_full, "values"):
        Y_full = Y_full.values
    Y_full = np.asarray(Y_full)

    mis = result_obj.best_mis
    if not mis or not mis.indices:
        return 0.0, 0.0

    indices = mis.indices
    Y_sub = Y_full[:, indices]

    # 1. True Front
    mask_true = get_nondominated_mask(Y_full)

    # 2. Surrogate Front
    mask_surr = get_nondominated_mask(Y_sub)

    # Metrics
    intersection = (mask_true & mask_surr).sum()

    # Precision: Of the points the surrogate thinks are optimal, how many are truly optimal?
    denom_p = mask_surr.sum()
    precision = intersection / denom_p if denom_p > 0 else 0.0

    return precision, recall


def evaluate_pareto_raw(Y, selected_indices, directions=None):
    """
    Evaluates Pareto precision and recall directly on raw objective matrix Y.
    Assumes minimization by default for all objectives unless directions specifies otherwise.

    Args:
        Y (np.ndarray or pd.DataFrame): Shape (N, M) matrix of objective values.
        selected_indices (sequence of int): Indices of selected/kept objectives.
        directions (sequence of int, optional): Objective optimization directions (+1 for max, -1 for min).

    Returns:
        tuple[float, float]: (precision, recall)
    """
    if hasattr(Y, "values"):
        data = np.asarray(Y.values, dtype=float)
    else:
        data = np.asarray(Y, dtype=float)

    N, M = data.shape
    if N == 0 or M == 0:
        return 0.0, 0.0

    if directions is not None:
        dirs = np.asarray(directions, dtype=float)
        Y_eval = data * (-dirs)
    else:
        Y_eval = data

    mask_true = get_nondominated_mask(Y_eval)
    sel = list(selected_indices)

    if len(sel) == M:
        return 1.0, 1.0

    if len(sel) == 0:
        return 0.0, 0.0

    Y_sub = Y_eval[:, sel]
    mask_surr = get_nondominated_mask(Y_sub)

    intersection = (mask_true & mask_surr).sum()

    denom_p = mask_surr.sum()
    precision = float(intersection / denom_p) if denom_p > 0 else 0.0

    denom_r = mask_true.sum()
    recall = float(intersection / denom_r) if denom_r > 0 else 0.0

    return precision, recall
