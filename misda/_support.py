# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Data-driven internal support diagnostics for MISDA dimensional estimates."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


TRANSITIVE_CHAINING = "TRANSITIVE_CHAINING"
HIDDEN_SPECTRAL_STRUCTURE = "HIDDEN_SPECTRAL_STRUCTURE"
SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"


def _rank_standardize(data):
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("data must be a non-empty two-dimensional matrix.")
    ranks = rankdata(matrix, axis=0, method="average")
    centered = ranks - np.mean(ranks, axis=0)
    norms = np.sqrt(np.sum(centered * centered, axis=0))
    standardized = np.zeros_like(centered, dtype=float)
    valid = norms > 0.0
    standardized[:, valid] = centered[:, valid] / norms[valid]
    return standardized, valid


def _rank_correlation(standardized, valid):
    correlation = standardized.T @ standardized
    np.clip(correlation, -1.0, 1.0, out=correlation)
    valid_indices = np.flatnonzero(valid)
    correlation[valid_indices, valid_indices] = 1.0
    return correlation


def _max_min_closure(weights):
    """Return widest-path strengths under max-min path composition."""

    closure = np.asarray(weights, dtype=float).copy()
    np.fill_diagonal(closure, 1.0)
    for intermediate in range(closure.shape[0]):
        closure = np.maximum(
            closure,
            np.minimum(
                closure[:, intermediate : intermediate + 1],
                closure[intermediate : intermediate + 1, :],
            ),
        )
    return closure


def _positive_and_closure(correlation):
    positive = np.maximum(correlation, 0.0)
    np.fill_diagonal(positive, 1.0)
    return positive, _max_min_closure(positive)


def _normalize_selected(selected_indices, n_objectives):
    selected = tuple(sorted(set(int(index) for index in selected_indices)))
    if not selected:
        raise ValueError("selected_indices must not be empty.")
    if any(index < 0 or index >= n_objectives for index in selected):
        raise IndexError("selected_indices contains an out-of-range objective.")
    return selected


def _transitivity_statistic_from_closure(
    positive,
    closure,
    selected_indices,
):
    """Largest indirect-minus-direct positive association to a retained objective."""

    n_objectives = positive.shape[0]
    selected = _normalize_selected(selected_indices, n_objectives)
    selected_set = set(selected)
    eliminated = tuple(
        index for index in range(n_objectives) if index not in selected_set
    )
    if not eliminated:
        return 0.0

    selected_array = np.asarray(selected, dtype=int)
    excesses = []
    for objective in eliminated:
        direct = float(np.max(positive[selected_array, objective], initial=0.0))
        indirect = float(np.max(closure[selected_array, objective], initial=0.0))
        excesses.append(max(0.0, indirect - direct))
    return float(max(excesses, default=0.0))


def _transitivity_statistics_from_closure(
    positive,
    closure,
    selections,
):
    """Vectorized transitivity statistics for several retained-objective sets.

    Candidates with the same cardinality are evaluated together.  Structural
    rank-tie groups naturally satisfy this condition, so a block case such as
    4 x K_5 evaluates all 625 tied MISs with two indexed reductions instead of
    625 Python-level candidate loops for every null permutation.
    """

    n_objectives = positive.shape[0]
    if closure.shape != positive.shape:
        raise ValueError("positive and closure must have the same shape.")
    if not selections:
        return np.empty(0, dtype=float)

    statistics = np.empty(len(selections), dtype=float)
    positions_by_size = {}
    for position, selected in enumerate(selections):
        positions_by_size.setdefault(len(selected), []).append(position)

    for size, positions in positions_by_size.items():
        if size == n_objectives:
            statistics[positions] = 0.0
            continue

        selected = np.asarray(
            [selections[position] for position in positions],
            dtype=int,
        )
        direct = np.max(positive[selected, :], axis=1)
        indirect = np.max(closure[selected, :], axis=1)
        excess = np.maximum(indirect - direct, 0.0)

        retained = np.zeros((len(positions), n_objectives), dtype=bool)
        rows = np.arange(len(positions))[:, None]
        retained[rows, selected] = True
        excess[retained] = 0.0
        statistics[positions] = np.max(excess, axis=1)

    return statistics


def _transitivity_statistic(correlation, selected_indices):
    positive, closure = _positive_and_closure(correlation)
    return _transitivity_statistic_from_closure(
        positive,
        closure,
        selected_indices,
    )


def _next_spectral_eigenvalue(
    correlation,
    valid,
    latent_dimension,
    n_constants,
):
    """Return the first rank-correlation eigenvalue beyond the estimated signal rank."""

    valid_indices = np.flatnonzero(valid)
    signal_dimension = max(0, int(latent_dimension) - int(n_constants))
    if signal_dimension >= len(valid_indices) or len(valid_indices) == 0:
        return 0.0, signal_dimension

    submatrix = correlation[np.ix_(valid_indices, valid_indices)]
    eigenvalues = np.linalg.eigvalsh(submatrix)[::-1]
    return float(eigenvalues[signal_dimension]), signal_dimension


def _derived_seed(seed, coordinate):
    sequence = np.random.SeedSequence([int(seed), int(coordinate)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def evaluate_dimensional_support_group(
    data,
    selected_sets,
    latent_dimension,
    *,
    seed=123,
):
    """Evaluate support for several tied candidates using one shared null.

    The observed rank correlation, every column-wise null permutation, the
    widest-path closure for each permutation, and the spectral diagnostic are
    computed once. Candidate-specific transitivity statistics are evaluated in
    vectorized cardinality groups from those shared structures. The scientific
    statistics are identical to evaluating each candidate separately with the
    same seed, but candidate-independent work and Python-level candidate loops
    are not repeated.
    """

    matrix = np.asarray(data, dtype=float)
    standardized, valid = _rank_standardize(matrix)
    correlation = _rank_correlation(standardized, valid)
    n_samples, n_objectives = matrix.shape
    selections = tuple(
        _normalize_selected(selected, n_objectives) for selected in selected_sets
    )
    if not selections:
        return tuple()

    n_constants = n_objectives - int(np.sum(valid))
    positive, closure = _positive_and_closure(correlation)
    observed_transitivity = _transitivity_statistics_from_closure(
        positive,
        closure,
        selections,
    )
    observed_spectral, signal_dimension = _next_spectral_eigenvalue(
        correlation,
        valid,
        latent_dimension,
        n_constants,
    )

    support_seed = _derived_seed(seed, 9101)
    rng = np.random.default_rng(support_seed)
    null_transitivity = np.empty((len(selections), n_samples), dtype=float)
    null_spectral = np.empty(n_samples, dtype=float)
    permuted = np.empty_like(standardized)

    for repetition in range(n_samples):
        for objective in range(n_objectives):
            permuted[:, objective] = rng.permutation(standardized[:, objective])
        null_correlation = _rank_correlation(permuted, valid)
        null_positive, null_closure = _positive_and_closure(null_correlation)
        null_transitivity[:, repetition] = _transitivity_statistics_from_closure(
            null_positive,
            null_closure,
            selections,
        )
        null_spectral[repetition], _ = _next_spectral_eigenvalue(
            null_correlation,
            valid,
            latent_dimension,
            n_constants,
        )

    mean_null_transitivity = np.mean(null_transitivity, axis=1)
    mean_null_spectral = float(np.mean(null_spectral))
    spectral_excess = float(observed_spectral - mean_null_spectral)

    results = []
    for position, observed in enumerate(observed_transitivity):
        transitivity_excess = float(observed - mean_null_transitivity[position])
        reasons = []
        if transitivity_excess > 0.0:
            reasons.append(TRANSITIVE_CHAINING)
        if spectral_excess > 0.0:
            reasons.append(HIDDEN_SPECTRAL_STRUCTURE)
        results.append(
            {
                "status": UNSUPPORTED if reasons else SUPPORTED,
                "reasons": tuple(reasons),
                "transitivity": {
                    "observed": float(observed),
                    "null": float(mean_null_transitivity[position]),
                    "excess": transitivity_excess,
                },
                "spectral": {
                    "tested_dimension": int(signal_dimension),
                    "observed_next_eigenvalue": observed_spectral,
                    "null_next_eigenvalue": mean_null_spectral,
                    "excess": spectral_excess,
                },
                "n_permutations": int(n_samples),
                "seed": support_seed,
            }
        )
    return tuple(results)


def evaluate_dimensional_support(
    data,
    selected_indices,
    latent_dimension,
    *,
    seed=123,
):
    """Evaluate support for one candidate using the shared-group engine."""

    return evaluate_dimensional_support_group(
        data,
        (selected_indices,),
        latent_dimension,
        seed=seed,
    )[0]
