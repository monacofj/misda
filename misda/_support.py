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


def _transitivity_statistic(correlation, selected_indices):
    """Largest indirect-minus-direct positive association to a retained objective."""

    n_objectives = correlation.shape[0]
    selected = tuple(sorted(set(int(index) for index in selected_indices)))
    if not selected:
        raise ValueError("selected_indices must not be empty.")
    if any(index < 0 or index >= n_objectives for index in selected):
        raise IndexError("selected_indices contains an out-of-range objective.")

    selected_set = set(selected)
    eliminated = tuple(
        index for index in range(n_objectives) if index not in selected_set
    )
    if not eliminated:
        return 0.0

    positive = np.maximum(correlation, 0.0)
    np.fill_diagonal(positive, 1.0)
    closure = _max_min_closure(positive)
    selected_array = np.asarray(selected, dtype=int)

    excesses = []
    for objective in eliminated:
        direct = float(np.max(positive[selected_array, objective], initial=0.0))
        indirect = float(np.max(closure[selected_array, objective], initial=0.0))
        excesses.append(max(0.0, indirect - direct))
    return float(max(excesses, default=0.0))


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


def evaluate_dimensional_support(
    data,
    selected_indices,
    latent_dimension,
    *,
    seed=123,
):
    """Evaluate whether the data contradict the graph dimensional estimate.

    Two rank-based diagnostics are calibrated against column-wise permutation
    nulls. The number of null permutations equals the sample size, so there is
    no user-set Monte Carlo budget.

    ``transitivity_excess`` detects chaining: indirect positive association to
    the retained MIS is stronger than direct association beyond what independent
    ranks generate by chance.

    ``spectral_excess`` detects hidden directions: the first eigenvalue beyond
    the estimated latent dimension is larger than its permutation-null mean.

    The final decision uses only the intrinsic zero boundary after null
    subtraction: any positive excess makes the estimate ``UNSUPPORTED``.
    """

    matrix = np.asarray(data, dtype=float)
    standardized, valid = _rank_standardize(matrix)
    correlation = _rank_correlation(standardized, valid)
    n_samples, n_objectives = matrix.shape
    n_constants = n_objectives - int(np.sum(valid))

    observed_transitivity = _transitivity_statistic(
        correlation,
        selected_indices,
    )
    observed_spectral, signal_dimension = _next_spectral_eigenvalue(
        correlation,
        valid,
        latent_dimension,
        n_constants,
    )

    support_seed = _derived_seed(seed, 9101)
    rng = np.random.default_rng(support_seed)
    null_transitivity = np.empty(n_samples, dtype=float)
    null_spectral = np.empty(n_samples, dtype=float)
    permuted = np.empty_like(standardized)

    for repetition in range(n_samples):
        for objective in range(n_objectives):
            permuted[:, objective] = rng.permutation(standardized[:, objective])
        null_correlation = _rank_correlation(permuted, valid)
        null_transitivity[repetition] = _transitivity_statistic(
            null_correlation,
            selected_indices,
        )
        null_spectral[repetition], _ = _next_spectral_eigenvalue(
            null_correlation,
            valid,
            latent_dimension,
            n_constants,
        )

    mean_null_transitivity = float(np.mean(null_transitivity))
    mean_null_spectral = float(np.mean(null_spectral))
    transitivity_excess = float(observed_transitivity - mean_null_transitivity)
    spectral_excess = float(observed_spectral - mean_null_spectral)

    reasons = []
    if transitivity_excess > 0.0:
        reasons.append(TRANSITIVE_CHAINING)
    if spectral_excess > 0.0:
        reasons.append(HIDDEN_SPECTRAL_STRUCTURE)

    return {
        "status": UNSUPPORTED if reasons else SUPPORTED,
        "reasons": tuple(reasons),
        "transitivity": {
            "observed": observed_transitivity,
            "null": mean_null_transitivity,
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
