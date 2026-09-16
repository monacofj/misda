# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Experimental Spearman-based discovery statistics for MISDA.

This module intentionally changes only the dependence coefficient used by the
static discovery pipeline.  The existing MISDA Fisher-z/log-alpha mapping and
fixed-budget permutation calibration are preserved so that the experimental
comparison isolates Pearson versus Spearman as closely as possible.
"""

import copy
from dataclasses import replace
from numbers import Integral
from typing import Any, Callable, Optional

import numpy as np
from scipy import stats

from ._statistics import (
    CorrelationStatistics,
    NullAlphaEstimate,
    estimate_null_envelope_from_maxima,
    interpolate_log_alpha,
    positive_correlation_log_p,
    separation_status,
)
from ._validation import NormalizedInput


def _rank_standardize(data: np.ndarray) -> np.ndarray:
    """Return columnwise average ranks centered and normalized to unit norm."""

    ranks = np.empty_like(data, dtype=float)
    for column in range(data.shape[1]):
        ranks[:, column] = stats.rankdata(data[:, column], method="average")

    centered = ranks - np.mean(ranks, axis=0)
    sum_squares = np.sum(centered * centered, axis=0)
    scales = np.sqrt(sum_squares)
    standardized = np.zeros_like(centered)
    valid = scales > 0.0
    standardized[:, valid] = centered[:, valid] / scales[valid]
    return standardized


def compute_correlation_statistics(normalized: NormalizedInput) -> CorrelationStatistics:
    """Compute signed Spearman correlations for the static discovery pipeline."""

    if not isinstance(normalized, NormalizedInput):
        raise TypeError("normalized must be a NormalizedInput instance.")

    data = normalized.data
    n_samples, n_objectives = data.shape
    standardized_ranks = _rank_standardize(data)
    raw_correlation = standardized_ranks.T @ standardized_ranks
    np.clip(raw_correlation, -1.0, 1.0, out=raw_correlation)

    nonconstant = ~normalized.constant_mask
    valid_matrix = np.outer(nonconstant, nonconstant)
    correlation = np.full((n_objectives, n_objectives), np.nan, dtype=float)
    correlation[valid_matrix] = raw_correlation[valid_matrix]

    diagonal = np.arange(n_objectives)
    correlation[diagonal[nonconstant], diagonal[nonconstant]] = 1.0

    valid_pairs = valid_matrix.copy()
    np.fill_diagonal(valid_pairs, False)

    # Keep MISDA's existing monotone r -> log-alpha mapping.  The permutation
    # null below is also Spearman-based, so the data-driven threshold is
    # recalibrated consistently while the experiment changes only the
    # dependence coefficient.
    log_p = np.full_like(correlation, np.nan)
    if np.any(valid_pairs):
        log_p[valid_pairs] = positive_correlation_log_p(
            np.abs(correlation[valid_pairs]),
            n_samples,
        )

    upper = np.triu(valid_pairs & (correlation > 0.0), k=1)
    if np.any(upper):
        log_alpha_onset = float(np.min(log_p[upper]))
    else:
        log_alpha_onset = None

    return CorrelationStatistics(
        correlation=correlation,
        log_p=log_p,
        valid_pairs=valid_pairs,
        labels=normalized.labels,
        constant_indices=normalized.constant_indices,
        n_samples=n_samples,
        log_alpha_onset=log_alpha_onset,
    )


def estimate_null_positive_correlation(
    normalized: NormalizedInput,
    *,
    signature: Callable[[float], Any],
    seed: int = 0,
    cancel_requested: Optional[Callable[[int], bool]] = None,
) -> NullAlphaEstimate:
    """Estimate the fixed-budget positive Spearman null envelope."""

    if not isinstance(normalized, NormalizedInput):
        raise TypeError("normalized must be a NormalizedInput instance.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer.")

    seed = int(seed)
    rng = np.random.default_rng(seed)

    data = normalized.data
    _, n_objectives = data.shape

    if n_objectives < 2:

        def maxima():
            while True:
                yield 0.0

    else:
        standardized_ranks = _rank_standardize(data)
        triu_idx = np.triu_indices(n_objectives, k=1)
        constant_list = list(normalized.constant_indices)

        def maxima():
            permuted = np.empty_like(standardized_ranks)
            while True:
                for column in range(n_objectives):
                    permuted[:, column] = rng.permutation(
                        standardized_ranks[:, column]
                    )
                correlation = permuted.T @ permuted
                np.clip(correlation, -1.0, 1.0, out=correlation)
                if constant_list:
                    correlation[constant_list, :] = 0.0
                    correlation[:, constant_list] = 0.0
                values = correlation[triu_idx]
                yield float(max(0.0, np.max(values, initial=0.0)))

    result = estimate_null_envelope_from_maxima(
        maxima(),
        n_samples=normalized.n_samples,
        signature=signature,
        cancel_requested=cancel_requested,
    )
    return replace(
        result,
        seed=seed,
        rng_state=copy.deepcopy(rng.bit_generator.state),
    )
