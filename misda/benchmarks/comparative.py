# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Comparison utilities for MISDA and PCA benchmark experiments."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from ..api import MISSet


COMMON_RECONSTRUCTION_METRIC = "global_standardized_external_r2"


def _as_matrix(data) -> np.ndarray:
    if hasattr(data, "to_numpy"):
        matrix = data.to_numpy(dtype=float)
    else:
        matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("data must be a two-dimensional matrix.")
    if matrix.shape[0] < 3:
        raise ValueError("data must contain at least three observations.")
    if matrix.shape[1] < 1:
        raise ValueError("data must contain at least one objective.")
    if not np.isfinite(matrix).all():
        raise ValueError("data must contain only finite values.")
    return matrix


def _validate_max_components(max_components, maximum):
    if isinstance(max_components, (bool, np.bool_)) or not isinstance(
        max_components, Integral
    ):
        raise TypeError("max_components must be an integer.")
    max_components = int(max_components)
    if max_components < 1:
        raise ValueError("max_components must be at least 1.")
    return min(max_components, int(maximum))


def global_standardized_reconstruction_r2(observed, reconstructed) -> float:
    """Return equal-objective-weight reconstruction R² in the original space.

    Each non-constant objective contributes its own reconstruction R² and the
    returned value is their arithmetic mean. This is equivalent to measuring
    squared reconstruction error after variance-standardizing each objective,
    so objectives with different physical scales receive equal weight.
    """

    observed = _as_matrix(observed)
    reconstructed = np.asarray(reconstructed, dtype=float)
    if reconstructed.shape != observed.shape:
        raise ValueError("reconstructed must have the same shape as observed.")
    if not np.isfinite(reconstructed).all():
        raise ValueError("reconstructed must contain only finite values.")

    centered = observed - np.mean(observed, axis=0)
    totals = np.sum(centered * centered, axis=0)
    valid = totals > np.finfo(float).eps
    if not np.any(valid):
        raise ValueError("global reconstruction R² is undefined for all-constant data.")
    residuals = np.sum((observed - reconstructed) ** 2, axis=0)
    per_objective = 1.0 - residuals[valid] / totals[valid]
    return float(np.mean(per_objective))


def misda_global_standardized_external_r2(data, result: MISSet) -> float:
    """Derive the common reconstruction score from an evaluated MISDA set.

    Preserved objectives are represented exactly and therefore contribute R²=1.
    Eliminated objectives contribute the external PRESS/LOO R² stored on the
    candidate selected by the canonical structural ranking. Constant objectives
    are excluded because R² is undefined for them.
    """

    matrix = _as_matrix(data)
    if not isinstance(result, MISSet):
        raise TypeError("result must be an MISSet returned by discover().")
    if matrix.shape != result._data.shape:
        raise ValueError("data shape must match the completed MISDA result.")
    preferred = result.structural_ranking.selected
    if preferred is None:
        raise ValueError("result has no selected structural candidate.")
    linear = preferred.linear
    if linear is None:
        raise ValueError(
            "selected MIS must have linear reconstruction evaluated."
        )

    labels = tuple(
        result.analysis.structural_graph.nodes[index]["label"]
        for index in range(result.analysis.original_dimension)
    )
    selected = set(preferred.indices)
    eliminated_scores = linear.r2_by_objective or {}
    centered = matrix - np.mean(matrix, axis=0)
    totals = np.sum(centered * centered, axis=0)

    scores = []
    for index, label in enumerate(labels):
        if totals[index] <= np.finfo(float).eps:
            continue
        if index in selected:
            scores.append(1.0)
            continue
        value = eliminated_scores.get(label)
        if value is None:
            raise ValueError(
                f"missing defined external reconstruction R² for objective {label!r}."
            )
        scores.append(float(value))
    if not scores:
        raise ValueError("global reconstruction R² is undefined for all-constant data.")
    return float(np.mean(scores))


def pca_in_sample_reconstruction_curve(data, max_components=10) -> list[dict]:
    """Return the conventional in-sample PCA standardized reconstruction curve."""

    matrix = _as_matrix(data)
    max_components = _validate_max_components(
        max_components,
        min(matrix.shape[0], matrix.shape[1]),
    )
    scaled = StandardScaler().fit_transform(matrix)
    curve = []
    for dimension in range(1, max_components + 1):
        pca = PCA(n_components=dimension)
        scores = pca.fit_transform(scaled)
        reconstructed = pca.inverse_transform(scores)
        curve.append(
            {
                "dimension": dimension,
                "global_standardized_r2": float(r2_score(scaled, reconstructed)),
            }
        )
    return curve


def pca_external_reconstruction_curve(data, max_components=10) -> list[dict]:
    """Return leave-one-out PCA reconstruction using the common external score.

    For each held-out row, centering, scaling, and principal directions are fit
    only on the remaining rows. The held-out row is projected into the learned
    PCA representation and reconstructed without influencing that representation.
    All requested dimensions are obtained from the same eigendecomposition for
    each fold, avoiding repeated PCA fits.
    """

    matrix = _as_matrix(data)
    n_samples, n_objectives = matrix.shape
    max_components = _validate_max_components(
        max_components,
        min(n_objectives, n_samples - 1),
    )
    predictions = np.empty(
        (max_components, n_samples, n_objectives),
        dtype=float,
    )

    for held_out in range(n_samples):
        training = np.arange(n_samples) != held_out
        train = matrix[training]
        mean = np.mean(train, axis=0)
        scale = np.std(train, axis=0, ddof=0)
        safe_scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
        standardized = (train - mean) / safe_scale
        covariance = standardized.T @ standardized
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        basis = eigenvectors[:, order[:max_components]]

        held_standardized = (matrix[held_out] - mean) / safe_scale
        coordinates = held_standardized @ basis
        reconstructed_standardized = np.zeros(n_objectives, dtype=float)
        for position in range(max_components):
            reconstructed_standardized += coordinates[position] * basis[:, position]
            predictions[position, held_out] = (
                reconstructed_standardized * safe_scale + mean
            )

    return [
        {
            "dimension": dimension,
            COMMON_RECONSTRUCTION_METRIC: global_standardized_reconstruction_r2(
                matrix,
                predictions[dimension - 1],
            ),
        }
        for dimension in range(1, max_components + 1)
    ]
