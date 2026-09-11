# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Performance-preserving linear reconstruction for full-candidate evaluation.

The public semantics are the same as ``_reconstruction.evaluate_linear_reconstruction``:
external PRESS/LOO R² with delete-one jackknife uncertainty.  For regular full-rank
OLS designs, the jackknife is obtained from one QR factorization using exact case-
deletion identities instead of recomputing a QR factorization for every omitted
sample.  Numerically delicate or rank-deficient cases fall back to the reference
implementation.
"""

from __future__ import annotations

import numpy as np

from ._reconstruction import (
    _jackknife_standard_error,
    _r2_metrics,
    evaluate_linear_reconstruction as _reference_linear_reconstruction,
)


def _fast_jackknife_r2(data, selected, eliminated):
    """Return delete-one jackknife R² values, or ``None`` when fallback is safer.

    For a full-data OLS fit with residual vector ``e`` and hat matrix ``H``, the
    prediction residual for observation ``i`` after observations ``i`` and ``j``
    are deleted is obtained from the 2x2 case-deletion system

        r_ij = ((1-h_jj)e_i + h_ij e_j)
               / ((1-h_ii)(1-h_jj) - h_ij**2).

    For jackknife replicate ``j``, these are exactly the PRESS residuals of the
    dataset with row ``j`` removed.  Thus all N jackknife replicates can be
    computed from one QR factorization of the full design.
    """

    matrix = np.asarray(data, dtype=float)
    n_samples = matrix.shape[0]
    design = np.column_stack((np.ones(n_samples), matrix[:, selected]))

    # A delete-one dataset must still contain enough rows for the regular
    # full-rank path.  The reference implementation remains authoritative for
    # small or degenerate cases.
    if n_samples <= design.shape[1] + 1:
        return None

    try:
        q, r = np.linalg.qr(design, mode="reduced")
    except np.linalg.LinAlgError:
        return None

    if q.shape[1] < design.shape[1]:
        return None
    rank_tolerance = (
        np.finfo(float).eps
        * max(design.shape)
        * max(1.0, float(np.linalg.norm(r, ord=np.inf)))
    )
    if np.any(np.abs(np.diag(r)) <= rank_tolerance):
        return None

    targets = matrix[:, eliminated]
    fitted = q @ (q.T @ targets)
    residuals = targets - fitted
    hat = q @ q.T
    one_minus_leverage = 1.0 - np.diag(hat)

    press_tolerance = np.finfo(float).eps * max(
        10.0, float(design.shape[1])
    )
    if np.any(np.abs(one_minus_leverage) <= press_tolerance):
        return None

    # Pair-deletion denominators.  A near-zero value indicates that at least
    # one reduced design is numerically singular; defer such cases to the
    # reference implementation, which performs its established explicit-LOO
    # fallback when necessary.
    denominator = (
        one_minus_leverage[:, np.newaxis]
        * one_minus_leverage[np.newaxis, :]
        - hat * hat
    )
    off_diagonal = ~np.eye(n_samples, dtype=bool)
    if np.any(np.abs(denominator[off_diagonal]) <= press_tolerance):
        return None

    # The diagonal corresponds to deleting the same observation twice and is
    # not part of any jackknife replicate.  Give it a harmless denominator so
    # vectorized division cannot emit a spurious divide-by-zero warning.
    np.fill_diagonal(denominator, 1.0)

    means = np.mean(targets, axis=0)
    full_sst = np.sum((targets - means) ** 2, axis=0)
    if np.any(full_sst <= np.finfo(float).eps):
        return None

    deleted_sst = (
        full_sst[np.newaxis, :]
        - (n_samples / (n_samples - 1.0))
        * (targets - means[np.newaxis, :]) ** 2
    )
    if np.any(deleted_sst <= np.finfo(float).eps):
        return None

    replicate_r2 = np.empty((n_samples, len(eliminated)), dtype=float)
    for position in range(len(eliminated)):
        e = residuals[:, position]
        pair_residuals = (
            e[:, np.newaxis] * one_minus_leverage[np.newaxis, :]
            + hat * e[np.newaxis, :]
        ) / denominator

        # Column j represents the jackknife dataset with row j omitted.
        # Row j itself is not part of that replicate.
        np.fill_diagonal(pair_residuals, 0.0)
        deleted_sse = np.sum(pair_residuals * pair_residuals, axis=0)
        replicate_r2[:, position] = (
            1.0 - deleted_sse / deleted_sst[:, position]
        )

    return replicate_r2


def evaluate_linear_reconstruction(data, selected_indices, labels):
    """Evaluate exact external linear reconstruction with fast jackknife.

    The returned schema and statistical definitions match the reference engine.
    The optimized path is used only for regular full-rank designs; every
    numerically delicate case delegates to the reference implementation.
    """

    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("data must be a two-dimensional matrix.")
    n_samples, n_objectives = matrix.shape
    selected = tuple(sorted(set(int(index) for index in selected_indices)))
    if not selected:
        raise ValueError("selected_indices must not be empty.")
    if any(index < 0 or index >= n_objectives for index in selected):
        raise IndexError("selected_indices contains an out-of-range objective.")
    if len(labels) != n_objectives:
        raise ValueError("labels must contain one value per objective.")

    eliminated = tuple(
        index for index in range(n_objectives) if index not in selected
    )
    if not eliminated:
        return _reference_linear_reconstruction(matrix, selected, labels)

    full = _r2_metrics(matrix, selected, eliminated, labels)
    replicate_r2 = _fast_jackknife_r2(matrix, selected, eliminated)
    if replicate_r2 is None:
        return _reference_linear_reconstruction(matrix, selected, labels)

    r2_se = {
        labels[objective]: _jackknife_standard_error(
            replicate_r2[:, position].tolist()
        )
        for position, objective in enumerate(eliminated)
    }
    mean_replicates = np.mean(replicate_r2, axis=1).tolist()
    worst_replicates = np.min(replicate_r2, axis=1).tolist()

    reason_by_metric = {}
    if full["mean_r2"] is None:
        reason_by_metric["mean_r2"] = "NO_DEFINED_TARGETS"
    if full["worst_r2"] is None:
        reason_by_metric["worst_r2"] = "NO_DEFINED_TARGETS"

    return {
        **full,
        "reason_by_metric": reason_by_metric,
        "jackknife": {
            "r2_se_by_objective": r2_se,
            "mean_r2_se": _jackknife_standard_error(mean_replicates),
            "worst_r2_se": _jackknife_standard_error(worst_replicates),
            "n_replicates": n_samples,
            "reason": None,
        },
    }
