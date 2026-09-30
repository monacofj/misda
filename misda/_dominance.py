# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Observed dominance-preservation diagnostics for objective projection."""

from __future__ import annotations

import numpy as np


def prepare_dominance_pairs(Y):
    """Precompute unordered-pair objective signs and the full-space no-D mask."""

    data = np.asarray(Y, dtype=float)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("Y must be a non-empty two-dimensional matrix.")

    left, right = np.triu_indices(data.shape[0], k=1)
    signs = np.sign(data[left] - data[right]).astype(np.int8, copy=False)

    left_dominates = (signs <= 0).all(axis=1) & (signs < 0).any(axis=1)
    right_dominates = (signs >= 0).all(axis=1) & (signs > 0).any(axis=1)
    no_dominance = ~(left_dominates | right_dominates)
    return signs, no_dominance


def evaluate_dominance_preservation(prepared, selected_indices):
    """Measure new dominance relations introduced by objective projection.

    The denominator contains unordered observation pairs for which neither row
    dominates the other in the full objective space. The numerator counts how
    many of those pairs acquire a strict dominance relation after retaining
    only the selected objectives.

    A value of zero means that the projection introduced no new observed
    dominance relation. Lower values are more conservative.
    """

    signs, no_dominance = prepared
    selected = tuple(sorted(set(int(index) for index in selected_indices)))
    if not selected:
        raise ValueError("selected_indices must not be empty.")
    if any(index < 0 or index >= signs.shape[1] for index in selected):
        raise IndexError("selected_indices contains an out-of-range objective.")

    projected = signs[:, selected]
    left_dominates = (projected <= 0).all(axis=1) & (projected < 0).any(axis=1)
    right_dominates = (projected >= 0).all(axis=1) & (projected > 0).any(axis=1)
    new_dominance = no_dominance & (left_dominates | right_dominates)

    denominator = int(no_dominance.sum())
    numerator = int(new_dominance.sum())
    rate = float(numerator / denominator) if denominator else 0.0

    return {
        "new_dominance_rate": rate,
        "new_dominance_pairs": numerator,
        "original_no_dominance_pairs": denominator,
        "exact_preservation": numerator == 0,
    }
