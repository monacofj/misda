# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Observed-data Pareto stability diagnostics.

These diagnostics use only the matrix ``Y`` supplied to MISDA. They do not
attempt to infer whether perturbations are measurement noise and they do not
consume benchmark truth. The purpose is to distinguish dimensional support
from the stability of exact Pareto membership and from the geometric quality
of a reduced front.

All objectives are minimized. Objective-wise empirical ranges are used for
normalization, yielding a unitless, data-derived scale without a user-selected
threshold. Constant objectives contribute zero normalized distance.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
from typing import Optional, Tuple

import numpy as np

from ._pareto import get_nondominated_mask_minimize


_api_module = importlib.import_module(f"{__package__}.api")
_BaseMISSet = _api_module.MISSet

if hasattr(_api_module, "_pareto_stability_original_evaluate"):
    _base_evaluate = _api_module._pareto_stability_original_evaluate
else:
    _base_evaluate = _api_module.evaluate
    _api_module._pareto_stability_original_evaluate = _base_evaluate


@dataclass(frozen=True)
class ParetoStabilityDiagnostics:
    """Pareto sensitivity evidence computed entirely from observed ``Y``.

    ``dominance_margin_*`` summarizes, over points in the observed full front,
    the smallest range-normalized additive worsening needed for another
    observed point to weakly dominate that front point. Smaller values indicate
    more perturbation-sensitive exact Pareto membership.

    ``epsilon_by_candidate`` stores the range-normalized additive epsilon
    approximation error from each evaluated reduced front ``P_R`` to the
    observed full front ``P_Y``, with candidates indexed in canonical MISSet
    order. Smaller values indicate a closer full-space geometric approximation.
    """

    observed_front_indices: Tuple[int, ...]
    observed_front_size: int
    observed_front_fraction: float
    dominance_margin_min: Optional[float]
    dominance_margin_median: Optional[float]
    dominance_margin_max: Optional[float]
    dominance_margin_by_front_index: Tuple[Tuple[int, float], ...]
    epsilon_by_candidate: Tuple[Optional[float], ...]
    normalization: str = "empirical_range"

    def epsilon_for_candidate(self, candidate_index: int) -> Optional[float]:
        index = int(candidate_index)
        if index < 0 or index >= len(self.epsilon_by_candidate):
            raise IndexError("candidate index out of range")
        return self.epsilon_by_candidate[index]


def _normalize_by_empirical_range(Y):
    data = np.asarray(Y, dtype=float)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("Y must be a non-empty two-dimensional matrix.")
    minimum = np.min(data, axis=0)
    maximum = np.max(data, axis=0)
    scale = maximum - minimum
    normalized = np.zeros_like(data, dtype=float)
    active = scale > 0.0
    if np.any(active):
        normalized[:, active] = (
            data[:, active] - minimum[active]
        ) / scale[active]
    return normalized


def _front_dominance_margins(normalized, front_indices):
    data = np.asarray(normalized, dtype=float)
    n_rows = data.shape[0]
    rows = np.arange(n_rows)
    margins = []
    for front_index in front_indices:
        competitors = rows[rows != front_index]
        if competitors.size == 0:
            continue
        # For minimization, max_j(a_j-b_j) is the additive worsening of b
        # required for competitor a to weakly dominate b. A nondominated b
        # therefore has a non-negative minimum over observed competitors,
        # apart from floating-point roundoff.
        gaps = np.max(data[competitors] - data[front_index], axis=1)
        margin = max(0.0, float(np.min(gaps)))
        margins.append((int(front_index), margin))
    return tuple(margins)


def _normalized_additive_epsilon(normalized, approximation_indices, target_indices):
    """Return unary additive epsilon I_eps+(A, B) for minimization.

    ``A`` is the reduced-front sample set and ``B`` is the observed full front;
    both are evaluated in the complete range-normalized objective space.
    """

    data = np.asarray(normalized, dtype=float)
    approximation = np.asarray(tuple(approximation_indices), dtype=int)
    target = np.asarray(tuple(target_indices), dtype=int)
    if approximation.size == 0 or target.size == 0:
        return None

    worst_target = 0.0
    for target_index in target:
        differences = data[approximation] - data[target_index]
        required = np.max(differences, axis=1)
        best_approximator = float(np.min(required))
        worst_target = max(worst_target, best_approximator)
    return max(0.0, float(worst_target))


def compute_pareto_stability(mis_set):
    """Compute observed-data Pareto stability for an evaluated ``MISSet``."""

    if not isinstance(mis_set, _BaseMISSet):
        raise TypeError("mis_set must be an MISSet.")

    data = np.asarray(mis_set._data, dtype=float)
    full_mask = get_nondominated_mask_minimize(data)
    full_indices = tuple(int(index) for index in np.flatnonzero(full_mask))
    normalized = _normalize_by_empirical_range(data)
    margins = _front_dominance_margins(normalized, full_indices)
    margin_values = tuple(value for _, value in margins)

    epsilon = []
    for candidate in mis_set:
        if candidate.pareto is None:
            epsilon.append(None)
        else:
            epsilon.append(
                _normalized_additive_epsilon(
                    normalized,
                    candidate.pareto.reduced_front_indices,
                    full_indices,
                )
            )

    return ParetoStabilityDiagnostics(
        observed_front_indices=full_indices,
        observed_front_size=len(full_indices),
        observed_front_fraction=float(len(full_indices) / data.shape[0]),
        dominance_margin_min=(min(margin_values) if margin_values else None),
        dominance_margin_median=(
            float(np.median(margin_values)) if margin_values else None
        ),
        dominance_margin_max=(max(margin_values) if margin_values else None),
        dominance_margin_by_front_index=margins,
        epsilon_by_candidate=tuple(epsilon),
    )


def evaluate(
    mis_set,
    *,
    metrics=("linear", "pareto"),
    candidates=None,
    null_reference=False,
    cancel_requested=None,
):
    """Run the canonical evaluator and attach Pareto stability when requested."""

    result = _base_evaluate(
        mis_set,
        metrics=metrics,
        candidates=candidates,
        null_reference=null_reference,
        cancel_requested=cancel_requested,
    )
    if "pareto" in tuple(metrics):
        result.pareto_stability = compute_pareto_stability(result)
    return result


def _install():
    """Install Pareto-stability enrichment on the internal evaluator."""

    module = sys.modules.get(f"{__package__}.api")
    if module is not None:
        module.evaluate = evaluate


_install()
