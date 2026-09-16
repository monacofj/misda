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
    _base_report = _api_module._pareto_stability_original_report
else:
    _base_evaluate = _api_module.evaluate
    _base_report = _BaseMISSet.report
    _api_module._pareto_stability_original_evaluate = _base_evaluate
    _api_module._pareto_stability_original_report = _base_report


@dataclass(frozen=True)
class ParetoStabilityDiagnostics:
    """Pareto sensitivity evidence computed entirely from observed ``Y``.

    ``dominance_margin_*`` summarizes, over points in the observed full front,
    the smallest range-normalized additive worsening needed for another
    observed point to weakly dominate that front point. Smaller values indicate
    more perturbation-sensitive exact Pareto membership.

    ``membership_loss_radius_*`` gives the symmetric range-normalized
    L-infinity perturbation radius at which a currently nondominated row can
    become dominated. ``membership_gain_radius_*`` gives the corresponding
    radius at which a currently dominated row can become nondominated.
    ``membership_radius`` is the smaller available radius and therefore the
    infimum perturbation magnitude capable of changing exact Pareto membership.
    These radii quantify susceptibility only; they do not estimate whether
    noise exists or how large it is.

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
    membership_loss_radius_min: Optional[float]
    membership_gain_radius_min: Optional[float]
    membership_radius: Optional[float]
    membership_loss_radius_by_front_index: Tuple[Tuple[int, float], ...]
    membership_gain_radius_by_dominated_index: Tuple[Tuple[int, float], ...]
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


def _front_membership_loss_radii(normalized, front_indices):
    """Return symmetric L-infinity radii for front points to become dominated.

    For front point ``i`` and competitor ``j``, the smallest symmetric
    entrywise perturbation that can make ``j`` dominate ``i`` is half the
    largest positive objective gap between them. The pointwise radius is the
    minimum over competitors.
    """

    margins = _front_dominance_margins(normalized, front_indices)
    return tuple((index, 0.5 * margin) for index, margin in margins)


def _dominated_membership_gain_radii(normalized, front_indices):
    """Return symmetric L-infinity radii for dominated points to enter the front.

    A dominated point ``i`` becomes nondominated only after every current
    dominator ``j`` is prevented from dominating it. For one such ``j``, the
    cheapest objective on which to reverse the relation has gap
    ``min_k(Y[i,k] - Y[j,k])``. Improving ``i`` and worsening ``j`` each by
    ``epsilon`` closes that gap at twice the perturbation rate. The pointwise
    gain radius is therefore half the largest such cheapest gap across all
    current dominators. The value is an infimum; tied coordinates can yield a
    zero radius.
    """

    data = np.asarray(normalized, dtype=float)
    n_rows = data.shape[0]
    front = set(int(index) for index in front_indices)
    gains = []

    for index in range(n_rows):
        if index in front:
            continue

        current = data[index]
        other_mask = np.arange(n_rows) != index
        others = data[other_mask]
        other_rows = np.arange(n_rows)[other_mask]
        dominates = (
            (others <= current).all(axis=1)
            & (others < current).any(axis=1)
        )
        dominator_rows = other_rows[dominates]
        if dominator_rows.size == 0:
            # Defensive fallback for direct helper use with inconsistent front
            # indices; this cannot occur when the indices come from ``data``.
            continue

        gaps = current - data[dominator_rows]
        cheapest_break_by_dominator = np.min(gaps, axis=1)
        radius = 0.5 * float(np.max(cheapest_break_by_dominator))
        gains.append((int(index), max(0.0, radius)))

    return tuple(gains)


def _pareto_membership_radii(normalized, front_indices):
    """Return loss, gain, and overall exact-membership stability radii."""

    loss = _front_membership_loss_radii(normalized, front_indices)
    gain = _dominated_membership_gain_radii(normalized, front_indices)
    loss_min = min((value for _, value in loss), default=None)
    gain_min = min((value for _, value in gain), default=None)
    available = tuple(
        value for value in (loss_min, gain_min) if value is not None
    )
    radius = min(available) if available else None
    return loss, gain, loss_min, gain_min, radius


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
    loss, gain, loss_min, gain_min, membership_radius = (
        _pareto_membership_radii(normalized, full_indices)
    )

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
        membership_loss_radius_min=loss_min,
        membership_gain_radius_min=gain_min,
        membership_radius=membership_radius,
        membership_loss_radius_by_front_index=loss,
        membership_gain_radius_by_dominated_index=gain,
        epsilon_by_candidate=tuple(epsilon),
    )


def _format_metric(value):
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def _report(self):
    lines = _base_report(self).splitlines()
    diagnostics = getattr(self, "pareto_stability", None)
    if diagnostics is None:
        return "\n".join(lines)

    lines.append("Pareto stability (observed Y only):")
    lines.append(
        "  Observed front: "
        f"{diagnostics.observed_front_size}/{self._data.shape[0]} "
        f"(fraction={_format_metric(diagnostics.observed_front_fraction)})"
    )
    lines.append(
        "  Dominance margin: "
        f"min={_format_metric(diagnostics.dominance_margin_min)}, "
        f"median={_format_metric(diagnostics.dominance_margin_median)}, "
        f"max={_format_metric(diagnostics.dominance_margin_max)} "
        "(smaller = more perturbation-sensitive exact membership)"
    )
    lines.append(
        "  Membership radius: "
        f"overall={_format_metric(diagnostics.membership_radius)}, "
        f"loss={_format_metric(diagnostics.membership_loss_radius_min)}, "
        f"gain={_format_metric(diagnostics.membership_gain_radius_min)} "
        "(range-normalized symmetric L_inf susceptibility; does not estimate noise)"
    )
    ranking = self.structural_ranking
    selected_index = ranking.indices[0] if ranking.indices else None
    selected_epsilon = (
        diagnostics.epsilon_for_candidate(selected_index)
        if selected_index is not None
        else None
    )
    lines.append(
        "  Additive epsilon+: "
        f"{_format_metric(selected_epsilon)} "
        "(range-normalized P_R -> P_Y; smaller = closer full-space approximation)"
    )
    return "\n".join(lines)


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
    """Install the additive observed-data Pareto diagnostics on the API module."""

    module = sys.modules.get(f"{__package__}.api")
    if module is not None:
        module.evaluate = evaluate
        module.MISSet.report = _report


_install()
