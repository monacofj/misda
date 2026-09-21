# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pareto observation diagnostics layered onto the benchmark API.

The benchmark truth remains defined externally (for controlled diagnostics,
on clean ``Z``). MISDA discovery/evaluation sees only the observed matrix ``Y``.
This module adds the missing comparison between the observed full Pareto front
``P_Y`` and the declared clean Pareto front ``P_Z`` while preserving the
existing reduction comparison ``P_R`` vs ``P_Y`` and end-to-end comparison
``P_R`` vs ``P_Z``.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
from typing import Optional

import numpy as np

from ._pareto import get_nondominated_mask_minimize


_benchmark_module = importlib.import_module(f"{__package__}.benchmark")
_BaseBenchmarkResult = _benchmark_module.BenchmarkResult

if hasattr(_benchmark_module, "_observation_original_benchmark"):
    _base_benchmark = _benchmark_module._observation_original_benchmark
    _base_compile_benchmark_summary = (
        _benchmark_module._observation_original_compile_benchmark_summary
    )
else:
    _base_benchmark = _benchmark_module.benchmark
    _base_compile_benchmark_summary = _benchmark_module.compile_benchmark_summary
    _benchmark_module._observation_original_benchmark = _base_benchmark
    _benchmark_module._observation_original_compile_benchmark_summary = (
        _base_compile_benchmark_summary
    )


def _set_agreement(predicted_indices, expected_indices):
    predicted = set(int(index) for index in predicted_indices)
    expected = set(int(index) for index in expected_indices)
    intersection = predicted & expected
    union = predicted | expected
    precision = float(len(intersection) / len(predicted)) if predicted else 0.0
    recall = float(len(intersection) / len(expected)) if expected else 0.0
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0
    )
    jaccard = float(len(intersection) / len(union)) if union else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "lost": len(expected - predicted),
        "spurious": len(predicted - expected),
        "exact": bool(predicted == expected),
    }


def _decision_space_effect(result, truth):
    """Return benchmark-only decision-space dimensions for the selected MIS."""
    dependencies = truth.get("objective_dependencies")
    original_dimension = truth.get("original_decision_dimension")
    if dependencies is None or original_dimension is None:
        return None, None, None

    preferred = result.structural_ranking.selected
    if preferred is None:
        return int(original_dimension), None, None

    active = set()
    for objective in preferred.objectives:
        if objective not in dependencies:
            raise ValueError(
                f"truth['objective_dependencies'] has no declaration for selected "
                f"objective {objective!r}."
            )
        active.update(str(variable) for variable in dependencies[objective])
    active_variables = tuple(sorted(active))
    return int(original_dimension), len(active_variables), active_variables


def _format_metric(value):
    if value is None:
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _benchmark_validation_lines(base_lines):
    """Extract only truth-dependent material from the legacy base renderer."""

    try:
        declaration_index = base_lines.index("Declaration")
        observed_index = base_lines.index("Observed analysis")
        assessment_index = base_lines.index("Declaration assessment")
    except ValueError as exc:
        raise RuntimeError(
            "benchmark report structure changed; validation extraction must be updated"
        ) from exc
    return [
        *base_lines[declaration_index:observed_index],
        *base_lines[assessment_index:],
    ]


@dataclass(frozen=True)
class ObservationBenchmarkResult(_BaseBenchmarkResult):
    """Benchmark result extended with ``P_Y`` versus ``P_Z`` evidence."""

    observed_pareto_indices: Optional[tuple] = None
    observation_pareto_precision: Optional[float] = None
    observation_pareto_recall: Optional[float] = None
    observation_pareto_f1: Optional[float] = None
    observation_pareto_jaccard: Optional[float] = None
    observation_pareto_lost: Optional[int] = None
    observation_pareto_spurious: Optional[int] = None
    observation_pareto_exact: Optional[bool] = None
    original_decision_dimension: Optional[int] = None
    active_decision_dimension: Optional[int] = None
    active_decision_variables: Optional[tuple] = None

    def report(self):
        base_lines = super().report().splitlines()
        lines = [
            *base_lines[:2],
            "",
            "MISDA measures (data-derived)",
            "-" * 72,
            *self.result.report().splitlines(),
            "",
            "Benchmark validation (requires declared truth)",
            "-" * 72,
            *_benchmark_validation_lines(base_lines),
        ]

        decision_marker = "Declaration assessment"
        if self.original_decision_dimension is not None:
            try:
                decision_index = lines.index(decision_marker)
            except ValueError:
                decision_index = len(lines)
            variables = (
                ", ".join(self.active_decision_variables)
                if self.active_decision_variables is not None
                else "N/A"
            )
            lines[decision_index:decision_index] = [
                "Decision-space effect of selected MIS",
                "  Original decision dimension : "
                f"{_format_metric(self.original_decision_dimension)}",
                "  Active decision dimension   : "
                f"{_format_metric(self.active_decision_dimension)}",
                f"  Active decision variables   : {variables}",
            ]

        marker = "Pareto declaration agreement"
        try:
            marker_index = lines.index(marker)
        except ValueError:
            marker_index = len(lines)

        if self.pareto_expected is not None:
            observed_size = (
                len(self.observed_pareto_indices)
                if self.observed_pareto_indices is not None
                else None
            )
            observation = [
                "Pareto observation agreement (P_Y vs P_Z)",
                "  Front sizes    : "
                f"clean={len(self.pareto_expected)}, "
                f"observed={_format_metric(observed_size)}",
                "  Set agreement  : "
                f"precision={_format_metric(self.observation_pareto_precision)}, "
                f"recall={_format_metric(self.observation_pareto_recall)}, "
                f"f1={_format_metric(self.observation_pareto_f1)}, "
                f"jaccard={_format_metric(self.observation_pareto_jaccard)}, "
                f"exact={_format_metric(self.observation_pareto_exact)}",
                "  Errors         : "
                f"lost={_format_metric(self.observation_pareto_lost)}, "
                f"spurious={_format_metric(self.observation_pareto_spurious)}",
            ]
            lines[marker_index:marker_index] = observation
            marker_index += len(observation)

        if marker_index < len(lines) and lines[marker_index] == marker:
            lines.insert(
                marker_index + 1,
                "  Pareto basis   : reduced P_R vs clean truth P_Z",
            )

        return "\n".join(lines)


def benchmark(result, truth):
    """Run the benchmark and add the observation effect ``P_Y`` vs ``P_Z``."""

    base = _base_benchmark(result, truth)
    observed_indices = None
    metrics = {
        "precision": None,
        "recall": None,
        "f1": None,
        "jaccard": None,
        "lost": None,
        "spurious": None,
        "exact": None,
    }

    original_decision_dimension, active_decision_dimension, active_decision_variables = (
        _decision_space_effect(result, truth)
    )

    if base.pareto_expected is not None:
        observed_mask = get_nondominated_mask_minimize(result._data)
        observed_indices = tuple(int(index) for index in np.flatnonzero(observed_mask))
        metrics = _set_agreement(observed_indices, base.pareto_expected)

    return ObservationBenchmarkResult(
        **base.__dict__,
        observed_pareto_indices=observed_indices,
        observation_pareto_precision=metrics["precision"],
        observation_pareto_recall=metrics["recall"],
        observation_pareto_f1=metrics["f1"],
        observation_pareto_jaccard=metrics["jaccard"],
        observation_pareto_lost=metrics["lost"],
        observation_pareto_spurious=metrics["spurious"],
        observation_pareto_exact=metrics["exact"],
        original_decision_dimension=original_decision_dimension,
        active_decision_dimension=active_decision_dimension,
        active_decision_variables=active_decision_variables,
    )


def compile_benchmark_summary(results_dict, sort_by=None):
    """Compile the existing summary plus the three-stage Pareto decomposition."""

    frame = _base_compile_benchmark_summary(results_dict, sort_by=None)
    if frame.empty:
        return frame

    observation_jaccard = []
    end_to_end_jaccard = []
    truth_size = []
    observed_size = []
    reduced_size = []
    observed_fraction = []
    additive_epsilon = []
    dominance_margin_min = []
    dominance_margin_median = []
    dominance_margin_max = []
    original_decision_dimension = []
    active_decision_dimension = []

    for case_name in frame["Case"]:
        item = results_dict[case_name]
        result = item.get("result_obj") if isinstance(item, dict) else item
        truth = item.get("truth", {}) if isinstance(item, dict) else {}
        observed = benchmark(result, truth)
        selected = result.structural_ranking.selected
        reduced = selected.pareto if selected is not None else None
        diagnostics = getattr(result, "pareto_stability", None)
        selected_index = (
            result.structural_ranking.indices[0]
            if result.structural_ranking.indices
            else None
        )

        observation_jaccard.append(observed.observation_pareto_jaccard)
        end_to_end_jaccard.append(observed.pareto_jaccard)
        truth_size.append(
            len(observed.pareto_expected)
            if observed.pareto_expected is not None
            else None
        )
        observed_size.append(
            len(observed.observed_pareto_indices)
            if observed.observed_pareto_indices is not None
            else None
        )
        reduced_size.append(reduced.reduced_front_size if reduced is not None else None)
        observed_fraction.append(
            diagnostics.observed_front_fraction if diagnostics is not None else None
        )
        additive_epsilon.append(
            diagnostics.epsilon_for_candidate(selected_index)
            if diagnostics is not None and selected_index is not None
            else None
        )
        dominance_margin_min.append(
            diagnostics.dominance_margin_min if diagnostics is not None else None
        )
        dominance_margin_median.append(
            diagnostics.dominance_margin_median if diagnostics is not None else None
        )
        dominance_margin_max.append(
            diagnostics.dominance_margin_max if diagnostics is not None else None
        )
        original_decision_dimension.append(observed.original_decision_dimension)
        active_decision_dimension.append(observed.active_decision_dimension)

    # Existing ParetoJaccard is P_R vs P_Y. Keep it for compatibility and add
    # an explicit alias beside the new observation and end-to-end quantities.
    frame["ParetoReductionJaccard"] = frame["ParetoJaccard"]
    frame["ParetoObservationJaccard"] = observation_jaccard
    frame["ParetoEndToEndJaccard"] = end_to_end_jaccard
    frame["ParetoTruthSize"] = truth_size
    frame["ParetoObservedSize"] = observed_size
    frame["ParetoReducedSize"] = reduced_size
    frame["ParetoObservedFraction"] = observed_fraction
    frame["ParetoAdditiveEpsilon"] = additive_epsilon
    frame["ParetoDominanceMarginMin"] = dominance_margin_min
    frame["ParetoDominanceMarginMedian"] = dominance_margin_median
    frame["ParetoDominanceMarginMax"] = dominance_margin_max
    frame["OriginalDecisionDimension"] = original_decision_dimension
    frame["ActiveDecisionDimension"] = active_decision_dimension

    if sort_by and sort_by in frame.columns:
        frame = frame.sort_values(by=sort_by)
    return frame


def _install():
    """Install the additive layer on the public benchmark module."""

    module = sys.modules.get(f"{__package__}.benchmark")
    if module is not None:
        module.benchmark = benchmark
        module.compile_benchmark_summary = compile_benchmark_summary


_install()
