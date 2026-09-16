# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rich stored-state reporting for the public MISDA result model.

The renderer is deliberately side-effect free: it consumes only information
already stored on an ``MISSet`` and its candidates.  It never runs discovery,
evaluation, ranking, or benchmark validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MetricMetadata:
    """Stable display metadata for one scalar result metric."""

    technical: str
    intuitive: str
    kind: str = "float"


METRIC_METADATA = {
    "mean_r2": MetricMetadata(
        "mean external R² over eliminated objectives",
        "average reconstruction quality",
    ),
    "worst_r2": MetricMetadata(
        "minimum external R² over eliminated objectives",
        "weakest reconstructed objective",
    ),
    "mean_r2_se": MetricMetadata(
        "delete-one jackknife SE of mean R²",
        "sampling uncertainty of the average",
    ),
    "worst_r2_se": MetricMetadata(
        "delete-one jackknife SE of worst R²",
        "sampling uncertainty of the weakest result",
    ),
    "jackknife_n": MetricMetadata(
        "number of delete-one jackknife replicates",
        "sampling-uncertainty replication count",
        "integer",
    ),
    "pareto_retention": MetricMetadata(
        "full-front points retained by the reduced front",
        "coverage of the observed trade-offs",
    ),
    "pareto_validity": MetricMetadata(
        "reduced-front points belonging to the full front",
        "precision of the reduced trade-offs",
    ),
    "pareto_jaccard": MetricMetadata(
        "Jaccard overlap of full and reduced observed fronts",
        "overall front agreement",
    ),
    "full_front_size": MetricMetadata(
        "number of observations on the full observed front",
        "size of the original trade-off set",
        "integer",
    ),
    "reduced_front_size": MetricMetadata(
        "number of observations on the reduced front",
        "size of the reduced trade-off set",
        "integer",
    ),
    "intersection_size": MetricMetadata(
        "observations shared by full and reduced fronts",
        "trade-offs preserved by both views",
        "integer",
    ),
    "union_size": MetricMetadata(
        "observations present on either observed front",
        "combined trade-off coverage",
        "integer",
    ),
    "exact_preservation": MetricMetadata(
        "equality of the observed full and reduced front masks",
        "whether every observed trade-off is preserved exactly",
        "boolean",
    ),
    "n_trees": MetricMetadata(
        "trees used by nonlinear reconstruction",
        "nonlinear model effort determined by the evaluator",
        "integer",
    ),
    "converged": MetricMetadata(
        "whether the evaluator's stopping criterion was met",
        "whether computational uncertainty is controlled",
        "boolean",
    ),
    "cancelled": MetricMetadata(
        "whether evaluation was explicitly cancelled",
        "whether the stored result is intentionally incomplete",
        "boolean",
    ),
    "mean_null_r2": MetricMetadata(
        "mean R² under destroyed association",
        "reconstruction expected from the nonlinear null reference",
    ),
    "above_null_r2": MetricMetadata(
        "observed nonlinear mean R² minus the null mean",
        "reconstruction beyond the null reference",
    ),
    "incidental_reconstruction_rate": MetricMetadata(
        "null exceedance frequency for reconstruction quality",
        "frequency of equally good incidental reconstruction",
    ),
    "n_permutations": MetricMetadata(
        "permutations used by the nonlinear null reference",
        "null-reference computational effort",
        "integer",
    ),
    "mc_se_mean_null_r2": MetricMetadata(
        "Monte Carlo SE of the nonlinear null mean R²",
        "computational uncertainty of the null baseline",
    ),
    "above_null_r2_se": MetricMetadata(
        "Monte Carlo SE carried by above-null R²",
        "computational uncertainty of the gain over the null",
    ),
    "incidental_reconstruction_rate_se": MetricMetadata(
        "Monte Carlo SE of the incidental reconstruction rate",
        "computational uncertainty of the null exceedance frequency",
    ),
}


def _format_value(value: Any, kind: str = "float") -> str:
    if value is None:
        return "N/A"
    if kind == "boolean":
        return "yes" if bool(value) else "no"
    if kind == "integer":
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _format_scalar(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4g}"
    return str(value)


def _metric_line(name, value, *, reason=None, indent="      "):
    metadata = METRIC_METADATA[name]
    rendered = _format_value(value, metadata.kind)
    if reason:
        if value is None or name in {"converged", "cancelled"}:
            rendered += f" [{reason}]"
    return (
        f"{indent}{name:<32}: {rendered:<18} — "
        f"{metadata.technical} ({metadata.intuitive})"
    )


def _reconstruction_lines(metrics, *, indent="      "):
    reasons = metrics.reason_by_metric or {}
    jackknife = metrics.jackknife
    return [
        _metric_line(
            "mean_r2",
            metrics.mean_r2,
            reason=reasons.get("mean_r2"),
            indent=indent,
        ),
        _metric_line(
            "worst_r2",
            metrics.worst_r2,
            reason=reasons.get("worst_r2"),
            indent=indent,
        ),
        _metric_line(
            "mean_r2_se",
            jackknife.mean_r2_se,
            reason=jackknife.reason,
            indent=indent,
        ),
        _metric_line(
            "worst_r2_se",
            jackknife.worst_r2_se,
            reason=jackknife.reason,
            indent=indent,
        ),
        _metric_line(
            "jackknife_n",
            jackknife.n_replicates,
            indent=indent,
        ),
    ]


def _pareto_lines(metrics, *, indent="      "):
    return [
        _metric_line("pareto_retention", metrics.retention, indent=indent),
        _metric_line("pareto_validity", metrics.validity, indent=indent),
        _metric_line("pareto_jaccard", metrics.jaccard, indent=indent),
        _metric_line("full_front_size", metrics.full_front_size, indent=indent),
        _metric_line("reduced_front_size", metrics.reduced_front_size, indent=indent),
        _metric_line("intersection_size", metrics.intersection_size, indent=indent),
        _metric_line("union_size", metrics.union_size, indent=indent),
        _metric_line(
            "exact_preservation",
            metrics.exact_preservation,
            indent=indent,
        ),
    ]


def _nonlinear_lines(metrics, *, indent="      "):
    lines = _reconstruction_lines(metrics, indent=indent)
    lines.extend(
        [
            _metric_line("n_trees", metrics.n_trees, indent=indent),
            _metric_line(
                "converged",
                metrics.converged,
                reason=metrics.convergence_reason,
                indent=indent,
            ),
            _metric_line(
                "cancelled",
                metrics.cancelled,
                reason=metrics.convergence_reason if metrics.cancelled else None,
                indent=indent,
            ),
        ]
    )
    null = metrics.null_reference
    if null is not None:
        lines.append("    null_reference")
        lines.extend(
            [
                _metric_line("mean_null_r2", null.mean_null_r2, indent=indent),
                _metric_line("above_null_r2", null.above_null_r2, indent=indent),
                _metric_line(
                    "incidental_reconstruction_rate",
                    null.incidental_reconstruction_rate,
                    indent=indent,
                ),
                _metric_line("n_permutations", null.n_permutations, indent=indent),
                _metric_line(
                    "mc_se_mean_null_r2",
                    null.mc_se_mean_null_r2,
                    indent=indent,
                ),
                _metric_line(
                    "above_null_r2_se",
                    null.above_null_r2_se,
                    indent=indent,
                ),
                _metric_line(
                    "incidental_reconstruction_rate_se",
                    null.incidental_reconstruction_rate_se,
                    indent=indent,
                ),
                _metric_line(
                    "converged",
                    null.converged,
                    reason=null.reason,
                    indent=indent,
                ),
                _metric_line(
                    "cancelled",
                    null.cancelled,
                    reason=null.reason if null.cancelled else None,
                    indent=indent,
                ),
            ]
        )
    return lines


def _candidate_lines(result, candidate_index, group_number):
    candidate = result[candidate_index]
    structural = candidate.structural
    labels = [str(label) for label in candidate.objectives]
    lines = [
        f"  candidate[{candidate_index}] group={group_number} "
        f"size={candidate.size} objectives={labels}",
        "    structural",
        f"      neighborhood                    : {structural.neighborhood}",
        f"      neighborhood_ratio              : {structural.neighborhood_ratio:.4f}",
        f"      span                            : {structural.span}",
        f"      avg_external_degree             : {structural.avg_external_degree:.4f}",
        f"      avg_internal_degree             : {structural.avg_internal_degree:.4f}",
    ]

    if candidate.linear is not None:
        lines.append("    linear_reconstruction")
        lines.extend(_reconstruction_lines(candidate.linear))

    if candidate.pareto is not None:
        lines.append("    pareto_preservation")
        lines.extend(_pareto_lines(candidate.pareto))

    if candidate.nonlinear is not None:
        lines.append("    nonlinear_reconstruction")
        lines.extend(_nonlinear_lines(candidate.nonlinear))

    if (
        candidate.linear is None
        and candidate.pareto is None
        and candidate.nonlinear is None
    ):
        lines.append("    evaluation not requested")
    return lines


def _support_lines(result):
    support = result.support
    if support is None:
        return ["Dimensional support: N/A"]

    lines = [f"Dimensional support: {support.status}"]
    for item in support.results:
        reasons = ", ".join(item.reasons) or "none"
        lines.append(
            f"  candidate[{item.candidate_index}]: {item.status}; reasons={reasons}"
        )
        lines.append(
            "    transitivity: "
            f"observed={item.transitivity_observed:.4f}; "
            f"null={item.transitivity_null:.4f}; "
            f"excess={item.transitivity_excess:.4f}"
        )
        lines.append(
            "    spectral    : "
            f"tested_dimension={item.spectral_tested_dimension}; "
            f"observed_next={item.spectral_observed_next_eigenvalue:.4f}; "
            f"null_next={item.spectral_null_next_eigenvalue:.4f}; "
            f"excess={item.spectral_excess:.4f}"
        )
        lines.append(
            f"    permutations: {item.n_permutations}; seed={item.seed}"
        )
    return lines


def _evaluation_scope_lines(result):
    lines = ["Evaluation scope:"]
    observed = False
    for family in ("linear", "pareto", "nonlinear"):
        scope = result.evaluation_scope(family)
        if scope is None:
            continue
        observed = True
        count, basis = scope
        lines.append(
            f"  {family:<9}: {count}/{len(result)} candidates ({basis})"
        )
        if count != len(result):
            lines.append(
                f"  Note: {family} metrics were evaluated for "
                f"{count} of {len(result)} candidates only ({basis})."
            )
    if not observed:
        lines.append("  no candidate evaluation requested")
    return lines


def _pareto_stability_lines(result):
    diagnostics = getattr(result, "pareto_stability", None)
    if diagnostics is None:
        return []
    ranking = result.structural_ranking
    selected_index = ranking.indices[0] if ranking.indices else None
    selected_epsilon = (
        diagnostics.epsilon_for_candidate(selected_index)
        if selected_index is not None
        else None
    )
    return [
        "Pareto stability (observed Y only):",
        "  Observed front: "
        f"{diagnostics.observed_front_size}/{result._data.shape[0]} "
        f"(fraction={_format_value(diagnostics.observed_front_fraction)})",
        "  Dominance margin: "
        f"min={_format_value(diagnostics.dominance_margin_min)}, "
        f"median={_format_value(diagnostics.dominance_margin_median)}, "
        f"max={_format_value(diagnostics.dominance_margin_max)} "
        "(smaller = more perturbation-sensitive exact membership)",
        "  Additive epsilon+: "
        f"{_format_value(selected_epsilon)} "
        "(range-normalized P_R -> P_Y; smaller = closer full-space approximation)",
    ]


def render_mis_set_report(result):
    """Render a rich audit of already stored MISDA discovery/evaluation state."""

    analysis = result.analysis
    ranking = result.structural_ranking
    separation = getattr(analysis.separation_status, "value", analysis.separation_status)
    selected_index = ranking.indices[0] if ranking.indices else None
    selected_dimension = ranking.selected_dimension

    lines = [f"MISDA report: {result.name or 'Untitled'}", "=" * 72]
    lines.extend(
        [
            "Dimensions:",
            f"  Original       : {analysis.original_dimension}",
            f"  Latent         : {analysis.latent_dimension}",
            f"  Structural     : {analysis.structural_dimension}",
            "  Selected       : "
            f"{selected_dimension if selected_dimension is not None else 'N/A'} "
            f"(candidate[{selected_index}] under {ranking.policy})"
            if selected_index is not None
            else "  Selected       : N/A",
            "Graph topology:",
            "  G± dependence  : "
            f"nodes={analysis.dependence_graph.number_of_nodes()}; "
            f"edges={analysis.dependence_graph.number_of_edges()}; "
            f"components={len(analysis.latent_components)}",
            "  G+ structural  : "
            f"nodes={analysis.structural_graph.number_of_nodes()}; "
            f"edges={analysis.structural_graph.number_of_edges()}; "
            f"components={len(analysis.structural_components)}",
            "Threshold calibration:",
            f"  alpha_onset    : {_format_scalar(analysis.alpha_onset)}",
            f"  alpha_null     : {_format_scalar(analysis.alpha_null)}",
            f"  alpha          : {_format_scalar(analysis.alpha)}",
            f"  aggressiveness : {analysis.aggressiveness:.4f}",
            f"  separation     : {separation}",
            "Null envelope:",
            "  completed      : "
            f"{'yes' if analysis.alpha_null_converged else 'no'}",
            f"  permutations   : {analysis.alpha_null_permutations}",
            f"  reason         : {analysis.alpha_null_reason or 'none'}",
            "Structural ranking:",
            f"  Policy         : {ranking.policy}",
            f"  MISs           : {len(result)}",
            f"  Selected       : candidate[{selected_index}]"
            if selected_index is not None
            else "  Selected       : N/A",
        ]
    )

    tie_counts = ", ".join(
        f"group {position}={len(group)}"
        for position, group in enumerate(ranking.groups, start=1)
    )
    lines.append(f"  Tie groups     : {tie_counts or 'none'}")
    lines.extend(_support_lines(result))
    lines.extend(_evaluation_scope_lines(result))

    representative_indices = []
    group_by_index = {}
    for group_number, group in enumerate(ranking.groups, start=1):
        for index in group:
            group_by_index[index] = group_number
        if group and len(representative_indices) < 3:
            representative_indices.append(group[0])

    for index, candidate in enumerate(result):
        if candidate.nonlinear is not None and index not in representative_indices:
            representative_indices.append(index)

    lines.append(
        "Candidates: one representative from the first three structural tie "
        "groups, plus any candidate with nonlinear evidence"
    )
    for index in representative_indices:
        lines.extend(_candidate_lines(result, index, group_by_index.get(index, "N/A")))

    lines.extend(_pareto_stability_lines(result))
    return "\n".join(lines)


def _install():
    """Install the rich renderer as the public ``MISSet.report`` implementation."""

    api_module = importlib.import_module(f"{__package__}.api")
    api_module.MISSet.report = render_mis_set_report


_install()
