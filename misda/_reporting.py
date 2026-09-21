# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rich stored-state reporting for the public MISDA result model.

The renderer is deliberately side-effect free: it consumes only information
already stored on an ``MISSet`` and its candidates.  It never runs discovery,
evaluation, ranking, or benchmark validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import textwrap
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


def _format_range(values, *, precision=4):
    values = tuple(values)
    if not values:
        return "N/A"
    low = min(values)
    high = max(values)
    low_text = f"{float(low):.{precision}f}"
    high_text = f"{float(high):.{precision}f}"
    if low_text == high_text:
        return low_text
    return f"{low_text}–{high_text}"


def _format_integer_range(values):
    values = tuple(int(value) for value in values)
    if not values:
        return "N/A"
    low = min(values)
    high = max(values)
    return str(low) if low == high else f"{low}–{high}"


REPORT_WIDTH = 100


def _wrapped_annotation(
    label,
    value,
    technical,
    intuitive=None,
    *,
    indent="  ",
    label_width=15,
    value_width=40,
    compact_label=False,
    width=REPORT_WIDTH,
):
    """Render an aligned value plus technical/intuitive interpretation.

    Short annotations stay on one line. When the complete annotation would
    exceed the target width, the value remains on the first line and
    explanatory text continues from the value column. The intuitive gloss,
    when present, occupies its own parenthesized continuation line.
    """

    rendered = str(value)
    if compact_label:
        prefix = f"{indent}{label}: "
    else:
        prefix = f"{indent}{label:<{label_width}}: "
    value_column = len(prefix)

    one_line = f"{prefix}{rendered:<{value_width}} — {technical}"
    if intuitive:
        one_line += f" ({intuitive})"
    if len(one_line) <= width:
        return one_line

    lines = [f"{prefix}{rendered}"]
    continuation = " " * value_column
    technical_prefix = f"{continuation}— "
    technical_width = max(20, width - len(technical_prefix))
    technical_parts = textwrap.wrap(
        technical,
        width=technical_width,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]
    lines.append(technical_prefix + technical_parts[0])
    for part in technical_parts[1:]:
        lines.append(f"{continuation}  {part}")

    if intuitive:
        intuitive_prefix = f"{continuation}  "
        intuitive_width = max(20, width - len(intuitive_prefix) - 2)
        intuitive_parts = textwrap.wrap(
            intuitive,
            width=intuitive_width,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        if len(intuitive_parts) == 1:
            lines.append(f"{intuitive_prefix}({intuitive_parts[0]})")
        else:
            lines.append(f"{intuitive_prefix}({intuitive_parts[0]}")
            for part in intuitive_parts[1:-1]:
                lines.append(f"{intuitive_prefix} {part}")
            lines.append(f"{intuitive_prefix} {intuitive_parts[-1]})")

    return "\n".join(lines)


def _explained_line(
    label,
    value,
    explanation,
    *,
    indent="  ",
    label_width=15,
    value_width=40,
    compact_label=False,
):
    """Render a scalar report line with a concise aligned interpretation."""

    return _wrapped_annotation(
        label,
        value,
        explanation,
        indent=indent,
        label_width=label_width,
        value_width=value_width,
        compact_label=compact_label,
    )


def _ranking_policy_explanation(policy):
    if policy == "size_span":
        return "MIS order: size descending, then span descending"
    return "named MIS ordering policy"


def _metric_line(name, value, *, reason=None, indent="      "):
    metadata = METRIC_METADATA[name]
    rendered = _format_value(value, metadata.kind)
    if reason:
        if value is None or name in {"converged", "cancelled"}:
            rendered += f" [{reason}]"
    return _wrapped_annotation(
        name,
        rendered,
        metadata.technical,
        metadata.intuitive,
        indent=indent,
        label_width=32,
        value_width=18,
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


def _pareto_lines(metrics, *, n_observations=None, indent="      "):
    full = int(metrics.full_front_size)
    preserved = int(metrics.intersection_size)
    lost = max(0, full - preserved)
    retention = metrics.retention
    front_loss = (lost / full) if full > 0 else None
    population_impact = (
        lost / int(n_observations)
        if n_observations is not None and int(n_observations) > 0
        else None
    )

    summary = [
        _explained_line(
            "Original front",
            f"{full}/{n_observations if n_observations is not None else '?'}",
            "full-space nondominated observations",
            indent=indent,
            label_width=32,
            value_width=20,
        ),
        _explained_line(
            "Preserved front",
            f"{preserved}/{full} ({_format_value(retention)})",
            "original-front observations retained after reduction",
            indent=indent,
            label_width=32,
            value_width=20,
        ),
        _explained_line(
            "Front loss",
            f"{lost}/{full} ({_format_value(front_loss)})",
            "fraction of the original front lost after reduction",
            indent=indent,
            label_width=32,
            value_width=20,
        ),
        _explained_line(
            "Population impact",
            (
                f"{lost}/{n_observations if n_observations is not None else '?'} "
                f"({_format_value(population_impact)})"
            ),
            "lost-front observations relative to the full sample",
            indent=indent,
            label_width=32,
            value_width=20,
        ),
    ]
    summary.extend(
        [
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
    )
    return summary


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
        _explained_line(
            "neighborhood",
            structural.neighborhood,
            "outside objectives adjacent to this MIS",
            indent="      ",
            label_width=32,
            value_width=18,
        ),
        _explained_line(
            "neighborhood_ratio",
            f"{structural.neighborhood_ratio:.4f}",
            "fraction of outside objectives covered by this MIS",
            indent="      ",
            label_width=32,
            value_width=18,
        ),
        _explained_line(
            "span",
            structural.span,
            "G+ edges crossing the retained/eliminated split",
            indent="      ",
            label_width=32,
            value_width=18,
        ),
        _explained_line(
            "avg_external_degree",
            f"{structural.avg_external_degree:.4f}",
            "mean crossing-edge degree of retained objectives",
            indent="      ",
            label_width=32,
            value_width=18,
        ),
        _explained_line(
            "avg_internal_degree",
            f"{structural.avg_internal_degree:.4f}",
            "mean G+ degree inside the MIS; zero for an independent set",
            indent="      ",
            label_width=32,
            value_width=18,
        ),
    ]

    if candidate.linear is not None:
        lines.append("    linear_reconstruction")
        lines.extend(_reconstruction_lines(candidate.linear))

    if candidate.pareto is not None:
        lines.append("    pareto_preservation")
        lines.extend(
            _pareto_lines(
                candidate.pareto,
                n_observations=result._data.shape[0],
            )
        )

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

    items = tuple(support.results)
    supported = tuple(item for item in items if item.status == "SUPPORTED")
    unsupported = tuple(item for item in items if item.status == "UNSUPPORTED")
    explanations = {
        "SUPPORTED": "no diagnostic contradiction found",
        "PARTIALLY_SUPPORTED": (
            "diagnostic contradiction found for some first-rank candidates"
        ),
        "UNSUPPORTED": (
            "diagnostic contradiction found for all first-rank candidates"
        ),
    }
    explanation = explanations.get(support.status, "diagnostic status")

    lines = [
        f"Dimensional support: {support.status} — {explanation}",
        _explained_line(
            "First-rank group",
            f"{len(items)} candidates",
            "top-tied MISs tested for dimensional support",
            label_width=17,
            value_width=18,
        ),
        _explained_line(
            "Supported",
            f"{len(supported)}/{len(items)}",
            "candidates with no diagnostic contradiction",
            label_width=17,
            value_width=18,
        ),
        _explained_line(
            "Unsupported",
            f"{len(unsupported)}/{len(items)}",
            "candidates with at least one diagnostic contradiction",
            label_width=17,
            value_width=18,
        ),
    ]
    if not items:
        return lines

    observed_transitivity = tuple(item.transitivity_observed for item in items)
    observed_transitivity_text = _format_range(observed_transitivity)
    if len(set(observed_transitivity)) == 1:
        observed_transitivity_text += " for all candidates"
    lines.extend(
        [
            "  Transitivity:",
            _explained_line(
                "observed",
                observed_transitivity_text,
                "largest indirect-minus-direct positive association",
                indent="    ",
                label_width=17,
                value_width=24,
            ),
            _explained_line(
                "null",
                _format_range(item.transitivity_null for item in items),
                "mean column-permutation reference",
                indent="    ",
                label_width=17,
                value_width=24,
            ),
            _explained_line(
                "excess",
                _format_range(item.transitivity_excess for item in items),
                "observed - null; positive flags transitive chaining",
                indent="    ",
                label_width=17,
                value_width=24,
            ),
            "  Spectral:",
            _explained_line(
                "tested_dimension",
                _format_integer_range(
                    item.spectral_tested_dimension for item in items
                ),
                "latent signal dimension tested",
                indent="    ",
                label_width=17,
                value_width=24,
            ),
            _explained_line(
                "observed_next",
                _format_range(
                    item.spectral_observed_next_eigenvalue for item in items
                ),
                "first rank-correlation eigenvalue beyond tested dimension",
                indent="    ",
                label_width=17,
                value_width=24,
            ),
            _explained_line(
                "null_next",
                _format_range(item.spectral_null_next_eigenvalue for item in items),
                "mean column-permutation reference for that eigenvalue",
                indent="    ",
                label_width=17,
                value_width=24,
            ),
            _explained_line(
                "excess",
                _format_range(item.spectral_excess for item in items),
                "observed_next - null_next; positive flags hidden structure",
                indent="    ",
                label_width=17,
                value_width=24,
            ),
            "  Null reference:",
            _explained_line(
                "permutations",
                _format_integer_range(item.n_permutations for item in items),
                "shared column-permutation replicates",
                indent="    ",
                label_width=17,
                value_width=24,
            ),
            _explained_line(
                "seed",
                _format_integer_range(item.seed for item in items),
                "derived seed shared by first-rank candidates",
                indent="    ",
                label_width=17,
                value_width=24,
            ),
        ]
    )

    reason_counts = {}
    for item in unsupported:
        for reason in item.reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if reason_counts:
        lines.append("  Reasons:")
        for reason, count in sorted(reason_counts.items()):
            lines.append(f"    {reason:<24}: {count} candidates")

    if 0 < len(unsupported) < len(items):
        limit = 10
        lines.append("  Unsupported candidates:")
        for item in unsupported[:limit]:
            reasons = ", ".join(item.reasons) or "none"
            lines.append(
                f"    candidate[{item.candidate_index}]: {reasons}"
            )
        if len(unsupported) > limit:
            lines.append(
                f"    ... and {len(unsupported) - limit} more"
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
            _explained_line(
                family,
                f"{count}/{len(result)} candidates ({basis})",
                "stored evaluation coverage for this metric family",
                label_width=9,
                value_width=42,
            )
        )
        if count != len(result):
            lines.append(
                f"  Note: {family} metrics were evaluated for "
                f"{count} of {len(result)} candidates only ({basis})."
            )
    if not observed:
        lines.append("  no candidate evaluation requested")
    return lines


def _pareto_stability_lines(result, ranking):
    diagnostics = getattr(result, "pareto_stability", None)
    if diagnostics is None:
        return []
    selected_index = ranking.indices[0] if ranking.indices else None
    selected_epsilon = (
        diagnostics.epsilon_for_candidate(selected_index)
        if selected_index is not None
        else None
    )
    return [
        "Pareto stability (observed Y only):",
        _explained_line(
            "Observed front",
            (
                f"{diagnostics.observed_front_size}/{result._data.shape[0]} "
                f"(fraction={_format_value(diagnostics.observed_front_fraction)})"
            ),
            "full-space empirical nondominated set",
            compact_label=True,
            value_width=34,
        ),
        _explained_line(
            "Dominance margin",
            (
                f"min={_format_value(diagnostics.dominance_margin_min)}, "
                f"median={_format_value(diagnostics.dominance_margin_median)}, "
                f"max={_format_value(diagnostics.dominance_margin_max)}"
            ),
            "smaller means more perturbation-sensitive exact membership",
            compact_label=True,
            value_width=48,
        ),
        _explained_line(
            "Additive epsilon+",
            _format_value(selected_epsilon),
            "range-normalized P_R -> P_Y; smaller means closer approximation",
            compact_label=True,
            value_width=34,
        ),
    ]


def _render_complete_report(result, ranking):
    """Render the complete stored-state report under one ranking view."""

    analysis = result.analysis
    separation = getattr(analysis.separation_status, "value", analysis.separation_status)
    selected_index = ranking.indices[0] if ranking.indices else None
    selected_dimension = ranking.selected_dimension

    selected_dimension_text = (
        (
            f"{selected_dimension} "
            f"(candidate[{selected_index}] under {ranking.policy})"
        )
        if selected_index is not None
        else "N/A"
    )
    selected_candidate_text = (
        f"candidate[{selected_index}]" if selected_index is not None else "N/A"
    )
    dependence_topology = (
        f"nodes={analysis.dependence_graph.number_of_nodes()}; "
        f"edges={analysis.dependence_graph.number_of_edges()}; "
        f"components={len(analysis.latent_components)}"
    )
    structural_topology = (
        f"nodes={analysis.structural_graph.number_of_nodes()}; "
        f"edges={analysis.structural_graph.number_of_edges()}; "
        f"components={len(analysis.structural_components)}"
    )

    lines = [f"MISDA report: {result.name or 'Untitled'}", "=" * 72]
    lines.extend(
        [
            "Dimensions:",
            _explained_line(
                "Original",
                analysis.original_dimension,
                "number of input objectives",
            ),
            _explained_line(
                "Latent",
                analysis.latent_dimension,
                "independence number of G± (signed-dependence dimension)",
            ),
            _explained_line(
                "Structural",
                analysis.structural_dimension,
                "independence number of G+ (positive-redundancy dimension)",
            ),
            _explained_line(
                "Selected",
                selected_dimension_text,
                "dimension of the top-ranked MIS",
            ),
            "Graph topology:",
            _explained_line(
                "G± dependence",
                dependence_topology,
                "signed-dependence graph; component count is topology, not dimension",
            ),
            _explained_line(
                "G+ structural",
                structural_topology,
                "positive-redundancy graph; component count is topology, not dimension",
            ),
            "Threshold calibration:",
            _explained_line(
                "alpha_onset",
                _format_scalar(analysis.alpha_onset),
                "observed positive-structure onset",
            ),
            _explained_line(
                "alpha_null",
                _format_scalar(analysis.alpha_null),
                "empirical permutation-null envelope endpoint",
            ),
            _explained_line(
                "alpha",
                _format_scalar(analysis.alpha),
                "threshold actually used to build G+ and G±",
            ),
            _explained_line(
                "aggressiveness",
                f"{analysis.aggressiveness:.4f}",
                "interpolation position: 0=onset, 1=null endpoint",
            ),
            _explained_line(
                "separation",
                separation,
                "whether observed onset is strictly separated from the null endpoint",
            ),
            "Null envelope:",
            _explained_line(
                "completed",
                "yes" if analysis.alpha_null_converged else "no",
                "whether the fixed B=N permutation envelope finished",
            ),
            _explained_line(
                "permutations",
                analysis.alpha_null_permutations,
                "null permutations actually completed",
            ),
            _explained_line(
                "reason",
                analysis.alpha_null_reason or "none",
                "completion or cancellation status of null-envelope calculation",
            ),
            "Structural ranking:",
            _explained_line(
                "Policy",
                ranking.policy,
                _ranking_policy_explanation(ranking.policy),
            ),
            _explained_line(
                "MISs",
                len(ranking),
                "MISs included in this ranking view",
            ),
            _explained_line(
                "Selected",
                selected_candidate_text,
                "top MIS under this ranking policy",
            ),
        ]
    )

    tie_counts = ", ".join(
        f"group {position}={len(group)}"
        for position, group in enumerate(ranking.groups, start=1)
    )
    lines.append(
        _explained_line(
            "Tie groups",
            tie_counts or "none",
            "scientific ties under the ranking criteria",
        )
    )
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

    lines.extend(_pareto_stability_lines(result, ranking=ranking))
    return "\n".join(lines)


def render_ranking_report(ranking):
    """Render a complete, self-contained report for one Ranking view."""

    return _render_complete_report(ranking.mis_set, ranking)


def render_mis_report(candidate):
    """Render evidence intrinsic to one already-selected MIS."""

    result = candidate._owner()
    structural = candidate.structural
    labels = [str(label) for label in candidate.objectives]
    lines = [
        f"MIS report: {result.name or 'Untitled'}",
        "=" * 72,
        f"Dimension       : {candidate.size}",
        f"Objectives      : {labels}",
        "Structural:",
        _explained_line(
            "neighborhood",
            structural.neighborhood,
            "outside objectives adjacent to this MIS",
            label_width=32,
            value_width=18,
        ),
        _explained_line(
            "neighborhood_ratio",
            f"{structural.neighborhood_ratio:.4f}",
            "fraction of outside objectives covered by this MIS",
            label_width=32,
            value_width=18,
        ),
        _explained_line(
            "span",
            structural.span,
            "G+ edges crossing the retained/eliminated split",
            label_width=32,
            value_width=18,
        ),
        _explained_line(
            "avg_external_degree",
            f"{structural.avg_external_degree:.4f}",
            "mean crossing-edge degree of retained objectives",
            label_width=32,
            value_width=18,
        ),
        _explained_line(
            "avg_internal_degree",
            f"{structural.avg_internal_degree:.4f}",
            "mean G+ degree inside the MIS; zero for an independent set",
            label_width=32,
            value_width=18,
        ),
    ]

    if candidate.linear is not None:
        lines.append("Linear reconstruction:")
        lines.extend(_reconstruction_lines(candidate.linear, indent="  "))

    if candidate.pareto is not None:
        lines.append("Pareto preservation:")
        lines.extend(
            _pareto_lines(
                candidate.pareto,
                n_observations=result._data.shape[0],
                indent="  ",
            )
        )

    if candidate.nonlinear is not None:
        lines.append("Nonlinear reconstruction:")
        lines.extend(_nonlinear_lines(candidate.nonlinear, indent="  "))

    if (
        candidate.linear is None
        and candidate.pareto is None
        and candidate.nonlinear is None
    ):
        lines.append("Evaluation: not requested")

    return "\n".join(lines)


