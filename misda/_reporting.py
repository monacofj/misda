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
        "mean external R²",
        "how well eliminated objectives reconstruct on average",
    ),
    "worst_r2": MetricMetadata(
        "minimum external R²",
        "how well the hardest eliminated objective reconstructs",
    ),
    "mean_r2_se": MetricMetadata(
        "jackknife SE of mean R²",
        "sampling uncertainty of average reconstruction",
    ),
    "worst_r2_se": MetricMetadata(
        "jackknife SE of worst R²",
        "sampling uncertainty of the weakest reconstruction",
    ),
    "jackknife_n": MetricMetadata(
        "jackknife replicate count",
        "how many delete-one replicates estimate sampling uncertainty",
        "integer",
    ),
    "pareto_retention": MetricMetadata(
        "full-front recall",
        "how much of the original observed trade-off front survives",
    ),
    "pareto_validity": MetricMetadata(
        "reduced-front precision",
        "how much of the reduced front belongs to the original front",
    ),
    "pareto_jaccard": MetricMetadata(
        "front Jaccard overlap",
        "overall agreement between original and reduced observed fronts",
    ),
    "full_front_size": MetricMetadata(
        "full-front size",
        "how many observations are nondominated before reduction",
        "integer",
    ),
    "reduced_front_size": MetricMetadata(
        "reduced-front size",
        "how many observations are nondominated after reduction",
        "integer",
    ),
    "intersection_size": MetricMetadata(
        "front intersection size",
        "how many trade-off observations both views preserve",
        "integer",
    ),
    "union_size": MetricMetadata(
        "front union size",
        "how many trade-off observations appear in either view",
        "integer",
    ),
    "exact_preservation": MetricMetadata(
        "front-mask equality",
        "whether reduction preserves every observed trade-off exactly",
        "boolean",
    ),
    "n_trees": MetricMetadata(
        "nonlinear tree count",
        "how much model effort the nonlinear evaluator used",
        "integer",
    ),
    "converged": MetricMetadata(
        "stopping criterion met",
        "whether computational uncertainty was controlled",
        "boolean",
    ),
    "cancelled": MetricMetadata(
        "evaluation cancelled",
        "whether this stored result was intentionally left incomplete",
        "boolean",
    ),
    "mean_null_r2": MetricMetadata(
        "null mean R²",
        "reconstruction expected after destroying association",
    ),
    "above_null_r2": MetricMetadata(
        "observed-minus-null R²",
        "reconstruction gained beyond the nonlinear null reference",
    ),
    "incidental_reconstruction_rate": MetricMetadata(
        "null exceedance rate",
        "how often the null reconstructs at least this well",
    ),
    "n_permutations": MetricMetadata(
        "null permutation count",
        "how many replicates built the nonlinear null reference",
        "integer",
    ),
    "mc_se_mean_null_r2": MetricMetadata(
        "MC SE of null mean R²",
        "computational uncertainty of the null baseline",
    ),
    "above_null_r2_se": MetricMetadata(
        "MC SE of above-null R²",
        "computational uncertainty of the gain over the null",
    ),
    "incidental_reconstruction_rate_se": MetricMetadata(
        "MC SE of exceedance rate",
        "computational uncertainty of incidental reconstruction frequency",
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


def _wrapped_annotation(
    label,
    value,
    technical,
    intuitive=None,
    *,
    indent="  ",
    label_width=15,
    value_width=12,
    compact_label=False,
):
    """Render one annotated report item on exactly one line."""

    rendered = str(value)
    if compact_label:
        prefix = f"{indent}{label}: "
    else:
        prefix = f"{indent}{label:<{label_width}}: "
    value_field = f"{rendered:<{value_width}}" if value_width > 0 else rendered
    line = f"{prefix}{value_field} — {technical}"
    if intuitive:
        line += f" ({intuitive})"
    return line


def _explained_line(
    label,
    value,
    technical,
    intuitive=None,
    *,
    indent="  ",
    label_width=15,
    value_width=12,
    compact_label=False,
):
    """Render one concise technical definition plus an intuitive gloss."""

    return _wrapped_annotation(
        label,
        value,
        technical,
        intuitive,
        indent=indent,
        label_width=label_width,
        value_width=value_width,
        compact_label=compact_label,
    )


def _ranking_policy_explanation(policy):
    if policy == "size_span":
        return "larger MISs first, then broader span"
    if policy == "size_pareto":
        return "larger MISs first, then greater observed Pareto-front retention"
    return "the named policy determines candidate order"


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
        value_width=10,
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
            "full-front size",
            "how many observations are nondominated before reduction",
            indent=indent,
            label_width=20,
            value_width=16,
        ),
        _explained_line(
            "Preserved front",
            f"{preserved}/{full} ({_format_value(retention)})",
            "retained full-front points",
            "how much of the original front survives reduction",
            indent=indent,
            label_width=20,
            value_width=16,
        ),
        _explained_line(
            "Front loss",
            f"{lost}/{full} ({_format_value(front_loss)})",
            "lost full-front fraction",
            "how much of the original front disappears",
            indent=indent,
            label_width=20,
            value_width=16,
        ),
        _explained_line(
            "Population impact",
            (
                f"{lost}/{n_observations if n_observations is not None else '?'} "
                f"({_format_value(population_impact)})"
            ),
            "lost-front sample fraction",
            "how much of the whole sample is affected",
            indent=indent,
            label_width=20,
            value_width=16,
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
            "external-neighbor count",
            "how many eliminated objectives touch this MIS in G+",
            indent="      ",
            label_width=22,
            value_width=10,
        ),
        _explained_line(
            "neighborhood_ratio",
            f"{structural.neighborhood_ratio:.4f}",
            "external coverage ratio",
            "fraction of eliminated objectives adjacent to this MIS",
            indent="      ",
            label_width=22,
            value_width=10,
        ),
        _explained_line(
            "span",
            structural.span,
            "retained/eliminated edge count",
            "how strongly this MIS connects across the reduction split",
            indent="      ",
            label_width=22,
            value_width=10,
        ),
        _explained_line(
            "avg_external_degree",
            f"{structural.avg_external_degree:.4f}",
            "mean external degree",
            "average eliminated neighbors per retained objective",
            indent="      ",
            label_width=22,
            value_width=10,
        ),
        _explained_line(
            "avg_internal_degree",
            f"{structural.avg_internal_degree:.4f}",
            "mean internal G+ degree",
            "should be zero because an MIS is independent",
            indent="      ",
            label_width=22,
            value_width=10,
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
        f"Dimensional support: {support.status} — support diagnostic ({explanation})",
        _explained_line(
            "First-rank group",
            f"{len(items)} candidates",
            "tested top-rank set",
            "how many tied leaders were checked for support",
            label_width=17,
            value_width=12,
        ),
        _explained_line(
            "Supported",
            f"{len(supported)}/{len(items)}",
            "supported candidates",
            "leaders with no diagnostic contradiction",
            label_width=17,
            value_width=12,
        ),
        _explained_line(
            "Unsupported",
            f"{len(unsupported)}/{len(items)}",
            "unsupported candidates",
            "leaders with at least one diagnostic contradiction",
            label_width=17,
            value_width=12,
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
                "observed transitivity",
                "strongest indirect-vs-direct positive-association gap",
                indent="    ",
                label_width=17,
                value_width=12,
            ),
            _explained_line(
                "null",
                _format_range(item.transitivity_null for item in items),
                "permutation-null mean",
                "expected value after breaking column association",
                indent="    ",
                label_width=17,
                value_width=12,
            ),
            _explained_line(
                "excess",
                _format_range(item.transitivity_excess for item in items),
                "observed − null",
                "positive values flag transitive chaining",
                indent="    ",
                label_width=17,
                value_width=12,
            ),
            "  Spectral:",
            _explained_line(
                "tested_dimension",
                _format_integer_range(
                    item.spectral_tested_dimension for item in items
                ),
                "tested latent dimension",
                "the assumed rank before checking leftover signal",
                indent="    ",
                label_width=17,
                value_width=12,
            ),
            _explained_line(
                "observed_next",
                _format_range(
                    item.spectral_observed_next_eigenvalue for item in items
                ),
                "next observed eigenvalue",
                "first rank-correlation eigenvalue beyond that dimension",
                indent="    ",
                label_width=17,
                value_width=12,
            ),
            _explained_line(
                "null_next",
                _format_range(item.spectral_null_next_eigenvalue for item in items),
                "next null eigenvalue",
                "its expected size after destroying association",
                indent="    ",
                label_width=17,
                value_width=12,
            ),
            _explained_line(
                "excess",
                _format_range(item.spectral_excess for item in items),
                "observed − null",
                "positive values flag hidden spectral structure",
                indent="    ",
                label_width=17,
                value_width=12,
            ),
            "  Null reference:",
            _explained_line(
                "permutations",
                _format_integer_range(item.n_permutations for item in items),
                "shared null replicates",
                "the same permutations are used across tied leaders",
                indent="    ",
                label_width=17,
                value_width=12,
            ),
            _explained_line(
                "seed",
                _format_integer_range(item.seed for item in items),
                "shared derived seed",
                "keeps tied-candidate checks directly comparable",
                indent="    ",
                label_width=17,
                value_width=12,
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
                "stored evaluation scope",
                "how much of the candidate set has this evidence",
                label_width=9,
                value_width=0,
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


def _pareto_stability_lines(result, ranking=None):
    diagnostics = getattr(result, "pareto_stability", None)
    if diagnostics is None:
        return []
    ranking = result.structural_ranking if ranking is None else ranking
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
            value_width=0,
        ),
        _explained_line(
            "Dominance margin",
            (
                f"min={_format_value(diagnostics.dominance_margin_min)}, "
                f"median={_format_value(diagnostics.dominance_margin_median)}, "
                f"max={_format_value(diagnostics.dominance_margin_max)}"
            ),
            "dominance robustness margin",
            "smaller values mean front membership is easier to perturb",
            compact_label=True,
            value_width=0,
        ),
        _explained_line(
            "Additive epsilon+",
            _format_value(selected_epsilon),
            "normalized P_R→P_Y error",
            "smaller means the reduced front better approximates the full front",
            compact_label=True,
            value_width=0,
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
                "observed objective dimension",
                "how many objectives we started with",
            ),
            _explained_line(
                "Latent",
                analysis.latent_dimension,
                "independence number of G±",
                "how many signed-dependence degrees remain",
            ),
            _explained_line(
                "Structural",
                analysis.structural_dimension,
                "independence number of G+",
                "how many positive-redundancy units remain",
            ),
            _explained_line(
                "Selected",
                selected_dimension_text,
                "top-ranked MIS size",
                "how many objectives the preferred reduction keeps",
            ),
            "Graph topology:",
            _explained_line(
                "G± dependence",
                dependence_topology,
                "signed-dependence graph",
                "components describe connectivity, not dimension",
            ),
            _explained_line(
                "G+ structural",
                structural_topology,
                "positive-redundancy graph",
                "components describe connectivity, not dimension",
            ),
            "Threshold calibration:",
            _explained_line(
                "alpha_onset",
                _format_scalar(analysis.alpha_onset),
                "structure-onset threshold",
                "where positive structure first appears",
            ),
            _explained_line(
                "alpha_null",
                _format_scalar(analysis.alpha_null),
                "permutation-null endpoint",
                "where dependence clearly exceeds the null envelope",
            ),
            _explained_line(
                "alpha",
                _format_scalar(analysis.alpha),
                "active dependence threshold",
                "the cutoff actually used to build G+ and G±",
            ),
            _explained_line(
                "aggressiveness",
                f"{analysis.aggressiveness:.4f}",
                "onset→null interpolation",
                "0 is conservative onset; 1 reaches the null endpoint",
            ),
            _explained_line(
                "separation",
                separation,
                "onset/null separation status",
                "whether observed structure separates cleanly from chance",
            ),
            "Null envelope:",
            _explained_line(
                "completed",
                "yes" if analysis.alpha_null_converged else "no",
                "null-envelope completion",
                "whether all planned permutations finished",
            ),
            _explained_line(
                "permutations",
                analysis.alpha_null_permutations,
                "completed null permutations",
                "how many null replicates were actually run",
            ),
            _explained_line(
                "reason",
                analysis.alpha_null_reason or "none",
                "null-envelope stop status",
                "why the null calculation finished or stopped",
            ),
            "Structural ranking:",
            _explained_line(
                "Policy",
                ranking.policy,
                "MIS ordering rule",
                _ranking_policy_explanation(ranking.policy),
            ),
            _explained_line(
                "MISs",
                len(ranking),
                "ranked MIS count",
                "how many candidates are present in this ranking view",
            ),
            _explained_line(
                "Selected",
                selected_candidate_text,
                "top-ranked MIS",
                "the candidate chosen by this ordering",
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
            "ranking ties",
            "candidates in the same group are scientifically tied",
        )
    )
    lines.extend(_support_lines(result))
    if ranking.policy != "size_span":
        lines.append(
            "  Ranking note: dimensional support above belongs to the canonical "
            "size_span first-rank group; alternative rankings do not recompute it."
        )
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


def render_mis_set_report(result):
    """Render the legacy-complete report using the canonical structural ranking."""

    return _render_complete_report(result, result.structural_ranking)


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
            "external-neighbor count",
            "how many eliminated objectives touch this MIS in G+",
            label_width=22,
            value_width=10,
        ),
        _explained_line(
            "neighborhood_ratio",
            f"{structural.neighborhood_ratio:.4f}",
            "external coverage ratio",
            "fraction of eliminated objectives adjacent to this MIS",
            label_width=22,
            value_width=10,
        ),
        _explained_line(
            "span",
            structural.span,
            "retained/eliminated edge count",
            "how strongly this MIS connects across the reduction split",
            label_width=22,
            value_width=10,
        ),
        _explained_line(
            "avg_external_degree",
            f"{structural.avg_external_degree:.4f}",
            "mean external degree",
            "average eliminated neighbors per retained objective",
            label_width=22,
            value_width=10,
        ),
        _explained_line(
            "avg_internal_degree",
            f"{structural.avg_internal_degree:.4f}",
            "mean internal G+ degree",
            "should be zero because an MIS is independent",
            label_width=22,
            value_width=10,
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


def _install():
    """Install the rich renderer as the public ``MISSet.report`` implementation."""

    api_module = importlib.import_module(f"{__package__}.api")
    api_module.MISSet.report = render_mis_set_report


_install()
