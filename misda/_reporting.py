# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Stored-result reporting and legacy textual helpers."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricMetadata:
    """Stable display metadata for one scalar result metric."""

    name: str
    technical: str
    intuitive: str
    kind: str = "float"


METRIC_METADATA = {
    "mean_r2": MetricMetadata(
        "mean_r2",
        "mean external R² over eliminated objectives",
        "average reconstruction quality",
    ),
    "worst_r2": MetricMetadata(
        "worst_r2",
        "minimum external R² over eliminated objectives",
        "weakest reconstructed objective",
    ),
    "mean_r2_se": MetricMetadata(
        "mean_r2_se",
        "delete-one jackknife SE of mean R²",
        "sampling uncertainty of the average",
    ),
    "worst_r2_se": MetricMetadata(
        "worst_r2_se",
        "delete-one jackknife SE of worst R²",
        "sampling uncertainty of the weakest result",
    ),
    "pareto_retention": MetricMetadata(
        "pareto_retention",
        "full-front points retained by the reduced front",
        "coverage of the original trade-offs",
    ),
    "pareto_validity": MetricMetadata(
        "pareto_validity",
        "reduced-front points belonging to the full front",
        "precision of the reduced trade-offs",
    ),
    "pareto_jaccard": MetricMetadata(
        "pareto_jaccard",
        "Jaccard overlap of full and reduced fronts",
        "overall front agreement",
    ),
    "full_front_size": MetricMetadata(
        "full_front_size",
        "number of observations on the full front",
        "size of the original trade-off set",
        "integer",
    ),
    "reduced_front_size": MetricMetadata(
        "reduced_front_size",
        "number of observations on the reduced front",
        "size of the reduced trade-off set",
        "integer",
    ),
    "intersection_size": MetricMetadata(
        "intersection_size",
        "observations shared by both fronts",
        "trade-offs preserved by both views",
        "integer",
    ),
    "union_size": MetricMetadata(
        "union_size",
        "observations present on either front",
        "combined trade-off coverage",
        "integer",
    ),
    "exact_preservation": MetricMetadata(
        "exact_preservation",
        "equality of the full and reduced front masks",
        "whether every trade-off is preserved exactly",
        "boolean",
    ),
    "n_trees": MetricMetadata(
        "n_trees",
        "trees required by the uncertainty stopping rule",
        "forest effort determined by the data",
        "integer",
    ),
    "converged": MetricMetadata(
        "converged",
        "whether the data-driven stopping rule was met",
        "whether computational uncertainty is controlled",
        "boolean",
    ),
    "mean_null_r2": MetricMetadata(
        "mean_null_r2",
        "Monte Carlo mean R² under destroyed association",
        "reconstruction expected by chance",
    ),
    "above_null_r2": MetricMetadata(
        "above_null_r2",
        "observed mean R² minus the null mean",
        "reconstruction beyond chance",
    ),
    "incidental_reconstruction_rate": MetricMetadata(
        "incidental_reconstruction_rate",
        "corrected null exceedance frequency",
        "chance of equally good incidental reconstruction",
    ),
    "n_permutations": MetricMetadata(
        "n_permutations",
        "permutations required by the Monte Carlo stopping rule",
        "null-calibration effort determined by the data",
        "integer",
    ),
    "mc_se_mean_null_r2": MetricMetadata(
        "mc_se_mean_null_r2",
        "Monte Carlo SE of the null mean R²",
        "computational uncertainty of the chance baseline",
    ),
    "above_null_r2_se": MetricMetadata(
        "above_null_r2_se",
        "Monte Carlo SE carried by above-null R²",
        "computational uncertainty of the gain",
    ),
    "incidental_reconstruction_rate_se": MetricMetadata(
        "incidental_reconstruction_rate_se",
        "Monte Carlo SE of the incidental rate",
        "computational uncertainty of the chance frequency",
    ),
}


def _format_metric_value(value, kind):
    if value is None:
        return "N/A"
    if kind == "boolean":
        return "yes" if bool(value) else "no"
    if kind == "integer":
        return str(int(value))
    return f"{float(value):.4f}"


def _metric_line(key, value, reason=None, *, indent="      "):
    metadata = METRIC_METADATA[key]
    rendered = _format_metric_value(value, metadata.kind)
    if reason and (value is None or (key == "converged" and not value)):
        rendered = f"{rendered} [{reason}]"
    return (
        f"{indent}{metadata.name} : {rendered}   "
        f"{metadata.technical} ({metadata.intuitive})"
    )


def _reconstruction_lines(block, *, indent="      "):
    reasons = block.get("reason_by_metric", {})
    lines = [
        _metric_line("mean_r2", block.get("mean_r2"), reasons.get("mean_r2"), indent=indent),
        _metric_line("worst_r2", block.get("worst_r2"), reasons.get("worst_r2"), indent=indent),
    ]
    jackknife = block.get("jackknife", {})
    jackknife_reason = jackknife.get("reason")
    lines.extend(
        [
            _metric_line(
                "mean_r2_se",
                jackknife.get("mean_r2_se"),
                jackknife_reason,
                indent=indent,
            ),
            _metric_line(
                "worst_r2_se",
                jackknife.get("worst_r2_se"),
                jackknife_reason,
                indent=indent,
            ),
        ]
    )
    return lines


def _evaluation_lines(evaluation):
    lines = []
    linear = evaluation.get("linear_reconstruction")
    if linear is not None:
        lines.append("    linear_reconstruction")
        lines.extend(_reconstruction_lines(linear))

    pareto = evaluation.get("pareto_preservation")
    if pareto is not None:
        lines.append("    pareto_preservation")
        for key in (
            "pareto_retention",
            "pareto_validity",
            "pareto_jaccard",
            "full_front_size",
            "reduced_front_size",
            "intersection_size",
            "union_size",
            "exact_preservation",
        ):
            lines.append(_metric_line(key, pareto.get(key)))

    nonlinear = evaluation.get("nonlinear_reconstruction")
    if nonlinear is not None:
        lines.append("    nonlinear_reconstruction")
        lines.extend(_reconstruction_lines(nonlinear))
        lines.append(_metric_line("n_trees", nonlinear.get("n_trees")))
        lines.append(
            _metric_line(
                "converged",
                nonlinear.get("converged"),
                nonlinear.get("convergence_reason"),
            )
        )
        null = nonlinear.get("null_reference")
        if null is not None:
            lines.append("    null_reference")
            for key in (
                "mean_null_r2",
                "above_null_r2",
                "incidental_reconstruction_rate",
                "n_permutations",
                "mc_se_mean_null_r2",
                "above_null_r2_se",
                "incidental_reconstruction_rate_se",
                "converged",
            ):
                lines.append(
                    _metric_line(key, null.get(key), null.get("reason"))
                )
    return lines


def render_result_report(result):
    """Render only metrics already stored in a refactored static result."""

    analysis = result.analysis
    lines = [f"MISDA static report: {result.name or 'Untitled'}"]
    lines.append(
        "Dimensions: "
        f"original={analysis.original_dimension}; "
        f"latent components={analysis.latent_dimension}; "
        f"structural components={analysis.structural_dimension}; "
        f"preferred MIS size={result.best_mis.size if result.best_mis else 'N/A'}"
    )
    lines.append(
        f"Separation: {analysis.separation_status.value}; "
        f"aggressiveness={analysis.aggressiveness:.4f}; "
        f"rank_policy={analysis.rank_policy}"
    )
    lines.append(
        f"MIS evaluation: {analysis.n_evaluated_mis} of {analysis.n_mis} "
        f"normally evaluated; {analysis.n_heavy_mis} heavy"
    )
    rank_counts = ", ".join(
        f"rank {rank}={count}"
        for rank, count in sorted(analysis.rank_counts.items())
    )
    lines.append(f"Rank counts: {rank_counts or 'none'}")

    representatives = []
    seen_ranks = set()
    for candidate in result.mis:
        if candidate.rank in seen_ranks:
            continue
        seen_ranks.add(candidate.rank)
        representatives.append(candidate)
        if len(representatives) == 3:
            break
    heavy_candidates = [
        candidate
        for candidate in result.mis
        if "nonlinear_reconstruction" in candidate.evaluation
        and candidate not in representatives
    ]

    lines.append("Candidates:")
    for candidate in [*representatives, *heavy_candidates]:
        lines.append(
            f"  {candidate.id} rank={candidate.rank} size={candidate.size} "
            f"objectives={list(candidate.objectives)}"
        )
        evaluation_lines = _evaluation_lines(candidate.evaluation)
        lines.extend(evaluation_lines or ["    evaluation not requested"])
    return "\n".join(lines)


def explain_ses(out, top_k=8, name=None, show_all=False):
    """
    Explains the result of calculate_ses(out). Returns string report.
    SES = Structural Evidence Score.
    """
    if out is None or not isinstance(out, dict):
        return "explain_ses: 'out' is invalid (expected dict)."

    lines = []
    def _p(x): lines.append(str(x))

    title = f"Structural Evidence Score for {name}" if name else "Structural Evidence Score"
    _p("\n" + " " * 72)
    _p(title)
    _p("-" * 72)

    status = out.get("status", None)
    mis = out.get("mis_size", None)
    if mis is not None:
        _p(f"Surrogate size (mis): {mis}")

    if status == "NO_REDUCTION":
        _p("Status: NO_REDUCTION (All objectives kept; reconstruction N/A).")
        _p("SES = N/A  |  F_real = N/A  |  F_null = N/A")
        return "\n".join(lines)

    ses = out.get("ses", None)
    F_real = out.get("F_real", None)
    F_null = out.get("F_null", None)
    r2_by_target = out.get("r2_real", None)

    if ses is None or F_real is None or F_null is None:
        _p("Status: UNDEFINED or missing metrics ('ses', 'F_real', 'F_null').")
        _p("SES = N/A  |  F_real = N/A  |  F_null = N/A")
        return "\n".join(lines)

    gap = F_real - F_null
    denom = max(1e-15, (1.0 - F_null))
    ses_recalc = np.clip(gap / denom, 0.0, 1.0)

    _p(f"SES = {ses:.4f}  (recalc = {ses_recalc:.4f})")
    _p(f"F_real = {F_real:.4f}  |  F_null = {F_null:.4f}  |  gap = {gap:.4f}")
    _p("Operational interpretation (Structural Evidence Score):")
    _p("  - SES≈1: surrogate reconstructs others very well, far above null.")
    _p("  - SES≈0: surrogate does not reconstruct better than null; suspicious reduction.")
    _p("  - intermediate values: some reconstruction, but there is relevant loss.")

    if ses >= 0.9:
        _p("Short read: strong SES (reduction tends to be safe for reconstruction).")
    elif ses >= 0.7:
        _p("Short read: moderate SES (reduction may work, but deserves checking).")
    else:
        _p("Short read: low SES (high risk of surrogate being too small).")

    if isinstance(r2_by_target, dict) and len(r2_by_target) > 0:
        items = list(r2_by_target.items())
        items_sorted = sorted(items, key=lambda kv: (-(np.inf) if kv[1] is None else kv[1]))
        items_sorted = [(k, (-np.inf if v is None else float(v))) for k, v in items_sorted]
        items_sorted = sorted(items_sorted, key=lambda kv: kv[1])

        worst = items_sorted[:min(top_k, len(items_sorted))]
        best = items_sorted[-min(top_k, len(items_sorted)):] if len(items_sorted) > 1 else []

        def _fmt_r2(v):
            if v is None or np.isneginf(v):
                return "N/A"
            return f"{v:.4f}"

        _p("\nWorst targets (lowest R² in test):")
        for k, v in worst:
            _p(f"  {k}: R² = {_fmt_r2(v)}")

        if best:
            _p("\nBest targets (highest R² in test):")
            for k, v in reversed(best):
                _p(f"  {k}: R² = {_fmt_r2(v)}")

        if show_all:
            _p("\nR² by target (all):")
            for k, v in items_sorted:
                _p(f"  {k}: R² = {_fmt_r2(v)}")
    else:
        _p("\nR² by target is not available.")

    return "\n".join(lines)
