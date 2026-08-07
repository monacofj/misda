# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Legacy static-analysis orchestration used by the MISDA public API."""

import time

import numpy as np
from scipy import stats

from ._graph import (
    build_dependency_graphs,
    calculate_component_compactness,
    find_maximal_independent_sets,
    make_structural_signature,
    rank_structural_mis,
    repair_mis_coverage,
)
from ._ranking import compute_mis_metrics, sort_mis_metrics
from ._pareto import (
    evaluate_pareto_preservation,
    get_nondominated_mask_minimize,
)
from ._reconstruction import evaluate_linear_reconstruction
from ._statistics import (
    _correlation_strength,
    AlphaRegime,
    compute_correlation_statistics,
    diagnose_alpha_regime,
    estimate_alpha_interval,
    estimate_null_positive_correlation,
    interpolate_log_alpha,
    separation_status,
    select_alpha,
)
from ._validation import (
    normalize_input_matrix,
    validate_aggressiveness,
    validate_max_evaluated_mis,
    validate_rank_policy,
)
from .result import (
    AnalysisResult,
    ExecutionResult,
    LegacyMISDAResult,
    MISCandidate,
    MISDAResult,
)

def report_significant_correlations(R, z_stat, z_crit, max_pairs=50, label_prefix="f"):
    """
    Generates a string report of significant correlations found in the data.

    Args:
        R (np.ndarray): The correlation matrix.
        z_stat (np.ndarray): The Fisher z-transformed correlation statistics.
        z_crit (float): The critical z-value for significance.
        max_pairs (int): Maximum number of significant pairs to report for each type (positive/negative).
        label_prefix (str): Prefix for feature labels (e.g., "f" for f1, f2).

    Returns:
        str: A formatted string report of significant correlations.
    """
    M = R.shape[0]
    pos_corr = []
    neg_corr = []

    for i in range(M):
        for j in range(i + 1, M):
            if abs(z_stat[i, j]) > z_crit:
                rij = R[i, j]
                if rij > 0:
                    pos_corr.append((i, j, rij))
                elif rij < 0:
                    neg_corr.append((i, j, rij))

    out = []
    out.append("\n--- SIGNIFICANT CORRELATIONS (Fisher z, two-tailed) ---")

    if pos_corr:
        out.append("\nSignificant POSITIVE correlation:")
        for i, j, r in pos_corr[:max_pairs]:
            out.append(f"  {label_prefix}{i+1} – {label_prefix}{j+1}:  ρ = {r:.4f}")
        if len(pos_corr) > max_pairs:
            out.append(f"  ... ({len(pos_corr) - max_pairs} pairs omitted)")
    else:
        out.append("\nSignificant POSITIVE correlation: none")

    if neg_corr:
        out.append("\nSignificant NEGATIVE correlation:")
        for i, j, r in neg_corr[:max_pairs]:
            out.append(f"  {label_prefix}{i+1} – {label_prefix}{j+1}:  ρ = {r:.4f}")
        if len(neg_corr) > max_pairs:
            out.append(f"  ... ({len(neg_corr) - max_pairs} pairs omitted)")
    else:
        out.append("\nSignificant NEGATIVE correlation: none")

    return "\n".join(out)


def misda_significance_from_corr(corr, N, M, alpha=0.05, labels=None, ensure_coverage=True, min_coverage=None):
    """
    Executes MISDA logic given a precomputed correlation matrix.
    """
    if labels is None:
        labels = [f"f{i+1}" for i in range(M)]

    corr = np.clip(corr, -0.999999, 0.999999)

    z = 0.5 * np.log((1 + corr) / (1 - corr))
    sigma_z = 1 / np.sqrt(N - 3)
    z_stat = z / sigma_z

    z_crit = stats.norm.isf(alpha / 2)
    z_threshold = z_crit * sigma_z
    r_crit = np.tanh(z_threshold)
    if r_crit >= 0.99999:
        r_crit = 0.99999

    corr_report = report_significant_correlations(corr, z_stat, z_crit, label_prefix="f")

    signif = (_correlation_strength(z_stat) > z_crit)
    adjacency = signif.astype(int)
    np.fill_diagonal(adjacency, 0)

    visited = [False] * M
    components = []

    def dfs(start):
        stack = [start]
        comp = []
        while stack:
            i = stack.pop()
            if not visited[i]:
                visited[i] = True
                comp.append(i)
                neighbors = np.where(adjacency[i] == 1)[0]
                for j in neighbors:
                    if not visited[j]:
                        stack.append(j)
        return sorted(comp)

    for i in range(M):
        if not visited[i]:
            components.append(dfs(i))

    components_labels = [[labels[i] for i in comp] for comp in components]

    mis_sets = find_maximal_independent_sets(adjacency)

    if ensure_coverage:
        repaired_sets = []
        coverage_threshold = min_coverage if min_coverage is not None else r_crit
        for ms in mis_sets:
            repaired = repair_mis_coverage(corr, ms, min_coverage=coverage_threshold)
            if repaired not in repaired_sets:
                repaired_sets.append(repaired)
        mis_sets = repaired_sets

    mis_sets_labels = [[labels[i] for i in mis] for mis in mis_sets]

    mis_metrics = compute_mis_metrics(mis_sets, adjacency, labels)
    mis_sorted = sort_mis_metrics(mis_metrics)

    def rank_key(m):
        return (
            m["neighborhood"],
            m["span"],
            round(m["avg_external_degree"], 4),
            m["size"],
        )

    mis_ranked = []
    rank_groups = {}
    current_rank = 0
    prev_key = None

    for m in mis_sorted:
        k = rank_key(m)
        if prev_key is None or k != prev_key:
            current_rank += 1
            prev_key = k
        m_with_rank = dict(m)
        m_with_rank["rank"] = current_rank
        mis_ranked.append(m_with_rank)
        rank_groups.setdefault(current_rank, []).append(m_with_rank)

    best_mis_rank1 = rank_groups.get(1, [None])[0]
    best_mis_rank2 = rank_groups.get(2, [None])[0] if 2 in rank_groups else None

    unique_metric_values = {
        "neighborhood": sorted({m["neighborhood"] for m in mis_metrics}),
        "neighborhood_ratio": sorted({m["neighborhood_ratio"] for m in mis_metrics}),
        "span": sorted({m["span"] for m in mis_metrics}),
        "avg_external_degree": sorted({m["avg_external_degree"] for m in mis_metrics}),
        "avg_internal_degree": sorted({m["avg_internal_degree"] for m in mis_metrics}),
    }

    min_compactness, component_metrics, homogeneity_stats = calculate_component_compactness(corr, components)

    return {
        "corr": corr,
        "adjacency": adjacency,
        "components": components,
        "components_labels": components_labels,
        "mis_sets": mis_sets,
        "mis_sets_labels": mis_sets_labels,
        "mis_metrics": mis_metrics,
        "mis_sorted": mis_sorted,
        "mis_ranked": mis_ranked,
        "rank_groups": rank_groups,
        "best_mis_rank1": best_mis_rank1,
        "best_mis_rank2": best_mis_rank2,
        "unique_metric_values": unique_metric_values,
        "min_component_compactness": min_compactness,
        "component_compactness": component_metrics,
        "homogeneity_stats": homogeneity_stats,
        "labels": labels,
        "alpha": alpha,
        "N": N,
        "M": M,
        "sigma_z": sigma_z,
        "z_crit": z_crit,
        "corr_report": corr_report
    }


def misda_significance(Y, alpha=0.05, ensure_coverage=True, min_coverage=None):
    """
    Executes the Maximal Independent Structural Dimensionality Analysis (MISDA) logic.
    """
    normalized = normalize_input_matrix(Y)
    X = normalized.data
    labels = list(normalized.labels)
    N, M = X.shape
    corr = np.corrcoef(X, rowvar=False)
    return misda_significance_from_corr(corr, N, M, alpha=alpha, labels=labels, ensure_coverage=ensure_coverage, min_coverage=min_coverage)


def _analyze_static_fast(Y, corr, alpha_min, alpha_max, alpha_exec, caution=1.0, name=None, ensure_coverage=True):
    """
    Executes static analysis using precomputed correlation matrix and alpha bounds.
    """
    metrics = diagnose_alpha_regime(alpha_min, alpha_max)
    regime = AlphaRegime(metrics["regime"])

    normalized = normalize_input_matrix(Y)
    data = normalized.data
    labels = list(normalized.labels)

    N, M = data.shape
    res = misda_significance_from_corr(corr, N, M, alpha=alpha_exec, labels=labels, ensure_coverage=ensure_coverage)

    return LegacyMISDAResult(
        Y=Y,
        caution=caution,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        metrics=metrics,
        regime=regime,
        alpha_exec=alpha_exec,
        isda_res=res,
        name=name
    )

def _analyze_static(Y, caution=1.0, name=None, ensure_coverage=True, alpha=None):
    """
    Executes the full MISDA pipeline on dataset Y.
    """
    normalize_input_matrix(Y)
    alpha_min, alpha_max, r_max_real, r_null = estimate_alpha_interval(Y)
    metrics = diagnose_alpha_regime(alpha_min, alpha_max)
    regime = AlphaRegime(metrics["regime"])

    if alpha is not None:
        alpha_exec = alpha
    else:
        alpha_exec = select_alpha(alpha_min, alpha_max, caution)

    res = misda_significance(Y, alpha=alpha_exec, ensure_coverage=ensure_coverage, min_coverage=None)

    return LegacyMISDAResult(
        Y=Y,
        caution=caution,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        metrics=metrics,
        regime=regime,
        alpha_exec=alpha_exec,
        isda_res=res,
        name=name
    )


def _analyze_static_v2(
    Y,
    *,
    aggressiveness=1.0,
    rank_policy="default",
    max_evaluated_mis=None,
    seed=123,
    name=None,
    cancel_requested=None,
):
    """Execute the refactored static core through ranking and result assembly."""

    total_start = time.perf_counter()
    normalized = normalize_input_matrix(Y)
    normalized_aggressiveness = validate_aggressiveness(aggressiveness)
    rank_policy = validate_rank_policy(rank_policy)
    max_evaluated_mis = validate_max_evaluated_mis(max_evaluated_mis)

    statistics_start = time.perf_counter()
    correlation_statistics = compute_correlation_statistics(normalized)
    signature = make_structural_signature(
        correlation_statistics,
        rank_policy=rank_policy,
    )
    null_estimate = estimate_null_positive_correlation(
        normalized,
        signature=signature,
        seed=seed,
        cancel_requested=cancel_requested,
    )
    status = separation_status(
        correlation_statistics.log_alpha_onset,
        null_estimate.log_alpha_null,
    )
    if correlation_statistics.log_alpha_onset is None:
        log_alpha = null_estimate.log_alpha_null
    else:
        log_alpha = interpolate_log_alpha(
            correlation_statistics.log_alpha_onset,
            null_estimate.log_alpha_null,
            normalized_aggressiveness,
        )
    statistics_seconds = time.perf_counter() - statistics_start

    graph_start = time.perf_counter()
    structure = build_dependency_graphs(correlation_statistics, log_alpha)
    ranked = rank_structural_mis(
        structure,
        normalized.labels,
        rank_policy=rank_policy,
    )
    graph_seconds = time.perf_counter() - graph_start

    candidates = tuple(
        MISCandidate(
            id=f"mis_{index:03d}",
            objectives=tuple(candidate["mis_labels"]),
            indices=tuple(candidate["mis_indices"]),
            size=int(candidate["size"]),
            rank=int(candidate["rank"]),
            rank_values=dict(candidate["rank_values"]),
        )
        for index, candidate in enumerate(ranked)
    )
    evaluation_start = time.perf_counter()
    if max_evaluated_mis is None:
        n_evaluated = len(candidates)
    else:
        n_evaluated = min(max_evaluated_mis, len(candidates))
    full_front = get_nondominated_mask_minimize(normalized.data)
    for candidate in candidates[:n_evaluated]:
        candidate.evaluation["linear_reconstruction"] = (
            evaluate_linear_reconstruction(
                normalized.data,
                candidate.indices,
                normalized.labels,
            )
        )
        candidate.evaluation["pareto_preservation"] = (
            evaluate_pareto_preservation(
                normalized.data,
                candidate.indices,
                full_front=full_front,
            )
        )
    evaluation_seconds = time.perf_counter() - evaluation_start
    rank_counts = {}
    for candidate in candidates:
        rank_counts[candidate.rank] = rank_counts.get(candidate.rank, 0) + 1

    graph_summaries = {
        "structural": {
            "nodes": structure.structural_graph.number_of_nodes(),
            "edges": structure.structural_graph.number_of_edges(),
            "components": structure.structural_component_count,
        },
        "dependence": {
            "nodes": structure.dependence_graph.number_of_nodes(),
            "edges": structure.dependence_graph.number_of_edges(),
            "components": structure.latent_component_count,
        },
    }
    analysis = AnalysisResult(
        original_dimension=normalized.n_objectives,
        latent_dimension=structure.latent_dimension,
        structural_dimension=structure.structural_dimension,
        alpha_onset=correlation_statistics.alpha_onset,
        log_alpha_onset=correlation_statistics.log_alpha_onset,
        alpha_null=null_estimate.alpha_null,
        log_alpha_null=null_estimate.log_alpha_null,
        alpha=float(np.exp(log_alpha)),
        log_alpha=log_alpha,
        aggressiveness=normalized_aggressiveness,
        separation_status=status,
        structural_graph=structure.structural_graph,
        dependence_graph=structure.dependence_graph,
        structural_components=structure.structural_components,
        latent_components=structure.latent_components,
        graph_summaries=graph_summaries,
        rank_policy=rank_policy,
        rank_counts=rank_counts,
        n_mis=len(candidates),
        n_evaluated_mis=n_evaluated,
        n_heavy_mis=0,
    )
    execution = ExecutionResult(
        configuration={
            "aggressiveness": normalized_aggressiveness,
            "rank_policy": rank_policy,
            "max_evaluated_mis": max_evaluated_mis,
        },
        seed=int(seed),
        timings={
            "statistics": statistics_seconds,
            "graph_and_ranking": graph_seconds,
            "evaluation": evaluation_seconds,
            "total": time.perf_counter() - total_start,
        },
        convergence={
            "alpha_null": {
                "converged": null_estimate.converged,
                "n_permutations": null_estimate.n_permutations,
                "r_null": null_estimate.r_null,
                "se_mc": null_estimate.se_mc,
                "r_interval": null_estimate.r_interval,
                "log_alpha_interval": null_estimate.log_alpha_interval,
                "seed": null_estimate.seed,
                "rng_state": null_estimate.rng_state,
            }
        },
    )
    data = normalized.data
    data.setflags(write=False)
    return MISDAResult(
        analysis=analysis,
        mis=candidates,
        execution=execution,
        name=name,
        _data=data,
    )
