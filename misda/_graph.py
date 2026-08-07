# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Legacy graph and maximal-independent-set operations used by MISDA."""

from dataclasses import dataclass
from typing import Tuple

import networkx as nx
import numpy as np

from ._ranking import DEFAULT_RANK_FIELDS, rank_mis_candidates
from ._statistics import (
    CorrelationStatistics,
    _correlation_strength,
    correlation_edge_masks,
)


def independence_number(graph: nx.Graph) -> int:
    """Return the exact independence number of ``graph``.

    The value is the maximum cardinality among independent vertex sets.  It is
    computed as the maximum clique size in the complement graph, without using
    MIS ranking or connected-component counts.
    """

    if graph.number_of_nodes() == 0:
        return 0
    complement = nx.complement(graph)
    return max(
        (len(clique) for clique in nx.find_cliques(complement)),
        default=0,
    )


@dataclass(frozen=True)
class GraphStructure:
    """Positive structural and signed dependence projections of one analysis."""

    structural_graph: nx.Graph
    dependence_graph: nx.Graph
    structural_components: Tuple[Tuple[int, ...], ...]
    latent_components: Tuple[Tuple[int, ...], ...]

    @property
    def structural_dimension(self) -> int:
        """Maximum number of mutually non-redundant vertices in ``G+``."""

        return independence_number(self.structural_graph)

    @property
    def latent_dimension(self) -> int:
        """Maximum number of mutually independent vertices in ``G±``."""

        return independence_number(self.dependence_graph)

    @property
    def structural_component_count(self) -> int:
        """Connected-component count of ``G+`` for topology diagnostics."""

        return len(self.structural_components)

    @property
    def latent_component_count(self) -> int:
        """Connected-component count of ``G±`` for topology diagnostics."""

        return len(self.latent_components)


@dataclass(frozen=True)
class StructuralSignature:
    """Output features whose stability terminates null estimation."""

    structural_dimension: int
    ranked_mis: Tuple[tuple, ...]


def _ordered_components(graph):
    components = [tuple(sorted(component)) for component in nx.connected_components(graph)]
    return tuple(sorted(components, key=lambda component: component[0]))


def build_dependency_graphs(
    correlation_statistics: CorrelationStatistics,
    log_alpha: float,
) -> GraphStructure:
    """Build positive ``G+`` and signed ``G±`` from one statistical threshold."""

    positive_mask, signed_mask = correlation_edge_masks(
        correlation_statistics,
        log_alpha,
    )
    structural_graph = nx.Graph()
    dependence_graph = nx.Graph()

    for index, label in enumerate(correlation_statistics.labels):
        attributes = {
            "index": index,
            "name": label,
            "label": label,
            "constant": index in correlation_statistics.constant_indices,
        }
        structural_graph.add_node(index, **attributes)
        dependence_graph.add_node(index, **attributes)

    n_objectives = len(correlation_statistics.labels)
    for left in range(n_objectives):
        for right in range(left + 1, n_objectives):
            if not signed_mask[left, right]:
                continue
            correlation = float(
                correlation_statistics.correlation[left, right]
            )
            attributes = {
                "correlation": correlation,
                "log_p": float(correlation_statistics.log_p[left, right]),
                "sign": 1 if correlation > 0.0 else -1,
            }
            dependence_graph.add_edge(left, right, **attributes)
            if positive_mask[left, right]:
                structural_graph.add_edge(left, right, **attributes)

    return GraphStructure(
        structural_graph=structural_graph,
        dependence_graph=dependence_graph,
        structural_components=_ordered_components(structural_graph),
        latent_components=_ordered_components(dependence_graph),
    )


def enumerate_structural_mis(structure: GraphStructure):
    """Enumerate every maximal independent set of ``G+`` without repair."""

    if not isinstance(structure, GraphStructure):
        raise TypeError("structure must be a GraphStructure instance.")
    n_objectives = structure.structural_graph.number_of_nodes()
    adjacency = nx.to_numpy_array(
        structure.structural_graph,
        nodelist=range(n_objectives),
        dtype=int,
        weight=None,
    )
    observed = find_maximal_independent_sets(adjacency)
    unique = {tuple(sorted(candidate)) for candidate in observed}
    return [list(candidate) for candidate in sorted(unique)]


def rank_structural_mis(
    structure: GraphStructure,
    labels,
    rank_policy="default",
):
    """Enumerate and rank the unmodified MISs of ``G+``."""

    if not isinstance(structure, GraphStructure):
        raise TypeError("structure must be a GraphStructure instance.")
    n_objectives = structure.structural_graph.number_of_nodes()
    if len(labels) != n_objectives:
        raise ValueError("labels must contain one value per graph vertex.")
    adjacency = nx.to_numpy_array(
        structure.structural_graph,
        nodelist=range(n_objectives),
        dtype=int,
        weight=None,
    )
    return rank_mis_candidates(
        enumerate_structural_mis(structure),
        adjacency,
        labels,
        rank_policy=rank_policy,
    )


def structural_signature(
    correlation_statistics: CorrelationStatistics,
    log_alpha: float,
    rank_policy="default",
) -> StructuralSignature:
    """Return the complete structural signature used by sequential estimation."""

    structure = build_dependency_graphs(correlation_statistics, log_alpha)
    ranked = rank_structural_mis(
        structure,
        correlation_statistics.labels,
        rank_policy=rank_policy,
    )
    signature_items = []
    for candidate in ranked:
        rank_values = tuple(
            candidate["rank_values"][field]
            for field in DEFAULT_RANK_FIELDS
        )
        signature_items.append(
            (
                tuple(candidate["mis_indices"]),
                candidate["rank"],
                rank_values,
            )
        )
    return StructuralSignature(
        structural_dimension=structure.structural_dimension,
        ranked_mis=tuple(signature_items),
    )


def make_structural_signature(
    correlation_statistics: CorrelationStatistics,
    rank_policy="default",
):
    """Bind observed correlations into the callback required by null estimation."""

    def signature(log_alpha):
        return structural_signature(
            correlation_statistics,
            log_alpha,
            rank_policy=rank_policy,
        )

    return signature


def find_maximal_independent_sets(adjacency):
    """
    Finds all Maximal Independent Sets (MIS) of a given graph represented by its adjacency matrix.
    Uses the Bron-Kerbosch algorithm.

    Args:
        adjacency (np.ndarray): The adjacency matrix of the graph.

    Returns:
        list: A list of lists, where each inner list represents an MIS (node indices).
    """
    adjacency = np.asarray(adjacency)
    M = adjacency.shape[0]

    # The Bron-Kerbosch algorithm typically works on the complement graph for MIS.
    # An independent set in G is a clique in G_complement.
    comp_adj = np.ones_like(adjacency, dtype=int)
    np.fill_diagonal(comp_adj, 0)
    comp_adj[adjacency == 1] = 0

    mis_list = []

    def neighbors_in_comp(v):
        """Returns neighbors of node v in the complement graph."""
        return {u for u in range(M) if comp_adj[v, u] == 1}

    def bron_kerbosch(R, P, X):
        """Recursive Bron-Kerbosch algorithm to find maximal cliques."""
        if not P and not X:
            mis_list.append(sorted(list(R)))
            return

        # Pivot selection: choose u in P union X with most neighbors in P
        u = None
        max_neighbors = -1
        for v_cand in P.union(X):
            num_neighbors = len(P.intersection(neighbors_in_comp(v_cand)))
            if num_neighbors > max_neighbors:
                max_neighbors = num_neighbors
                u = v_cand

        # Iterate over P \ N(u)
        for v in list(P.difference(neighbors_in_comp(u))):
            N_v = neighbors_in_comp(v)
            bron_kerbosch(R.union({v}), P.intersection(N_v), X.intersection(N_v))
            P.remove(v)
            X.add(v)

    bron_kerbosch(set(), set(range(M)), set())
    return mis_list


def calculate_component_compactness(corr_matrix, components):
    """
    Calculates component homogeneity metrics (Compactness and Ratio) for each connected component.
    Compactness is the minimum absolute correlation within a component.
    Ratio is min_corr / max_corr within a component.

    Args:
        corr_matrix (np.ndarray): The full correlation matrix.
        components (list): A list of lists, where each inner list represents the
                           indices of nodes in a connected component.

    Returns:
        tuple: A tuple containing:
            - float: The lowest internal correlation (min_compactness) across all components.
            - dict: A dictionary mapping component index to its compactness (min internal correlation).
            - dict: A dictionary with 'min_ratio', 'worst_comp_idx', 'ratios', and 'details'.
    """
    metrics = {}
    ratios = {}
    details = {}
    min_compactness = 1.0
    min_ratio = 1.0
    worst_comp_idx = -1

    for idx, comp in enumerate(components):
        if len(comp) < 2:
            metrics[idx] = 1.0
            ratios[idx] = 1.0
            details[idx] = {
                "min_r": 1.0,
                "max_r": 1.0,
                "ratio": 1.0,
            }
            continue

        # Extract submatrix
        sub_corr = corr_matrix[np.ix_(comp, comp)]
        sub_corr_abs = _correlation_strength(sub_corr)

        mask = np.ones_like(sub_corr_abs, dtype=bool)
        np.fill_diagonal(mask, False)
        off_diag = sub_corr_abs[mask]

        if len(off_diag) > 0:
            c_min = float(np.min(off_diag))
            c_max = float(np.max(off_diag))
            ratio = c_min / c_max if c_max > 0 else 0.0
        else:
            c_min, c_max, ratio = 1.0, 1.0, 1.0

        metrics[idx] = c_min
        ratios[idx] = ratio
        details[idx] = {
            "min_r": c_min,
            "max_r": c_max,
            "ratio": ratio,
        }

        if c_min < min_compactness:
            min_compactness = c_min

        if ratio < min_ratio:
            min_ratio = ratio
            worst_comp_idx = idx

    homogeneity_stats = {
        "min_ratio": min_ratio,
        "worst_comp_idx": worst_comp_idx,
        "ratios": ratios,
        "details": details,
    }

    return min_compactness, metrics, homogeneity_stats


def repair_mis_coverage(corr_matrix, mis_indices, min_coverage=0.7):
    """
    Iteratively repairs the MIS to ensure all variables are covered by at least one
    member of the MIS with correlation > min_coverage.

    Args:
        corr_matrix (np.ndarray): MxM correlation matrix.
        mis_indices (list): List of indices currently in the MIS.
        min_coverage (float): Minimum absolute correlation required to consider a variable 'covered'.

    Returns:
        list: List of indices in the repaired (expanded) MIS.
    """
    M = corr_matrix.shape[0]
    current_mis = list(mis_indices)

    # Identify orphans (variables not sufficiently covered by any current MIS member)
    while True:
        orphans = []
        # Calculate max coverage for each variable
        # We look at |Corr(i, m)| for all m in current_mis
        if not current_mis:
            # Should not happen in ISDA context, but handle gracefully
            orphans = list(range(M))
        else:
            mis_cols = corr_matrix[:, current_mis]
            max_corrs = np.max(_correlation_strength(mis_cols), axis=1)  # (M,)

            # Find those below threshold
            orphans = np.where(max_corrs < min_coverage)[0]

        if len(orphans) == 0:
            break

        # Select the best candidate to cover orphans
        # Heuristic: Pick the orphan that is "most central" among the remaining orphans?
        # Or simplify: pick the first orphan?
        # Better: Pick the orphan that covers the most other orphans.

        best_candidate = -1
        best_cover_count = -1

        # Optimization: only check nodes within the orphan set as candidates
        # (Though a non-orphan could arguably cover them too, but non-orphans are already 'represented')
        subset_corr = _correlation_strength(corr_matrix[np.ix_(orphans, orphans)])

        # Count how many orphans each orphan covers
        coverage_counts = np.sum(subset_corr > min_coverage, axis=1)

        best_idx_local = np.argmax(coverage_counts)
        best_candidate = orphans[best_idx_local]

        current_mis.append(best_candidate)

    return sorted(current_mis)
