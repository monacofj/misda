# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Graph construction and maximal-independent-set operations for MISDA."""

from dataclasses import dataclass
from typing import Tuple

import networkx as nx
import numpy as np

from ._statistics import CorrelationStatistics, correlation_edge_masks


def independence_number(graph: nx.Graph) -> int:
    """Return the exact independence number of ``graph``."""

    if graph.number_of_nodes() == 0:
        return 0
    complement = nx.complement(graph)
    return max((len(clique) for clique in nx.find_cliques(complement)), default=0)


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
        return len(self.structural_components)

    @property
    def latent_component_count(self) -> int:
        return len(self.latent_components)


def _ordered_components(graph):
    components = [
        tuple(sorted(component)) for component in nx.connected_components(graph)
    ]
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
            correlation = float(correlation_statistics.correlation[left, right])
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


def find_maximal_independent_sets(adjacency):
    """Enumerate every maximal independent set of an adjacency matrix."""

    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix.")
    n_vertices = adjacency.shape[0]

    complement = np.ones_like(adjacency, dtype=int)
    np.fill_diagonal(complement, 0)
    complement[adjacency == 1] = 0

    maximal_sets = []

    def neighbors(vertex):
        return {
            other
            for other in range(n_vertices)
            if complement[vertex, other] == 1
        }

    def bron_kerbosch(current, prospective, excluded):
        if not prospective and not excluded:
            maximal_sets.append(sorted(current))
            return

        pivot = None
        pivot_score = -1
        for candidate in prospective | excluded:
            score = len(prospective & neighbors(candidate))
            if score > pivot_score:
                pivot = candidate
                pivot_score = score

        pivot_neighbors = neighbors(pivot) if pivot is not None else set()
        for vertex in list(prospective - pivot_neighbors):
            vertex_neighbors = neighbors(vertex)
            bron_kerbosch(
                current | {vertex},
                prospective & vertex_neighbors,
                excluded & vertex_neighbors,
            )
            prospective.remove(vertex)
            excluded.add(vertex)

    bron_kerbosch(set(), set(range(n_vertices)), set())
    return maximal_sets


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
