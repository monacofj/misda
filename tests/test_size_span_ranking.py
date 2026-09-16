"""Regression tests for the canonical size-span structural ranking."""

import itertools

import networkx as nx
import numpy as np

import misda.api as api
from misda._ranking import compute_mis_metrics


def _maximal_independent_sets(graph):
    nodes = tuple(graph.nodes)
    result = []
    for size in range(1, len(nodes) + 1):
        for candidate in itertools.combinations(nodes, size):
            selected = set(candidate)
            if any(graph.has_edge(u, v) for u, v in itertools.combinations(candidate, 2)):
                continue
            if any(
                all(not graph.has_edge(vertex, member) for member in selected)
                for vertex in set(nodes) - selected
            ):
                continue
            result.append(tuple(sorted(candidate)))
    return result


def _old_redundant_key(metric):
    return (
        -metric["size"],
        -metric["neighborhood"],
        -metric["avg_external_degree"],
        -metric["span"],
        tuple(repr(label) for label in metric["mis_labels"]),
    )


def test_size_span_key_matches_former_redundant_key_on_actual_mis_candidates():
    graph = nx.Graph()
    graph.add_nodes_from(range(7))
    graph.add_edges_from(
        [
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (2, 4),
            (3, 4),
            (3, 5),
            (4, 6),
            (5, 6),
        ]
    )
    adjacency = nx.to_numpy_array(graph, nodelist=range(7), dtype=int, weight=None)
    labels = tuple(f"f{i + 1}" for i in range(7))
    metrics = compute_mis_metrics(_maximal_independent_sets(graph), adjacency, labels)

    old_order = [item["mis_indices"] for item in sorted(metrics, key=_old_redundant_key)]
    new_order = [item["mis_indices"] for item in sorted(metrics, key=api._structural_sort_key)]

    assert new_order == old_order


def test_maximality_makes_neighborhood_redundant_with_size():
    graph = nx.path_graph(6)
    adjacency = nx.to_numpy_array(graph, nodelist=range(6), dtype=int, weight=None)
    labels = tuple(f"f{i + 1}" for i in range(6))
    metrics = compute_mis_metrics(_maximal_independent_sets(graph), adjacency, labels)

    for metric in metrics:
        assert metric["neighborhood"] == 6 - metric["size"]
        if metric["size"]:
            assert np.isclose(
                metric["avg_external_degree"],
                metric["span"] / metric["size"],
            )


def test_scientific_ties_depend_only_on_size_and_span():
    first = {
        "size": 3,
        "span": 7,
        "mis_labels": ["a", "c", "e"],
    }
    second = {
        "size": 3,
        "span": 7,
        "mis_labels": ["b", "d", "f"],
    }

    assert api._structural_rank_value(first) == api._structural_rank_value(second)
    assert api._structural_sort_key(first) != api._structural_sort_key(second)
