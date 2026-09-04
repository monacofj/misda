"""Tests for signed graphs, dimensions, structural MISs, and discovery signature."""

import numpy as np
import pytest

import misda.newapi as newapi
from misda import _graph, _statistics


def _statistics_from_relations(relations, n_objectives, labels=None):
    correlation = np.zeros((n_objectives, n_objectives), dtype=float)
    np.fill_diagonal(correlation, 1.0)
    log_p = np.full((n_objectives, n_objectives), -1.0, dtype=float)
    np.fill_diagonal(log_p, np.nan)
    valid_pairs = np.ones((n_objectives, n_objectives), dtype=bool)
    np.fill_diagonal(valid_pairs, False)

    for left, right, sign in relations:
        correlation[left, right] = correlation[right, left] = 0.8 * sign
        log_p[left, right] = log_p[right, left] = -10.0

    return _statistics.CorrelationStatistics(
        correlation=correlation,
        log_p=log_p,
        valid_pairs=valid_pairs,
        labels=tuple(labels or [f"f{index + 1}" for index in range(n_objectives)]),
        constant_indices=(),
        n_samples=30,
        log_alpha_onset=-10.0 if any(sign > 0 for _, _, sign in relations) else None,
    )


def _edges(graph):
    return {tuple(sorted(edge)) for edge in graph.edges}


def test_empty_graph_keeps_all_objectives_in_one_mis():
    structure = _graph.build_dependency_graphs(
        _statistics_from_relations([], 4), log_alpha=-5.0
    )

    assert structure.structural_dimension == 4
    assert structure.latent_dimension == 4
    assert structure.structural_components == ((0,), (1,), (2,), (3,))
    assert _graph.enumerate_structural_mis(structure) == [[0, 1, 2, 3]]


def test_complete_positive_graph_has_singleton_maximal_sets():
    relations = [
        (left, right, 1)
        for left in range(4)
        for right in range(left + 1, 4)
    ]
    structure = _graph.build_dependency_graphs(
        _statistics_from_relations(relations, 4), log_alpha=-5.0
    )

    assert structure.structural_dimension == 1
    assert structure.latent_dimension == 1
    assert structure.structural_component_count == 1
    assert structure.latent_component_count == 1
    assert _graph.enumerate_structural_mis(structure) == [[0], [1], [2], [3]]


def test_signed_edges_join_latent_but_not_structural_redundancy():
    statistics = _statistics_from_relations(
        [(0, 1, 1), (2, 3, 1), (1, 2, -1)],
        4,
        labels=["cost-a", "cost-b", "gain-a", "gain-b"],
    )
    structure = _graph.build_dependency_graphs(statistics, log_alpha=-5.0)

    assert _edges(structure.structural_graph) == {(0, 1), (2, 3)}
    assert _edges(structure.dependence_graph) == {(0, 1), (1, 2), (2, 3)}
    assert structure.structural_components == ((0, 1), (2, 3))
    assert structure.latent_components == ((0, 1, 2, 3),)
    assert structure.structural_dimension == 2
    assert structure.latent_dimension == 2
    assert structure.dependence_graph.edges[1, 2]["sign"] == -1


def test_positive_and_negative_pair_have_different_structural_dimensions():
    positive = _graph.build_dependency_graphs(
        _statistics_from_relations([(0, 1, 1)], 2), log_alpha=-5.0
    )
    negative = _graph.build_dependency_graphs(
        _statistics_from_relations([(0, 1, -1)], 2), log_alpha=-5.0
    )

    assert positive.latent_dimension == positive.structural_dimension == 1
    assert negative.latent_dimension == 1
    assert negative.structural_dimension == 2


def test_path_dimension_is_independence_number_not_component_count():
    relations = [(index, index + 1, 1) for index in range(19)]
    statistics = _statistics_from_relations(relations, 20)
    structure = _graph.build_dependency_graphs(statistics, log_alpha=-5.0)
    ranked, groups = newapi._rank_structural_coverage(
        structure, statistics.labels
    )

    assert structure.structural_component_count == 1
    assert structure.structural_dimension == 10
    assert max(len(candidate) for candidate in _graph.enumerate_structural_mis(structure)) == 10
    assert ranked[0]["size"] == 10
    assert groups[0]


def test_two_anticorrelated_positive_cliques_have_structural_two_latent_one():
    first = range(10)
    second = range(10, 20)
    relations = []
    relations.extend(
        (left, right, 1)
        for group in (first, second)
        for left in group
        for right in group
        if left < right
    )
    relations.extend((left, right, -1) for left in first for right in second)
    statistics = _statistics_from_relations(relations, 20)
    structure = _graph.build_dependency_graphs(statistics, log_alpha=-5.0)
    ranked, _ = newapi._rank_structural_coverage(structure, statistics.labels)

    assert structure.structural_dimension == 2
    assert structure.latent_dimension == 1
    assert structure.structural_component_count == 2
    assert structure.latent_component_count == 1
    assert len(ranked) == 100
    assert all(candidate["size"] == 2 for candidate in ranked)
    assert all(
        len(set(candidate["mis_indices"]) & set(first)) == 1
        and len(set(candidate["mis_indices"]) & set(second)) == 1
        for candidate in ranked
    )


def test_constant_objectives_remain_isolated_vertices():
    statistics = _statistics_from_relations([(0, 1, 1)], 3)
    correlation = statistics.correlation.copy()
    log_p = statistics.log_p.copy()
    valid_pairs = statistics.valid_pairs.copy()
    correlation[2, :] = correlation[:, 2] = np.nan
    log_p[2, :] = log_p[:, 2] = np.nan
    valid_pairs[2, :] = valid_pairs[:, 2] = False
    statistics = _statistics.CorrelationStatistics(
        correlation=correlation,
        log_p=log_p,
        valid_pairs=valid_pairs,
        labels=statistics.labels,
        constant_indices=(2,),
        n_samples=30,
        log_alpha_onset=-10.0,
    )

    structure = _graph.build_dependency_graphs(statistics, log_alpha=-5.0)

    assert structure.structural_components == ((0, 1), (2,))
    assert structure.latent_components == ((0, 1), (2,))
    assert structure.structural_dimension == 2
    assert structure.latent_dimension == 2
    assert structure.structural_graph.nodes[2]["constant"]
    assert structure.structural_graph.degree[2] == 0
    assert structure.dependence_graph.degree[2] == 0


@pytest.mark.parametrize(
    "relations",
    [
        [],
        [(0, 1, 1), (1, 2, 1), (2, 3, 1)],
        [(0, 1, 1), (2, 3, 1)],
        [
            (left, right, 1)
            for left in range(4)
            for right in range(left + 1, 4)
        ],
    ],
)
def test_enumerated_sets_are_unique_maximal_and_deterministic(relations):
    statistics = _statistics_from_relations(relations, 4)
    structure = _graph.build_dependency_graphs(statistics, log_alpha=-5.0)
    first = _graph.enumerate_structural_mis(structure)
    second = _graph.enumerate_structural_mis(structure)

    assert first == second
    assert len(first) == len({tuple(candidate) for candidate in first})
    for candidate in first:
        selected = set(candidate)
        assert all(
            not structure.structural_graph.has_edge(left, right)
            for left in selected
            for right in selected
            if left < right
        )
        assert all(
            any(
                structure.structural_graph.has_edge(vertex, chosen)
                for chosen in selected
            )
            for vertex in set(structure.structural_graph) - selected
        )


def test_structural_coverage_groups_equal_scientific_values_and_uses_labels_only_inside_tie(monkeypatch):
    class Structure:
        structural_graph = __import__("networkx").empty_graph(4)

    measured = [
        {
            "mis_indices": [0, 1], "mis_labels": ["z", "y"], "size": 2,
            "neighborhood": 0, "neighborhood_ratio": 0.0,
            "avg_external_degree": 0.0, "avg_internal_degree": 0.0, "span": 0,
        },
        {
            "mis_indices": [2, 3], "mis_labels": ["a", "b"], "size": 2,
            "neighborhood": 0, "neighborhood_ratio": 0.0,
            "avg_external_degree": 0.0, "avg_internal_degree": 0.0, "span": 0,
        },
    ]
    monkeypatch.setattr(newapi, "enumerate_structural_mis", lambda structure: [[0, 1], [2, 3]])
    monkeypatch.setattr(newapi, "compute_mis_metrics", lambda *args: measured)

    ranked, groups = newapi._rank_structural_coverage(
        Structure(), ["z", "y", "a", "b"]
    )

    assert [item["mis_indices"] for item in ranked] == [[2, 3], [0, 1]]
    assert groups == ((0, 1),)


def test_discovery_signature_contains_ds_dl_and_tie_groups(monkeypatch):
    class FakeStructure:
        structural_dimension = 3
        latent_dimension = 2

    ranked = [
        {"mis_indices": [0, 2]},
        {"mis_indices": [1, 3]},
        {"mis_indices": [0, 3]},
    ]
    monkeypatch.setattr(newapi, "build_dependency_graphs", lambda *args: FakeStructure())
    monkeypatch.setattr(
        newapi,
        "_rank_structural_coverage",
        lambda *args: (ranked, ((0, 1), (2,))),
    )

    observed = newapi._discovery_signature(
        _statistics_from_relations([], 4), -5.0
    )

    assert observed == (3, 2, (((0, 2), (1, 3)), ((0, 3),)))


def test_discovery_signature_ignores_order_inside_true_tie(monkeypatch):
    class FakeStructure:
        structural_dimension = 2
        latent_dimension = 2

    monkeypatch.setattr(newapi, "build_dependency_graphs", lambda *args: FakeStructure())
    statistics = _statistics_from_relations([], 4)

    first = [
        {"mis_indices": [0, 2]},
        {"mis_indices": [1, 3]},
    ]
    monkeypatch.setattr(
        newapi,
        "_rank_structural_coverage",
        lambda *args: (first, ((0, 1),)),
    )
    signature_a = newapi._discovery_signature(statistics, -5.0)

    second = list(reversed(first))
    monkeypatch.setattr(
        newapi,
        "_rank_structural_coverage",
        lambda *args: (second, ((0, 1),)),
    )
    signature_b = newapi._discovery_signature(statistics, -5.0)

    assert signature_a == signature_b
