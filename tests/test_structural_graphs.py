"""Tests for signed graphs, structural dimensions, MISs, and ranking."""

import numpy as np
import pytest

from misda import _graph, _ranking, _statistics


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


def _positive_edges(graph):
    return {tuple(sorted(edge)) for edge in graph.edges}


def test_empty_graph_keeps_all_objectives_in_one_mis():
    statistics = _statistics_from_relations([], 4)
    structure = _graph.build_dependency_graphs(statistics, log_alpha=-5.0)

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
    statistics = _statistics_from_relations(relations, 4)
    structure = _graph.build_dependency_graphs(statistics, log_alpha=-5.0)

    assert structure.structural_dimension == 1
    assert structure.latent_dimension == 1
    assert _graph.enumerate_structural_mis(structure) == [[0], [1], [2], [3]]


def test_signed_edges_join_latent_but_not_structural_components():
    statistics = _statistics_from_relations(
        [(0, 1, 1), (2, 3, 1), (1, 2, -1)],
        4,
        labels=["cost-a", "cost-b", "gain-a", "gain-b"],
    )
    structure = _graph.build_dependency_graphs(statistics, log_alpha=-5.0)

    assert _positive_edges(structure.structural_graph) == {(0, 1), (2, 3)}
    assert _positive_edges(structure.dependence_graph) == {
        (0, 1),
        (1, 2),
        (2, 3),
    }
    assert structure.structural_components == ((0, 1), (2, 3))
    assert structure.latent_components == ((0, 1, 2, 3),)
    assert structure.structural_dimension == 2
    assert structure.latent_dimension == 1
    assert structure.dependence_graph.edges[1, 2]["sign"] == -1
    assert structure.structural_graph.nodes[0]["name"] == "cost-a"
    assert structure.structural_graph.edges[0, 1]["correlation"] == 0.8
    assert structure.structural_graph.edges[0, 1]["log_p"] == -10.0


def test_case_5_path_has_one_component_and_maximum_mis_size_ten():
    relations = [(index, index + 1, 1) for index in range(19)]
    statistics = _statistics_from_relations(relations, 20)
    structure = _graph.build_dependency_graphs(statistics, log_alpha=-5.0)
    ranked = _graph.rank_structural_mis(structure, statistics.labels)

    assert structure.structural_dimension == 1
    assert max(len(candidate) for candidate in _graph.enumerate_structural_mis(structure)) == 10
    assert len(ranked[0]["mis_indices"]) == 10
    assert ranked[0]["mis_indices"] != list(range(20))


def test_case_7_two_anticorrelated_groups_has_dimensions_two_and_one():
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
    ranked = _graph.rank_structural_mis(structure, statistics.labels)

    assert structure.structural_dimension == 2
    assert structure.latent_dimension == 1
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
            any(structure.structural_graph.has_edge(vertex, chosen) for chosen in selected)
            for vertex in set(structure.structural_graph) - selected
        )


def test_default_policy_preserves_legacy_effective_order_and_exposes_values():
    adjacency = np.array(
        [
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
        ],
        dtype=int,
    )
    mis_sets = [[0, 2, 4], [1, 3], [0, 2], [1, 4], [3]]
    labels = ["z", "a", "m", "b", "q"]

    ranked = _ranking.rank_mis_candidates(mis_sets, adjacency, labels)

    assert [candidate["mis_indices"] for candidate in ranked] == [
        [0, 2, 4],
        [1, 3],
        [1, 4],
        [0, 2],
        [3],
    ]
    assert ranked[0]["rank_values"] == {
        "size": 3,
        "neighborhood": 2,
        "avg_external_degree": 2.0,
        "span": 6,
    }
    assert [candidate["rank"] for candidate in ranked] == [1, 2, 3, 4, 5]
    assert all("total_correlation" not in candidate for candidate in ranked)
    assert all("max_correlation" not in candidate for candidate in ranked)


def test_equal_rank_values_share_rank_and_labels_only_break_ties():
    adjacency = np.zeros((4, 4), dtype=int)
    labels = ["z", "y", "a", "b"]
    ranked = _ranking.rank_mis_candidates(
        [[0, 1], [2, 3]],
        adjacency,
        labels,
    )

    assert [candidate["mis_indices"] for candidate in ranked] == [[2, 3], [0, 1]]
    assert [candidate["rank"] for candidate in ranked] == [1, 1]
    assert ranked[0]["rank_values"] == ranked[1]["rank_values"]


def test_structural_signature_contains_dimension_order_ranks_and_values():
    statistics = _statistics_from_relations(
        [(0, 1, 1), (1, 2, 1), (2, 3, 1)],
        4,
    )
    observed = _graph.structural_signature(statistics, log_alpha=-5.0)

    assert observed.structural_dimension == 1
    assert observed.ranked_mis
    assert all(len(item) == 3 for item in observed.ranked_mis)
    assert observed == _graph.make_structural_signature(statistics)(-5.0)


def test_null_estimator_accepts_complete_structural_signature():
    statistics = _statistics_from_relations([(0, 1, 1)], 3)
    normalized_data = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 3.0],
            [3.0, 3.0, 2.0],
        ]
    )
    from misda import _validation

    normalized = _validation.normalize_input_matrix(normalized_data)
    observed = _statistics.estimate_null_from_maxima(
        [0.2, 0.2, 0.2, 0.2],
        n_samples=normalized.n_samples,
        signature=_graph.make_structural_signature(statistics),
    )

    assert observed.converged
    assert isinstance(observed.lower_r_signature, _graph.StructuralSignature)
    assert observed.lower_r_signature == observed.upper_r_signature
