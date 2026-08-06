"""Characterization tests for the extracted legacy ranking layer."""

import numpy as np
import pytest

import misda
from misda import _ranking


PUBLIC_RANKING_OPERATIONS = (
    "compute_mis_metrics",
    "sort_mis_metrics",
)


@pytest.fixture
def ranking_example():
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
    return mis_sets, adjacency, labels


@pytest.mark.parametrize("name", PUBLIC_RANKING_OPERATIONS)
def test_legacy_ranking_operations_are_reexported_from_package(name):
    assert getattr(misda, name) is getattr(_ranking, name)


def test_empty_candidate_list_is_unchanged():
    adjacency = np.zeros((3, 3), dtype=int)
    assert misda.compute_mis_metrics([], adjacency, ["a", "b", "c"]) == []
    assert misda.sort_mis_metrics([]) == []


def test_metric_values_are_unchanged(ranking_example):
    observed = misda.compute_mis_metrics(*ranking_example)

    assert observed == [
        {
            "mis_indices": [0, 2, 4],
            "mis_labels": ["z", "m", "q"],
            "size": 3,
            "neighborhood": 2,
            "neighborhood_ratio": 1.0,
            "span": 6,
            "avg_external_degree": 2.0,
            "avg_internal_degree": 0.0,
        },
        {
            "mis_indices": [1, 3],
            "mis_labels": ["a", "b"],
            "size": 2,
            "neighborhood": 3,
            "neighborhood_ratio": 1.0,
            "span": 6,
            "avg_external_degree": 3.0,
            "avg_internal_degree": 0.0,
        },
        {
            "mis_indices": [0, 2],
            "mis_labels": ["z", "m"],
            "size": 2,
            "neighborhood": 2,
            "neighborhood_ratio": 2 / 3,
            "span": 4,
            "avg_external_degree": 2.0,
            "avg_internal_degree": 0.0,
        },
        {
            "mis_indices": [1, 4],
            "mis_labels": ["a", "q"],
            "size": 2,
            "neighborhood": 3,
            "neighborhood_ratio": 1.0,
            "span": 3,
            "avg_external_degree": 1.5,
            "avg_internal_degree": 1.0,
        },
        {
            "mis_indices": [3],
            "mis_labels": ["b"],
            "size": 1,
            "neighborhood": 3,
            "neighborhood_ratio": 0.75,
            "span": 3,
            "avg_external_degree": 3.0,
            "avg_internal_degree": 0.0,
        },
    ]


def test_candidate_indices_are_sorted_before_metrics_are_computed():
    adjacency = np.zeros((3, 3), dtype=int)
    observed = misda.compute_mis_metrics(
        [[2, 0, 1]], adjacency, ["first", "second", "third"]
    )

    assert observed[0]["mis_indices"] == [0, 1, 2]
    assert observed[0]["mis_labels"] == ["first", "second", "third"]


def test_legacy_sort_priority_is_unchanged(ranking_example):
    metrics = misda.compute_mis_metrics(*ranking_example)
    observed = misda.sort_mis_metrics(metrics)

    assert [item["mis_indices"] for item in observed] == [
        [0, 2, 4],
        [1, 3],
        [1, 4],
        [0, 2],
        [3],
    ]


def test_legacy_label_tie_breaker_is_unchanged():
    tied = [
        {
            "size": 2,
            "neighborhood": 3,
            "avg_external_degree": 1.5,
            "span": 3,
            "mis_labels": ["z", "a"],
        },
        {
            "size": 2,
            "neighborhood": 3,
            "avg_external_degree": 1.5,
            "span": 3,
            "mis_labels": ["a", "z"],
        },
    ]

    assert misda.sort_mis_metrics(tied) == [tied[1], tied[0]]
