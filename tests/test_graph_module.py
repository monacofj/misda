"""Characterization tests for the extracted legacy graph layer."""

import numpy as np
import pytest

import misda
from misda import _graph


PUBLIC_GRAPH_OPERATIONS = (
    "find_maximal_independent_sets",
    "calculate_component_compactness",
    "repair_mis_coverage",
)


@pytest.mark.parametrize("name", PUBLIC_GRAPH_OPERATIONS)
def test_legacy_graph_operations_are_reexported_from_package(name):
    assert getattr(misda, name) is getattr(_graph, name)


@pytest.mark.parametrize(
    ("adjacency", "expected"),
    [
        (np.zeros((4, 4), dtype=int), [[0, 1, 2, 3]]),
        (
            np.ones((4, 4), dtype=int) - np.eye(4, dtype=int),
            [[0], [1], [2], [3]],
        ),
        (
            np.array(
                [
                    [0, 1, 0, 0],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                ],
                dtype=int,
            ),
            [[0, 2], [0, 3], [1, 3]],
        ),
        (
            np.array(
                [
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ],
                dtype=int,
            ),
            [[0, 2], [0, 3], [1, 2], [1, 3]],
        ),
    ],
)
def test_maximal_independent_set_enumeration_is_unchanged(adjacency, expected):
    assert misda.find_maximal_independent_sets(adjacency) == expected


def test_component_compactness_values_are_unchanged():
    correlation = np.array(
        [
            [1.0, 0.8, 0.4, 0.0],
            [0.8, 1.0, 0.5, 0.0],
            [0.4, 0.5, 1.0, -0.6],
            [0.0, 0.0, -0.6, 1.0],
        ]
    )

    observed = misda.calculate_component_compactness(
        correlation, [[0, 1, 2], [3]]
    )

    assert observed == (
        0.4,
        {0: 0.4, 1: 1.0},
        {
            "min_ratio": 0.5,
            "worst_comp_idx": 0,
            "ratios": {0: 0.5, 1: 1.0},
            "details": {
                0: {"min_r": 0.4, "max_r": 0.8, "ratio": 0.5},
                1: {"min_r": 1.0, "max_r": 1.0, "ratio": 1.0},
            },
        },
    )


@pytest.mark.parametrize(
    ("initial", "expected"),
    [
        ([], [0, 2, 3]),
        ([0], [0, 2, 3]),
        ([0, 3], [0, 2, 3]),
    ],
)
def test_legacy_coverage_repair_is_unchanged(initial, expected):
    correlation = np.array(
        [
            [1.0, 0.8, 0.4, 0.0],
            [0.8, 1.0, 0.5, 0.0],
            [0.4, 0.5, 1.0, -0.6],
            [0.0, 0.0, -0.6, 1.0],
        ]
    )

    observed = misda.repair_mis_coverage(
        correlation, initial, min_coverage=0.55
    )

    assert observed == expected
