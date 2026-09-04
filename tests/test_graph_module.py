"""Tests for the graph layer used by static discovery."""

import networkx as nx
import numpy as np
import pytest

from misda._graph import find_maximal_independent_sets, independence_number


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
def test_maximal_independent_set_enumeration(adjacency, expected):
    assert find_maximal_independent_sets(adjacency) == expected


def test_independence_number_is_not_component_count_for_chain():
    graph = nx.path_graph(5)

    assert nx.number_connected_components(graph) == 1
    assert independence_number(graph) == 3


def test_independence_number_extremes():
    assert independence_number(nx.empty_graph(5)) == 5
    assert independence_number(nx.complete_graph(5)) == 1


def test_mis_input_must_be_square():
    with pytest.raises(ValueError, match="square"):
        find_maximal_independent_sets(np.zeros((2, 3), dtype=int))
