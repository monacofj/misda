"""Tests for structural candidate metrics used by ranking policies."""

import numpy as np

from misda._ranking import compute_mis_metrics


def test_structural_metric_values():
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
    observed = compute_mis_metrics(
        [[0, 2, 4], [1, 3]],
        adjacency,
        ["z", "a", "m", "b", "q"],
    )

    assert observed[0] == {
        "mis_indices": [0, 2, 4],
        "mis_labels": ["z", "m", "q"],
        "size": 3,
        "neighborhood": 2,
        "neighborhood_ratio": 1.0,
        "span": 6,
        "avg_external_degree": 2.0,
        "avg_internal_degree": 0.0,
    }
    assert observed[1]["size"] == 2
    assert observed[1]["neighborhood"] == 3
    assert observed[1]["span"] == 6


def test_candidate_indices_are_canonicalized_before_measurement():
    adjacency = np.zeros((3, 3), dtype=int)
    observed = compute_mis_metrics(
        [[2, 0, 1]], adjacency, ["first", "second", "third"]
    )

    assert observed[0]["mis_indices"] == [0, 1, 2]
    assert observed[0]["mis_labels"] == ["first", "second", "third"]


def test_empty_candidate_universe_has_no_metrics():
    assert compute_mis_metrics([], np.zeros((3, 3), dtype=int), ["a", "b", "c"]) == []
