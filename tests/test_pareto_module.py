"""Characterization tests for the extracted legacy Pareto layer."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import misda
from misda import _pareto


PUBLIC_PARETO_OPERATIONS = (
    "get_nondominated_mask",
    "evaluate_pareto_consistency",
    "evaluate_pareto_raw",
)


@pytest.fixture
def pareto_example():
    return np.array(
        [
            [0.0, 3.0, 2.0],
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 3.0],
            [1.5, 1.5, 1.5],
        ]
    )


@pytest.mark.parametrize("name", PUBLIC_PARETO_OPERATIONS)
def test_legacy_pareto_operations_are_reexported_from_package(name):
    assert getattr(misda, name) is getattr(_pareto, name)


def test_nondominated_mask_is_unchanged(pareto_example):
    np.testing.assert_array_equal(
        misda.get_nondominated_mask(pareto_example),
        [True, True, True, True, True],
    )


@pytest.mark.parametrize(
    ("selected", "expected"),
    [
        ([0, 1, 2], (1.0, 1.0)),
        ([0, 1], (1.0, 1.0)),
        ([0], (1.0, 0.2)),
        ([], (0.0, 0.0)),
    ],
)
def test_raw_pareto_values_are_unchanged(
    pareto_example, selected, expected
):
    assert misda.evaluate_pareto_raw(pareto_example, selected) == expected


def test_raw_pareto_dataframe_and_directions_are_unchanged(pareto_example):
    frame = pd.DataFrame(pareto_example, columns=["a", "b", "c"])

    assert misda.evaluate_pareto_raw(
        frame,
        [0, 1],
        directions=[-1, 1, -1],
    ) == (1.0, 1 / 3)


def test_consistency_empty_selection_behavior_is_unchanged(pareto_example):
    result = SimpleNamespace(
        Y=pareto_example,
        best_mis=SimpleNamespace(indices=[]),
    )

    assert misda.evaluate_pareto_consistency(result) == (0.0, 0.0)


def test_consistency_undefined_recall_bug_is_preserved(pareto_example):
    result = SimpleNamespace(
        Y=pareto_example,
        best_mis=SimpleNamespace(indices=[0]),
    )

    with pytest.raises(NameError, match="recall"):
        misda.evaluate_pareto_consistency(result)
