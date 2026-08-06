"""Tests for the MISDA input and argument validation contract."""

import numpy as np
import pandas as pd
import pytest

from misda import _validation


def _valid_array():
    return np.array(
        [
            [0.0, 3.0, 8.0],
            [1.0, 2.0, 8.0],
            [2.0, 1.0, 8.0],
            [3.0, 0.0, 8.0],
        ]
    )


def test_array_and_dataframe_normalize_to_equivalent_values():
    array = _valid_array()
    frame = pd.DataFrame(array, columns=["cost", "time", "constant"])

    array_input = _validation.normalize_input_matrix(array)
    frame_input = _validation.normalize_input_matrix(frame)

    np.testing.assert_array_equal(array_input.data, frame_input.data)
    assert array_input.labels == ("f1", "f2", "f3")
    assert frame_input.labels == ("cost", "time", "constant")
    assert array_input.constant_indices == frame_input.constant_indices == (2,)
    assert frame_input.constant_labels == ("constant",)
    np.testing.assert_array_equal(
        frame_input.constant_mask,
        np.array([False, False, True]),
    )


def test_normalization_returns_an_independent_float_copy():
    source = _valid_array().astype(np.int64)
    normalized = _validation.normalize_input_matrix(source)

    source[0, 0] = 99

    assert normalized.data.dtype == float
    assert normalized.data[0, 0] == 0.0
    assert normalized.n_samples == 4
    assert normalized.n_objectives == 3


def test_legacy_adapter_delegates_to_the_single_normalization_contract():
    data, labels = _validation._validate_input_matrix(_valid_array())

    np.testing.assert_array_equal(data, _valid_array())
    assert labels == ["f1", "f2", "f3"]


@pytest.mark.parametrize(
    "value",
    [
        [[0.0, 1.0], [1.0, 0.0]],
        (0.0, 1.0, 2.0, 3.0),
    ],
)
def test_only_arrays_and_dataframes_are_accepted(value):
    with pytest.raises(
        TypeError,
        match="must be a numpy.ndarray or pandas.DataFrame",
    ):
        _validation.normalize_input_matrix(value)


@pytest.mark.parametrize(
    "value",
    [
        np.arange(4.0),
        np.zeros((4, 2, 1)),
    ],
)
def test_input_must_be_two_dimensional(value):
    with pytest.raises(ValueError, match="two-dimensional matrix"):
        _validation.normalize_input_matrix(value)


@pytest.mark.parametrize(
    "value",
    [
        np.empty((0, 2)),
        np.empty((4, 0)),
    ],
)
def test_input_must_not_be_empty(value):
    with pytest.raises(ValueError, match="must not be empty"):
        _validation.normalize_input_matrix(value)


def test_fisher_z_requires_four_observations_by_default():
    with pytest.raises(ValueError, match="at least four observations"):
        _validation.normalize_input_matrix(np.ones((3, 2)))


def test_small_input_is_allowed_when_fisher_z_is_not_required():
    normalized = _validation.normalize_input_matrix(
        np.ones((3, 2)),
        require_fisher_z=False,
    )

    assert normalized.n_samples == 3
    assert normalized.constant_indices == (0, 1)


@pytest.mark.parametrize(
    "value",
    [
        np.array([["1", "2"]] * 4),
        np.array([[1 + 2j, 3 + 4j]] * 4),
        np.array([[True, False]] * 4),
    ],
)
def test_array_must_contain_real_numeric_values(value):
    with pytest.raises(TypeError, match="only real numeric values"):
        _validation.normalize_input_matrix(value)


def test_dataframe_error_identifies_every_non_numeric_column():
    frame = pd.DataFrame(
        {
            "ok": [1.0, 2.0, 3.0, 4.0],
            "kind": ["a", "b", "c", "d"],
            "flag": [True, False, True, False],
        }
    )

    with pytest.raises(
        TypeError,
        match=r"non-numeric columns: 'kind', 'flag'\.",
    ):
        _validation.normalize_input_matrix(frame)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_nonfinite_error_identifies_columns_in_input_order(nonfinite):
    frame = pd.DataFrame(
        {
            "cost": [0.0, nonfinite, 2.0, 3.0],
            "time": [np.inf, 2.0, 1.0, 0.0],
            "quality": [3.0, 2.0, 1.0, 0.0],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"non-finite values in columns: 'cost', 'time'\.",
    ):
        _validation.normalize_input_matrix(frame)


def test_multiple_and_entirely_constant_matrices_are_recorded_not_rejected():
    partly_constant = _validation.normalize_input_matrix(_valid_array())
    entirely_constant = _validation.normalize_input_matrix(np.ones((4, 3)))

    assert partly_constant.constant_indices == (2,)
    assert entirely_constant.constant_indices == (0, 1, 2)
    assert entirely_constant.constant_labels == ("f1", "f2", "f3")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 0.0), (0.25, 0.25), (1, 1.0), (np.float64(0.5), 0.5)],
)
def test_aggressiveness_accepts_the_closed_unit_interval(value, expected):
    assert _validation.validate_aggressiveness(value) == expected


@pytest.mark.parametrize("value", [-0.01, 1.01, np.nan, np.inf, -np.inf])
def test_aggressiveness_rejects_values_outside_the_closed_interval(value):
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        _validation.validate_aggressiveness(value)


@pytest.mark.parametrize("value", [True, "0.5", None])
def test_aggressiveness_rejects_non_real_values(value):
    with pytest.raises(TypeError, match="must be a real number"):
        _validation.validate_aggressiveness(value)


def test_only_default_rank_policy_is_currently_valid():
    assert _validation.validate_rank_policy("default") == "default"

    with pytest.raises(ValueError, match="must be 'default'"):
        _validation.validate_rank_policy("alternative")
    with pytest.raises(TypeError, match="must be a string"):
        _validation.validate_rank_policy(None)


@pytest.mark.parametrize("value", [None, 1, 5, np.int64(2)])
def test_max_evaluated_mis_accepts_none_or_positive_integers(value):
    expected = None if value is None else int(value)
    assert _validation.validate_max_evaluated_mis(value) == expected


@pytest.mark.parametrize("value", [0, -1])
def test_max_evaluated_mis_rejects_nonpositive_integers(value):
    with pytest.raises(ValueError, match="must be a positive integer"):
        _validation.validate_max_evaluated_mis(value)


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_max_evaluated_mis_rejects_noninteger_values(value):
    with pytest.raises(TypeError, match="None or a positive integer"):
        _validation.validate_max_evaluated_mis(value)


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (2, (2,)),
        (range(1, 4), (1, 2, 3)),
        ([3, 1], (3, 1)),
        ((0, 4), (0, 4)),
        (np.array([2, 0]), (2, 0)),
    ],
)
def test_heavy_selection_normalization(selection, expected):
    assert _validation.normalize_mis_selection(selection, n_mis=5) == expected


@pytest.mark.parametrize("selection", [[], range(0), np.array([])])
def test_heavy_selection_must_not_be_empty(selection):
    with pytest.raises(ValueError, match="at least one MIS index"):
        _validation.normalize_mis_selection(selection, n_mis=5)


def test_heavy_selection_rejects_duplicates_and_invalid_indices():
    with pytest.raises(ValueError, match="duplicate"):
        _validation.normalize_mis_selection([1, 1], n_mis=5)
    with pytest.raises(IndexError, match=r"out of range for 5 MISs: \[-1, 5\]"):
        _validation.normalize_mis_selection([-1, 5], n_mis=5)


@pytest.mark.parametrize("selection", [True, 1.5, "1", {1, 2}])
def test_heavy_selection_rejects_unsupported_forms(selection):
    with pytest.raises(TypeError, match="index, range, or sequence"):
        _validation.normalize_mis_selection(selection, n_mis=5)


def test_heavy_selection_array_must_be_one_dimensional_and_integral():
    with pytest.raises(ValueError, match="one-dimensional"):
        _validation.normalize_mis_selection(np.array([[0, 1]]), n_mis=5)
    with pytest.raises(TypeError, match="indices must be integers"):
        _validation.normalize_mis_selection([0, 1.5], n_mis=5)
