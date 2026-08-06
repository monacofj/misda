"""Characterization tests for the extracted legacy input validation."""

import numpy as np
import pandas as pd
import pytest

import misda
from misda import _validation


def test_legacy_input_validation_is_reexported_from_package():
    assert misda._validate_input_matrix is _validation._validate_input_matrix


def test_array_validation_values_and_generated_labels_are_unchanged():
    data = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])

    observed, labels = misda._validate_input_matrix(data)

    np.testing.assert_array_equal(observed, data)
    assert labels == ["f1", "f2"]


def test_dataframe_validation_preserves_labels():
    frame = pd.DataFrame(
        [[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]],
        columns=["cost", "time"],
    )

    observed, labels = misda._validate_input_matrix(frame)

    np.testing.assert_array_equal(observed, frame.values)
    assert labels == ["cost", "time"]


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_nonfinite_error_is_unchanged(nonfinite):
    data = np.array([[0.0, 1.0], [nonfinite, 2.0]])

    with pytest.raises(ValueError, match="contains non-finite values"):
        misda._validate_input_matrix(data)


def test_constant_objective_error_is_unchanged():
    data = np.array([[1.0, 4.0], [1.0, 5.0]])

    with pytest.raises(
        ValueError,
        match=r"Objective 'f1' \(column index 0\) has zero range",
    ):
        misda._validate_input_matrix(data)
