"""Contracts for the Pareto-preservation engine used by evaluate()."""

import numpy as np
import pytest

from misda import _pareto


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


def test_nondominated_mask_engine_is_stable(pareto_example):
    np.testing.assert_array_equal(
        _pareto.get_nondominated_mask_minimize(pareto_example),
        [True, True, True, True, True],
    )


def test_pareto_preservation_reports_retention_validity_and_jaccard():
    data = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0], [1.5, 1.5]])

    observed = _pareto.evaluate_pareto_preservation(data, [0])

    assert observed["full_front_size"] == 3
    assert observed["reduced_front_size"] == 1
    assert observed["intersection_size"] == 1
    assert observed["pareto_retention"] == pytest.approx(1 / 3)
    assert observed["pareto_validity"] == 1.0
    assert observed["pareto_jaccard"] == pytest.approx(1 / 3)
    assert observed["exact_preservation"] is False
    assert observed["reduced_front_indices"] == (0,)


def test_pareto_preservation_rejects_empty_selection(pareto_example):
    with pytest.raises(ValueError, match="selected_indices must not be empty"):
        _pareto.evaluate_pareto_preservation(pareto_example, [])


def test_pareto_preservation_rejects_mixed_directions_for_now(pareto_example):
    with pytest.raises(ValueError, match="Mixed objective directions are not supported"):
        _pareto.evaluate_pareto_preservation(
            pareto_example,
            [0, 1],
            directions=[-1, 1, -1],
        )
