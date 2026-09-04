"""Contracts for reconstruction engines consumed by evaluate()."""

import numpy as np
import pytest

from misda import _reconstruction


@pytest.fixture
def reconstruction_example():
    rng = np.random.default_rng(20260806)
    source = rng.normal(size=120)
    target = 1.75 * source + rng.normal(scale=0.1, size=120)
    noise = rng.normal(size=120)
    return np.column_stack([source, target, noise])


def test_linear_reconstruction_reports_typed_source_metrics(reconstruction_example):
    observed = _reconstruction.evaluate_linear_reconstruction(
        reconstruction_example,
        selected_indices=(0, 2),
        labels=("source", "target", "noise"),
    )

    assert set(observed["r2_by_objective"]) == {"target"}
    assert observed["mean_r2"] is not None
    assert observed["worst_r2"] is not None
    assert observed["mean_r2"] == pytest.approx(observed["worst_r2"])
    assert "jackknife" in observed


def test_linear_reconstruction_no_reduction_is_explicit(reconstruction_example):
    observed = _reconstruction.evaluate_linear_reconstruction(
        reconstruction_example,
        selected_indices=(0, 1, 2),
        labels=("source", "target", "noise"),
    )

    assert observed["r2_by_objective"] is None
    assert observed["mean_r2"] is None
    assert observed["worst_r2"] is None
    assert observed["reason_by_metric"] == {
        "r2_by_objective": "NO_ELIMINATED_OBJECTIVES",
        "mean_r2": "NO_ELIMINATED_OBJECTIVES",
        "worst_r2": "NO_ELIMINATED_OBJECTIVES",
    }


def test_linear_reconstruction_rejects_empty_selection(reconstruction_example):
    with pytest.raises(ValueError, match="selected_indices must not be empty"):
        _reconstruction.evaluate_linear_reconstruction(
            reconstruction_example,
            selected_indices=(),
            labels=("source", "target", "noise"),
        )


def test_linear_reconstruction_rejects_out_of_range_selection(reconstruction_example):
    with pytest.raises(IndexError, match="out-of-range"):
        _reconstruction.evaluate_linear_reconstruction(
            reconstruction_example,
            selected_indices=(3,),
            labels=("source", "target", "noise"),
        )
