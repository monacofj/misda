"""Tests for on-demand nonlinear and null-reference evaluation."""

import numpy as np
import pytest

import misda
from misda import _reconstruction


def _four_mis_result():
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    data = np.column_stack([x, 2.0 * x, -x, -2.0 * x])
    return misda.analyze(data, max_evaluated_mis=1, seed=8)


def _fake_nonlinear_result(value=0.75, *, cancelled=False):
    return {
        "r2_by_objective": {"target": value},
        "r2_reason_by_objective": {},
        "mean_r2": value,
        "worst_r2": value,
        "reason_by_metric": {},
        "jackknife": {
            "r2_se_by_objective": {"target": 0.1},
            "mean_r2_se": 0.1,
            "worst_r2_se": 0.1,
            "n_replicates": 6,
            "reason": None,
        },
        "tree_se_by_objective": {"target": 0.05},
        "n_trees": 6,
        "configuration_counts": {
            "max_features=1,min_samples_leaf=1": 6
        },
        "converged": not cancelled,
        "cancelled": cancelled,
    }


def test_heavy_complements_selected_candidates_in_place():
    result = _four_mis_result()
    light = dict(result.mis[0].evaluation)
    calls = []

    def evaluator(data, selection, labels, *, seed, cancel_requested):
        calls.append((selection, labels, seed))
        return _fake_nonlinear_result()

    observed = misda.heavy(result, [0, 2], _evaluator=evaluator)

    assert observed is result
    assert result.mis[0].evaluation["linear_reconstruction"] == light[
        "linear_reconstruction"
    ]
    assert result.mis[0].evaluation["pareto_preservation"] == light[
        "pareto_preservation"
    ]
    assert "nonlinear_reconstruction" in result.mis[0].evaluation
    assert "nonlinear_reconstruction" in result.mis[2].evaluation
    assert "nonlinear_reconstruction" not in result.mis[1].evaluation
    assert len(calls) == 2
    assert result.analysis.n_heavy_mis == 2
    assert result.execution.timings["heavy"] >= 0.0


def test_heavy_is_idempotent_and_accepts_ranges():
    result = _four_mis_result()
    calls = 0

    def evaluator(data, selection, labels, *, seed, cancel_requested):
        nonlocal calls
        calls += 1
        return _fake_nonlinear_result()

    misda.heavy(result, range(1, 3), _evaluator=evaluator)
    misda.heavy(result, [1, 2], _evaluator=evaluator)

    assert calls == 2
    assert result.analysis.n_heavy_mis == 2


def test_heavy_stops_after_an_explicitly_cancelled_candidate():
    result = _four_mis_result()
    calls = 0

    def evaluator(data, selection, labels, *, seed, cancel_requested):
        nonlocal calls
        calls += 1
        return _fake_nonlinear_result(cancelled=True)

    misda.heavy(result, [0, 1, 2], _evaluator=evaluator)

    assert calls == 1
    assert result.analysis.n_heavy_mis == 1
    assert "nonlinear_reconstruction" not in result.mis[1].evaluation


@pytest.mark.parametrize("selection", [[], [0, 0], [-1], [99], 1.5])
def test_heavy_validates_selection(selection):
    result = _four_mis_result()
    with pytest.raises((TypeError, ValueError, IndexError)):
        misda.heavy(result, selection, _evaluator=lambda *args, **kwargs: {})


def test_heavy_rejects_legacy_results_and_invalid_controls():
    data = np.arange(24.0).reshape(6, 4)
    legacy = misda._analyze_static(data)

    with pytest.raises(TypeError, match="refactored MISDAResult"):
        misda.heavy(legacy, 0)
    with pytest.raises(TypeError, match="null_reference"):
        misda.heavy(_four_mis_result(), 0, null_reference=1)
    with pytest.raises(TypeError, match="cancel_requested"):
        misda.heavy(_four_mis_result(), 0, cancel_requested=True)


def test_nonlinear_no_reduction_is_explicitly_undefined():
    data = np.arange(18.0).reshape(6, 3)
    observed = _reconstruction.evaluate_nonlinear_reconstruction(
        data,
        selected_indices=[0, 1, 2],
        labels=("a", "b", "c"),
    )

    assert observed["r2_by_objective"] is None
    assert observed["mean_r2"] is None
    assert observed["worst_r2"] is None
    assert observed["reason_by_metric"] == {
        "r2_by_objective": "NO_ELIMINATED_OBJECTIVES",
        "mean_r2": "NO_ELIMINATED_OBJECTIVES",
        "worst_r2": "NO_ELIMINATED_OBJECTIVES",
    }
    assert observed["n_trees"] == 0
    assert observed["converged"]


def test_nonlinear_cancellation_before_work_has_its_own_reason():
    data = np.column_stack([np.arange(5.0), np.arange(5.0) ** 2])
    observed = _reconstruction.evaluate_nonlinear_reconstruction(
        data,
        selected_indices=[0],
        labels=("x", "square"),
        cancel_requested=lambda: True,
    )

    assert observed["mean_r2"] is None
    assert observed["reason_by_metric"]["mean_r2"] == (
        "CANCELLED_BEFORE_EVALUATION"
    )
    assert observed["jackknife"]["reason"] == "CANCELLED_BEFORE_EVALUATION"
    assert observed["convergence_reason"] == "CANCELLED_BEFORE_EVALUATION"
    assert not observed["converged"]
    assert observed["cancelled"]


@pytest.mark.slow
def test_complete_nonlinear_protocol_is_reproducible_on_a_small_case():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    data = np.column_stack([x, x**2])
    kwargs = {
        "selected_indices": [0],
        "labels": ("x", "square"),
        "seed": 11,
    }

    first = _reconstruction.evaluate_nonlinear_reconstruction(data, **kwargs)
    second = _reconstruction.evaluate_nonlinear_reconstruction(data, **kwargs)

    assert first == second
    assert first["n_trees"] >= data.shape[0]
    assert first["converged"]
    assert not first["cancelled"]
    assert first["configuration_by_outer_fold"]
    assert first["r2_by_objective"]["square"] < 0.0


@pytest.mark.slow
def test_undefined_sample_uncertainty_does_not_claim_tree_convergence():
    data = np.column_stack(
        [
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        ]
    )
    observed = _reconstruction.evaluate_nonlinear_reconstruction(
        data,
        selected_indices=[0],
        labels=("x", "rare"),
        seed=12,
    )

    assert observed["r2_by_objective"]["rare"] is not None
    assert observed["jackknife"]["r2_se_by_objective"]["rare"] is None
    assert not observed["converged"]
    assert not observed["cancelled"]
    assert observed["convergence_reason"] == "SAMPLE_UNCERTAINTY_UNDEFINED"


def test_inner_model_selection_breaks_error_ties_toward_simplicity():
    class MeanModel:
        def __init__(self, **configuration):
            self.configuration = configuration

        def fit(self, predictors, target):
            self.mean = float(np.mean(target))
            return self

        def predict(self, predictors):
            return np.full(predictors.shape[0], self.mean)

    predictors = np.column_stack([np.arange(5.0), np.arange(5.0)])
    target = np.arange(5.0)
    observed = _reconstruction._select_rf_configuration(
        predictors,
        target,
        n_trees=5,
        seed=4,
        model_factory=MeanModel,
    )

    assert observed["max_features"] == 1
    assert observed["min_samples_leaf"] == 2


def test_tree_stopping_compares_every_defined_target():
    assert _reconstruction._tree_stopping_reached(
        {"a": 0.1, "b": 0.2},
        {"a": 0.1, "b": 0.3},
    )
    assert not _reconstruction._tree_stopping_reached(
        {"a": 0.1, "b": 0.4},
        {"a": 0.1, "b": 0.3},
    )
    assert _reconstruction._tree_stopping_reached(
        {"constant": None},
        {"constant": None},
    )


def test_null_reference_starts_at_n_and_uses_observed_uncertainty():
    values = iter([-0.2, -0.1, 0.0, 0.1])

    def evaluator(data, selection, labels, *, seed, cancel_requested):
        value = next(values)
        result = _fake_nonlinear_result(value)
        result["jackknife"]["mean_r2_se"] = 0.5
        return result

    data = np.column_stack([np.arange(4.0), np.arange(4.0) ** 2])
    observed = _fake_nonlinear_result(0.5)
    observed["jackknife"]["mean_r2_se"] = 1.0

    null = _reconstruction.evaluate_null_reconstruction(
        data,
        [0],
        ("x", "target"),
        observed,
        seed=9,
        evaluator=evaluator,
    )

    assert null["n_permutations"] == data.shape[0]
    assert null["converged"]
    assert null["mean_null_r2"] == pytest.approx(-0.05)
    assert null["above_null_r2"] == pytest.approx(0.55)
    assert null["incidental_reconstruction_rate"] == pytest.approx(0.2)
    assert null["above_null_r2_se"] == pytest.approx(
        np.std([-0.2, -0.1, 0.0, 0.1], ddof=1) / 2
    )


def test_null_reference_reports_external_cancellation_without_hidden_budget():
    data = np.column_stack([np.arange(4.0), np.arange(4.0) ** 2])
    observed = _fake_nonlinear_result(0.5)

    null = _reconstruction.evaluate_null_reconstruction(
        data,
        [0],
        ("x", "target"),
        observed,
        cancel_requested=lambda: True,
    )

    assert null["n_permutations"] == 0
    assert not null["converged"]
    assert null["cancelled"]
    assert null["reason"] == "CANCELLED_BEFORE_NULL_EVALUATION"


def test_heavy_can_add_optional_null_reference_without_replacing_metrics():
    result = _four_mis_result()

    def evaluator(data, selection, labels, *, seed, cancel_requested):
        return _fake_nonlinear_result()

    def null_evaluator(
        data,
        selection,
        labels,
        observed,
        *,
        seed,
        cancel_requested,
        evaluator,
    ):
        assert observed["mean_r2"] == 0.75
        return {
            "above_null_r2": 0.5,
            "incidental_reconstruction_rate": 0.1,
            "converged": True,
            "cancelled": False,
        }

    misda.heavy(
        result,
        0,
        null_reference=True,
        _evaluator=evaluator,
        _null_evaluator=null_evaluator,
    )

    nonlinear = result.mis[0].evaluation["nonlinear_reconstruction"]
    assert nonlinear["mean_r2"] == 0.75
    assert nonlinear["null_reference"] == {
        "above_null_r2": 0.5,
        "incidental_reconstruction_rate": 0.1,
        "converged": True,
        "cancelled": False,
    }
