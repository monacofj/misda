"""Tests for PRESS reconstruction and Pareto preservation in analyze()."""

import numpy as np
import pandas as pd
import pytest

import misda
from misda import _pareto, _reconstruction


def test_press_matches_explicit_leave_one_out_predictions():
    x = np.arange(8.0)
    data = np.column_stack([x, 3.0 + 2.0 * x])
    observed = _reconstruction.evaluate_linear_reconstruction(
        data,
        selected_indices=[0],
        labels=("x", "target"),
    )

    assert observed["r2_by_objective"]["target"] == pytest.approx(1.0)
    assert observed["mean_r2"] == pytest.approx(1.0)
    assert observed["worst_r2"] == pytest.approx(1.0)
    assert observed["jackknife"]["n_replicates"] == 8
    assert observed["jackknife"]["mean_r2_se"] == pytest.approx(0.0)


def test_linear_reconstruction_preserves_negative_r2():
    rng = np.random.default_rng(14)
    data = np.column_stack([rng.normal(size=20), rng.normal(size=20)])
    observed = _reconstruction.evaluate_linear_reconstruction(
        data,
        selected_indices=[0],
        labels=("source", "noise"),
    )

    assert observed["r2_by_objective"]["noise"] < 0.0
    assert observed["mean_r2"] < 0.0
    assert observed["worst_r2"] < 0.0


def test_no_reduction_is_requested_but_mathematically_undefined():
    data = np.arange(12.0).reshape(6, 2)
    observed = _reconstruction.evaluate_linear_reconstruction(
        data,
        selected_indices=[0, 1],
        labels=("a", "b"),
    )

    assert observed["r2_by_objective"] is None
    assert observed["mean_r2"] is None
    assert observed["worst_r2"] is None
    assert observed["reason_by_metric"] == {
        "r2_by_objective": "NO_ELIMINATED_OBJECTIVES",
        "mean_r2": "NO_ELIMINATED_OBJECTIVES",
        "worst_r2": "NO_ELIMINATED_OBJECTIVES",
    }


def test_constant_target_is_isolated_without_contaminating_other_targets():
    x = np.arange(8.0)
    data = np.column_stack([x, 2.0 * x, np.ones_like(x)])
    observed = _reconstruction.evaluate_linear_reconstruction(
        data,
        selected_indices=[0],
        labels=("x", "linear", "constant"),
    )

    assert observed["r2_by_objective"]["linear"] == pytest.approx(1.0)
    assert observed["r2_by_objective"]["constant"] is None
    assert observed["r2_reason_by_objective"] == {
        "constant": "CONSTANT_TARGET"
    }
    assert observed["mean_r2"] == pytest.approx(1.0)
    assert observed["worst_r2"] == pytest.approx(1.0)
    assert observed["jackknife"]["r2_se_by_objective"]["constant"] is None


def test_singular_design_is_handled_by_stable_pseudoinverse():
    x = np.arange(8.0)
    data = np.column_stack([x, 2.0 * x, 4.0 * x])
    observed = _reconstruction.evaluate_linear_reconstruction(
        data,
        selected_indices=[0, 1],
        labels=("x", "duplicate", "target"),
    )

    assert observed["r2_by_objective"]["target"] == pytest.approx(1.0)


def test_pareto_retention_one_third_example():
    data = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    observed = _pareto.evaluate_pareto_preservation(data, [0])

    assert observed == {
        "pareto_retention": pytest.approx(1 / 3),
        "pareto_validity": pytest.approx(1.0),
        "pareto_jaccard": pytest.approx(1 / 3),
        "full_front_size": 3,
        "reduced_front_size": 1,
        "intersection_size": 1,
        "union_size": 3,
        "exact_preservation": False,
        "reduced_front_indices": (0,),
    }


def test_pareto_duplicates_are_deduplicated_and_mapped_to_original_rows():
    data = np.array(
        [[0.0, 2.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0]]
    )
    full = _pareto.get_nondominated_mask_minimize(data)
    observed = _pareto.evaluate_pareto_preservation(
        data,
        [0, 1],
        full_front=full,
    )

    np.testing.assert_array_equal(full, [True, True, True, True])
    assert observed["full_front_size"] == 4
    assert observed["reduced_front_size"] == 4
    assert observed["reduced_front_indices"] == (0, 1, 2, 3)
    assert observed["exact_preservation"]
    assert observed["pareto_retention"] == 1.0
    assert observed["pareto_validity"] == 1.0
    assert observed["pareto_jaccard"] == 1.0


def test_mixed_directions_fail_explicitly():
    data = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="Mixed objective directions"):
        _pareto.evaluate_pareto_preservation(
            data,
            [0],
            directions=[-1, 1],
        )


@pytest.mark.parametrize(
    ("limit", "expected"),
    [(1, 1), (100, 4), (None, 4)],
)
def test_analyze_evaluates_only_the_requested_ranked_prefix(limit, expected):
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    data = np.column_stack([x, 2.0 * x, -x, -2.0 * x])
    observed = misda.analyze(
        data,
        max_evaluated_mis=limit,
        seed=8,
    )

    assert observed.analysis.n_mis == 4
    assert observed.analysis.n_evaluated_mis == expected
    for index, candidate in enumerate(observed.mis):
        if index < expected:
            assert set(candidate.evaluation) == {
                "linear_reconstruction",
                "pareto_preservation",
            }
        else:
            assert candidate.evaluation == {}


def test_analyze_reuses_full_pareto_front(monkeypatch):
    calls = 0
    original = _pareto.get_nondominated_mask_minimize

    def counted(data):
        nonlocal calls
        calls += 1
        return original(data)

    monkeypatch.setattr(_pareto, "get_nondominated_mask_minimize", counted)
    monkeypatch.setattr(misda.api, "get_nondominated_mask_minimize", counted)
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    data = np.column_stack([x, 2.0 * x, -x, -2.0 * x])

    observed = misda.analyze(data, max_evaluated_mis=2, seed=8)

    assert observed.analysis.n_evaluated_mis == 2
    assert calls == 3  # one full front plus one reduced front per evaluated MIS


def test_static_public_api_supports_deprecated_caution_and_ignores_coverage():
    x = np.arange(6.0)
    data = np.column_stack([x, 2.0 * x])

    with pytest.warns(DeprecationWarning, match="caution"):
        caution_result = misda.analyze(data, caution=0.5, seed=2)
    direct_result = misda.analyze(data, aggressiveness=0.5, seed=2)
    with pytest.warns(DeprecationWarning, match="ensure_coverage"):
        coverage_result = misda.analyze(
            data,
            aggressiveness=0.5,
            ensure_coverage=True,
            seed=2,
        )

    assert caution_result.analysis.log_alpha == direct_result.analysis.log_alpha
    assert coverage_result.best_mis.indices == direct_result.best_mis.indices


def test_static_public_api_rejects_manual_alpha_and_conflicting_names():
    x = np.arange(6.0)
    data = np.column_stack([x, 2.0 * x])

    with pytest.raises(ValueError, match="alpha is not supported"):
        misda.analyze(data, alpha=0.05)
    with pytest.warns(DeprecationWarning, match="caution"):
        with pytest.raises(ValueError, match="conflicting"):
            misda.analyze(data, caution=0.2, aggressiveness=0.8)
