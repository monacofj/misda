import numpy as np

import misda
from misda._pareto_stability import (
    _front_dominance_margins,
    _normalize_by_empirical_range,
    _normalized_additive_epsilon,
)


def test_dominance_margin_and_additive_epsilon_have_direct_geometry():
    Y = np.array(
        [
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.0],
        ]
    )
    normalized = _normalize_by_empirical_range(Y)

    margins = _front_dominance_margins(normalized, (0, 1, 2))
    assert [index for index, _ in margins] == [0, 1, 2]
    np.testing.assert_allclose([value for _, value in margins], [0.5, 0.5, 0.5])

    epsilon = _normalized_additive_epsilon(
        normalized,
        approximation_indices=(0,),
        target_indices=(0, 1, 2),
    )
    assert epsilon == 1.0


def test_range_normalized_diagnostics_are_invariant_to_positive_affine_scaling():
    Y = np.array(
        [
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.0],
        ]
    )
    transformed = Y * np.array([1000.0, 0.01]) + np.array([17.0, -4.0])

    first = _normalize_by_empirical_range(Y)
    second = _normalize_by_empirical_range(transformed)
    np.testing.assert_allclose(first, second)

    first_epsilon = _normalized_additive_epsilon(first, (0,), (0, 1, 2))
    second_epsilon = _normalized_additive_epsilon(second, (0,), (0, 1, 2))
    np.testing.assert_allclose(first_epsilon, second_epsilon)


def test_public_pareto_evaluation_attaches_y_only_stability_diagnostics():
    rng = np.random.default_rng(123)
    Y = rng.normal(size=(48, 4))

    result = misda.discover(Y, seed=123)
    assert not hasattr(result, "pareto_stability")

    returned = misda.evaluate(result, metrics=("pareto",))
    assert returned is result
    diagnostics = result.pareto_stability

    assert isinstance(diagnostics, misda.ParetoStabilityDiagnostics)
    assert diagnostics.observed_front_size == len(diagnostics.observed_front_indices)
    assert diagnostics.observed_front_fraction == diagnostics.observed_front_size / 48
    assert len(diagnostics.epsilon_by_candidate) == len(result)
    assert diagnostics.normalization == "empirical_range"

    evaluated = [
        value for value in diagnostics.epsilon_by_candidate if value is not None
    ]
    assert evaluated
    assert all(value >= 0.0 for value in evaluated)

    report = result.report()
    assert "Pareto stability (observed Y only):" in report
    assert "Observed front:" in report
    assert "Dominance margin:" in report
    assert "Additive epsilon+:" in report


def test_stability_layer_does_not_require_benchmark_truth():
    x = np.linspace(0.0, 1.0, 40)
    Y = np.column_stack([x, 1.0 - x, x + 0.01 * np.sin(7.0 * x)])

    result = misda.discover(Y, seed=321)
    misda.evaluate(result, metrics=("pareto",))

    diagnostics = result.pareto_stability
    assert 0.0 < diagnostics.observed_front_fraction <= 1.0
    assert diagnostics.dominance_margin_min is not None
    assert diagnostics.dominance_margin_median is not None
    assert diagnostics.dominance_margin_max is not None
