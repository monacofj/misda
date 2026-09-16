import numpy as np

import misda
from misda._pareto_stability import (
    _dominated_membership_gain_radii,
    _front_dominance_margins,
    _front_membership_loss_radii,
    _normalize_by_empirical_range,
    _normalized_additive_epsilon,
    _pareto_membership_radii,
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


def test_membership_loss_radius_is_half_the_symmetric_dominance_gap():
    normalized = np.array(
        [
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.0],
        ]
    )

    loss = _front_membership_loss_radii(normalized, (0, 1, 2))
    assert [index for index, _ in loss] == [0, 1, 2]
    np.testing.assert_allclose([value for _, value in loss], [0.25, 0.25, 0.25])

    _, gain, loss_min, gain_min, radius = _pareto_membership_radii(
        normalized, (0, 1, 2)
    )
    assert gain == ()
    assert loss_min == 0.25
    assert gain_min is None
    assert radius == 0.25


def test_membership_gain_radius_breaks_all_current_dominators():
    normalized = np.array(
        [
            [0.0, 0.0],
            [0.1, 1.0],
            [1.0, 0.2],
        ]
    )

    gain = _dominated_membership_gain_radii(normalized, (0,))
    assert [index for index, _ in gain] == [1, 2]
    np.testing.assert_allclose([value for _, value in gain], [0.05, 0.10])

    loss, _, loss_min, gain_min, radius = _pareto_membership_radii(
        normalized, (0,)
    )
    assert loss == ((0, 0.5),)
    assert loss_min == 0.5
    assert gain_min == 0.05
    assert radius == 0.05


def test_range_normalized_diagnostics_are_invariant_to_positive_affine_scaling():
    Y = np.array(
        [
            [0.0, 1.0],
            [0.5, 0.5],
            [1.0, 0.0],
            [0.6, 0.6],
        ]
    )
    transformed = Y * np.array([1000.0, 0.01]) + np.array([17.0, -4.0])

    first = _normalize_by_empirical_range(Y)
    second = _normalize_by_empirical_range(transformed)
    np.testing.assert_allclose(first, second)

    first_epsilon = _normalized_additive_epsilon(first, (0,), (0, 1, 2))
    second_epsilon = _normalized_additive_epsilon(second, (0,), (0, 1, 2))
    np.testing.assert_allclose(first_epsilon, second_epsilon)

    first_radii = _pareto_membership_radii(first, (0, 1, 2))
    second_radii = _pareto_membership_radii(second, (0, 1, 2))
    np.testing.assert_allclose(
        [first_radii[2], first_radii[3], first_radii[4]],
        [second_radii[2], second_radii[3], second_radii[4]],
    )


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

    assert diagnostics.membership_radius is not None
    assert diagnostics.membership_radius >= 0.0
    assert diagnostics.membership_loss_radius_min is not None
    if diagnostics.membership_gain_radius_min is not None:
        assert diagnostics.membership_gain_radius_min >= 0.0

    evaluated = [
        value for value in diagnostics.epsilon_by_candidate if value is not None
    ]
    assert evaluated
    assert all(value >= 0.0 for value in evaluated)

    report = result.report()
    assert "Pareto stability (observed Y only):" in report
    assert "Observed front:" in report
    assert "Dominance margin:" in report
    assert "Membership radius:" in report
    assert "does not estimate noise" in report
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
    assert diagnostics.membership_radius is not None
