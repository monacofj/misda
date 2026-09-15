"""Contracts for the fixed-budget empirical alpha-null envelope."""

import math

import numpy as np
import pytest

from misda import _statistics, _validation


def test_controlled_envelope_uses_maximum_of_exactly_n_null_maxima():
    observed = _statistics.estimate_null_envelope_from_maxima(
        [0.10, 0.40, 0.30, 0.20, 0.90],
        n_samples=4,
        signature=lambda log_alpha: ("threshold", log_alpha),
    )

    assert observed.converged
    assert observed.reason is None
    assert observed.n_permutations == 4
    assert observed.samples == pytest.approx((0.10, 0.40, 0.30, 0.20))
    assert observed.r_null == pytest.approx(0.40)
    assert np.isnan(observed.se_mc)
    assert observed.r_interval == pytest.approx((0.40, 0.40))
    assert observed.log_alpha_null == pytest.approx(
        _statistics.positive_correlation_log_p(0.40, 4)
    )
    assert observed.log_alpha_interval == pytest.approx(
        (observed.log_alpha_null, observed.log_alpha_null)
    )
    assert observed.lower_r_signature == observed.upper_r_signature


def test_controlled_envelope_preserves_explicit_cancellation():
    observed = _statistics.estimate_null_envelope_from_maxima(
        [0.10, 0.30, 0.20, 0.90, 0.80],
        n_samples=5,
        signature=lambda log_alpha: log_alpha,
        cancel_requested=lambda count: count == 3,
    )

    assert not observed.converged
    assert observed.reason == "CANCELLED"
    assert observed.n_permutations == 3
    assert observed.r_null == pytest.approx(0.30)
    assert observed.samples == pytest.approx((0.10, 0.30, 0.20))


def test_public_envelope_is_reproducible_and_uses_exactly_n_permutations():
    rng = np.random.default_rng(91)
    normalized = _validation.normalize_input_matrix(rng.normal(size=(12, 4)))

    first = _statistics.estimate_null_positive_correlation(
        normalized,
        signature=lambda log_alpha: log_alpha,
        seed=123,
    )
    second = _statistics.estimate_null_positive_correlation(
        normalized,
        signature=lambda log_alpha: log_alpha,
        seed=123,
    )

    assert first == second
    assert first.converged
    assert first.reason is None
    assert first.n_permutations == normalized.n_samples
    assert len(first.samples) == normalized.n_samples
    assert first.r_null == max(first.samples)
    assert np.isnan(first.se_mc)
    assert first.r_interval == pytest.approx((first.r_null, first.r_null))
    assert first.log_alpha_interval == pytest.approx(
        (first.log_alpha_null, first.log_alpha_null)
    )
    assert first.seed == 123
    assert first.rng_state["bit_generator"] == "PCG64"


def test_all_constant_input_has_zero_empirical_envelope():
    normalized = _validation.normalize_input_matrix(np.ones((6, 3)))
    observed = _statistics.estimate_null_positive_correlation(
        normalized,
        signature=lambda log_alpha: log_alpha,
        seed=7,
    )

    assert observed.converged
    assert observed.n_permutations == 6
    assert observed.r_null == 0.0
    assert np.isnan(observed.se_mc)
    assert observed.r_interval == (0.0, 0.0)
    assert observed.log_alpha_null == pytest.approx(math.log(0.5))
