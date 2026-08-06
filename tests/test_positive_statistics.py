"""Tests for the positive, log-domain statistical layer."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from misda import _statistics, _validation


def _signed_input():
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    frame = pd.DataFrame(
        {
            "x": x,
            "positive": 2.0 * x,
            "negative": -x,
            "constant": np.full_like(x, 7.0),
        }
    )
    return _validation.normalize_input_matrix(frame)


def test_positive_log_p_is_one_tailed_and_preserves_extreme_values():
    observed = _statistics.positive_correlation_log_p(0.5, 30)
    expected = stats.norm.logsf(np.arctanh(0.5) * np.sqrt(27))

    assert observed == pytest.approx(expected)
    assert observed != pytest.approx(math.log(2.0) + expected)
    assert _statistics.positive_correlation_log_p(0.0, 30) == pytest.approx(
        math.log(0.5)
    )
    assert _statistics.positive_correlation_log_p(1.0, 30) == -math.inf
    assert np.isfinite(
        _statistics.positive_correlation_log_p(np.nextafter(1.0, 0.0), 30)
    )


def test_positive_log_p_rejects_negative_nonfinite_and_invalid_sample_size():
    with pytest.raises(ValueError, match=r"correlations in \[0, 1\]"):
        _statistics.positive_correlation_log_p(-0.1, 30)
    with pytest.raises(ValueError, match=r"correlations in \[0, 1\]"):
        _statistics.positive_correlation_log_p(np.nan, 30)
    with pytest.raises(ValueError, match="greater than or equal to 4"):
        _statistics.positive_correlation_log_p(0.5, 3)


def test_signed_correlations_and_constant_pairs_remain_explicit():
    observed = _statistics.compute_correlation_statistics(_signed_input())

    assert observed.labels == ("x", "positive", "negative", "constant")
    assert observed.constant_indices == (3,)
    assert observed.correlation[0, 1] == pytest.approx(1.0)
    assert observed.correlation[0, 2] == pytest.approx(-1.0)
    assert np.isnan(observed.correlation[0, 3])
    assert np.isnan(observed.correlation[3, 3])
    assert not observed.valid_pairs[0, 0]
    assert not observed.valid_pairs[0, 3]
    assert np.isnan(observed.log_p[0, 3])
    assert observed.log_alpha_onset == -math.inf
    assert observed.alpha_onset == 0.0


def test_no_positive_pair_has_no_onset():
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    normalized = _validation.normalize_input_matrix(np.column_stack([x, -x]))
    observed = _statistics.compute_correlation_statistics(normalized)

    assert observed.log_alpha_onset is None
    assert observed.alpha_onset is None


def test_edge_masks_exclude_negative_relations_from_positive_graph():
    observed = _statistics.compute_correlation_statistics(_signed_input())
    positive, signed = _statistics.correlation_edge_masks(
        observed,
        log_alpha=-math.inf,
    )

    assert positive[0, 1]
    assert not positive[0, 2]
    assert signed[0, 1]
    assert signed[0, 2]
    assert not signed[0, 3]
    np.testing.assert_array_equal(positive, positive.T)
    np.testing.assert_array_equal(signed, signed.T)


def test_edge_threshold_is_inclusive_and_monotone():
    rng = np.random.default_rng(22)
    x = rng.normal(size=40)
    data = np.column_stack(
        [x, 0.8 * x + 0.2 * rng.normal(size=40), rng.normal(size=40)]
    )
    correlation = _statistics.compute_correlation_statistics(
        _validation.normalize_input_matrix(data)
    )
    threshold = float(correlation.log_p[0, 1])

    at_threshold, _ = _statistics.correlation_edge_masks(correlation, threshold)
    below_threshold, _ = _statistics.correlation_edge_masks(
        correlation,
        np.nextafter(threshold, -math.inf),
    )
    more_aggressive, _ = _statistics.correlation_edge_masks(
        correlation,
        math.log(0.5),
    )

    assert at_threshold[0, 1]
    assert not below_threshold[0, 1]
    assert np.all(at_threshold <= more_aggressive)


def test_separation_status_uses_strict_log_domain_ordering():
    assert _statistics.separation_status(-10.0, -5.0) is (
        _statistics.SeparationStatus.NULL_SEPARATION
    )
    assert _statistics.separation_status(-5.0, -5.0) is (
        _statistics.SeparationStatus.NO_NULL_SEPARATION
    )
    assert _statistics.separation_status(-4.0, -5.0) is (
        _statistics.SeparationStatus.NO_NULL_SEPARATION
    )
    assert _statistics.separation_status(None, -5.0) is (
        _statistics.SeparationStatus.NO_NULL_SEPARATION
    )


def test_log_interpolation_matches_arithmetic_alpha_interpolation():
    log_onset = -1000.0
    log_null = math.log(0.1)

    assert _statistics.interpolate_log_alpha(log_onset, log_null, 0.0) == log_onset
    assert _statistics.interpolate_log_alpha(log_onset, log_null, 1.0) == log_null
    observed = _statistics.interpolate_log_alpha(log_onset, log_null, 0.25)
    expected = np.logaddexp(
        math.log(0.75) + log_onset,
        math.log(0.25) + log_null,
    )
    assert observed == pytest.approx(expected)
    assert np.isfinite(observed)


def test_log_interpolation_requires_an_onset():
    with pytest.raises(ValueError, match="without a positive onset"):
        _statistics.interpolate_log_alpha(None, -2.0, 0.5)


def test_sequential_null_stops_at_initial_n_when_signatures_match():
    observed = _statistics.estimate_null_from_maxima(
        [0.2, 0.2, 0.2, 0.2, 0.9],
        n_samples=4,
        signature=lambda log_alpha: int(log_alpha <= -1.0),
    )

    assert observed.converged
    assert observed.n_permutations == 4
    assert observed.r_null == pytest.approx(0.2)
    assert observed.se_mc == pytest.approx(0.0)
    assert observed.r_interval == pytest.approx((0.2, 0.2))
    assert observed.samples == pytest.approx((0.2, 0.2, 0.2, 0.2))


def test_sequential_null_checks_each_additional_permutation():
    checks = []

    def cancel(count):
        checks.append(count)
        return count == 6

    observed = _statistics.estimate_null_from_maxima(
        [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.5],
        n_samples=4,
        signature=lambda log_alpha: int(log_alpha < -1.3),
        cancel_requested=cancel,
    )

    assert not observed.converged
    assert observed.n_permutations == 6
    assert checks == [4, 5, 6]
    assert observed.lower_r_signature != observed.upper_r_signature


def test_exhausted_controlled_sequence_returns_explicit_nonconvergence():
    observed = _statistics.estimate_null_from_maxima(
        [0.0, 0.0, 1.0, 1.0],
        n_samples=4,
        signature=lambda log_alpha: int(log_alpha < -1.3),
    )

    assert not observed.converged
    assert observed.n_permutations == 4
    assert observed.lower_r_signature != observed.upper_r_signature


def test_permutation_estimator_is_reproducible_and_records_rng_state():
    rng = np.random.default_rng(91)
    normalized = _validation.normalize_input_matrix(rng.normal(size=(8, 3)))

    def signature(log_alpha):
        return int(log_alpha <= math.log(0.5))

    first = _statistics.estimate_null_positive_correlation(
        normalized,
        signature=signature,
        seed=123,
    )
    second = _statistics.estimate_null_positive_correlation(
        normalized,
        signature=signature,
        seed=123,
    )

    assert first == second
    assert first.converged
    assert first.n_permutations >= normalized.n_samples
    assert first.seed == 123
    assert first.rng_state["bit_generator"] == "PCG64"


def test_entirely_constant_input_has_zero_null_reference():
    normalized = _validation.normalize_input_matrix(np.ones((5, 3)))
    observed = _statistics.estimate_null_positive_correlation(
        normalized,
        signature=lambda log_alpha: log_alpha,
        seed=7,
    )

    assert observed.converged
    assert observed.n_permutations == 5
    assert observed.r_null == 0.0
    assert observed.se_mc == 0.0
    assert observed.log_alpha_null == pytest.approx(math.log(0.5))
