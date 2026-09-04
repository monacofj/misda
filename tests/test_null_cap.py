"""Contracts for autonomous structural alpha-null termination."""

import itertools
import math

import numpy as np
import pytest

import misda
import misda.api as api
from misda import _statistics, _validation


def test_controlled_null_sequence_stops_exactly_at_explicit_cap():
    observed = _statistics.estimate_null_from_maxima(
        itertools.cycle((0.0, 1.0)),
        n_samples=4,
        signature=lambda _log_alpha: object(),
        max_permutations=40,
    )

    assert not observed.converged
    assert observed.n_permutations == 40
    assert observed.reason == "MAX_PERMUTATIONS_REACHED"


def test_public_null_estimator_warns_once_at_10n_cap():
    rng = np.random.default_rng(91)
    normalized = _validation.normalize_input_matrix(rng.normal(size=(8, 3)))

    with pytest.warns(RuntimeWarning, match=r"B_max=10N") as recorded:
        observed = _statistics.estimate_null_positive_correlation(
            normalized,
            signature=lambda _log_alpha: object(),
            seed=123,
        )

    assert len(recorded) == 1
    assert not observed.converged
    assert observed.n_permutations == 80
    assert observed.reason == "MAX_PERMUTATIONS_REACHED"


def test_cancellation_remains_distinct_from_autonomous_cap():
    observed = _statistics.estimate_null_from_maxima(
        itertools.cycle((0.0, 1.0)),
        n_samples=4,
        signature=lambda _log_alpha: object(),
        cancel_requested=lambda count: count == 6,
        max_permutations=40,
    )

    assert not observed.converged
    assert observed.n_permutations == 6
    assert observed.reason == "CANCELLED"


def test_alpha_null_nonconvergence_reason_reaches_discovery(monkeypatch):
    fake = _statistics.NullAlphaEstimate(
        r_null=0.2,
        se_mc=0.01,
        r_interval=(0.19, 0.21),
        log_alpha_null=math.log(0.25),
        log_alpha_interval=(math.log(0.2), math.log(0.3)),
        n_permutations=80,
        converged=False,
        lower_r_signature=("lower",),
        upper_r_signature=("upper",),
        samples=(0.2,) * 80,
        seed=123,
        rng_state={"bit_generator": "PCG64"},
        reason="MAX_PERMUTATIONS_REACHED",
    )
    monkeypatch.setattr(
        api,
        "estimate_null_positive_correlation",
        lambda *args, **kwargs: fake,
    )

    rng = np.random.default_rng(5)
    result = misda.discover(rng.normal(size=(8, 3)), seed=123)

    assert result.analysis.alpha_null_converged is False
    assert result.analysis.alpha_null_reason == "MAX_PERMUTATIONS_REACHED"
    assert result.analysis.alpha_null_permutations == 80
