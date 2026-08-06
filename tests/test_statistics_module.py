"""Characterization tests for the extracted legacy statistics layer."""

import math

import numpy as np
import pytest

import misda
from misda import _statistics


PUBLIC_STATISTICS = (
    "alpha_from_r",
    "max_abs_corr",
    "estimate_null_max_r",
    "estimate_alpha_interval",
    "select_alpha",
    "AlphaRegime",
    "diagnose_alpha_regime",
    "describe_alpha_regime",
)


def _characterization_matrix():
    rng = np.random.default_rng(1234)
    data = rng.normal(size=(16, 4))
    data[:, 3] = 0.75 * data[:, 0] + 0.25 * data[:, 3]
    return data


@pytest.mark.parametrize("name", PUBLIC_STATISTICS)
def test_legacy_statistics_are_reexported_from_package(name):
    assert getattr(misda, name) is getattr(_statistics, name)


def test_alpha_and_interval_values_are_unchanged():
    data = _characterization_matrix()

    assert misda.alpha_from_r(0.5, 30) == pytest.approx(
        0.004313470570616613
    )
    assert misda.estimate_alpha_interval(
        data, B=12, random_state=9
    ) == pytest.approx(
        (
            8.754710673628166e-10,
            0.06956117348646912,
            0.9354494486453092,
            0.46472665562780285,
        )
    )


def test_null_estimate_samples_are_reproducible():
    expected = np.array(
        [
            0.22148993,
            0.20679142,
            0.28703502,
            0.04427084,
            0.11494778,
            0.26869077,
            0.29357991,
            0.20269646,
            0.27352336,
            0.25796449,
            0.44898421,
            0.46472666,
        ]
    )

    observed_max, observed = misda.estimate_null_max_r(
        _characterization_matrix(), B=12, random_state=9
    )

    np.testing.assert_allclose(observed, expected, rtol=1e-7, atol=1e-8)
    assert observed_max == observed.max()


@pytest.mark.parametrize(
    ("alpha_min", "alpha_max", "regime", "s", "s_norm"),
    [
        (0.2, 0.1, misda.AlphaRegime.SIGNAL_BELOW_NOISE, -math.log(2), math.nan),
        (0.0, 0.0, misda.AlphaRegime.END_OF_SCALE, math.nan, math.nan),
        (0.0, 0.1, misda.AlphaRegime.IMMEDIATE_SEPARATION, math.inf, math.nan),
        (0.01, 0.1, misda.AlphaRegime.LIMINAL_SEPARATION, math.log(10), 0.5),
    ],
)
def test_alpha_regime_diagnosis_is_unchanged(
    alpha_min, alpha_max, regime, s, s_norm
):
    observed = misda.diagnose_alpha_regime(alpha_min, alpha_max)

    assert observed["regime"] == int(regime)
    assert observed["S"] == pytest.approx(s, nan_ok=True)
    assert observed["S_norm"] == pytest.approx(s_norm, nan_ok=True)
    assert "Statistical regime:" in misda.describe_alpha_regime(observed)
