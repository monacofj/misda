"""Regression tests for rank-deficient linear PRESS designs."""

import numpy as np

from misda import _reconstruction


def _rank_deficient_example(n=80):
    rng = np.random.default_rng(123)
    x = rng.uniform(0.0, 1.0, size=n)
    y = 1.0 - x
    target = x**2
    return np.column_stack([x, y, target])


def test_press_falls_back_to_explicit_loo_for_rank_deficient_design(monkeypatch):
    data = _rank_deficient_example()
    selected = (0, 1)
    eliminated = (2,)
    expected = _reconstruction._explicit_loo_predictions(
        data, selected, eliminated
    )

    original = _reconstruction._explicit_loo_predictions
    calls = []

    def spy(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(_reconstruction, "_explicit_loo_predictions", spy)
    observed = _reconstruction._press_predictions(data, selected, eliminated)

    assert calls
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)


def test_press_keeps_fast_qr_path_for_full_rank_design(monkeypatch):
    rng = np.random.default_rng(42)
    data = rng.normal(size=(80, 5))

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("full-rank design unexpectedly used explicit LOO")

    monkeypatch.setattr(
        _reconstruction,
        "_explicit_loo_predictions",
        fail_if_called,
    )
    observed = _reconstruction._press_predictions(data, (0, 1), (2, 3, 4))

    assert observed.shape == (80, 3)
    assert np.all(np.isfinite(observed))
