"""Regression contracts for the exact fast linear jackknife path."""

import numpy as np

from misda import _linear, _reconstruction
from misda._tolerances import GATE_ATOL, GATE_RTOL


def _assert_optional_close(observed, expected):
    if expected is None:
        assert observed is None
    else:
        assert np.isclose(
            observed,
            expected,
            rtol=GATE_RTOL,
            atol=GATE_ATOL,
        )


def _assert_linear_equivalent(observed, expected):
    assert observed["r2_reason_by_objective"] == expected["r2_reason_by_objective"]
    assert observed["reason_by_metric"] == expected["reason_by_metric"]

    expected_r2 = expected["r2_by_objective"]
    observed_r2 = observed["r2_by_objective"]
    if expected_r2 is None:
        assert observed_r2 is None
    else:
        assert set(observed_r2) == set(expected_r2)
        for label in expected_r2:
            _assert_optional_close(observed_r2[label], expected_r2[label])

    _assert_optional_close(observed["mean_r2"], expected["mean_r2"])
    _assert_optional_close(observed["worst_r2"], expected["worst_r2"])

    observed_jackknife = observed["jackknife"]
    expected_jackknife = expected["jackknife"]
    assert observed_jackknife["n_replicates"] == expected_jackknife["n_replicates"]
    assert observed_jackknife["reason"] == expected_jackknife["reason"]

    expected_se = expected_jackknife["r2_se_by_objective"]
    observed_se = observed_jackknife["r2_se_by_objective"]
    if expected_se is None:
        assert observed_se is None
    else:
        assert set(observed_se) == set(expected_se)
        for label in expected_se:
            _assert_optional_close(observed_se[label], expected_se[label])

    _assert_optional_close(
        observed_jackknife["mean_r2_se"],
        expected_jackknife["mean_r2_se"],
    )
    _assert_optional_close(
        observed_jackknife["worst_r2_se"],
        expected_jackknife["worst_r2_se"],
    )


def test_fast_linear_jackknife_matches_reference_with_gate_tolerance():
    rng = np.random.default_rng(20260911)
    sources = rng.normal(size=(48, 3))
    targets = sources @ np.array(
        [
            [1.2, -0.4, 0.6],
            [0.3, 1.5, -0.2],
            [-0.8, 0.2, 1.1],
        ]
    ) + rng.normal(scale=0.15, size=(48, 3))
    data = np.column_stack((sources, targets))
    labels = tuple(f"f{index + 1}" for index in range(data.shape[1]))
    selected = (0, 1, 2)

    expected = _reconstruction.evaluate_linear_reconstruction(
        data,
        selected,
        labels,
    )
    observed = _linear.evaluate_linear_reconstruction(
        data,
        selected,
        labels,
    )

    _assert_linear_equivalent(observed, expected)


def test_regular_design_uses_fast_path(monkeypatch):
    rng = np.random.default_rng(1109)
    data = rng.normal(size=(40, 6))
    labels = tuple(f"f{index + 1}" for index in range(data.shape[1]))

    def fail_reference(*_args, **_kwargs):
        raise AssertionError("regular design unexpectedly used reference fallback")

    monkeypatch.setattr(_linear, "_reference_linear_reconstruction", fail_reference)
    observed = _linear.evaluate_linear_reconstruction(
        data,
        selected_indices=(0, 2, 4),
        labels=labels,
    )

    assert observed["jackknife"]["n_replicates"] == data.shape[0]


def test_rank_deficient_design_falls_back_to_reference(monkeypatch):
    rng = np.random.default_rng(911)
    source = rng.normal(size=30)
    target = 2.0 * source + rng.normal(scale=0.1, size=30)
    data = np.column_stack((source, source, target))
    labels = ("source", "duplicate", "target")
    sentinel = {"fallback": True}

    def reference(*_args, **_kwargs):
        return sentinel

    monkeypatch.setattr(_linear, "_reference_linear_reconstruction", reference)
    observed = _linear.evaluate_linear_reconstruction(
        data,
        selected_indices=(0, 1),
        labels=labels,
    )

    assert observed is sentinel
