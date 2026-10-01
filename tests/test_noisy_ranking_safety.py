# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pandas as pd

from benchmarks.run_noisy_dtlz5_stress_optimization import screen_condition
from benchmarks.run_noisy_moeabench_stress_optimization import (
    screen_condition as screen_moeabench_condition,
)
from benchmarks.run_noisy_ranking_optimization import (
    _degradation_vs_full,
    _noisy_discovery_case,
)
from benchmarks.run_noisy_ranking_safety import observe, run_audit


def test_observe_sigma_zero_is_identity():
    frame = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 9.0]})
    observed = observe(frame, sigma=0.0, observation_seed=123)
    pd.testing.assert_frame_equal(observed, frame)
    assert observed is not frame


def test_observe_scales_same_standard_noise_linearly():
    frame = pd.DataFrame({"f1": [1.0, 2.0, 4.0], "f2": [3.0, 8.0, 9.0]})
    low = observe(frame, sigma=0.1, observation_seed=456)
    high = observe(frame, sigma=0.2, observation_seed=456)
    np.testing.assert_allclose(
        high.to_numpy() - frame.to_numpy(),
        2.0 * (low.to_numpy() - frame.to_numpy()),
    )


def test_small_audit_preserves_clean_reference_at_sigma_zero():
    result = run_audit(
        screen_power=5,
        sigmas=(0.0,),
        observation_seeds=(101,),
    )
    assert {row["problem"] for row in result["rows"]} == {"DTLZ5", "DPF1"}
    for row in result["rows"]:
        for policy in ("size_span", "dominance_preservation"):
            assert row["policies"][policy]["matches_clean_selection"] is True


def test_degradation_positive_means_reduced_is_worse():
    full = {"gdplus": 1.0, "igdplus": 2.0, "relative_hv": 0.8}
    reduced = {"gdplus": 1.2, "igdplus": 2.3, "relative_hv": 0.7}
    loss = _degradation_vs_full(reduced, full)
    assert loss == {
        "gdplus_loss": 0.19999999999999996,
        "igdplus_loss": 0.2999999999999998,
        "relative_hv_loss": 0.10000000000000009,
    }


def test_controlled_condition_reproduces_known_review09_divergence():
    audit = _noisy_discovery_case(
        "antagonistic_nonlinear_groups",
        sigma=0.10,
        replicate_seed=202,
        n=300,
        misda_seed=123,
    )
    assert audit["comparison"]["truth_outcome"] == "regressed"
    assert audit["policies"]["size_span"]["selected_labels"] == ["f1", "f11"]
    assert audit["policies"]["dominance_preservation"]["selected_labels"] == ["f10", "f18"]
    assert audit["policies"]["size_span"]["reduction_assessment"]["trustworthy"] is True
    assert audit["policies"]["dominance_preservation"]["reduction_assessment"]["trustworthy"] is True


def test_dtlz5_high_noise_stress_reproduces_support_asymmetry():
    _, screening = screen_condition(sigma=2.0, observation_seed=101)
    size = screening["policies"]["size_span"]
    dominance = screening["policies"]["dominance_preservation"]

    assert size["selected_labels"] == ["f1", "f2", "f4", "f5", "f6", "f8", "f10"]
    assert size["assessment"]["trustworthy"] is True
    assert size["support"]["reasons"] == []

    assert dominance["selected_labels"] == ["f1", "f2", "f5", "f7", "f8", "f9", "f10"]
    assert dominance["assessment"]["trustworthy"] is False
    assert dominance["support"]["reasons"] == ["TRANSITIVE_CHAINING"]


def test_supported_moeabench_stress_conditions_reproduce_policy_disagreement():
    conditions = (
        ("DTLZ5", 1.0, 404),
        ("DPF1", 2.0, 505),
    )
    for problem, sigma, observation_seed in conditions:
        _, screening = screen_moeabench_condition(
            problem,
            sigma=sigma,
            observation_seed=observation_seed,
        )
        assert screening["same_selected_mis"] is False
        for policy in ("size_span", "dominance_preservation"):
            assert screening["policies"][policy]["assessment"]["trustworthy"] is True
            assert screening["policies"][policy]["support"]["status"] == "SUPPORTED"
