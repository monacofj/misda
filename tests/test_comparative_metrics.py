"""Contracts for direct MISDA/PCA reconstruction comparison."""

import numpy as np
import pandas as pd
import pytest

import misda
import misda.benchmarks as bench


def test_global_standardized_reconstruction_r2_equal_weights_objectives():
    observed = np.column_stack(
        [
            np.array([-1.0, 0.0, 1.0, 2.0]),
            np.array([-100.0, 0.0, 100.0, 200.0]),
        ]
    )
    reconstructed = observed.copy()
    reconstructed[:, 0] += np.array([0.1, -0.1, 0.1, -0.1])
    reconstructed[:, 1] += np.array([10.0, -10.0, 10.0, -10.0])

    score = bench.global_standardized_reconstruction_r2(observed, reconstructed)

    per_objective = []
    for column in range(observed.shape[1]):
        target = observed[:, column]
        predicted = reconstructed[:, column]
        total = np.sum((target - np.mean(target)) ** 2)
        residual = np.sum((target - predicted) ** 2)
        per_objective.append(1.0 - residual / total)
    assert score == pytest.approx(np.mean(per_objective))


def test_misda_common_score_counts_preserved_objectives_as_exact():
    x = np.linspace(-2.0, 2.0, 12)
    frame = pd.DataFrame({"f1": x, "f2": 2.0 * x, "f3": -3.0 * x})
    result = misda.discover(frame, seed=123)
    ranking = misda.rank(result)
    misda.evaluate(result, metrics=("linear",), candidates=ranking[:1])
    selected_candidate = ranking.selected

    score = bench.misda_global_standardized_external_r2(frame, result)

    selected = set(selected_candidate.indices)
    expected = []
    for index, label in enumerate(frame.columns):
        if index in selected:
            expected.append(1.0)
        else:
            expected.append(selected_candidate.linear.r2(label))
    assert score == pytest.approx(np.mean(expected))


def test_external_pca_curve_uses_common_metric_and_reaches_full_reconstruction():
    rng = np.random.default_rng(44)
    frame = pd.DataFrame(rng.normal(size=(24, 4)), columns=list("abcd"))
    curve = bench.pca_external_reconstruction_curve(frame, max_components=4)

    assert [point["dimension"] for point in curve] == [1, 2, 3, 4]
    assert all(bench.COMMON_RECONSTRUCTION_METRIC in point for point in curve)
    assert curve[-1][bench.COMMON_RECONSTRUCTION_METRIC] == pytest.approx(1.0, abs=1e-12)


def test_in_sample_pca_metric_remains_separate_from_external_comparison_metric():
    rng = np.random.default_rng(91)
    frame = pd.DataFrame(rng.normal(size=(20, 3)))
    native = bench.pca_in_sample_reconstruction_curve(frame, max_components=2)
    external = bench.pca_external_reconstruction_curve(frame, max_components=2)

    assert set(native[0]) == {"dimension", "global_standardized_r2"}
    assert set(external[0]) == {"dimension", bench.COMMON_RECONSTRUCTION_METRIC}
