"""Regression checks for dimensional-support visibility in benchmark reports."""

import numpy as np

import misda


def test_benchmark_report_surfaces_stored_dimensional_support():
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    data = np.column_stack([x, 2.0 * x, -x, -2.0 * x])
    result = misda.discover(data, seed=19)

    report = misda.benchmark(
        result,
        {"latent_expected": 1, "structural_expected": 2},
    ).report()

    assert f"Dim. support   : {result.support.status}" in report
    for support_result in result.support.results:
        assert result.support.for_candidate(support_result.candidate_index) is support_result
