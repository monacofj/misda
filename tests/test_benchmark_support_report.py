"""Regression checks for dimensional-support visibility in benchmark reports."""

import numpy as np

import misda


def test_benchmark_report_surfaces_stored_dimensional_support():
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    data = np.column_stack([x, 2.0 * x, -x, -2.0 * x])
    result = misda.analyze(data, max_evaluated_mis=1, seed=19)
    result.analysis.dimensional_support = {
        "status": "UNSUPPORTED",
        "reasons": ("TRANSITIVE_CHAINING",),
        "transitivity": {"excess": 0.125},
        "spectral": {"excess": -0.25},
    }

    report = misda.benchmark(
        result,
        {"latent_expected": 1, "structural_expected": 2},
    ).report()

    assert "Dim. support   : UNSUPPORTED" in report
    assert "transitivity_excess=0.1250" in report
    assert "spectral_excess=-0.2500" in report
    assert "Support reason : TRANSITIVE_CHAINING" in report
