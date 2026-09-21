"""Regression tests for the conceptual boundary in benchmark reports."""

import numpy as np

import misda


def _misda_block(report):
    lines = report.splitlines()
    start = lines.index("MISDA measures (data-derived)") + 2
    end = lines.index("Benchmark validation (requires declared truth)")
    return "\n".join(lines[start:end]).rstrip()


def test_benchmark_report_embeds_ranking_report_verbatim():
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    data = np.column_stack([x, 2.0 * x, -x, -2.0 * x])
    result = misda.discover(data, seed=19)
    result.evaluate(metrics=("linear", "pareto"), candidates="all")

    report = misda.benchmark(
        result,
        {
            "name": "report boundary",
            "latent_expected": 1,
            "structural_expected": 2,
            "components_expected": [["f1", "f2"], ["f3", "f4"]],
        },
    ).report()

    assert _misda_block(report) == misda.rank(result).report()


def test_benchmark_validation_remains_truth_dependent_and_separate():
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    data = np.column_stack([x, 2.0 * x, -x, -2.0 * x])
    result = misda.discover(data, seed=19)

    report = misda.benchmark(
        result,
        {
            "name": "report boundary",
            "latent_expected": 1,
            "structural_expected": 2,
        },
    ).report()
    lines = report.splitlines()
    validation_index = lines.index("Benchmark validation (requires declared truth)")
    validation_block = "\n".join(lines[validation_index:])

    assert "expected=" not in _misda_block(report)
    assert "relative_error=" not in _misda_block(report)
    assert "Declaration assessment" not in _misda_block(report)
    assert "Dimensional accuracy" not in _misda_block(report)

    assert "Declaration" in validation_block
    assert "Declaration assessment" in validation_block
    assert "Dimensional accuracy" in validation_block
    assert "expected=1" in validation_block
