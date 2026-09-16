"""Regression tests for the conceptual boundary in benchmark reports."""

import numpy as np

import misda


def test_benchmark_report_separates_misda_measures_from_truth_validation():
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    data = np.column_stack([x, 2.0 * x, -x, -2.0 * x])
    result = misda.discover(data, seed=19)
    misda.evaluate(result, metrics=("linear", "pareto"), candidates="all")

    report = misda.benchmark(
        result,
        {
            "name": "report boundary",
            "latent_expected": 1,
            "structural_expected": 2,
            "components_expected": [["f1", "f2"], ["f3", "f4"]],
        },
    ).report()
    lines = report.splitlines()

    misda_index = lines.index("MISDA measures (data-derived)")
    observed_index = lines.index("Observed analysis")
    evidence_index = lines.index("Candidate evaluation evidence")
    validation_index = lines.index("Benchmark validation (requires declared truth)")
    declaration_index = lines.index("Declaration")
    assessment_index = lines.index("Declaration assessment")
    accuracy_index = lines.index("Dimensional accuracy")

    assert misda_index < observed_index < evidence_index < validation_index
    assert validation_index < declaration_index < assessment_index < accuracy_index

    misda_block = "\n".join(lines[misda_index:validation_index])
    validation_block = "\n".join(lines[validation_index:])

    assert "expected=" not in misda_block
    assert "relative_error=" not in misda_block
    assert "Declaration assessment" not in misda_block
    assert "Dimensional accuracy" not in misda_block

    assert "Declaration" in validation_block
    assert "Declaration assessment" in validation_block
    assert "Dimensional accuracy" in validation_block
    assert "expected=1" in validation_block
