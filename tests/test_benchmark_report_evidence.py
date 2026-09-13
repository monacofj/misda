"""Regression tests for rich observed evidence in benchmark reports."""

import numpy as np

import misda


def _evaluated_result():
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    data = np.column_stack([x, 2.0 * x, -x, -2.0 * x])
    result = misda.discover(data, seed=19, name="two groups")
    misda.evaluate(result, metrics=("linear", "pareto"), candidates="all")
    return result


def test_report_exposes_observed_evidence_without_external_metric_truth():
    result = _evaluated_result()
    observed = misda.benchmark(result, {"name": "observed evidence"})

    report = observed.report()

    assert "Candidate evaluation evidence" in report
    assert f"Linear scope   : {len(result)}/{len(result)} candidates" in report
    assert "Linear selected: mean_r2=" in report
    assert "Linear across  : mean_r2 min=" in report
    assert "worst_r2 min=" in report
    assert f"Pareto scope   : {len(result)}/{len(result)} candidates" in report
    assert "Pareto selected: retention=" in report
    assert "Pareto fronts  : full=" in report
    assert "Pareto across  : retention min=" in report
    assert "Pareto across  : validity min=" in report
    assert "Pareto across  : jaccard min=" in report
    assert "Pareto declaration agreement" in report
    assert "N/A — pareto_expected was not declared" in report
    assert "Expected components" not in report


def test_report_distinguishes_generating_families_structural_units_and_components():
    result = _evaluated_result()
    truth = {
        "name": "declaration semantics",
        "structural_expected": 2,
        "families_expected": [["f1", "f2", "f3", "f4"]],
        "blocks_expected": [["f1", "f2"], ["f3", "f4"]],
        "components_expected": [["f1", "f2"], ["f3", "f4"]],
    }
    report = misda.benchmark(result, truth).report()

    assert "Generating families : {f1, f2, f3, f4}" in report
    assert "Structural units    : {f1, f2} | {f3, f4}" in report
    assert "Expected components : {f1, f2} | {f3, f4}" in report
    assert "Expected blocks:" not in report


def test_report_states_when_candidate_families_were_not_evaluated():
    data = np.eye(6, 4)
    result = misda.discover(data, seed=7)
    observed = misda.benchmark(result, {})

    report = observed.report()

    assert "Linear scope   : not evaluated" in report
    assert "Linear selected: N/A" in report
    assert "Pareto scope   : not evaluated" in report
    assert "Pareto selected: N/A" in report
