"""Regression tests for benchmark report evidence and declarations."""

import numpy as np

import misda


def _evaluated_result():
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    data = np.column_stack([x, 2.0 * x, -x, -2.0 * x])
    result = misda.discover(data, seed=19, name="two groups")
    result.evaluate(metrics=("linear", "pareto"), candidates="all")
    return result


def _misda_block(report):
    lines = report.splitlines()
    start = lines.index("MISDA measures (data-derived)") + 2
    end = lines.index("Benchmark validation (requires declared truth)")
    return "\n".join(lines[start:end]).rstrip()


def test_report_uses_native_misda_report_without_benchmark_only_expansion():
    result = _evaluated_result()
    observed = misda.benchmark(result, {"name": "observed evidence"})

    report = observed.report()

    assert _misda_block(report) == misda.rank(result).report()
    assert "Candidate evaluation evidence" not in report
    assert "Linear selected:" not in report
    assert "Pareto selected:" not in report
    assert "Pareto declaration agreement" in report
    assert "N/A — pareto_expected was not declared" in report
    assert "Expected components" not in report
    assert "Structural component reconstruction" not in report


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
    assert "Structural component reconstruction" in report
    assert "Expected blocks:" not in report

    lines = report.splitlines()
    selected_index = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("  selected_structural_units: ")
    )
    continuation = " " * len("  selected_structural_units: ")
    assert "status=" in lines[selected_index]
    assert lines[selected_index + 1].startswith(continuation + "observed=")
    assert lines[selected_index + 2].startswith(continuation + "expected=")
    assert lines[selected_index + 3].startswith(continuation + "reason=")


def test_report_does_not_invent_unevaluated_candidate_metric_sections():
    data = np.eye(6, 4)
    result = misda.discover(data, seed=7)
    observed = misda.benchmark(result, {})

    report = observed.report()

    assert _misda_block(report) == result.report()
    assert "Linear scope" not in report
    assert "Linear selected" not in report
    assert "Pareto scope" not in report
    assert "Pareto selected" not in report
