"""Regression tests for notebook-visible benchmark declaration assessment."""

import numpy as np

import misda
from misda.benchmark import (
    DECLARATION_MISMATCH,
    EXPECTED_DECLARATION_MISMATCH,
)


def _result():
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    data = np.column_stack([x, 2.0 * x, -x, -2.0 * x])
    return misda.discover(data, seed=19, name="two groups")


def test_public_benchmark_report_exposes_structured_declaration_checks():
    result = _result()
    observed = misda.benchmark(
        result,
        {
            "name": "wrong topology declaration",
            "latent_expected": 1,
            "structural_expected": 2,
            "graph_expectations": {"structural": {"components": 1}},
        },
    )

    assert observed.assessment["status"] == DECLARATION_MISMATCH
    report = observed.report()
    assert "Declaration assessment" in report
    assert "Status         : DECLARATION_MISMATCH" in report
    assert "graphs.structural.components: status=DECLARATION_MISMATCH" in report


def test_public_benchmark_report_exposes_expected_mismatch_reason():
    result = _result()
    observed = misda.benchmark(
        result,
        {
            "latent_expected": 4,
            "expected_mismatches": {"latent_dimension": "KNOWN_LIMITATION"},
        },
    )

    assert observed.assessment["status"] == EXPECTED_DECLARATION_MISMATCH
    report = observed.report()
    assert "latent_dimension: status=EXPECTED_DECLARATION_MISMATCH" in report
    assert "reason=KNOWN_LIMITATION" in report
