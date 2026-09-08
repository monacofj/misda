"""Regression tests for structured benchmark graph expectations."""

import numpy as np

import misda
from misda.benchmark import (
    BenchmarkCase,
    DECLARATION_MATCH,
    DECLARATION_MISMATCH,
)


def _two_group_result():
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    data = np.column_stack([x, 2.0 * x, -x, -2.0 * x])
    return misda.discover(data, seed=19, name="two groups")


def test_from_truth_preserves_structured_graph_expectations():
    result = _two_group_result()
    truth = {
        "name": "two groups",
        "latent_expected": 1,
        "structural_expected": 2,
        "blocks_expected": [["f1", "f2"], ["f3", "f4"]],
        "graph_expected": "two positive components; one signed component",
        "graph_expectations": {
            "structural": {"edges": 2, "components": 2},
            "dependence": {"edges": 6, "components": 1},
        },
    }

    case = BenchmarkCase.from_truth("x", truth)
    assert case.graph_expectations == truth["graph_expectations"]

    assessment = case.evaluate(result)
    graph_checks = [
        check for check in assessment["checks"]
        if check["field"].startswith("graphs.")
    ]
    assert graph_checks
    assert all(check["status"] == DECLARATION_MATCH for check in graph_checks)


def test_wrong_structured_graph_expectation_is_a_declaration_mismatch():
    result = _two_group_result()
    case = BenchmarkCase.from_truth(
        "x",
        {
            "name": "wrong graph",
            "latent_expected": 1,
            "structural_expected": 2,
            "blocks_expected": [["f1", "f2"], ["f3", "f4"]],
            "graph_expectations": {
                "structural": {"components": 1},
            },
        },
    )

    assessment = case.evaluate(result)
    graph_check = next(
        check for check in assessment["checks"]
        if check["field"] == "graphs.structural.components"
    )
    assert graph_check["observed"] == 2
    assert graph_check["expected"] == 1
    assert graph_check["status"] == DECLARATION_MISMATCH
    assert assessment["status"] == DECLARATION_MISMATCH
