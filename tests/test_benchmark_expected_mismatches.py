"""Regression tests for expected versus unexpected benchmark mismatches."""

import numpy as np
import pytest

import misda
from examples.benchmarks.run_benchmark import (
    enforce_scientific_assessment,
    unexpected_mismatch_case_ids,
)
from misda.benchmark import (
    BenchmarkCase,
    DECLARATION_MISMATCH,
    EXPECTED_DECLARATION_MISMATCH,
)
from misda.benchmarks.cases import make_case5_chain_structure


def _two_group_result():
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    data = np.column_stack([x, 2.0 * x, -x, -2.0 * x])
    return misda.discover(data, seed=19, name="two groups")


def test_expected_mismatches_are_field_specific_not_blanket_adversarial():
    result = _two_group_result()
    case = BenchmarkCase.from_truth(
        "x",
        {
            "name": "field-specific mismatch",
            "latent_expected": 4,
            "structural_expected": 4,
            "blocks_expected": [["f1", "f2"], ["f3", "f4"]],
            "graph_expectations": {"structural": {"components": 1}},
            "expected_mismatches": {
                "latent_dimension": "TRANSITIVE_CHAINING",
                "structural_dimension": "TRANSITIVE_CHAINING",
            },
        },
    )

    assessment = case.evaluate(result)
    checks = {check["field"]: check for check in assessment["checks"]}

    assert checks["latent_dimension"]["status"] == EXPECTED_DECLARATION_MISMATCH
    assert checks["latent_dimension"]["reason"] == "TRANSITIVE_CHAINING"
    assert checks["structural_dimension"]["status"] == EXPECTED_DECLARATION_MISMATCH
    assert checks["graphs.structural.components"]["status"] == DECLARATION_MISMATCH
    assert assessment["status"] == DECLARATION_MISMATCH


def test_case5_declares_only_dimensional_chaining_mismatches():
    _frame, truth = make_case5_chain_structure(N=64, seed=123)

    assert truth["expected_mismatches"] == {
        "latent_dimension": "TRANSITIVE_CHAINING",
        "structural_dimension": "TRANSITIVE_CHAINING",
    }


def test_strict_gate_rejects_only_unexpected_declaration_mismatches():
    artifact = {
        "cases": [
            {
                "case_id": "case_05",
                "assessment": {"status": EXPECTED_DECLARATION_MISMATCH},
            },
            {
                "case_id": "case_03",
                "assessment": {"status": "DECLARATION_MATCH"},
            },
        ]
    }
    assert unexpected_mismatch_case_ids(artifact) == ()
    enforce_scientific_assessment(artifact)

    artifact["cases"][1]["assessment"]["status"] = DECLARATION_MISMATCH
    assert unexpected_mismatch_case_ids(artifact) == ("case_03",)
    with pytest.raises(SystemExit, match="case_03"):
        enforce_scientific_assessment(artifact)
