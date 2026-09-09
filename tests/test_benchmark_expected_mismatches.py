"""Regression tests for expected versus unexpected benchmark mismatches."""

from types import SimpleNamespace

import numpy as np
import pytest

import misda
from examples.benchmarks.run_benchmark import (
    enforce_scientific_assessment,
    unexpected_mismatch_case_ids,
    unexpected_mismatch_details,
)
from misda.benchmark import (
    BenchmarkCase,
    DECLARATION_MISMATCH,
    EXPECTED_DECLARATION_MISMATCH,
)
from misda.benchmarks.cases import make_case5_chain_structure
from misda.benchmarks.mop import mopF_regime_switching


def _two_group_result():
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    data = np.column_stack([x, 2.0 * x, -x, -2.0 * x])
    return misda.discover(data, seed=19, name="two groups")


def _set_support_reasons(result, *reasons):
    result.support = SimpleNamespace(
        status="UNSUPPORTED" if reasons else "SUPPORTED",
        results=(SimpleNamespace(reasons=tuple(reasons)),),
    )


def test_expected_mismatches_are_field_specific_not_blanket_adversarial():
    result = _two_group_result()
    _set_support_reasons(result, "TRANSITIVE_CHAINING")
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
    assert assessment["observed_support_reasons"] == ("TRANSITIVE_CHAINING",)


def test_expected_mismatch_requires_matching_observed_diagnostic():
    result = _two_group_result()
    _set_support_reasons(result)
    case = BenchmarkCase.from_truth(
        "x",
        {
            "name": "diagnostic-gated mismatch",
            "latent_expected": 4,
            "expected_mismatches": {
                "latent_dimension": "TRANSITIVE_CHAINING",
            },
        },
    )

    assessment = case.evaluate(result)
    latent = next(
        check for check in assessment["checks"]
        if check["field"] == "latent_dimension"
    )
    assert latent["status"] == DECLARATION_MISMATCH
    assert latent["reason"] == (
        "DECLARED_DIMENSION_MISMATCH; "
        "EXPECTED_DIAGNOSTIC_NOT_OBSERVED:TRANSITIVE_CHAINING"
    )
    assert assessment["observed_support_reasons"] == ()

    _set_support_reasons(result, "TRANSITIVE_CHAINING")
    assessment = case.evaluate(result)
    latent = next(
        check for check in assessment["checks"]
        if check["field"] == "latent_dimension"
    )
    assert latent["status"] == EXPECTED_DECLARATION_MISMATCH
    assert latent["reason"] == "TRANSITIVE_CHAINING"


def test_case5_declares_only_dimensional_chaining_mismatches():
    _frame, truth = make_case5_chain_structure(N=64, seed=123)

    assert truth["expected_mismatches"] == {
        "latent_dimension": "TRANSITIVE_CHAINING",
        "structural_dimension": "TRANSITIVE_CHAINING",
    }


def test_mopf_declares_only_spectral_structure_mismatches():
    _frame, truth = mopF_regime_switching(N=64, seed=123)

    assert truth["expected_mismatches"] == {
        "latent_dimension": "HIDDEN_SPECTRAL_STRUCTURE",
        "structural_dimension": "HIDDEN_SPECTRAL_STRUCTURE",
        "selected_structural_units": "HIDDEN_SPECTRAL_STRUCTURE",
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

    artifact["cases"][1]["assessment"] = {
        "status": DECLARATION_MISMATCH,
        "checks": [
            {
                "field": "structural_dimension",
                "status": DECLARATION_MISMATCH,
                "observed": 3,
                "expected": 4,
                "reason": "DECLARED_DIMENSION_MISMATCH",
            }
        ],
    }
    assert unexpected_mismatch_case_ids(artifact) == ("case_03",)
    details = unexpected_mismatch_details(artifact)
    assert details == (
        (
            "case_03",
            (
                {
                    "field": "structural_dimension",
                    "observed": 3,
                    "expected": 4,
                    "reason": "DECLARED_DIMENSION_MISMATCH",
                },
            ),
        ),
    )
    with pytest.raises(SystemExit, match="structural_dimension: observed=3, expected=4"):
        enforce_scientific_assessment(artifact)
