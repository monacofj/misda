"""Contracts for portable floating-point regression gates."""

import copy
import importlib

from misda import _tolerances


benchmark = importlib.import_module("misda.benchmark")


def _comparison_artifact():
    return {
        "input_sha256": "same",
        "declared": {"latent_dimension": 2, "structural_dimension": 2},
        "estimated": {"latent_dimension": 2, "structural_dimension": 2},
        "graphs": {"structural": {"edges": 4, "components": 2}},
        "selected_indices": [0, 2],
        "n_mis": 4,
        "ranking_groups": [[0, 1, 2, 3]],
        "linear_reconstruction": {"mean_r2": 0.7, "worst_r2": 0.5},
        "pareto_preservation": {
            "retention": 0.8,
            "validity": 0.9,
            "jaccard": 0.7,
        },
    }


def test_gate_tolerance_policy_is_explicit_and_versioned():
    assert _tolerances.GATE_TOLERANCE_VERSION == 1
    assert _tolerances.GATE_RTOL == 0.0
    assert _tolerances.GATE_ATOL == 1e-12


def test_gate_tolerance_accepts_only_absolute_last_bit_scale_variation():
    assert _tolerances.gate_isclose(1.0, 1.0 + 5e-13)
    assert not _tolerances.gate_isclose(1.0, 1.0 + 2e-12)
    assert not _tolerances.gate_isclose(1e9, 1e9 + 1e-4)


def test_gate_accepts_float_variation_within_tolerance():
    baseline = _comparison_artifact()
    candidate = copy.deepcopy(baseline)
    candidate["linear_reconstruction"]["mean_r2"] -= 5e-13

    observed = benchmark.compare_results(baseline, candidate)

    assert observed["status"] == "PASS"
    check = next(
        item
        for item in observed["checks"]
        if item["field"] == "linear_reconstruction.mean_r2"
    )
    assert check["status"] == "PASS"


def test_gate_rejects_float_degradation_above_tolerance():
    baseline = _comparison_artifact()
    candidate = copy.deepcopy(baseline)
    candidate["linear_reconstruction"]["mean_r2"] -= 2e-12

    observed = benchmark.compare_results(baseline, candidate)

    assert observed["status"] == "REGRESSION"
    check = next(
        item
        for item in observed["checks"]
        if item["field"] == "linear_reconstruction.mean_r2"
    )
    assert check["status"] == "REGRESSION"


def test_discrete_gate_fields_remain_exact():
    baseline = _comparison_artifact()
    candidate = copy.deepcopy(baseline)
    candidate["selected_indices"] = [0, 3]

    observed = benchmark.compare_results(baseline, candidate)

    assert observed["status"] == "REGRESSION"
    check = next(
        item for item in observed["checks"] if item["field"] == "selected_indices"
    )
    assert check["status"] == "REGRESSION"
