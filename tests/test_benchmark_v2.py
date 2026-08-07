"""Tests for benchmark declarations kept outside the MISDA result."""

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

import misda
from examples.benchmarks.run_benchmark import run_benchmark


benchmark = importlib.import_module("misda.benchmark")


def _two_group_result():
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    data = np.column_stack([x, 2.0 * x, -x, -2.0 * x])
    return data, misda.analyze(
        data,
        max_evaluated_mis=1,
        seed=19,
        name="two groups",
    )


def _connected_two_representative_result():
    data, result = _two_group_result()
    result.analysis.structural_graph.add_edge(1, 3)
    result.analysis.structural_components = ((0, 1, 2, 3),)
    result.analysis.structural_dimension = 1
    result.analysis.graph_summaries["structural"]["components"] = 1
    return data, result


def _case7(adversarial=False):
    return benchmark.BenchmarkCase(
        case_id="case_07",
        name="two groups",
        latent_dimension=1,
        structural_dimension=2,
        structural_units=(("f1", "f2"), ("f3", "f4")),
        graph_expectations={
            "structural": {"components": 2, "edges": 2},
            "dependence": {"components": 1, "edges": 6},
        },
        adversarial=adversarial,
    )


def test_benchmark_api_is_reexported_from_the_package():
    assert misda.benchmark is benchmark.benchmark
    assert misda.BenchmarkResult is benchmark.BenchmarkResult
    assert misda.BenchmarkCase is benchmark.BenchmarkCase
    assert misda.BenchmarkSuite is benchmark.BenchmarkSuite
    assert misda.compare_results is benchmark.compare_results
    assert misda.serialize_benchmark_result is benchmark.serialize_benchmark_result


def test_public_benchmark_returns_wrapper_and_preserves_truth():
    _data, result = _two_group_result()
    truth = {
        "name": "two groups",
        "latent_expected": 1,
        "structural_expected": 2,
        "blocks_expected": [["f3", "f4"], ["f1", "f2"]],
        "pareto_expected": [0, 1, 2, 3],
        "feature": "two antagonistic families",
        "intuition": "one representative per family",
        "graph_expected": "two positive components",
        "notes": "external declaration",
    }

    observed = misda.benchmark(result, truth)

    assert isinstance(observed, benchmark.BenchmarkResult)
    assert observed.result is result
    assert observed.truth is truth
    assert observed.name == truth["name"]
    assert observed.feature == truth["feature"]
    assert observed.intuition == truth["intuition"]
    assert observed.graph_expected == truth["graph_expected"]
    assert observed.notes == truth["notes"]
    assert observed.blocks_expected == (
        ("f3", "f4"),
        ("f1", "f2"),
    )


def test_public_benchmark_uses_selected_dimension_for_structural_accuracy():
    _data, result = _two_group_result()
    observed = misda.benchmark(
        result,
        {
            "latent_expected": 1,
            "structural_expected": 2,
            "blocks_expected": [["f3", "f4"], ["f1", "f2"]],
        },
    )

    assert observed.selected_dimension == 2
    assert observed.latent_error is None
    assert observed.latent_relative_error is None
    assert observed.latent_exact is None
    assert observed.unavailable_reasons["latent_dimension"] == (
        "the current method does not estimate latent dimension"
    )
    assert observed.structural_error == 0
    assert observed.structural_relative_error == 0.0
    assert observed.structural_dimension_exact
    assert observed.structural_jaccard == 1.0
    assert observed.structural_precision == 1.0
    assert observed.structural_recall == 1.0
    assert observed.structural_f1 == 1.0
    assert observed.structural_partition_exact


def test_report_separates_graph_components_from_selected_dimension():
    _data, result = _connected_two_representative_result()
    observed = misda.benchmark(
        result,
        {
            "latent_expected": 1,
            "structural_expected": 2,
        },
    )

    report = observed.report()

    assert "Original dim.  : 4" in report
    assert "Positive comps.: 1" in report
    assert "Depend. comps. : 1" in report
    assert "Selected dim.  : 2" in report
    assert "Latent         : expected=1, estimated=N/A" in report
    assert "Structural     : expected=2, selected=2, error=0" in report
    assert observed.structural_dimension_exact


def test_structural_metrics_penalize_fusion_and_use_one_to_one_matching():
    _data, result = _two_group_result()
    observed = misda.benchmark(
        result,
        {"blocks_expected": [["f1", "f2", "f3", "f4"]]},
    )

    assert observed.structural_jaccard == pytest.approx(0.25)
    assert observed.structural_precision == 1.0
    assert observed.structural_recall == pytest.approx(1 / 3)
    assert observed.structural_f1 == pytest.approx(0.5)
    assert not observed.structural_partition_exact


def test_pareto_metrics_compare_same_row_indices_without_original_data():
    _data, result = _two_group_result()
    assert result.best_mis.evaluation["pareto_preservation"][
        "reduced_front_indices"
    ] == (0, 1, 2, 3, 4, 5)

    observed = misda.benchmark(
        result,
        {"pareto_expected": [0, 1, 2, 3]},
    )

    assert observed.pareto_precision == pytest.approx(4 / 6)
    assert observed.pareto_recall == 1.0
    assert observed.pareto_f1 == pytest.approx(0.8)
    assert observed.pareto_jaccard == pytest.approx(4 / 6)
    assert observed.pareto_lost == 0
    assert observed.pareto_spurious == 2


def test_missing_references_produce_none_and_reported_na():
    _data, result = _two_group_result()
    observed = misda.benchmark(result, {"name": "undeclared"})

    assert observed.latent_error is None
    assert observed.structural_error is None
    assert observed.structural_jaccard is None
    assert observed.pareto_recall is None
    assert observed.unavailable_reasons == {
        "structural": "blocks_expected was not declared",
        "pareto": "pareto_expected was not declared",
        "latent_dimension": "latent_expected was not declared",
        "structural_dimension": "structural_expected was not declared",
    }
    report = observed.report()
    assert report.startswith("MISDA benchmark report: undeclared")
    assert "Observed analysis" in report
    assert "Structural reconstruction" in report
    assert "Pareto-front preservation" in report
    assert "N/A — blocks_expected was not declared" in report
    assert "N/A — pareto_expected was not declared" in report


def test_truth_reference_validation_is_explicit():
    _data, result = _two_group_result()

    with pytest.raises(TypeError, match="truth must be a mapping"):
        misda.benchmark(result, object())
    with pytest.raises(TypeError, match="latent_expected"):
        misda.benchmark(result, {"latent_expected": 1.5})
    with pytest.raises(ValueError, match="duplicate"):
        misda.benchmark(result, {"pareto_expected": [0, 0]})


def test_case7_external_declarations_pass_without_entering_result():
    _data, result = _two_group_result()
    observed = _case7().evaluate(result)

    assert observed["status"] == "PASS"
    assert observed["dimension_errors"] == {"latent": 0, "structural": 0}
    assert observed["preferred_unit_adequacy"]
    assert observed["preferred_unit_counts"] == [1, 1]
    assert all(check["status"] == "PASS" for check in observed["checks"])
    assert not hasattr(result, "declared")
    assert not hasattr(result.analysis, "latent_expected")


def test_known_adversarial_case_records_expected_change_instead_of_false_pass():
    _data, result = _two_group_result()
    case = benchmark.BenchmarkCase(
        case_id="case_05",
        name="known chain limitation",
        latent_dimension=20,
        structural_dimension=20,
        graph_expectations={
            "structural": {"components": 20},
            "dependence": {"components": 20},
        },
        adversarial=True,
    )

    observed = case.evaluate(result)

    assert observed["status"] == "EXPECTED_CHANGE"
    assert observed["known_adversarial"]
    assert any(
        check["reason"] == "KNOWN_ADVERSARIAL_CASE"
        for check in observed["checks"]
    )


def test_ambiguous_units_are_skipped_instead_of_becoming_method_metrics():
    _data, result = _two_group_result()
    case = benchmark.BenchmarkCase(
        case_id="ambiguous",
        name="ambiguous",
        latent_dimension=1,
        structural_dimension=2,
        structural_units=(("f1",), ("f2",), ("f3", "f4")),
    )

    observed = case.evaluate(result)

    unit_check = next(
        check
        for check in observed["checks"]
        if check["field"] == "preferred_structural_units"
    )
    assert unit_check["status"] == "SKIP"
    assert unit_check["reason"] == "DECLARATION_NOT_UNAMBIGUOUS"
    assert observed["preferred_unit_adequacy"] is None


def test_suite_aggregates_ready_results_and_rejects_missing_cases():
    _data, result = _two_group_result()
    suite = benchmark.BenchmarkSuite("small", (_case7(),))

    observed = suite.evaluate({"case_07": result})

    assert observed["suite"] == "small"
    assert observed["status"] == "PASS"
    assert observed["cases"][0]["case_id"] == "case_07"
    with pytest.raises(KeyError, match="case_07"):
        suite.evaluate({})


def test_serializer_uses_only_a_completed_result_and_external_case():
    data, result = _two_group_result()
    observed = benchmark.serialize_benchmark_result(
        _case7(),
        result,
        data,
        seed=19,
    )

    assert observed["declared"] == {
        "latent_dimension": 1,
        "structural_dimension": 2,
    }
    assert observed["estimated"] == {
        "latent_dimension": 1,
        "structural_dimension": 2,
        "preferred_mis_size": 2,
    }
    assert observed["graphs"] == result.analysis.graph_summaries
    assert observed["linear_reconstruction"] is (
        result.best_mis.evaluation["linear_reconstruction"]
    )
    assert observed["pareto_preservation"] is (
        result.best_mis.evaluation["pareto_preservation"]
    )
    assert len(observed["input_sha256"]) == 64
    assert observed["assessment"]["status"] == "PASS"


def _comparison_artifact():
    return {
        "input_sha256": "same",
        "declared": {"latent_dimension": 2, "structural_dimension": 2},
        "estimated": {"latent_dimension": 3, "structural_dimension": 2},
        "graphs": {"structural": {"edges": 4, "components": 2}},
        "preferred_indices": [0, 2],
        "n_mis": 4,
        "rank_counts": {"1": 4},
        "linear_reconstruction": {"mean_r2": 0.7, "worst_r2": 0.5},
        "pareto_preservation": {
            "pareto_retention": 0.8,
            "pareto_validity": 0.9,
            "pareto_jaccard": 0.7,
        },
    }


def test_compare_results_distinguishes_pass_improvement_and_regression():
    baseline = _comparison_artifact()
    assert benchmark.compare_results(baseline, copy.deepcopy(baseline))["status"] == "PASS"

    improved = copy.deepcopy(baseline)
    improved["estimated"]["latent_dimension"] = 2
    improved["linear_reconstruction"]["mean_r2"] = 0.8
    assert benchmark.compare_results(baseline, improved)["status"] == "IMPROVED"

    regressed = copy.deepcopy(baseline)
    regressed["input_sha256"] = "different"
    assert benchmark.compare_results(baseline, regressed)["status"] == "REGRESSION"


def test_compare_results_requires_a_versioned_reason_for_expected_change():
    baseline = _comparison_artifact()
    candidate = copy.deepcopy(baseline)
    candidate["preferred_indices"] = [1, 3]

    without_rule = benchmark.compare_results(baseline, candidate)
    with_rule = benchmark.compare_results(
        baseline,
        candidate,
        expected_changes={
            "preferred_indices": "SIGNED_GRAPH_METHOD_CHANGE",
        },
    )

    assert without_rule["status"] == "REGRESSION"
    assert with_rule["status"] == "EXPECTED_CHANGE"
    changed = next(
        check for check in with_rule["checks"] if check["field"] == "preferred_indices"
    )
    assert changed["reason"] == "SIGNED_GRAPH_METHOD_CHANGE"


@pytest.mark.slow
def test_canonical_static_battery_has_no_regression_against_frozen_baseline():
    manifest = json.loads(
        (Path(__file__).parent / "baselines" / "refactor_preimplementation.json")
        .read_text(encoding="utf-8")
    )
    baseline = manifest["artifacts"]["benchmark"]["cases"]
    candidate = run_benchmark(n=1000, seed=123)["cases"]

    comparisons = {
        current["case_id"]: benchmark.compare_results(previous, current)
        for previous, current in zip(baseline, candidate)
    }

    assert set(comparisons) == {case["case_id"] for case in baseline}
    assert all(
        comparison["status"] == "PASS"
        for comparison in comparisons.values()
    )
    assert next(
        case for case in candidate if case["case_id"] == "case_05"
    )["assessment"]["status"] == "EXPECTED_CHANGE"
    case7 = next(case for case in candidate if case["case_id"] == "case_07")
    assert case7["estimated"] == {
        "latent_dimension": 1,
        "structural_dimension": 2,
        "preferred_mis_size": 2,
    }
    json.dumps(candidate, allow_nan=False)
