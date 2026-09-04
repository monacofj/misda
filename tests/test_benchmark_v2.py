"""Tests for benchmark declarations kept outside the discovered MISSet."""

import copy
import importlib

import numpy as np
import pytest

import misda


benchmark_module = importlib.import_module("misda.benchmark")


def _two_group_result():
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    data = np.column_stack([x, 2.0 * x, -x, -2.0 * x])
    result = misda.discover(data, seed=19, name="two groups")
    misda.evaluate(result, metrics=("linear", "pareto"), candidates=1)
    return data, result


def _case7(adversarial=False):
    return benchmark_module.BenchmarkCase(
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


def test_benchmark_api_is_reexported_from_package():
    assert misda.benchmark is benchmark_module.benchmark
    assert misda.BenchmarkResult is benchmark_module.BenchmarkResult
    assert misda.BenchmarkCase is benchmark_module.BenchmarkCase
    assert misda.BenchmarkSuite is benchmark_module.BenchmarkSuite


def test_public_benchmark_preserves_external_truth():
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

    assert isinstance(observed, benchmark_module.BenchmarkResult)
    assert observed.result is result
    assert observed.truth is truth
    assert observed.name == truth["name"]
    assert observed.feature == truth["feature"]
    assert observed.blocks_expected == (("f3", "f4"), ("f1", "f2"))
    assert not hasattr(result.analysis, "latent_expected")
    assert not hasattr(result.analysis, "structural_expected")


def test_benchmark_compares_dimensions_and_partition_to_graph_outputs():
    _data, result = _two_group_result()
    observed = misda.benchmark(
        result,
        {
            "latent_expected": 1,
            "structural_expected": 2,
            "blocks_expected": [["f3", "f4"], ["f1", "f2"]],
        },
    )

    assert observed.selected_dimension == result.structural_ranking.selected_dimension == 2
    assert observed.latent_error == 0
    assert observed.structural_error == 0
    assert observed.latent_exact
    assert observed.structural_dimension_exact
    assert observed.structural_jaccard == 1.0
    assert observed.structural_precision == 1.0
    assert observed.structural_recall == 1.0
    assert observed.structural_partition_exact


def test_benchmark_pareto_uses_stored_selected_candidate_evidence_only():
    _data, result = _two_group_result()
    selected = result.structural_ranking.selected
    assert selected.pareto.reduced_front_indices == (0, 1, 2, 3, 4, 5)

    observed = misda.benchmark(result, {"pareto_expected": [0, 1, 2, 3]})

    assert observed.pareto_precision == pytest.approx(4 / 6)
    assert observed.pareto_recall == 1.0
    assert observed.pareto_jaccard == pytest.approx(4 / 6)
    assert observed.pareto_lost == 0
    assert observed.pareto_spurious == 2


def test_benchmark_does_not_trigger_missing_pareto_evaluation():
    data = np.arange(24.0).reshape(6, 4)
    result = misda.discover(data, seed=31)

    observed = misda.benchmark(result, {"pareto_expected": [0]})

    assert observed.pareto_recall is None
    assert observed.unavailable_reasons["pareto"] == (
        "the selected MIS Pareto frontier was not evaluated"
    )
    assert result.structural_ranking.selected.pareto is None


def test_missing_declarations_are_explicit_na():
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
    assert "N/A — blocks_expected was not declared" in report
    assert "N/A — pareto_expected was not declared" in report


def test_truth_validation_is_explicit():
    _data, result = _two_group_result()
    with pytest.raises(TypeError, match="truth must be a mapping"):
        misda.benchmark(result, object())
    with pytest.raises(TypeError, match="latent_expected"):
        misda.benchmark(result, {"latent_expected": 1.5})
    with pytest.raises(ValueError, match="duplicate"):
        misda.benchmark(result, {"pareto_expected": [0, 0]})


def test_case_declarations_match_without_entering_discovery():
    _data, result = _two_group_result()
    observed = _case7().evaluate(result)

    assert observed["status"] == benchmark_module.DECLARATION_MATCH
    assert observed["dimension_errors"] == {"latent": 0, "structural": 0}
    assert observed["selected_unit_adequacy"]
    assert observed["selected_unit_counts"] == [1, 1]


def test_from_truth_does_not_infer_component_counts_from_dimensions():
    case = benchmark_module.BenchmarkCase.from_truth(
        "x",
        {
            "name": "connected chain",
            "latent_expected": 3,
            "structural_expected": 3,
        },
    )

    assert case.graph_expectations == {}


def test_suite_aggregates_ready_results_and_rejects_missing_cases():
    _data, result = _two_group_result()
    suite = benchmark_module.BenchmarkSuite("small", (_case7(),))

    observed = suite.evaluate({"case_07": result})

    assert observed["status"] == benchmark_module.DECLARATION_MATCH
    with pytest.raises(KeyError, match="case_07"):
        suite.evaluate({})


def test_serializer_records_ranking_not_intrinsic_candidate_rank():
    data, result = _two_group_result()
    observed = benchmark_module.serialize_benchmark_result(
        _case7(), result, data, seed=19
    )

    assert observed["estimated"] == {
        "latent_dimension": 1,
        "structural_dimension": 2,
        "selected_dimension": 2,
    }
    assert observed["ranking_policy"] == "structural_coverage"
    assert observed["selected_indices"] == list(result.structural_ranking.selected.indices)
    assert "preferred_mis_size" not in observed["estimated"]
    assert "rank_counts" not in observed
    assert len(observed["input_sha256"]) == 64


def _comparison_artifact():
    return {
        "input_sha256": "same",
        "declared": {"latent_dimension": 2, "structural_dimension": 2},
        "estimated": {
            "latent_dimension": 3,
            "structural_dimension": 2,
            "selected_dimension": 2,
        },
        "graphs": {"structural": {"edges": 4, "components": 2}},
        "selected_indices": [0, 2],
        "n_mis": 4,
        "ranking_groups": [[0, 1], [2, 3]],
        "linear_reconstruction": {"mean_r2": 0.7, "worst_r2": 0.5},
        "pareto_preservation": {
            "retention": 0.8,
            "validity": 0.9,
            "jaccard": 0.7,
        },
    }


def test_compare_results_distinguishes_pass_improvement_and_regression():
    baseline = _comparison_artifact()
    assert benchmark_module.compare_results(baseline, copy.deepcopy(baseline))["status"] == "PASS"

    improved = copy.deepcopy(baseline)
    improved["estimated"]["latent_dimension"] = 2
    improved["linear_reconstruction"]["mean_r2"] = 0.8
    assert benchmark_module.compare_results(baseline, improved)["status"] == "IMPROVED"

    regressed = copy.deepcopy(baseline)
    regressed["input_sha256"] = "different"
    assert benchmark_module.compare_results(baseline, regressed)["status"] == "REGRESSION"
