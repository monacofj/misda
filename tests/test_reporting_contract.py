"""Non-regression contract for the public MISDA report.

This file intentionally protects user-visible reporting capability.  Refactors
may change implementation and formatting, but deleting one of these evidence
families requires an explicit methodological/API decision rather than silently
shrinking ``MISSet.report()``.
"""

import numpy as np

import misda
import misda.api as api


def _evaluated_result():
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    data = np.column_stack([x, 2.0 * x, -x, -2.0 * x])
    result = misda.discover(data, seed=19, name="report contract")
    misda.evaluate(result, metrics=("linear", "pareto"), candidates="all")
    return result


def test_report_preserves_rich_public_audit_contract():
    result = _evaluated_result()
    report = result.report()

    for section in (
        "MISDA report: report contract",
        "Dimensions:",
        "Graph topology:",
        "Threshold calibration:",
        "Null envelope:",
        "Structural ranking:",
        "Tie groups",
        "Dimensional support:",
        "Reduction assessment:",
        "use-status annotation",
        "Transitivity:",
        "Spectral:",
        "Evaluation scope:",
        "Candidates:",
        "structural",
        "linear_reconstruction",
        "pareto_preservation",
        "Pareto stability (observed Y only):",
    ):
        assert section in report

    for metric in (
        "mean_r2",
        "worst_r2",
        "mean_r2_se",
        "worst_r2_se",
        "pareto_retention",
        "pareto_validity",
        "pareto_jaccard",
        "full_front_size",
        "reduced_front_size",
        "intersection_size",
        "union_size",
        "exact_preservation",
        "Front loss",
        "Population impact",
    ):
        assert f"{metric}" in report

    assert "external R²" in report
    assert "how well eliminated objectives reconstruct on average" in report
    assert "G± dependence" in report
    assert "G+ structural" in report
    assert "candidate[0]" in report


def test_report_preserves_nonlinear_and_null_reference_evidence():
    result = _evaluated_result()
    jackknife = misda.JackknifeMetrics(
        r2_se_by_objective={},
        mean_r2_se=0.10,
        worst_r2_se=0.20,
        n_replicates=6,
        reason=None,
    )
    null = misda.NullReferenceMetrics(
        mean_null_r2=-0.10,
        above_null_r2=0.90,
        incidental_reconstruction_rate=0.05,
        n_permutations=20,
        mc_se_mean_null_r2=0.03,
        above_null_r2_se=0.03,
        incidental_reconstruction_rate_se=0.01,
        converged=True,
        cancelled=False,
        reason=None,
    )
    nonlinear = misda.NonlinearMetrics(
        r2_by_objective={},
        r2_reason_by_objective={},
        mean_r2=0.80,
        worst_r2=0.70,
        reason_by_metric={},
        jackknife=jackknife,
        tree_se_by_objective={},
        n_trees=12,
        configuration_counts={},
        configuration_by_outer_fold={},
        converged=True,
        cancelled=False,
        convergence_reason=None,
        null_reference=null,
    )
    object.__setattr__(result[0], "nonlinear", nonlinear)
    result._evaluation_scopes["nonlinear"] = (1, "report contract fixture")

    report = result.report()

    assert "nonlinear_reconstruction" in report
    assert "null_reference" in report
    for metric in (
        "n_trees",
        "mean_null_r2",
        "above_null_r2",
        "incidental_reconstruction_rate",
        "n_permutations",
        "mc_se_mean_null_r2",
        "above_null_r2_se",
        "incidental_reconstruction_rate_se",
    ):
        assert metric in report


def test_report_compacts_first_rank_dimensional_support():
    x = np.linspace(-1.0, 1.0, 20)
    data = np.column_stack([x for _ in range(20)])
    result = misda.discover(data, seed=19, name="support aggregation")
    report = result.report()

    start = report.index("Dimensional support:")
    end = report.index("Evaluation scope:")
    support_block = report[start:end]

    assert len(result.support.results) == 20
    assert "First-rank group : 20 candidates" in support_block
    assert "Supported        : 20/20" in support_block
    assert "Unsupported      : 0/20" in support_block
    assert "Transitivity:" in support_block
    assert "Spectral:" in support_block
    assert "Null reference:" in support_block
    assert "candidate[" not in support_block


def test_report_exposes_front_loss_and_population_impact():
    result = _evaluated_result()
    metrics = misda.ParetoMetrics(
        retention=1.0 / 3.0,
        validity=1.0,
        jaccard=1.0 / 3.0,
        full_front_size=3,
        reduced_front_size=1,
        intersection_size=1,
        union_size=3,
        exact_preservation=False,
        reduced_front_indices=(0,),
    )
    object.__setattr__(result[0], "pareto", metrics)

    report = result.report()

    lines = report.splitlines()
    assert any("Original front" in line and ": 3/6" in line for line in lines)
    assert any("Preserved front" in line and ": 1/3 (0.3333)" in line for line in lines)
    assert any("Front loss" in line and ": 2/3 (0.6667)" in line for line in lines)
    assert any("Population impact" in line and ": 2/6 (0.3333)" in line for line in lines)


def test_report_never_runs_hidden_scientific_evaluation(monkeypatch):
    result = _evaluated_result()

    def fail(*args, **kwargs):
        raise AssertionError("report attempted a new scientific calculation")

    monkeypatch.setattr(api, "evaluate_linear_reconstruction", fail)
    monkeypatch.setattr(api, "evaluate_pareto_preservation", fail)
    monkeypatch.setattr(api, "evaluate_nonlinear_reconstruction", fail)
    monkeypatch.setattr(api, "evaluate_null_reconstruction", fail)

    report = result.report()
    assert "MISDA report:" in report


def test_benchmark_embeds_the_native_report_verbatim():
    result = _evaluated_result()
    native = result.report()
    report = misda.benchmark(
        result,
        {"name": "contract", "latent_expected": 1, "structural_expected": 2},
    ).report()
    lines = report.splitlines()
    start = lines.index("MISDA measures (data-derived)") + 2
    end = lines.index("Benchmark validation (requires declared truth)")
    embedded = "\n".join(lines[start:end]).rstrip()

    assert embedded == native


def test_ranking_report_is_complete_and_preserves_default_report_contract():
    result = _evaluated_result()
    ranking = misda.rank(result)

    report = ranking.report()

    assert report == result.report()
    for section in (
        "Dimensions:",
        "Graph topology:",
        "Threshold calibration:",
        "Null envelope:",
        "Structural ranking:",
        "Dimensional support:",
        "Evaluation scope:",
        "Candidates:",
        "Pareto stability (observed Y only):",
    ):
        assert section in report


def test_mis_report_focuses_on_intrinsic_stored_evidence():
    result = _evaluated_result()
    ranking = misda.rank(result)

    report = ranking.mis().report()

    assert "MIS report: report contract" in report
    assert "Dimension" in report
    assert "Objectives" in report
    assert "Structural:" in report
    assert "Linear reconstruction:" in report
    assert "Pareto preservation:" in report
    assert "Front loss" in report
    assert "Population impact" in report
    assert "Structural ranking:" not in report
    assert "candidate[" not in report


def test_ranking_and_mis_reports_never_run_hidden_evaluation(monkeypatch):
    result = _evaluated_result()
    ranking = misda.rank(result)

    def fail(*args, **kwargs):
        raise AssertionError("report attempted a new scientific calculation")

    monkeypatch.setattr(api, "evaluate_linear_reconstruction", fail)
    monkeypatch.setattr(api, "evaluate_pareto_preservation", fail)
    monkeypatch.setattr(api, "evaluate_nonlinear_reconstruction", fail)
    monkeypatch.setattr(api, "evaluate_null_reconstruction", fail)

    assert "MISDA report:" in ranking.report()
    assert "MIS report:" in ranking.mis().report()


def test_report_explains_top_level_fields_without_losing_values():
    result = _evaluated_result()
    report = result.report()

    expected_explanations = (
        "observed objective dimension",
        "how many objectives we started with",
        "independence number of G±",
        "how many signed-dependence degrees remain",
        "independence number of G+",
        "how many positive-redundancy units remain",
        "top-ranked MIS size",
        "components describe connectivity, not dimension",
        "structure-onset threshold",
        "permutation-null endpoint",
        "active dependence threshold",
        "onset→null interpolation",
        "onset/null separation status",
        "null-envelope completion",
        "completed null permutations",
        "MIS ordering rule",
        "larger MISs first, then broader span",
        "ranking ties",
    )
    for explanation in expected_explanations:
        assert explanation in report

    # Existing values and labels remain present; explanations are additive.
    assert f"Original       : {result.analysis.original_dimension}" in report
    assert f"Latent         : {result.analysis.latent_dimension}" in report
    assert f"Structural     : {result.analysis.structural_dimension}" in report
    assert "G± dependence" in report
    assert "G+ structural" in report
    assert "alpha_onset" in report
    assert "alpha_null" in report
    assert "aggressiveness" in report


def test_report_explains_support_and_intrinsic_mis_metrics():
    result = _evaluated_result()
    ranking = misda.rank(result)

    complete = ranking.report()
    for explanation in (
        "tested top-rank set",
        "leaders with no diagnostic contradiction",
        "observed transitivity",
        "expected value after breaking column association",
        "positive values flag transitive chaining",
        "next observed eigenvalue",
        "positive values flag hidden spectral structure",
        "shared null replicates",
    ):
        assert explanation in complete

    intrinsic = ranking.mis().report()
    for explanation in (
        "external-neighbor count",
        "fraction of eliminated objectives adjacent to this MIS",
        "retained/eliminated edge count",
        "average eliminated neighbors per retained objective",
        "should be zero because an MIS is independent",
        "how many observations are nondominated before reduction",
        "how much of the original front disappears",
        "how much of the whole sample is affected",
    ):
        assert explanation in intrinsic


def test_default_ranking_report_remains_exactly_the_complete_mis_set_report():
    result = _evaluated_result()

    assert misda.rank(result).report() == result.report()


def test_metric_annotations_keep_technical_and_intuitive_text_on_one_line():
    result = _evaluated_result()
    report_lines = result.report().splitlines()

    line = next(
        line for line in report_lines
        if line.lstrip().startswith("mean_r2")
    )

    assert "— mean external R²" in line
    assert "(how well eliminated objectives reconstruct on average)" in line
    assert not any(line.lstrip().startswith("— ") for line in report_lines)


def test_top_level_annotations_keep_technical_and_intuitive_text_on_one_line():
    result = _evaluated_result()
    report_lines = result.report().splitlines()

    original = next(line for line in report_lines if line.startswith("  Original"))
    latent = next(line for line in report_lines if line.startswith("  Latent"))

    assert "— observed objective dimension" in original
    assert "(how many objectives we started with)" in original
    assert "— independence number of G±" in latent
    assert "(how many signed-dependence degrees remain)" in latent
    assert not any(line.lstrip().startswith("— ") for line in report_lines)


def test_wrapped_report_preserves_legacy_pareto_stability_labels():
    result = _evaluated_result()
    report = result.report()

    assert "Observed front:" in report
    assert "Dominance margin:" in report
    assert "Additive epsilon+:" in report


def test_dominance_ranking_report_exposes_metric_and_trust_annotation():
    result = _evaluated_result()
    result.evaluate(metrics=("dominance",), candidates="all")
    ranking = misda.rank(result, policy=misda.DOMINANCE_PRESERVATION)

    report = ranking.report()

    assert "Policy" in report and "dominance_preservation" in report
    assert "Reduction assessment:" in report
    assert ranking.assessment.status in report
    assert "dominance_preservation" in report
    assert "new_dominance_rate" in report
    assert "new observed dominance fraction" in report
