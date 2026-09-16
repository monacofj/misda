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
        "transitivity:",
        "spectral    :",
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
    ):
        assert f"{metric}" in report

    assert "external R²" in report
    assert "average reconstruction quality" in report
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
