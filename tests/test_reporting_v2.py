"""Tests for the stored-result report and structural graph facade."""

import copy
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest

import misda
from misda import _reporting


def _two_group_result(max_evaluated_mis=2):
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    data = np.column_stack([x, 2.0 * x, -x, -2.0 * x])
    return misda.analyze(
        data,
        max_evaluated_mis=max_evaluated_mis,
        seed=19,
        name="two groups",
    )


def test_report_contains_global_summary_rank_counts_and_stored_light_metrics():
    result = _two_group_result()
    report = result.report()

    assert "MISDA static report: two groups" in report
    assert (
        "original=4; latent=1; structural=2; preferred MIS size=2"
    ) in report
    assert "Graph topology: G± components=1; G+ components=2" in report
    assert "MIS evaluation: 2 of 4 normally evaluated; 0 heavy" in report
    assert "Rank counts: rank 1=4" in report
    assert "mis_000 rank=1 size=2" in report
    for metric in (
        "mean_r2",
        "worst_r2",
        "mean_r2_se",
        "worst_r2_se",
        "pareto_retention",
        "pareto_validity",
        "pareto_jaccard",
        "exact_preservation",
    ):
        assert f"{metric} :" in report
    assert "nonlinear_reconstruction" not in report


def test_report_uses_central_metric_metadata_and_one_line_format():
    result = _two_group_result()
    report = result.report()

    assert {
        "mean_r2",
        "worst_r2",
        "pareto_retention",
        "above_null_r2",
        "incidental_reconstruction_rate",
    } <= set(_reporting.METRIC_METADATA)
    mean_lines = [
        line for line in report.splitlines() if line.strip().startswith("mean_r2 :")
    ]
    assert len(mean_lines) == 1
    assert "external R²" in mean_lines[0]
    assert "(average reconstruction quality)" in mean_lines[0]


def test_report_marks_requested_but_undefined_metrics_with_their_reason():
    data = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [-2.0, -2.0, -2.0],
            [-2.0, 2.0, 2.0],
            [2.0, -2.0, 2.0],
            [2.0, 2.0, -2.0],
        ]
    )
    result = misda.analyze(data, max_evaluated_mis=1, seed=5)
    report = result.report()

    assert result.best_mis.size == 3
    assert "mean_r2 : N/A [NO_ELIMINATED_OBJECTIVES]" in report
    assert "worst_r2 : N/A [NO_ELIMINATED_OBJECTIVES]" in report


def test_report_shows_at_most_one_representative_from_first_three_ranks():
    result = copy.deepcopy(_two_group_result(max_evaluated_mis=1))
    result.mis = tuple(
        replace(
            candidate,
            id=f"rank_{index + 1}",
            rank=index + 1,
            evaluation={} if index else candidate.evaluation,
        )
        for index, candidate in enumerate(result.mis)
    )
    result.analysis.rank_counts = {1: 1, 2: 1, 3: 1, 4: 1}
    report = result.report()

    assert "rank_1 rank=1" in report
    assert "rank_2 rank=2" in report
    assert "rank_3 rank=3" in report
    assert "rank_4 rank=4" not in report


def test_heavy_metrics_stay_attached_to_a_nonrepresentative_candidate():
    result = _two_group_result(max_evaluated_mis=1)
    result.mis[2].evaluation["nonlinear_reconstruction"] = {
        "r2_by_objective": {"f2": 0.8},
        "r2_reason_by_objective": {},
        "mean_r2": 0.8,
        "worst_r2": 0.8,
        "reason_by_metric": {},
        "jackknife": {
            "mean_r2_se": 0.1,
            "worst_r2_se": 0.1,
            "reason": None,
        },
        "n_trees": 6,
        "converged": True,
        "convergence_reason": None,
    }
    result.analysis.n_heavy_mis = 1
    report = result.report()

    assert "mis_002 rank=1 size=2" in report
    section = report.split("mis_002 rank=1 size=2", 1)[1]
    assert "nonlinear_reconstruction" in section
    assert "mean_r2 : 0.8000" in section


def test_report_includes_every_scalar_from_requested_null_calibration():
    result = _two_group_result(max_evaluated_mis=1)
    result.mis[0].evaluation["nonlinear_reconstruction"] = {
        "r2_by_objective": {"f2": 0.8},
        "r2_reason_by_objective": {},
        "mean_r2": 0.8,
        "worst_r2": 0.7,
        "reason_by_metric": {},
        "jackknife": {
            "mean_r2_se": 0.1,
            "worst_r2_se": 0.2,
            "reason": None,
        },
        "n_trees": 12,
        "converged": False,
        "convergence_reason": "EXTERNAL_CANCELLATION",
        "null_reference": {
            "mean_null_r2": -0.1,
            "above_null_r2": 0.9,
            "incidental_reconstruction_rate": 0.05,
            "n_permutations": 20,
            "mc_se_mean_null_r2": 0.03,
            "above_null_r2_se": 0.03,
            "incidental_reconstruction_rate_se": 0.01,
            "converged": True,
            "reason": None,
        },
    }
    result.analysis.n_heavy_mis = 1
    report = result.report()

    for metric in (
        "mean_null_r2",
        "above_null_r2",
        "incidental_reconstruction_rate",
        "n_permutations",
        "mc_se_mean_null_r2",
        "above_null_r2_se",
        "incidental_reconstruction_rate_se",
    ):
        assert f"{metric} :" in report
    assert "converged : no [EXTERNAL_CANCELLATION]" in report


def test_report_never_uses_abandoned_claims_or_prints_voluminous_objects():
    report = _two_group_result().report()

    for abandoned in (
        "SES",
        "F_real",
        "F_null",
        "Ideal (Disjoint Cliques)",
        "correlation matrix",
        "r2_by_objective",
        "structural_graph",
    ):
        assert abandoned not in report


def test_report_does_not_run_new_analysis(monkeypatch):
    result = _two_group_result()

    def fail(*args, **kwargs):
        raise AssertionError("report attempted a new calculation")

    monkeypatch.setattr(misda._reconstruction, "evaluate_linear_reconstruction", fail)
    monkeypatch.setattr(misda._pareto, "evaluate_pareto_preservation", fail)

    assert "MISDA static report" in result.report()


def test_graph_plot_and_deprecated_plot_alias_render_equivalent_figures():
    result = _two_group_result()
    direct = result.graph_plot(show=False)
    with pytest.warns(DeprecationWarning, match="graph_plot"):
        alias = result.plot(show=False)

    assert direct.axes[0].get_title() == alias.axes[0].get_title()
    assert len(direct.axes[0].collections) == len(alias.axes[0].collections)
    assert len(direct.axes[0].texts) == len(alias.axes[0].texts) == 4
    plt.close(direct)
    plt.close(alias)
