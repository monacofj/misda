"""Tests for clean -> observed -> reduced Pareto decomposition."""

import importlib

import numpy as np
import pytest

import misda


benchmark_module = importlib.import_module("misda.benchmark")


def _tradeoff_data(n=12):
    x = np.arange(float(n))
    return np.column_stack([x, (n - 1) - x])


def _misda_block(report):
    lines = report.splitlines()
    start = lines.index("MISDA measures (data-derived)") + 2
    end = lines.index("Benchmark validation (requires declared truth)")
    return "\n".join(lines[start:end]).rstrip()


def test_public_benchmark_module_uses_observation_aware_wrapper():
    assert misda.benchmark is benchmark_module.benchmark
    assert misda.compile_benchmark_summary is benchmark_module.compile_benchmark_summary


def test_clean_observation_layer_is_identity_when_y_equals_z():
    z = _tradeoff_data()
    result = misda.discover(z, seed=17)
    misda.evaluate(result, metrics=("pareto",), candidates=1)
    truth = {"name": "clean", "pareto_expected": list(range(len(z)))}

    observed = misda.benchmark(result, truth)

    assert observed.observed_pareto_indices == tuple(range(len(z)))
    assert observed.observation_pareto_precision == 1.0
    assert observed.observation_pareto_recall == 1.0
    assert observed.observation_pareto_f1 == 1.0
    assert observed.observation_pareto_jaccard == 1.0
    assert observed.observation_pareto_lost == 0
    assert observed.observation_pareto_spurious == 0
    assert observed.observation_pareto_exact

    native_report = result.report()
    report = observed.report()
    assert _misda_block(report) == native_report
    assert "Pareto stability (observed Y only):" in native_report
    assert "Observed front:" in native_report
    assert "Additive epsilon+:" in native_report
    assert "Dominance margin:" in native_report
    assert "Pareto observation agreement (P_Y vs P_Z)" in report
    assert "clean=12, observed=12" in report
    assert "jaccard=1.0000, exact=yes" in report
    assert "Pareto basis   : reduced P_R vs clean truth P_Z" in report
    assert "Pareto basis   : reduced P_R vs observed full P_Y" not in report


def test_noisy_observation_layer_separates_observation_from_reduction():
    z = _tradeoff_data()
    y = z.copy()
    y[5] = (20.0, 20.0)  # observation alone makes the clean point dominated

    result = misda.discover(y, seed=17)
    misda.evaluate(result, metrics=("pareto",), candidates=1)
    truth = {"name": "observed", "pareto_expected": list(range(len(z)))}

    observed = misda.benchmark(result, truth)

    expected_observed = tuple(index for index in range(len(z)) if index != 5)
    assert observed.observed_pareto_indices == expected_observed
    assert observed.observation_pareto_precision == 1.0
    assert observed.observation_pareto_recall == pytest.approx(11 / 12)
    assert observed.observation_pareto_jaccard == pytest.approx(11 / 12)
    assert observed.observation_pareto_lost == 1
    assert observed.observation_pareto_spurious == 0
    assert not observed.observation_pareto_exact

    selected = result.structural_ranking.selected
    assert selected is not None and selected.pareto is not None
    # Existing candidate Pareto evidence remains P_R vs P_Y.
    assert observed.pareto_jaccard is not None  # existing P_R vs P_Z end-to-end field


def test_summary_exposes_all_three_pareto_stages_without_changing_old_column():
    z = _tradeoff_data()
    result = misda.discover(z, seed=17)
    misda.evaluate(result, metrics=("pareto",), candidates=1)
    truth = {"pareto_expected": list(range(len(z)))}

    summary = misda.compile_benchmark_summary(
        {"clean": {"result_obj": result, "truth": truth}}
    )

    assert summary.loc[0, "ParetoObservationJaccard"] == 1.0
    assert summary.loc[0, "ParetoEndToEndJaccard"] == summary.loc[0, "ParetoJaccard"]
    assert summary.loc[0, "ParetoReductionJaccard"] == summary.loc[0, "ParetoJaccard"]
    assert summary.loc[0, "ParetoTruthSize"] == len(z)
    assert summary.loc[0, "ParetoObservedSize"] == len(z)
    assert summary.loc[0, "ParetoReducedSize"] == result.structural_ranking.selected.pareto.reduced_front_size
    assert summary.loc[0, "ParetoObservedFraction"] == 1.0
    assert summary.loc[0, "ParetoAdditiveEpsilon"] >= 0.0
    assert summary.loc[0, "ParetoDominanceMarginMin"] >= 0.0
    assert summary.loc[0, "ParetoDominanceMarginMedian"] >= 0.0
    assert summary.loc[0, "ParetoDominanceMarginMax"] >= 0.0
