# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

import inspect

import numpy as np
import pytest

import misda
import misda.newapi as newapi


def _two_blocks(seed=7, n=40):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    return np.column_stack([
        x,
        2.0 * x + 0.01 * rng.normal(size=n),
        y,
        -3.0 * y + 0.01 * rng.normal(size=n),
    ])


def test_public_surface_is_discover_evaluate_rank():
    assert callable(misda.discover)
    assert callable(misda.evaluate)
    assert callable(misda.rank)
    assert not hasattr(misda, "analyze")
    assert not hasattr(misda, "heavy")


def test_discover_has_no_policy_parameter():
    parameters = inspect.signature(misda.discover).parameters
    assert "policy" not in parameters
    assert "rank_policy" not in parameters


def test_discover_returns_canonical_mis_set():
    result = misda.discover(_two_blocks(), seed=11)

    assert isinstance(result, misda.MISSet)
    assert len(result) >= 1
    assert result.structural_ranking.policy == "structural_coverage"
    assert result.structural_ranking[0] is result[0]
    assert (
        result.structural_ranking.selected_dimension
        == result.structural_ranking.selected.size
    )
    assert result.analysis.structural_dimension == result[0].size


def test_candidate_has_no_contextual_rank_or_artificial_id():
    candidate = misda.discover(_two_blocks(), seed=13)[0]

    assert not hasattr(candidate, "rank")
    assert not hasattr(candidate, "rank_values")
    assert not hasattr(candidate, "id")
    assert candidate.size == len(candidate.indices)


def test_ranking_slice_returns_ranking_view():
    result = misda.discover(_two_blocks(), seed=17)
    ranking = misda.rank(result)
    sliced = ranking[:1]

    assert isinstance(sliced, misda.Ranking)
    assert sliced.mis_set is result
    assert sliced.policy == ranking.policy
    assert len(sliced) == 1
    assert sliced[0] is ranking[0]


def test_rank_does_not_reorder_mis_set():
    result = misda.discover(_two_blocks(), seed=19)
    before = tuple(candidate.indices for candidate in result)

    ranking = misda.rank(result)

    assert tuple(candidate.indices for candidate in result) == before
    assert tuple(candidate.indices for candidate in ranking) == before


def test_linear_evaluation_defaults_to_all_candidates(monkeypatch):
    result = misda.discover(_two_blocks(n=20), seed=23)
    calls = []

    def fake_linear(data, selected_indices, labels):
        calls.append(tuple(selected_indices))
        return {
            "r2_by_objective": {},
            "r2_reason_by_objective": {},
            "mean_r2": 0.5,
            "worst_r2": 0.4,
            "reason_by_metric": {},
            "jackknife": {
                "r2_se_by_objective": {},
                "mean_r2_se": 0.1,
                "worst_r2_se": 0.1,
                "n_replicates": len(data),
                "reason": None,
            },
        }

    monkeypatch.setattr(newapi, "evaluate_linear_reconstruction", fake_linear)
    misda.evaluate(result, metrics=("linear",))

    assert len(calls) == len(result)
    assert result.evaluation_scope("linear")[0] == len(result)
    assert all(candidate.linear is not None for candidate in result)


def test_nonlinear_default_scope_is_one_candidate(monkeypatch):
    result = misda.discover(_two_blocks(n=12), seed=29)
    calls = []

    def fake_nonlinear(data, selected_indices, labels, **kwargs):
        calls.append(tuple(selected_indices))
        return {
            "r2_by_objective": {},
            "r2_reason_by_objective": {},
            "mean_r2": 0.6,
            "worst_r2": 0.5,
            "reason_by_metric": {},
            "jackknife": {
                "r2_se_by_objective": {},
                "mean_r2_se": 0.1,
                "worst_r2_se": 0.1,
                "n_replicates": len(data),
                "reason": None,
            },
            "tree_se_by_objective": {},
            "n_trees": len(data),
            "configuration_counts": {},
            "configuration_by_outer_fold": {},
            "converged": True,
            "cancelled": False,
            "convergence_reason": None,
        }

    monkeypatch.setattr(newapi, "evaluate_nonlinear_reconstruction", fake_nonlinear)
    misda.evaluate(result, metrics=("nonlinear",))

    assert len(calls) == 1
    assert result[0].nonlinear is not None
    assert result.evaluation_scope("nonlinear")[0] == 1
    assert "1 of" in result.report()


def test_combined_expensive_call_uses_one_common_scope(monkeypatch):
    result = misda.discover(_two_blocks(n=12), seed=31)
    linear_calls = []
    nonlinear_calls = []

    def fake_linear(data, selected_indices, labels):
        linear_calls.append(tuple(selected_indices))
        return {
            "r2_by_objective": {},
            "r2_reason_by_objective": {},
            "mean_r2": 0.5,
            "worst_r2": 0.4,
            "reason_by_metric": {},
            "jackknife": {
                "r2_se_by_objective": {},
                "mean_r2_se": 0.1,
                "worst_r2_se": 0.1,
                "n_replicates": len(data),
                "reason": None,
            },
        }

    def fake_nonlinear(data, selected_indices, labels, **kwargs):
        nonlinear_calls.append(tuple(selected_indices))
        return {
            "r2_by_objective": {},
            "r2_reason_by_objective": {},
            "mean_r2": 0.6,
            "worst_r2": 0.5,
            "reason_by_metric": {},
            "jackknife": {
                "r2_se_by_objective": {},
                "mean_r2_se": 0.1,
                "worst_r2_se": 0.1,
                "n_replicates": len(data),
                "reason": None,
            },
            "tree_se_by_objective": {},
            "n_trees": len(data),
            "configuration_counts": {},
            "configuration_by_outer_fold": {},
            "converged": True,
            "cancelled": False,
            "convergence_reason": None,
        }

    monkeypatch.setattr(newapi, "evaluate_linear_reconstruction", fake_linear)
    monkeypatch.setattr(newapi, "evaluate_nonlinear_reconstruction", fake_nonlinear)

    misda.evaluate(result, metrics=("linear", "nonlinear"))

    assert len(linear_calls) == 1
    assert len(nonlinear_calls) == 1


def test_partial_scope_can_follow_ranking_slice(monkeypatch):
    result = misda.discover(_two_blocks(n=20), seed=37)
    ranking = misda.rank(result)
    calls = []

    def fake_linear(data, selected_indices, labels):
        calls.append(tuple(selected_indices))
        return {
            "r2_by_objective": {},
            "r2_reason_by_objective": {},
            "mean_r2": 0.5,
            "worst_r2": 0.4,
            "reason_by_metric": {},
            "jackknife": {
                "r2_se_by_objective": {},
                "mean_r2_se": 0.1,
                "worst_r2_se": 0.1,
                "n_replicates": len(data),
                "reason": None,
            },
        }

    monkeypatch.setattr(newapi, "evaluate_linear_reconstruction", fake_linear)
    misda.evaluate(result, metrics=("linear",), candidates=ranking[:1])

    assert len(calls) == 1
    count, basis = result.evaluation_scope("linear")
    assert count == 1
    assert "Ranking view" in basis


def test_support_exposes_individual_first_rank_candidates():
    result = misda.discover(_two_blocks(), seed=41)

    assert result.support.status in {
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "UNSUPPORTED",
    }
    assert len(result.support.results) == len(result.structural_ranking.groups[0])
    for support_result in result.support.results:
        observed = result.support.for_candidate(support_result.candidate_index)
        assert observed is support_result
        assert observed.status in {"SUPPORTED", "UNSUPPORTED"}
