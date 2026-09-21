# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

import misda
import misda.api as api


def _two_candidate_data(n=24):
    """Return a deterministic graph with exactly two maximal independent sets."""

    x = np.arange(n, dtype=float) - (n - 1) / 2.0
    return np.column_stack([x, 2.0 * x, x * x])


def _fake_linear(data, selected_indices, labels):
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


def test_evaluation_scope_accumulates_across_calls(monkeypatch):
    result = misda.discover(_two_candidate_data(), seed=17)
    assert len(result) == 2
    monkeypatch.setattr(api, "evaluate_linear_reconstruction", _fake_linear)

    ranking = misda.rank(result)
    result.evaluate(metrics=("linear",), candidates=[ranking.mis(0, 0)])
    result.evaluate(metrics=("linear",), candidates=[ranking.mis(0, 1)])

    count, basis = result.evaluation_scope("linear")
    assert count == 2
    assert basis == "explicit MIS sequence"


def test_scope_note_disappears_after_family_is_complete(monkeypatch):
    result = misda.discover(_two_candidate_data(), seed=19)
    assert len(result) == 2
    monkeypatch.setattr(api, "evaluate_linear_reconstruction", _fake_linear)

    ranking = misda.rank(result)
    result.evaluate(metrics=("linear",), candidates=ranking.mis())
    assert "linear metrics were evaluated for" in ranking.report()

    result.evaluate(metrics=("linear",), candidates="all")

    count, _ = result.evaluation_scope("linear")
    assert count == len(result)
    assert "linear metrics were evaluated for" not in ranking.report()


def test_integer_and_index_candidate_selectors_are_rejected():
    result = misda.discover(_two_candidate_data(), seed=23)

    for selector in (1, [0], (0, 1)):
        with pytest.raises(TypeError, match="not part of the alpha API"):
            result.evaluate(metrics=("linear",), candidates=selector)
