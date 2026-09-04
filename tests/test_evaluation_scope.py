# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

import misda
import misda.api as api


def _independent_data(seed=123, n=24, m=4):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, m))


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
    result = misda.discover(_independent_data(), seed=17)
    assert len(result) >= 2
    monkeypatch.setattr(api, "evaluate_linear_reconstruction", _fake_linear)

    misda.evaluate(result, metrics=("linear",), candidates=[0])
    misda.evaluate(result, metrics=("linear",), candidates=[1])

    count, basis = result.evaluation_scope("linear")
    assert count == 2
    assert basis == "explicit candidate indices"


def test_scope_note_disappears_after_family_is_complete(monkeypatch):
    result = misda.discover(_independent_data(seed=321), seed=19)
    monkeypatch.setattr(api, "evaluate_linear_reconstruction", _fake_linear)

    misda.evaluate(result, metrics=("linear",), candidates=1)
    assert "linear metrics were evaluated for" in result.report()

    misda.evaluate(result, metrics=("linear",), candidates="all")

    count, _ = result.evaluation_scope("linear")
    assert count == len(result)
    assert "linear metrics were evaluated for" not in result.report()
