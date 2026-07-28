# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest
import misda


def test_evaluate_pareto_raw():
    Y = np.array([
        [1.0, 5.0],
        [2.0, 4.0],
        [3.0, 3.0],
        [4.0, 2.0],
        [5.0, 1.0],
    ])
    p_full, r_full = misda.evaluate_pareto_raw(Y, [0, 1])
    assert p_full == 1.0
    assert r_full == 1.0

    p_sub, r_sub = misda.evaluate_pareto_raw(Y, [0])
    assert p_sub == 1.0
    assert pytest.approx(r_sub) == 0.2


def test_adaptive_execution_and_types():
    rng = np.random.default_rng(123)
    x1 = rng.normal(size=150)
    x2 = x1 + 0.02 * rng.normal(size=150)
    x3 = rng.normal(size=150)
    x4 = x3 + 0.02 * rng.normal(size=150)
    Y = np.column_stack([x1, x2, x3, x4])

    res = misda.analyze(Y, method='adaptive', b_bootstrap=10, seed=42)

    assert isinstance(res, misda.AdaptiveResult)
    assert isinstance(res.static_candidate, misda.AdaptiveCandidate)
    assert len(res.candidates) >= 1

    rec = res.recommended
    assert isinstance(rec, misda.AdaptiveCandidate)
    assert rec.oob is not None
    assert isinstance(rec.oob, misda.OOBSummary)
    assert len(rec.oob.observations) == 10
    assert isinstance(rec.oob.observations[0], misda.BootstrapObservation)

    df_summary = res.to_pandas()
    assert isinstance(df_summary, pd.DataFrame)
    assert "candidate_id" in df_summary.columns
    assert "oob_recall_mean" in df_summary.columns

    summary_str = res.summary()
    assert "MISDA ADAPTIVE ANALYSIS SUMMARY" in summary_str
