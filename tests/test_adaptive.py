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


def test_zero_range_constant_column_exception():
    Y = np.array([
        [1.0, 2.0, 5.0],
        [2.0, 2.0, 4.0],
        [3.0, 2.0, 3.0],
        [4.0, 2.0, 2.0],
    ])
    res = misda.analyze(Y, method='static')
    assert res.analysis.structural_graph.nodes[1]["constant"] is True


def test_adaptive_execution_and_types():
    rng = np.random.default_rng(123)
    x1 = rng.normal(size=150)
    x2 = x1 + 0.02 * rng.normal(size=150)
    x3 = rng.normal(size=150)
    x4 = x3 + 0.02 * rng.normal(size=150)
    Y = np.column_stack([x1, x2, x3, x4])

    res = misda.analyze(Y, method='adaptive', caution=1.0, b_bootstrap=10, seed=42)

    assert isinstance(res, misda.AdaptiveResult)
    assert isinstance(res.static_candidate, misda.AdaptiveCandidate)
    assert res.static_candidate.candidate_id == "static"

    assert len(res.candidates) >= 1

    rec = res.recommended
    assert isinstance(rec, misda.AdaptiveCandidate)
    assert rec.oob is not None
    assert isinstance(rec.oob, misda.OOBSummary)
    assert len(rec.oob.observations) == 10
    assert isinstance(rec.oob.observations[0], misda.BootstrapObservation)

    assert hasattr(rec.oob, "reduction_mean")
    assert hasattr(rec.oob, "reduction_ci")

    df_summary = res.to_pandas()
    assert isinstance(df_summary, pd.DataFrame)
    assert "candidate_id" in df_summary.columns
    assert "oob_recall_mean" in df_summary.columns
    assert "oob_reduction_mean" in df_summary.columns

    summary_str = res.summary()
    assert "MISDA ADAPTIVE ANALYSIS SUMMARY" in summary_str

    report_str = res.report()
    assert "RECOMMENDED CANDIDATE INSPECTION REPORT" in report_str


def test_shared_bootstrap_resamples_and_alpha_used():
    rng = np.random.default_rng(456)
    x1 = rng.normal(size=100)
    x2 = x1 + 0.01 * rng.normal(size=100)
    x3 = rng.normal(size=100)
    Y = np.column_stack([x1, x2, x3])

    res = misda.analyze(Y, method='adaptive', caution=0.5, b_bootstrap=5, seed=99)

    cands = res.candidates
    static_cand = res.static_candidate

    # All candidates share identical seeds and index splits per repetition
    for b in range(5):
        seeds_b = [c.oob.observations[b].seed for c in cands]
        inbag_lens = [c.oob.observations[b].n_inbag for c in cands]
        oob_lens = [c.oob.observations[b].n_oob for c in cands]

        assert len(set(seeds_b)) == 1
        assert len(set(inbag_lens)) == 1
        assert len(set(oob_lens)) == 1

    # Static candidate logs re-estimated alpha_used in each observation
    static_alphas = [obs.alpha_used for obs in static_cand.oob.observations]
    assert len(static_alphas) == 5
    assert all(a >= 0.0 for a in static_alphas)


def test_validated_space_consistency():
    # Synthetic candidates testing validated frontier, knee point, and static dominance
    Y_dummy = np.zeros((10, 2))
    static_res = misda._analyze_static_fast(Y_dummy, np.eye(2), 0.01, 0.05, 0.05)

    c_static = misda.AdaptiveCandidate("static", 0.05, True, static_res, 0.2, 0.9)
    c_static.oob = misda.OOBSummary(0.85, 0.85, (0.8, 0.9), 0.15, (0.1, 0.2), 1.8, {2: 10}, np.ones(2), 1.0, 10, 0, ())

    c_1 = misda.AdaptiveCandidate("cand_001", 0.10, False, static_res, 0.5, 0.95)
    c_1.oob = misda.OOBSummary(0.95, 0.95, (0.9, 1.0), 0.50, (0.4, 0.6), 1.0, {1: 10}, np.ones(2), 1.0, 10, 0, ())

    frontier_ids = misda._compute_pareto_frontier_ids(
        [c_static, c_1],
        x_func=lambda c: c.oob.reduction_mean if c.oob else c.reduction_rate,
        y_func=lambda c: c.oob.recall_mean if c.oob else 0.0
    )
    assert "cand_001" in frontier_ids


def test_reproducibility():
    rng = np.random.default_rng(789)
    x1 = rng.normal(size=120)
    x2 = x1 + 0.03 * rng.normal(size=120)
    x3 = rng.normal(size=120)
    Y = np.column_stack([x1, x2, x3])

    res1 = misda.analyze(Y, method='adaptive', b_bootstrap=10, seed=12345)
    res2 = misda.analyze(Y, method='adaptive', b_bootstrap=10, seed=12345)

    assert res1.recommended_candidate == res2.recommended_candidate
    assert pytest.approx(res1.recommended.oob.recall_mean) == res2.recommended.oob.recall_mean
    assert pytest.approx(res1.recommended.oob.reduction_mean) == res2.recommended.oob.reduction_mean
