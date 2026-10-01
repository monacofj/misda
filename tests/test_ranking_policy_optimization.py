# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

import misda
import moeabench as mb
from benchmarks import run_ranking_policy_optimization as probe


def test_objective_projection_delegates_to_original_mop():
    mop = mb.mops.DTLZ5(M=5)
    rng = np.random.default_rng(123)
    X = rng.uniform(
        np.asarray(mop.xl, dtype=float),
        np.asarray(mop.xu, dtype=float),
        size=(8, mop.N),
    )
    projected = probe.ObjectiveProjectionMOP(mop, (1, 4))

    expected = np.asarray(mop.evaluation(X)["F"], dtype=float)[:, (1, 4)]
    observed = np.asarray(projected.evaluation(X)["F"], dtype=float)

    np.testing.assert_allclose(observed, expected)


def test_validate_checkpoints_rejects_out_of_budget_values():
    assert probe._validate_checkpoints((5, 2, 5), 5) == (2, 5)
    with pytest.raises(ValueError):
        probe._validate_checkpoints((0, 1), 5)
    with pytest.raises(ValueError):
        probe._validate_checkpoints((2, 6), 5)


def test_screen_rankings_uses_both_policies_on_one_discovery():
    mop = mb.mops.DTLZ5(M=5)
    screening = probe.screen_rankings(mop, power=5, seed=123)

    assert set(screening["policies"]) == {
        misda.SIZE_SPAN,
        misda.DOMINANCE_PRESERVATION,
    }
    for policy in (misda.SIZE_SPAN, misda.DOMINANCE_PRESERVATION):
        snapshot = screening["policies"][policy]
        assert snapshot["selected_indices"]
        assert snapshot["selected_dimension"] == len(snapshot["selected_indices"])
        assert snapshot["dominance"]["new_dominance_rate"] >= 0.0

    expected_delta = (
        screening["policies"][misda.DOMINANCE_PRESERVATION]["dominance"][
            "new_dominance_rate"
        ]
        - screening["policies"][misda.SIZE_SPAN]["dominance"][
            "new_dominance_rate"
        ]
    )
    assert screening["dominance_delta"] == pytest.approx(expected_delta)
