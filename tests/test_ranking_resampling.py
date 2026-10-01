"""Contracts for the ranking-evidence resampling probe."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks import run_ranking_resampling_probe as probe


def test_bootstrap_delta_is_deterministic_and_partitions_outcomes():
    data = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 2.0, 0.0],
            [0.5, 1.5, 1.5],
            [1.5, 0.5, 0.5],
        ]
    )

    first = probe.bootstrap_dominance_delta(
        data,
        dominance_indices=(0, 2),
        structural_indices=(0, 1),
        replicates=32,
        seed=123,
    )
    second = probe.bootstrap_dominance_delta(
        data,
        dominance_indices=(0, 2),
        structural_indices=(0, 1),
        replicates=32,
        seed=123,
    )

    assert first == second
    assert first["replicates"] == 32
    assert first["original_delta"] <= 0.0
    assert first["negative_rate"] + first["tie_rate"] + first["positive_rate"] == pytest.approx(1.0)
    assert first["q05_delta"] <= first["median_delta"] <= first["q95_delta"]


def test_bootstrap_same_candidate_has_zero_delta_everywhere():
    rng = np.random.default_rng(17)
    data = rng.normal(size=(24, 4))

    observed = probe.bootstrap_dominance_delta(
        data,
        dominance_indices=(0, 2),
        structural_indices=(0, 2),
        replicates=16,
        seed=456,
    )

    assert observed["original_delta"] == 0.0
    assert observed["negative_rate"] == 0.0
    assert observed["positive_rate"] == 0.0
    assert observed["tie_rate"] == 1.0
    assert observed["mean_delta"] == 0.0
    assert observed["sign_reversal_rate"] == 0.0


def test_bootstrap_rejects_invalid_replicate_count():
    with pytest.raises(ValueError, match="replicates"):
        probe.bootstrap_dominance_delta(
            np.ones((4, 2)),
            dominance_indices=(0,),
            structural_indices=(1,),
            replicates=0,
        )
