# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pandas as pd

from benchmarks.run_noisy_ranking_safety import observe, run_audit


def test_observe_sigma_zero_is_identity():
    frame = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 9.0]})
    observed = observe(frame, sigma=0.0, observation_seed=123)
    pd.testing.assert_frame_equal(observed, frame)
    assert observed is not frame


def test_observe_scales_same_standard_noise_linearly():
    frame = pd.DataFrame({"f1": [1.0, 2.0, 4.0], "f2": [3.0, 8.0, 9.0]})
    low = observe(frame, sigma=0.1, observation_seed=456)
    high = observe(frame, sigma=0.2, observation_seed=456)
    np.testing.assert_allclose(
        high.to_numpy() - frame.to_numpy(),
        2.0 * (low.to_numpy() - frame.to_numpy()),
    )


def test_small_audit_preserves_clean_reference_at_sigma_zero():
    result = run_audit(
        screen_power=5,
        sigmas=(0.0,),
        observation_seeds=(101,),
    )
    assert {row["problem"] for row in result["rows"]} == {"DTLZ5", "DPF1"}
    for row in result["rows"]:
        for policy in ("size_span", "dominance_preservation"):
            assert row["policies"][policy]["matches_clean_selection"] is True
