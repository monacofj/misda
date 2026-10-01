# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

import misda


def _two_blocks(seed=7, n=36):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    return np.column_stack(
        [
            x,
            2.0 * x + 0.01 * rng.normal(size=n),
            y,
            -3.0 * y + 0.01 * rng.normal(size=n),
        ]
    )


def test_public_profile_exposes_ordered_regimes_and_selected_boundary():
    observed = misda.profile(_two_blocks(), seed=11, name="two blocks")

    assert isinstance(observed, misda.Profile)
    assert len(observed) >= 1
    aggressiveness = [regime.aggressiveness for regime in observed]
    assert aggressiveness == sorted(aggressiveness)
    assert len(set(aggressiveness)) == len(aggressiveness)

    acceptable = [regime for regime in observed if regime.acceptable]
    if acceptable:
        assert observed.selected is acceptable[-1]
        assert observed.selected.aggressiveness == max(
            regime.aggressiveness for regime in acceptable
        )
    else:
        assert observed.selected is None

    text = observed.report()
    assert "alpha_onset=" in text
    assert "alpha_null=" in text
    assert "Selected regime:" in text


def test_profile_report_is_side_effect_free_for_candidate_evidence():
    observed = misda.profile(_two_blocks(), seed=13)
    before = [
        tuple(candidate.dominance for candidate in regime.mis_set)
        for regime in observed
    ]

    observed.report()

    after = [
        tuple(candidate.dominance for candidate in regime.mis_set)
        for regime in observed
    ]
    assert after == before
    assert all(value is None for row in after for value in row)


def test_discovery_from_profile_defaults_to_dominance_preservation():
    observed = misda.profile(_two_blocks(), seed=17)
    result = misda.discovery(observed)

    assert isinstance(result, misda.DiscoveryResult)
    if observed.selected is None:
        assert result.status == misda.ABSTAINED
        assert result.selected is None
    else:
        assert result.regime is observed.selected
        assert result.rank_policy == misda.DOMINANCE_PRESERVATION
        assert result.ranking.policy == misda.DOMINANCE_PRESERVATION
        assert result.selected is result.ranking.selected
        assert all(candidate.dominance is not None for candidate in result.mis_set)


def test_discovery_y_is_equivalent_to_profile_then_discovery():
    data = _two_blocks(seed=19)
    direct = misda.discovery(data, seed=23)
    staged_profile = misda.profile(data, seed=23)
    staged = misda.discovery(staged_profile)

    assert direct.status == staged.status
    assert direct.rank_policy == staged.rank_policy
    if direct.selected is None:
        assert staged.selected is None
    else:
        assert direct.selected.indices == staged.selected.indices
        assert direct.regime.analysis.structural_dimension == staged.regime.analysis.structural_dimension
        assert direct.regime.aggressiveness == staged.regime.aggressiveness


def test_size_span_remains_available_and_low_level_default_is_unchanged():
    observed = misda.profile(_two_blocks(seed=29), seed=31)
    if observed.selected is None:
        return

    high_level = misda.discovery(observed, rank_policy=misda.SIZE_SPAN)
    low_level = misda.rank(observed.selected.mis_set)

    assert high_level.rank_policy == misda.SIZE_SPAN
    assert high_level.ranking.policy == misda.SIZE_SPAN
    assert high_level.selected.indices == low_level.selected.indices
    assert low_level.policy == misda.SIZE_SPAN


def test_discovery_abstains_when_profile_has_no_selected_regime():
    base = misda.profile(_two_blocks(seed=37), seed=41)
    abstaining = misda.Profile(
        regimes=base.regimes,
        selected_index=None,
        alpha_onset=base.alpha_onset,
        alpha_null=base.alpha_null,
        separation=base.separation_status,
        seed=base.seed,
        name="forced abstention contract",
        correlation=base.correlation,
        experimental=base.experimental,
    )

    result = misda.discovery(abstaining)

    assert result.status == misda.ABSTAINED
    assert result.regime is None
    assert result.ranking is None
    assert result.selected is None
    assert "ABSTAINED" in result.report()
