# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

import inspect

import numpy as np
import pytest

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
    assert "no MIS has been selected yet" in text


def test_profile_regime_status_is_aggregate_discovery_support_not_ranking():
    observed = misda.profile(_two_blocks(), seed=13)

    for regime in observed:
        analysis = regime.analysis
        if analysis.structural_dimension == analysis.original_dimension:
            assert regime.status == misda.NO_REDUNDANCY
        elif regime.mis_set.support.status == "SUPPORTED":
            assert regime.status == misda.SUPPORTED_REDUCTION
        elif regime.mis_set.support.status == misda.PARTIALLY_SUPPORTED:
            assert regime.status == misda.PARTIALLY_SUPPORTED
            assert not regime.acceptable
        else:
            assert regime.status == misda.UNSUPPORTED_REDUCTION
            assert not regime.acceptable


def test_profile_stores_support_for_every_discovered_mis():
    observed = misda.profile(_two_blocks(seed=15), seed=17)

    for regime in observed:
        for candidate in regime.mis_set:
            support = regime.mis_set.support_for(candidate)
            assert support.candidate_index >= 0


def test_profile_report_is_side_effect_free_for_candidate_evidence():
    observed = misda.profile(_two_blocks(), seed=19)
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


def test_discovery_contract_has_no_ranking_policy_and_returns_mis_set():
    assert "rank_policy" not in inspect.signature(misda.discovery).parameters

    observed = misda.profile(_two_blocks(), seed=23)
    if observed.selected is None:
        pytest.skip("fixture produced no admissible profile regime")

    mis_set = misda.discovery(observed)

    assert isinstance(mis_set, misda.MISSet)
    assert mis_set is observed.selected.mis_set
    assert all(candidate.dominance is None for candidate in mis_set)


def test_discovery_y_is_equivalent_to_profile_then_discovery():
    data = _two_blocks(seed=29)
    staged_profile = misda.profile(data, seed=31)
    if staged_profile.selected is None:
        with pytest.raises(RuntimeError, match="no selected regime"):
            misda.discovery(data, seed=31)
        return

    direct = misda.discovery(data, seed=31)
    staged = misda.discovery(staged_profile)

    assert isinstance(direct, misda.MISSet)
    assert isinstance(staged, misda.MISSet)
    assert tuple(candidate.indices for candidate in direct) == tuple(
        candidate.indices for candidate in staged
    )
    assert direct.analysis.structural_dimension == staged.analysis.structural_dimension
    assert direct.analysis.aggressiveness == staged.analysis.aggressiveness


def test_rank_default_is_dominance_and_size_span_remains_explicit():
    observed = misda.profile(_two_blocks(seed=37), seed=41)
    if observed.selected is None:
        pytest.skip("fixture produced no admissible profile regime")
    mis_set = misda.discovery(observed)

    default_ranking = misda.rank(mis_set)
    structural_ranking = misda.rank(mis_set, policy=misda.SIZE_SPAN)

    assert default_ranking.policy == misda.DOMINANCE_PRESERVATION
    assert all(candidate.dominance is not None for candidate in mis_set)
    assert structural_ranking.policy == misda.SIZE_SPAN
    assert tuple(structural_ranking.indices) == tuple(
        mis_set.structural_ranking.indices
    )


def test_ranking_does_not_change_discovery_universe_or_canonical_order():
    observed = misda.profile(_two_blocks(seed=43), seed=47)
    if observed.selected is None:
        pytest.skip("fixture produced no admissible profile regime")
    mis_set = misda.discovery(observed)

    candidates_before = tuple(candidate.indices for candidate in mis_set)
    canonical_before = tuple(mis_set.structural_ranking.indices)
    dimensions_before = (
        mis_set.analysis.original_dimension,
        mis_set.analysis.latent_dimension,
        mis_set.analysis.structural_dimension,
    )
    graph_before = frozenset(mis_set.analysis.structural_graph.edges())

    ranking = misda.rank(mis_set)

    assert ranking.mis_set is mis_set
    assert tuple(candidate.indices for candidate in mis_set) == candidates_before
    assert tuple(mis_set.structural_ranking.indices) == canonical_before
    assert (
        mis_set.analysis.original_dimension,
        mis_set.analysis.latent_dimension,
        mis_set.analysis.structural_dimension,
    ) == dimensions_before
    assert frozenset(mis_set.analysis.structural_graph.edges()) == graph_before


def test_discovery_refuses_profile_without_selected_regime():
    base = misda.profile(_two_blocks(seed=53), seed=59)
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

    with pytest.raises(RuntimeError, match="no selected regime"):
        misda.discovery(abstaining)
