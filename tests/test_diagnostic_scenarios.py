"""Scientific contract checks for the unified diagnostic catalogue."""

import pytest

from misda.benchmarks import DIAGNOSTIC_BY_ID, DIAGNOSTIC_SCENARIOS


def test_diagnostic_catalogue_contains_thirteen_unique_scenarios():
    assert len(DIAGNOSTIC_SCENARIOS) == 13
    assert len(DIAGNOSTIC_BY_ID) == 13
    assert tuple(DIAGNOSTIC_BY_ID) == tuple(s.id for s in DIAGNOSTIC_SCENARIOS)


@pytest.mark.parametrize("scenario", DIAGNOSTIC_SCENARIOS, ids=lambda s: s.id)
def test_diagnostic_spec_agrees_with_legacy_generator(scenario):
    assert scenario.variables
    assert scenario.clean_map
    assert scenario.observation
    assert scenario.tags
    assert scenario.latent_expected >= 1
    assert scenario.structural_expected >= 1
    assert sum(scenario.family_sizes) == 20
    scenario.validate_legacy_contract(N=32, seed=123)


def test_generating_families_are_not_used_as_structural_dimension():
    chain = DIAGNOSTIC_BY_ID["transitive_chain"]
    tradeoff = DIAGNOSTIC_BY_ID["tradeoff_redundancies"]

    # One cumulative generating family contains 20 independent innovations.
    assert chain.family_sizes == (20,)
    assert chain.structural_expected == 20

    # Three functional families are driven by a 2D structural truth.
    assert tradeoff.family_sizes == (7, 7, 6)
    assert tradeoff.structural_expected == 2


def test_case5_innovations_are_declared_as_structure_not_observation_noise():
    chain = DIAGNOSTIC_BY_ID["transitive_chain"]
    assert "identity" in chain.observation
    assert "not noise" in chain.observation
    assert len(chain.variables) == 20
    assert chain.latent_expected == 20
    assert chain.structural_expected == 20
