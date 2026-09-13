"""Scientific contract checks for the unified diagnostic catalogue."""

import pytest

from misda.benchmarks import DIAGNOSTIC_BY_ID, DIAGNOSTIC_SCENARIOS


CANONICAL_CASES = (
    ("independence", "Case 1 - Independent objectives"),
    ("total_redundancy", "Case 2 - Complete positive redundancy"),
    ("blocks_4x5", "Case 3 - Four redundant blocks"),
    ("blocks_2x10", "Case 4 - Two redundant blocks"),
    (
        "mixed_independent_and_blocks",
        "Case 5 - Mixed independent and redundant objectives",
    ),
    ("monotonic_redundancy", "Case 6 - Nonlinear monotonic redundancy"),
    ("antagonistic_linear_groups", "Case 7 - Antagonistic linear groups"),
    ("tradeoff_redundancies", "Case 8 - Trade-off with redundant families"),
    ("nonlinear_blocks_4x5", "Case 9 - Nonlinear redundant blocks"),
    (
        "antagonistic_nonlinear_groups",
        "Case 10 - Antagonistic nonlinear groups",
    ),
    ("overlapping_factors", "Case 11 - Overlapping latent factors"),
    ("transitive_chain", "Case 12 - Transitive positive chain"),
    ("regime_switching", "Case 13 - Regime-switching dependence"),
)


def test_diagnostic_catalogue_contains_thirteen_unique_scenarios():
    assert len(DIAGNOSTIC_SCENARIOS) == 13
    assert len(DIAGNOSTIC_BY_ID) == 13
    assert tuple(DIAGNOSTIC_BY_ID) == tuple(s.id for s in DIAGNOSTIC_SCENARIOS)


def test_diagnostic_catalogue_uses_canonical_numbered_order_and_names():
    assert tuple((scenario.id, scenario.name) for scenario in DIAGNOSTIC_SCENARIOS) == (
        CANONICAL_CASES
    )
    assert all("known_failure_mode" not in scenario.tags for scenario in DIAGNOSTIC_SCENARIOS[:-2])
    assert all("known_failure_mode" in scenario.tags for scenario in DIAGNOSTIC_SCENARIOS[-2:])


@pytest.mark.parametrize("scenario", DIAGNOSTIC_SCENARIOS, ids=lambda s: s.id)
def test_diagnostic_spec_agrees_with_legacy_generator(scenario):
    assert scenario.variables
    assert scenario.clean_map
    assert scenario.observation
    assert scenario.tags
    assert scenario.latent_expected >= 1
    assert scenario.structural_expected >= 1
    assert sum(scenario.family_sizes) == 20
    if scenario.structural_unit_sizes is not None:
        assert sum(scenario.structural_unit_sizes) == 20
        assert len(scenario.structural_unit_sizes) == scenario.structural_expected
    scenario.validate_legacy_contract(N=32, seed=123)


def test_generating_families_are_not_used_as_structural_dimension_or_units():
    chain = DIAGNOSTIC_BY_ID["transitive_chain"]
    tradeoff = DIAGNOSTIC_BY_ID["tradeoff_redundancies"]

    # One cumulative generating family contains 20 independent innovations.
    assert chain.family_sizes == (20,)
    assert chain.structural_expected == 20
    assert chain.structural_unit_sizes == (1,) * 20

    # Three functional families are driven by a 2D structural truth, but do not
    # define an unambiguous two-unit structural partition.
    assert tradeoff.family_sizes == (7, 7, 6)
    assert tradeoff.structural_expected == 2
    assert tradeoff.structural_unit_sizes is None


def test_overlapping_factors_do_not_declare_family_partition_as_structural_units():
    overlapping = DIAGNOSTIC_BY_ID["overlapping_factors"]
    assert overlapping.family_sizes == (10, 4, 6)
    assert overlapping.structural_expected == 2
    assert overlapping.structural_unit_sizes is None


def test_case5_innovations_are_declared_as_structure_not_observation_noise():
    chain = DIAGNOSTIC_BY_ID["transitive_chain"]
    assert "identity" in chain.observation
    assert "not noise" in chain.observation
    assert len(chain.variables) == 20
    assert chain.latent_expected == 20
    assert chain.structural_expected == 20
