"""Regression tests for machine-verifiable MOP graph declarations."""

from misda.benchmarks.mop import (
    mopA_monotonic_redundancy,
    mopB_tradeoff_with_redundancies,
    mopC_latent_blocks_4x5,
    mopD_pure_conflict_groups,
    mopE_partial_redundancy_noisy,
    mopF_regime_switching,
)


def _truth(generator):
    _data, truth = generator(N=64, seed=123)
    return truth


def test_exact_mop_graph_claims_are_structured_exactly():
    expected = {
        mopA_monotonic_redundancy: {
            "structural": {"nodes": 20, "edges": 190, "components": 1},
            "dependence": {"nodes": 20, "edges": 190, "components": 1},
        },
        mopC_latent_blocks_4x5: {
            "structural": {"nodes": 20, "edges": 40, "components": 4},
            "dependence": {"nodes": 20, "edges": 40, "components": 4},
        },
        mopD_pure_conflict_groups: {
            "structural": {"nodes": 20, "edges": 90, "components": 2},
            "dependence": {"nodes": 20, "edges": 190, "components": 1},
        },
        mopF_regime_switching: {
            "structural": {"nodes": 20, "edges": 190, "components": 1},
            "dependence": {"nodes": 20, "edges": 190, "components": 1},
        },
    }

    for generator, graph_expectations in expected.items():
        assert _truth(generator)["graph_expectations"] == graph_expectations


def test_qualitative_generating_families_are_not_misdeclared_as_graph_clusters():
    for generator in (mopB_tradeoff_with_redundancies, mopE_partial_redundancy_noisy):
        truth = _truth(generator)
        assert truth["graph_expectations"] == {
            "structural": {"nodes": 20, "components": 1},
            "dependence": {"nodes": 20, "components": 1},
        }
        assert "declared separately from graph topology" in truth["graph_expected"]
