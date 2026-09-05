import numpy as np

import misda
from misda._support import (
    HIDDEN_SPECTRAL_STRUCTURE,
    SUPPORTED,
    TRANSITIVE_CHAINING,
    UNSUPPORTED,
    _positive_and_closure,
    _transitivity_statistic_from_closure,
    _transitivity_statistics_from_closure,
    evaluate_dimensional_support,
)
from misda.benchmarks.cases import (
    make_case2_total_redundancy,
    make_case5_chain_structure,
)
from misda.benchmarks.mop import (
    mopB_tradeoff_with_redundancies,
    mopF_regime_switching,
)


def test_transitivity_support_separates_redundancy_from_chain():
    redundant, _ = make_case2_total_redundancy(N=100, seed=123)
    chain, _ = make_case5_chain_structure(N=100, seed=123)

    redundant_support = evaluate_dimensional_support(
        redundant,
        selected_indices=(0,),
        latent_dimension=1,
        seed=123,
    )
    chain_support = evaluate_dimensional_support(
        chain,
        selected_indices=(0,),
        latent_dimension=1,
        seed=123,
    )

    assert redundant_support["status"] == SUPPORTED
    assert redundant_support["transitivity"]["excess"] <= 0.0
    assert chain_support["status"] == UNSUPPORTED
    assert chain_support["transitivity"]["excess"] > 0.0
    assert TRANSITIVE_CHAINING in chain_support["reasons"]


def test_spectral_support_separates_good_2d_from_regime_underestimate():
    good, _ = mopB_tradeoff_with_redundancies(N=100, seed=123)
    regime, _ = mopF_regime_switching(N=100, seed=123)

    good_support = evaluate_dimensional_support(
        good,
        selected_indices=(0, 14),
        latent_dimension=2,
        seed=123,
    )
    regime_support = evaluate_dimensional_support(
        regime,
        selected_indices=(0,),
        latent_dimension=1,
        seed=123,
    )

    assert good_support["status"] == SUPPORTED
    assert good_support["spectral"]["excess"] <= 0.0
    assert regime_support["status"] == UNSUPPORTED
    assert regime_support["spectral"]["excess"] > 0.0
    assert HIDDEN_SPECTRAL_STRUCTURE in regime_support["reasons"]


def test_support_uses_sample_size_as_null_budget_and_is_reproducible():
    data, _ = make_case5_chain_structure(N=80, seed=7)

    first = evaluate_dimensional_support(data, (0,), 1, seed=19)
    second = evaluate_dimensional_support(data, (0,), 1, seed=19)

    assert first == second
    assert first["n_permutations"] == len(data)


def test_vectorized_transitivity_matches_scalar_for_mixed_cardinalities():
    rng = np.random.default_rng(2026)
    raw = rng.uniform(-1.0, 1.0, size=(8, 8))
    correlation = (raw + raw.T) / 2.0
    np.fill_diagonal(correlation, 1.0)
    positive, closure = _positive_and_closure(correlation)
    selections = (
        (0,),
        (1, 3),
        (0, 2, 5),
        (1, 2, 4, 6),
        tuple(range(8)),
        (0, 3),
    )

    scalar = np.asarray(
        [
            _transitivity_statistic_from_closure(
                positive,
                closure,
                selected,
            )
            for selected in selections
        ]
    )
    vectorized = _transitivity_statistics_from_closure(
        positive,
        closure,
        selections,
    )

    np.testing.assert_allclose(vectorized, scalar, rtol=0.0, atol=0.0)


def test_discover_attaches_group_support_and_report_renders_it():
    data, _ = make_case2_total_redundancy(N=40, seed=123)
    result = misda.discover(data, seed=123)

    assert result.support is not None
    assert result.support.status in {
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "UNSUPPORTED",
    }
    assert len(result.support.results) == len(result.structural_ranking.groups[0])
    report = result.report()
    assert "Dimensional support:" in report
