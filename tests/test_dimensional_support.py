import numpy as np

from misda import api
from misda._support import (
    HIDDEN_SPECTRAL_STRUCTURE,
    SUPPORTED,
    TRANSITIVE_CHAINING,
    UNSUPPORTED,
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


def test_analyze_attaches_support_and_report_renders_it(monkeypatch):
    marker = {
        "status": SUPPORTED,
        "reasons": (),
        "transitivity": {"observed": 0.0, "null": 0.1, "excess": -0.1},
        "spectral": {
            "tested_dimension": 1,
            "observed_next_eigenvalue": 0.1,
            "null_next_eigenvalue": 1.0,
            "excess": -0.9,
        },
        "n_permutations": 20,
        "seed": 1,
    }

    monkeypatch.setattr(
        api,
        "evaluate_dimensional_support",
        lambda *args, **kwargs: marker,
    )
    x = np.linspace(0.0, 1.0, 20)
    data = np.column_stack((x, 2.0 * x))

    result = api._analyze_static_v2(
        data,
        seed=123,
        max_evaluated_mis=1,
    )

    assert result.analysis.dimensional_support is marker
    report = result.report()
    assert "Dimensional support: SUPPORTED" in report
    assert "transitivity excess=-0.1000" in report
    assert "spectral excess=-0.9000" in report
