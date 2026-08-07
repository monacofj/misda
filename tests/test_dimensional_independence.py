"""Regression tests for independence-number dimensional estimates."""

import misda
from misda.benchmarks.mop import (
    mopA_monotonic_redundancy,
    mopB_tradeoff_with_redundancies,
)


def _analyze(generator, n=300):
    frame, truth = generator(N=n, seed=123)
    result = misda.analyze(
        frame,
        method="static",
        max_evaluated_mis=1,
        seed=123,
        name=truth["name"],
    )
    return result, truth


def test_mop_a_independence_dimensions_are_one():
    result, truth = _analyze(mopA_monotonic_redundancy)

    assert truth["latent_expected"] == 1
    assert truth["structural_expected"] == 1
    assert result.analysis.latent_dimension == 1
    assert result.analysis.structural_dimension == 1
    assert result.analysis.graph_summaries["dependence"]["components"] == 1
    assert result.analysis.graph_summaries["structural"]["components"] == 1
    assert result.selected_dimension == 1


def test_mop_b_connected_graph_has_two_independent_dimensions():
    result, truth = _analyze(mopB_tradeoff_with_redundancies)

    assert truth["latent_expected"] == 2
    assert truth["structural_expected"] == 2
    assert result.analysis.latent_dimension == 2
    assert result.analysis.structural_dimension == 2
    assert result.analysis.graph_summaries["dependence"]["components"] == 1
    assert result.analysis.graph_summaries["structural"]["components"] == 1
    assert result.selected_dimension == 2
