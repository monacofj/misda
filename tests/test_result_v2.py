"""Tests for the refactored static result tree and core assembly."""

import math

import networkx as nx
import numpy as np
import pytest

import misda
from misda import result


def _two_conflicting_groups(include_constant=False):
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    columns = [x, 2.0 * x, -x, -2.0 * x]
    if include_constant:
        columns.append(np.full_like(x, 7.0))
    return np.column_stack(columns)


@pytest.fixture
def refactored_result():
    return misda._analyze_static_v2(
        _two_conflicting_groups(),
        aggressiveness=1.0,
        rank_policy="default",
        max_evaluated_mis=2,
        seed=19,
        name="two groups",
    )


def test_refactored_core_returns_public_result_tree(refactored_result):
    observed = refactored_result

    assert type(observed) is result.MISDAResult
    assert isinstance(observed.analysis, result.AnalysisResult)
    assert isinstance(observed.execution, result.ExecutionResult)
    assert all(type(candidate) is result.MISCandidate for candidate in observed.mis)
    assert observed.name == "two groups"


def test_dimensions_and_components_are_distinct_graph_properties(
    refactored_result,
):
    analysis = refactored_result.analysis

    assert analysis.original_dimension == 4
    assert analysis.structural_dimension == 2
    assert analysis.latent_dimension == 1
    assert analysis.structural_components == ((0, 1), (2, 3))
    assert analysis.latent_components == ((0, 1, 2, 3),)
    assert analysis.structural_dimension == nx.algorithms.clique.graph_clique_number(
        nx.complement(analysis.structural_graph)
    )
    assert analysis.latent_dimension == nx.algorithms.clique.graph_clique_number(
        nx.complement(analysis.dependence_graph)
    )
    assert analysis.graph_summaries == {
        "structural": {"nodes": 4, "edges": 2, "components": 2},
        "dependence": {"nodes": 4, "edges": 6, "components": 1},
    }


def test_all_ranked_mis_are_stored_with_explicit_identity(refactored_result):
    observed = refactored_result

    assert observed.analysis.n_mis == len(observed.mis) == 4
    assert observed.analysis.rank_counts == {1: 4}
    assert [candidate.id for candidate in observed.mis] == [
        "mis_000",
        "mis_001",
        "mis_002",
        "mis_003",
    ]
    assert all(candidate.size == 2 for candidate in observed.mis)
    assert all(candidate.rank == 1 for candidate in observed.mis)
    assert all(
        set(candidate.evaluation) == {
            "linear_reconstruction",
            "pareto_preservation",
        }
        for candidate in observed.mis[:2]
    )
    assert all(candidate.evaluation == {} for candidate in observed.mis[2:])
    assert all(
        len(set(candidate.indices) & {0, 1}) == 1
        and len(set(candidate.indices) & {2, 3}) == 1
        for candidate in observed.mis
    )
    assert observed.best_mis is observed.mis[0]
    assert observed.best_mis_indices == list(observed.mis[0].indices)
    assert observed.best_mis_labels == list(observed.mis[0].objectives)


def test_light_evaluation_records_the_requested_prefix(
    refactored_result,
):
    observed = refactored_result

    assert observed.analysis.n_evaluated_mis == 2
    assert observed.analysis.n_heavy_mis == 0
    assert observed.execution.configuration == {
        "aggressiveness": 1.0,
        "rank_policy": "default",
        "max_evaluated_mis": 2,
    }


def test_null_estimation_diagnostics_are_complete(refactored_result):
    diagnostics = refactored_result.execution.convergence["alpha_null"]

    assert diagnostics["converged"]
    assert diagnostics["n_permutations"] >= 6
    assert diagnostics["se_mc"] >= 0.0
    assert len(diagnostics["r_interval"]) == 2
    assert len(diagnostics["log_alpha_interval"]) == 2
    assert diagnostics["seed"] == 19
    assert diagnostics["rng_state"]["bit_generator"] == "PCG64"
    assert refactored_result.execution.timings["total"] >= 0.0


def test_thresholds_status_and_aggressiveness_are_stored(refactored_result):
    analysis = refactored_result.analysis

    assert analysis.log_alpha_onset == -math.inf
    assert analysis.alpha_onset == 0.0
    assert np.isfinite(analysis.log_alpha_null)
    assert analysis.alpha_null > 0.0
    assert analysis.log_alpha == analysis.log_alpha_null
    assert analysis.alpha == analysis.alpha_null
    assert analysis.aggressiveness == 1.0
    assert analysis.separation_status is misda._statistics.SeparationStatus.NULL_SEPARATION


def test_no_positive_relation_keeps_full_structural_selection():
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    observed = misda._analyze_static_v2(
        np.column_stack([x, -x]),
        seed=3,
    )

    assert observed.analysis.log_alpha_onset is None
    assert observed.analysis.alpha_onset is None
    assert observed.analysis.separation_status is (
        misda._statistics.SeparationStatus.NO_NULL_SEPARATION
    )
    assert observed.analysis.structural_dimension == 2
    assert observed.analysis.latent_dimension == 1
    assert observed.best_mis.indices == (0, 1)
    assert not observed.reduction_applied


def test_constant_objective_is_preserved_as_an_isolated_unit():
    observed = misda._analyze_static_v2(
        _two_conflicting_groups(include_constant=True),
        seed=11,
    )

    assert observed.analysis.structural_dimension == 3
    assert observed.analysis.latent_dimension == 2
    assert observed.best_mis.size == 3
    assert 4 in observed.best_mis.indices
    assert observed.analysis.structural_graph.nodes[4]["constant"]


def test_transitional_unambiguous_properties_warn_and_forward(refactored_result):
    observed = refactored_result

    with pytest.warns(DeprecationWarning, match="alpha_min"):
        assert observed.alpha_min == observed.analysis.alpha_onset
    with pytest.warns(DeprecationWarning, match="alpha_max"):
        assert observed.alpha_max == observed.analysis.alpha_null
    with pytest.warns(DeprecationWarning, match="caution"):
        assert observed.caution == observed.analysis.aggressiveness
    with pytest.warns(DeprecationWarning, match="mis_sets"):
        assert observed.mis_sets == list(observed.mis)
    with pytest.warns(DeprecationWarning, match="ranked_mis_sets"):
        grouped = observed.ranked_mis_sets
    assert grouped == {1: list(observed.mis)}


def test_abandoned_metrics_and_legacy_result_tree_have_no_aliases(
    refactored_result,
):
    observed = refactored_result

    for name in (
        "ses_results",
        "ses_nonlinear",
        "pareto_recall",
        "validation_metrics",
        "isda_results",
    ):
        assert not hasattr(observed, name)


def test_summary_uses_only_stored_result_values(refactored_result):
    summary = refactored_result.summary()

    assert "MISDA Analysis Summary: two groups" in summary
    assert "Dimensions: original=4, latent=1, structural=2" in summary
    assert "Selected dimension: 2" in summary
    assert "MISs: 4; evaluated=2; heavy=0" in summary
    assert "Preferred MIS: mis_000" in summary
    assert refactored_result.report().startswith("MISDA static report: two groups")


def test_input_copy_is_read_only(refactored_result):
    with pytest.raises(ValueError, match="read-only"):
        refactored_result._data[0, 0] = 99.0
