"""Regression tests for the Case 5 transitive-chaining declaration."""

from misda.benchmarks.cases import make_case5_chain_structure


def test_case5_declares_complete_threshold_graph_but_full_generating_dimension():
    _frame, truth = make_case5_chain_structure(N=300, seed=123)

    assert truth["latent_expected"] == 20
    assert truth["structural_expected"] == 20
    assert truth["graph_expectations"] == {
        "structural": {"edges": 190, "components": 1},
        "dependence": {"edges": 190, "components": 1},
    }
    assert "K_20" in truth["graph_expected"]
    assert "generating mechanism is a chain" in truth["graph_expected"]
    assert truth["expected_mismatches"] == {
        "latent_dimension": "TRANSITIVE_CHAINING",
        "structural_dimension": "TRANSITIVE_CHAINING",
    }
    assert "not ground-truth redundancy" in truth["notes"]
