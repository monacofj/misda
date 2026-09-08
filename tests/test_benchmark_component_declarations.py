"""Regression tests for benchmark component declarations."""

from misda.benchmark import _structural_metrics, _truth_components
from misda.benchmarks.cases import make_case3_block_structure
from misda.benchmarks.mop import mopB_tradeoff_with_redundancies


def test_component_truth_is_distinct_from_generating_blocks_when_needed():
    _frame, truth = mopB_tradeoff_with_redundancies(N=64, seed=123)

    assert [len(block) for block in truth["blocks_expected"]] == [7, 7, 6]
    assert [len(block) for block in truth["components_expected"]] == [20]
    assert _truth_components(truth) == tuple(
        tuple(block) for block in truth["components_expected"]
    )

    found_components = (tuple(f"f{i}" for i in range(1, 21)),)
    component_metrics = _structural_metrics(
        found_components,
        truth["components_expected"],
    )
    block_metrics = _structural_metrics(
        found_components,
        truth["blocks_expected"],
    )

    assert component_metrics["structural_partition_exact"]
    assert component_metrics["structural_jaccard"] == 1.0
    assert not block_metrics["structural_partition_exact"]
    assert block_metrics["structural_jaccard"] < 1.0


def test_canonical_block_cases_keep_component_partition_when_they_coincide():
    _frame, truth = make_case3_block_structure(N=64, seed=123)

    assert truth["components_expected"] == truth["blocks_expected"]
    metrics = _structural_metrics(
        truth["components_expected"],
        truth["blocks_expected"],
    )
    assert metrics["structural_partition_exact"]
    assert metrics["structural_jaccard"] == 1.0
