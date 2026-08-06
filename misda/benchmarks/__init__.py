"""
misda.benchmarks — MISDA Benchmark Suite and Problem Generators.

Exposes canonical structural cases, synthetic MOP problem generators, DTLZ benchmarks,
and evaluation summary utilities.
"""

from .cases import (
    CANONICAL_CASES,
    make_case1_independence,
    make_case2_total_redundancy,
    make_case3_block_structure,
    make_case4_two_big_blocks,
    make_case5_chain_structure,
    make_case6_mixed_structure,
    make_case7_pure_conflict_groups,
)
from .mop import (
    MOP_CASES,
    mopA_monotonic_redundancy,
    mopB_tradeoff_with_redundancies,
    mopC_latent_blocks_4x5,
    mopD_pure_conflict_groups,
    mopE_partial_redundancy_noisy,
    mopF_regime_switching,
    generate_dtlz2,
    generate_dtlz5,
)
from ..benchmark import (
    BenchmarkCase,
    BenchmarkSuite,
    compare_results,
    compile_benchmark_summary,
    serialize_benchmark_result,
)

__all__ = [
    "CANONICAL_CASES",
    "MOP_CASES",
    "make_case1_independence",
    "make_case2_total_redundancy",
    "make_case3_block_structure",
    "make_case4_two_big_blocks",
    "make_case5_chain_structure",
    "make_case6_mixed_structure",
    "make_case7_pure_conflict_groups",
    "mopA_monotonic_redundancy",
    "mopB_tradeoff_with_redundancies",
    "mopC_latent_blocks_4x5",
    "mopD_pure_conflict_groups",
    "mopE_partial_redundancy_noisy",
    "mopF_regime_switching",
    "generate_dtlz2",
    "generate_dtlz5",
    "BenchmarkCase",
    "BenchmarkSuite",
    "compare_results",
    "compile_benchmark_summary",
    "serialize_benchmark_result",
]
