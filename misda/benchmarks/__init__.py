"""
misda.benchmarks — MISDA Benchmark Suite and Problem Generators.

Exposes the unified controlled diagnostic catalogue and executable X -> Z -> Y
problems, historical generator names, classical DTLZ generators, and evaluation
summary utilities.
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
from .diagnostics import (
    DIAGNOSTIC_BY_ID,
    DIAGNOSTIC_SCENARIOS,
    DiagnosticScenario,
)
from .problems import (
    PROBLEM_BY_ID,
    PROBLEMS,
    DiagnosticDataset,
    DiagnosticProblem,
)
from .comparative import (
    COMMON_RECONSTRUCTION_METRIC,
    global_standardized_reconstruction_r2,
    misda_global_standardized_external_r2,
    pca_external_reconstruction_curve,
    pca_in_sample_reconstruction_curve,
)
from ..benchmark import (
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkSuite,
    benchmark,
    compare_results,
    compile_benchmark_summary,
    serialize_benchmark_result,
)

__all__ = [
    "DiagnosticScenario",
    "DIAGNOSTIC_SCENARIOS",
    "DIAGNOSTIC_BY_ID",
    "DiagnosticProblem",
    "DiagnosticDataset",
    "PROBLEMS",
    "PROBLEM_BY_ID",
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
    "COMMON_RECONSTRUCTION_METRIC",
    "global_standardized_reconstruction_r2",
    "misda_global_standardized_external_r2",
    "pca_external_reconstruction_curve",
    "pca_in_sample_reconstruction_curve",
    "BenchmarkCase",
    "BenchmarkResult",
    "BenchmarkSuite",
    "benchmark",
    "compare_results",
    "compile_benchmark_summary",
    "serialize_benchmark_result",
]
