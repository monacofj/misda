"""Canonical synthetic cases used by the MISDA benchmark notebook (backward-compatibility shim)."""

from misda.benchmarks.cases import (
    _truth,
    make_case1_independence,
    make_case2_total_redundancy,
    make_case3_block_structure,
    make_case4_two_big_blocks,
    make_case5_chain_structure,
    make_case6_mixed_structure,
    make_case7_pure_conflict_groups,
    CANONICAL_CASES,
)

__all__ = [
    "_truth",
    "make_case1_independence",
    "make_case2_total_redundancy",
    "make_case3_block_structure",
    "make_case4_two_big_blocks",
    "make_case5_chain_structure",
    "make_case6_mixed_structure",
    "make_case7_pure_conflict_groups",
    "CANONICAL_CASES",
]
