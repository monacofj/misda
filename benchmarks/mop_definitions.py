"""Synthetic Multi-Objective Problem Generators (backward-compatibility shim)."""

from misda.benchmarks.mop import (
    _mop_truth,
    _mop_df,
    _mk_block_names,
    _repeat_with_small_noise,
    mopA_monotonic_redundancy,
    mopB_tradeoff_with_redundancies,
    mopC_latent_blocks_4x5,
    mopD_pure_conflict_groups,
    mopE_partial_redundancy_noisy,
    mopF_regime_switching,
    generate_dtlz2,
    generate_dtlz5,
    MOP_CASES,
)

__all__ = [
    "_mop_truth",
    "_mop_df",
    "_mk_block_names",
    "_repeat_with_small_noise",
    "mopA_monotonic_redundancy",
    "mopB_tradeoff_with_redundancies",
    "mopC_latent_blocks_4x5",
    "mopD_pure_conflict_groups",
    "mopE_partial_redundancy_noisy",
    "mopF_regime_switching",
    "generate_dtlz2",
    "generate_dtlz5",
    "MOP_CASES",
]
