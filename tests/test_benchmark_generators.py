"""Characterization tests for data generators extracted from the notebooks."""

import hashlib
import inspect

import numpy as np
import pytest

from examples.benchmarks.cases import (
    make_case1_independence,
    make_case2_total_redundancy,
    make_case3_block_structure,
    make_case4_two_big_blocks,
    make_case5_chain_structure,
    make_case6_mixed_structure,
    make_case7_pure_conflict_groups,
)
from examples.mop_definitions import (
    mopA_monotonic_redundancy,
    mopB_tradeoff_with_redundancies,
    mopC_latent_blocks_4x5,
    mopD_pure_conflict_groups,
    mopE_partial_redundancy_noisy,
    mopF_regime_switching,
)


def _matrix_hash(frame):
    matrix = np.ascontiguousarray(frame.to_numpy(dtype=np.float64))
    return hashlib.sha256(matrix.tobytes()).hexdigest()


CANONICAL_CASES = [
    (
        make_case1_independence,
        "c748ec226df5b37d7ade443741f432599c344e005c5cecdf3d797b55aa905343",
        "Case 1 - Total independence",
        20,
        20,
        [1] * 20,
    ),
    (
        make_case2_total_redundancy,
        "060f075873f09507abdd6648093e8db7b02a8d133976b1c0aab5eb6c5763fe04",
        "Case 2 - Total redundancy",
        1,
        1,
        [20],
    ),
    (
        make_case3_block_structure,
        "a0f3ce0c2920e2d4a331d8b20a3d29b3d57b5dfdb1cde602ee785da6de6d3b5f",
        "Case 3 - Blocks (4 x 5)",
        4,
        4,
        [5, 5, 5, 5],
    ),
    (
        make_case4_two_big_blocks,
        "26787fa6276b66d52ad0a3794df5719d878ad6096fe122e1851074d3a22595f7",
        "Case 4 - Blocks (2 x 10)",
        2,
        2,
        [10, 10],
    ),
    (
        make_case5_chain_structure,
        "2fcde82b61cd49f5e46c6e9ab1eacbc6e6a2c329dcd00cc041d2ef9c6411e887",
        "Case 5 - Chain",
        20,
        20,
        [20],
    ),
    (
        make_case6_mixed_structure,
        "f0c2bccd73e12642aa118393859a591b7cbbee26f8b66120c94c464a97ac7689",
        "Case 6 - Mixed (indep + latents)",
        12,
        12,
        [1] * 10 + [5, 5],
    ),
    (
        make_case7_pure_conflict_groups,
        "3fdbb9e5159701a932d2ff3e4ba6ff3d16f101ac630d850554efb5239cbfc534",
        "Case 7 - Structural conflict (anti-corr) 2-groups",
        1,
        2,
        [10, 10],
    ),
]


MOP_CASES = [
    (
        mopA_monotonic_redundancy,
        "1d8c547a97d13fc1fd57532910f9160000e8f3e6479049dccc634a52dd3622c4",
        "MOP-A — Monotonic redundancy (1D, M=20)",
        1,
        1,
        [20],
    ),
    (
        mopB_tradeoff_with_redundancies,
        "cbf1127348eb48cd2a6c6024bbefe76d788e83a16d3acf4c9ed34143d78e29a8",
        "MOP-B — Trade-off + redundancies (~2D, M=20)",
        2,
        2,
        [7, 7, 6],
    ),
    (
        mopC_latent_blocks_4x5,
        "41cbc1f98cef896b805d0462e1d1a3e721a518204815018b7ff789d1d829f237",
        "MOP-C — Latent blocks (4×5, M=20)",
        4,
        4,
        [5, 5, 5, 5],
    ),
    (
        mopD_pure_conflict_groups,
        "6dfa6b76589ccd51c593061fff4aac2b0f796c3cb3ef7173b72b550112757fde",
        "MOP-D — Structural conflict (anti-corr) 2-groups (M=20)",
        1,
        2,
        [10, 10],
    ),
    (
        mopE_partial_redundancy_noisy,
        "0cfcd11707738503ba1adeb65c8430118b7ff818aaa19aaf618a652de9c1d4ea",
        "MOP-E — Partial redundancy + noise (M=20)",
        2,
        2,
        [10, 4, 6],
    ),
    (
        mopF_regime_switching,
        "ae687dbdb76cb3034e96b946c23d8272412412eaf791c7aaaa9802e961835e62",
        "MOP-F — Regimes (mixture, M=20)",
        2,
        2,
        [10, 10],
    ),
]


@pytest.mark.parametrize(
    "generator,expected_hash,name,latent,structural,block_sizes", CANONICAL_CASES
)
def test_canonical_case_matches_notebook_baseline(
    generator, expected_hash, name, latent, structural, block_sizes
):
    frame, truth = generator()

    assert frame.shape == (1000, 20)
    assert list(frame.columns) == [f"f{i}" for i in range(1, 21)]
    assert _matrix_hash(frame) == expected_hash
    assert truth["name"] == name
    assert truth["latent_expected"] == latent
    assert truth["structural_expected"] == structural
    assert [len(block) for block in truth["blocks_expected"]] == block_sizes
    assert truth["feature"]
    assert truth["intuition"]
    assert truth["graph_expected"]


@pytest.mark.parametrize(
    "generator,expected_hash,name,latent,structural,block_sizes", MOP_CASES
)
def test_mop_matches_notebook_baseline(
    generator, expected_hash, name, latent, structural, block_sizes
):
    frame, truth = generator()

    assert frame.shape == (1000, 20)
    assert list(frame.columns) == [f"f{i}" for i in range(1, 21)]
    assert _matrix_hash(frame) == expected_hash
    assert truth["name"] == name
    assert truth["latent_expected"] == latent
    assert truth["structural_expected"] == structural
    assert [len(block) for block in truth["blocks_expected"]] == block_sizes
    assert truth["notes"]
    assert truth["feature"]
    assert truth["intuition"]
    assert truth["graph_expected"]


@pytest.mark.parametrize(
    "generator", [case[0] for case in CANONICAL_CASES + MOP_CASES]
)
def test_generators_preserve_default_seed_and_are_reproducible(generator):
    assert inspect.signature(generator).parameters["seed"].default == 123

    first, _ = generator(N=64, seed=991)
    repeated, _ = generator(N=64, seed=991)
    other_seed, _ = generator(N=64, seed=992)

    np.testing.assert_array_equal(first.to_numpy(), repeated.to_numpy())
    assert not np.array_equal(first.to_numpy(), other_seed.to_numpy())
