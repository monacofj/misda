"""Canonical synthetic cases used by the MISDA benchmark notebook."""

import numpy as np
import pandas as pd


def _truth(
    name,
    latent_expected,
    structural_expected,
    blocks_expected,
    feature="",
    intuition="",
    graph_expected="",
    pareto_expected=None,
    notes="",
):
    return {
        "name": name,
        "latent_expected": (
            int(latent_expected)
            if latent_expected != "" and latent_expected is not None
            else None
        ),
        "structural_expected": (
            int(structural_expected)
            if structural_expected != "" and structural_expected is not None
            else None
        ),
        "blocks_expected": blocks_expected,
        "pareto_expected": (
            None
            if pareto_expected is None
            else [int(index) for index in pareto_expected]
        ),
        "feature": feature,
        "intuition": intuition,
        "graph_expected": graph_expected,
        "notes": notes,
    }


def make_case1_independence(N=1000, M=20, seed=123):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(N, M))
    cols = [f"f{i+1}" for i in range(M)]
    df = pd.DataFrame(Y, columns=cols)
    truth = _truth(
        name="Case 1 - Total independence",
        latent_expected=M,
        structural_expected=M,
        blocks_expected=[[c] for c in cols],
        feature="All 20 objectives are mutually independent i.i.d. Gaussian noise variables.",
        intuition="20 completely unrelated random sensors; knowing one tells you nothing about any other. MISDA should keep all 20.",
        graph_expected="20 isolated nodes (0 edges, 20 connected components)",
    )
    return df, truth


def make_case2_total_redundancy(N=1000, M=20, seed=123):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(N, 1))
    noise = rng.normal(scale=0.05, size=(N, M))
    Y = latent + noise
    cols = [f"f{i+1}" for i in range(M)]
    df = pd.DataFrame(Y, columns=cols)
    truth = _truth(
        name="Case 2 - Total redundancy",
        latent_expected=1,
        structural_expected=1,
        blocks_expected=[cols],
        feature="All 20 objectives are noisy linear copies of a single 1D latent factor.",
        intuition="20 identical thermometers measuring the exact same room temperature with minor noise. MISDA should keep just 1.",
        graph_expected="1 fully connected graph (K_20, 190 edges, 1 connected component)",
    )
    return df, truth


def make_case3_block_structure(N=1000, M=20, seed=123):
    rng = np.random.default_rng(seed)
    assert M == 20
    latent_blocks = rng.normal(size=(N, 4))
    Y = np.zeros((N, M))
    for b in range(4):
        for j in range(5):
            idx = 5 * b + j
            Y[:, idx] = latent_blocks[:, b] + rng.normal(scale=0.2, size=N)
    cols = [f"f{i+1}" for i in range(M)]
    df = pd.DataFrame(Y, columns=cols)
    blocks = [
        [f"f{i}" for i in range(1, 6)],
        [f"f{i}" for i in range(6, 11)],
        [f"f{i}" for i in range(11, 16)],
        [f"f{i}" for i in range(16, 21)],
    ]
    truth = _truth(
        name="Case 3 - Blocks (4 x 5)",
        latent_expected=4,
        structural_expected=4,
        blocks_expected=blocks,
        feature="4 independent latent factors; each factor generates a cluster of 5 redundant objectives.",
        intuition="4 physical properties (e.g., Temp, Pressure, Humidity, Speed), each measured by 5 duplicate sensors. MISDA should reduce 20 sensors to 4.",
        graph_expected="4 disjoint complete subgraphs of 5 nodes each (4 x K_5, 40 total edges)",
    )
    return df, truth


def make_case4_two_big_blocks(N=1000, M=20, seed=123):
    rng = np.random.default_rng(seed)
    assert M == 20
    latent_blocks = rng.normal(size=(N, 2))
    Y = np.zeros((N, M))
    for i in range(10):
        Y[:, i] = latent_blocks[:, 0] + rng.normal(scale=0.2, size=N)
    for i in range(10, 20):
        Y[:, i] = latent_blocks[:, 1] + rng.normal(scale=0.2, size=N)
    cols = [f"f{i+1}" for i in range(M)]
    df = pd.DataFrame(Y, columns=cols)
    truth = _truth(
        name="Case 4 - Blocks (2 x 10)",
        latent_expected=2,
        structural_expected=2,
        blocks_expected=[
            [f"f{i}" for i in range(1, 11)],
            [f"f{i}" for i in range(11, 21)],
        ],
        feature="2 independent latent factors; each factor generates a cluster of 10 redundant objectives.",
        intuition="Measuring 2 goals (e.g., Cost and Weight), but using 10 duplicate formulas for Cost and 10 for Weight. MISDA should reduce 20 formulas to 2.",
        graph_expected="2 disjoint complete subgraphs of 10 nodes each (2 x K_10, 90 total edges)",
    )
    return df, truth


def make_case5_chain_structure(N=1000, M=20, seed=123):
    rng = np.random.default_rng(seed)
    Y = np.zeros((N, M))
    Y[:, 0] = rng.normal(size=N)
    for j in range(1, M):
        Y[:, j] = Y[:, j - 1] + rng.normal(scale=0.2, size=N)
    cols = [f"f{i+1}" for i in range(M)]
    df = pd.DataFrame(Y, columns=cols)
    truth = _truth(
        name="Case 5 - Chain",
        latent_expected=M,
        structural_expected=M,
        blocks_expected=[cols],
        feature="Markovian random walk chain where correlation decays smoothly with index distance.",
        intuition="A chain of 20 dominoes: adjacent dominoes are strongly linked, but the 1st and 20th are far apart. Tests gradual, step-by-step continuous dependency.",
        graph_expected="1 connected chain/band graph (adjacent node edges, 1 connected component)",
    )
    return df, truth


def make_case6_mixed_structure(N=1000, M=20, seed=123):
    rng = np.random.default_rng(seed)
    assert M == 20
    Y = np.zeros((N, M))
    Y[:, :10] = rng.normal(size=(N, 10))
    latent1 = rng.normal(size=N)
    latent2 = rng.normal(size=N)
    for j in range(10, 15):
        Y[:, j] = latent1 + rng.normal(scale=0.2, size=N)
    for j in range(15, 20):
        Y[:, j] = latent2 + rng.normal(scale=0.2, size=N)
    cols = [f"f{i+1}" for i in range(M)]
    df = pd.DataFrame(Y, columns=cols)
    truth = _truth(
        name="Case 6 - Mixed (indep + latents)",
        latent_expected=12,
        structural_expected=12,
        blocks_expected=[
            [f"f{i}"] for i in range(1, 11)
        ]
        + [
            [f"f{i}" for i in range(11, 16)],
            [f"f{i}" for i in range(16, 21)],
        ],
        feature="Heterogeneous structure: 10 independent noise objectives (f1..f10) and 2 redundant blocks of 5.",
        intuition="10 random independent variables mixed with 2 redundant groups of 5 sensors each. MISDA should keep 10 + 2 = 12 objectives.",
        graph_expected="10 isolated nodes and 2 disjoint complete subgraphs of 5 nodes each (10 x K_1 + 2 x K_5)",
    )
    return df, truth


def make_case7_pure_conflict_groups(
    N=1000, M=20, noise=0.05, seed=123, **kwargs
):
    rng = np.random.default_rng(seed)
    if M < 2:
        raise ValueError("M must be >= 2")
    M_pos = (M + 1) // 2
    M_neg = M - M_pos
    x = rng.normal(size=N)
    Y_pos = np.column_stack(
        [x + noise * rng.normal(size=N) for _ in range(M_pos)]
    )
    Y_neg = np.column_stack(
        [(-x) + noise * rng.normal(size=N) for _ in range(M_neg)]
    )
    Y = np.column_stack([Y_pos, Y_neg])
    cols = [f"f{i+1}" for i in range(M)]
    Y = pd.DataFrame(Y, columns=cols)
    truth = _truth(
        name="Case 7 - Structural conflict (anti-corr) 2-groups",
        latent_expected=1,
        structural_expected=2,
        blocks_expected=[cols[:M_pos], cols[M_pos:]],
        feature="Two groups (+x and -x) with internal redundancy and strong structural conflict (anti-correlation).",
        intuition="10 sensors measuring Car Speed (+x) and 10 measuring Remaining Travel Time (-x). Speed and Time conflict, but both are essential! MISDA must keep 1 of each.",
        graph_expected="2 disjoint complete subgraphs of 10 nodes each (2 x K_10, 90 total edges, 2 connected components)",
    )
    return Y, truth


CANONICAL_CASES = [
    ("Case 1 - Total independence", make_case1_independence),
    ("Case 2 - Total redundancy", make_case2_total_redundancy),
    ("Case 3 - Blocks (4 x 5)", make_case3_block_structure),
    ("Case 4 - Blocks (2 x 10)", make_case4_two_big_blocks),
    ("Case 5 - Chain", make_case5_chain_structure),
    ("Case 6 - Mixed (indep + latents)", make_case6_mixed_structure),
    (
        "Case 7 - Structural conflict (anti-corr) with groups",
        make_case7_pure_conflict_groups,
    ),
]
