
import numpy as np
import pandas as pd

"""
mop_definitions.py - Synthetic Multi-Objective Problem Generators

This module defines test cases (MOP-A to MOP-F) for verifying the MISDA algorithm.
Each generator produces a synthetic dataset (N samples x M objectives) with
known ground truth regarding intrinsic dimensionality and redundancy structure.
"""

def _mop_truth(
    name,
    latent_expected,
    structural_expected,
    blocks_expected,
    notes="",
    feature="",
    intuition="",
    graph_expected="",
):
    return {
        "name": name,
        "latent_expected": int(latent_expected),
        "structural_expected": int(structural_expected),
        "blocks_expected": blocks_expected,  # list of lists of names "f1","f2",...
        "notes": notes,
        "feature": feature,
        "intuition": intuition,
        "graph_expected": graph_expected,
    }

def _mop_df(Y):
    return pd.DataFrame(Y, columns=[f"f{i+1}" for i in range(Y.shape[1])])

def _mk_block_names(start, size):
    # start is 1-based
    return [f"f{i}" for i in range(start, start + size)]

def _repeat_with_small_noise(base, rng, noise):
    # base: (N,) -> returns perturbed (N,)
    return base + noise * rng.normal(size=base.shape[0])


# ------------------------------------------------------------
# MOP-A — Monotonic redundancy (1D) with 20 objectives
# Expected: dim=1; single block of 20
# ------------------------------------------------------------
def mopA_monotonic_redundancy(N=1000, seed=123, noise=0.0):
    """
    MOP-A: Monotonic Redundancy (M=20).
    
    Generates 20 objectives that are all monotonic transformations of a single latent variable.
    Tests the algorithm's ability to detect non-linear (but monotonic) redundancy.
    
    Expected Intrinsic Dimension: 1
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=N)

    # 20 monotonic transformations (all 1D redundant)
    feats = [
        x,
        2.0 * x + 0.1,
        np.log(1.0 + 9.0 * x),
        x**2,
        np.sqrt(np.maximum(x, 0.0)),
        x**3,
        np.exp(0.5 * x) - 1.0,
        1.0 / (1.0 + np.exp(-10.0 * (x - 0.5))),
        (x + 0.2) ** 2,
        np.log(1.0 + 3.0 * x),
        np.tanh(2.0 * x),
        (1.0 + x) ** 1.5,
        np.clip(x + 0.05, 0, 1),
        np.clip(1.2 * x, 0, 1),
        np.log1p(20.0 * x) / np.log1p(20.0),
        (x + 1e-6) ** 0.25,
        (x + 0.1) ** 3,
        np.sqrt(np.maximum(0.1 + x, 0.0)),
        np.exp(x) - 1.0,
        (x + 0.3) ** 2,
    ]
    Y = np.vstack([_repeat_with_small_noise(f, rng, noise) for f in feats]).T

    truth = _mop_truth(
        name="MOP-A — Monotonic redundancy (1D, M=20)",
        latent_expected=1,
        structural_expected=1,
        blocks_expected=[_mk_block_names(1, 20)],
        notes="20 objectives as monotonic (and redundant) transformations of the same latent x.",
        feature="20 non-linear monotonic transformations driven by a single 1D decision variable x.",
        intuition="20 different formulas (squares, roots, logs) calculated from a single input x. Since all move in sync, MISDA should collapse all 20 to 1.",
        graph_expected="1 fully connected graph (K_20, 190 edges, 1 connected component)",
    )
    return _mop_df(Y), truth


def mopB_tradeoff_with_redundancies(N=1000, seed=123, noise=0.02):
    """
    MOP-B: Trade-off with Redundancies (M=20).
    
    Simulates a 3-objective problem (Cost, Consumption, Performance) where Performance
    depends on the others, creating a 2D manifold. Objectives are expanded with redundancies.
    
    Expected Intrinsic Dimension: 2
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.0, 1.0, size=N)
    b = rng.uniform(0.0, 1.0, size=N)

    # Plausible latents
    C = 0.6 * a + 0.8 * b            # cost in ~[0,1.4]
    E = b + 0.3 * (1.0 - a)          # consumption in ~[0,1.3]
    P = a * (1.0 - b) + 0.2 * a      # performance can go up to 1.2 -> BUG for Q

    # FIX: force performance to stay in [0,1] so that Q=1-P stays in [0,1]
    P = np.clip(P, 0.0, 1.0)
    Q = 1.0 - P

    # 7 "cost" objectives
    cost_feats = [
        C,
        _repeat_with_small_noise(C, rng, noise),
        1.0 + 2.0 * C,
        np.log1p(9.0 * C),
        np.sqrt(np.maximum(C, 0.0)),
        C**2,
        (C + 0.1) ** 1.5,
    ]

    # 7 "consumption" objectives
    cons_feats = [
        E,
        _repeat_with_small_noise(E, rng, noise),
        np.sqrt(np.maximum(E, 0.0)),
        np.log1p(9.0 * E),
        E**2,
        (E + 0.05),
        (E + 0.2) ** 1.3,
    ]

    # 6 "performance" objectives (minimization via 1-P), with protected domain
    Q_rep = np.clip(_repeat_with_small_noise(Q, rng, noise), 0.0, 1.0)

    perf_feats = [
        Q,
        Q_rep,
        Q**2,
        np.sqrt(np.maximum(Q, 0.0)),
        np.log1p(9.0 * Q),            # now Q ∈ [0,1] -> always valid
        (Q + 0.1) ** 1.2,             # now Q+0.1 ∈ [0.1,1.1] -> always valid
    ]

    feats = cost_feats + cons_feats + perf_feats
    Y = np.vstack(feats).T

    truth = _mop_truth(
        name="MOP-B — Trade-off + redundancies (~2D, M=20)",
        latent_expected=2,
        structural_expected=2,
        blocks_expected=[_mk_block_names(1, 7), _mk_block_names(8, 7), _mk_block_names(15, 6)],
        notes="Three families (cost/consumption/performance) with internal redundancies; effective tends to ~2.",
        feature="Three functional engineering families (7 cost, 7 consumption, 6 performance) driven by 2 decision variables.",
        intuition="An engineering problem with 3 main goals: Cost, Energy, and Performance, each measured in multiple redundant ways. MISDA should shrink 20 to ~2-3 core trade-offs.",
        graph_expected="1 connected graph with 3 dense functional clusters (1 connected component, effective dim ~2)",
    )
    return _mop_df(Y), truth



# ------------------------------------------------------------
# MOP-C — Latent blocks (4 independent factors) with 20 objectives
# Here: 4 blocks of 5 (total 20). Expected: dim=4.
# ------------------------------------------------------------
def mopC_latent_blocks_4x5(N=1000, seed=123, noise=0.02):
    """
    MOP-C: Latent Blocks (4x5, M=20).
    
    Generates 4 independent latent factors. Each factor drives a block of 5
    redundant objectives (some non-linear).
    
    Expected Intrinsic Dimension: 4
    """
    rng = np.random.default_rng(seed)
    u, v, w, z = rng.uniform(0.0, 1.0, size=(4, N))
    eps = rng.normal(size=N)

    b1 = [u, 2*u, u**2, np.sqrt(np.maximum(u,0.0)), np.log1p(9*u)]
    b2 = [v, v+0.5, np.log1p(9*v), v**2, np.sqrt(np.maximum(v,0.0))]
    b3 = [w, w+noise*eps, np.sqrt(np.maximum(w,0.0)), np.log1p(9*w), (w+0.1)**2]
    b4 = [z, (1.0+z)**2, np.exp(z)-1.0, np.log1p(9*z), np.sqrt(np.maximum(z,0.0))]

    feats = b1 + b2 + b3 + b4
    Y = np.vstack(feats).T

    truth = _mop_truth(
        name="MOP-C — Latent blocks (4×5, M=20)",
        latent_expected=4,
        structural_expected=4,
        blocks_expected=[_mk_block_names(1,5), _mk_block_names(6,5), _mk_block_names(11,5), _mk_block_names(16,5)],
        notes="Four independent factors; each block (5 objectives) is internally redundant.",
        feature="4 independent decision factors; each factor generates a block of 5 non-linearly transformed objectives.",
        intuition="4 control dials, where turning each dial affects 5 non-linear indicators. MISDA should extract 4 independent representatives (1 per dial).",
        graph_expected="4 disjoint dense subgraphs of 5 nodes each (4 x K_5, 4 connected components)",
    )
    return _mop_df(Y), truth


# ------------------------------------------------------------
# MOP-D — Pure conflict (anti-corr) with 20 objectives
# Idea: 2 internally redundant groups (+x and 1-x), conflicting with each other.
# (This is, in practice, your Case 7 in 'MOP' format.)
# Expected: conflict must be preserved; internal redundancy can be reduced.
# ------------------------------------------------------------
def mopD_pure_conflict_groups(N=1000, seed=123, noise=0.0):
    """
    MOP-D: Pure Conflict Groups (M=20).
    
    Two large groups of internally redundant objectives. The two groups are 
    strongly anti-correlated (conflicting). Tests preservation of conflict.
    
    Expected Intrinsic Dimension: 2
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=N)

    g1 = [
        x,
        2*x + 0.1,
        np.log1p(9*x),
        x**2,
        np.sqrt(np.maximum(x,0.0)),
        x**3,
        np.tanh(2*x),
        np.log1p(3*x),
        (x+0.2)**2,
        (1.0 + x)**1.5,
    ]
    y = 1.0 - x
    g2 = [
        y,
        2*y + 0.1,
        np.log1p(9*y),
        y**2,
        np.sqrt(np.maximum(y,0.0)),
        y**3,
        np.tanh(2*y),
        np.log1p(3*y),
        (y+0.2)**2,
        (1.0 + y)**1.5,
    ]

    feats = [_repeat_with_small_noise(f, rng, noise) for f in (g1 + g2)]
    Y = np.vstack(feats).T

    truth = _mop_truth(
        name="MOP-D — Structural conflict (anti-corr) 2-groups (M=20)",
        latent_expected=1,
        structural_expected=2,
        blocks_expected=[_mk_block_names(1,10), _mk_block_names(11,10)],
        notes="Two internally redundant groups (+x and 1-x), but antagonistic to each other: conflict must be preserved.",
        feature="Two antagonistic non-linear objective families (+x vs 1-x) with internal redundancy and trade-off conflict.",
        intuition="10 indicators measuring Benefit (+x) vs 10 measuring Risk (1-x). Benefit and Risk directly conflict. MISDA must preserve 1 Benefit and 1 Risk indicator.",
        graph_expected="2 disjoint complete subgraphs of 10 nodes each (2 x K_10, 90 total edges, 2 connected components)",
    )
    return _mop_df(Y), truth


# ------------------------------------------------------------
# MOP-E — Partial redundancy + noise + new objective + mixtures with 20 objectives
# Here: three subfamilies (10 + 4 + 6 = 20) maintaining the original idea.
# Expected: dim≈2 (maintained).
# ------------------------------------------------------------
def mopE_partial_redundancy_noisy(N=1000, seed=123, noise=0.05):
    """
    MOP-E: Partial Redundancy + Noise (M=20).
    
    Mixture of redundant groups and compound objectives (sums/mixtures).
    Includes significant noise to test robustness.
    
    Expected Intrinsic Dimension: 2
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.0, 1.0, size=N)
    b = rng.uniform(0.0, 1.0, size=N)
    eps = rng.normal(size=N)

    # subfamily A: redundant around 'a' (10)
    A = [
        a,
        a + noise*eps,
        a - noise*eps,
        2*a + 0.1,
        a**2,
        np.sqrt(np.maximum(a,0.0)),
        np.log1p(9*a),
        (a+0.2)**2,
        np.tanh(2*a),
        (1.0+a)**1.2,
    ]

    # subfamily B: "b" (4)
    B = [
        b,
        b + 0.5,
        np.sqrt(np.maximum(b,0.0)),
        np.log1p(9*b),
    ]

    # mixtures/compounds: functions of s=a+b (6)
    s = a + b
    C = [
        s,
        s**2,
        np.sqrt(np.maximum(s,0.0)),
        np.log1p(9*s),
        (s+0.1)**1.5,
        1.0/(1.0+np.exp(-10*(s-1.0))),
    ]

    feats = A + B + C
    Y = np.vstack(feats).T

    truth = _mop_truth(
        name="MOP-E — Partial redundancy + noise (M=20)",
        latent_expected=2,
        structural_expected=2,
        blocks_expected=[_mk_block_names(1,10), _mk_block_names(11,4), _mk_block_names(15,6)],
        notes="Trio/quartet of 'a' extended to 10 redundants; 'b' (4); and 6 compounds around s=a+b.",
        feature="Partial redundancy across 2 latent drivers (a,b): 10 objectives on a, 4 on b, and 6 compounds on s=a+b.",
        intuition="Overlapping signals: some indicators monitor Engine A, some monitor Engine B, and some monitor both combined (A+B). Tests if MISDA untangles blended signals.",
        graph_expected="1 connected graph with 3 dense overlapping clusters (Subfamily A: 10, B: 4, C: 6)",
    )
    return _mop_df(Y), truth


# ------------------------------------------------------------
# MOP-F — Regimes (mixture) with 20 objectives
# Here: 10 objectives based on L (mixture by regime) + 10 based on b
# Expected: dim≈2 (maintained), but global correlation can be misleading.
# ------------------------------------------------------------
def mopF_regime_switching(N=1000, seed=123, sharpness=20.0, noise=0.0):
    """
    MOP-F: Regime Switching / Mixture (M=20).
    
    A variable L switches behavior between 'a' and 'b' depending on the value of 'a'.
    Tests the algorithm's behavior on manifold mixtures/discontinuities.
    
    Expected Intrinsic Dimension: 2 (but may be detected as 1 due to collapse).
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.0, 1.0, size=N)
    b = rng.uniform(0.0, 1.0, size=N)

    s = 1.0 / (1.0 + np.exp(-sharpness * (a - 0.5)))
    L = (1.0 - s) * a + s * b

    eps = rng.normal(size=N)

    L_feats = [
        L,
        L**2,
        np.log1p(9*L),
        np.sqrt(np.maximum(L,0.0)),
        (L+0.1)**1.5,
        np.tanh(2*L),
        np.exp(0.5*L)-1.0,
        (L+0.2)**2,
        np.log1p(3*L),
        _repeat_with_small_noise(L, rng, 0.02) if noise == 0.0 else _repeat_with_small_noise(L, rng, noise),
    ]

    b_feats = [
        b,
        np.sqrt(np.maximum(b,0.0)),
        np.log1p(9*b),
        b**2,
        (b+0.1)**1.5,
        np.tanh(2*b),
        np.exp(0.5*b)-1.0,
        (b+0.2)**2,
        np.log1p(3*b),
        _repeat_with_small_noise(b, rng, 0.02) if noise == 0.0 else _repeat_with_small_noise(b, rng, noise),
    ]

    feats = L_feats + b_feats
    Y = np.vstack(feats).T

    truth = _mop_truth(
        name="MOP-F — Regimes (mixture, M=20)",
        latent_expected=2,
        structural_expected=2,
        blocks_expected=[_mk_block_names(1,10), _mk_block_names(11,10)],
        notes="10 objectives redundant around L (mixture by regime) + 10 redundant around b; global correlation can be misleading.",
        feature="Non-linear regime-switching mixture: 10 objectives on regime-dependent mixture L(a,b) and 10 on b.",
        intuition="System switching: indicators change behavior depending on whether the system operates in High-Power or Low-Power mode. Tests MISDA under shifting states.",
        graph_expected="1 connected graph with 2 dense interconnected clusters (10 on mixture L, 10 on b)",
    )
    return _mop_df(Y), truth


# ------------------------------------------------------------
# DTLZ Suite (Geometric Manifolds)
# ------------------------------------------------------------

def generate_dtlz2(N=1000, M=3, n_vars=12, on_front=False):
    """
    Generates N samples of DTLZ2 with M objectives.
    """
    import math
    rng = np.random.default_rng()
    k = n_vars - M + 1
    X = rng.uniform(0.0, 1.0, size=(N, n_vars))
    if on_front:
        X[:, (M-1):] = 0.5
    xm = X[:, (M-1):] 
    g = np.sum((xm - 0.5)**2, axis=1)
    F = np.zeros((N, M))
    for i in range(M):
        f = (1.0 + g)
        for j in range(M - 1 - i):
            f *= np.cos(X[:, j] * math.pi / 2.0)
        if i > 0:
            f *= np.sin(X[:, M - 1 - i] * math.pi / 2.0)
        F[:, i] = f
    return F, X

def generate_dtlz5(N=1000, M=3, n_vars=12, on_front=False):
    """
    Generates N samples of DTLZ5 (Degenerate curve).
    """
    import math
    rng = np.random.default_rng()
    k = n_vars - M + 1
    X = rng.uniform(0.0, 1.0, size=(N, n_vars))
    if on_front:
        X[:, (M-1):] = 0.5
    xm = X[:, (M-1):]
    g = np.sum((xm - 0.5)**2, axis=1)
    theta = np.zeros((N, M-1))
    theta[:, 0] = X[:, 0] * math.pi / 2.0
    gr = g[:, np.newaxis]
    for i in range(1, M-1):
        theta[:, i] = ((math.pi / (4.0 * (1.0 + gr))) * (1.0 + 2.0 * gr * X[:, i][:, np.newaxis])).ravel()
    F = np.zeros((N, M))
    for i in range(M):
        f = (1.0 + g)
        for j in range(M - 1 - i):
            f *= np.cos(theta[:, j])
        if i > 0:
            f *= np.sin(theta[:, M - 1 - i])
        F[:, i] = f
    return F, X
