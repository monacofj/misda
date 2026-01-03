```python
# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import misda
import importlib

"""
This script contains functions to generate synthetic datasets for testing and
demonstrating the MISDA library. Each function creates a DataFrame (Y) and
a dictionary (truth) describing the expected intrinsic dimensionality and
block structure.

It also includes a main execution loop to run MISDA analysis on these
generated cases and print a summary.
"""

# 1. Data Generators
def _truth(name, intrinsic_dim_expected, blocks_expected, notes=""):
    return {
        "name": name,
        "intrinsic_dim_expected": int(intrinsic_dim_expected),
        "blocks_expected": blocks_expected,
        "notes": notes,
    }

def make_case1_independence(N=1000, M=20, seed=123):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(N, M))
    cols = [f"f{i+1}" for i in range(M)]
    df = pd.DataFrame(Y, columns=cols)
    truth = _truth(
        name="Case 1 - Total independence",
        intrinsic_dim_expected=M,
        blocks_expected=[[c] for c in cols],
        notes="Each objective is independent (Gaussian noise)."
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
        intrinsic_dim_expected=1,
        blocks_expected=[cols],
        notes="A single latent; all objectives are noisy copies."
    )
    return df, truth

def make_case3_block_structure(N=1000, M=20, seed=123):
    rng = np.random.default_rng(seed)
    assert M == 20
    latent_blocks = rng.normal(size=(N, 4))
    Y = np.zeros((N, M))
    for b in range(4):
        for j in range(5):
            idx = 5*b + j
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
        intrinsic_dim_expected=4,
        blocks_expected=blocks,
        notes="4 independent latents; each generates 5 objectives."
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
        intrinsic_dim_expected=2,
        blocks_expected=[
            [f"f{i}" for i in range(1, 11)],
            [f"f{i}" for i in range(11, 21)],
        ],
        notes="2 independent latents; each generates 10 objectives."
    )
    return df, truth

def make_case5_chain_structure(N=1000, M=20, seed=123):
    rng = np.random.default_rng(seed)
    Y = np.zeros((N, M))
    Y[:, 0] = rng.normal(size=N)
    for j in range(1, M):
        Y[:, j] = Y[:, j-1] + rng.normal(scale=0.2, size=N)
    cols = [f"f{i+1}" for i in range(M)]
    df = pd.DataFrame(Y, columns=cols)
    truth = _truth(
        name="Case 5 - Chain",
        intrinsic_dim_expected=M,
        blocks_expected=[cols],
        notes="Sequential dependency."
    )
    return df, truth

def make_case6_mixed_structure(N=1000, M=20, seed=123):
    rng = np.random.default_rng(seed)
    assert M == 20
    Y = np.zeros((N, M))
    # First 10: independent
    Y[:, :10] = rng.normal(size=(N, 10))
    # Last 10: two latents
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
        intrinsic_dim_expected=12,
        blocks_expected=[[f"f{i}" for i in range(1, 11)] + [[f"f{i}" for i in range(11, 16)], [f"f{i}" for i in range(16, 21)]]],
        notes="f1..f10 independent; f11..f15 latent1; f16..f20 latent2."
    )
    return df, truth

def make_case7_pure_conflict_groups(N=1000, M=20, noise=0.05, seed=123, **kwargs):
    rng = np.random.default_rng(seed)
    if M < 2:
        raise ValueError("M must be >= 2")
    M_pos = (M + 1) // 2
    M_neg = M - M_pos
    x = rng.normal(size=N)
    Y_pos = np.column_stack([x + noise * rng.normal(size=N) for _ in range(M_pos)])
    Y_neg = np.column_stack([(-x) + noise * rng.normal(size=N) for _ in range(M_neg)])
    Y = np.column_stack([Y_pos, Y_neg])
    cols = [f"f{i+1}" for i in range(M)]
    Y = pd.DataFrame(Y, columns=cols)
    truth = {
        "name": f"Case 7 - Structural conflict (anti-corr) 2-groups",
        "intrinsic_dim_expected": 2,
        "blocks_expected": [cols[:M_pos], cols[M_pos:]],
        "notes": "Conflict groups (+x and -x).",
    }
    return Y, truth

# --- MOP Cases ---
def _make_mop_case(name, N, M, seed):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(N, M))
    cols = [f"f{i+1}" for i in range(M)]
    df = pd.DataFrame(Y, columns=cols)
    truth = {
        "name": name,
        "intrinsic_dim_expected": M,
        "blocks_expected": [[c] for c in cols],
        "notes": "MOP placeholder structure."
    }
    return df, truth

def make_mop_dtlz2_3d(N=1000, M=3, seed=123):
    return _make_mop_case(name=f"DTLZ2 (M={M})", N=N, M=M, seed=seed)

def make_mop_wfg_3d(N=1000, M=3, seed=123):
    return _make_mop_case(name=f"WFG (M={M})", N=N, M=M, seed=seed)


# 2. Main Execution Loop
caution = misda.CONSERVATIVE # Set caution level (0 to 1)

def run_cases(cases_list, N=200): # Reduced N for speed
    summary_data = []

    for name, gen in cases_list:
        print("\n\n" + "=" * 70)
        print(name)
        print("=" * 70)

        Y, truth = gen(N=N)
        print(f"[OK] Data generated: Y.shape = {Y.shape}")
        
        # 1. Pipeline execution
        # Use high-level analyze for convenience as it handles everything
        result = misda.analyze(Y, caution=caution, run_ses=True, name=name)
        
        # Print standard reports
        print(misda.describe_alpha_regime(result.metrics))
        print(result.summary())
        
        # Collect metrics for final table
        mis_size = len(result.best_mis['mis_indices']) if result.best_mis else 0
        exp_dim = truth.get('intrinsic_dim_expected', 0)
        
        f_real = 0.0
        ses = 0.0
        if result.ses_results:
            f_real = result.ses_results.get('F_real', 0.0)
            ses = result.ses_results.get('ses', 0.0)
            
        # Determine Status
        # Simple heuristic: Good if Fidelity is high (>0.9) OR if MIS size matches expectation (for cases where we know dim)
        status = "Bad"
        if f_real >= 0.9:
            status = "OK"
        elif exp_dim > 0 and mis_size == exp_dim: # If dimensionality matches exactly
            status = "OK"
        elif result.regime == misda.AlphaRegime.SIGNAL_BELOW_NOISE:
             status = "Noise"
             
        summary_data.append({
            "Case": name,
            "Regime": result.regime.name,
            "Alpha Min": f"{result.alpha_min:.2e}",
            "Alpha Max": f"{result.alpha_max:.2e}",
            "Exp Dim": exp_dim,
            "MIS Size": mis_size,
            "F_real": f"{f_real:.4f}",
            "SES": f"{ses:.4f}",
            "Status": status
        })
        
    return pd.DataFrame(summary_data)

battery1 = [
    ("Case 1 - Total independence", make_case1_independence),
    ("Case 2 - Total redundancy", make_case2_total_redundancy),
    ("Case 3 - Blocks (4 x 5)", make_case3_block_structure),
    ("Case 4 - Blocks (2 x 10)", make_case4_two_big_blocks),
    ("Case 5 - Chain", make_case5_chain_structure),
    ("Case 6 - Mixed (indep + latents)", make_case6_mixed_structure),
    ("Case 7 - Structural conflict (anti-corr) with groups", make_case7_pure_conflict_groups),
]

battery2 = [
    ("Case 8 - DTLZ2 (M=3)", make_mop_dtlz2_3d),
    ("Case 9 - WFG (M=3)", make_mop_wfg_3d),
]

print("\n=== RUNNING STANDARD CORRELATION BATTERY ===")
df1 = run_cases(battery1, N=300)

print("\n\n=== RUNNING MOP BENCHMARK BATTERY ===")
df2 = run_cases(battery2, N=300)

print("\n\n" + "="*100)
print("FINAL SUMMARY REPORT")
print("="*100)
final_df = pd.concat([df1, df2], ignore_index=True)
print(final_df.to_string(index=False))

