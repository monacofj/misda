
import numpy as np
import pandas as pd
import misda

def make_case3_block_structure(N=1000, M=20, seed=123):
    rng = np.random.default_rng(seed)
    assert M == 20
    latent_blocks = rng.normal(size=(N, 4))
    Y = np.zeros((N, M))
    for b in range(4):
        for j in range(5):
            idx = 5*b + j
            Y[:, idx] = latent_blocks[:, b] + np.random.normal(scale=0.2, size=N)
    cols = [f"f{i+1}" for i in range(M)]
    df = pd.DataFrame(Y, columns=cols)
    return df

def make_case4_two_big_blocks(N=1000, M=20, seed=123):
    rng = np.random.default_rng(seed)
    assert M == 20
    latent_blocks = rng.normal(size=(N, 2))
    Y = np.zeros((N, M))
    for i in range(10):
        Y[:, i] = latent_blocks[:, 0] + np.random.normal(scale=0.2, size=N)
    for i in range(10, 20):
        Y[:, i] = latent_blocks[:, 1] + np.random.normal(scale=0.2, size=N)
    cols = [f"f{i+1}" for i in range(M)]
    df = pd.DataFrame(Y, columns=cols)
    return df

print("=== Testing Case 3 (Expected Dim: 4) ===")
Y3 = make_case3_block_structure()
# Using current working directory for imports
import os
import sys
# sys.path.insert(0, os.getcwd()) # Not needed if misda is installed or correctly namespaced
res3 = misda.analyze(Y3, name="Case 3")
print(f"Result Dim: {len(res3.best_mis.indices)}")
print(f"MIS Indices: {res3.best_mis.indices}")

print("\n=== Testing Case 4 (Expected Dim: 2) ===")
Y4 = make_case4_two_big_blocks()
res4 = misda.analyze(Y4, name="Case 4")
print(f"Result Dim: {len(res4.best_mis.indices)}")
print(f"MIS Indices: {res4.best_mis.indices}")
