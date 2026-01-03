
import sys
import os
import numpy as np
import pandas as pd
import isda
import mop_definitions

def run_verification():
    """
    Runs a verification process for ISDA (Independent Set Discovery Algorithm)
    against a suite of MOP (Method of Patterns) synthetic datasets (A-F).

    For each MOP case, it performs the following steps:
    1. Generates synthetic data using the MOP's defined characteristics.
    2. Estimates the alpha interval for ISDA.
    3. Selects a conservative alpha value for ISDA execution.
    4. Runs ISDA to identify maximally independent sets (MIS).
    5. Compares the size of the best MIS found by ISDA with the
       expected intrinsic dimensionality provided by the MOP definition.
    6. Reports the status (PASS/WARN/FAIL) for each case and compiles a summary.
    """
    print("Running ISDA Validation on MOP Cases A-F...")
    print("-" * 60)
    
    cases = [
        ("MOP-A", mop_definitions.mopA_monotonic_redundancy),
        ("MOP-B", mop_definitions.mopB_tradeoff_with_redundancies),
        ("MOP-C", mop_definitions.mopC_latent_blocks_4x5),
        ("MOP-D", mop_definitions.mopD_pure_conflict_groups),
        ("MOP-E", mop_definitions.mopE_partial_redundancy_noisy),
        ("MOP-F", mop_definitions.mopF_regime_switching),
    ]

    results_summary = []

    for name, func in cases:
        print(f"\nTesting {name}...")
        try:
            # Generate data
            df, truth = func(N=1000, seed=42)
            
            # Estimate Alpha
            alpha_min, alpha_max, _, _ = isda.estimate_alpha_interval(df)
            metrics = isda.diagnose_alpha_regime(alpha_min, alpha_max)
            
            # Select Alpha (Automatic / Conservative)
            # Using caution=1 (Conservative) as default for robust verification
            alpha_exec = isda.select_alpha(alpha_min, alpha_max, caution=1.0)
            
            # Run ISDA
            isda_results = isda.isda_significance(df, alpha=alpha_exec)
            
            # Check results
            mis_sets = isda_results['mis_sets']
            num_mis = len(mis_sets)
            expected_dim = truth['intrinsic_dim_expected']
            
            # For MOP-D (Conflict), we expect dim=2 (one from each group)
            # For others, typically dim=expected_dim
            
            # Basic validation: is the number of MIS close to expected?
            # Note: Bron-Kerbosch finds ALL MIS. 
            # The "Rank 1" MIS is the one we usually care about for dimensionality reduction.
            # But the 'intrinsic_dim_expected' in truth maps more to the count of independent factors.
            # If ISDA works well, size of MIS (nodes kept) should be small? 
            # WAIT: MIS are the "Independent Sets". 
            # The size of an MIS is the number of independent variables kept.
            # So len(best_mis) should be approx intrinsic_dim_expected.
            
            best_mis = isda_results['mis_ranked'][0]['mis_indices']
            dim_found = len(best_mis)
            
            match = (dim_found == expected_dim)
            status = "PASS" if match else "WARN"
            
            print(f"  -> Truth: Dim={expected_dim}")
            print(f"  -> Found: Dim={dim_found} (Best MIS size)")
            print(f"  -> Status: {status}")
            
            results_summary.append({
                "Case": name,
                "Expected": expected_dim,
                "Found": dim_found,
                "Status": status,
                "Notes": truth['notes']
            })
            
        except Exception as e:
            print(f"  -> ERROR: {e}")
            results_summary.append({
                "Case": name,
                "Expected": "?",
                "Found": "Error",
                "Status": "FAIL",
                "Notes": str(e)
            })

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    df_res = pd.DataFrame(results_summary)
    print(df_res[["Case", "Expected", "Found", "Status"]])
    print("="*60)

if __name__ == "__main__":
    run_verification()
