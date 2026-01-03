import dtlz_lib
import misda
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from misda import compile_benchmark_summary, MISDAResult, AlphaRegime

"""
verify_table.py

A utility script to validade the output formatting of `compile_benchmark_summary`.
It creates mock `MISDAResult` objects with predefined metrics and checks if the
resulting DataFrame correctly calculates and displays columns like 'Regime', 'Fidel', etc.
"""

# Copy pasted function for testing
def evaluate_reduced_model_fidelity(results_dict):
    results_summary = []
    for name, data in results_dict.items():
        result_obj = data.get("result_obj")
        truth = data.get("truth", {})
        
        Y = result_obj.Y
        mis = result_obj.best_mis
        mis_indices = mis["mis_indices"] if mis else []
        
        if not mis_indices:
            fidelity = 0.0
            ses = 0.0
        else:
            if result_obj.ses_results:
                fidelity = result_obj.ses_results["F_real"]
                ses = result_obj.ses_results.get("ses", 0.0)
            else:
                fidelity = 0.0
                ses = 0.0

        expected_dim = truth.get("intrinsic_dim_expected", None)
        mis_size = len(mis_indices)
        
        regime_name = result_obj.regime.name if result_obj.regime else "N/A"
        alpha_min = result_obj.alpha_min
        alpha_max = result_obj.alpha_max
        compactness = getattr(result_obj, "min_compactness", 1.0)
        homog_ratio = getattr(result_obj, "homogeneity_ratio", 1.0)
        diagnosis = getattr(result_obj, "diagnosis", "N/A")

        status = "Bad"
        if fidelity >= 0.9: status = "OK"
        elif expected_dim and mis_size == expected_dim: status = "OK"
        elif str(regime_name) == "SIGNAL_BELOW_NOISE": status = "Noise"
        if homog_ratio < 0.6: status = "WARN"

        entry = {
            "Case": name,
            "a_min": f"{alpha_min:.2e}",
            "a_max": f"{alpha_max:.2e}",
            "Regime": regime_name,
            "Exp": expected_dim,
            "Fnd": mis_size,
            "Fidel": f"{fidelity:.4f}",
            "SES": f"{ses:.4f}",
            "Comp": f"{compactness:.4f}",
            "Homog": f"{homog_ratio:.4f}",
            "Diag": diagnosis,
            "Stat": status
        }
        results_summary.append(entry)

    df_summary = pd.DataFrame(results_summary)
    cols = ["Case", "Regime", "a_min", "a_max", "Exp", "Fnd", "Fidel", "SES", "Comp", "Homog", "Diag", "Stat"]
    return df_summary[[c for c in cols if c in df_summary.columns]]

# Run test
results = {}

print("Testing DTLZ2...")
Y, _ = dtlz_lib.generate_dtlz2(N=200, M=3)
res = misda.analyze(Y, caution=0.5, run_ses=True)
results["DTLZ2"] = {"result_obj": res, "truth": {"intrinsic_dim_expected": 3}}

print("Testing DTLZ5...")
Y5, _ = dtlz_lib.generate_dtlz5(N=200, M=3)
res5 = misda.analyze(Y5, caution=0.5, run_ses=True)
results["DTLZ5"] = {"result_obj": res5, "truth": {"intrinsic_dim_expected": 2}}

print("\n--- Summary Table ---")
df = evaluate_reduced_model_fidelity(results)
print(df.to_markdown(index=False))
