# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Legacy benchmark-summary infrastructure."""

import pandas as pd

from ._pareto import evaluate_pareto_consistency
from ._reconstruction import calculate_ses_nonlinear
from ._statistics import _CORRELATION_MODE


def compile_benchmark_summary(results_dict, sort_by=None):
    """
    Standardizes the summary table generation for MISDA benchmarks.
    Extracts all key metrics including Alpha regimes, Homogeneity, Linear/Non-Linear Fidelity, and Pareto Consistency.

    Args:
        results_dict (dict): Dictionary mapping Case Name -> MISDAResult object.
                             Alternatively, can be a dictionary where values are dicts containing 'result_obj'.
        sort_by (str): Column name to sort by.

    Returns:
        pd.DataFrame: Comprehensive summary table.
    """
    rows = []

    for case_name, item in results_dict.items():
        # Handle both direct MISDAResult and wrapper dicts (e.g. from benchmark.ipynb)
        # item could be MISDAResult or dict
        res = None
        truth_dim = None

        if hasattr(item, 'best_mis'):
            res = item
        elif isinstance(item, dict) and 'result_obj' in item:
            res = item['result_obj']
            truth_dim = item.get('truth', {}).get('structural_expected' if _CORRELATION_MODE == "positive" else 'latent_expected', item.get('truth', {}).get('intrinsic_dim_expected', None))

        if res is None:
            continue

        # Basic Params
        # Handle res.Y being dataframe or numpy
        N, M = res.Y.shape
        algo_alpha = res.alpha

        # MIS Info
        mis_indices = res.best_mis.indices if res.best_mis else []
        dim_red = len(mis_indices)

        # Fidelity (Linear)
        fidel_lin = None
        if res.ses_results and isinstance(res.ses_results, dict):
            fidel_lin = res.ses_results.get("F_real", None)

        # Fidelity (Non-Linear)
        fidel_nl = None
        try:
            if N <= 5000: # Per safeguard
                nl_out = calculate_ses_nonlinear(res.Y, mis_indices, n_estimators=50, return_details=True)
                if isinstance(nl_out, dict):
                    fidel_nl = nl_out.get("F_real", None)
                else:
                    fidel_nl = nl_out
        except Exception:
            pass

        # Pareto Consistency
        prec, rec = 0.0, 0.0
        try:
             prec, rec = evaluate_pareto_consistency(res, res.Y)
        except Exception:
             pass

        # Homogeneity
        homog = res.homogeneity_ratio

        # Status
        status = "OK"
        low_lin = (fidel_lin is not None and fidel_lin < 0.9)
        low_nl = (fidel_nl is not None and fidel_nl < 0.9)
        if low_lin and low_nl:
            status = "LOW_FIDEL"
        if prec < 1.0:
            status = "UNSAFE(Prec)"
        if truth_dim and dim_red != truth_dim:
             status += f"|DimMismatch({dim_red}!={truth_dim})"

        # Alpha Bounds
        a_min = res.alpha_min if hasattr(res, 'alpha_min') else 0.0
        a_max = res.alpha_max if hasattr(res, 'alpha_max') else 1.0

        def _fmt_f(val):
            return f"{val:.2f}" if val is not None else "N/A"

        row = {
            "Case": case_name,
            "Regime": res.regime.name if res.regime else "N/A",
            "N": N,
            "M": M,
            "Dim(Red)": dim_red,
            "Alpha": f"{algo_alpha:.4f}",
            "Min": f"{a_min:.2f}",
            "Max": f"{a_max:.2f}",
            "Homog": f"{homog:.2f}",
            "Fidel(Lin)": _fmt_f(fidel_lin),
            "Fidel(NL)": _fmt_f(fidel_nl),
            "Prec": f"{prec:.2f}",
            "Rec": f"{rec:.2f}",
            "Status": status
        }

        if truth_dim is not None:
            row["Exp"] = truth_dim

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Reorder columns if Exp exists
    cols = ["Case", "Regime", "N", "M", "Exp", "Dim(Red)", "Alpha", "Min", "Max", "Homog", "Fidel(Lin)", "Fidel(NL)", "Prec", "Rec", "Status"]
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols]

    if sort_by and sort_by in df.columns:
        df = df.sort_values(by=sort_by)

    return df
