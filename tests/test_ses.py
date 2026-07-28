import pytest
import numpy as np
import pandas as pd
import misda
from misda import calculate_ses, calculate_ses_linear, calculate_ses_nonlinear

def test_no_reduction_returns_none():
    """When all features are kept (no reduction), SES and F_real must be None (N/A)."""
    X = np.random.randn(100, 3)
    out_lin = calculate_ses_linear(X, mis=[0, 1, 2], return_details=True)
    assert out_lin["status"] == "NO_REDUCTION"
    assert out_lin["ses"] is None
    assert out_lin["F_real"] is None
    assert out_lin["F_null"] is None

    out_nl_scalar = calculate_ses_nonlinear(X, mis=[0, 1, 2], return_details=False)
    assert out_nl_scalar is None

    out_nl = calculate_ses_nonlinear(X, mis=[0, 1, 2], return_details=True)
    assert out_nl["status"] == "NO_REDUCTION"
    assert out_nl["ses"] is None


def test_linear_dependency():
    """Linear dependency: f2 = 2*f1 + noise. Linear and Non-linear SES should both be high."""
    rng = np.random.default_rng(42)
    f1 = rng.uniform(-5, 5, 200)
    f2 = 2.0 * f1 + rng.normal(0, 0.05, 200)
    Y = np.column_stack([f1, f2])

    res_lin = calculate_ses_linear(Y, mis=[0], seed=42, return_details=True)
    assert res_lin["status"] == "SUCCESS"
    assert res_lin["ses"] > 0.8
    assert res_lin["F_real"] > 0.8

    res_nl = calculate_ses_nonlinear(Y, mis=[0], seed=42, return_details=True)
    assert res_nl["status"] == "SUCCESS"
    assert res_nl["ses"] > 0.8
    assert res_nl["F_real"] > 0.8


def test_nonlinear_quad_dependency():
    """Non-linear quadratic relation: f2 = f1**2 + noise, mis=[0] (target = f2).
    Linear SES will fail (low score), but RF Non-linear SES should score significantly higher.
    """
    rng = np.random.default_rng(123)
    f1 = rng.uniform(-3, 3, 300)
    f2 = (f1 ** 2) + rng.normal(0, 0.1, 300)
    Y = np.column_stack([f1, f2])

    res_lin = calculate_ses_linear(Y, mis=[0], seed=123, return_details=True)
    res_nl = calculate_ses_nonlinear(Y, mis=[0], seed=123, n_estimators=50, return_details=True)

    assert res_lin["status"] == "SUCCESS"
    assert res_nl["status"] == "SUCCESS"

    # Non-linear SES (RF) must strictly outperform Linear SES (OLS) on quadratic data
    assert res_nl["F_real"] > res_lin["F_real"] + 0.3
    assert res_nl["ses"] > res_lin["ses"] + 0.3


def test_constant_target_returns_none_r2():
    """Target objective constant in test set should result in None R2 without division by zero errors."""
    Y = np.zeros((100, 2))
    Y[:, 0] = np.random.randn(100)
    # Y[:, 1] is constantly 0.0

    res = calculate_ses_linear(Y, mis=[0], seed=42, return_details=True)
    assert res["status"] == "UNDEFINED_TARGETS"
    assert res["ses"] is None
    assert res["F_real"] is None


def test_reproducibility():
    """Same seed must yield identical results for both linear and non-linear SES."""
    Y = np.random.default_rng(7).normal(0, 1, (150, 4))
    res1_lin = calculate_ses_linear(Y, mis=[0, 1], seed=999, return_details=True)
    res2_lin = calculate_ses_linear(Y, mis=[0, 1], seed=999, return_details=True)
    assert res1_lin["ses"] == res2_lin["ses"]
    assert res1_lin["F_real"] == res2_lin["F_real"]

    res1_nl = calculate_ses_nonlinear(Y, mis=[0, 1], seed=999, return_details=True)
    res2_nl = calculate_ses_nonlinear(Y, mis=[0, 1], seed=999, return_details=True)
    assert res1_nl["ses"] == res2_nl["ses"]
    assert res1_nl["F_real"] == res2_nl["F_real"]


def test_no_data_leakage_target_exclusion():
    """Targets reconstructed must strictly be T (eliminated objectives)."""
    Y = np.random.randn(100, 4)
    res = calculate_ses_linear(Y, mis=[0, 2], return_details=True)
    assert res["targets_reconstructed"] == ["f2", "f4"]
    assert "f1" not in res["targets_reconstructed"]
    assert "f3" not in res["targets_reconstructed"]


def test_result_ses_properties():
    """Verify property separation between scalar ses_nonlinear and dict ses_nonlinear_results."""
    rng = np.random.default_rng(42)
    f1 = rng.normal(size=200)
    f2 = 2.0 * f1 + rng.normal(scale=0.1, size=200)
    Y = pd.DataFrame({"f1": f1, "f2": f2})

    res = misda.analyze(Y, caution=1.0)
    res.validate(check_linear=True, check_nonlinear=True)

    # ses_nonlinear property should be scalar float or None
    assert isinstance(res.ses_nonlinear, float)
    # ses_nonlinear_results property should be dict
    assert isinstance(res.ses_nonlinear_results, dict)
    assert res.ses_nonlinear_results["ses"] == res.ses_nonlinear


def test_diagnosis_presentation():
    """Verify diagnosis text for no reduction and disjoint cliques."""
    # Case 1: Total independence -> No reduction -> Valid (No Reduction Required)
    Y_indep = np.random.randn(100, 3)
    res1 = misda.analyze(Y_indep, caution=1.0)
    res1.validate()
    assert res1.diagnosis == "Valid (No Reduction Required)"

    # Case 7: Disjoint cliques (+x and -x groups) -> Ideal (Disjoint Cliques)
    rng = np.random.default_rng(123)
    x = rng.normal(size=200)
    pos = np.column_stack([x + 0.01 * rng.normal(size=200) for _ in range(5)])
    neg = np.column_stack([(-x) + 0.01 * rng.normal(size=200) for _ in range(5)])
    Y_c7 = np.column_stack([pos, neg])
    res7 = misda.analyze(Y_c7, caution=1.0)
    res7.validate()
    assert res7.diagnosis == "Ideal (Disjoint Cliques)"


def test_component_report_homogeneity_details():
    """Verify that res.report() formats internal correlation and homogeneity ratio without N/A for components."""
    rng = np.random.default_rng(123)
    x = rng.normal(size=200)
    pos = np.column_stack([x + 0.01 * rng.normal(size=200) for _ in range(5)])
    neg = np.column_stack([(-x) + 0.01 * rng.normal(size=200) for _ in range(5)])
    Y_c7 = np.column_stack([pos, neg])
    res7 = misda.analyze(Y_c7, caution=1.0)
    report_text = res7.report()

    assert "Internal Correlation: [N/A ... N/A]" not in report_text
    assert "Homogeneity: N/A" not in report_text
    assert "Internal Correlation: [" in report_text
    assert "(Tight)" in report_text
