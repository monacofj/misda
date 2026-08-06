# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Legacy statistical threshold calculations used by MISDA."""

import math
from enum import IntEnum

import numpy as np
from scipy import stats


# Internal Correlation Mode Configuration
# "absolute": |r| — Structural dependence / latent dimension (reconstruction)
# "positive": max(r, 0) — Directional redundancy / Pareto conflict preservation
# (r < 0 preserved)
# _CORRELATION_MODE = "absolute"
_CORRELATION_MODE = "positive"


def _correlation_strength(r):
    """Calculates correlation strength based on internal _CORRELATION_MODE."""
    if _CORRELATION_MODE == "absolute":
        return np.abs(r)
    if _CORRELATION_MODE == "positive":
        return np.maximum(r, 0.0)
    raise ValueError(
        f"Unknown _CORRELATION_MODE: {_CORRELATION_MODE}. "
        "Expected 'absolute' or 'positive'."
    )


def alpha_from_r(r, n):
    """
    Converts a correlation coefficient |r| to a two-tailed p-value (alpha).

    Args:
        r (float): The absolute value of the correlation coefficient.
        n (int): The number of samples.

    Returns:
        float: The two-tailed p-value (alpha).
    """
    r = float(abs(r))
    if r <= 0.0:
        return 1.0

    # Use survival function (sf = 1 - cdf) for better precision at tails
    # Handle r -> 1.0 case implicitly via large z, clamping p at the end
    if r >= 1.0 - 1e-15:
        # Avoid arctanh(1) singularity
        z_stat = np.inf
    else:
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3)
        z_stat = z / se

    # Use sf instead of (1-cdf) to avoid precision loss near 0
    p = 2.0 * stats.norm.sf(abs(z_stat))

    # Clamp to machine epsilon to represent "extremely significant" rather than 0
    # This allows z_crit lookup to return a finite large number instead of inf
    min_float = np.finfo(float).tiny
    if p < min_float:
        p = min_float

    return float(p)


def max_abs_corr(Y):
    """
    Calculates the largest absolute correlation coefficient among columns of Y
    and returns the full correlation matrix.

    Args:
        Y (np.ndarray or pd.DataFrame): Input data.

    Returns:
        tuple: A tuple containing:
            - float: The maximum absolute correlation coefficient.
            - np.ndarray: The correlation matrix.
    """
    if hasattr(Y, "values"):
        data = np.asarray(Y.values, dtype=float)
    else:
        data = np.asarray(Y, dtype=float)
    n, m = data.shape
    corr = np.corrcoef(data, rowvar=False)
    iu = np.triu_indices(m, k=1)
    vals = _correlation_strength(corr[iu])
    r_max = float(vals.max()) if vals.size > 0 else 0.0
    return r_max, corr


def estimate_null_max_r(Y, B=500, random_state=None):
    """
    Estimates, via permutation, the largest absolute correlation coefficient
    expected under the null hypothesis (no correlation).

    Args:
        Y (np.ndarray or pd.DataFrame): Input data.
        B (int): Number of permutations to perform.
        random_state (int or np.random.Generator, optional): Seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - float: The maximum absolute correlation coefficient under the null hypothesis.
            - np.ndarray: Array of maximum absolute correlations from each permutation.
    """
    if hasattr(Y, "values"):
        data = np.asarray(Y.values, dtype=float)
    else:
        data = np.asarray(Y, dtype=float)
    n, m = data.shape
    rng = np.random.default_rng(random_state)
    max_nulls = []
    for _ in range(B):
        perm = np.empty_like(data)
        for j in range(m):
            perm[:, j] = rng.permutation(data[:, j])
        corr_perm = np.corrcoef(perm, rowvar=False)
        iu = np.triu_indices(m, k=1)
        max_nulls.append(_correlation_strength(corr_perm[iu]).max())
    max_nulls = np.asarray(max_nulls, dtype=float)
    r_max_null = float(max_nulls.max()) if max_nulls.size > 0 else 0.0
    return r_max_null, max_nulls


def estimate_alpha_interval(Y, B=500, random_state=0):
    """
    Estimates the (alpha_min, alpha_max) interval from the input data Y.
    alpha_min corresponds to the most significant observed correlation.
    alpha_max corresponds to the most significant correlation expected under the null.

    Args:
        Y (np.ndarray or pd.DataFrame): Input data.
        B (int): Number of permutations for null estimation.
        random_state (int, optional): Seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - float: alpha_min (p-value of the strongest real correlation).
            - float: alpha_max (p-value of the strongest null correlation).
            - float: r_max_real (strongest real correlation).
            - float: r_max_null (strongest null correlation).
    """
    if hasattr(Y, "values"):
        data = np.asarray(Y.values, dtype=float)
    else:
        data = np.asarray(Y, dtype=float)
    n, m = data.shape
    r_max_real, corr_real = max_abs_corr(data)
    r_max_null, null_samples = estimate_null_max_r(
        data, B=B, random_state=random_state
    )
    alpha_min = alpha_from_r(r_max_real, n)
    alpha_max = alpha_from_r(r_max_null, n)
    return alpha_min, alpha_max, r_max_real, r_max_null


def select_alpha(alpha_min: float, alpha_max: float, caution: float) -> float:
    """
    A caution of 1.0 (conservative) targets alpha_max (noise floor) to ensure structural
    integrity by identifying more potential dependencies. A caution of 0.0 (aggressive)
    targets alpha_min (signal floor), prioritizing statistical pureness over structure.

    Args:
        alpha_min (float): The minimum alpha value (most significant real correlation).
        alpha_max (float): The maximum alpha value (most significant null correlation).
        caution (float): A value between 0 and 1, indicating the level of caution.

    Returns:
        float: The selected alpha value.

    Raises:
        ValueError: If caution is not between 0 and 1.
    """
    if not (0 <= caution <= 1):
        raise ValueError("Caution must be between 0 and 1.")
    # Consistent mapping:
    # caution=1.0 -> DEFAULT/STABLE -> alpha_max (Noise floor)
    # caution=0.0 -> SIGNAL_ONLY   -> alpha_min (Signal floor)
    return alpha_min * (1 - caution) + alpha_max * caution


class AlphaRegime(IntEnum):
    SIGNAL_BELOW_NOISE = 1  # alpha_min > alpha_max
    END_OF_SCALE = 2  # alpha_min = 0, alpha_max = 0
    IMMEDIATE_SEPARATION = 3  # alpha_min = 0, alpha_max > 0
    LIMINAL_SEPARATION = 4  # 0 < alpha_min <= alpha_max


def diagnose_alpha_regime(alpha_min: float, alpha_max: float):
    """
    Diagnoses the statistical regime based on alpha_min and alpha_max,
    and calculates related metrics like S and S_norm.

    Args:
        alpha_min (float): The minimum alpha value.
        alpha_max (float): The maximum alpha value.

    Returns:
        dict: A dictionary containing the regime, alpha values, S, and S_norm.
    """
    if alpha_min > alpha_max:
        regime = AlphaRegime.SIGNAL_BELOW_NOISE
        try:
            S = math.log(alpha_max / alpha_min)
        except ValueError:
            S = math.nan
        S_norm = math.nan

    elif alpha_min == 0 and alpha_max == 0:
        regime = AlphaRegime.END_OF_SCALE
        S = math.nan
        S_norm = math.nan

    elif alpha_min == 0 and alpha_max > 0:
        regime = AlphaRegime.IMMEDIATE_SEPARATION
        S = math.inf
        S_norm = math.nan

    else:
        # REGULAR: 0 < alpha_min <= alpha_max
        regime = AlphaRegime.LIMINAL_SEPARATION
        S = math.log(alpha_max / alpha_min)
        S_norm = S / math.log(1.0 / alpha_min)

    return {
        "regime": int(regime),
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "S": S,
        "S_norm": S_norm,
    }


def describe_alpha_regime(metrics: dict) -> str:
    """
    Generates a human-readable text report describing the diagnosed alpha regime.

    Args:
        metrics (dict): A dictionary containing regime diagnosis metrics
                        (output of `diagnose_alpha_regime`).

    Returns:
        str: A formatted string report of the statistical regime.
    """
    regime = AlphaRegime(int(metrics["regime"]))
    alpha_min = float(metrics["alpha_min"])
    alpha_max = float(metrics["alpha_max"])
    S = float(metrics["S"])
    S_norm = float(metrics["S_norm"])

    def _fmt(x):
        if math.isnan(x):
            return "N/A"
        if math.isinf(x):
            return "+inf" if x > 0 else "-inf"
        return f"{x:.6g}"

    def _fp_rate(a):
        if not (a > 0) or math.isnan(a) or math.isinf(a):
            return "N/A"
        return f"≈ 1 in {1.0/a:.6g}"

    def _log10(a):
        if not (a > 0) or math.isnan(a) or math.isinf(a):
            return math.nan
        return math.log10(a)

    if regime == AlphaRegime.SIGNAL_BELOW_NOISE:
        condition = "α_min > α_max"
        name = "SIGNAL BELOW NOISE"
        interpretation = "There is no statistical evidence of dependence."
        mis_action = "Do not reduce dimensionality."
        S_meaning = "S is negative due to inversion."
        S_norm_meaning = "N/A"
    elif regime == AlphaRegime.END_OF_SCALE:
        condition = "α_min = 0 and α_max = 0"
        name = "END OF SCALE"
        interpretation = "Criterion collapsed."
        mis_action = "Do not reduce dimensionality."
        S_meaning = "S is undefined."
        S_norm_meaning = "N/A"
    elif regime == AlphaRegime.IMMEDIATE_SEPARATION:
        condition = "α_min = 0 and α_max > 0"
        name = "IMMEDIATE SEPARATION"
        interpretation = "Dependencies are robust."
        mis_action = "Reduction allowed."
        S_meaning = "S diverges."
        S_norm_meaning = "N/A"
    else:
        condition = "0 < α_min ≤ α_max"
        name = "LIMINAL SEPARATION"
        interpretation = "Valid interval found."
        mis_action = "Reduction allowed."
        S_meaning = "S measures separability on log scale."
        S_norm_meaning = "S_norm measures fraction of potential gap."

    log10_min = _log10(alpha_min)
    log10_max = _log10(alpha_max)

    report = (
        f"\nCondition: {condition}\n"
        f"Statistical regime: {name} (id={int(regime)})\n\n"
        f"Interpretation: {interpretation}\n"
        f"Action on MIS: {mis_action}\n"
        f"Parameters:\n"
        f"  α_min = {_fmt(alpha_min)}  ({_fp_rate(alpha_min)});  "
        f"log10(α_min) = {_fmt(log10_min)}\n"
        f"  α_max = {_fmt(alpha_max)}  ({_fp_rate(alpha_max)});  "
        f"log10(α_max) = {_fmt(log10_max)}\n"
        f"Metrics:\n"
        f"  S = {_fmt(S)}  -> {S_meaning}\n"
        f"  S_norm = {_fmt(S_norm)}  -> {S_norm_meaning}\n"
    )
    return report


def calculate_spectral_entropy(Y):
    """
    Calculates the normalized spectral entropy of the correlation matrix of Y.
    High entropy (~1.0) indicates complex, spherical, or random structure.
    Low entropy (~0.0) indicates high redundancy/dimensionality reduction potential.
    """
    if hasattr(Y, "values"):
        data = np.asarray(Y.values, dtype=float)
    else:
        data = np.asarray(Y, dtype=float)

    n, m = data.shape
    if m < 2:
        return 0.0

    # Correlation matrix
    corr = np.corrcoef(data, rowvar=False)
    # Eigenvalues (Hermitian/Symmetric)
    eigvals = np.linalg.eigvalsh(corr)

    # Normalize eigenvalues to probability distribution
    # Filter small negative/zeros due to precision
    eigvals = eigvals[eigvals > 1e-9]
    if len(eigvals) == 0:
        return 0.0

    p = eigvals / np.sum(eigvals)

    # Entropy
    se = -np.sum(p * np.log(p))

    # Normalize by log(M)
    # Note: Max entropy for M variables is log(M) when all eigenvalues = 1
    # However, number of non-zero eigenvalues could be < M if N < M.
    # Usually we norm by log(min(N, M)) or log(len(eigvals)).
    # Using log(len(eigvals)) is safer.
    denom = np.log(len(eigvals))
    if denom == 0:
        return 0.0

    return se / denom
