# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Statistical threshold calculations used by MISDA."""

import copy
import math
import warnings
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from numbers import Integral
from typing import Any, Callable, Iterable, Optional, Tuple

import numpy as np
from scipy import stats

from ._validation import NormalizedInput, validate_aggressiveness


class SeparationStatus(str, Enum):
    """Whether the observed positive onset is separated from the null bound."""

    NULL_SEPARATION = "NULL_SEPARATION"
    NO_NULL_SEPARATION = "NO_NULL_SEPARATION"


@dataclass(frozen=True)
class CorrelationStatistics:
    """Signed correlations and one-tailed log probabilities for valid pairs."""

    correlation: np.ndarray
    log_p: np.ndarray
    valid_pairs: np.ndarray
    labels: Tuple[Any, ...]
    constant_indices: Tuple[int, ...]
    n_samples: int
    log_alpha_onset: Optional[float]

    @property
    def alpha_onset(self) -> Optional[float]:
        if self.log_alpha_onset is None:
            return None
        return float(np.exp(self.log_alpha_onset))


@dataclass(frozen=True)
class NullAlphaEstimate:
    """Sequential estimate of the expected maximum positive null correlation."""

    r_null: float
    se_mc: float
    r_interval: Tuple[float, float]
    log_alpha_null: float
    log_alpha_interval: Tuple[float, float]
    n_permutations: int
    converged: bool
    lower_r_signature: Any
    upper_r_signature: Any
    samples: Tuple[float, ...]
    seed: Optional[int] = None
    rng_state: Optional[dict] = None
    reason: Optional[str] = None

    @property
    def alpha_null(self) -> float:
        return float(np.exp(self.log_alpha_null))


def positive_correlation_log_p(r, n_samples):
    """Return the one-tailed Fisher-z log probability for positive ``r``.

    The result remains in the logarithmic domain.  Exact perfect correlation
    therefore maps to ``-inf`` rather than to zero or a machine-dependent floor.
    """

    if (
        isinstance(n_samples, (bool, np.bool_))
        or not isinstance(n_samples, Integral)
        or int(n_samples) < 4
    ):
        raise ValueError("n_samples must be an integer greater than or equal to 4.")

    values = np.asarray(r, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("r must contain finite correlations in [0, 1].")

    with np.errstate(divide="ignore", invalid="ignore"):
        z_stat = np.arctanh(values) * np.sqrt(int(n_samples) - 3)
        result = stats.norm.logsf(z_stat)
    if values.ndim == 0:
        return float(result)
    return np.asarray(result, dtype=float)


def compute_correlation_statistics(normalized: NormalizedInput) -> CorrelationStatistics:
    """Compute signed correlations while marking constant pairs as undefined."""

    if not isinstance(normalized, NormalizedInput):
        raise TypeError("normalized must be a NormalizedInput instance.")

    data = normalized.data
    n_samples, n_objectives = data.shape
    centered = data - np.mean(data, axis=0)
    sum_squares = np.sum(centered * centered, axis=0)
    denominator = np.sqrt(np.outer(sum_squares, sum_squares))
    numerator = centered.T @ centered

    correlation = np.full((n_objectives, n_objectives), np.nan, dtype=float)
    np.divide(
        numerator,
        denominator,
        out=correlation,
        where=denominator > 0.0,
    )
    finite_correlation = np.isfinite(correlation)
    correlation[finite_correlation] = np.clip(
        correlation[finite_correlation], -1.0, 1.0
    )

    nonconstant = ~normalized.constant_mask
    diagonal = np.arange(n_objectives)
    correlation[diagonal[nonconstant], diagonal[nonconstant]] = 1.0

    valid_pairs = np.outer(nonconstant, nonconstant)
    np.fill_diagonal(valid_pairs, False)

    log_p = np.full_like(correlation, np.nan)
    if np.any(valid_pairs):
        log_p[valid_pairs] = positive_correlation_log_p(
            np.abs(correlation[valid_pairs]),
            n_samples,
        )

    upper = np.triu(valid_pairs & (correlation > 0.0), k=1)
    if np.any(upper):
        log_alpha_onset = float(np.min(log_p[upper]))
    else:
        log_alpha_onset = None

    return CorrelationStatistics(
        correlation=correlation,
        log_p=log_p,
        valid_pairs=valid_pairs,
        labels=normalized.labels,
        constant_indices=normalized.constant_indices,
        n_samples=n_samples,
        log_alpha_onset=log_alpha_onset,
    )


def correlation_edge_masks(
    correlation_statistics: CorrelationStatistics,
    log_alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return provisional adjacency masks for positive and signed dependence."""

    if not isinstance(correlation_statistics, CorrelationStatistics):
        raise TypeError(
            "correlation_statistics must be a CorrelationStatistics instance."
        )
    if not isinstance(log_alpha, (int, float, np.floating)) or np.isnan(log_alpha):
        raise ValueError("log_alpha must be a real number other than NaN.")

    significant = (
        correlation_statistics.valid_pairs
        & (correlation_statistics.log_p <= float(log_alpha))
    )
    positive = significant & (correlation_statistics.correlation > 0.0)
    signed = significant.copy()
    np.fill_diagonal(positive, False)
    np.fill_diagonal(signed, False)
    return positive, signed


def separation_status(
    log_alpha_onset: Optional[float],
    log_alpha_null: float,
) -> SeparationStatus:
    """Classify strict separation between observed onset and null reference."""

    if np.isnan(log_alpha_null):
        raise ValueError("log_alpha_null must not be NaN.")
    if log_alpha_onset is None:
        return SeparationStatus.NO_NULL_SEPARATION
    if np.isnan(log_alpha_onset):
        raise ValueError("log_alpha_onset must not be NaN.")
    if log_alpha_onset < log_alpha_null:
        return SeparationStatus.NULL_SEPARATION
    return SeparationStatus.NO_NULL_SEPARATION


def interpolate_log_alpha(
    log_alpha_onset: Optional[float],
    log_alpha_null: float,
    aggressiveness,
) -> float:
    """Interpolate alpha arithmetically while remaining in the log domain."""

    normalized_aggressiveness = validate_aggressiveness(aggressiveness)
    if log_alpha_onset is None:
        raise ValueError("Cannot interpolate alpha without a positive onset.")
    if np.isnan(log_alpha_onset) or np.isnan(log_alpha_null):
        raise ValueError("log-alpha bounds must not be NaN.")
    if normalized_aggressiveness == 0.0:
        return float(log_alpha_onset)
    if normalized_aggressiveness == 1.0:
        return float(log_alpha_null)
    return float(
        np.logaddexp(
            np.log1p(-normalized_aggressiveness) + log_alpha_onset,
            np.log(normalized_aggressiveness) + log_alpha_null,
        )
    )


def _null_estimate_snapshot(
    samples,
    n_samples,
    signature: Callable[[float], Any],
    converged,
    reason=None,
) -> NullAlphaEstimate:
    values = np.asarray(samples, dtype=float)
    count = len(values)
    mean = float(np.mean(values)) if count else math.nan
    if count >= 2:
        se_mc = float(np.std(values, ddof=1) / np.sqrt(count))
    else:
        se_mc = math.inf

    if count:
        lower_r = max(0.0, mean - se_mc)
        upper_r = min(1.0, mean + se_mc)
        log_alpha_null = positive_correlation_log_p(mean, n_samples)
        log_at_lower_r = positive_correlation_log_p(lower_r, n_samples)
        log_at_upper_r = positive_correlation_log_p(upper_r, n_samples)
        lower_r_signature = signature(log_at_lower_r)
        upper_r_signature = signature(log_at_upper_r)
        log_interval = (log_at_upper_r, log_at_lower_r)
    else:
        lower_r = math.nan
        upper_r = math.nan
        log_alpha_null = math.nan
        log_interval = (math.nan, math.nan)
        lower_r_signature = None
        upper_r_signature = None

    return NullAlphaEstimate(
        r_null=mean,
        se_mc=se_mc,
        r_interval=(lower_r, upper_r),
        log_alpha_null=log_alpha_null,
        log_alpha_interval=log_interval,
        n_permutations=count,
        converged=converged,
        lower_r_signature=lower_r_signature,
        upper_r_signature=upper_r_signature,
        samples=tuple(float(value) for value in values),
        reason=reason,
    )


def estimate_null_from_maxima(
    maxima: Iterable[float],
    *,
    n_samples: int,
    signature: Callable[[float], Any],
    cancel_requested: Optional[Callable[[int], bool]] = None,
    max_permutations: Optional[int] = None,
) -> NullAlphaEstimate:
    """Run the sequential stopping rule on controlled null maxima.

    ``max_permutations`` is optional for controlled/internal sequences.  Public
    structural null estimation sets it to ``10 * N`` so termination is
    autonomous even when the structural signatures do not stabilize.
    """

    if (
        isinstance(n_samples, (bool, np.bool_))
        or not isinstance(n_samples, Integral)
        or int(n_samples) < 4
    ):
        raise ValueError("n_samples must be an integer greater than or equal to 4.")
    if not callable(signature):
        raise TypeError("signature must be callable.")
    if cancel_requested is not None and not callable(cancel_requested):
        raise TypeError("cancel_requested must be callable or None.")
    if max_permutations is not None:
        if (
            isinstance(max_permutations, (bool, np.bool_))
            or not isinstance(max_permutations, Integral)
        ):
            raise TypeError("max_permutations must be an integer or None.")
        max_permutations = int(max_permutations)
        if max_permutations < int(n_samples):
            raise ValueError("max_permutations must be greater than or equal to n_samples.")

    samples = []
    for maximum in maxima:
        value = float(maximum)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("null maxima must be finite values in [0, 1].")
        samples.append(value)

        if len(samples) < int(n_samples):
            continue
        snapshot = _null_estimate_snapshot(
            samples,
            int(n_samples),
            signature,
            converged=False,
        )
        if snapshot.lower_r_signature == snapshot.upper_r_signature:
            return replace(snapshot, converged=True, reason=None)
        if cancel_requested is not None and cancel_requested(len(samples)):
            return replace(snapshot, reason="CANCELLED")
        if max_permutations is not None and len(samples) >= max_permutations:
            return replace(snapshot, reason="MAX_PERMUTATIONS_REACHED")

    return _null_estimate_snapshot(
        samples,
        int(n_samples),
        signature,
        converged=False,
        reason="SEQUENCE_EXHAUSTED",
    )


def _maximum_positive_correlation(data: np.ndarray, constant_indices) -> float:
    n_objectives = data.shape[1]
    if n_objectives < 2:
        return 0.0
    centered = data - np.mean(data, axis=0)
    sum_squares = np.sum(centered * centered, axis=0)
    denominator = np.sqrt(np.outer(sum_squares, sum_squares))
    correlation = np.zeros((n_objectives, n_objectives), dtype=float)
    np.divide(
        centered.T @ centered,
        denominator,
        out=correlation,
        where=denominator > 0.0,
    )
    np.clip(correlation, -1.0, 1.0, out=correlation)
    if constant_indices:
        correlation[list(constant_indices), :] = 0.0
        correlation[:, list(constant_indices)] = 0.0
    values = correlation[np.triu_indices(n_objectives, k=1)]
    return float(max(0.0, np.max(values, initial=0.0)))


def estimate_null_positive_correlation(
    normalized: NormalizedInput,
    *,
    signature: Callable[[float], Any],
    seed: int = 0,
    cancel_requested: Optional[Callable[[int], bool]] = None,
) -> NullAlphaEstimate:
    """Estimate the expected maximum positive correlation under permutation."""

    if not isinstance(normalized, NormalizedInput):
        raise TypeError("normalized must be a NormalizedInput instance.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer.")

    seed = int(seed)
    rng = np.random.default_rng(seed)

    data = normalized.data
    n_samples, n_objectives = data.shape

    if n_objectives < 2:
        def maxima():
            while True:
                yield 0.0
    else:
        centered = data - np.mean(data, axis=0)
        sum_squares = np.sum(centered * centered, axis=0)
        stds = np.sqrt(sum_squares)
        valid = stds > 0.0
        Z = np.zeros_like(data)
        Z[:, valid] = centered[:, valid] / stds[valid]
        triu_idx = np.triu_indices(n_objectives, k=1)
        constant_list = list(normalized.constant_indices)

        def maxima():
            permuted = np.empty_like(Z)
            while True:
                for col in range(n_objectives):
                    permuted[:, col] = rng.permutation(Z[:, col])
                corr = permuted.T @ permuted
                np.clip(corr, -1.0, 1.0, out=corr)
                if constant_list:
                    corr[constant_list, :] = 0.0
                    corr[:, constant_list] = 0.0
                values = corr[triu_idx]
                yield float(max(0.0, np.max(values, initial=0.0)))

    result = estimate_null_from_maxima(
        maxima(),
        n_samples=normalized.n_samples,
        signature=signature,
        cancel_requested=cancel_requested,
        max_permutations=10 * normalized.n_samples,
    )
    if not result.converged and result.reason == "MAX_PERMUTATIONS_REACHED":
        warnings.warn(
            "Structural alpha_null estimation did not converge by B_max=10N; "
            "returning the current null estimate with converged=False.",
            RuntimeWarning,
            stacklevel=2,
        )
    return replace(
        result,
        seed=seed,
        rng_state=copy.deepcopy(rng.bit_generator.state),
    )


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
        str: A formatted string report describing the statistical regime.
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

    corr = np.corrcoef(data, rowvar=False)
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = eigvals[eigvals > 1e-9]
    if len(eigvals) == 0:
        return 0.0

    p = eigvals / np.sum(eigvals)
    se = -np.sum(p * np.log(p))
    denom = np.log(len(eigvals))
    if denom == 0:
        return 0.0

    return se / denom
