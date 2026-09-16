"""Experimental Y-only Pareto perturbation-response diagnostics.

This module is intentionally benchmark/research-facing.  It estimates how exact
Pareto membership responds to hypothetical perturbations of an observed matrix
``Y``.  It does not infer that observation noise exists, identify its law, or
estimate its magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from scipy.stats import spearmanr

from misda._pareto import get_nondominated_mask_minimize
from misda.benchmark import DEFAULT_SEED, FORMAT_VERSION, software_versions
from misda.benchmarks.problems import PROBLEM_BY_ID
from misda.benchmarks.validation import (
    NOISE_REPLICATE_SEEDS,
    NOISE_SIGMAS,
    NOISY_ROBUSTNESS_PROBLEM_IDS,
)


@dataclass(frozen=True)
class ParetoResponsePoint:
    """One point of a conditional Pareto perturbation-response curve."""

    sigma: float
    mean_jaccard: float
    mc_se: Optional[float]
    draws: int


def _validate_matrix(Y) -> np.ndarray:
    data = np.asarray(Y, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] == 0:
        raise ValueError("Y must be a two-dimensional matrix with at least two rows.")
    if not np.isfinite(data).all():
        raise ValueError("Y must contain only finite values.")
    return data


def _front_jaccard(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.sum(first & second))
    union = int(np.sum(first | second))
    return float(intersection / union) if union else 1.0


def _standard_perturbation(
    rng: np.random.Generator,
    shape: tuple[int, int],
    *,
    distribution: str,
    gaussian_rho: float,
) -> np.ndarray:
    n_rows, n_objectives = shape
    if distribution == "gaussian":
        rho = float(gaussian_rho)
        if rho < 0.0 or rho > 1.0:
            raise ValueError("gaussian_rho must lie in [0, 1].")
        if rho == 0.0:
            return rng.normal(size=shape)
        common = rng.normal(size=(n_rows, 1))
        if rho == 1.0:
            return np.repeat(common, n_objectives, axis=1)
        independent = rng.normal(size=shape)
        return np.sqrt(rho) * common + np.sqrt(1.0 - rho) * independent

    if gaussian_rho != 0.0:
        raise ValueError("gaussian_rho is only defined for Gaussian perturbations.")
    if distribution == "laplace":
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), size=shape)
    if distribution == "uniform":
        limit = np.sqrt(3.0)
        return rng.uniform(-limit, limit, size=shape)
    raise ValueError("distribution must be 'gaussian', 'laplace', or 'uniform'.")


def pareto_perturbation_response(
    Y,
    sigmas: Iterable[float],
    *,
    seed: int = DEFAULT_SEED,
    draws: Optional[int] = None,
    distribution: str = "gaussian",
    gaussian_rho: float = 0.0,
) -> tuple[ParetoResponsePoint, ...]:
    """Estimate a conditional exact-membership response curve from observed Y.

    Each objective is perturbed at ``sigma * sample_std(objective)``.  The
    default Monte Carlo budget is data-derived: one perturbation draw per row of
    ``Y``.  Common random numbers are used across sigma values within each draw.

    The returned curve is conditional on the explicitly requested perturbation
    law.  It is not an estimate of the unknown observation-noise process.
    """

    data = _validate_matrix(Y)
    sigma_values = tuple(float(value) for value in sigmas)
    if not sigma_values:
        raise ValueError("At least one sigma is required.")
    if any(value < 0.0 for value in sigma_values):
        raise ValueError("sigma values must be non-negative.")

    n_draws = data.shape[0] if draws is None else int(draws)
    if n_draws < 1:
        raise ValueError("draws must be at least 1.")

    base_front = get_nondominated_mask_minimize(data)
    scales = np.std(data, axis=0, ddof=1)
    rng = np.random.default_rng(int(seed))
    values = np.empty((n_draws, len(sigma_values)), dtype=float)

    for draw_index in range(n_draws):
        standard = _standard_perturbation(
            rng,
            data.shape,
            distribution=distribution,
            gaussian_rho=gaussian_rho,
        )
        for sigma_index, sigma in enumerate(sigma_values):
            if sigma == 0.0:
                values[draw_index, sigma_index] = 1.0
                continue
            perturbed = data + sigma * scales[np.newaxis, :] * standard
            perturbed_front = get_nondominated_mask_minimize(perturbed)
            values[draw_index, sigma_index] = _front_jaccard(
                base_front, perturbed_front
            )

    points = []
    for sigma_index, sigma in enumerate(sigma_values):
        sample = values[:, sigma_index]
        mc_se = (
            float(np.std(sample, ddof=1) / np.sqrt(n_draws))
            if n_draws > 1
            else None
        )
        points.append(
            ParetoResponsePoint(
                sigma=sigma,
                mean_jaccard=float(np.mean(sample)),
                mc_se=mc_se,
                draws=n_draws,
            )
        )
    return tuple(points)


def _curve_lookup(points: tuple[ParetoResponsePoint, ...]) -> dict[float, ParetoResponsePoint]:
    return {point.sigma: point for point in points}


def _mean(records: list[dict], key: str) -> Optional[float]:
    values = [float(record[key]) for record in records if record[key] is not None]
    return float(np.mean(values)) if values else None


def _spearman(first: list[float], second: list[float]) -> Optional[float]:
    if len(first) < 2:
        return None
    result = float(spearmanr(first, second).statistic)
    return result if np.isfinite(result) else None


def run_pareto_response_validation(
    *,
    n: int = 300,
    sigmas: Iterable[float] = NOISE_SIGMAS,
    replicate_seeds: Iterable[int] = NOISE_REPLICATE_SEEDS,
    problem_ids: Iterable[str] = NOISY_ROBUSTNESS_PROBLEM_IDS,
    draws: Optional[int] = None,
) -> dict:
    """Compare the Gaussian Y-only curve with controlled perturbation families."""

    sigma_values = tuple(float(value) for value in sigmas)
    replicate_values = tuple(int(value) for value in replicate_seeds)
    problem_values = tuple(problem_ids)
    if not sigma_values or not replicate_values or not problem_values:
        raise ValueError("sigmas, replicate_seeds, and problem_ids must be non-empty.")
    unknown = set(problem_values) - set(PROBLEM_BY_ID)
    if unknown:
        raise ValueError(f"Unknown problem id(s): {', '.join(sorted(unknown))}")

    model_specs = (
        ("gaussian_iid", "gaussian", 0.0),
        ("laplace_iid", "laplace", 0.0),
        ("uniform_iid", "uniform", 0.0),
        ("gaussian_rho_0_5", "gaussian", 0.5),
        ("gaussian_rho_1_0", "gaussian", 1.0),
    )
    records: list[dict] = []

    for problem_position, problem_id in enumerate(problem_values):
        problem = PROBLEM_BY_ID[problem_id]
        for replicate_seed in replicate_values:
            sample_seed = int(
                np.random.SeedSequence([replicate_seed, problem_position, 1])
                .generate_state(1)[0]
            )
            X = problem.sample(N=n, seed=sample_seed)
            Z_frame = problem.evaluate(X)
            Z = Z_frame.to_numpy(dtype=float)
            base_front = get_nondominated_mask_minimize(Z)
            scales = np.std(Z, axis=0, ddof=1)
            observation_noise = np.random.default_rng(
                np.random.SeedSequence([replicate_seed, problem_position, 2])
            ).normal(size=Z.shape)

            curves = {}
            for model_index, (name, distribution, rho) in enumerate(model_specs):
                model_seed = int(
                    np.random.SeedSequence(
                        [replicate_seed, problem_position, 100 + model_index]
                    ).generate_state(1)[0]
                )
                curves[name] = _curve_lookup(
                    pareto_perturbation_response(
                        Z,
                        sigma_values,
                        seed=model_seed,
                        draws=draws,
                        distribution=distribution,
                        gaussian_rho=rho,
                    )
                )

            for sigma in sigma_values:
                if sigma == 0.0:
                    observed_jaccard = 1.0
                else:
                    observed = Z + sigma * scales[np.newaxis, :] * observation_noise
                    observed_jaccard = _front_jaccard(
                        base_front,
                        get_nondominated_mask_minimize(observed),
                    )
                record = {
                    "problem_id": problem_id,
                    "replicate_seed": replicate_seed,
                    "sample_seed": sample_seed,
                    "sigma": sigma,
                    "observed_gaussian_one_draw": observed_jaccard,
                }
                for name, _, _ in model_specs:
                    point = curves[name][sigma]
                    record[name] = point.mean_jaccard
                    record[f"{name}_mc_se"] = point.mc_se
                records.append(record)

    problem_summary = []
    for problem_id in problem_values:
        for sigma in sigma_values:
            group = [
                record
                for record in records
                if record["problem_id"] == problem_id and record["sigma"] == sigma
            ]
            row = {
                "problem_id": problem_id,
                "sigma": sigma,
                "replicates": len(group),
                "observed_gaussian_one_draw": _mean(
                    group, "observed_gaussian_one_draw"
                ),
            }
            for name, _, _ in model_specs:
                row[name] = _mean(group, name)
            problem_summary.append(row)

    comparison_summary = []
    comparison_names = tuple(name for name, _, _ in model_specs if name != "gaussian_iid")
    for sigma in sigma_values:
        group = [row for row in problem_summary if row["sigma"] == sigma]
        gaussian = [float(row["gaussian_iid"]) for row in group]
        observed = [float(row["observed_gaussian_one_draw"]) for row in group]
        comparison_summary.append(
            {
                "sigma": sigma,
                "target": "observed_gaussian_one_draw",
                "spearman": _spearman(gaussian, observed),
                "mae": float(np.mean(np.abs(np.asarray(gaussian) - np.asarray(observed)))),
            }
        )
        for name in comparison_names:
            target = [float(row[name]) for row in group]
            comparison_summary.append(
                {
                    "sigma": sigma,
                    "target": name,
                    "spearman": _spearman(gaussian, target),
                    "mae": float(
                        np.mean(np.abs(np.asarray(gaussian) - np.asarray(target)))
                    ),
                }
            )

    effective_draws = n if draws is None else int(draws)
    return {
        "format_version": FORMAT_VERSION,
        "suite": "pareto_perturbation_response",
        "method": "conditional Y-only Pareto perturbation response",
        "parameters": {
            "n": int(n),
            "sigmas": list(sigma_values),
            "replicate_seeds": list(replicate_values),
            "problem_ids": list(problem_values),
            "draws": effective_draws,
            "scale": "sample_std_per_objective",
        },
        "software": software_versions(),
        "records": records,
        "problem_summary": problem_summary,
        "comparison_summary": comparison_summary,
    }
