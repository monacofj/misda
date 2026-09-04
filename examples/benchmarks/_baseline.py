"""Historical baseline helpers used only to read/compare pre-new-API artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np

from misda._pareto import get_nondominated_mask


FORMAT_VERSION = 1
METHOD = "static"
DEFAULT_SEED = 123


def matrix_sha256(frame) -> str:
    matrix = np.ascontiguousarray(frame.to_numpy(dtype=np.float64))
    return hashlib.sha256(matrix.tobytes()).hexdigest()


def software_versions() -> dict[str, str]:
    packages = ("misda", "numpy", "pandas", "scipy", "scikit-learn")
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in packages},
    }


def write_json(artifact: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(
            artifact,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")


def _linear_press_reconstruction(frame, selected: list[int]) -> dict:
    """Evaluate eliminated objectives with deterministic PRESS/leave-one-out R²."""
    data = frame.to_numpy(dtype=float)
    n, m = data.shape
    eliminated = [index for index in range(m) if index not in selected]
    if not eliminated:
        return {
            "r2_by_objective": {},
            "mean_r2": None,
            "worst_r2": None,
            "status": "NO_REDUCTION",
        }

    predictors = data[:, selected]
    design = np.column_stack((np.ones(n), predictors))
    design_pinv = np.linalg.pinv(design)
    targets = data[:, eliminated]
    fitted = design @ (design_pinv @ targets)
    residuals = targets - fitted
    leverage = np.einsum("ij,ji->i", design, design_pinv)
    press_denominator = 1.0 - leverage
    if np.any(np.abs(press_denominator) <= np.finfo(float).eps * 10):
        raise ValueError("PRESS is undefined because one or more leverage values equal 1")

    loo_predictions = targets - residuals / press_denominator[:, np.newaxis]
    r2_by_objective = {}
    defined_values = []
    for position, objective in enumerate(eliminated):
        observed = targets[:, position]
        ss_total = float(np.sum((observed - np.mean(observed)) ** 2))
        if ss_total <= np.finfo(float).eps:
            value = None
        else:
            ss_residual = float(
                np.sum((observed - loo_predictions[:, position]) ** 2)
            )
            value = float(1.0 - ss_residual / ss_total)
            defined_values.append(value)
        r2_by_objective[str(frame.columns[objective])] = value

    return {
        "r2_by_objective": r2_by_objective,
        "mean_r2": (
            float(np.mean(defined_values)) if defined_values else None
        ),
        "worst_r2": min(defined_values) if defined_values else None,
        "status": "SUCCESS" if defined_values else "UNDEFINED_TARGETS",
    }


def _pareto_preservation(frame, selected: list[int]) -> dict:
    data = frame.to_numpy(dtype=float)
    full_front = get_nondominated_mask(data)
    reduced_front = get_nondominated_mask(data[:, selected])
    intersection = full_front & reduced_front
    union = full_front | reduced_front
    n_full = int(np.sum(full_front))
    n_reduced = int(np.sum(reduced_front))
    n_intersection = int(np.sum(intersection))
    n_union = int(np.sum(union))
    return {
        "retention": float(n_intersection / n_full) if n_full else None,
        "validity": float(n_intersection / n_reduced) if n_reduced else None,
        "jaccard": float(n_intersection / n_union) if n_union else None,
        "full_front_size": n_full,
        "reduced_front_size": n_reduced,
        "intersection_size": n_intersection,
        "exact_preservation": bool(np.array_equal(full_front, reduced_front)),
    }
