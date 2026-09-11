"""Ground-truth helpers for controlled diagnostic problems.

All Pareto truth in the R5 diagnostic architecture is defined on the sampled
clean objective matrix Z, never on the observed matrix Y.  Objectives are
minimized. Exact duplicate objective vectors are all retained as nondominated
whenever their common vector is nondominated, preserving sample-row identity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._pareto import get_nondominated_mask_minimize


def sampled_pareto_indices(Z: pd.DataFrame | np.ndarray) -> list[int]:
    """Return sample indices nondominated in the clean minimization matrix Z."""
    values = Z.to_numpy(dtype=float) if isinstance(Z, pd.DataFrame) else np.asarray(Z, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Z must be a non-empty two-dimensional objective matrix")
    mask = get_nondominated_mask_minimize(values)
    return [int(index) for index in np.flatnonzero(mask)]


def diagnostic_truth(problem, Z: pd.DataFrame) -> dict:
    """Build truth from the theoretical problem declaration and its clean Z."""
    scenario = problem.scenario
    families: list[list[str]] = []
    start = 1
    for size in scenario.family_sizes:
        families.append([f"f{i}" for i in range(start, start + size)])
        start += size

    return {
        "name": scenario.historical_name,
        "problem_id": problem.id,
        "latent_expected": scenario.latent_expected,
        "structural_expected": scenario.structural_expected,
        "blocks_expected": families,
        "pareto_expected": sampled_pareto_indices(Z),
        "tags": sorted(scenario.tags),
    }
