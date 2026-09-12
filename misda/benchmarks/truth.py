"""Ground-truth helpers for controlled diagnostic problems.

All Pareto truth in the R5 diagnostic architecture is defined on the sampled
clean objective matrix Z, never on the observed matrix Y. Objectives are
minimized. Exact duplicate objective vectors are all retained as nondominated
whenever their common vector is nondominated, preserving sample-row identity.

Generating families and structural units are separate declarations. Families
describe how objectives are generated; structural units are declared only when
the problem specification supplies an unambiguous structural partition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._pareto import get_nondominated_mask_minimize


_EXPECTED_MISMATCHES = {
    "transitive_chain": {
        "latent_dimension": "TRANSITIVE_CHAINING",
        "structural_dimension": "TRANSITIVE_CHAINING",
        "selected_structural_units": "TRANSITIVE_CHAINING",
    },
    "regime_switching": {
        "latent_dimension": "HIDDEN_SPECTRAL_STRUCTURE",
        "structural_dimension": "HIDDEN_SPECTRAL_STRUCTURE",
        "selected_structural_units": "HIDDEN_SPECTRAL_STRUCTURE",
    },
}


def _label_groups(sizes: tuple[int, ...]) -> list[list[str]]:
    groups: list[list[str]] = []
    start = 1
    for size in sizes:
        groups.append([f"f{i}" for i in range(start, start + size)])
        start += size
    return groups


def sampled_pareto_indices(Z: pd.DataFrame | np.ndarray) -> list[int]:
    """Return sample indices nondominated in the clean minimization matrix Z."""
    values = (
        Z.to_numpy(dtype=float)
        if isinstance(Z, pd.DataFrame)
        else np.asarray(Z, dtype=float)
    )
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Z must be a non-empty two-dimensional objective matrix")
    mask = get_nondominated_mask_minimize(values)
    return [int(index) for index in np.flatnonzero(mask)]


def diagnostic_truth(problem, Z: pd.DataFrame) -> dict:
    """Build truth from the theoretical problem declaration and its clean Z."""
    scenario = problem.scenario
    truth = {
        "name": scenario.historical_name,
        "problem_id": problem.id,
        "latent_expected": scenario.latent_expected,
        "structural_expected": scenario.structural_expected,
        "families_expected": _label_groups(scenario.family_sizes),
        "pareto_expected": sampled_pareto_indices(Z),
        "tags": sorted(scenario.tags),
        "expected_mismatches": dict(_EXPECTED_MISMATCHES.get(problem.id, {})),
    }
    if scenario.structural_unit_sizes is not None:
        truth["structural_units_expected"] = _label_groups(
            scenario.structural_unit_sizes
        )
    return truth
