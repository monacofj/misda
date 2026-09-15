"""Ground-truth helpers for controlled diagnostic problems.

All Pareto truth in the R5 diagnostic architecture is defined on the sampled
clean objective matrix Z, never on the observed matrix Y. Objectives are
minimized. Exact duplicate objective vectors are all retained as nondominated
whenever their common vector is nondominated, preserving sample-row identity.

Generating families and structural units are separate declarations. Families
describe how objectives are generated; ``blocks_expected`` keeps its public
benchmark meaning as structural units and is declared only when the problem
specification supplies an unambiguous structural partition.

Human-readable feature, intuition, graph expectation, and notes are benchmark
metadata. They intentionally live in this benchmark module rather than in the
scientific ``DiagnosticScenario`` specification.
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


# Presentation/declaration prose belongs exclusively to the benchmark layer.
# Keep every canonical problem explicit here so missing metadata fails loudly
# instead of silently degrading reports to N/A.
_BENCHMARK_DESCRIPTIONS = {
    "independence": {
        "feature": "All 20 objectives are mutually independent i.i.d. Gaussian noise variables.",
        "intuition": "20 completely unrelated random sensors; knowing one tells you nothing about any other. MISDA should keep all 20.",
        "graph_expected": "20 isolated nodes (0 edges, 20 connected components)",
        "notes": "",
    },
    "total_redundancy": {
        "feature": "All 20 objectives are noisy linear copies of a single 1D latent factor.",
        "intuition": "20 identical thermometers measuring the exact same room temperature with minor noise. MISDA should keep just 1.",
        "graph_expected": "1 fully connected graph (K_20, 190 edges, 1 connected component)",
        "notes": "",
    },
    "blocks_4x5": {
        "feature": "4 independent latent factors; each factor generates a cluster of 5 redundant objectives.",
        "intuition": "4 physical properties (e.g., Temp, Pressure, Humidity, Speed), each measured by 5 duplicate sensors. MISDA should reduce 20 sensors to 4.",
        "graph_expected": "4 disjoint complete subgraphs of 5 nodes each (4 x K_5, 40 total edges)",
        "notes": "",
    },
    "blocks_2x10": {
        "feature": "2 independent latent factors; each factor generates a cluster of 10 redundant objectives.",
        "intuition": "Measuring 2 goals (e.g., Cost and Weight), but using 10 duplicate formulas for Cost and 10 for Weight. MISDA should reduce 20 formulas to 2.",
        "graph_expected": "2 disjoint complete subgraphs of 10 nodes each (2 x K_10, 90 total edges)",
        "notes": "",
    },
    "mixed_independent_and_blocks": {
        "feature": "Heterogeneous structure: 10 independent noise objectives (f1..f10) and 2 redundant blocks of 5.",
        "intuition": "10 random independent variables mixed with 2 redundant groups of 5 sensors each. MISDA should keep 10 + 2 = 12 objectives.",
        "graph_expected": "10 isolated nodes and 2 disjoint complete subgraphs of 5 nodes each (10 x K_1 + 2 x K_5)",
        "notes": "",
    },
    "monotonic_redundancy": {
        "feature": "20 non-linear monotonic transformations driven by a single 1D decision variable x.",
        "intuition": "20 different formulas (squares, roots, logs) calculated from a single input x. Since all move in sync, MISDA should collapse all 20 to 1.",
        "graph_expected": "G+ and G± are K_20 (20 nodes, 190 edges, 1 connected component).",
        "notes": "20 objectives as monotonic (and redundant) transformations of the same latent x.",
    },
    "antagonistic_linear_groups": {
        "feature": "Two groups (+x and -x) with internal redundancy and strong structural conflict (anti-correlation).",
        "intuition": "10 sensors measuring Car Speed (+x) and 10 measuring Remaining Travel Time (-x). Speed and Time conflict, but both are essential! MISDA must keep 1 of each.",
        "graph_expected": "G+ is 2 disjoint K_10 components (20 nodes, 90 edges); G± is K_20 (20 nodes, 190 edges, 1 connected component) because the two groups are mutually anti-correlated.",
        "notes": "",
    },
    "tradeoff_redundancies": {
        "feature": "Three functional engineering families (7 cost, 7 consumption, 6 performance) driven by 2 decision variables.",
        "intuition": "An engineering problem with 3 main goals: Cost, Energy, and Performance, each measured in multiple redundant ways. MISDA should shrink 20 to ~2-3 core trade-offs.",
        "graph_expected": "G+ is one connected 20-node graph; the 3 generating families are declared separately from graph topology.",
        "notes": "Three families (cost/consumption/performance) with internal redundancies; effective tends to ~2.",
    },
    "nonlinear_blocks_4x5": {
        "feature": "4 independent decision factors; each factor generates a block of 5 non-linearly transformed objectives.",
        "intuition": "4 control dials, where turning each dial affects 5 non-linear indicators. MISDA should extract 4 independent representatives (1 per dial).",
        "graph_expected": "G+ and G± are 4 disjoint K_5 components (20 nodes, 40 edges, 4 connected components).",
        "notes": "Four independent factors; each block (5 objectives) is internally redundant.",
    },
    "antagonistic_nonlinear_groups": {
        "feature": "Two antagonistic non-linear objective families (+x vs 1-x) with internal redundancy and trade-off conflict.",
        "intuition": "10 indicators measuring Benefit (+x) vs 10 measuring Risk (1-x). Benefit and Risk directly conflict. MISDA must preserve 1 Benefit and 1 Risk indicator.",
        "graph_expected": "G+ is 2 disjoint K_10 components (20 nodes, 90 edges); G± is K_20 because the groups are mutually antagonistic.",
        "notes": "Two internally redundant groups (+x and 1-x), but antagonistic to each other: conflict must be preserved.",
    },
    "overlapping_factors": {
        "feature": "Partial redundancy across 2 latent drivers (a,b): 10 objectives on a, 4 on b, and 6 compounds on s=a+b.",
        "intuition": "Overlapping signals: some indicators monitor Engine A, some monitor Engine B, and some monitor both combined (A+B). Tests if MISDA untangles blended signals.",
        "graph_expected": "G+ is one connected 20-node graph; the A, B, and A+B generating families are declared separately from graph topology.",
        "notes": "Trio/quartet of 'a' extended to 10 redundants; 'b' (4); and 6 compounds around s=a+b.",
    },
    "transitive_chain": {
        "feature": "Cumulative random-walk chain with one independent innovation added at each objective; pairwise correlations decay with index distance but remain strongly positive at this noise scale.",
        "intuition": "Each objective adds new information to the previous one, so the generating dimension remains 20 even though accumulated pairwise correlation can make all objectives pass the positive-dependence threshold. This is the intended transitive-chaining failure mode.",
        "graph_expected": "Thresholded positive/dependence graph saturates to K_20 (190 edges, 1 connected component) even though the generating mechanism is a chain.",
        "notes": "The complete threshold graph is not ground-truth redundancy: TRANSITIVE_CHAINING is expected to flag the collapse from the 20-dimensional generating process to graph-derived dimension 1.",
    },
    "regime_switching": {
        "feature": "Non-linear regime-switching mixture: 10 objectives on regime-dependent mixture L(a,b) and 10 on b.",
        "intuition": "System switching: indicators change behavior depending on whether the system operates in High-Power or Low-Power mode. Tests MISDA under shifting states.",
        "graph_expected": "At the canonical threshold, G+ and G± saturate to K_20 (20 nodes, 190 edges, 1 connected component), hiding the 2-driver generating structure.",
        "notes": "10 objectives redundant around L (mixture by regime) + 10 redundant around b; global correlation can be misleading.",
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
    """Build benchmark truth and declaration metadata from clean Z."""
    scenario = problem.scenario
    try:
        description = _BENCHMARK_DESCRIPTIONS[problem.id]
    except KeyError as exc:
        raise KeyError(
            f"Missing benchmark declaration metadata for problem {problem.id!r}."
        ) from exc

    truth = {
        "name": scenario.name,
        "problem_id": problem.id,
        "latent_expected": scenario.latent_expected,
        "structural_expected": scenario.structural_expected,
        "families_expected": _label_groups(scenario.family_sizes),
        "pareto_expected": sampled_pareto_indices(Z),
        "tags": sorted(scenario.tags),
        "expected_mismatches": dict(_EXPECTED_MISMATCHES.get(problem.id, {})),
        "feature": description["feature"],
        "intuition": description["intuition"],
        "graph_expected": description["graph_expected"],
        "notes": description["notes"],
    }
    if scenario.structural_unit_sizes is not None:
        truth["blocks_expected"] = _label_groups(scenario.structural_unit_sizes)
    return truth
