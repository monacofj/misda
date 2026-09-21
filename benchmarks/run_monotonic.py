"""Discriminating benchmark for monotonic dependence in MISDA discovery.

The cases are deliberately small and controlled.  They separate linear
association from monotonic association while retaining positive, negative,
non-monotonic, and independent controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

import misda


DEFAULT_N = 300
DEFAULT_SEED = 123
EXPONENT_STEEPNESS = 650.0


def _edges(graph):
    return [list(edge) for edge in sorted(tuple(sorted(edge)) for edge in graph.edges())]


def _focus_correlations(Y, pair):
    i, j = pair
    x = np.asarray(Y[:, i], dtype=float)
    y = np.asarray(Y[:, j], dtype=float)
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(stats.spearmanr(x, y).statistic)
    return {"pearson": pearson, "spearman": spearman}


def build_cases(n=DEFAULT_N, seed=DEFAULT_SEED):
    """Return deterministic cases and their semantic dimensional truth."""

    n = int(n)
    seed = int(seed)
    if n < 4:
        raise ValueError("n must be at least 4")

    x01 = np.linspace(0.0, 1.0, n)
    x11 = np.linspace(-1.0, 1.0, n)
    # This is exp(k*x) multiplied by exp(-k).  Pearson and Spearman are
    # invariant to that positive scaling, while the bounded range (0, 1]
    # avoids overflow.  k=650 keeps all values representable for N=300 while
    # making Pearson weak and Spearman exactly monotonic.
    steep = np.exp(EXPONENT_STEEPNESS * (x01 - 1.0))

    rng = np.random.default_rng(seed)
    independent = rng.normal(size=n)
    independent_pair = rng.normal(size=(n, 2))

    return (
        {
            "id": "linear_positive_control",
            "description": "Exact positive linear redundancy: y = 2x.",
            "Y": np.column_stack((x01, 2.0 * x01)),
            "focus_pair": (0, 1),
            "latent_expected": 1,
            "structural_expected": 1,
        },
        {
            "id": "mild_monotonic_control",
            "description": "Smooth monotonic nonlinear redundancy: y = x^3 on [0,1].",
            "Y": np.column_stack((x01, x01**3)),
            "focus_pair": (0, 1),
            "latent_expected": 1,
            "structural_expected": 1,
        },
        {
            "id": "steep_positive_monotonic",
            "description": (
                "Strictly increasing but strongly nonlinear redundancy: "
                "y = exp(650(x-1))."
            ),
            "Y": np.column_stack((x01, steep)),
            "focus_pair": (0, 1),
            "latent_expected": 1,
            "structural_expected": 1,
        },
        {
            "id": "steep_negative_monotonic",
            "description": (
                "Strictly decreasing strong nonlinear dependence: "
                "y = -exp(650(x-1)). It is latent dependence but not positive redundancy."
            ),
            "Y": np.column_stack((x01, -steep)),
            "focus_pair": (0, 1),
            "latent_expected": 1,
            "structural_expected": 2,
        },
        {
            "id": "steep_positive_plus_independent",
            "description": (
                "One steep positive monotonic redundant pair plus one independent objective."
            ),
            "Y": np.column_stack((x01, steep, independent)),
            "focus_pair": (0, 1),
            "latent_expected": 2,
            "structural_expected": 2,
        },
        {
            "id": "nonmonotonic_quadratic_limit",
            "description": (
                "Deterministic but globally non-monotonic latent dependence: y = x^2 "
                "on [-1,1]. Latent truth is one dimension, but there is no global "
                "positive monotonic redundancy, so structural truth is two."
            ),
            "Y": np.column_stack((x11, x11**2)),
            "focus_pair": (0, 1),
            "latent_expected": 1,
            "structural_expected": 2,
        },
        {
            "id": "independence_control",
            "description": "Independent Gaussian objectives.",
            "Y": independent_pair,
            "focus_pair": (0, 1),
            "latent_expected": 2,
            "structural_expected": 2,
        },
    )


def run_monotonic(n=DEFAULT_N, seed=DEFAULT_SEED):
    """Run all monotonic discrimination cases with the installed MISDA build."""

    records = []
    for case in build_cases(n=n, seed=seed):
        Y = case["Y"]
        result = misda.discover(Y, name=case["id"], seed=seed)
        analysis = result.analysis
        ranking = misda.rank(result)

        latent_observed = int(analysis.latent_dimension)
        structural_observed = int(analysis.structural_dimension)
        latent_expected = int(case["latent_expected"])
        structural_expected = int(case["structural_expected"])

        records.append(
            {
                "id": case["id"],
                "description": case["description"],
                "n_objectives": int(Y.shape[1]),
                "focus_pair": list(case["focus_pair"]),
                "focus_correlation": _focus_correlations(Y, case["focus_pair"]),
                "latent_expected": latent_expected,
                "latent_observed": latent_observed,
                "latent_exact": latent_observed == latent_expected,
                "structural_expected": structural_expected,
                "structural_observed": structural_observed,
                "structural_exact": structural_observed == structural_expected,
                "exact": (
                    latent_observed == latent_expected
                    and structural_observed == structural_expected
                ),
                "g_positive_edges": _edges(analysis.structural_graph),
                "g_signed_edges": _edges(analysis.dependence_graph),
                "mis_count": len(result),
                "selected_objectives": list(ranking.mis().objectives),
                "selected_dimension": int(ranking.selected_dimension),
                "separation_status": str(analysis.separation_status.value),
                "alpha_onset": analysis.alpha_onset,
                "alpha_null": analysis.alpha_null,
                "alpha": analysis.alpha,
                "support_status": result.support.status,
            }
        )

    return {
        "suite": "monotonic",
        "purpose": (
            "Discriminate linear Pearson dependence from rank-monotonic Spearman "
            "dependence under the same MISDA discovery pipeline."
        ),
        "parameters": {
            "n": int(n),
            "seed": int(seed),
            "exponent_steepness": EXPONENT_STEEPNESS,
        },
        "summary": {
            "cases": len(records),
            "exact_cases": sum(record["exact"] for record in records),
        },
        "cases": records,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = _parse_args()
    artifact = run_monotonic(n=args.n, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Monotonic benchmark: {artifact['summary']['exact_cases']}/{artifact['summary']['cases']} exact")
    for case in artifact["cases"]:
        rho = case["focus_correlation"]
        print(
            f"{case['id']}: "
            f"Pearson={rho['pearson']:.4f}, Spearman={rho['spearman']:.4f}; "
            f"dl={case['latent_observed']}/{case['latent_expected']}, "
            f"ds={case['structural_observed']}/{case['structural_expected']}; "
            f"exact={'yes' if case['exact'] else 'no'}"
        )


if __name__ == "__main__":
    main()
