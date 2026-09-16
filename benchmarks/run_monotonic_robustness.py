"""Robustness experiment for Pearson versus Spearman MISDA discovery.

The experiment reuses the controlled monotonic cases from ``run_monotonic``
and probes two distinct perturbations:

1. bootstrap resampling of rows from the clean data;
2. additive observation noise, scaled by each objective's sample standard
   deviation and coupled across sigma levels with common random numbers.

Discovery's own seed is held fixed so variation is attributable to the data
perturbation rather than to MISDA's internal randomization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import misda

try:
    from benchmarks.run_monotonic import DEFAULT_N, DEFAULT_SEED, build_cases
except ModuleNotFoundError:  # allows this script to drive an installed baseline checkout
    from run_monotonic import DEFAULT_N, DEFAULT_SEED, build_cases


RESAMPLE_SEEDS = tuple(range(2001, 2011))
NOISE_SEEDS = (3001, 3002, 3003, 3004, 3005)
NOISE_SIGMAS = (0.00, 0.01, 0.02, 0.05, 0.10)


def _mean(records, key):
    if not records:
        return None
    return float(sum(float(record[key]) for record in records) / len(records))


def _mode_dimension(records, key):
    values = [int(record[key]) for record in records]
    if not values:
        return None
    counts = {value: values.count(value) for value in sorted(set(values))}
    return min(counts, key=lambda value: (-counts[value], value))


def _discover_record(Y, case, *, discovery_seed, perturbation, perturbation_seed, sigma=None):
    result = misda.discover(Y, name=case["id"], seed=discovery_seed)
    analysis = result.analysis
    i, j = case["focus_pair"]
    latent_observed = int(analysis.latent_dimension)
    structural_observed = int(analysis.structural_dimension)
    latent_expected = int(case["latent_expected"])
    structural_expected = int(case["structural_expected"])

    return {
        "case_id": case["id"],
        "perturbation": perturbation,
        "perturbation_seed": int(perturbation_seed),
        "sigma": None if sigma is None else float(sigma),
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
        "focus_positive_edge": bool(analysis.structural_graph.has_edge(i, j)),
        "focus_signed_edge": bool(analysis.dependence_graph.has_edge(i, j)),
        "selected_dimension": int(result.structural_ranking.selected_dimension),
        "separation_status": str(analysis.separation_status.value),
        "support_status": result.support.status,
    }


def _summarize(records, case_ids, *, perturbation, sigmas=None):
    summary = []
    sigma_values = (None,) if sigmas is None else tuple(sigmas)
    for case_id in case_ids:
        for sigma in sigma_values:
            group = [
                record
                for record in records
                if record["case_id"] == case_id
                and record["perturbation"] == perturbation
                and (sigma is None or record["sigma"] == float(sigma))
            ]
            if not group:
                continue
            item = {
                "case_id": case_id,
                "perturbation": perturbation,
                "replicates": len(group),
                "latent_recovery": _mean(group, "latent_exact"),
                "structural_recovery": _mean(group, "structural_exact"),
                "exact_recovery": _mean(group, "exact"),
                "focus_positive_edge_rate": _mean(group, "focus_positive_edge"),
                "focus_signed_edge_rate": _mean(group, "focus_signed_edge"),
                "modal_latent_dimension": _mode_dimension(group, "latent_observed"),
                "modal_structural_dimension": _mode_dimension(group, "structural_observed"),
            }
            if sigma is not None:
                item["sigma"] = float(sigma)
            summary.append(item)
    return summary


def run_monotonic_robustness(
    *,
    n=DEFAULT_N,
    base_seed=DEFAULT_SEED,
    discovery_seed=DEFAULT_SEED,
    resample_seeds=RESAMPLE_SEEDS,
    noise_seeds=NOISE_SEEDS,
    noise_sigmas=NOISE_SIGMAS,
    label="unspecified",
):
    """Return paired bootstrap/noise robustness evidence for the monotonic suite."""

    n = int(n)
    base_seed = int(base_seed)
    discovery_seed = int(discovery_seed)
    resample_seeds = tuple(int(value) for value in resample_seeds)
    noise_seeds = tuple(int(value) for value in noise_seeds)
    noise_sigmas = tuple(float(value) for value in noise_sigmas)
    if not resample_seeds:
        raise ValueError("At least one resampling seed is required.")
    if not noise_seeds:
        raise ValueError("At least one noise seed is required.")
    if not noise_sigmas:
        raise ValueError("At least one noise sigma is required.")

    cases = build_cases(n=n, seed=base_seed)
    case_ids = [case["id"] for case in cases]
    records = []

    # Bootstrap rows from one common clean realization.  This preserves the
    # semantic relation while changing the empirical sample and its leverage.
    for case in cases:
        Y = np.asarray(case["Y"], dtype=float)
        for resample_seed in resample_seeds:
            rng = np.random.default_rng(resample_seed)
            indices = rng.integers(0, len(Y), size=len(Y))
            records.append(
                _discover_record(
                    Y[indices],
                    case,
                    discovery_seed=discovery_seed,
                    perturbation="bootstrap",
                    perturbation_seed=resample_seed,
                )
            )

    # Use common random numbers across sigma values within each replicate, so
    # deterioration with sigma is not confounded by a different noise draw.
    for case_position, case in enumerate(cases):
        Y = np.asarray(case["Y"], dtype=float)
        scale = np.std(Y, axis=0, ddof=1)
        scale = np.where(scale > 0.0, scale, 1.0)
        for noise_seed in noise_seeds:
            sequence = np.random.SeedSequence([noise_seed, case_position, base_seed])
            epsilon = np.random.default_rng(sequence).normal(size=Y.shape)
            for sigma in noise_sigmas:
                noisy = Y + float(sigma) * scale * epsilon
                records.append(
                    _discover_record(
                        noisy,
                        case,
                        discovery_seed=discovery_seed,
                        perturbation="noise",
                        perturbation_seed=noise_seed,
                        sigma=sigma,
                    )
                )

    bootstrap_summary = _summarize(
        records, case_ids, perturbation="bootstrap"
    )
    noise_summary = _summarize(
        records,
        case_ids,
        perturbation="noise",
        sigmas=noise_sigmas,
    )

    return {
        "suite": "monotonic_robustness",
        "label": str(label),
        "purpose": (
            "Assess whether monotonic-dependence recovery is stable under "
            "bootstrap resampling and scaled additive observation noise."
        ),
        "parameters": {
            "n": n,
            "base_seed": base_seed,
            "discovery_seed": discovery_seed,
            "resample_seeds": list(resample_seeds),
            "noise_seeds": list(noise_seeds),
            "noise_sigmas": list(noise_sigmas),
            "noise_scaling": "per-objective sample standard deviation",
            "common_random_numbers_across_sigma": True,
        },
        "records": records,
        "bootstrap_summary": bootstrap_summary,
        "noise_summary": noise_summary,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="unspecified")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--discovery-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    kwargs = {}
    if args.quick:
        kwargs.update(
            resample_seeds=RESAMPLE_SEEDS[:2],
            noise_seeds=NOISE_SEEDS[:2],
            noise_sigmas=(0.0, 0.05),
        )
    artifact = run_monotonic_robustness(
        n=args.n,
        base_seed=args.base_seed,
        discovery_seed=args.discovery_seed,
        label=args.label,
        **kwargs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Monotonic robustness ({artifact['label']})")
    print("Bootstrap resampling:")
    for row in artifact["bootstrap_summary"]:
        print(
            f"  {row['case_id']}: exact={row['exact_recovery']:.2f}, "
            f"dl={row['latent_recovery']:.2f}, ds={row['structural_recovery']:.2f}, "
            f"G+={row['focus_positive_edge_rate']:.2f}, "
            f"G±={row['focus_signed_edge_rate']:.2f}"
        )
    print("Observation noise:")
    for row in artifact["noise_summary"]:
        print(
            f"  {row['case_id']} sigma={row['sigma']:.2f}: "
            f"exact={row['exact_recovery']:.2f}, "
            f"dl={row['latent_recovery']:.2f}, ds={row['structural_recovery']:.2f}, "
            f"G+={row['focus_positive_edge_rate']:.2f}, "
            f"G±={row['focus_signed_edge_rate']:.2f}"
        )


if __name__ == "__main__":
    main()
