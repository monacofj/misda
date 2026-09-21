"""Map Pearson/Spearman recovery across monotonic nonlinearity and noise.

The sweep isolates one family of strictly increasing objective relations,

    y = exp(k (x - 1)),  x in [0, 1],

and varies both the nonlinearity parameter ``k`` and scaled additive
observation noise.  Two contexts are evaluated: the monotonic pair alone and
the same pair with one independent objective.  The latter probes the
multivariate threshold effect that discriminated Pearson and Spearman in the
``monotonic`` benchmark.

Noise is scaled by each objective's sample standard deviation, matching the
robustness experiment, and common random numbers are used across k and sigma
within each context/replicate.  MISDA's discovery seed is held fixed.
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
K_VALUES = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0, 650.0)
NOISE_SIGMAS = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10)
NOISE_SEEDS = tuple(range(4001, 4011))
CONTEXTS = ("pair", "plus_independent")


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


def _build_clean(n, seed, k, context):
    x = np.linspace(0.0, 1.0, int(n))
    y = np.exp(float(k) * (x - 1.0))
    if context == "pair":
        Y = np.column_stack((x, y))
        expected = (1, 1)
    elif context == "plus_independent":
        independent = np.random.default_rng(int(seed)).normal(size=int(n))
        Y = np.column_stack((x, y, independent))
        expected = (2, 2)
    else:
        raise ValueError(f"Unknown context: {context}")
    return Y, expected


def _clean_correlations(Y):
    x = np.asarray(Y[:, 0], dtype=float)
    y = np.asarray(Y[:, 1], dtype=float)
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(stats.spearmanr(x, y).statistic),
    }


def _discover_record(
    Y,
    *,
    context,
    k,
    sigma,
    noise_seed,
    discovery_seed,
    expected,
):
    result = misda.discover(
        Y,
        name=f"monotonic_sweep_{context}_k{k:g}",
        seed=int(discovery_seed),
    )
    analysis = result.analysis
    latent_expected, structural_expected = expected
    latent_observed = int(analysis.latent_dimension)
    structural_observed = int(analysis.structural_dimension)
    return {
        "context": context,
        "k": float(k),
        "sigma": float(sigma),
        "noise_seed": int(noise_seed),
        "latent_expected": int(latent_expected),
        "latent_observed": latent_observed,
        "latent_exact": latent_observed == int(latent_expected),
        "structural_expected": int(structural_expected),
        "structural_observed": structural_observed,
        "structural_exact": structural_observed == int(structural_expected),
        "exact": (
            latent_observed == int(latent_expected)
            and structural_observed == int(structural_expected)
        ),
        "focus_positive_edge": bool(analysis.structural_graph.has_edge(0, 1)),
        "focus_signed_edge": bool(analysis.dependence_graph.has_edge(0, 1)),
        "selected_dimension": int(misda.rank(result).selected_dimension),
        "separation_status": str(analysis.separation_status.value),
        "support_status": result.support.status,
    }


def _summarize(records, contexts, k_values, sigmas):
    summary = []
    for context in contexts:
        for k in k_values:
            for sigma in sigmas:
                group = [
                    row
                    for row in records
                    if row["context"] == context
                    and row["k"] == float(k)
                    and row["sigma"] == float(sigma)
                ]
                summary.append(
                    {
                        "context": context,
                        "k": float(k),
                        "sigma": float(sigma),
                        "replicates": len(group),
                        "exact_recovery": _mean(group, "exact"),
                        "latent_recovery": _mean(group, "latent_exact"),
                        "structural_recovery": _mean(group, "structural_exact"),
                        "focus_positive_edge_rate": _mean(group, "focus_positive_edge"),
                        "focus_signed_edge_rate": _mean(group, "focus_signed_edge"),
                        "modal_latent_dimension": _mode_dimension(group, "latent_observed"),
                        "modal_structural_dimension": _mode_dimension(
                            group, "structural_observed"
                        ),
                    }
                )
    return summary


def _tolerance(summary, contexts, k_values, threshold=0.8):
    rows = []
    for context in contexts:
        for k in k_values:
            eligible = [
                row["sigma"]
                for row in summary
                if row["context"] == context
                and row["k"] == float(k)
                and row["exact_recovery"] >= float(threshold)
            ]
            rows.append(
                {
                    "context": context,
                    "k": float(k),
                    "recovery_threshold": float(threshold),
                    "max_sigma_meeting_threshold": max(eligible) if eligible else None,
                }
            )
    return rows


def run_monotonic_sweep(
    *,
    n=DEFAULT_N,
    base_seed=DEFAULT_SEED,
    discovery_seed=DEFAULT_SEED,
    k_values=K_VALUES,
    noise_sigmas=NOISE_SIGMAS,
    noise_seeds=NOISE_SEEDS,
    contexts=CONTEXTS,
    label="unspecified",
):
    n = int(n)
    base_seed = int(base_seed)
    discovery_seed = int(discovery_seed)
    k_values = tuple(float(value) for value in k_values)
    noise_sigmas = tuple(float(value) for value in noise_sigmas)
    noise_seeds = tuple(int(value) for value in noise_seeds)
    contexts = tuple(str(value) for value in contexts)
    if not k_values or not noise_sigmas or not noise_seeds or not contexts:
        raise ValueError("k values, sigmas, seeds, and contexts must be non-empty")

    records = []
    clean = []

    # One noise field per context/replicate is reused across every k and sigma.
    # This makes changes along either axis attributable to the experimental
    # factor rather than to a different noise realization.
    epsilons = {}
    for context_position, context in enumerate(contexts):
        width = 2 if context == "pair" else 3
        for noise_seed in noise_seeds:
            sequence = np.random.SeedSequence(
                [noise_seed, context_position, base_seed, 17]
            )
            epsilons[(context, noise_seed)] = np.random.default_rng(sequence).normal(
                size=(n, width)
            )

    for context in contexts:
        for k in k_values:
            Y, expected = _build_clean(n, base_seed, k, context)
            clean.append(
                {
                    "context": context,
                    "k": float(k),
                    "correlation": _clean_correlations(Y),
                    "latent_expected": int(expected[0]),
                    "structural_expected": int(expected[1]),
                }
            )
            scale = np.std(Y, axis=0, ddof=1)
            scale = np.where(scale > 0.0, scale, 1.0)
            for noise_seed in noise_seeds:
                epsilon = epsilons[(context, noise_seed)]
                for sigma in noise_sigmas:
                    noisy = Y + float(sigma) * scale * epsilon
                    records.append(
                        _discover_record(
                            noisy,
                            context=context,
                            k=k,
                            sigma=sigma,
                            noise_seed=noise_seed,
                            discovery_seed=discovery_seed,
                            expected=expected,
                        )
                    )

    summary = _summarize(records, contexts, k_values, noise_sigmas)
    return {
        "suite": "monotonic_sweep",
        "label": str(label),
        "purpose": (
            "Map monotonic-dependence recovery across nonlinearity strength k "
            "and scaled observation noise sigma."
        ),
        "parameters": {
            "n": n,
            "base_seed": base_seed,
            "discovery_seed": discovery_seed,
            "k_values": list(k_values),
            "noise_sigmas": list(noise_sigmas),
            "noise_seeds": list(noise_seeds),
            "contexts": list(contexts),
            "noise_scaling": "per-objective sample standard deviation",
            "common_random_numbers_across_k_and_sigma": True,
        },
        "clean": clean,
        "records": records,
        "summary": summary,
        "tolerance_80": _tolerance(summary, contexts, k_values, threshold=0.8),
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
            k_values=(1.0, 20.0, 100.0, 650.0),
            noise_sigmas=(0.0, 0.01, 0.05),
            noise_seeds=NOISE_SEEDS[:2],
        )
    artifact = run_monotonic_sweep(
        n=args.n,
        base_seed=args.base_seed,
        discovery_seed=args.discovery_seed,
        label=args.label,
        **kwargs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Monotonic k x noise sweep ({artifact['label']})")
    print("Clean correlations:")
    for row in artifact["clean"]:
        if row["context"] != "pair":
            continue
        corr = row["correlation"]
        print(
            f"  k={row['k']:g}: Pearson={corr['pearson']:.4f}, "
            f"Spearman={corr['spearman']:.4f}"
        )
    print("80% exact-recovery noise tolerance:")
    for row in artifact["tolerance_80"]:
        tolerance = row["max_sigma_meeting_threshold"]
        text = "none" if tolerance is None else f"{tolerance:.3f}"
        print(f"  {row['context']} k={row['k']:g}: sigma<={text}")


if __name__ == "__main__":
    main()
