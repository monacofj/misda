# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""End-to-end optimization audit for MISDA ranking policies.

The experiment keeps discovery fixed and compares the optimization consequence
of the MIS selected by ``size_span`` with the MIS selected by
``dominance_preservation``. All optimizer outputs are re-evaluated in the
original objective space and scored against the same MoeaBench calibration.

This is research instrumentation. It does not change MISDA's public ranking
default and it deliberately does not infer a winner from the screening score.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

import misda
import moeabench as mb
from moeabench.core.run import Population


FORMAT_VERSION = 1
M = 10
SCREEN_POWER = 9
POPULATION = 60
GENERATIONS = 400
CHECKPOINTS = (50, 100, 200, 400)
MISDA_SEED = 123
MOEA_SEEDS = (321,)
REF_DIRS_SEED = 456
HV_MC_SAMPLES = 20_000
HV_MC_SEED = 789
POLICIES = (misda.SIZE_SPAN, misda.DOMINANCE_PRESERVATION)


class ObjectiveProjectionMOP(mb.mops.BaseMop):
    """Expose selected objectives of a source MOP without duplicating formulas."""

    def __init__(self, source_mop, objective_indices):
        self.source_mop = source_mop
        self.objective_indices = tuple(int(i) for i in objective_indices)
        if len(self.objective_indices) < 2:
            raise ValueError("NSGA-III requires at least two selected objectives.")
        super().__init__(
            name=f"{source_mop.name}[MISDA:{','.join(map(str, self.objective_indices))}]",
            M=len(self.objective_indices),
            N=source_mop.N,
            xl=np.asarray(source_mop.xl, dtype=float),
            xu=np.asarray(source_mop.xu, dtype=float),
        )

    def evaluation(self, X, n_ieq_constr=0):
        result = dict(self.source_mop.evaluation(X, n_ieq_constr))
        result["F"] = np.asarray(result["F"], dtype=float)[:, self.objective_indices]
        return result

    def ps(self, n_points=100):
        return self.source_mop.ps(n_points)


def problem_suite():
    return {
        "DTLZ5": mb.mops.DTLZ5(M=M),
        "DPF1": mb.mops.DPF1(M=M, D=2, K=5),
    }


def _safe_float(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _nd_front(F):
    return np.asarray(Population(np.asarray(F, dtype=float)).non_dominated().objectives)


def _canonical_rows(X):
    X = np.asarray(X, dtype=float)
    order = np.lexsort(X.T[::-1])
    return X[order]


def _paired_initial_population(mop, *, size, seed):
    rng = np.random.default_rng(int(seed))
    return rng.uniform(
        np.asarray(mop.xl, dtype=float),
        np.asarray(mop.xu, dtype=float),
        size=(int(size), mop.N),
    )


def _sample_screening(mop, *, power, seed):
    sampler = qmc.Sobol(d=mop.N, scramble=True, seed=int(seed))
    X = qmc.scale(
        sampler.random_base2(m=int(power)),
        np.asarray(mop.xl, dtype=float),
        np.asarray(mop.xu, dtype=float),
    )
    F = np.asarray(mop.evaluation(X)["F"], dtype=float)
    frame = pd.DataFrame(F, columns=[f"f{i + 1}" for i in range(mop.M)])
    return X, frame


def _candidate_snapshot(mis_set, ranking):
    candidate = ranking.mis()
    support = mis_set.support_for(candidate)
    return {
        "policy": ranking.policy,
        "selected_indices": [int(i) for i in candidate.indices],
        "selected_labels": [str(label) for label in candidate.objectives],
        "selected_dimension": int(candidate.size),
        "dominance": {
            "new_dominance_rate": float(candidate.dominance.new_dominance_rate),
            "new_dominance_pairs": int(candidate.dominance.new_dominance_pairs),
            "exact_preservation": bool(candidate.dominance.exact_preservation),
        },
        "pareto": (
            {
                "retention": _safe_float(candidate.pareto.retention),
                "validity": _safe_float(candidate.pareto.validity),
                "jaccard": _safe_float(candidate.pareto.jaccard),
                "exact_preservation": bool(candidate.pareto.exact_preservation),
            }
            if candidate.pareto is not None
            else None
        ),
        "linear": (
            {
                "mean_r2": _safe_float(candidate.linear.mean_r2),
                "worst_r2": _safe_float(candidate.linear.worst_r2),
            }
            if candidate.linear is not None
            else None
        ),
        "support": {
            "status": support.status,
            "reasons": list(support.reasons),
        },
        "assessment": {
            "status": ranking.assessment.status,
            "trustworthy": bool(ranking.assessment.trustworthy),
        },
    }


def screen_rankings(mop, *, power=SCREEN_POWER, seed=MISDA_SEED):
    """Discover once, then derive both ranking views from the same MIS universe."""

    X, frame = _sample_screening(mop, power=power, seed=seed)
    mis_set = misda.discover(frame, name=f"{mop.name} ranking-policy screening", seed=seed)
    mis_set.evaluate(metrics=("dominance",), candidates="all")

    rankings = {policy: misda.rank(mis_set, policy=policy) for policy in POLICIES}
    selected = []
    for ranking in rankings.values():
        candidate = ranking.mis()
        if all(candidate is not observed for observed in selected):
            selected.append(candidate)
    mis_set.evaluate(metrics=("linear", "pareto"), candidates=selected)

    snapshots = {
        policy: _candidate_snapshot(mis_set, ranking)
        for policy, ranking in rankings.items()
    }
    return {
        "X": X,
        "F": frame,
        "mis_set": mis_set,
        "rankings": rankings,
        "policies": snapshots,
        "same_selected_mis": (
            snapshots[misda.SIZE_SPAN]["selected_indices"]
            == snapshots[misda.DOMINANCE_PRESERVATION]["selected_indices"]
        ),
        "dominance_delta": (
            snapshots[misda.DOMINANCE_PRESERVATION]["dominance"]["new_dominance_rate"]
            - snapshots[misda.SIZE_SPAN]["dominance"]["new_dominance_rate"]
        ),
    }


def _calibrated_ground_truth(mop):
    sidecar = (
        Path(tempfile.gettempdir())
        / f"misda-ranking-policy-{mop.name}-M{mop.M}-calibration.json"
    )
    mop.calibrate(source_baseline=str(sidecar), force=True)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    gt = np.asarray(payload["gt_reference"], dtype=float)
    if gt.ndim != 2 or gt.shape[1] != mop.M:
        raise ValueError(
            f"Calibrated GT for {mop.name} has shape {gt.shape}; expected (*, {mop.M})."
        )
    return gt, str(sidecar)


def _make_experiment(mop, *, X0, generations, population, seed, ref_dirs_seed, name):
    experiment = mb.experiment(
        mop=mop,
        moea=mb.moeas.NSGA3(
            population=int(population),
            generations=int(generations),
            seed=int(seed),
            ref_dirs_seed=int(ref_dirs_seed),
            sampling=np.asarray(X0, dtype=float).copy(),
        ),
    )
    experiment.name = name
    experiment.run(repeat=1, silent=True)
    np.testing.assert_allclose(
        _canonical_rows(experiment[0].history("x")[0]),
        _canonical_rows(X0),
    )
    return experiment


def _front_at_generation(experiment, original_mop, generation):
    history = experiment[0].history("x")
    generation = int(generation)
    if generation < 1 or generation > len(history):
        raise ValueError(
            f"generation {generation} outside recorded history 1..{len(history)}"
        )
    X = np.asarray(history[generation - 1], dtype=float)
    F = np.asarray(original_mop.evaluation(X)["F"], dtype=float)
    return _nd_front(F)


def _front_metrics(front, gt, *, mc_seed=HV_MC_SEED):
    front = np.asarray(front, dtype=float)
    gt = np.asarray(gt, dtype=float)
    gd = float(mb.metrics.gdplus(front, ref=gt, progress=False))
    igd = float(mb.metrics.igdplus(front, ref=gt, progress=False))
    hv = float(
        mb.metrics.hypervolume(
            front,
            ref=gt,
            mode="auto",
            scale="rel",
            n_samples=HV_MC_SAMPLES,
            mc_seed=int(mc_seed),
            progress=False,
        )
    )
    return {
        "gdplus": gd,
        "igdplus": igd,
        "relative_hv": hv,
        "nd_size": int(len(front)),
    }


def _metric_delta(left, right):
    return {
        "gdplus": float(left["gdplus"] - right["gdplus"]),
        "igdplus": float(left["igdplus"] - right["igdplus"]),
        "relative_hv": float(left["relative_hv"] - right["relative_hv"]),
    }


def _validate_checkpoints(checkpoints, generations):
    values = tuple(sorted(set(int(value) for value in checkpoints)))
    if not values or values[0] < 1:
        raise ValueError("checkpoints must contain positive generation numbers")
    if values[-1] > int(generations):
        raise ValueError("the largest checkpoint cannot exceed generations")
    return values


def run_seed(
    name,
    mop,
    screening,
    gt,
    *,
    generations,
    checkpoints,
    population,
    moea_seed,
    ref_dirs_seed,
):
    X0 = _paired_initial_population(mop, size=population, seed=moea_seed)

    full = _make_experiment(
        mop,
        X0=X0,
        generations=generations,
        population=population,
        seed=moea_seed,
        ref_dirs_seed=ref_dirs_seed,
        name=f"{name} — Full — seed {moea_seed}",
    )

    experiments = {"full": full}
    for policy in POLICIES:
        selected = screening["policies"][policy]["selected_indices"]
        reduced_mop = ObjectiveProjectionMOP(mop, selected)
        experiments[policy] = _make_experiment(
            reduced_mop,
            X0=X0,
            generations=generations,
            population=population,
            seed=moea_seed,
            ref_dirs_seed=ref_dirs_seed,
            name=f"{name} — Reduced/{policy} — seed {moea_seed}",
        )

    checkpoint_rows = []
    for generation in checkpoints:
        metrics = {}
        for arm, experiment in experiments.items():
            front = _front_at_generation(experiment, mop, generation)
            metrics[arm] = _front_metrics(front, gt)
        checkpoint_rows.append(
            {
                "generation": int(generation),
                "arms": metrics,
                "dominance_minus_size_span": _metric_delta(
                    metrics[misda.DOMINANCE_PRESERVATION],
                    metrics[misda.SIZE_SPAN],
                ),
            }
        )

    final = checkpoint_rows[-1]
    return {
        "moea_seed": int(moea_seed),
        "checkpoints": checkpoint_rows,
        "final": final,
        "full_minus_reduced": {
            policy: _metric_delta(final["arms"]["full"], final["arms"][policy])
            for policy in POLICIES
        },
    }


def _mean_metrics(rows, arm):
    keys = ("gdplus", "igdplus", "relative_hv", "nd_size")
    return {
        key: float(np.mean([row["final"]["arms"][arm][key] for row in rows]))
        for key in keys
    }


def summarize_runs(runs):
    summary = {arm: _mean_metrics(runs, arm) for arm in ("full", *POLICIES)}
    summary["dominance_minus_size_span"] = _metric_delta(
        summary[misda.DOMINANCE_PRESERVATION],
        summary[misda.SIZE_SPAN],
    )
    return summary


def run_case(
    name,
    mop,
    *,
    screen_power=SCREEN_POWER,
    misda_seed=MISDA_SEED,
    population=POPULATION,
    generations=GENERATIONS,
    checkpoints=CHECKPOINTS,
    moea_seeds=MOEA_SEEDS,
    ref_dirs_seed=REF_DIRS_SEED,
):
    checkpoints = _validate_checkpoints(checkpoints, generations)
    screening = screen_rankings(mop, power=screen_power, seed=misda_seed)
    gt, calibration_sidecar = _calibrated_ground_truth(mop)

    runs = [
        run_seed(
            name,
            mop,
            screening,
            gt,
            generations=generations,
            checkpoints=checkpoints,
            population=population,
            moea_seed=seed,
            ref_dirs_seed=ref_dirs_seed,
        )
        for seed in moea_seeds
    ]

    return {
        "problem": name,
        "M": int(mop.M),
        "N": int(mop.N),
        "screening": {
            "n": int(len(screening["F"])),
            "misda_seed": int(misda_seed),
            "same_selected_mis": bool(screening["same_selected_mis"]),
            "dominance_delta": float(screening["dominance_delta"]),
            "policies": screening["policies"],
        },
        "optimization": {
            "population": int(population),
            "generations": int(generations),
            "checkpoints": [int(value) for value in checkpoints],
            "moea_seeds": [int(seed) for seed in moea_seeds],
            "ref_dirs_seed": int(ref_dirs_seed),
            "gt_size": int(len(gt)),
            "calibration_sidecar": calibration_sidecar,
            "runs": runs,
            "mean_final": summarize_runs(runs),
        },
    }


def _print_case(result):
    print(f"\n=== {result['problem']} ===")
    screening = result["screening"]
    for policy in POLICIES:
        row = screening["policies"][policy]
        print(
            f"{policy}: {row['selected_labels']} "
            f"(D={row['dominance']['new_dominance_rate']:.6g}, "
            f"status={row['assessment']['status']})"
        )
    print(
        "screening dominance delta (dominance_preservation - size_span): "
        f"{screening['dominance_delta']:.6g}"
    )
    print("mean final original-space metrics:")
    for arm, metrics in result["optimization"]["mean_final"].items():
        if arm == "dominance_minus_size_span":
            continue
        print(
            f"  {arm}: GD+={metrics['gdplus']:.6g}, "
            f"IGD+={metrics['igdplus']:.6g}, HV={metrics['relative_hv']:.6g}, "
            f"ND={metrics['nd_size']:.1f}"
        )
    delta = result["optimization"]["mean_final"]["dominance_minus_size_span"]
    print(
        "  dominance - size_span: "
        f"ΔGD+={delta['gdplus']:.6g}, ΔIGD+={delta['igdplus']:.6g}, "
        f"ΔHV={delta['relative_hv']:.6g}"
    )
    print("Interpret policy differences only after inspecting Full checkpoint stabilization.")


def run_audit(
    *,
    problem_names=("DTLZ5", "DPF1"),
    screen_power=SCREEN_POWER,
    misda_seed=MISDA_SEED,
    population=POPULATION,
    generations=GENERATIONS,
    checkpoints=CHECKPOINTS,
    moea_seeds=MOEA_SEEDS,
    ref_dirs_seed=REF_DIRS_SEED,
):
    suite = problem_suite()
    unknown = sorted(set(problem_names) - set(suite))
    if unknown:
        raise ValueError(f"unknown problem(s): {', '.join(unknown)}")
    results = [
        run_case(
            name,
            suite[name],
            screen_power=screen_power,
            misda_seed=misda_seed,
            population=population,
            generations=generations,
            checkpoints=checkpoints,
            moea_seeds=moea_seeds,
            ref_dirs_seed=ref_dirs_seed,
        )
        for name in problem_names
    ]
    return {
        "format_version": FORMAT_VERSION,
        "question": (
            "Does the dominance-preservation advantage observed on Y predict "
            "a better end-to-end Reduced optimization result than size_span?"
        ),
        "results": results,
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem",
        action="append",
        choices=("DTLZ5", "DPF1"),
        dest="problems",
        help="Problem to run; repeat to select both. Defaults to DTLZ5 and DPF1.",
    )
    parser.add_argument("--screen-power", type=int, default=SCREEN_POWER)
    parser.add_argument("--misda-seed", type=int, default=MISDA_SEED)
    parser.add_argument("--population", type=int, default=POPULATION)
    parser.add_argument("--generations", type=int, default=GENERATIONS)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=int,
        dest="checkpoints",
        help="1-based generation checkpoint; repeat as needed.",
    )
    parser.add_argument(
        "--moea-seed",
        action="append",
        type=int,
        dest="moea_seeds",
        help="Independent optimizer seed; repeat for multi-seed runs.",
    )
    parser.add_argument("--ref-dirs-seed", type=int, default=REF_DIRS_SEED)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    result = run_audit(
        problem_names=tuple(args.problems or ("DTLZ5", "DPF1")),
        screen_power=args.screen_power,
        misda_seed=args.misda_seed,
        population=args.population,
        generations=args.generations,
        checkpoints=tuple(args.checkpoints or CHECKPOINTS),
        moea_seeds=tuple(args.moea_seeds or MOEA_SEEDS),
        ref_dirs_seed=args.ref_dirs_seed,
    )
    for case in result["results"]:
        _print_case(case)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote {args.output}")
    return result


if __name__ == "__main__":
    main()
