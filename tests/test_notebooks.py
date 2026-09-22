"""Executable checks for the benchmark notebook front ends."""

import json
from pathlib import Path

import pytest

import misda
from misda.benchmark import BenchmarkResult


BANNED_SOURCE = (
    "importlib.reload",
    "target_fidelity",
    ".validate(",
    "FeatureAgglomeration",
    "method='adaptive'",
    'method="adaptive"',
    "misda.analyze(",
    "misda.heavy(",
)

BENCHMARK_CASES = (
    ("independence", "Case 1 - Independent objectives"),
    ("total_redundancy", "Case 2 - Complete positive redundancy"),
    ("blocks_4x5", "Case 3 - Four redundant blocks"),
    ("blocks_2x10", "Case 4 - Two redundant blocks"),
    ("mixed_independent_and_blocks", "Case 5 - Mixed independent and redundant objectives"),
    ("monotonic_redundancy", "Case 6 - Nonlinear monotonic redundancy"),
    ("antagonistic_linear_groups", "Case 7 - Antagonistic linear groups"),
    ("tradeoff_redundancies", "Case 8 - Trade-off with redundant families"),
    ("nonlinear_blocks_4x5", "Case 9 - Nonlinear redundant blocks"),
    ("antagonistic_nonlinear_groups", "Case 10 - Antagonistic nonlinear groups"),
    ("overlapping_factors", "Case 11 - Overlapping latent factors"),
    ("transitive_chain", "Case 12 - Transitive positive chain"),
    ("regime_switching", "Case 13 - Regime-switching dependence"),
)


def _read_notebook(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    return notebook, source


def _execute_notebook(path, monkeypatch, overrides=None, skip_tags=("setup",)):
    notebook, _ = _read_notebook(path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {"__name__": f"notebook_{path.stem}"}
    overrides = overrides or {}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tags = cell.get("metadata", {}).get("tags", [])
        if any(tag in tags for tag in skip_tags):
            continue
        for tag, values in overrides.items():
            if tag in tags:
                namespace.update(values)
        cell_source = "".join(cell.get("source", []))
        exec(compile(cell_source, f"{path}:cell-{index}", "exec"), namespace)
    return notebook, namespace


def _case_headings(notebook):
    return [
        "".join(cell.get("source", [])).splitlines()[0]
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
        and "".join(cell.get("source", [])).startswith("## Case ")
    ]


def test_controlled_notebook_runs_explicit_documented_suite(monkeypatch):
    path = Path("benchmarks/controlled.ipynb")
    notebook, source = _read_notebook(path)

    assert notebook["nbformat"] == 4
    assert all(term not in source for term in BANNED_SOURCE)
    assert "for problem in bench.PROBLEMS" not in source
    assert "bench.PROBLEM_BY_ID" in source
    assert "def run_case(problem_id)" in source
    assert "problem.generate(N=N, seed=SEED, sigma=SIGMA)" in source
    assert "bench.diagnostic_truth(problem, dataset.Z)" in source
    assert "dataset.Y" in source
    assert "misda.discover(" in source
    assert 'mis_set.evaluate(metrics=("linear", "pareto"))' in source
    assert "ranking = misda.rank(mis_set)" in source
    assert "ranking.mis().graph_plot()" in source
    assert "ranking.mis().front_plot()" in source
    assert "misda.benchmark(mis_set, truth)" in source
    assert "# Adversarial diagnostics" in source
    assert "benchmark_summary = misda.compile_benchmark_summary(benchmark_results)" in source

    assert _case_headings(notebook) == [f"## {name}" for _, name in BENCHMARK_CASES]
    for index, (problem_id, _name) in enumerate(BENCHMARK_CASES, start=1):
        assert f'case_{index} = run_case("{problem_id}")' in source

    _, namespace = _execute_notebook(
        path,
        monkeypatch,
        overrides={"benchmark-run": {"N": 64}},
    )
    results = namespace["benchmark_results"]
    assert tuple(results) == tuple(problem_id for problem_id, _ in BENCHMARK_CASES)
    assert [item["truth"]["name"] for item in results.values()] == [
        name for _, name in BENCHMARK_CASES
    ]
    assert len(results) == 13
    assert all(isinstance(item["result_obj"], misda.MISSet) for item in results.values())
    assert all(isinstance(item["ranking_obj"], misda.Ranking) for item in results.values())
    assert all(isinstance(item["benchmark_obj"], BenchmarkResult) for item in results.values())
    assert all(item["dataset"].sigma == 0.0 for item in results.values())
    assert all(item["dataset"].Y.equals(item["dataset"].Z) for item in results.values())
    assert len(namespace["benchmark_summary"]) == 13


def test_controlled_noisy_notebook_is_runner_frontend_and_runs_suite(monkeypatch):
    path = Path("benchmarks/controlled_noisy.ipynb")
    notebook, source = _read_notebook(path)

    assert notebook["nbformat"] == 4
    assert all(term not in source for term in BANNED_SOURCE)
    assert "analyze_controlled_noisy_problem" in source
    assert "def run_case(problem_id)" in source
    assert "OBSERVATION_SEED = 456" in source
    assert "SIGMA = 0.10" in source
    assert "noisy_summary = misda.compile_benchmark_summary(noisy_results)" in source
    assert _case_headings(notebook) == [f"## {name}" for _, name in BENCHMARK_CASES]
    for index, (problem_id, _name) in enumerate(BENCHMARK_CASES, start=1):
        assert f'case_{index} = run_case("{problem_id}")' in source

    _, namespace = _execute_notebook(
        path,
        monkeypatch,
        overrides={"benchmark-run": {"N": 64}},
    )
    results = namespace["noisy_results"]
    assert tuple(results) == tuple(problem_id for problem_id, _ in BENCHMARK_CASES)
    assert len(results) == 13
    assert all(isinstance(item["result_obj"], misda.MISSet) for item in results.values())
    assert all(isinstance(item["benchmark_obj"], BenchmarkResult) for item in results.values())
    assert all(item["dataset"].sigma == pytest.approx(0.10) for item in results.values())
    assert all(item["dataset"].sample_seed == 123 for item in results.values())
    assert all(item["dataset"].observation_seed == 456 for item in results.values())
    assert all(not item["dataset"].Y.equals(item["dataset"].Z) for item in results.values())
    assert len(namespace["noisy_summary"]) == 13


def test_sampling_robustness_notebook_is_runner_frontend(monkeypatch):
    path = Path("benchmarks/sampling_robustness.ipynb")
    notebook, source = _read_notebook(path)

    assert notebook["nbformat"] == 4
    assert all(term not in source for term in BANNED_SOURCE)
    assert "run_sampling_robustness" in source
    assert "CONTROLLED_PROBLEM_IDS" in source
    assert "SAMPLING_REPLICATE_SEEDS" in source
    assert "sampling_artifact" in source
    assert "sampling_summary" in source

    _, namespace = _execute_notebook(
        path,
        monkeypatch,
        overrides={
            "sampling-run": {
                "N": 48,
                "PROBLEM_IDS": ("independence", "total_redundancy"),
                "REPLICATE_SEEDS": (101, 202),
            }
        },
    )
    sampling = namespace["sampling"]
    summary = namespace["sampling_summary"]
    assert len(sampling) == 4
    assert len(summary) == 2
    assert set(sampling["problem_id"]) == {"independence", "total_redundancy"}
    assert sampling.groupby("problem_id")["sample_seed"].nunique().eq(2).all()
    assert set(summary["replicates"]) == {2}


def test_noise_robustness_notebook_is_runner_frontend(monkeypatch):
    path = Path("benchmarks/noisy_robustness.ipynb")
    notebook, source = _read_notebook(path)

    assert notebook["nbformat"] == 4
    assert all(term not in source for term in BANNED_SOURCE)
    assert "controlled_noisy.ipynb" in source
    assert "run_noisy_robustness" in source
    assert "NOISE_SIGMAS" in source
    assert "NOISE_REPLICATE_SEEDS" in source
    assert "robustness_artifact" in source
    assert "pareto_observation_jaccard" in source
    assert "pareto_reduction_jaccard" in source
    assert "pareto_end_to_end_jaccard" in source

    _, namespace = _execute_notebook(
        path,
        monkeypatch,
        overrides={
            "robustness-run": {
                "N": 48,
                "PROBLEM_IDS": ("independence", "total_redundancy"),
                "SIGMAS": (0.0, 0.10),
                "REPLICATE_SEEDS": (101,),
            }
        },
        skip_tags=("setup", "visualization"),
    )
    robustness = namespace["robustness"]
    summary = namespace["robustness_summary"]
    assert len(robustness) == 4
    assert len(summary) == 4
    assert set(robustness["problem_id"]) == {"independence", "total_redundancy"}
    assert set(robustness["sigma"]) == {0.0, 0.10}
    assert robustness.groupby("problem_id")["sample_seed"].nunique().eq(1).all()
    assert "pareto_end_to_end_jaccard" in robustness.columns


def test_comparison_notebook_uses_diagnostic_truth_without_pca_dimension_cutoff(monkeypatch):
    path = Path("benchmarks/comparison.ipynb")
    notebook, source = _read_notebook(path)

    assert notebook["nbformat"] == 4
    assert all(term not in source for term in BANNED_SOURCE)
    assert "COMPARISON_PROBLEM_IDS" in source
    assert "bench.PROBLEM_BY_ID" in source
    assert "problem.generate(N=N, seed=SEED, sigma=0.0)" in source
    assert "bench.diagnostic_truth(problem, dataset.Z)" in source
    assert "pca_external_reconstruction_curve" in source
    assert "pca_at_latent_truth" in source
    assert "pca_at_structural_truth" in source
    assert "misda_latent_error" in source
    assert "misda_structural_error" in source
    assert "explained_variance" not in source
    assert "COMPARATIVE_CASES" not in source

    _, namespace = _execute_notebook(
        path,
        monkeypatch,
        overrides={
            "comparison-run": {
                "N": 48,
                "COMPARISON_PROBLEM_IDS": (
                    "total_redundancy",
                    "antagonistic_linear_groups",
                ),
            }
        },
    )
    observed = namespace["comparison_results"]
    comparison = namespace["comparison"]
    assert set(observed) == {"total_redundancy", "antagonistic_linear_groups"}
    assert len(comparison) == 2
    assert set(comparison["problem_id"]) == set(observed)
    assert comparison["misda_latent_error"].ge(0).all()
    assert comparison["misda_structural_error"].ge(0).all()
    assert comparison["pca_at_misda_dimension"].notna().all()
    assert comparison["pca_at_latent_truth"].notna().all()
    assert comparison["pca_at_structural_truth"].notna().all()


def test_classical_notebook_keeps_reference_geometry_separate_from_misda_truth(monkeypatch):
    path = Path("benchmarks/classical.ipynb")
    notebook, source = _read_notebook(path)

    assert notebook["nbformat"] == 4
    assert all(term not in source for term in BANNED_SOURCE)
    assert "bench.CLASSICAL_MOPS" in source
    assert "on_front=ON_FRONT" in source
    assert "seed=SEED" in source
    assert "pareto_manifold_dimension" in source
    assert "misda_latent" in source
    assert "misda_structural" in source
    assert "misda.benchmark(" not in source
    assert "latent_expected" not in source
    assert "structural_expected" not in source

    _, namespace = _execute_notebook(
        path,
        monkeypatch,
        overrides={
            "classical-run": {
                "N": 48,
                "M": 5,
                "N_VARS": 14,
                "PROBLEM_IDS": ("dtlz2", "dtlz5"),
            }
        },
    )
    results = namespace["classical_results"]
    summary = namespace["classical_summary"]
    assert set(results) == {"dtlz2", "dtlz5"}
    assert len(summary) == 2
    assert set(summary["pareto_manifold_dimension"]) == {1, 4}
    assert summary["misda_latent"].notna().all()
    assert summary["misda_structural"].notna().all()
    assert summary["pareto_jaccard"].notna().all()



def test_optimization_notebook_uses_paired_original_space_protocol():
    path = Path("benchmarks/optimization.ipynb")
    notebook, source = _read_notebook(path)

    assert notebook["nbformat"] == 4
    assert "misda[benchmarks]" in source
    assert "git+https://github.com/monacofj/moeabench" not in source
    assert "pip install moeabench" not in source

    # Global pre-optimization screening, independent of the MOEA and Pareto GT.
    assert "qmc.Sobol" in source
    assert "random_base2" in source
    assert "qmc.scale" in source
    assert "joint Sobol sample" in source
    assert "one-factor-at-a-time" in source

    # Objective reduction delegates to the original MOP; no benchmark formula is copied.
    assert "class ObjectiveProjectionMOP" in source
    assert "self.source_mop.evaluation" in source
    assert 'result["F"] = np.asarray(result["F"], dtype=float)[:, self.objective_indices]' in source

    # Short pilot only.
    assert 'run_optimization_case("DTLZ2", PROBLEMS["DTLZ2"])' in source
    assert 'run_optimization_case("DPF1", PROBLEMS["DPF1"])' in source
    for problem in ("DTLZ5", "DTLZ7", "DPF3", "DPF5"):
        assert f'run_optimization_case("{problem}"' not in source

    assert "M = 10" in source
    assert "GENERATIONS = 50" in source
    assert "POPULATION = 60" in source

    # Pairing: same search seed and initial X, separate reference-direction RNG.
    assert "mb.moeas.NSGA3" in source
    assert source.count("\n            seed=MOEA_SEED,\n") == 2
    assert source.count("\n            ref_dirs_seed=REF_DIRS_SEED,\n") == 2
    assert source.count("sampling=X0.copy()") == 2
    assert "def _paired_initial_population" in source
    assert "def _canonical_rows" in source
    assert "np.testing.assert_allclose" in source

    # Decision-space reduction is measured only; the MOEA domain stays unchanged.
    assert "def _active_decision_dimension" in source
    assert "measured only; MOEA domain unchanged" in source

    # Every Reduced generation is returned to original objective space.
    assert 'exp[0].history("x")' in source
    assert "original_mop.evaluation" in source
    assert "Reduced decision vectors are re-evaluated" in source
    assert "never feeds back into the Reduced search" in source

    # Pareto GT is a separate ruler for optimization quality.
    assert "misda_truth" in source
    assert "Pareto ground truth" in source
    assert "mb.metrics.gdplus" in source
    assert "mb.metrics.igdplus" in source
    assert "mb.metrics.hypervolume" in source
    assert 'scale="abs"' in source
    assert source.count("initial_data=initial_original") == 2
    assert source.count("k=POPULATION") == 2

    # Generations and total represented evaluations are explicit.
    assert "def _history_evaluations" in source
    assert '"Evaluations": result["evaluations"]' in source

    assert "mb.view.topology" in source
    assert "mb.view.radar" in source
    assert "mb.view.history" in source
    assert "mb.view.perf_history" not in source



def test_optimization_notebook_runtime_pairs_initial_population(monkeypatch):
    """Execute the paired NSGA-III initialization contract, not only source checks."""
    path = Path("benchmarks/optimization.ipynb")
    notebook, _ = _read_notebook(path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {"__name__": "optimization_runtime_smoke"}

    wanted = {
        "optimization-imports",
        "optimization-helpers-1",
        "optimization-helpers-2",
    }
    for cell in notebook["cells"]:
        if cell.get("id") in wanted:
            exec(
                compile("".join(cell.get("source", [])), str(path), "exec"),
                namespace,
            )

    mb = namespace["mb"]
    np = namespace["np"]
    mop = mb.mops.DTLZ2(M=10)
    reduced_mop = namespace["ObjectiveProjectionMOP"](mop, (0, 1, 2))
    X0 = namespace["_paired_initial_population"](mop, size=12, seed=321)

    def _run(problem):
        exp = mb.experiment(
            mop=problem,
            moea=mb.moeas.NSGA3(
                population=12,
                generations=2,
                seed=321,
                ref_dirs_seed=456,
                sampling=X0.copy(),
            ),
        )
        exp.run(repeat=1, silent=True)
        return exp

    full = _run(mop)
    reduced = _run(reduced_mop)
    canonical = namespace["_canonical_rows"]

    np.testing.assert_allclose(
        canonical(full[0].history("x")[0]),
        canonical(X0),
    )
    np.testing.assert_allclose(
        canonical(reduced[0].history("x")[0]),
        canonical(X0),
    )



def test_optimization_notebook_full_dtlz2_pilot_executes(monkeypatch):
    """Diagnostic execution of the complete DTLZ2 pilot, except rendering."""
    path = Path("benchmarks/optimization.ipynb")
    notebook, _ = _read_notebook(path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {"__name__": "optimization_full_dtlz2_smoke"}

    wanted = {
        "optimization-imports",
        "optimization-helpers-1",
        "optimization-helpers-2",
        "optimization-runner",
    }
    for cell in notebook["cells"]:
        if cell.get("id") in wanted:
            exec(
                compile("".join(cell.get("source", [])), str(path), "exec"),
                namespace,
            )

    mb = namespace["mb"]
    monkeypatch.setattr(mb.view, "topology", lambda *args, **kwargs: None)
    monkeypatch.setattr(mb.view, "radar", lambda *args, **kwargs: None)
    monkeypatch.setattr(mb.view, "history", lambda *args, **kwargs: None)
    namespace["display"] = lambda *args, **kwargs: None

    result = namespace["run_optimization_case"](
        "DTLZ2", mb.mops.DTLZ2(M=namespace["M"])
    )

    assert result["mop"].M == 10
    assert result["evaluations"] > 0
    assert len(result["full_history"]) == len(result["reduced_history"])
