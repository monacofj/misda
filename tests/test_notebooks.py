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
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "moeabench.git@5b7b379dd03bc5aa11a74447a914c724aa88c010" in pyproject
    assert "git+https://github.com/monacofj/moeabench" not in source
    assert "pip install moeabench" not in source

    # Global pre-optimization screening, independent of the MOEA and Pareto GT.
    assert "qmc.Sobol" in source
    assert "random_base2" in source
    assert "qmc.scale" in source
    assert "joint Sobol sample" in source
    assert "one-factor-at-a-time" in source
    assert 'REMOTE_REF = "issue-75-optimization-benchmark"' in source
    assert "/blob/issue-75-optimization-benchmark/benchmarks/optimization.ipynb" in source
    assert 'mis_set.evaluate(metrics=("linear", "pareto"), candidates=selected)' in source
    assert "print(ranking.report())" in source
    assert "selected.graph_plot()" in source
    assert "Selected MIS support:" in source
    assert '"front_loss"' in source
    assert '"population_impact"' in source
    assert '"selected_support"' in source
    assert '"screening_diagnostics": screening_diagnostics' in source
    runner_source = "".join(
        next(
            cell["source"]
            for cell in notebook["cells"]
            if cell.get("id") == "optimization-runner"
        )
    )
    assert runner_source.index(
        "screening_diagnostics = _diagnose_screening(screening)"
    ) < runner_source.index("full.run(repeat=1")
    assert "optimization_confrontation = optimization_summary[" in source
    assert "dtlz2_objective_truth = _dtlz2_objective_irredundancy_check" in source
    assert "dtlz5_objective_truth = _dtlz5_safe_reduction_check" in source
    assert '"DTLZ5": mb.mops.DTLZ5(M=M)' in source
    assert 'dtlz5_screening = _screen_misda(PROBLEMS["DTLZ5"])' in source
    assert 'dtlz5_screening_diagnostics = _diagnose_screening(dtlz5_screening)' in source
    assert 'dtlz5_budget = _full_budget_calibration("DTLZ5", PROBLEMS["DTLZ5"])' in source
    assert "BUDGET_CHECKPOINTS = (50, 100, 200, 400)" in source
    assert "mb.metrics.gdplus(full, ref=gt, progress=False)" in source
    assert "mb.metrics.igdplus(full, ref=gt, progress=False)" in source
    assert "serially dependent observations" in source
    assert "_dpf1_g_from_original_objectives" not in source
    assert "_plot_dpf1_g_convergence" not in source
    assert "dpf1_gt_projection = _dpf1_gt_projection_check" in source
    assert "mop.ps(n_points=n_points)" in source
    assert source.index("dpf1_gt_projection = ") < source.index('run_optimization_case("DPF1"')

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
    assert source.count("\n            seed=MOEA_SEED,\n") == 3
    assert source.count("\n            ref_dirs_seed=REF_DIRS_SEED,\n") == 3
    assert source.count("sampling=X0.copy()") == 3
    assert "def _paired_initial_population" in source
    assert "def _canonical_rows" in source
    assert "np.testing.assert_allclose" in source

    # Decision-space reduction is measured only; the MOEA domain stays unchanged.
    assert "def _active_decision_dimension" in source
    assert "measured only; MOEA domain unchanged" in source

    # Every Reduced-search generation is returned to original objective space as Full_r.
    assert 'exp[0].history("x")' in source
    assert "original_mop.evaluation" in source
    assert "Reduced-search decision vectors re-evaluated" in source
    assert "never feeds back into the Reduced search" in source

    # Pareto GT is a separate ruler for optimization quality.
    assert "misda_truth" in source
    assert "Pareto ground truth" in source
    assert "mb.metrics.gdplus" in source
    assert "mb.metrics.igdplus" in source
    assert "mb.metrics.hypervolume" in source
    assert 'scale="rel"' in source
    assert "def _calibrated_ground_truth" in source
    assert "mop.calibrate(" in source
    assert '"gt_reference"' in source
    assert "optimal(n_points=" not in source
    assert source.count("source_baseline=calibration_sidecar") == 2
    assert source.count("initial_data=initial_original") == 2
    assert "population_size=population" not in source
    assert "k=POPULATION" not in source
    assert "Full_r" in source
    assert '"full_r_front"' in source
    assert '"full_r_history"' in source
    assert "Original-space ND cardinality" in source

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


def test_optimization_screening_diagnostics_run_without_moea_or_gt(monkeypatch, capsys):
    """The selected MIS's stored diagnostics depend only on Y_screen."""
    path = Path("benchmarks/optimization.ipynb")
    notebook, _ = _read_notebook(path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {"__name__": "optimization_screening_smoke"}
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
    mop = namespace["mb"].mops.DPF1(M=10, D=2, K=5)
    screening = namespace["_screen_misda"](mop, power=5, seed=123)
    diagnostics = namespace["_diagnose_screening"](screening, show_plots=False)
    assert diagnostics["n_screen"] == 32
    assert diagnostics["selected_indices"] == screening["indices"]
    assert diagnostics["selected_dimension"] == len(screening["indices"])
    assert diagnostics["selected_support"] in {"SUPPORTED", "UNSUPPORTED"}
    assert screening["selected"].linear is not None
    assert screening["selected"].pareto is not None
    assert diagnostics["pareto_jaccard"] is not None
    assert diagnostics["front_loss"] is not None
    assert 0 <= diagnostics["population_impact"] <= 1
    out = capsys.readouterr().out
    assert "Discovery evidence: MISDA" in out
    assert "Selected MIS support:" in out
    assert "neither Pareto GT nor NSGA-III" in out


def test_optimization_dpf1_analytical_projections_without_optimizer(monkeypatch):
    """The base and previously selected objective pairs retain sampled analytic GT."""
    path = Path("benchmarks/optimization.ipynb")
    notebook, _ = _read_notebook(path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {"__name__": "dpf1_projection_smoke"}
    wanted = {"optimization-imports", "optimization-helpers-1", "optimization-helpers-2"}
    for cell in notebook["cells"]:
        if cell.get("id") in wanted:
            exec(compile("".join(cell["source"]), str(path), "exec"), namespace)

    mop = namespace["mb"].mops.DPF1(M=10, D=2, K=5)
    result = namespace["_dpf1_gt_projection_check"](mop, n_points=64)
    table = result["table"]
    assert result["gt"].shape == (64, 10)
    assert table["Objectives"].tolist() == ["f1, f2", "f2, f8"]
    assert table["Full GT ND"].tolist() == [64, 64]
    assert table["Projected GT ND"].tolist() == [64, 64]
    assert table["All GT points retained"].all()


def test_optimization_dtlz2_exact_objective_irredundancy(monkeypatch):
    """Every DTLZ2 objective has an exact axis-point dominance witness."""
    path = Path("benchmarks/optimization.ipynb")
    notebook, _ = _read_notebook(path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {"__name__": "dtlz2_truth_smoke"}
    wanted = {"optimization-imports", "optimization-helpers-1", "optimization-helpers-2"}
    for cell in notebook["cells"]:
        if cell.get("id") in wanted:
            exec(compile("".join(cell["source"]), str(path), "exec"), namespace)

    mop = namespace["mb"].mops.DTLZ2(M=10)
    table = namespace["_dtlz2_objective_irredundancy_check"](mop)
    assert len(table) == 10
    assert table["Full-space ND"].eq(2).all()
    assert table["Projected ND"].eq(1).all()
    assert table["Dominance changed"].all()


def test_optimization_dtlz5_specific_safe_pair_and_unsafe_pair(monkeypatch):
    """DTLZ5 f_(M-1),f_M is safe; degeneracy does not make f1,f_M safe."""
    path = Path("benchmarks/optimization.ipynb")
    notebook, _ = _read_notebook(path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {"__name__": "dtlz5_truth_smoke"}
    wanted = {"optimization-imports", "optimization-helpers-1", "optimization-helpers-2"}
    for cell in notebook["cells"]:
        if cell.get("id") in wanted:
            exec(compile("".join(cell["source"]), str(path), "exec"), namespace)

    mop = namespace["mb"].mops.DTLZ5(M=10)
    result = namespace["_dtlz5_safe_reduction_check"](
        mop, n_front=128, stress_power=6, seed=123
    )
    assert result["safe_pair"] == (8, 9)
    assert result["unsafe_pair"] == (0, 9)
    assert len(result["gt"]) == 128
    assert result["table"].iloc[0]["Projected PF ND"] == 128
    assert result["matched_front_dominates"].all()
    assert result["unsafe_witness"]


def test_optimization_budget_checkpoint_table_uses_metricmatrix_rows(monkeypatch):
    path = Path("benchmarks/optimization.ipynb")
    notebook, _ = _read_notebook(path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {"__name__": "budget_checkpoint_smoke"}
    wanted = {"optimization-imports", "optimization-helpers-1", "optimization-helpers-2"}
    for cell in notebook["cells"]:
        if cell.get("id") in wanted:
            exec(compile("".join(cell["source"]), str(path), "exec"), namespace)

    Matrix = namespace["mb"].metrics.MetricMatrix
    metrics = {
        "GD+": Matrix([[4.0], [3.0], [2.0], [1.0]], metric_name="GD+"),
        "IGD+": Matrix([[8.0], [6.0], [4.0], [2.0]], metric_name="IGD+"),
        "HV (relative to GT)": Matrix([[0.1], [0.2], [0.3], [0.4]], metric_name="Hypervolume (Rel)"),
    }
    table = namespace["_budget_checkpoint_table"](metrics, (1, 2, 4))
    assert table["Generation"].tolist() == [1, 2, 4]
    assert table["GD+"].tolist() == [4.0, 3.0, 1.0]
    assert table["IGD+"].tolist() == [8.0, 6.0, 2.0]
    assert table["HV (relative to GT)"].tolist() == [0.1, 0.2, 0.4]


def test_optimization_dtlz5_misda_screening_is_truth_independent(monkeypatch):
    path = Path("benchmarks/optimization.ipynb")
    notebook, _ = _read_notebook(path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {"__name__": "dtlz5_screening_smoke"}
    wanted = {"optimization-imports", "optimization-helpers-1", "optimization-helpers-2"}
    for cell in notebook["cells"]:
        if cell.get("id") in wanted:
            exec(compile("".join(cell["source"]), str(path), "exec"), namespace)

    mop = namespace["mb"].mops.DTLZ5(M=10)
    screening = namespace["_screen_misda"](mop, power=5, seed=123)
    diagnostics = namespace["_diagnose_screening"](screening, show_plots=False)
    assert diagnostics["n_screen"] == 32
    assert diagnostics["selected_indices"] == screening["indices"]
    assert diagnostics["selected_support"] in {"SUPPORTED", "UNSUPPORTED"}
    assert screening["selected"].linear is not None
    assert screening["selected"].pareto is not None
