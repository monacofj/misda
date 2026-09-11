"""Executable checks for the notebook front ends."""

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


def _read_notebook(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    return notebook, source


def test_diagnostic_clean_notebook_runs_unified_suite(monkeypatch):
    path = Path("examples/diagnostic_clean.ipynb")
    notebook, source = _read_notebook(path)

    assert notebook["nbformat"] == 4
    assert all(term not in source for term in BANNED_SOURCE)
    assert "git+https://github.com/monacofj/misda.git@main#egg=misda[benchmarks]" in source
    assert "bench.PROBLEMS" in source
    assert "problem.generate(N=N, seed=SEED, sigma=SIGMA)" in source
    assert "bench.diagnostic_truth(problem, dataset.Z)" in source
    assert "dataset.Y" in source
    assert "CANONICAL_CASES" not in source
    assert "MOP_CASES" not in source
    assert "misda.discover(" in source
    assert 'misda.evaluate(mis_set, metrics=("linear", "pareto"))' in source
    assert "misda.benchmark(mis_set, truth)" in source
    assert "print(benchmark_result.report())" in source
    assert "mis_set.graph_plot()" in source

    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {"__name__": "notebook_diagnostic_clean"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tags = cell.get("metadata", {}).get("tags", [])
        if "setup" in tags:
            continue
        if "benchmark-run" in tags:
            namespace["N"] = 64
        cell_source = "".join(cell.get("source", []))
        exec(compile(cell_source, f"{path}:cell-{index}", "exec"), namespace)

    results = namespace["diagnostic_results"]
    assert len(results) == 13
    assert all(isinstance(item["result_obj"], misda.MISSet) for item in results.values())
    assert all(
        isinstance(item["benchmark_obj"], BenchmarkResult)
        for item in results.values()
    )
    assert all(
        item["benchmark_obj"].result is item["result_obj"]
        for item in results.values()
    )
    assert all(item["dataset"].sigma == 0.0 for item in results.values())
    assert all(
        item["dataset"].Y.equals(item["dataset"].Z) for item in results.values()
    )


def test_diagnostic_noisy_notebook_runs_unified_suite(monkeypatch):
    path = Path("examples/diagnostic_noisy.ipynb")
    notebook, source = _read_notebook(path)

    assert notebook["nbformat"] == 4
    assert all(term not in source for term in BANNED_SOURCE)
    assert "git+https://github.com/monacofj/misda.git@main#egg=misda[benchmarks]" in source
    assert "bench.PROBLEMS" in source
    assert "OBSERVATION_SEED = 456" in source
    assert "SIGMA = 0.10" in source
    assert "sigma=SIGMA" in source
    assert "observation_seed=OBSERVATION_SEED" in source
    assert "bench.diagnostic_truth(problem, dataset.Z)" in source
    assert "dataset.Y" in source
    assert "misda.discover(" in source
    assert 'misda.evaluate(mis_set, metrics=("linear", "pareto"))' in source
    assert "misda.benchmark(mis_set, truth)" in source
    assert "print(benchmark_result.report())" in source
    assert "mis_set.graph_plot()" in source

    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {"__name__": "notebook_diagnostic_noisy"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tags = cell.get("metadata", {}).get("tags", [])
        if "setup" in tags:
            continue
        if "benchmark-run" in tags:
            namespace["N"] = 64
        cell_source = "".join(cell.get("source", []))
        exec(compile(cell_source, f"{path}:cell-{index}", "exec"), namespace)

    results = namespace["noisy_results"]
    assert len(results) == 13
    assert all(isinstance(item["result_obj"], misda.MISSet) for item in results.values())
    assert all(
        isinstance(item["benchmark_obj"], BenchmarkResult)
        for item in results.values()
    )
    assert all(
        item["benchmark_obj"].result is item["result_obj"]
        for item in results.values()
    )
    assert all(item["dataset"].sigma == pytest.approx(0.10) for item in results.values())
    assert all(item["dataset"].sample_seed == 123 for item in results.values())
    assert all(item["dataset"].observation_seed == 456 for item in results.values())
    assert all(
        not item["dataset"].Y.equals(item["dataset"].Z)
        for item in results.values()
    )
    bench = namespace["bench"]
    by_id = {problem.id: problem for problem in bench.PROBLEMS}
    for problem_id, item in results.items():
        assert item["truth"] == bench.diagnostic_truth(
            by_id[problem_id],
            item["dataset"].Z,
        )


def test_diagnostic_robustness_notebook_runs_lightweight_controlled_sweep(monkeypatch):
    path = Path("examples/diagnostic_robustness.ipynb")
    notebook, source = _read_notebook(path)

    assert notebook["nbformat"] == 4
    assert all(term not in source for term in BANNED_SOURCE)
    assert "PROBLEM_IDS" in source
    assert "SIGMAS = (0.00, 0.05, 0.10, 0.20, 0.40)" in source
    assert "REPLICATE_SEEDS = (101, 202, 303, 404, 505)" in source
    assert "problem.observe(Z, sigma=sigma, standard_noise=epsilon)" in source
    assert "epsilon = np.random.default_rng(observation_sequence).normal(size=Z.shape)" in source
    assert 'misda.evaluate(mis_set, metrics=("pareto",), candidates=1)' in source
    assert "bench.diagnostic_truth(problem, Z)" in source
    assert "transitive_chaining_rate" in source

    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {"__name__": "notebook_diagnostic_robustness"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tags = cell.get("metadata", {}).get("tags", [])
        if "setup" in tags or "visualization" in tags:
            continue
        if "robustness-run" in tags:
            namespace["N"] = 48
            namespace["PROBLEM_IDS"] = ("independence", "total_redundancy")
            namespace["SIGMAS"] = (0.0, 0.10)
            namespace["REPLICATE_SEEDS"] = (101,)
        cell_source = "".join(cell.get("source", []))
        exec(compile(cell_source, f"{path}:cell-{index}", "exec"), namespace)

    robustness = namespace["robustness"]
    summary = namespace["robustness_summary"]
    assert len(robustness) == 4
    assert len(summary) == 4
    assert set(robustness["problem_id"]) == {"independence", "total_redundancy"}
    assert set(robustness["sigma"]) == {0.0, 0.10}
    assert robustness.groupby("problem_id")["sample_seed"].nunique().eq(1).all()
    assert robustness["pareto_jaccard"].notna().all()
    assert summary["replicates"].eq(1).all()


def test_comparison_notebook_uses_public_api_and_runs_three_experiments(monkeypatch):
    path = Path("examples/comparison.ipynb")
    notebook, source = _read_notebook(path)

    assert notebook["nbformat"] == 4
    assert all(term not in source for term in BANNED_SOURCE)
    assert "git+https://github.com/monacofj/misda.git@main#egg=misda[benchmarks]" in source
    assert "misda.discover(" in source
    assert "misda.rank(mis_set)" in source
    assert "misda.evaluate(" in source
    assert "misda.benchmark(mis_set, truth)" in source
    assert "print(benchmark_result.report())" in source
    assert "mis_set.graph_plot(ranking=structural)" in source
    assert "COMPARATIVE_CASES" in source
    assert "run_comparative" not in source

    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {"__name__": "notebook_comparison"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tags = cell.get("metadata", {}).get("tags", [])
        if "setup" in tags:
            continue
        if "comparative-run" in tags:
            namespace["N"] = 32
        cell_source = "".join(cell.get("source", []))
        exec(compile(cell_source, f"{path}:cell-{index}", "exec"), namespace)

    observed = namespace["comparative_results"]
    assert len(observed) == 3
    assert all(isinstance(item["result_obj"], misda.MISSet) for item in observed.values())
    assert all(
        isinstance(item["benchmark_obj"], BenchmarkResult)
        for item in observed.values()
    )
    assert len(namespace["comparison"]) == 3
    assert set(namespace["comparison"]["case_id"]) == {"exp_01", "exp_02", "exp_03"}


def test_comparison_notebook_keeps_native_estimands_and_adds_common_score():
    _, source = _read_notebook(Path("examples/comparison.ipynb"))

    assert "mean_eliminated_objective_r2" in source
    assert "worst_eliminated_objective_r2" in source
    assert "global_standardized_r2" in source
    assert "global_standardized_external_r2" in source
    assert "misda_reconstruction" in source
    assert "pca_reconstruction" in source
    assert "misda_minus_pca" in source
