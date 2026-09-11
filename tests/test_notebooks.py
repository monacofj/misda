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


def test_diagnostic_clean_notebook_runs_unified_suite(monkeypatch):
    path = Path("examples/diagnostic_clean.ipynb")
    notebook, source = _read_notebook(path)

    assert notebook["nbformat"] == 4
    assert all(term not in source for term in BANNED_SOURCE)
    assert "bench.PROBLEMS" in source
    assert "problem.generate(N=N, seed=SEED, sigma=SIGMA)" in source
    assert "bench.diagnostic_truth(problem, dataset.Z)" in source
    assert "dataset.Y" in source
    assert "misda.discover(" in source
    assert 'misda.evaluate(mis_set, metrics=("linear", "pareto"))' in source
    assert "misda.benchmark(mis_set, truth)" in source

    _, namespace = _execute_notebook(
        path,
        monkeypatch,
        overrides={"benchmark-run": {"N": 64}},
    )
    results = namespace["diagnostic_results"]
    assert len(results) == 13
    assert all(isinstance(item["result_obj"], misda.MISSet) for item in results.values())
    assert all(isinstance(item["benchmark_obj"], BenchmarkResult) for item in results.values())
    assert all(item["dataset"].sigma == 0.0 for item in results.values())
    assert all(item["dataset"].Y.equals(item["dataset"].Z) for item in results.values())


def test_diagnostic_noisy_notebook_runs_unified_suite(monkeypatch):
    path = Path("examples/diagnostic_noisy.ipynb")
    notebook, source = _read_notebook(path)

    assert notebook["nbformat"] == 4
    assert all(term not in source for term in BANNED_SOURCE)
    assert "bench.PROBLEMS" in source
    assert "OBSERVATION_SEED = 456" in source
    assert "SIGMA = 0.10" in source
    assert "observation_seed=OBSERVATION_SEED" in source
    assert "bench.diagnostic_truth(problem, dataset.Z)" in source

    _, namespace = _execute_notebook(
        path,
        monkeypatch,
        overrides={"benchmark-run": {"N": 64}},
    )
    results = namespace["noisy_results"]
    assert len(results) == 13
    assert all(item["dataset"].sigma == pytest.approx(0.10) for item in results.values())
    assert all(item["dataset"].sample_seed == 123 for item in results.values())
    assert all(item["dataset"].observation_seed == 456 for item in results.values())
    assert all(not item["dataset"].Y.equals(item["dataset"].Z) for item in results.values())


def test_diagnostic_robustness_notebook_runs_lightweight_controlled_sweep(monkeypatch):
    path = Path("examples/diagnostic_robustness.ipynb")
    notebook, source = _read_notebook(path)

    assert notebook["nbformat"] == 4
    assert all(term not in source for term in BANNED_SOURCE)
    assert "PROBLEM_IDS" in source
    assert "SIGMAS = (0.00, 0.05, 0.10, 0.20, 0.40)" in source
    assert "REPLICATE_SEEDS = (101, 202, 303, 404, 505)" in source
    assert "problem.observe(Z, sigma=sigma, standard_noise=epsilon)" in source
    assert 'misda.evaluate(mis_set, metrics=("pareto",), candidates=1)' in source
    assert "transitive_chaining_rate" in source

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
    assert robustness["pareto_jaccard"].notna().all()


def test_comparison_notebook_uses_diagnostic_truth_without_pca_dimension_cutoff(monkeypatch):
    path = Path("examples/comparison.ipynb")
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
