"""Executable checks for the notebook front ends."""

import json
from pathlib import Path

import pytest

from misda.benchmark import BenchmarkResult
from misda.result import MISDAResult


BANNED_SOURCE = (
    "importlib.reload",
    "target_fidelity",
    ".validate(",
    "FeatureAgglomeration",
    "method='adaptive'",
    'method="adaptive"',
)


def _read_notebook(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    return notebook, source


def test_benchmark_notebook_runs_and_displays_each_case(monkeypatch):
    path = Path("examples/benchmark.ipynb")
    notebook, source = _read_notebook(path)

    assert notebook["nbformat"] == 4
    assert all(term not in source for term in BANNED_SOURCE)
    assert "git+https://github.com/monacofj/misda.git@efficient#egg=misda[benchmarks]" in source
    assert 'method="static"' in source
    assert "misda.benchmark(result, truth)" in source
    assert "print(benchmark_result.report())" in source
    assert "result.graph_plot()" in source
    assert "CANONICAL_CASES" in source
    assert "MOP_CASES" in source

    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {"__name__": "notebook_benchmark"}
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

    assert len(namespace["canonical_results"]) == 7
    assert len(namespace["mop_results"]) == 6
    results = (
        list(namespace["canonical_results"].values())
        + list(namespace["mop_results"].values())
    )
    assert all(isinstance(item["result_obj"], MISDAResult) for item in results)
    assert all(
        isinstance(item["benchmark_obj"], BenchmarkResult)
        for item in results
    )
    assert all(
        item["benchmark_obj"].result is item["result_obj"]
        for item in results
    )


def test_comparative_notebook_uses_public_api_and_runs_three_experiments(monkeypatch):
    path = Path("examples/comparative.ipynb")
    notebook, source = _read_notebook(path)

    assert notebook["nbformat"] == 4
    assert all(term not in source for term in BANNED_SOURCE)
    assert "git+https://github.com/monacofj/misda.git@efficient#egg=misda[benchmarks]" in source
    assert 'method="static"' in source
    assert "misda.analyze(" in source
    assert "misda.benchmark(result, truth)" in source
    assert "print(benchmark_result.report())" in source
    assert "result.graph_plot()" in source
    assert "COMPARATIVE_CASES" in source
    assert "run_comparative" not in source

    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {"__name__": "notebook_comparative"}
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
    assert all(isinstance(item["result_obj"], MISDAResult) for item in observed.values())
    assert all(
        isinstance(item["benchmark_obj"], BenchmarkResult)
        for item in observed.values()
    )
    assert len(namespace["comparison"]) == 3
    assert set(namespace["comparison"]["case_id"]) == {"exp_01", "exp_02", "exp_03"}


def test_comparative_notebook_keeps_native_estimands_and_adds_common_score():
    _, source = _read_notebook(Path("examples/comparative.ipynb"))

    assert "mean_eliminated_objective_r2" in source
    assert "worst_eliminated_objective_r2" in source
    assert "global_standardized_r2" in source
    assert "global_standardized_external_r2" in source
    assert "misda_reconstruction" in source
    assert "pca_reconstruction" in source
    assert "misda_minus_pca" in source
