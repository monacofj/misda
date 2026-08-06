"""Executable checks for the notebook front ends."""

import json
from pathlib import Path

import pytest


NOTEBOOKS = (
    (Path("examples/benchmark.ipynb"), "benchmark", 13),
    (Path("examples/comparative.ipynb"), "comparative", 3),
)

BANNED_SOURCE = (
    "pip install",
    "importlib.reload",
    "target_fidelity",
    ".validate(",
    "FeatureAgglomeration",
    "method='adaptive'",
    'method="adaptive"',
)


@pytest.mark.parametrize("path,suite,expected_cases", NOTEBOOKS)
def test_notebook_is_thin_static_front_end(path, suite, expected_cases):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert notebook["nbformat"] == 4
    assert all(term not in source for term in BANNED_SOURCE)
    assert f"run_{suite}" in source
    assert "method='static'" not in source
    assert "method=\"static\"" not in source
    assert sum(cell["cell_type"] == "code" for cell in notebook["cells"]) <= 4

    namespace = {"__name__": f"notebook_{suite}"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        cell_source = "".join(cell.get("source", []))
        exec(compile(cell_source, f"{path}:cell-{index}", "exec"), namespace)

    artifact = namespace["artifact"]
    assert artifact["suite"] == suite
    assert len(artifact["cases"]) == expected_cases
    assert artifact.get("method", artifact.get("methods", [None])[0]) == "static"


def test_comparative_notebook_keeps_estimands_separate():
    notebook = json.loads(
        Path("examples/comparative.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "mean_eliminated_objective_r2" in source
    assert "worst_eliminated_objective_r2" in source
    assert "global_standardized_r2" in source
    assert "misda_reconstruction" in source
    assert "pca_reconstruction" in source
