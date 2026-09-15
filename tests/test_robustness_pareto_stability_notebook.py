"""Regression checks for Pareto-stability evidence in noisy robustness validation."""

import json
from pathlib import Path


def _notebook_source(path):
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    return "\n".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )


def test_noisy_robustness_frontend_exposes_pareto_decomposition_without_thresholds():
    notebook_source = _notebook_source("benchmarks/noisy_robustness.ipynb")
    runner_source = Path("misda/benchmarks/validation.py").read_text(encoding="utf-8")
    source = notebook_source + "\n" + runner_source

    required = (
        "pareto_observation_jaccard",
        "pareto_reduction_jaccard",
        "pareto_end_to_end_jaccard",
        "pareto_observed_fraction",
        "pareto_additive_epsilon",
        "pareto_dominance_margin_min",
        "pareto_dominance_margin_median",
        "pareto_dominance_margin_max",
    )
    for field in required:
        assert field in source

    assert "pareto_stability" in runner_source
    assert "epsilon_for_candidate(selected_index)" in runner_source
    assert "does **not** define a pass/fail threshold" in notebook_source
    assert "no fixed pass/fail cutoff" in notebook_source
