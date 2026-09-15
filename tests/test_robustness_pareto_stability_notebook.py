"""Regression checks for Pareto-stability evidence in the robustness notebook."""

import json
from pathlib import Path


def test_robustness_notebook_tracks_observed_pareto_stability_without_thresholds():
    notebook = json.loads(Path("examples/diagnostic_robustness.ipynb").read_text())
    source = "\n".join(
        line
        for cell in notebook["cells"]
        for line in cell.get("source", [])
    )

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

    assert "mis_set.pareto_stability" in source
    assert "epsilon_for_candidate(selected_index)" in source
    assert "does **not** define a pass/fail threshold" in source
    assert "pass/fail cutoff" in source
