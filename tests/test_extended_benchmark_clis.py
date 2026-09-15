"""Smoke tests for extended reproducible benchmark-validation front ends."""

import json
import subprocess
import sys

import pytest

from misda.benchmarks.validation import (
    CONTROLLED_PROBLEM_IDS,
    PRESENTATION_CASE,
    SERIALIZATION_CASE_ID,
)
from benchmarks.run_controlled import BENCHMARK_CASES


def _run_cli(tmp_path, module, *args):
    output = tmp_path / (module.rsplit(".", 1)[-1] + ".json")
    completed = subprocess.run(
        [sys.executable, "-m", module, *args, "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(output.read_text(encoding="utf-8"))


def test_validation_case_ids_match_controlled_runner_contract():
    observed = {problem.id: case_id for case_id, problem in BENCHMARK_CASES}
    assert SERIALIZATION_CASE_ID == observed
    assert tuple(sorted(PRESENTATION_CASE, key=PRESENTATION_CASE.get)) == CONTROLLED_PROBLEM_IDS


def test_controlled_noisy_cli_writes_reference_artifact(tmp_path):
    artifact = _run_cli(
        tmp_path,
        "benchmarks.run_controlled_noisy",
        "--quick",
        "--problem-id",
        "independence",
    )
    assert artifact["suite"] == "controlled_noisy"
    assert artifact["parameters"] == {
        "n": 64,
        "observation_seed": 456,
        "seed": 123,
        "sigma": pytest.approx(0.10),
    }
    assert len(artifact["cases"]) == 1
    case = artifact["cases"][0]
    assert case["problem_id"] == "independence"
    assert case["presentation_case"] == 1
    assert case["observation"] == {"observation_seed": 456, "sigma": pytest.approx(0.10)}
    assert len(case["clean_input_sha256"]) == 64
    assert "pareto_decomposition" in case


def test_sampling_robustness_cli_quick_shape(tmp_path):
    artifact = _run_cli(tmp_path, "benchmarks.run_sampling_robustness", "--quick")
    assert artifact["suite"] == "sampling_robustness"
    assert artifact["parameters"]["n"] == 48
    assert artifact["parameters"]["replicate_seeds"] == [1001]
    assert len(artifact["records"]) == 2
    assert len(artifact["summary"]) == 2
    assert {row["problem_id"] for row in artifact["summary"]} == {
        "independence",
        "total_redundancy",
    }


def test_noisy_robustness_cli_quick_shape(tmp_path):
    artifact = _run_cli(tmp_path, "benchmarks.run_noisy_robustness", "--quick")
    assert artifact["suite"] == "noisy_robustness"
    assert artifact["parameters"]["n"] == 48
    assert artifact["parameters"]["sigmas"] == [0.0, 0.10]
    assert artifact["parameters"]["replicate_seeds"] == [101]
    assert len(artifact["records"]) == 4
    assert len(artifact["summary"]) == 4
    assert all("pareto_observation_jaccard" in row for row in artifact["summary"])
    assert all("pareto_reduction_jaccard" in row for row in artifact["summary"])
    assert all("pareto_end_to_end_jaccard" in row for row in artifact["summary"])
