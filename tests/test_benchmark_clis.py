"""Smoke tests for executable benchmark front ends."""

import json
import importlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from examples.benchmarks._baseline import (
    _linear_press_reconstruction,
    _pareto_preservation,
)


def test_press_reconstruction_matches_explicit_leave_one_out():
    x = np.linspace(-1.5, 2.0, 12)
    frame = pd.DataFrame(
        {
            "f1": x,
            "f2": 1.2 + 2.5 * x + 0.1 * x**2,
            "f3": -0.7 * x + np.sin(x),
        }
    )

    observed = _linear_press_reconstruction(frame, [0])
    expected = {}
    for target in (1, 2):
        predictions = []
        for left_out in range(len(frame)):
            train = np.arange(len(frame)) != left_out
            design = np.column_stack((np.ones(np.sum(train)), x[train]))
            beta, *_ = np.linalg.lstsq(
                design,
                frame.iloc[train, target].to_numpy(),
                rcond=None,
            )
            predictions.append(beta[0] + beta[1] * x[left_out])
        actual = frame.iloc[:, target].to_numpy()
        ss_residual = np.sum((actual - predictions) ** 2)
        ss_total = np.sum((actual - np.mean(actual)) ** 2)
        expected[frame.columns[target]] = 1.0 - ss_residual / ss_total

    assert observed["r2_by_objective"] == pytest.approx(expected)
    assert observed["mean_r2"] == pytest.approx(np.mean(list(expected.values())))
    assert observed["worst_r2"] == pytest.approx(min(expected.values()))


def test_pareto_preservation_uses_distinct_retention_and_validity():
    frame = pd.DataFrame(
        [[0.0, 2.0], [1.0, 1.0], [2.0, 0.0], [1.5, 1.5]],
        columns=["f1", "f2"],
    )

    metrics = _pareto_preservation(frame, [0])

    assert metrics["full_front_size"] == 3
    assert metrics["reduced_front_size"] == 1
    assert metrics["intersection_size"] == 1
    assert metrics["retention"] == pytest.approx(1 / 3)
    assert metrics["validity"] == 1.0
    assert metrics["jaccard"] == pytest.approx(1 / 3)
    assert metrics["exact_preservation"] is False


@pytest.mark.parametrize(
    "module,case_id,suite",
    [
        ("examples.benchmarks.run_benchmark", "case_01", "benchmark"),
        ("examples.benchmarks.run_comparative", "exp_01", "comparative"),
    ],
)
def test_benchmark_cli_writes_normalized_static_json(
    tmp_path, module, case_id, suite
):
    output = tmp_path / f"{suite}.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            "--quick",
            "--case-id",
            case_id,
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["format_version"] == 3
    assert artifact["suite"] == suite
    assert artifact.get("method", artifact.get("methods", [None])[0]) == "static"
    assert artifact["parameters"] == {"n": 64, "seed": 123}
    assert len(artifact["cases"]) == 1

    case = artifact["cases"][0]
    assert case["case_id"] == case_id
    assert case["seed"] == 123
    assert case["n"] == 64
    assert case["m"] == 20
    assert len(case["input_sha256"]) == 64
    assert isinstance(case["estimated"]["latent_dimension"], int)
    assert case["estimated"]["preferred_mis_size"] == len(
        case["preferred_indices"]
    )
    assert sum(case["rank_counts"].values()) == case["n_mis"]
    assert set(case["graphs"]["dependence"]) == {
        "nodes",
        "edges",
        "components",
    }
    assert case["separation_status"] in {
        "NULL_SEPARATION",
        "NO_NULL_SEPARATION",
    }
    assert set(case["linear_reconstruction"]) == {
        "r2_by_objective",
        "r2_reason_by_objective",
        "mean_r2",
        "worst_r2",
        "reason_by_metric",
        "jackknife",
    }
    assert set(case["pareto_preservation"]) == {
        "pareto_retention",
        "pareto_validity",
        "pareto_jaccard",
        "full_front_size",
        "reduced_front_size",
        "intersection_size",
        "union_size",
        "exact_preservation",
        "reduced_front_indices",
    }
    assert case["assessment"]["case_id"] == case_id
    assert case["assessment"]["status"] in {
        "PASS",
        "EXPECTED_CHANGE",
        "REGRESSION",
    }
    if suite == "comparative":
        assert artifact["methods"] == ["static", "pca"]
        assert case["pca"]["metric"] == "global_standardized_r2"
        assert len(case["pca"]["curve"]) == 10


def test_unknown_case_id_is_rejected(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.benchmarks.run_benchmark",
            "--quick",
            "--case-id",
            "not_a_case",
            "--output",
            str(tmp_path / "unused.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "Unknown case id(s): not_a_case" in completed.stderr


def test_cli_never_passes_case_declarations_into_analyze(monkeypatch):
    module = importlib.import_module("examples.benchmarks.run_benchmark")
    original = module.misda.analyze
    observed_kwargs = []

    def capture(data, **kwargs):
        observed_kwargs.append(dict(kwargs))
        return original(data, **kwargs)

    monkeypatch.setattr(module.misda, "analyze", capture)
    artifact = module.run_benchmark(n=32, case_ids={"case_02"})

    assert len(artifact["cases"]) == 1
    assert observed_kwargs == [
        {
            "method": "static",
            "name": "Case 2 - Total redundancy",
            "seed": 123,
            "max_evaluated_mis": 1,
        }
    ]
