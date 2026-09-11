"""Tests for reproducible classical DTLZ reference generators and runner."""

import inspect
import json
import subprocess
import sys

import numpy as np
import pytest

from misda.benchmarks import CLASSICAL_MOPS, generate_dtlz2, generate_dtlz5


@pytest.mark.parametrize("generator", (generate_dtlz2, generate_dtlz5))
def test_classical_dtlz_generators_are_seeded_and_reproducible(generator):
    assert inspect.signature(generator).parameters["seed"].default == 123

    first_f, first_x = generator(N=64, M=5, n_vars=14, seed=991)
    repeated_f, repeated_x = generator(N=64, M=5, n_vars=14, seed=991)
    other_f, other_x = generator(N=64, M=5, n_vars=14, seed=992)

    np.testing.assert_array_equal(first_f, repeated_f)
    np.testing.assert_array_equal(first_x, repeated_x)
    assert not np.array_equal(first_x, other_x)
    assert not np.array_equal(first_f, other_f)


@pytest.mark.parametrize("generator", (generate_dtlz2, generate_dtlz5))
def test_classical_dtlz_on_front_fixes_distance_variables_and_unit_norm(generator):
    F, X = generator(N=73, M=5, n_vars=14, on_front=True, seed=123)

    assert F.shape == (73, 5)
    assert X.shape == (73, 14)
    np.testing.assert_array_equal(X[:, 4:], np.full((73, 10), 0.5))
    np.testing.assert_allclose(np.linalg.norm(F, axis=1), 1.0, rtol=1e-12, atol=1e-12)
    assert np.isfinite(F).all()
    assert np.all(F >= 0.0)


def test_dtlz5_front_is_one_dimensional_under_fixed_first_coordinate():
    F, X = generate_dtlz5(N=51, M=6, n_vars=15, on_front=True, seed=123)

    order = np.argsort(X[:, 0])
    ordered = F[order]
    assert np.all(np.diff(ordered[:, -1]) >= 0.0)
    assert np.all(np.diff(ordered[:, 0]) <= 0.0)


def test_classical_catalogue_declares_geometry_without_misda_dimension_truth():
    assert set(CLASSICAL_MOPS) == {"dtlz2", "dtlz5"}
    assert CLASSICAL_MOPS["dtlz2"]["pareto_manifold_dimension"](10) == 9
    assert CLASSICAL_MOPS["dtlz5"]["pareto_manifold_dimension"](10) == 1
    for item in CLASSICAL_MOPS.values():
        assert "latent_expected" not in item
        assert "structural_expected" not in item
        assert item["pareto_geometry"]


@pytest.mark.parametrize("generator", (generate_dtlz2, generate_dtlz5))
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"N": 0}, "N must be at least 1"),
        ({"M": 1}, "M must be at least 2"),
        ({"M": 6, "n_vars": 5}, "n_vars must be at least M"),
    ],
)
def test_classical_dtlz_parameter_validation(generator, kwargs, match):
    with pytest.raises(ValueError, match=match):
        generator(**kwargs)


def test_classical_runner_writes_reference_artifact_without_dimension_truth(tmp_path):
    output = tmp_path / "classical.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.benchmarks.run_classical_mops",
            "--quick",
            "--problem-id",
            "dtlz5",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["suite"] == "classical_mops"
    assert artifact["parameters"] == {
        "n": 64,
        "m": 5,
        "n_vars": 14,
        "seed": 123,
        "on_front": True,
    }
    assert len(artifact["cases"]) == 1
    case = artifact["cases"][0]
    assert case["case_id"] == "dtlz5"
    assert case["declared"] == {
        "latent_dimension": None,
        "structural_dimension": None,
    }
    assert case["assessment"]["status"] == "NO_DECLARATION"
    assert case["reference_geometry"]["pareto_manifold_dimension"] == 1
    assert case["reference_geometry"]["misda_dimension_truth"] is None
    assert case["sample_on_pareto_front"] is True


def test_unknown_classical_problem_id_is_rejected(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.benchmarks.run_classical_mops",
            "--quick",
            "--problem-id",
            "not_a_problem",
            "--output",
            str(tmp_path / "unused.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "Unknown classical problem id(s): not_a_problem" in completed.stderr
