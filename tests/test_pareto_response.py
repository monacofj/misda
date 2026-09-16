import numpy as np
import pytest

from misda.benchmarks.pareto_response import (
    pareto_perturbation_response,
    run_pareto_response_validation,
)


def test_response_is_reproducible_and_sigma_zero_is_identity():
    rng = np.random.default_rng(7)
    Y = rng.normal(size=(24, 4))

    first = pareto_perturbation_response(
        Y, (0.0, 0.10), seed=123, draws=12
    )
    second = pareto_perturbation_response(
        Y, (0.0, 0.10), seed=123, draws=12
    )

    assert first == second
    assert first[0].sigma == 0.0
    assert first[0].mean_jaccard == 1.0
    assert first[0].mc_se == 0.0
    assert first[0].draws == 12
    assert 0.0 <= first[1].mean_jaccard <= 1.0
    assert first[1].mc_se is not None
    assert first[1].mc_se >= 0.0


def test_default_monte_carlo_budget_is_number_of_rows():
    Y = np.column_stack(
        [
            np.linspace(0.0, 1.0, 17),
            np.linspace(1.0, 0.0, 17),
        ]
    )
    point = pareto_perturbation_response(Y, (0.0,), seed=3)[0]
    assert point.draws == len(Y)


def test_response_is_invariant_to_positive_affine_objective_scaling():
    rng = np.random.default_rng(11)
    Y = rng.normal(size=(30, 3))
    transformed = Y * np.array([100.0, 0.01, 7.0]) + np.array([4.0, -8.0, 2.0])

    first = pareto_perturbation_response(
        Y, (0.0, 0.05, 0.20), seed=77, draws=20
    )
    second = pareto_perturbation_response(
        transformed, (0.0, 0.05, 0.20), seed=77, draws=20
    )

    np.testing.assert_allclose(
        [point.mean_jaccard for point in first],
        [point.mean_jaccard for point in second],
    )
    np.testing.assert_allclose(
        [point.mc_se for point in first],
        [point.mc_se for point in second],
    )


def test_distribution_and_correlation_contracts_are_explicit():
    x = np.linspace(0.0, 1.0, 20)
    Y = np.column_stack([x, 1.0 - x, np.ones_like(x)])

    for distribution in ("gaussian", "laplace", "uniform"):
        point = pareto_perturbation_response(
            Y,
            (0.10,),
            seed=9,
            draws=5,
            distribution=distribution,
        )[0]
        assert 0.0 <= point.mean_jaccard <= 1.0

    correlated = pareto_perturbation_response(
        Y,
        (0.10,),
        seed=9,
        draws=5,
        distribution="gaussian",
        gaussian_rho=0.5,
    )[0]
    assert 0.0 <= correlated.mean_jaccard <= 1.0

    with pytest.raises(ValueError, match="only defined for Gaussian"):
        pareto_perturbation_response(
            Y,
            (0.10,),
            seed=9,
            draws=2,
            distribution="laplace",
            gaussian_rho=0.5,
        )


def test_validation_runner_exposes_matching_and_misspecified_models():
    artifact = run_pareto_response_validation(
        n=24,
        sigmas=(0.0, 0.10),
        replicate_seeds=(101,),
        problem_ids=("independence", "monotonic_redundancy"),
        draws=4,
    )

    assert artifact["suite"] == "pareto_perturbation_response"
    assert artifact["parameters"]["draws"] == 4
    assert len(artifact["records"]) == 4
    assert len(artifact["problem_summary"]) == 4
    targets = {row["target"] for row in artifact["comparison_summary"]}
    assert targets == {
        "observed_gaussian_one_draw",
        "laplace_iid",
        "uniform_iid",
        "gaussian_rho_0_5",
        "gaussian_rho_1_0",
    }
    for record in artifact["records"]:
        assert "gaussian_iid" in record
        assert "gaussian_iid_mc_se" in record
        assert "gaussian_rho_1_0" in record
