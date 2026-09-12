"""Tests for the explicit X -> Z -> Y diagnostic problem architecture."""

import numpy as np
import pandas as pd
import pytest

from misda.benchmarks import DIAGNOSTIC_SCENARIOS, PROBLEM_BY_ID, PROBLEMS


def test_problem_catalogue_matches_scenario_catalogue():
    assert len(PROBLEMS) == 13
    assert tuple(problem.id for problem in PROBLEMS) == tuple(
        scenario.id for scenario in DIAGNOSTIC_SCENARIOS
    )
    assert set(PROBLEM_BY_ID) == {scenario.id for scenario in DIAGNOSTIC_SCENARIOS}


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_clean_generation_is_explicit_identity_observation(problem):
    dataset = problem.generate(N=37, seed=321, sigma=0.0, observation_seed=999)

    assert isinstance(dataset.X, pd.DataFrame)
    assert isinstance(dataset.Z, pd.DataFrame)
    assert isinstance(dataset.Y, pd.DataFrame)
    assert len(dataset.X) == 37
    assert dataset.Z.shape == (37, 20)
    assert dataset.Y.shape == (37, 20)
    np.testing.assert_array_equal(dataset.Y.to_numpy(), dataset.Z.to_numpy())
    assert dataset.truth["problem_id"] == problem.id
    assert dataset.truth["latent_expected"] == problem.scenario.latent_expected
    assert dataset.truth["structural_expected"] == problem.scenario.structural_expected
    assert [len(group) for group in dataset.truth["families_expected"]] == list(
        problem.scenario.family_sizes
    )
    if problem.scenario.structural_unit_sizes is None:
        assert "blocks_expected" not in dataset.truth
    else:
        assert [len(group) for group in dataset.truth["blocks_expected"]] == list(
            problem.scenario.structural_unit_sizes
        )


@pytest.mark.parametrize("problem", PROBLEMS, ids=lambda p: p.id)
def test_observation_rng_is_independent_of_problem_sampling(problem):
    clean = problem.generate(N=41, seed=101, sigma=0.0, observation_seed=1)
    noisy_a = problem.generate(N=41, seed=101, sigma=0.1, observation_seed=2)
    noisy_b = problem.generate(N=41, seed=101, sigma=0.1, observation_seed=3)

    np.testing.assert_array_equal(clean.X.to_numpy(), noisy_a.X.to_numpy())
    np.testing.assert_array_equal(clean.X.to_numpy(), noisy_b.X.to_numpy())
    np.testing.assert_array_equal(clean.Z.to_numpy(), noisy_a.Z.to_numpy())
    np.testing.assert_array_equal(clean.Z.to_numpy(), noisy_b.Z.to_numpy())
    assert not np.array_equal(noisy_a.Y.to_numpy(), noisy_b.Y.to_numpy())


def test_observer_scales_same_noise_realization_across_sigma():
    problem = PROBLEM_BY_ID["independence"]
    X = problem.sample(N=64, seed=123)
    Z = problem.evaluate(X)
    epsilon = np.random.default_rng(44).normal(size=Z.shape)

    y_small = problem.observe(Z, sigma=0.05, standard_noise=epsilon)
    y_large = problem.observe(Z, sigma=0.10, standard_noise=epsilon)

    np.testing.assert_allclose(
        y_large.to_numpy() - Z.to_numpy(),
        2.0 * (y_small.to_numpy() - Z.to_numpy()),
        rtol=1e-12,
        atol=1e-12,
    )


def test_case5_clean_map_uses_innovations_as_generating_variables():
    problem = PROBLEM_BY_ID["transitive_chain"]
    dataset = problem.generate(N=53, seed=123, sigma=0.0)
    X = dataset.X.to_numpy()
    Z = dataset.Z.to_numpy()

    np.testing.assert_allclose(Z[:, 0], X[:, 0])
    for j in range(1, 20):
        np.testing.assert_allclose(Z[:, j] - Z[:, j - 1], 0.2 * X[:, j])


def test_case7_clean_problem_separates_conflict_from_noise():
    problem = PROBLEM_BY_ID["antagonistic_linear_groups"]
    dataset = problem.generate(N=29, seed=777, sigma=0.0)
    x = dataset.X["x"].to_numpy()
    Z = dataset.Z.to_numpy()

    np.testing.assert_array_equal(Z[:, :10], np.column_stack([x] * 10))
    np.testing.assert_array_equal(Z[:, 10:], np.column_stack([-x] * 10))


def test_mop_f_clean_problem_contains_no_embedded_replica_noise():
    problem = PROBLEM_BY_ID["regime_switching"]
    dataset = problem.generate(N=71, seed=123, sigma=0.0)
    Z = dataset.Z

    np.testing.assert_array_equal(Z["f1"].to_numpy(), Z["f10"].to_numpy())
    np.testing.assert_array_equal(Z["f11"].to_numpy(), Z["f20"].to_numpy())


def test_sigma_must_be_nonnegative():
    problem = PROBLEM_BY_ID["independence"]
    Z = problem.evaluate(problem.sample(N=5, seed=1))
    with pytest.raises(ValueError, match="non-negative"):
        problem.observe(Z, sigma=-0.1)
