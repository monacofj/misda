"""Ground-truth checks for sampled clean diagnostic objectives."""

import numpy as np

from misda.benchmarks import (
    PROBLEM_BY_ID,
    PROBLEMS,
    diagnostic_truth,
    sampled_pareto_indices,
)


def test_pareto_truth_is_available_for_every_clean_diagnostic_problem():
    for problem in PROBLEMS:
        dataset = problem.generate(N=48, seed=123, sigma=0.0)
        truth = diagnostic_truth(problem, dataset.Z)
        assert truth["pareto_expected"]
        assert all(0 <= index < 48 for index in truth["pareto_expected"])


def test_generating_families_are_declared_for_every_diagnostic_problem():
    for problem in PROBLEMS:
        dataset = problem.generate(N=24, seed=123, sigma=0.0)
        truth = diagnostic_truth(problem, dataset.Z)
        assert [len(group) for group in truth["families_expected"]] == list(
            problem.scenario.family_sizes
        )


def test_case5_family_and_structural_units_are_distinct_declarations():
    problem = PROBLEM_BY_ID["transitive_chain"]
    dataset = problem.generate(N=32, seed=123, sigma=0.0)
    truth = diagnostic_truth(problem, dataset.Z)

    assert [len(group) for group in truth["families_expected"]] == [20]
    assert [len(group) for group in truth["blocks_expected"]] == [1] * 20
    assert truth["expected_mismatches"]["selected_structural_units"] == "TRANSITIVE_CHAINING"


def test_mop_b_families_do_not_become_structural_units():
    problem = PROBLEM_BY_ID["tradeoff_redundancies"]
    dataset = problem.generate(N=32, seed=123, sigma=0.0)
    truth = diagnostic_truth(problem, dataset.Z)

    assert [len(group) for group in truth["families_expected"]] == [7, 7, 6]
    assert "blocks_expected" not in truth


def test_pareto_truth_does_not_change_when_observation_noise_changes():
    problem = PROBLEM_BY_ID["monotonic_redundancy"]
    clean = problem.generate(N=64, seed=123, sigma=0.0, observation_seed=1)
    noisy = problem.generate(N=64, seed=123, sigma=0.25, observation_seed=999)

    np.testing.assert_array_equal(clean.Z.to_numpy(), noisy.Z.to_numpy())
    assert diagnostic_truth(problem, clean.Z)["pareto_expected"] == diagnostic_truth(
        problem, noisy.Z
    )["pareto_expected"]
    assert not np.array_equal(clean.Y.to_numpy(), noisy.Y.to_numpy())


def test_mop_a_clean_truth_is_argmin_x_even_when_y_is_noisy():
    problem = PROBLEM_BY_ID["monotonic_redundancy"]
    dataset = problem.generate(N=80, seed=123, sigma=0.20, observation_seed=456)
    expected = [int(np.argmin(dataset.X["x"].to_numpy()))]

    assert diagnostic_truth(problem, dataset.Z)["pareto_expected"] == expected


def test_mop_d_clean_truth_contains_every_sample_even_when_y_is_noisy():
    problem = PROBLEM_BY_ID["antagonistic_nonlinear_groups"]
    dataset = problem.generate(N=80, seed=123, sigma=0.20, observation_seed=456)

    assert diagnostic_truth(problem, dataset.Z)["pareto_expected"] == list(range(80))


def test_duplicate_nondominated_rows_preserve_all_sample_indices():
    Z = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 2.0],
        ]
    )

    assert sampled_pareto_indices(Z) == [0, 1, 2]


def test_pareto_truth_is_based_on_z_not_observed_y():
    problem = PROBLEM_BY_ID["monotonic_redundancy"]
    dataset = problem.generate(N=96, seed=321, sigma=1.0, observation_seed=777)
    clean_truth = diagnostic_truth(problem, dataset.Z)["pareto_expected"]
    observed_front = sampled_pareto_indices(dataset.Y)

    # Large observation noise can alter the observed front; it must not alter truth.
    assert clean_truth == [int(np.argmin(dataset.X["x"].to_numpy()))]
    assert diagnostic_truth(problem, dataset.Z)["pareto_expected"] == clean_truth
    assert isinstance(observed_front, list)
