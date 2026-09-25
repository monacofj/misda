"""Contracts for the Pareto-retention resampling robustness probe."""

import pandas as pd

from benchmarks import run_ranking_policy_robustness as probe


def test_ranking_policy_robustness_smoke_uses_independent_sample_seeds():
    results = probe.run_robustness(
        sample_seeds=(101, 202),
        screen_power=5,
        misda_seed=123,
    )
    summary = probe.summarize_robustness(results)

    assert len(results) == 8
    assert set(results["problem"]) == {"DTLZ2", "DTLZ5", "DPF1", "SAFE_PATH"}
    assert results.groupby("problem")["sample_seed"].nunique().eq(2).all()
    assert set(results["misda_seed"]) == {123}
    assert set(results["n_screen"]) == {32}

    assert len(summary) == 4
    assert set(summary["problem"]) == {"DTLZ2", "DTLZ5", "DPF1", "SAFE_PATH"}
    assert summary["replicates"].eq(2).all()

    dtlz5 = summary.loc[summary["problem"] == "DTLZ5"].iloc[0]
    dpf1 = summary.loc[summary["problem"] == "DPF1"].iloc[0]
    assert "safe_present_rate" in dtlz5.index
    assert "safe_pareto_top_given_present_rate" in dtlz5.index
    assert "safe_base_present_rate" in dpf1.index
    assert "safe_base_pareto_top_given_present_rate" in dpf1.index
    safe_path = summary.loc[summary["problem"] == "SAFE_PATH"].iloc[0]
    assert "safe_small_global_spurious_top_given_present_rate" in safe_path.index
    assert "safe_large_global_spurious_top_given_present_rate" in safe_path.index


def test_summary_separates_discovery_from_conditional_ranking_success():
    results = pd.DataFrame(
        [
            {
                "problem": "DTLZ5",
                "n_screen": 512,
                "n_mis": 9,
                "candidate_size_min": 2,
                "candidate_size_max": 2,
                "global_distortion_selected_size": 2,
                "global_distortion_selected_is_max_size": True,
                "pareto_selected_retention": 0.20,
                "safe_present": True,
                "safe_size_span_top": False,
                "safe_pareto_top": True,
                "safe_global_spurious_top": True,
                "safe_front_spurious_top": True,
                "safe_retention": 0.20,
                "safe_global_spurious_rate": 0.02,
                "safe_front_spurious_rate": 0.10,
                "unsafe_witness_present": True,
                "unsafe_witness_size_span_top": True,
                "unsafe_witness_pareto_top": False,
                "unsafe_witness_global_spurious_top": False,
                "unsafe_witness_front_spurious_top": False,
                "unsafe_witness_retention": 0.10,
                "unsafe_witness_global_spurious_rate": 0.08,
                "unsafe_witness_front_spurious_rate": 0.30,
            },
            {
                "problem": "DTLZ5",
                "n_screen": 512,
                "n_mis": 8,
                "candidate_size_min": 2,
                "candidate_size_max": 2,
                "global_distortion_selected_size": 2,
                "global_distortion_selected_is_max_size": True,
                "pareto_selected_retention": 0.18,
                "safe_present": False,
                "safe_size_span_top": False,
                "safe_pareto_top": False,
                "safe_global_spurious_top": False,
                "safe_front_spurious_top": False,
                "safe_retention": None,
                "safe_global_spurious_rate": None,
                "safe_front_spurious_rate": None,
                "unsafe_witness_present": True,
                "unsafe_witness_size_span_top": True,
                "unsafe_witness_pareto_top": False,
                "unsafe_witness_retention": 0.09,
            },
        ]
    )

    summary = probe.summarize_robustness(results).iloc[0]

    assert summary["safe_present_rate"] == 0.5
    assert summary["safe_pareto_top_rate"] == 0.5
    assert summary["safe_pareto_top_given_present_rate"] == 1.0
    assert summary["safe_global_spurious_top_given_present_rate"] == 1.0
    assert summary["safe_front_spurious_top_given_present_rate"] == 1.0
