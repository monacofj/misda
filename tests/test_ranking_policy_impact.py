"""Contracts for the pre-default ranking policy impact audit."""

from __future__ import annotations

import json
import subprocess
import sys

import misda
from benchmarks import run_ranking_policy_aggressiveness_probe as aggressiveness_probe
from benchmarks import run_ranking_policy_impact as impact
from misda.benchmarks import PROBLEM_BY_ID, diagnostic_truth


def test_audit_compares_rankings_without_changing_default():
    problem = PROBLEM_BY_ID["total_redundancy"]
    dataset = problem.generate(N=32, seed=123, sigma=0.0)
    truth = diagnostic_truth(problem, dataset.Z)

    observed = impact._audit_case(
        suite="controlled",
        problem_id="total_redundancy",
        data=dataset.Y,
        truth=truth,
        misda_seed=123,
        metadata={"sample_seed": 123, "sigma": 0.0},
    )

    assert observed["invariants_preserved"] is True
    assert observed["policies"][misda.SIZE_SPAN]["policy"] == misda.SIZE_SPAN
    assert (
        observed["policies"][misda.DOMINANCE_PRESERVATION]["policy"]
        == misda.DOMINANCE_PRESERVATION
    )
    assert (
        observed["policies"][misda.DOMINANCE_PRESERVATION]["dominance"]
        ["new_dominance_rate"]
        <= observed["policies"][misda.SIZE_SPAN]["dominance"]
        ["new_dominance_rate"]
    )

    fresh = misda.discover(dataset.Y, seed=123)
    assert misda.rank(fresh).policy == misda.SIZE_SPAN


def test_audit_summary_accounts_for_every_run():
    artifact = impact.run_policy_impact_audit(
        suites=("controlled",),
        n=32,
        problem_ids=("total_redundancy", "blocks_4x5"),
    )

    summary = artifact["summary"]["overall"]
    assert summary["runs"] == 2
    assert summary["same_selected_mis"] + summary["policy_divergences"] == 2
    assert (
        summary["truth_improved"]
        + summary["truth_regressed"]
        + summary["truth_mixed"]
        + summary["truth_neutral"]
        + summary["truth_not_declared"]
        == 2
    )
    assert all(record["invariants_preserved"] for record in artifact["records"])


def test_policy_impact_cli_writes_json(tmp_path):
    output = tmp_path / "ranking-policy-impact.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.run_ranking_policy_impact",
            "--quick",
            "--suite",
            "controlled",
            "--problem-id",
            "total_redundancy",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["format_version"] == 1
    assert artifact["suite"] == "ranking_policy_impact"
    assert artifact["policies"] == {
        "baseline": "size_span",
        "candidate": "dominance_preservation",
    }
    assert artifact["summary"]["overall"]["runs"] == 1
    assert artifact["records"][0]["invariants_preserved"] is True
    assert "overall:" in completed.stdout


def test_aggressiveness_state_records_both_policy_views():
    problem = PROBLEM_BY_ID["blocks_4x5"]
    dataset = problem.generate(N=32, seed=123, sigma=0.1, observation_seed=456)
    case = {
        "problem_id": "blocks_4x5",
        "data": dataset.Y,
        "truth": diagnostic_truth(problem, dataset.Z),
        "misda_seed": 123,
        "metadata": {"sigma": 0.1},
    }

    observed = aggressiveness_probe._state(case, 0.5)

    assert observed["aggressiveness"] == 0.5
    assert 0.0 < observed["alpha"] <= 1.0
    assert observed["structural_dimension"] >= 1
    assert observed["n_mis"] >= 1
    assert set(observed["policies"]) == {
        misda.SIZE_SPAN,
        misda.DOMINANCE_PRESERVATION,
    }
    assert observed["comparison"]["truth_outcome"] in {
        "improved",
        "regressed",
        "mixed",
        "neutral",
        "not_declared",
    }


def test_aggressiveness_probe_cli_writes_json_without_changing_default(tmp_path):
    output = tmp_path / "ranking-policy-aggressiveness.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.run_ranking_policy_aggressiveness_probe",
            "--quick",
            "--problem-id",
            "blocks_4x5",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["format_version"] == 1
    assert artifact["suite"] == "ranking_policy_aggressiveness_probe"
    assert artifact["parameters"]["aggressiveness_grid"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert artifact["screened_noisy_cases"] == 2
    assert "divergences at a=1" in completed.stdout

    problem = PROBLEM_BY_ID["blocks_4x5"]
    dataset = problem.generate(N=32, seed=123, sigma=0.0)
    fresh = misda.discover(dataset.Y, seed=123)
    assert misda.rank(fresh).policy == misda.SIZE_SPAN
