"""Contracts for the pre-default ranking policy impact audit."""

from __future__ import annotations

import json
import subprocess
import sys

import misda
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
