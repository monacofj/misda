"""Regression guards for the benchmark acceptance workflow."""

from pathlib import Path


def _workflow_text():
    return Path(".github/workflows/newapi.yml").read_text(encoding="utf-8")


def test_acceptance_workflow_preserves_benchmark_json_artifacts():
    workflow = _workflow_text()

    assert "uses: actions/upload-artifact@v4" in workflow
    assert "name: misda-diagnostic-clean" in workflow
    assert "path: /tmp/misda-diagnostic-clean.json" in workflow
    assert "name: misda-comparative" in workflow
    assert "path: /tmp/misda-comparative.json" in workflow
    assert workflow.count("if: always()") >= 2


def test_acceptance_workflow_runs_for_main_prs_and_pushes():
    workflow = _workflow_text()

    assert "name: acceptance gate" in workflow
    assert "  push:\n    branches: [main]\n" in workflow
    assert "  pull_request:\n    branches: [main]\n" in workflow
    assert "newapi" not in workflow
