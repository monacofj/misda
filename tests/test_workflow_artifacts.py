"""Regression guard for benchmark artifacts in the acceptance workflow."""

from pathlib import Path


def test_acceptance_workflow_preserves_benchmark_json_artifacts():
    workflow = Path(".github/workflows/newapi.yml").read_text(encoding="utf-8")

    assert "uses: actions/upload-artifact@v4" in workflow
    assert "name: misda-benchmark" in workflow
    assert "path: /tmp/misda-benchmark.json" in workflow
    assert "name: misda-comparative" in workflow
    assert "path: /tmp/misda-comparative.json" in workflow
    assert workflow.count("if: always()") >= 2
