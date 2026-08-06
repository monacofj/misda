"""Reproduce the frozen scientific baseline from the pre-refactor code."""

import json
from pathlib import Path

from examples.benchmarks.run_benchmark import run_benchmark
from examples.benchmarks.run_comparative import run_comparative


BASELINE_PATH = (
    Path(__file__).parent / "baselines" / "refactor_preimplementation.json"
)
SOURCE_COMMIT = "2e08b9aeb952039f19f607045c457d0d2de23ff4"


def _scientific_payload(artifact: dict) -> dict:
    """Exclude diagnostic environment versions from scientific comparison."""
    payload = dict(artifact)
    payload.pop("software")
    return payload


def test_preimplementation_manifest_reproduces_exact_scientific_results(monkeypatch):
    def legacy_analyze(Y, method="static", name=None, **kwargs):
        assert method == "static"
        return __import__("misda")._analyze_static(
            Y,
            caution=1.0,
            name=name,
            ensure_coverage=True,
        )

    monkeypatch.setattr(__import__("misda"), "analyze", legacy_analyze)
    manifest = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    assert manifest["baseline_format_version"] == 1
    assert manifest["source"] == {
        "branch": "refactor",
        "commit": SOURCE_COMMIT,
        "method": "static",
    }

    expected = manifest["artifacts"]
    observed = {
        "benchmark": run_benchmark(n=1000, seed=123),
        "comparative": run_comparative(n=500, seed=123),
    }

    for suite in ("benchmark", "comparative"):
        assert set(observed[suite]["software"]) == set(
            expected[suite]["software"]
        )
        assert _scientific_payload(observed[suite]) == _scientific_payload(
            expected[suite]
        )

    # This fails if any generated metric is NaN or infinite.
    json.dumps(observed, allow_nan=False)
