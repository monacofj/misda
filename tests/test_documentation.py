"""Contract checks for the active static documentation."""

import re
from pathlib import Path

import pytest


ACTIVE_DOCS = (
    Path("README.md"),
    Path("docs/userguide.md"),
    Path("docs/design_notes.md"),
)


@pytest.mark.parametrize("path", ACTIVE_DOCS)
def test_active_documentation_describes_static_v2_contract(path):
    text = path.read_text(encoding="utf-8")

    for term in (
        "static",
        "structural dimension",
        "latent dimension",
        "aggressiveness",
    ):
        assert term in text.lower()

    assert "target_fidelity" not in text
    assert "method='adaptive'" not in text
    assert 'method="adaptive"' not in text


@pytest.mark.parametrize("path", (Path("README.md"), Path("docs/userguide.md")))
def test_python_examples_do_not_use_removed_workflow(path):
    text = path.read_text(encoding="utf-8")
    python_blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)

    assert python_blocks
    for index, source in enumerate(python_blocks):
        compile(source, f"{path}:python-block-{index}", "exec")
        assert ".validate(" not in source
        assert ".plot(" not in source
        assert ".to_pandas(" not in source
        assert "caution=" not in source


def test_readme_points_to_executable_static_benchmarks_only():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "examples.benchmarks.run_benchmark" in text
    assert "examples.benchmarks.run_comparative" in text
    assert "examples/benchmark.ipynb" in text
    assert "examples/comparative.ipynb" in text
    assert "examples/dtlz.ipynb" not in text
    assert "FeatureAgglomeration" not in text


def test_changelog_records_static_refactor_and_suspended_scope():
    text = Path("CHANGELOG.md").read_text(encoding="utf-8")

    unreleased = text.split("## [0.4.1]", 1)[0]
    assert "## [Unreleased]" in unreleased
    assert "misda.heavy()" in unreleased
    assert "adaptive implementation is suspended" in unreleased
