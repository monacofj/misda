"""Contract checks for the active new static API documentation."""

import re
from pathlib import Path

import pytest


ACTIVE_DOCS = (
    Path("README.md"),
    Path("docs/userguide.md"),
    Path("docs/design_notes.md"),
)


@pytest.mark.parametrize("path", ACTIVE_DOCS)
def test_active_documentation_describes_new_static_contract(path):
    text = path.read_text(encoding="utf-8").lower()

    for term in (
        "discover",
        "evaluate",
        "rank",
        "structural_coverage",
        "aggressiveness",
    ):
        assert term in text

    assert "target_fidelity" not in text
    assert "method='adaptive'" not in text
    assert 'method="adaptive"' not in text


def test_design_notes_state_current_dimension_semantics_and_null_signature():
    text = Path("docs/design_notes.md").read_text(encoding="utf-8")

    assert "structural_dimension = alpha(G+)" in text
    assert "latent_dimension     = alpha(G±)" in text
    assert "complete structural_coverage tie-group ordering" in text
    assert "Literal graph equality" in text
    assert "Raw metric equality" in text


@pytest.mark.parametrize("path", (Path("README.md"), Path("docs/userguide.md")))
def test_python_examples_use_only_new_public_workflow(path):
    text = path.read_text(encoding="utf-8")
    python_blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)

    assert python_blocks
    joined = "\n".join(python_blocks)
    assert "misda.discover(" in joined
    assert "misda.evaluate(" in joined
    assert "misda.rank(" in joined
    assert "misda.analyze(" not in joined
    assert "misda.heavy(" not in joined
    assert ".validate(" not in joined
    assert "caution=" not in joined


def test_readme_points_to_main_and_executable_benchmarks():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "examples.benchmarks.run_benchmark" in text
    assert "examples.benchmarks.run_comparative" in text
    assert "examples/benchmark.ipynb" in text
    assert "examples/comparative.ipynb" in text
    assert "blob/main/examples/benchmark.ipynb" in text
    assert "@refactor" not in text
    assert "@efficient" not in text
