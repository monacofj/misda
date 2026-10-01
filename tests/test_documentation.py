"""Contract checks for the active new static API documentation."""

import re
from pathlib import Path

import pytest


ACTIVE_DOCS = (
    Path("README.md"),
    Path("docs/userguide.md"),
)


@pytest.mark.parametrize("path", ACTIVE_DOCS)
def test_active_documentation_describes_new_static_contract(path):
    text = path.read_text(encoding="utf-8").lower()

    for term in (
        "discover",
        "evaluate",
        "rank",
        "size_span",
        "aggressiveness",
    ):
        assert term in text

    assert "target_fidelity" not in text
    assert "method='adaptive'" not in text
    assert 'method="adaptive"' not in text


def test_documentation_has_single_normative_and_research_ledgers():
    adr = Path("docs/adr/README.md")
    notebook = Path("docs/research-notes/README.md")

    assert adr.exists()
    assert notebook.exists()
    assert "authoritative architectural and methodological specification" in adr.read_text(
        encoding="utf-8"
    ).lower()
    assert "laboratory notebook" in notebook.read_text(encoding="utf-8").lower()

    for obsolete in (
        Path("docs/decisions.md"),
        Path("docs/design_notes.md"),
        Path("docs/validation_results.md"),
    ):
        assert not obsolete.exists()


@pytest.mark.parametrize("path", (Path("README.md"), Path("docs/userguide.md")))
def test_python_examples_use_only_new_public_workflow(path):
    text = path.read_text(encoding="utf-8")
    python_blocks = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)

    assert python_blocks
    joined = "\n".join(python_blocks)
    assert "misda.discover(" in joined
    assert "mis_set.evaluate(" in joined
    assert "misda.rank(" in joined
    assert "ranking.mis(" in joined
    assert "misda.evaluate(mis_set" in text
    assert "misda.analyze(" not in joined
    assert "misda.heavy(" not in joined
    assert ".validate(" not in joined
    assert "caution=" not in joined


def test_readme_points_to_main_and_executable_benchmarks():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "benchmarks.run_controlled" in text
    assert "benchmarks.run_comparison" in text
    assert "benchmarks.run_classical" in text
    assert "benchmarks/controlled.ipynb" in text
    assert "benchmarks/controlled_noisy.ipynb" in text
    assert "benchmarks/sampling_robustness.ipynb" in text
    assert "benchmarks/noisy_robustness.ipynb" in text
    assert "benchmarks/comparison.ipynb" in text
    assert "benchmarks/classical.ipynb" in text
    assert "blob/main/benchmarks/controlled.ipynb" in text
    assert "examples/" not in text
    assert "examples.benchmarks" not in text
    assert "@refactor" not in text
    assert "@efficient" not in text


def test_readme_keeps_classical_front_geometry_separate_from_misda_truth():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "analytical Pareto-manifold geometry" in text
    assert "not re-labelled as MISDA latent or structural ground truth" in text
