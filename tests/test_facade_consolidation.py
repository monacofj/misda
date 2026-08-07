"""Characterization tests for the consolidated legacy static facade."""

import importlib

import numpy as np
import pandas as pd
import pytest

import misda
from misda import _metadata, _plotting, _reporting, _statistics


benchmark_module = importlib.import_module("misda.benchmark")


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("calculate_spectral_entropy", _statistics),
        ("_enforce_min_distance", _plotting),
        ("_parse_node_to_1based", _plotting),
        ("_extract_mis_nodes_1based", _plotting),
        ("plot_custom_misda_graph", _plotting),
        ("explain_ses", _reporting),
        ("compile_benchmark_summary", benchmark_module),
    ],
)
def test_consolidated_operations_are_reexported_from_package(name, module):
    assert getattr(misda, name) is getattr(module, name)


def test_package_version_comes_from_metadata_module():
    assert misda.__version__ is _metadata.__version__
    assert misda.__version__ == "0.4.1"


def test_spectral_entropy_values_are_unchanged():
    fully_redundant = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    uncorrelated = np.array(
        [[1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]]
    )

    assert misda.calculate_spectral_entropy(fully_redundant) == 0.0
    assert misda.calculate_spectral_entropy(uncorrelated) == pytest.approx(1.0)


def test_reporting_helper_value_is_unchanged():
    observed = misda.explain_ses(
        {
            "status": "NO_REDUCTION",
            "mis_size": 3,
            "ses": None,
            "F_real": None,
            "F_null": None,
        },
        name="example",
    )

    assert "Structural Evidence Score for example" in observed
    assert "Status: NO_REDUCTION" in observed
    assert "SES = N/A" in observed


def test_empty_benchmark_summary_schema_is_unchanged():
    observed = misda.compile_benchmark_summary({})

    assert isinstance(observed, pd.DataFrame)
    assert observed.empty
