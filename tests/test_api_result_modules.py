"""Characterization tests for the extracted static API and result objects."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import misda
from misda import api, result


PUBLIC_STATIC_API = (
    "report_significant_correlations",
    "misda_significance_from_corr",
    "misda_significance",
    "_analyze_static_fast",
    "_analyze_static",
    "_analyze_static_v2",
)

PUBLIC_RESULT_TYPES = (
    "MISCandidate",
    "MISDAResult",
)


@pytest.mark.parametrize("name", PUBLIC_STATIC_API)
def test_static_api_operations_are_reexported_from_package(name):
    assert getattr(misda, name) is getattr(api, name)


@pytest.mark.parametrize("name", PUBLIC_RESULT_TYPES)
def test_result_types_are_reexported_from_package(name):
    assert getattr(misda, name) is getattr(result, name)


@pytest.fixture
def static_result():
    data = np.array(
        [
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 0.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 2.5],
        ]
    )
    corr = np.corrcoef(data, rowvar=False)
    return misda._analyze_static_fast(
        data,
        corr,
        alpha_min=0.001,
        alpha_max=0.05,
        alpha_exec=0.05,
        caution=1.0,
        name="characterization",
        ensure_coverage=False,
    )


def test_static_pipeline_returns_extracted_result_type(static_result):
    assert type(static_result) is result.LegacyMISDAResult
    assert isinstance(static_result.best_mis, result.LegacyMISCandidate)
    assert static_result.name == "characterization"
    assert static_result.best_mis_indices == [0, 2]
    assert static_result.best_mis_labels == ["f1", "f3"]


def test_result_text_methods_keep_working_after_extraction(static_result):
    summary = static_result.summary()
    report = static_result.report(top_k=1)

    assert "MISDA Analysis Summary: characterization" in summary
    assert "Best MIS Labels: ['f1', 'f3']" in summary
    assert "MISDA Version: 0.4.2" in report
    assert "DETAILED INSPECTION REPORT" in report


def test_result_plot_keeps_working_after_extraction(static_result):
    figure = static_result.plot(show=False)

    assert figure.axes
    plt.close(figure)


def test_candidate_characterization_is_unchanged():
    candidate = misda.LegacyMISCandidate(
        {
            "mis_indices": [0, 2],
            "mis_labels": ["f1", "f3"],
            "rank": 2,
        }
    )

    assert candidate.indices == [0, 2]
    assert candidate.labels == ["f1", "f3"]
    assert candidate.rank == 2
    assert candidate.size == 2
    assert repr(candidate) == "<MISCandidate: ['f1', 'f3'] (Size=2, Rank=2)>"
