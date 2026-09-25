# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""MISDA public API.

The alpha-stage public surface is intentionally small. Static analysis keeps
three responsibilities separate:

``discover``        -> structural inference and MIS universe
``MISSet.evaluate`` -> candidate-level evidence
``rank``            -> an ordered view whose ``mis()`` selector exposes one MIS

The module-level ``evaluate(mis_set, ...)`` form remains public and equivalent.
Adaptive analysis and the previous ``analyze``/``heavy`` result model are not
part of this API.
"""

from ._metadata import __version__

# The canonical discovery implementation lives in ``api``.  Correlation
# selection is injected only at the statistics boundary so Pearson and the
# experimental Spearman backend share the complete downstream pipeline.
from . import api as _api
from ._correlation_backend import (
    compute_correlation_statistics as _compute_correlation_statistics,
    correlation_backend as _correlation_backend,
    estimate_null_positive_correlation as _estimate_null_positive_correlation,
)

_api.compute_correlation_statistics = _compute_correlation_statistics
_api.estimate_null_positive_correlation = _estimate_null_positive_correlation
_discover_impl = _api.discover

from .api import (
    PARTIALLY_SUPPORTED,
    SIZE_PARETO,
    SIZE_SPAN,
    STRUCTURAL_COVERAGE,
    CandidateSupport,
    DimensionalSupport,
    DiscoveryAnalysis,
    JackknifeMetrics,
    LinearMetrics,
    MISCandidate,
    MISSet,
    NonlinearMetrics,
    NullReferenceMetrics,
    ParetoMetrics,
    Ranking,
    rank,
)


def discover(
    Y,
    *,
    aggressiveness=1.0,
    seed=123,
    name=None,
    cancel_requested=None,
    correlation="pearson",
    experimental=False,
):
    """Discover the complete static structural MIS universe.

    Pearson is the canonical MISDA correlation backend.  Spearman is retained
    for reproducible research and requires the explicit opt-in
    ``experimental=True``.
    """

    with _correlation_backend(correlation, experimental) as backend:
        result = _discover_impl(
            Y,
            aggressiveness=aggressiveness,
            seed=seed,
            name=name,
            cancel_requested=cancel_requested,
        )
    # MISSet is intentionally a mutable result container for evaluation state;
    # record the discovery backend so experimental results remain auditable.
    result.correlation = backend
    result.experimental = backend != "pearson"
    return result


# Keep ``misda.api.discover`` and ``misda.discover`` consistent for callers
# that import the implementation module directly.
_api.discover = discover

from ._pareto_stability import ParetoStabilityDiagnostics, evaluate
from . import _reporting as _reporting  # installs the legacy-complete MISSet report
from . import _front_plotting as _front_plotting  # installs graph/front plot views
from .benchmark import (
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkSuite,
    compare_results,
    serialize_benchmark_result,
)
from ._benchmark_observation import benchmark, compile_benchmark_summary

__all__ = [
    "__version__",
    "SIZE_SPAN",
    "SIZE_PARETO",
    "STRUCTURAL_COVERAGE",
    "PARTIALLY_SUPPORTED",
    "StructuralMetrics",
    "JackknifeMetrics",
    "LinearMetrics",
    "NonlinearMetrics",
    "NullReferenceMetrics",
    "ParetoMetrics",
    "ParetoStabilityDiagnostics",
    "MISCandidate",
    "CandidateSupport",
    "DimensionalSupport",
    "DiscoveryAnalysis",
    "MISSet",
    "Ranking",
    "discover",
    "evaluate",
    "rank",
    "BenchmarkCase",
    "BenchmarkResult",
    "BenchmarkSuite",
    "benchmark",
    "compare_results",
    "compile_benchmark_summary",
    "serialize_benchmark_result",
]
