# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""MISDA public API.

The canonical user workflow separates structural regime diagnosis from the
final representative choice::

    profile = misda.profile(Y)
    result = misda.discovery(profile)

The lower-level ``discover`` / ``evaluate`` / ``rank`` primitives remain public
for scientific instrumentation and advanced use.
"""

from ._metadata import __version__

# The canonical low-level discovery implementation lives in ``api``.
# Correlation selection is injected only at the statistics boundary so Pearson
# and the experimental Spearman backend share the complete downstream pipeline.
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
    DOMINANCE_PRESERVATION,
    NO_REDUNDANCY,
    PARETO_RETENTION,
    SIZE_SPAN,
    SUPPORTED_REDUCTION,
    UNSUPPORTED_REDUCTION,
    STRUCTURAL_COVERAGE,
    CandidateSupport,
    DimensionalSupport,
    DiscoveryAnalysis,
    DominanceMetrics,
    JackknifeMetrics,
    LinearMetrics,
    MISCandidate,
    MISSet,
    NonlinearMetrics,
    NullReferenceMetrics,
    ParetoMetrics,
    Ranking,
    ReductionAssessment,
    StructuralMetrics,
    rank,
)
from ._profile import (
    ABSTAINED,
    DiscoveryResult,
    Profile,
    ProfileRegime,
    discovery as _discovery_high_level,
    profile as _profile_impl,
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
    """Low-level discovery of the complete static structural MIS universe.

    Pearson is the canonical MISDA correlation backend. Spearman is retained
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
    result.correlation = backend
    result.experimental = backend != "pearson"
    return result


# Keep ``misda.api.discover`` and ``misda.discover`` consistent for callers
# that import the implementation module directly.
_api.discover = discover


def profile(
    Y,
    *,
    seed=123,
    name=None,
    cancel_requested=None,
    correlation="pearson",
    experimental=False,
):
    """Map distinct structural regimes across the calibrated alpha path."""

    with _correlation_backend(correlation, experimental) as backend:
        result = _profile_impl(
            Y,
            seed=seed,
            name=name,
            cancel_requested=cancel_requested,
        )
    result.correlation = backend
    result.experimental = backend != "pearson"
    for regime in result:
        regime.mis_set.correlation = backend
        regime.mis_set.experimental = backend != "pearson"
    return result


def discovery(
    source,
    *,
    rank_policy=DOMINANCE_PRESERVATION,
    seed=123,
    name=None,
    cancel_requested=None,
    correlation="pearson",
    experimental=False,
):
    """Return a high-level reduction decision from observed Y or a Profile.

    When ``source`` is raw Y, profiling is performed first. When it is already
    a :class:`Profile`, the stored selected regime is used directly. The
    high-level default is ``dominance_preservation``; low-level ``rank()`` keeps
    its historical ``size_span`` default.
    """

    if isinstance(source, Profile):
        return _discovery_high_level(source, rank_policy=rank_policy)
    observed_profile = profile(
        source,
        seed=seed,
        name=name,
        cancel_requested=cancel_requested,
        correlation=correlation,
        experimental=experimental,
    )
    return _discovery_high_level(observed_profile, rank_policy=rank_policy)


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
    "PARETO_RETENTION",
    "DOMINANCE_PRESERVATION",
    "NO_REDUNDANCY",
    "SUPPORTED_REDUCTION",
    "UNSUPPORTED_REDUCTION",
    "STRUCTURAL_COVERAGE",
    "PARTIALLY_SUPPORTED",
    "ABSTAINED",
    "StructuralMetrics",
    "JackknifeMetrics",
    "LinearMetrics",
    "NonlinearMetrics",
    "NullReferenceMetrics",
    "ParetoMetrics",
    "DominanceMetrics",
    "ReductionAssessment",
    "ParetoStabilityDiagnostics",
    "MISCandidate",
    "CandidateSupport",
    "DimensionalSupport",
    "DiscoveryAnalysis",
    "MISSet",
    "Ranking",
    "ProfileRegime",
    "Profile",
    "DiscoveryResult",
    "profile",
    "discovery",
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
