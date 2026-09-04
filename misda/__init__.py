# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""MISDA public API.

The alpha-stage public surface is intentionally small. Static analysis is
expressed as three separate operations:

``discover`` -> structural inference and MIS universe
``evaluate`` -> candidate-level evidence
``rank``     -> an ordered view under a named policy

Adaptive analysis and the previous ``analyze``/``heavy`` result model are not
part of this API.
"""

from ._metadata import __version__
from .newapi import (
    PARTIALLY_SUPPORTED,
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
    StructuralMetrics,
    discover,
    evaluate,
    rank,
)
from .benchmark import (
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkSuite,
    benchmark,
    compare_results,
    compile_benchmark_summary,
    serialize_benchmark_result,
)

__all__ = [
    "__version__",
    "STRUCTURAL_COVERAGE",
    "PARTIALLY_SUPPORTED",
    "StructuralMetrics",
    "JackknifeMetrics",
    "LinearMetrics",
    "NonlinearMetrics",
    "NullReferenceMetrics",
    "ParetoMetrics",
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
