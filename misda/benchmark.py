# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""External benchmark declarations, evaluation, and legacy summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
from itertools import combinations
from numbers import Integral
import platform
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ._pareto import evaluate_pareto_consistency
from ._reconstruction import calculate_ses_nonlinear
from ._statistics import _CORRELATION_MODE
from .result import MISDAResult


FORMAT_VERSION = 3
METHOD = "static"
DEFAULT_SEED = 123


def software_versions():
    packages = ("numpy", "pandas", "scipy", "scikit-learn")
    res = {"python": platform.python_version()}
    try:
        res["misda"] = importlib.metadata.version("misda")
    except importlib.metadata.PackageNotFoundError:
        from ._metadata import __version__

        res["misda"] = __version__
    for name in packages:
        try:
            res[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            res[name] = "unknown"
    return res


def write_json(artifact, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(
            artifact,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")


def matrix_sha256(data) -> str:
    """Hash the canonical float64 bytes of a benchmark input matrix."""

    if hasattr(data, "to_numpy"):
        matrix = data.to_numpy(dtype=np.float64)
    else:
        matrix = np.asarray(data, dtype=np.float64)
    canonical = np.ascontiguousarray(matrix)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _truth_text(truth, field_name):
    value = truth.get(field_name)
    return None if value is None else str(value)


def _truth_dimension(truth, field_name):
    value = truth.get(field_name)
    if value is None or value == "":
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"truth[{field_name!r}] must be an integer or None.")
    value = int(value)
    if value < 0:
        raise ValueError(f"truth[{field_name!r}] must not be negative.")
    return value


def _truth_blocks(truth):
    blocks = truth.get("blocks_expected")
    if blocks is None:
        return None
    try:
        normalized = tuple(tuple(block) for block in blocks)
        for block in normalized:
            for label in block:
                hash(label)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "truth['blocks_expected'] must be a sequence of label sequences."
        ) from exc
    return normalized


def _truth_pareto_indices(truth):
    indices = truth.get("pareto_expected")
    if indices is None:
        return None
    normalized = []
    for index in indices:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
            raise TypeError(
                "truth['pareto_expected'] must contain integer row indices."
            )
        index = int(index)
        if index < 0:
            raise ValueError(
                "truth['pareto_expected'] must not contain negative indices."
            )
        normalized.append(index)
    if len(normalized) != len(set(normalized)):
        raise ValueError("truth['pareto_expected'] contains duplicate indices.")
    return tuple(sorted(normalized))


def _dimension_metrics(observed, expected):
    if expected is None:
        return None, None, None
    error = abs(int(observed) - expected)
    relative_error = float(error / expected) if expected else None
    return error, relative_error, bool(error == 0)


def _found_structural_blocks(result):
    graph = result.analysis.structural_graph
    return tuple(
        tuple(graph.nodes[index]["label"] for index in component)
        for component in result.analysis.structural_components
    )


def _jaccard(left, right):
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def _pair_set(blocks):
    pairs = set()
    for block in blocks:
        unique = tuple(dict.fromkeys(block))
        pairs.update(frozenset(pair) for pair in combinations(unique, 2))
    return pairs


def _precision_recall_f1(predicted, expected):
    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    intersection = len(predicted & expected)
    precision = float(intersection / len(predicted)) if predicted else 0.0
    recall = float(intersection / len(expected)) if expected else 0.0
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def _structural_metrics(found_blocks, expected_blocks):
    found_sets = tuple(set(block) for block in found_blocks)
    expected_sets = tuple(set(block) for block in expected_blocks)
    size = max(len(found_sets), len(expected_sets))
    if size == 0:
        matched_jaccard = 1.0
    else:
        similarities = np.zeros((size, size), dtype=float)
        for row, found in enumerate(found_sets):
            for column, expected in enumerate(expected_sets):
                similarities[row, column] = _jaccard(found, expected)
        rows, columns = linear_sum_assignment(-similarities)
        matched_jaccard = float(similarities[rows, columns].sum() / size)

    found_partition = frozenset(frozenset(block) for block in found_sets)
    expected_partition = frozenset(
        frozenset(block) for block in expected_sets
    )
    precision, recall, f1 = _precision_recall_f1(
        _pair_set(found_blocks),
        _pair_set(expected_blocks),
    )
    return {
        "structural_jaccard": matched_jaccard,
        "structural_precision": precision,
        "structural_recall": recall,
        "structural_f1": f1,
        "structural_partition_exact": bool(
            found_partition == expected_partition
        ),
    }


def _pareto_metrics(predicted_indices, expected_indices):
    predicted = set(predicted_indices)
    expected = set(expected_indices)
    precision, recall, f1 = _precision_recall_f1(predicted, expected)
    union = predicted | expected
    jaccard = float(len(predicted & expected) / len(union)) if union else 1.0
    return {
        "pareto_precision": precision,
        "pareto_recall": recall,
        "pareto_f1": f1,
        "pareto_jaccard": jaccard,
        "pareto_lost": len(expected - predicted),
        "pareto_spurious": len(predicted - expected),
    }


def _format_metric(value):
    if value is None:
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _format_blocks(blocks):
    if blocks is None:
        return "N/A"
    if not blocks:
        return "[]"
    return " | ".join(
        "{" + ", ".join(str(label) for label in block) + "}"
        for block in blocks
    )


@dataclass(frozen=True)
class BenchmarkResult:
    """Comparison of one MISDA result with an external truth declaration."""

    result: MISDAResult
    truth: Mapping[str, Any]

    name: Optional[str]
    feature: Optional[str]
    intuition: Optional[str]
    graph_expected: Optional[str]
    notes: Optional[str]
    latent_expected: Optional[int]
    structural_expected: Optional[int]
    blocks_expected: Optional[tuple]
    pareto_expected: Optional[tuple]
    found_blocks: tuple

    latent_error: Optional[int]
    latent_relative_error: Optional[float]
    latent_exact: Optional[bool]
    structural_error: Optional[int]
    structural_relative_error: Optional[float]
    structural_dimension_exact: Optional[bool]

    structural_jaccard: Optional[float]
    structural_precision: Optional[float]
    structural_recall: Optional[float]
    structural_f1: Optional[float]
    structural_partition_exact: Optional[bool]

    pareto_precision: Optional[float]
    pareto_recall: Optional[float]
    pareto_f1: Optional[float]
    pareto_jaccard: Optional[float]
    pareto_lost: Optional[int]
    pareto_spurious: Optional[int]
    unavailable_reasons: Mapping[str, str]

    def report(self):
        """Render declaration, observed result, and external comparisons."""

        analysis = self.result.analysis
        preferred = self.result.best_mis
        lines = [f"MISDA benchmark report: {self.name or 'Untitled'}"]
        lines.append("=" * 72)
        lines.append("Declaration")
        lines.append(f"  Feature        : {self.feature or 'N/A'}")
        lines.append(f"  Intuition      : {self.intuition or 'N/A'}")
        lines.append(f"  Expected graph : {self.graph_expected or 'N/A'}")
        lines.append(f"  Expected blocks: {_format_blocks(self.blocks_expected)}")
        if self.notes:
            lines.append(f"  Notes          : {self.notes}")

        lines.append("Observed analysis")
        lines.append(
            "  Dimensions     : "
            f"original={analysis.original_dimension}, "
            f"latent={analysis.latent_dimension}, "
            f"structural={analysis.structural_dimension}"
        )
        lines.append(
            "  Preferred MIS  : "
            + (
                f"{preferred.id} {_format_blocks((preferred.objectives,))}"
                if preferred is not None
                else "N/A"
            )
        )
        lines.append(f"  Found blocks   : {_format_blocks(self.found_blocks)}")

        lines.append("Dimensional accuracy")
        lines.append(
            "  Latent         : "
            f"expected={_format_metric(self.latent_expected)}, "
            f"error={_format_metric(self.latent_error)}, "
            f"relative_error={_format_metric(self.latent_relative_error)}, "
            f"exact={_format_metric(self.latent_exact)}"
        )
        lines.append(
            "  Structural     : "
            f"expected={_format_metric(self.structural_expected)}, "
            f"error={_format_metric(self.structural_error)}, "
            f"relative_error={_format_metric(self.structural_relative_error)}, "
            f"exact={_format_metric(self.structural_dimension_exact)}"
        )

        lines.append("Structural reconstruction")
        if "structural" in self.unavailable_reasons:
            lines.append(
                "  N/A — " + self.unavailable_reasons["structural"]
            )
        else:
            lines.append(
                "  Matched Jaccard: "
                f"{_format_metric(self.structural_jaccard)}; "
                f"partition_exact={_format_metric(self.structural_partition_exact)}"
            )
            lines.append(
                "  Pairwise       : "
                f"precision={_format_metric(self.structural_precision)}, "
                f"recall={_format_metric(self.structural_recall)}, "
                f"f1={_format_metric(self.structural_f1)}"
            )

        lines.append("Pareto-front preservation")
        if "pareto" in self.unavailable_reasons:
            lines.append("  N/A — " + self.unavailable_reasons["pareto"])
        else:
            lines.append(
                "  Set agreement  : "
                f"precision={_format_metric(self.pareto_precision)}, "
                f"recall={_format_metric(self.pareto_recall)}, "
                f"f1={_format_metric(self.pareto_f1)}, "
                f"jaccard={_format_metric(self.pareto_jaccard)}"
            )
            lines.append(
                "  Errors         : "
                f"lost={_format_metric(self.pareto_lost)}, "
                f"spurious={_format_metric(self.pareto_spurious)}"
            )
        return "\n".join(lines)


def benchmark(result, truth):
    """Compare a completed analysis with an external declaration.

    The function never reads benchmark truth during analysis and never needs
    the original objective matrix. Pareto comparison uses the reduced-front
    indices already stored in the preferred MIS evaluation.
    """

    if not isinstance(result, MISDAResult):
        raise TypeError("result must be a refactored MISDAResult.")
    if not isinstance(truth, Mapping):
        raise TypeError("truth must be a mapping.")

    latent_expected = _truth_dimension(truth, "latent_expected")
    structural_expected = _truth_dimension(truth, "structural_expected")
    blocks_expected = _truth_blocks(truth)
    pareto_expected = _truth_pareto_indices(truth)
    found_blocks = _found_structural_blocks(result)

    latent = _dimension_metrics(
        result.analysis.latent_dimension,
        latent_expected,
    )
    structural_dimension = _dimension_metrics(
        result.analysis.structural_dimension,
        structural_expected,
    )
    unavailable = {}

    if blocks_expected is None:
        structural = {
            "structural_jaccard": None,
            "structural_precision": None,
            "structural_recall": None,
            "structural_f1": None,
            "structural_partition_exact": None,
        }
        unavailable["structural"] = "blocks_expected was not declared"
    else:
        structural = _structural_metrics(found_blocks, blocks_expected)

    if pareto_expected is None:
        pareto = {
            "pareto_precision": None,
            "pareto_recall": None,
            "pareto_f1": None,
            "pareto_jaccard": None,
            "pareto_lost": None,
            "pareto_spurious": None,
        }
        unavailable["pareto"] = "pareto_expected was not declared"
    else:
        preferred_evaluation = (
            result.best_mis.evaluation if result.best_mis is not None else {}
        )
        reduced_indices = preferred_evaluation.get(
            "pareto_preservation", {}
        ).get("reduced_front_indices")
        if reduced_indices is None:
            pareto = {
                "pareto_precision": None,
                "pareto_recall": None,
                "pareto_f1": None,
                "pareto_jaccard": None,
                "pareto_lost": None,
                "pareto_spurious": None,
            }
            unavailable["pareto"] = (
                "the preferred MIS Pareto frontier was not evaluated"
            )
        else:
            pareto = _pareto_metrics(reduced_indices, pareto_expected)

    if latent_expected is None:
        unavailable["latent_dimension"] = "latent_expected was not declared"
    if structural_expected is None:
        unavailable["structural_dimension"] = (
            "structural_expected was not declared"
        )

    return BenchmarkResult(
        result=result,
        truth=truth,
        name=_truth_text(truth, "name"),
        feature=_truth_text(truth, "feature"),
        intuition=_truth_text(truth, "intuition"),
        graph_expected=_truth_text(truth, "graph_expected"),
        notes=_truth_text(truth, "notes"),
        latent_expected=latent_expected,
        structural_expected=structural_expected,
        blocks_expected=blocks_expected,
        pareto_expected=pareto_expected,
        found_blocks=found_blocks,
        latent_error=latent[0],
        latent_relative_error=latent[1],
        latent_exact=latent[2],
        structural_error=structural_dimension[0],
        structural_relative_error=structural_dimension[1],
        structural_dimension_exact=structural_dimension[2],
        unavailable_reasons=unavailable,
        **structural,
        **pareto,
    )


@dataclass(frozen=True)
class BenchmarkCase:
    """External declaration for one synthetic or empirical benchmark case."""

    case_id: str
    name: str
    latent_dimension: Optional[int] = None
    structural_dimension: Optional[int] = None
    structural_units: tuple[tuple[Any, ...], ...] = ()
    graph_expectations: Mapping[str, Mapping[str, int]] = field(
        default_factory=dict
    )
    adversarial: bool = False
    notes: str = ""

    @classmethod
    def from_truth(cls, case_id, truth, *, adversarial=False):
        """Normalize an examples-only declaration without touching analysis."""

        return cls(
            case_id=str(case_id),
            name=str(truth["name"]),
            latent_dimension=truth.get("latent_expected"),
            structural_dimension=truth.get("structural_expected"),
            structural_units=tuple(
                tuple(unit) for unit in truth.get("blocks_expected", ())
            ),
            graph_expectations={
                "structural": {
                    "components": truth.get("structural_expected")
                },
                "dependence": {
                    "components": truth.get("latent_expected")
                },
            },
            adversarial=bool(adversarial),
            notes=str(truth.get("notes", "")),
        )

    def _units_are_unambiguous(self, result):
        labels = tuple(
            result.analysis.structural_graph.nodes[index]["label"]
            for index in range(result.analysis.original_dimension)
        )
        flattened = [label for unit in self.structural_units for label in unit]
        return bool(
            self.structural_dimension is not None
            and len(self.structural_units) == self.structural_dimension
            and len(flattened) == len(set(flattened)) == len(labels)
            and set(flattened) == set(labels)
        )

    def evaluate(self, result):
        """Compare a completed result with declarations kept outside MISDA."""

        if not isinstance(result, MISDAResult):
            raise TypeError("result must be a refactored MISDAResult.")
        analysis = result.analysis
        checks = []

        def dimension_check(name, observed, expected):
            if expected is None:
                checks.append(
                    {
                        "field": name,
                        "status": "SKIP",
                        "observed": observed,
                        "expected": None,
                        "reason": "NO_DECLARATION",
                    }
                )
                return None
            error = abs(int(observed) - int(expected))
            if error == 0:
                status = "PASS"
                reason = None
            elif self.adversarial:
                status = "EXPECTED_CHANGE"
                reason = "KNOWN_ADVERSARIAL_CASE"
            else:
                status = "REGRESSION"
                reason = "DECLARED_DIMENSION_MISMATCH"
            checks.append(
                {
                    "field": name,
                    "status": status,
                    "observed": int(observed),
                    "expected": int(expected),
                    "absolute_error": error,
                    "reason": reason,
                }
            )
            return error

        latent_error = dimension_check(
            "latent_dimension",
            analysis.latent_dimension,
            self.latent_dimension,
        )
        structural_error = dimension_check(
            "structural_dimension",
            analysis.structural_dimension,
            self.structural_dimension,
        )

        for graph_name, expectations in self.graph_expectations.items():
            observed_summary = analysis.graph_summaries.get(graph_name, {})
            for metric, expected in expectations.items():
                if expected is None:
                    continue
                observed = observed_summary.get(metric)
                matches = observed == expected
                if matches:
                    graph_status = "PASS"
                    graph_reason = None
                elif self.adversarial:
                    graph_status = "EXPECTED_CHANGE"
                    graph_reason = "KNOWN_ADVERSARIAL_CASE"
                else:
                    graph_status = "REGRESSION"
                    graph_reason = "DECLARED_GRAPH_MISMATCH"
                checks.append(
                    {
                        "field": f"graphs.{graph_name}.{metric}",
                        "status": graph_status,
                        "observed": observed,
                        "expected": expected,
                        "reason": graph_reason,
                    }
                )

        unit_adequacy = None
        unit_reason = "DECLARATION_NOT_UNAMBIGUOUS"
        unit_counts = None
        if self._units_are_unambiguous(result):
            selected = set(result.best_mis.objectives if result.best_mis else ())
            unit_counts = [
                len(selected.intersection(unit))
                for unit in self.structural_units
            ]
            unit_adequacy = bool(
                result.best_mis
                and all(count == 1 for count in unit_counts)
                and result.best_mis.size == len(self.structural_units)
            )
            if unit_adequacy:
                unit_status = "PASS"
                unit_reason = None
            elif self.adversarial:
                unit_status = "EXPECTED_CHANGE"
                unit_reason = "KNOWN_ADVERSARIAL_CASE"
            else:
                unit_status = "REGRESSION"
                unit_reason = "DECLARED_UNIT_MISMATCH"
            checks.append(
                {
                    "field": "preferred_structural_units",
                    "status": unit_status,
                    "observed": unit_counts,
                    "expected": [1] * len(self.structural_units),
                    "reason": unit_reason,
                }
            )
        else:
            checks.append(
                {
                    "field": "preferred_structural_units",
                    "status": "SKIP",
                    "observed": None,
                    "expected": None,
                    "reason": unit_reason,
                }
            )

        statuses = {check["status"] for check in checks}
        if "REGRESSION" in statuses:
            status = "REGRESSION"
        elif "EXPECTED_CHANGE" in statuses:
            status = "EXPECTED_CHANGE"
        else:
            status = "PASS"
        return {
            "case_id": self.case_id,
            "status": status,
            "known_adversarial": self.adversarial,
            "dimension_errors": {
                "latent": latent_error,
                "structural": structural_error,
            },
            "preferred_unit_adequacy": unit_adequacy,
            "preferred_unit_counts": unit_counts,
            "checks": checks,
        }


@dataclass(frozen=True)
class BenchmarkSuite:
    """Evaluate already-produced results against external case declarations."""

    name: str
    cases: tuple[BenchmarkCase, ...]

    def evaluate(self, results):
        missing = [case.case_id for case in self.cases if case.case_id not in results]
        if missing:
            raise KeyError(f"Missing benchmark result(s): {', '.join(missing)}")
        assessments = [
            case.evaluate(results[case.case_id]) for case in self.cases
        ]
        statuses = {assessment["status"] for assessment in assessments}
        if "REGRESSION" in statuses:
            status = "REGRESSION"
        elif "EXPECTED_CHANGE" in statuses:
            status = "EXPECTED_CHANGE"
        else:
            status = "PASS"
        return {
            "suite": self.name,
            "status": status,
            "cases": assessments,
        }


def serialize_benchmark_result(case, result, data, *, seed):
    """Serialize one completed static result without consulting declarations."""

    if not isinstance(case, BenchmarkCase):
        raise TypeError("case must be a BenchmarkCase.")
    if not isinstance(result, MISDAResult):
        raise TypeError("result must be a refactored MISDAResult.")
    preferred = result.best_mis
    evaluation = preferred.evaluation if preferred is not None else {}
    linear = evaluation.get("linear_reconstruction")
    pareto = evaluation.get("pareto_preservation")
    analysis = result.analysis
    return {
        "case_id": case.case_id,
        "name": case.name,
        "seed": int(seed),
        "n": int(result._data.shape[0]),
        "m": int(result._data.shape[1]),
        "input_sha256": matrix_sha256(data),
        "declared": {
            "latent_dimension": case.latent_dimension,
            "structural_dimension": case.structural_dimension,
        },
        "estimated": {
            "latent_dimension": analysis.latent_dimension,
            "structural_dimension": analysis.structural_dimension,
            "preferred_mis_size": preferred.size if preferred else None,
        },
        "preferred_indices": list(preferred.indices) if preferred else [],
        "preferred_labels": (
            [str(label) for label in preferred.objectives] if preferred else []
        ),
        "preferred_rank": preferred.rank if preferred else None,
        "preferred_rank_values": dict(preferred.rank_values) if preferred else {},
        "n_mis": analysis.n_mis,
        "rank_counts": {
            str(rank): count for rank, count in sorted(analysis.rank_counts.items())
        },
        "graphs": {
            name: dict(summary)
            for name, summary in analysis.graph_summaries.items()
        },
        "linear_reconstruction": linear,
        "pareto_preservation": pareto,
        "separation_status": analysis.separation_status.value,
        "assessment": case.evaluate(result),
    }


def compare_results(baseline, candidate, *, expected_changes=None):
    """Compare normalized case artifacts with explicit scientific rules."""

    expected_changes = expected_changes or {}
    checks = []

    def metric_value(artifact, block, metric):
        value = artifact.get(block, {}).get(metric)
        if value is not None or block != "pareto_preservation":
            return value
        legacy_names = {
            "pareto_retention": "retention",
            "pareto_validity": "validity",
            "pareto_jaccard": "jaccard",
        }
        return artifact.get("pareto", {}).get(legacy_names[metric])

    def record(field, before, after, status, reason=None):
        if field in expected_changes and before != after:
            status = "EXPECTED_CHANGE"
            reason = expected_changes[field]
        checks.append(
            {
                "field": field,
                "baseline": before,
                "candidate": after,
                "status": status,
                "reason": reason,
            }
        )

    same_input = baseline.get("input_sha256") == candidate.get("input_sha256")
    record(
        "input_sha256",
        baseline.get("input_sha256"),
        candidate.get("input_sha256"),
        "PASS" if same_input else "REGRESSION",
        None if same_input else "INPUT_CHANGED",
    )

    declared = candidate.get("declared", {})
    for dimension in ("latent_dimension", "structural_dimension"):
        expected = declared.get(dimension)
        before = baseline.get("estimated", {}).get(dimension)
        after = candidate.get("estimated", {}).get(dimension)
        if expected is None or before is None or after is None:
            continue
        before_error = abs(before - expected)
        after_error = abs(after - expected)
        if after_error < before_error:
            status = "IMPROVED"
        elif after_error == before_error:
            status = "PASS"
        else:
            status = "REGRESSION"
        record(
            f"estimated.{dimension}",
            before,
            after,
            status,
            None if status != "REGRESSION" else "DIMENSION_ERROR_INCREASED",
        )

    same_structural_graph = all(
        baseline.get("graphs", {}).get("structural", {}).get(field)
        == candidate.get("graphs", {}).get("structural", {}).get(field)
        for field in ("edges", "components")
    )
    if same_structural_graph:
        for field in ("n_mis", "rank_counts", "preferred_indices"):
            before = baseline.get(field)
            after = candidate.get(field)
            record(
                field,
                before,
                after,
                "PASS" if before == after else "REGRESSION",
                None if before == after else "SAME_GRAPH_RESULT_CHANGED",
            )

    if baseline.get("preferred_indices") == candidate.get("preferred_indices"):
        for block, metrics in (
            ("linear_reconstruction", ("mean_r2", "worst_r2")),
            (
                "pareto_preservation",
                ("pareto_retention", "pareto_validity", "pareto_jaccard"),
            ),
        ):
            for metric in metrics:
                before = metric_value(baseline, block, metric)
                after = metric_value(candidate, block, metric)
                if before is None or after is None:
                    continue
                status = "PASS" if after == before else (
                    "IMPROVED" if after > before else "REGRESSION"
                )
                record(
                    f"{block}.{metric}",
                    before,
                    after,
                    status,
                    None if status != "REGRESSION" else "QUALITY_DECREASED",
                )

    statuses = {check["status"] for check in checks}
    if "REGRESSION" in statuses:
        status = "REGRESSION"
    elif "EXPECTED_CHANGE" in statuses:
        status = "EXPECTED_CHANGE"
    elif "IMPROVED" in statuses:
        status = "IMPROVED"
    else:
        status = "PASS"
    return {"status": status, "checks": checks}


def compile_benchmark_summary(results_dict, sort_by=None):
    """
    Standardizes the summary table generation for MISDA benchmarks.
    Extracts all key metrics including Alpha regimes, Homogeneity, Linear/Non-Linear Fidelity, and Pareto Consistency.

    Args:
        results_dict (dict): Dictionary mapping Case Name -> MISDAResult object.
                             Alternatively, can be a dictionary where values are dicts containing 'result_obj'.
        sort_by (str): Column name to sort by.

    Returns:
        pd.DataFrame: Comprehensive summary table.
    """
    rows = []

    for case_name, item in results_dict.items():
        # Handle both direct MISDAResult and wrapper dicts (e.g. from benchmark.ipynb)
        # item could be MISDAResult or dict
        res = None
        truth_dim = None

        if hasattr(item, 'best_mis'):
            res = item
        elif isinstance(item, dict) and 'result_obj' in item:
            res = item['result_obj']
            truth_dim = item.get('truth', {}).get('structural_expected' if _CORRELATION_MODE == "positive" else 'latent_expected', item.get('truth', {}).get('intrinsic_dim_expected', None))

        if res is None:
            continue

        # Basic Params
        # Handle res.Y being dataframe or numpy
        N, M = res.Y.shape
        algo_alpha = res.alpha

        # MIS Info
        mis_indices = res.best_mis.indices if res.best_mis else []
        dim_red = len(mis_indices)

        # Fidelity (Linear)
        fidel_lin = None
        if res.ses_results and isinstance(res.ses_results, dict):
            fidel_lin = res.ses_results.get("F_real", None)

        # Fidelity (Non-Linear)
        fidel_nl = None
        try:
            if N <= 5000: # Per safeguard
                nl_out = calculate_ses_nonlinear(res.Y, mis_indices, n_estimators=50, return_details=True)
                if isinstance(nl_out, dict):
                    fidel_nl = nl_out.get("F_real", None)
                else:
                    fidel_nl = nl_out
        except Exception:
            pass

        # Pareto Consistency
        prec, rec = 0.0, 0.0
        try:
             prec, rec = evaluate_pareto_consistency(res, res.Y)
        except Exception:
             pass

        # Homogeneity
        homog = res.homogeneity_ratio

        # Status
        status = "OK"
        low_lin = (fidel_lin is not None and fidel_lin < 0.9)
        low_nl = (fidel_nl is not None and fidel_nl < 0.9)
        if low_lin and low_nl:
            status = "LOW_FIDEL"
        if prec < 1.0:
            status = "UNSAFE(Prec)"
        if truth_dim and dim_red != truth_dim:
             status += f"|DimMismatch({dim_red}!={truth_dim})"

        # Alpha Bounds
        a_min = res.alpha_min if hasattr(res, 'alpha_min') else 0.0
        a_max = res.alpha_max if hasattr(res, 'alpha_max') else 1.0

        def _fmt_f(val):
            return f"{val:.2f}" if val is not None else "N/A"

        row = {
            "Case": case_name,
            "Regime": res.regime.name if res.regime else "N/A",
            "N": N,
            "M": M,
            "Dim(Red)": dim_red,
            "Alpha": f"{algo_alpha:.4f}",
            "Min": f"{a_min:.2f}",
            "Max": f"{a_max:.2f}",
            "Homog": f"{homog:.2f}",
            "Fidel(Lin)": _fmt_f(fidel_lin),
            "Fidel(NL)": _fmt_f(fidel_nl),
            "Prec": f"{prec:.2f}",
            "Rec": f"{rec:.2f}",
            "Status": status
        }

        if truth_dim is not None:
            row["Exp"] = truth_dim

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Reorder columns if Exp exists
    cols = ["Case", "Regime", "N", "M", "Exp", "Dim(Red)", "Alpha", "Min", "Max", "Homog", "Fidel(Lin)", "Fidel(NL)", "Prec", "Rec", "Status"]
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols]

    if sort_by and sort_by in df.columns:
        df = df.sort_values(by=sort_by)

    return df
