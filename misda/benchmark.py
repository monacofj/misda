# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""External benchmark declarations, evaluation, and legacy summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from ._pareto import evaluate_pareto_consistency
from ._reconstruction import calculate_ses_nonlinear
from ._statistics import _CORRELATION_MODE
from .result import MISDAResult


FORMAT_VERSION = 2
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
