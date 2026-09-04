# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Static MISDA discovery, evaluation, and ranking API."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import networkx as nx
import numpy as np

from ._graph import build_dependency_graphs, enumerate_structural_mis
from ._pareto import evaluate_pareto_preservation, get_nondominated_mask_minimize
from ._ranking import compute_mis_metrics
from ._reconstruction import (
    _derive_seed,
    evaluate_linear_reconstruction,
    evaluate_nonlinear_reconstruction,
    evaluate_null_reconstruction,
)
from ._statistics import (
    compute_correlation_statistics,
    estimate_null_positive_correlation,
    interpolate_log_alpha,
    separation_status,
)
from ._support import (
    SUPPORTED,
    UNSUPPORTED,
    evaluate_dimensional_support_group,
)
from ._validation import normalize_input_matrix, validate_aggressiveness


STRUCTURAL_COVERAGE = "structural_coverage"
PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"


@dataclass(frozen=True)
class StructuralMetrics:
    """Structural measurements of one maximal independent set."""

    neighborhood: int
    neighborhood_ratio: float
    span: int
    avg_external_degree: float
    avg_internal_degree: float


@dataclass(frozen=True)
class JackknifeMetrics:
    """Sampling uncertainty attached to reconstruction metrics."""

    r2_se_by_objective: Any
    mean_r2_se: Optional[float]
    worst_r2_se: Optional[float]
    n_replicates: int
    reason: Optional[str]

    def r2_se(self, objective):
        values = self.r2_se_by_objective
        if values is None:
            return None
        return values.get(objective)


@dataclass(frozen=True)
class LinearMetrics:
    """Linear external reconstruction evidence for one candidate."""

    r2_by_objective: Any
    r2_reason_by_objective: Any
    mean_r2: Optional[float]
    worst_r2: Optional[float]
    reason_by_metric: Any
    jackknife: JackknifeMetrics

    def r2(self, objective):
        if self.r2_by_objective is None:
            return None
        return self.r2_by_objective.get(objective)

    def reason(self, objective):
        return self.r2_reason_by_objective.get(objective)


@dataclass(frozen=True)
class NullReferenceMetrics:
    """Permutation-null evidence attached to nonlinear reconstruction."""

    mean_null_r2: Optional[float]
    above_null_r2: Optional[float]
    incidental_reconstruction_rate: Optional[float]
    n_permutations: int
    mc_se_mean_null_r2: Optional[float]
    above_null_r2_se: Optional[float]
    incidental_reconstruction_rate_se: Optional[float]
    converged: bool
    cancelled: bool
    reason: Optional[str]


@dataclass(frozen=True)
class NonlinearMetrics:
    """Nonlinear external reconstruction evidence for one candidate."""

    r2_by_objective: Any
    r2_reason_by_objective: Any
    mean_r2: Optional[float]
    worst_r2: Optional[float]
    reason_by_metric: Any
    jackknife: JackknifeMetrics
    tree_se_by_objective: Any
    n_trees: int
    configuration_counts: Any
    configuration_by_outer_fold: Any
    converged: bool
    cancelled: bool
    convergence_reason: Optional[str]
    null_reference: Optional[NullReferenceMetrics] = None

    def r2(self, objective):
        if self.r2_by_objective is None:
            return None
        return self.r2_by_objective.get(objective)

    def reason(self, objective):
        return self.r2_reason_by_objective.get(objective)


@dataclass(frozen=True)
class ParetoMetrics:
    """Pareto-front preservation evidence for one candidate."""

    retention: Optional[float]
    validity: Optional[float]
    jaccard: Optional[float]
    full_front_size: int
    reduced_front_size: int
    intersection_size: int
    union_size: int
    exact_preservation: bool
    reduced_front_indices: Tuple[int, ...]


@dataclass(frozen=True)
class MISCandidate:
    """One discovered structural maximal independent set."""

    objectives: Tuple[Any, ...]
    indices: Tuple[int, ...]
    structural: StructuralMetrics
    linear: Optional[LinearMetrics] = field(default=None, compare=False)
    nonlinear: Optional[NonlinearMetrics] = field(default=None, compare=False)
    pareto: Optional[ParetoMetrics] = field(default=None, compare=False)

    @property
    def size(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class CandidateSupport:
    """Dimensional-support evidence for one structurally preferred candidate."""

    candidate_index: int
    status: str
    reasons: Tuple[str, ...]
    transitivity_observed: float
    transitivity_null: float
    transitivity_excess: float
    spectral_tested_dimension: int
    spectral_observed_next_eigenvalue: float
    spectral_null_next_eigenvalue: float
    spectral_excess: float
    n_permutations: int
    seed: int


class DimensionalSupport:
    """Aggregate support over the complete first structural-rank tie group."""

    def __init__(self, candidate_results, candidates):
        self._results = tuple(candidate_results)
        self._candidates = candidates
        statuses = tuple(item.status for item in self._results)
        if statuses and all(status == SUPPORTED for status in statuses):
            self.status = SUPPORTED
        elif statuses and all(status == UNSUPPORTED for status in statuses):
            self.status = UNSUPPORTED
        else:
            self.status = PARTIALLY_SUPPORTED

    @property
    def candidates(self):
        return tuple(self._candidates[item.candidate_index] for item in self._results)

    @property
    def supported(self):
        return tuple(
            self._candidates[item.candidate_index]
            for item in self._results
            if item.status == SUPPORTED
        )

    @property
    def unsupported(self):
        return tuple(
            self._candidates[item.candidate_index]
            for item in self._results
            if item.status == UNSUPPORTED
        )

    @property
    def results(self):
        return self._results

    def for_candidate(self, candidate):
        if isinstance(candidate, (int, np.integer)):
            index = int(candidate)
        else:
            index = next(
                (
                    position
                    for position, observed in enumerate(self._candidates)
                    if observed is candidate
                ),
                None,
            )
            if index is None:
                raise ValueError("candidate does not belong to this MISSet.")
        for result in self._results:
            if result.candidate_index == index:
                return result
        raise ValueError("candidate is not in the first structural-rank group.")


@dataclass(frozen=True)
class DiscoveryAnalysis:
    """Global outputs of structural discovery."""

    original_dimension: int
    latent_dimension: int
    structural_dimension: int
    alpha_onset: Optional[float]
    log_alpha_onset: Optional[float]
    alpha_null: float
    log_alpha_null: float
    alpha: float
    log_alpha: float
    aggressiveness: float
    separation_status: Any
    structural_graph: Any
    dependence_graph: Any
    structural_components: Tuple[Tuple[int, ...], ...]
    latent_components: Tuple[Tuple[int, ...], ...]
    alpha_null_converged: bool
    alpha_null_reason: Optional[str]
    alpha_null_permutations: int
    alpha_null_se_mc: float
    alpha_null_r_interval: Tuple[float, float]
    alpha_null_log_interval: Tuple[float, float]


class MISSet:
    """Complete immutable-order universe returned by :func:`discover`."""

    def __init__(
        self,
        *,
        analysis,
        candidates,
        rank_groups,
        data,
        labels,
        seed,
        name=None,
        support=None,
        timings=None,
    ):
        self.analysis = analysis
        self._candidates = tuple(candidates)
        self._rank_groups = tuple(tuple(group) for group in rank_groups)
        self._data = data
        self._labels = tuple(labels)
        self.seed = int(seed)
        self.name = name
        self.support = support
        self.timings = dict(timings or {})
        self._evaluation_scopes = {}

    def __len__(self):
        return len(self._candidates)

    def __iter__(self):
        return iter(self._candidates)

    def __getitem__(self, key):
        return self._candidates[key]

    @property
    def structural_ranking(self):
        return Ranking(
            self,
            tuple(range(len(self))),
            policy=STRUCTURAL_COVERAGE,
            groups=self._rank_groups,
        )

    def evaluation_scope(self, family):
        if family not in {"linear", "pareto", "nonlinear"}:
            return self._evaluation_scopes.get(family)
        basis_entry = self._evaluation_scopes.get(family)
        evaluated = sum(
            getattr(candidate, family) is not None
            for candidate in self._candidates
        )
        if basis_entry is None and evaluated == 0:
            return None
        basis = basis_entry[1] if basis_entry is not None else "existing metrics"
        return evaluated, basis

    def report(self):
        ranking = self.structural_ranking
        lines = [f"MISDA discovery: {self.name or 'Untitled'}"]
        lines.append(
            "Dimensions: "
            f"original={self.analysis.original_dimension}, "
            f"latent={self.analysis.latent_dimension}, "
            f"structural={self.analysis.structural_dimension}"
        )
        lines.append(
            f"Structural ranking: policy={ranking.policy}; "
            f"selected_dimension={ranking.selected_dimension}; MISs={len(self)}"
        )
        lines.append(
            f"Dimensional support: "
            f"{self.support.status if self.support is not None else 'N/A'}"
        )
        if self.support is not None and len(self.support.results) > 1:
            for item in self.support.results:
                reasons = ", ".join(item.reasons) or "none"
                lines.append(
                    f"  candidate[{item.candidate_index}]: {item.status}; "
                    f"reasons={reasons}"
                )
        for family in ("linear", "pareto", "nonlinear"):
            scope = self.evaluation_scope(family)
            if scope is not None and scope[0] != len(self):
                lines.append(
                    f"Note: {family} metrics were evaluated for "
                    f"{scope[0]} of {len(self)} candidates only "
                    f"({scope[1]})."
                )
        return "\n".join(lines)

    def graph_plot(self, show=True, ranking=None):
        """Plot the stored positive structural graph and a ranking selection."""

        from ._plotting import plot_mis_set_graph

        return plot_mis_set_graph(
            self,
            ranking=self.structural_ranking if ranking is None else ranking,
            show=show,
        )


class Ranking:
    """Snapshot ordering view over candidates in one MISSet."""

    def __init__(self, mis_set, indices, *, policy, groups=None):
        self.mis_set = mis_set
        self.indices = tuple(int(index) for index in indices)
        self.policy = policy
        if groups is None:
            self.groups = tuple((index,) for index in self.indices)
        else:
            allowed = set(self.indices)
            self.groups = tuple(
                tuple(index for index in group if index in allowed)
                for group in groups
                if any(index in allowed for index in group)
            )

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return (self.mis_set[index] for index in self.indices)

    def __getitem__(self, key):
        if isinstance(key, slice):
            selected = self.indices[key]
            return Ranking(
                self.mis_set,
                selected,
                policy=self.policy,
                groups=self.groups,
            )
        return self.mis_set[self.indices[key]]

    @property
    def selected(self):
        return self[0] if self.indices else None

    @property
    def selected_dimension(self):
        return self.selected.size if self.selected is not None else None

    def position(self, candidate):
        canonical = next(
            (
                index
                for index, observed in enumerate(self.mis_set)
                if observed is candidate
            ),
            None,
        )
        if canonical is None:
            raise ValueError("candidate does not belong to this MISSet.")
        try:
            return self.indices.index(canonical)
        except ValueError as exc:
            raise ValueError("candidate is outside this Ranking view.") from exc


def _structural_sort_key(metric):
    return (
        -metric["size"],
        -metric["neighborhood"],
        -metric["avg_external_degree"],
        -metric["span"],
        tuple(repr(label) for label in metric["mis_labels"]),
    )


def _structural_rank_value(metric):
    return (
        metric["size"],
        metric["neighborhood"],
        metric["avg_external_degree"],
        metric["span"],
    )


def _rank_structural_coverage(structure, labels):
    n_objectives = structure.structural_graph.number_of_nodes()
    adjacency = nx.to_numpy_array(
        structure.structural_graph,
        nodelist=range(n_objectives),
        dtype=int,
        weight=None,
    )
    measured = compute_mis_metrics(
        enumerate_structural_mis(structure), adjacency, labels
    )
    ordered = sorted(measured, key=_structural_sort_key)
    groups = []
    previous = object()
    for index, metric in enumerate(ordered):
        value = _structural_rank_value(metric)
        if not groups or value != previous:
            groups.append([])
            previous = value
        groups[-1].append(index)
    return ordered, tuple(tuple(group) for group in groups)


def _discovery_signature(correlation_statistics, log_alpha):
    structure = build_dependency_graphs(correlation_statistics, log_alpha)
    ranked, groups = _rank_structural_coverage(
        structure, correlation_statistics.labels
    )
    grouped_mis = tuple(
        tuple(
            sorted(
                tuple(ranked[index]["mis_indices"])
                for index in group
            )
        )
        for group in groups
    )
    return (
        structure.structural_dimension,
        structure.latent_dimension,
        grouped_mis,
    )


def _candidate_support(raw, index):
    return CandidateSupport(
        candidate_index=index,
        status=raw["status"],
        reasons=tuple(raw["reasons"]),
        transitivity_observed=float(raw["transitivity"]["observed"]),
        transitivity_null=float(raw["transitivity"]["null"]),
        transitivity_excess=float(raw["transitivity"]["excess"]),
        spectral_tested_dimension=int(raw["spectral"]["tested_dimension"]),
        spectral_observed_next_eigenvalue=float(
            raw["spectral"]["observed_next_eigenvalue"]
        ),
        spectral_null_next_eigenvalue=float(
            raw["spectral"]["null_next_eigenvalue"]
        ),
        spectral_excess=float(raw["spectral"]["excess"]),
        n_permutations=int(raw["n_permutations"]),
        seed=int(raw["seed"]),
    )


def discover(
    Y,
    *,
    aggressiveness=1.0,
    seed=123,
    name=None,
    cancel_requested=None,
):
    """Discover the complete static structural MIS universe."""

    total_start = time.perf_counter()
    normalized = normalize_input_matrix(Y)
    aggressiveness = validate_aggressiveness(aggressiveness)

    statistics_start = time.perf_counter()
    correlation_statistics = compute_correlation_statistics(normalized)

    def signature(log_alpha):
        return _discovery_signature(correlation_statistics, log_alpha)

    null_estimate = estimate_null_positive_correlation(
        normalized,
        signature=signature,
        seed=seed,
        cancel_requested=cancel_requested,
    )
    status = separation_status(
        correlation_statistics.log_alpha_onset,
        null_estimate.log_alpha_null,
    )
    if correlation_statistics.log_alpha_onset is None:
        log_alpha = null_estimate.log_alpha_null
    else:
        log_alpha = interpolate_log_alpha(
            correlation_statistics.log_alpha_onset,
            null_estimate.log_alpha_null,
            aggressiveness,
        )
    statistics_seconds = time.perf_counter() - statistics_start

    graph_start = time.perf_counter()
    structure = build_dependency_graphs(correlation_statistics, log_alpha)
    ranked, groups = _rank_structural_coverage(structure, normalized.labels)
    candidates = tuple(
        MISCandidate(
            objectives=tuple(item["mis_labels"]),
            indices=tuple(item["mis_indices"]),
            structural=StructuralMetrics(
                neighborhood=int(item["neighborhood"]),
                neighborhood_ratio=float(item["neighborhood_ratio"]),
                span=int(item["span"]),
                avg_external_degree=float(item["avg_external_degree"]),
                avg_internal_degree=float(item["avg_internal_degree"]),
            ),
        )
        for item in ranked
    )
    graph_seconds = time.perf_counter() - graph_start

    support_start = time.perf_counter()
    first_group = groups[0] if groups else tuple()
    raw_support = evaluate_dimensional_support_group(
        normalized.data,
        tuple(candidates[index].indices for index in first_group),
        structure.latent_dimension,
        seed=seed,
    )
    support_results = tuple(
        _candidate_support(raw, index)
        for raw, index in zip(raw_support, first_group)
    )
    support_seconds = time.perf_counter() - support_start

    analysis = DiscoveryAnalysis(
        original_dimension=normalized.n_objectives,
        latent_dimension=structure.latent_dimension,
        structural_dimension=structure.structural_dimension,
        alpha_onset=correlation_statistics.alpha_onset,
        log_alpha_onset=correlation_statistics.log_alpha_onset,
        alpha_null=null_estimate.alpha_null,
        log_alpha_null=null_estimate.log_alpha_null,
        alpha=float(np.exp(log_alpha)),
        log_alpha=log_alpha,
        aggressiveness=aggressiveness,
        separation_status=status,
        structural_graph=structure.structural_graph,
        dependence_graph=structure.dependence_graph,
        structural_components=structure.structural_components,
        latent_components=structure.latent_components,
        alpha_null_converged=bool(null_estimate.converged),
        alpha_null_reason=null_estimate.reason,
        alpha_null_permutations=int(null_estimate.n_permutations),
        alpha_null_se_mc=float(null_estimate.se_mc),
        alpha_null_r_interval=tuple(null_estimate.r_interval),
        alpha_null_log_interval=tuple(null_estimate.log_alpha_interval),
    )

    data = normalized.data
    data.setflags(write=False)
    result = MISSet(
        analysis=analysis,
        candidates=candidates,
        rank_groups=groups,
        data=data,
        labels=normalized.labels,
        seed=seed,
        name=name,
        timings={
            "statistics": statistics_seconds,
            "graph_and_ranking": graph_seconds,
            "dimensional_support": support_seconds,
            "total": time.perf_counter() - total_start,
        },
    )
    result.support = DimensionalSupport(support_results, result._candidates)
    return result


def _jackknife(raw):
    value = raw["jackknife"]
    return JackknifeMetrics(
        r2_se_by_objective=value["r2_se_by_objective"],
        mean_r2_se=value["mean_r2_se"],
        worst_r2_se=value["worst_r2_se"],
        n_replicates=int(value["n_replicates"]),
        reason=value["reason"],
    )


def _linear_metrics(raw):
    return LinearMetrics(
        r2_by_objective=raw["r2_by_objective"],
        r2_reason_by_objective=raw["r2_reason_by_objective"],
        mean_r2=raw["mean_r2"],
        worst_r2=raw["worst_r2"],
        reason_by_metric=raw["reason_by_metric"],
        jackknife=_jackknife(raw),
    )


def _null_metrics(raw):
    if raw is None:
        return None
    return NullReferenceMetrics(
        mean_null_r2=raw.get("mean_null_r2"),
        above_null_r2=raw.get("above_null_r2"),
        incidental_reconstruction_rate=raw.get("incidental_reconstruction_rate"),
        n_permutations=int(raw.get("n_permutations", 0)),
        mc_se_mean_null_r2=raw.get("mc_se_mean_null_r2"),
        above_null_r2_se=raw.get("above_null_r2_se"),
        incidental_reconstruction_rate_se=raw.get(
            "incidental_reconstruction_rate_se"
        ),
        converged=bool(raw.get("converged", False)),
        cancelled=bool(raw.get("cancelled", False)),
        reason=raw.get("reason"),
    )


def _nonlinear_metrics(raw):
    return NonlinearMetrics(
        r2_by_objective=raw["r2_by_objective"],
        r2_reason_by_objective=raw["r2_reason_by_objective"],
        mean_r2=raw["mean_r2"],
        worst_r2=raw["worst_r2"],
        reason_by_metric=raw["reason_by_metric"],
        jackknife=_jackknife(raw),
        tree_se_by_objective=raw["tree_se_by_objective"],
        n_trees=int(raw["n_trees"]),
        configuration_counts=raw["configuration_counts"],
        configuration_by_outer_fold=raw["configuration_by_outer_fold"],
        converged=bool(raw["converged"]),
        cancelled=bool(raw["cancelled"]),
        convergence_reason=raw["convergence_reason"],
        null_reference=_null_metrics(raw.get("null_reference")),
    )


def _pareto_metrics(raw):
    return ParetoMetrics(
        retention=raw["pareto_retention"],
        validity=raw["pareto_validity"],
        jaccard=raw["pareto_jaccard"],
        full_front_size=int(raw["full_front_size"]),
        reduced_front_size=int(raw["reduced_front_size"]),
        intersection_size=int(raw["intersection_size"]),
        union_size=int(raw["union_size"]),
        exact_preservation=bool(raw["exact_preservation"]),
        reduced_front_indices=tuple(raw["reduced_front_indices"]),
    )


def _candidate_indices(mis_set, candidates, metrics):
    if candidates is None:
        return (
            (0,)
            if "nonlinear" in metrics and len(mis_set)
            else tuple(range(len(mis_set)))
        ), "default scope"
    if isinstance(candidates, str):
        if candidates != "all":
            raise ValueError("the only string candidate selector is 'all'.")
        return tuple(range(len(mis_set))), "all candidates"
    if isinstance(candidates, Ranking):
        if candidates.mis_set is not mis_set:
            raise ValueError("Ranking belongs to a different MISSet.")
        return candidates.indices, f"{candidates.policy} Ranking view"
    if (
        isinstance(candidates, (int, np.integer))
        and not isinstance(candidates, (bool, np.bool_))
    ):
        count = int(candidates)
        if count < 0:
            raise ValueError("candidates must be non-negative.")
        return (
            tuple(range(min(count, len(mis_set)))),
            f"first {count} in {STRUCTURAL_COVERAGE} order",
        )
    try:
        selected = tuple(int(index) for index in candidates)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "candidates must be 'all', an integer, a Ranking, or an index sequence."
        ) from exc
    if len(set(selected)) != len(selected):
        raise ValueError("candidate indices must be unique.")
    if any(index < 0 or index >= len(mis_set) for index in selected):
        raise IndexError("candidate index out of range.")
    return selected, "explicit candidate indices"


def evaluate(
    mis_set,
    *,
    metrics=("linear", "pareto"),
    candidates=None,
    null_reference=False,
    cancel_requested=None,
):
    """Enrich candidates with requested metric families without reordering."""

    if not isinstance(mis_set, MISSet):
        raise TypeError("mis_set must be an MISSet.")
    requested = tuple(metrics)
    allowed = {"structural", "linear", "nonlinear", "pareto"}
    unknown = tuple(metric for metric in requested if metric not in allowed)
    if unknown:
        raise ValueError(f"Unknown metric families: {unknown!r}.")
    if not isinstance(null_reference, (bool, np.bool_)):
        raise TypeError("null_reference must be a boolean.")
    if null_reference and "nonlinear" not in requested:
        raise ValueError("null_reference requires nonlinear metrics.")
    if cancel_requested is not None and not callable(cancel_requested):
        raise TypeError("cancel_requested must be callable or None.")

    selected, basis = _candidate_indices(mis_set, candidates, requested)
    full_front = None
    if "pareto" in requested:
        full_front = get_nondominated_mask_minimize(mis_set._data)

    started = time.perf_counter()
    for index in selected:
        candidate = mis_set[index]
        if "linear" in requested and candidate.linear is None:
            raw = evaluate_linear_reconstruction(
                mis_set._data, candidate.indices, mis_set._labels
            )
            object.__setattr__(candidate, "linear", _linear_metrics(raw))

        if "pareto" in requested and candidate.pareto is None:
            raw = evaluate_pareto_preservation(
                mis_set._data,
                candidate.indices,
                full_front=full_front,
            )
            object.__setattr__(candidate, "pareto", _pareto_metrics(raw))

        if "nonlinear" in requested:
            nonlinear = candidate.nonlinear
            if nonlinear is None:
                raw = evaluate_nonlinear_reconstruction(
                    mis_set._data,
                    candidate.indices,
                    mis_set._labels,
                    seed=_derive_seed(mis_set.seed, 8001, index),
                    cancel_requested=cancel_requested,
                )
                nonlinear = _nonlinear_metrics(raw)
                object.__setattr__(candidate, "nonlinear", nonlinear)
            if nonlinear.cancelled:
                break
            if null_reference and nonlinear.null_reference is None:
                raw_nonlinear = {
                    "r2_by_objective": nonlinear.r2_by_objective,
                    "r2_reason_by_objective": nonlinear.r2_reason_by_objective,
                    "mean_r2": nonlinear.mean_r2,
                    "worst_r2": nonlinear.worst_r2,
                    "reason_by_metric": nonlinear.reason_by_metric,
                    "jackknife": {
                        "r2_se_by_objective": nonlinear.jackknife.r2_se_by_objective,
                        "mean_r2_se": nonlinear.jackknife.mean_r2_se,
                        "worst_r2_se": nonlinear.jackknife.worst_r2_se,
                        "n_replicates": nonlinear.jackknife.n_replicates,
                        "reason": nonlinear.jackknife.reason,
                    },
                    "tree_se_by_objective": nonlinear.tree_se_by_objective,
                    "n_trees": nonlinear.n_trees,
                    "configuration_counts": nonlinear.configuration_counts,
                    "configuration_by_outer_fold": nonlinear.configuration_by_outer_fold,
                    "converged": nonlinear.converged,
                    "cancelled": nonlinear.cancelled,
                    "convergence_reason": nonlinear.convergence_reason,
                }
                raw_null = evaluate_null_reconstruction(
                    mis_set._data,
                    candidate.indices,
                    mis_set._labels,
                    raw_nonlinear,
                    seed=_derive_seed(mis_set.seed, 8002, index),
                    cancel_requested=cancel_requested,
                    evaluator=evaluate_nonlinear_reconstruction,
                )
                object.__setattr__(
                    candidate,
                    "nonlinear",
                    NonlinearMetrics(
                        **{
                            **nonlinear.__dict__,
                            "null_reference": _null_metrics(raw_null),
                        }
                    ),
                )
                if raw_null.get("cancelled", False):
                    break

    for family in requested:
        if family != "structural":
            previous = mis_set._evaluation_scopes.get(family)
            if previous is None or previous[1] == basis:
                scope_basis = basis
            else:
                scope_basis = "multiple evaluation calls"
            mis_set._evaluation_scopes[family] = (0, scope_basis)
    mis_set.timings["evaluation"] = (
        mis_set.timings.get("evaluation", 0.0) + time.perf_counter() - started
    )
    return mis_set


def rank(
    mis_set,
    policy=STRUCTURAL_COVERAGE,
    *,
    candidates="all",
    accept_cost=False,
):
    """Create a ranking snapshot over an already discovered MISSet."""

    if not isinstance(mis_set, MISSet):
        raise TypeError("mis_set must be an MISSet.")
    if policy != STRUCTURAL_COVERAGE:
        raise ValueError(
            f"Unsupported ranking policy {policy!r}; currently only "
            f"{STRUCTURAL_COVERAGE!r} is defined."
        )
    if not isinstance(accept_cost, (bool, np.bool_)):
        raise TypeError("accept_cost must be a boolean.")
    selected, _ = _candidate_indices(
        mis_set,
        candidates,
        metrics=("structural",),
    )
    allowed = set(selected)
    ordered = tuple(index for index in range(len(mis_set)) if index in allowed)
    groups = tuple(
        tuple(index for index in group if index in allowed)
        for group in mis_set._rank_groups
        if any(index in allowed for index in group)
    )
    return Ranking(
        mis_set,
        ordered,
        policy=STRUCTURAL_COVERAGE,
        groups=groups,
    )
