# SPDX-FileCopyrightText: 2026 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""High-level structural profile and discovery workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import api as _api
from ._graph import build_dependency_graphs
from ._statistics import interpolate_log_alpha, separation_status
from ._support import evaluate_dimensional_support_group
from ._validation import normalize_input_matrix


@dataclass(frozen=True)
class ProfileRegime:
    """One distinct structural regime along the alpha/aggressiveness path."""

    index: int
    aggressiveness: float
    alpha: float
    log_alpha: float
    mis_set: _api.MISSet

    @property
    def analysis(self):
        return self.mis_set.analysis

    @property
    def status(self):
        """Aggregate discovery status for this structural regime.

        This is deliberately independent of any ranking policy.  A regime with
        no structural reduction is admissible by construction; otherwise the
        status is the aggregate dimensional-support state of the complete first
        canonical structural tie group.
        """

        if self.analysis.structural_dimension == self.analysis.original_dimension:
            return _api.NO_REDUNDANCY
        support = self.mis_set.support
        if support is None:
            return _api.UNSUPPORTED_REDUCTION
        if support.status == "SUPPORTED":
            return _api.SUPPORTED_REDUCTION
        if support.status == _api.PARTIALLY_SUPPORTED:
            return _api.PARTIALLY_SUPPORTED
        return _api.UNSUPPORTED_REDUCTION

    @property
    def acceptable(self):
        """Whether this regime is safe for automatic profile selection."""

        return self.status in {_api.NO_REDUNDANCY, _api.SUPPORTED_REDUCTION}


class Profile:
    """Diagnostic map of structural regimes inferred from one observed Y."""

    def __init__(
        self,
        *,
        regimes,
        selected_index,
        alpha_onset,
        alpha_null,
        separation,
        seed,
        name=None,
        correlation="pearson",
        experimental=False,
    ):
        self._regimes = tuple(regimes)
        self.selected_index = selected_index
        self.alpha_onset = alpha_onset
        self.alpha_null = float(alpha_null)
        self.separation_status = separation
        self.seed = int(seed)
        self.name = name
        self.correlation = correlation
        self.experimental = bool(experimental)

    def __len__(self):
        return len(self._regimes)

    def __iter__(self):
        return iter(self._regimes)

    def __getitem__(self, key):
        return self._regimes[key]

    @property
    def regimes(self):
        return self._regimes

    @property
    def selected(self) -> Optional[ProfileRegime]:
        if self.selected_index is None:
            return None
        return self._regimes[self.selected_index]

    def report(self):
        lines = [f"MISDA profile: {self.name or 'Untitled'}"]
        onset = "None" if self.alpha_onset is None else f"{self.alpha_onset:.6g}"
        lines.append(
            f"Threshold path: alpha_onset={onset}; alpha_null={self.alpha_null:.6g}; "
            f"separation={getattr(self.separation_status, 'value', self.separation_status)}"
        )
        lines.append(
            "Regimes: idx  aggressiveness  alpha        latent  structural  MISs  assessment"
        )
        interpretations = {
            _api.NO_REDUNDANCY: "no reduction",
            _api.SUPPORTED_REDUCTION: "reduction supported",
            _api.PARTIALLY_SUPPORTED: "mixed support; not auto-selected",
            _api.UNSUPPORTED_REDUCTION: "reduction not supported",
        }
        for regime in self._regimes:
            marker = "*" if self.selected_index == regime.index else " "
            analysis = regime.analysis
            interpretation = interpretations.get(regime.status, "inspect regime")
            lines.append(
                f"{marker} {regime.index:>3}  {regime.aggressiveness:>14.6f}  "
                f"{regime.alpha:>10.6g}  {analysis.latent_dimension:>6}  "
                f"{analysis.structural_dimension:>10}  {len(regime.mis_set):>4}  "
                f"{regime.status} — {interpretation}"
            )
        if self.selected is None:
            lines.append("Selected regime: none — no fully supported admissible regime.")
        else:
            lines.append(
                f"Selected regime: {self.selected.index} — most aggressive admissible "
                f"regime (aggressiveness={self.selected.aggressiveness:.6f}, "
                f"alpha={self.selected.alpha:.6g})."
            )
        lines.append("This selects the threshold regime only; no MIS has been selected yet.")
        return "\n".join(lines)


def _critical_aggressiveness(correlation_statistics, alpha_onset, alpha_null):
    """Return boundary probes spanning every threshold transition on the path."""

    if alpha_onset is None:
        return (1.0,)
    alpha_onset = float(alpha_onset)
    alpha_null = float(alpha_null)
    if alpha_onset == alpha_null:
        return (1.0,)

    upper = np.triu(correlation_statistics.valid_pairs, k=1)
    pair_log_p = correlation_statistics.log_p[upper]
    pair_alpha = np.exp(pair_log_p[np.isfinite(pair_log_p)])
    lo = min(alpha_onset, alpha_null)
    hi = max(alpha_onset, alpha_null)
    pair_alpha = np.unique(pair_alpha[(pair_alpha >= lo) & (pair_alpha <= hi)])

    executed = [alpha_onset, alpha_null]
    for threshold in pair_alpha:
        threshold = float(threshold)
        if threshold == alpha_onset or threshold == alpha_null:
            continue
        for direction in (-np.inf, 0.0, np.inf):
            value = threshold if direction == 0.0 else float(np.nextafter(threshold, direction))
            if lo <= value <= hi:
                executed.append(value)

    denominator = alpha_null - alpha_onset
    values = []
    for alpha in executed:
        aggressiveness = (float(alpha) - alpha_onset) / denominator
        aggressiveness = min(1.0, max(0.0, float(aggressiveness)))
        values.append(aggressiveness)
    return tuple(sorted(set(values)))


def _state_signature(correlation_statistics, log_alpha):
    structure = build_dependency_graphs(correlation_statistics, log_alpha)
    ranked, groups = _api._rank_size_span(structure, correlation_statistics.labels)
    grouped_mis = tuple(
        tuple(tuple(ranked[index]["mis_indices"]) for index in group)
        for group in groups
    )
    return (
        structure.structural_dimension,
        structure.latent_dimension,
        structure.structural_components,
        structure.latent_components,
        grouped_mis,
    )


def _build_regime(
    *,
    normalized,
    correlation_statistics,
    null_estimate,
    separation,
    aggressiveness,
    seed,
    name,
    support_cache,
):
    if correlation_statistics.log_alpha_onset is None:
        log_alpha = null_estimate.log_alpha_null
    else:
        log_alpha = interpolate_log_alpha(
            correlation_statistics.log_alpha_onset,
            null_estimate.log_alpha_null,
            aggressiveness,
        )
    structure = build_dependency_graphs(correlation_statistics, log_alpha)
    ranked, groups = _api._rank_size_span(structure, normalized.labels)
    candidates = tuple(
        _api.MISCandidate(
            objectives=tuple(item["mis_labels"]),
            indices=tuple(item["mis_indices"]),
            structural=_api.StructuralMetrics(
                neighborhood=int(item["neighborhood"]),
                neighborhood_ratio=float(item["neighborhood_ratio"]),
                span=int(item["span"]),
                avg_external_degree=float(item["avg_external_degree"]),
                avg_internal_degree=float(item["avg_internal_degree"]),
            ),
        )
        for item in ranked
    )

    # Profile is a structural scan, not a ranking decision.  Store support for
    # every discovered MIS exactly as low-level discover() does, while retaining
    # the historical aggregate support over the complete first size-span tie
    # group.  Shared regimes reuse identical permutation work when possible.
    all_indices = tuple(range(len(candidates)))
    selected_sets = tuple(candidates[index].indices for index in all_indices)
    support_key = (int(structure.latent_dimension), selected_sets)
    raw_support = support_cache.get(support_key)
    if raw_support is None:
        raw_support = evaluate_dimensional_support_group(
            normalized.data,
            selected_sets,
            structure.latent_dimension,
            seed=seed,
        )
        support_cache[support_key] = raw_support
    all_support_results = tuple(
        _api._candidate_support(raw, index)
        for raw, index in zip(raw_support, all_indices)
    )
    support_by_index = {
        item.candidate_index: item for item in all_support_results
    }
    first_group = groups[0] if groups else tuple()
    support_results = tuple(support_by_index[index] for index in first_group)

    analysis = _api.DiscoveryAnalysis(
        original_dimension=normalized.n_objectives,
        latent_dimension=structure.latent_dimension,
        structural_dimension=structure.structural_dimension,
        alpha_onset=correlation_statistics.alpha_onset,
        log_alpha_onset=correlation_statistics.log_alpha_onset,
        alpha_null=null_estimate.alpha_null,
        log_alpha_null=null_estimate.log_alpha_null,
        alpha=float(np.exp(log_alpha)),
        log_alpha=float(log_alpha),
        aggressiveness=float(aggressiveness),
        separation_status=separation,
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
    result = _api.MISSet(
        analysis=analysis,
        candidates=candidates,
        rank_groups=groups,
        data=normalized.data,
        labels=normalized.labels,
        seed=seed,
        name=name,
    )
    result.support = _api.DimensionalSupport(support_results, result._candidates)
    result._support_by_index = support_by_index
    return result


def profile(Y, *, seed=123, name=None, cancel_requested=None):
    """Map distinct structural regimes over the calibrated alpha path."""

    normalized = normalize_input_matrix(Y)
    correlation_statistics = _api.compute_correlation_statistics(normalized)

    def signature(log_alpha):
        return _api._discovery_signature(correlation_statistics, log_alpha)

    null_estimate = _api.estimate_null_positive_correlation(
        normalized,
        signature=signature,
        seed=seed,
        cancel_requested=cancel_requested,
    )
    separation = separation_status(
        correlation_statistics.log_alpha_onset,
        null_estimate.log_alpha_null,
    )

    points = _critical_aggressiveness(
        correlation_statistics,
        correlation_statistics.alpha_onset,
        null_estimate.alpha_null,
    )

    compressed = []
    for aggressiveness in points:
        if correlation_statistics.log_alpha_onset is None:
            log_alpha = null_estimate.log_alpha_null
        else:
            log_alpha = interpolate_log_alpha(
                correlation_statistics.log_alpha_onset,
                null_estimate.log_alpha_null,
                aggressiveness,
            )
        current_signature = _state_signature(correlation_statistics, log_alpha)
        if compressed and compressed[-1][0] == current_signature:
            compressed[-1] = (current_signature, float(aggressiveness))
        else:
            compressed.append((current_signature, float(aggressiveness)))

    regimes = []
    support_cache = {}
    for _signature, aggressiveness in compressed:
        mis_set = _build_regime(
            normalized=normalized,
            correlation_statistics=correlation_statistics,
            null_estimate=null_estimate,
            separation=separation,
            aggressiveness=aggressiveness,
            seed=seed,
            name=name,
            support_cache=support_cache,
        )
        regimes.append(
            ProfileRegime(
                index=len(regimes),
                aggressiveness=float(aggressiveness),
                alpha=float(mis_set.analysis.alpha),
                log_alpha=float(mis_set.analysis.log_alpha),
                mis_set=mis_set,
            )
        )

    selected_index = next(
        (regime.index for regime in reversed(regimes) if regime.acceptable),
        None,
    )
    return Profile(
        regimes=regimes,
        selected_index=selected_index,
        alpha_onset=correlation_statistics.alpha_onset,
        alpha_null=null_estimate.alpha_null,
        separation=separation,
        seed=seed,
        name=name,
    )


def discovery(source, **profile_kwargs):
    """Return the MISSet for the regime selected by a Profile.

    Raw observed Y is accepted as a convenience and is profiled first.  This
    operation never ranks or selects an MIS; ranking policy belongs exclusively
    to :func:`misda.rank`.
    """

    if isinstance(source, Profile):
        if profile_kwargs:
            unexpected = ", ".join(sorted(profile_kwargs))
            raise TypeError(
                f"profile options are not accepted when source is already a Profile: {unexpected}"
            )
        observed_profile = source
    else:
        observed_profile = profile(source, **profile_kwargs)

    regime = observed_profile.selected
    if regime is None:
        raise RuntimeError(
            "profile has no selected regime; inspect profile.report() and the "
            "individual regimes before requesting discovery."
        )
    return regime.mis_set
