# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""
isda.py - Maximal Independent Structural Dimensionality Analysis

MISDA is a graph-theoretic framework designed for dimensionality reduction in Multi-Objective Problems (MOPs). It identifies the Maximal Independent Set (MIS) of objectives within a data-driven dependency network. Unlike projection-based methods like PCA, which transform attributes into abstract components, MISDA analyzes the structural topology of the correlation graph to extract the largest possible subset of original features that are mutually independent. By mathematically maximizing this independent set, the algorithm recovers the problem's intrinsic dimensionality while ensuring that no redundant information is retained. This Python module implements the core functionality of MISDA. Refere to the documentation for further information.
"""

import numpy as np
import pandas as pd
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any

from ._statistics import (
    _CORRELATION_MODE,
    _correlation_strength,
    alpha_from_r,
    max_abs_corr,
    estimate_null_max_r,
    estimate_alpha_interval,
    select_alpha,
    AlphaRegime,
    diagnose_alpha_regime,
    describe_alpha_regime,
    calculate_spectral_entropy,
)
from ._graph import (
    find_maximal_independent_sets,
    calculate_component_compactness,
    repair_mis_coverage,
)
from ._ranking import (
    compute_mis_metrics,
    sort_mis_metrics,
)
from ._reconstruction import (
    _calculate_ses_core,
    calculate_ses_linear,
    calculate_ses,
    calculate_ses_nonlinear,
    evaluate_linear_reconstruction,
)
from ._heavy import heavy
from ._pareto import (
    evaluate_pareto_preservation,
    get_nondominated_mask,
    get_nondominated_mask_minimize,
    evaluate_pareto_consistency,
    evaluate_pareto_raw,
)
from ._validation import _validate_input_matrix, normalize_input_matrix
from ._metadata import __version__
from ._plotting import (
    _enforce_min_distance,
    _parse_node_to_1based,
    _extract_mis_nodes_1based,
    plot_custom_misda_graph,
)
from ._reporting import explain_ses
from .result import (
    AnalysisResult,
    ExecutionResult,
    LegacyMISCandidate,
    LegacyMISDAResult,
    MISCandidate,
    MISDAResult,
)
from .api import (
    report_significant_correlations,
    misda_significance_from_corr,
    misda_significance,
    _analyze_static_fast,
    _analyze_static,
    _analyze_static_v2,
)
from .benchmark import compile_benchmark_summary

# Constants
AGGRESSIVE = 0
MODERATE = 0.5
CONSERVATIVE = 1

# Utilities













# Stats / Alpha / Regime (reexported from misda._statistics)
# Graph / MIS (reexported from misda._graph)
# Ranking (reexported from misda._ranking)
# Core ISDA







# -------------------------------------------------------------------------
# MOP (Multi-Objective Pruning) - aka "Reduction" Helpers (for validation)
# -------------------------------------------------------------------------



# --------------------------------------------------------------------------------------
# HIGH-LEVEL API
# --------------------------------------------------------------------------------------




# --------------------------------------------------------------------------------------
# ADAPTIVE DATA STRUCTURES & ALGORITHM
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BootstrapObservation:
    repetition: int
    seed: int
    alpha_used: float
    selected: tuple
    dimension: int
    reduction_rate: float
    pareto_recall: float
    n_inbag: int
    n_oob: int


@dataclass
class OOBSummary:
    recall_mean: float
    recall_median: float
    recall_ci: tuple
    reduction_mean: float
    reduction_ci: tuple
    dimension_mean: float
    dimension_distribution: dict
    objective_frequencies: np.ndarray
    subset_stability: float
    valid_repetitions: int
    failed_repetitions: int
    observations: tuple


@dataclass
class AdaptiveCandidate:
    candidate_id: str
    alpha: float
    is_static: bool
    result: MISDAResult
    reduction_rate: float
    fitted_recall: float
    oob: Optional[OOBSummary] = None
    ses: Optional[dict] = None


@dataclass
class AdaptiveResult:
    static_candidate: AdaptiveCandidate
    candidates: tuple
    fitted_frontier: tuple
    validated_frontier: tuple
    recommended_candidate: str
    static_dominators: tuple
    dominated_candidates: tuple
    bootstrap_config: dict
    adaptive_config: dict

    @property
    def recommended(self) -> AdaptiveCandidate:
        """Returns the recommended AdaptiveCandidate object (knee point on validated frontier)."""
        for c in self.candidates:
            if c.candidate_id == self.recommended_candidate:
                return c
        return self.static_candidate

    def get_candidate(self, candidate_id: str) -> Optional[AdaptiveCandidate]:
        """Finds candidate by candidate_id."""
        for c in self.candidates:
            if c.candidate_id == candidate_id:
                return c
        return None

    def to_pandas(self) -> pd.DataFrame:
        """Exports comparative table of all candidates to pandas DataFrame."""
        rows = []
        for c in self.candidates:
            oob_rec = c.oob.recall_mean if c.oob else np.nan
            oob_red = c.oob.reduction_mean if c.oob else c.reduction_rate
            oob_stab = c.oob.subset_stability if c.oob else np.nan
            dim = c.result.best_mis.size if c.result.best_mis else c.result.Y.shape[1]
            rows.append({
                "candidate_id": c.candidate_id,
                "is_static": c.is_static,
                "alpha": c.alpha,
                "dimension": dim,
                "reduction_rate": c.reduction_rate,
                "fitted_recall": c.fitted_recall,
                "oob_reduction_mean": oob_red,
                "oob_recall_mean": oob_rec,
                "subset_stability": oob_stab,
                "in_fitted_frontier": c.candidate_id in self.fitted_frontier,
                "in_validated_frontier": c.candidate_id in self.validated_frontier,
                "is_recommended": c.candidate_id == self.recommended_candidate,
            })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Returns textual summary of Adaptive analysis."""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("                    MISDA ADAPTIVE ANALYSIS SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Total Candidates Analyzed: {len(self.candidates)}")
        lines.append(f"Recommended Candidate: {self.recommended_candidate}")
        rec = self.recommended
        dim = rec.result.best_mis.size if rec.result.best_mis else "Full"
        lines.append(f"  Alpha: {rec.alpha:.6g}")
        lines.append(f"  Dimension: {dim} (Fitted Reduction: {rec.reduction_rate:.2%})")
        lines.append(f"  Fitted Recall: {rec.fitted_recall:.4f}")
        if rec.oob:
            lines.append(f"  OOB Mean Reduction: {rec.oob.reduction_mean:.2%} (95% CI: [{rec.oob.reduction_ci[0]:.2%}, {rec.oob.reduction_ci[1]:.2%}])")
            lines.append(f"  OOB Mean Recall: {rec.oob.recall_mean:.4f} (95% CI: [{rec.oob.recall_ci[0]:.4f}, {rec.oob.recall_ci[1]:.4f}])")
            lines.append(f"  Subset Stability (Jaccard): {rec.oob.subset_stability:.4f}")
        lines.append(f"Fitted Frontier: {list(self.fitted_frontier)}")
        lines.append(f"Validated Frontier: {list(self.validated_frontier)}")
        if self.static_dominators:
            lines.append(f"Candidates Dominating Static Baseline: {list(self.static_dominators)}")
        else:
            lines.append("Candidates Dominating Static Baseline: None (Static is Non-Dominated)")
        return "\n".join(lines)

    def report(self) -> str:
        base_summary = self.summary()
        lines = [base_summary, "\n" + "=" * 70, "              RECOMMENDED CANDIDATE INSPECTION REPORT", "=" * 70]
        rec = self.recommended
        lines.append(rec.result.report())
        return "\n".join(lines)


def analyze(
    Y,
    method='static',
    caution=None,
    name=None,
    ensure_coverage=None,
    alpha=None,
    target_fidelity=None,
    max_iter=None,
    b_bootstrap=50,
    seed=123,
    aggressiveness=1.0,
    rank_policy="default",
    max_evaluated_mis=None,
):
    """
    Executes the MISDA pipeline on dataset Y.
    
    Strategies:
    - 'static' (Default): Uses the positive, data-driven structural pipeline.
    - 'adaptive': Searches discrete critical alpha levels for optimal Pareto reduction-recall trade-off with OOB bootstrap.
    """
    if method == 'static':
        if caution is not None:
            import warnings
            warnings.warn(
                "caution is deprecated; use aggressiveness.",
                DeprecationWarning,
                stacklevel=2,
            )
            if aggressiveness != 1.0 and aggressiveness != caution:
                raise ValueError(
                    "caution and aggressiveness specify conflicting values."
                )
            aggressiveness = caution
        if ensure_coverage is not None:
            import warnings
            warnings.warn(
                "ensure_coverage is deprecated and no longer alters MISs.",
                DeprecationWarning,
                stacklevel=2,
            )
        if alpha is not None:
            raise ValueError(
                "alpha is not supported by the refactored static method; "
                "use aggressiveness."
            )
        return _analyze_static_v2(
            Y,
            aggressiveness=aggressiveness,
            rank_policy=rank_policy,
            max_evaluated_mis=max_evaluated_mis,
            seed=seed,
            name=name,
        )
    elif method == 'adaptive':
        legacy_caution = 1.0 if caution is None else caution
        legacy_coverage = True if ensure_coverage is None else ensure_coverage
        return _analyze_adaptive(Y, caution=legacy_caution, b_bootstrap=b_bootstrap, seed=seed, name=name, ensure_coverage=legacy_coverage)
    else:
        raise ValueError(f"Unknown method '{method}'. Valid options: 'static', 'adaptive'")


def _generate_critical_alphas(corr, n_samples, alpha_static):
    """
    Generates discrete critical alpha levels corresponding to positive correlation thresholds
    in the range (alpha_static, 1.0].
    """
    M = corr.shape[0]
    iu = np.triu_indices(M, k=1)
    r_vals = _correlation_strength(corr[iu])

    alpha_events = []
    for r in r_vals:
        if r > 0:
            a = alpha_from_r(r, n_samples)
            alpha_events.append(a)

    valid_events = [a for a in alpha_events if a > alpha_static]
    valid_events = sorted(set(valid_events))

    critical_alphas = []
    for a in valid_events:
        a_next = float(np.nextafter(a, 1.0))
        if a_next <= 1.0 and a_next not in critical_alphas:
            critical_alphas.append(a_next)

    return sorted(critical_alphas)


def _compute_pareto_frontier_ids(candidates, x_func, y_func):
    """
    Computes non-dominated candidate IDs for two maximized objectives (x_func, y_func).
    """
    pts = []
    for c in candidates:
        pts.append((x_func(c), y_func(c), c.candidate_id))

    frontier = []
    for i, p_i in enumerate(pts):
        is_dominated = False
        for j, p_j in enumerate(pts):
            if i == j:
                continue
            if p_j[0] >= p_i[0] and p_j[1] >= p_i[1] and (p_j[0] > p_i[0] or p_j[1] > p_i[1]):
                is_dominated = True
                break
        if not is_dominated:
            frontier.append(p_i[2])

    frontier_cands = [next(c for c in candidates if c.candidate_id == cid) for cid in frontier]
    frontier_cands = sorted(frontier_cands, key=lambda c: (x_func(c), y_func(c)))
    return [c.candidate_id for c in frontier_cands]


def _select_knee_candidate_id(frontier_candidates):
    """
    Selects recommended candidate using maximum distance to chord line between extremes on validated OOB domain.
    """
    if not frontier_candidates:
        return "static"
    if len(frontier_candidates) <= 2:
        best = sorted(frontier_candidates, key=lambda c: (
            -(c.oob.recall_mean if c.oob else 0.0),
            -(c.oob.reduction_mean if c.oob else c.reduction_rate),
            c.alpha
        ))[0]
        return best.candidate_id

    cands = sorted(frontier_candidates, key=lambda c: (c.oob.reduction_mean if c.oob else c.reduction_rate))
    p_first = (
        cands[0].oob.reduction_mean if cands[0].oob else cands[0].reduction_rate,
        cands[0].oob.recall_mean if cands[0].oob else 0.0
    )
    p_last = (
        cands[-1].oob.reduction_mean if cands[-1].oob else cands[-1].reduction_rate,
        cands[-1].oob.recall_mean if cands[-1].oob else 0.0
    )

    x1, y1 = p_first
    x2, y2 = p_last
    denom = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)

    scored = []
    for c in cands:
        x0 = c.oob.reduction_mean if c.oob else c.reduction_rate
        y0 = c.oob.recall_mean if c.oob else 0.0
        if denom < 1e-12:
            dist = 0.0
        else:
            num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
            dist = num / denom

        scored.append((
            dist,
            c.oob.recall_mean if c.oob else 0.0,
            c.oob.reduction_mean if c.oob else c.reduction_rate,
            -c.alpha,
            c.candidate_id,
            c
        ))

    scored.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], item[4]))
    return scored[0][5].candidate_id




def _analyze_adaptive(
    Y,
    caution=1.0,
    b_bootstrap=50,
    seed=123,
    name=None,
    ensure_coverage=True,
    target_fidelity=None,
    max_iter=None
):
    """Internal implementation of adaptive search."""
    data, labels = _validate_input_matrix(Y)
    N, M = data.shape

    # 1. Statistical Baseline (Static run with user-supplied caution)
    alpha_min, alpha_max, r_max_real, r_null = estimate_alpha_interval(data)
    corr = np.corrcoef(data, rowvar=False)

    alpha_static = select_alpha(alpha_min, alpha_max, caution)

    res_static = _analyze_static_fast(
        Y, corr, alpha_min, alpha_max, alpha_static,
        caution=caution,
        name=f"{name}_static" if name else "static",
        ensure_coverage=ensure_coverage
    )

    mis_static = res_static.best_mis_indices
    subset_static = tuple(sorted(mis_static)) if mis_static else tuple(range(M))
    red_static = float(1.0 - len(subset_static) / M)
    prec_static, rec_static = evaluate_pareto_raw(data, subset_static)

    cand_static = AdaptiveCandidate(
        candidate_id="static",
        alpha=float(res_static.alpha),
        is_static=True,
        result=res_static,
        reduction_rate=red_static,
        fitted_recall=rec_static
    )

    # 2. Discrete Critical Level Generation
    critical_alphas = _generate_critical_alphas(corr, N, alpha_static=res_static.alpha)

    # 3. Candidate Generation and Subset Deduplication (Retain SMALLEST alpha per unique subset)
    subset_to_cand_info = {subset_static: (cand_static, float(res_static.alpha))}

    for a_crit in critical_alphas:
        res_k = _analyze_static_fast(
            Y, corr, alpha_min, alpha_max, a_crit,
            caution=caution,
            name=f"{name}_alpha_{a_crit:.4f}" if name else f"alpha_{a_crit:.4f}",
            ensure_coverage=ensure_coverage
        )
        mis_k = res_k.best_mis_indices
        subset_k = tuple(sorted(mis_k)) if mis_k else tuple(range(M))

        if subset_k not in subset_to_cand_info:
            red_k = float(1.0 - len(subset_k) / M)
            prec_k, rec_k = evaluate_pareto_raw(data, subset_k)
            cand_k = AdaptiveCandidate(
                candidate_id="", # placeholder
                alpha=float(a_crit),
                is_static=False,
                result=res_k,
                reduction_rate=red_k,
                fitted_recall=rec_k
            )
            subset_to_cand_info[subset_k] = (cand_k, float(a_crit))
        else:
            prev_cand, prev_alpha = subset_to_cand_info[subset_k]
            if not prev_cand.is_static and a_crit < prev_alpha:
                red_k = float(1.0 - len(subset_k) / M)
                prec_k, rec_k = evaluate_pareto_raw(data, subset_k)
                cand_k = AdaptiveCandidate(
                    candidate_id="",
                    alpha=float(a_crit),
                    is_static=False,
                    result=res_k,
                    reduction_rate=red_k,
                    fitted_recall=rec_k
                )
                subset_to_cand_info[subset_k] = (cand_k, float(a_crit))

    # Assign deterministic IDs
    all_candidates_list = [cand_static]
    non_static_cands = [cand for sub, (cand, a) in subset_to_cand_info.items() if not cand.is_static]
    non_static_cands.sort(key=lambda c: c.alpha)

    for idx, c in enumerate(non_static_cands, start=1):
        c.candidate_id = f"cand_{idx:03d}"
        all_candidates_list.append(c)

    candidates_dict = {c.candidate_id: c for c in all_candidates_list}

    # 4. Out-of-Bag (OOB) Bootstrap Validation (SHARED PAIRED RESAMPLES)
    rng_master = np.random.default_rng(seed)
    bootstrap_seeds = [int(s) for s in rng_master.integers(0, 10**9, size=b_bootstrap)]

    boot_splits = []
    for b in range(b_bootstrap):
        b_seed = bootstrap_seeds[b]
        rng_b = np.random.default_rng(b_seed)
        inbag_idx = rng_b.choice(N, size=N, replace=True)
        oob_mask = np.ones(N, dtype=bool)
        oob_mask[inbag_idx] = False
        oob_idx = np.where(oob_mask)[0]
        boot_splits.append((b_seed, inbag_idx, oob_idx))

    for cand in all_candidates_list:
        sub_full = tuple(sorted(cand.result.best_mis_indices)) if cand.result.best_mis else tuple(range(M))
        obs_list = []
        recalls_oob = []
        reductions_oob = []
        dims = []
        freq_counts = np.zeros(M, dtype=int)
        jaccard_sum = 0.0
        failed_reps = 0

        for b in range(b_bootstrap):
            b_seed, inbag_idx, oob_idx = boot_splits[b]

            if len(oob_idx) == 0:
                failed_reps += 1
                continue

            Y_inbag = data[inbag_idx, :]
            Y_oob = data[oob_idx, :]

            corr_inbag = np.corrcoef(Y_inbag, rowvar=False)

            if cand.is_static:
                a_min_b, a_max_b, _, _ = estimate_alpha_interval(Y_inbag)
                alpha_used_b = select_alpha(a_min_b, a_max_b, caution)
            else:
                alpha_used_b = cand.alpha

            res_b = misda_significance_from_corr(corr_inbag, len(inbag_idx), M, alpha_used_b, ensure_coverage=ensure_coverage)

            if res_b.get('mis_ranked'):
                mis_b = res_b['mis_ranked'][0]['mis_indices']
            else:
                mis_b = list(range(M))

            sub_b = tuple(sorted(mis_b))
            dim_b = len(sub_b)
            red_b = float(1.0 - dim_b / M)

            _, rec_oob = evaluate_pareto_raw(Y_oob, sub_b)

            for idx in sub_b:
                freq_counts[idx] += 1

            set_full = set(sub_full)
            set_b = set(sub_b)
            union_len = len(set_full.union(set_b))
            jacc = len(set_full.intersection(set_b)) / union_len if union_len > 0 else 1.0
            jaccard_sum += jacc

            obs = BootstrapObservation(
                repetition=b,
                seed=b_seed,
                alpha_used=float(alpha_used_b),
                selected=sub_b,
                dimension=dim_b,
                reduction_rate=red_b,
                pareto_recall=float(rec_oob),
                n_inbag=len(inbag_idx),
                n_oob=len(oob_idx)
            )
            obs_list.append(obs)
            recalls_oob.append(rec_oob)
            reductions_oob.append(red_b)
            dims.append(dim_b)

        valid_reps = len(obs_list)
        if valid_reps == 0:
            raise RuntimeError(f"Bootstrap validation failed for candidate '{cand.candidate_id}': 0 valid OOB samples found across {b_bootstrap} repetitions.")

        recalls_arr = np.asarray(recalls_oob, dtype=float)
        mean_rec = float(np.mean(recalls_arr))
        med_rec = float(np.median(recalls_arr))
        ci_rec_lower = float(np.percentile(recalls_arr, 2.5))
        ci_rec_upper = float(np.percentile(recalls_arr, 97.5))

        reductions_arr = np.asarray(reductions_oob, dtype=float)
        mean_red = float(np.mean(reductions_arr))
        ci_red_lower = float(np.percentile(reductions_arr, 2.5))
        ci_red_upper = float(np.percentile(reductions_arr, 97.5))

        dim_counts = {}
        for d in dims:
            dim_counts[d] = dim_counts.get(d, 0) + 1

        cand.oob = OOBSummary(
            recall_mean=mean_rec,
            recall_median=med_rec,
            recall_ci=(ci_rec_lower, ci_rec_upper),
            reduction_mean=mean_red,
            reduction_ci=(ci_red_lower, ci_red_upper),
            dimension_mean=float(np.mean(dims)),
            dimension_distribution=dim_counts,
            objective_frequencies=freq_counts / valid_reps,
            subset_stability=float(jaccard_sum / valid_reps),
            valid_repetitions=valid_reps,
            failed_repetitions=failed_reps,
            observations=tuple(obs_list)
        )

    # 5. Frontier Construction & Knee-Point Recommendation (STRICT OOB VALIDATED DOMAIN CONSISTENCY)
    fitted_frontier_ids = _compute_pareto_frontier_ids(
        all_candidates_list,
        x_func=lambda c: c.reduction_rate,
        y_func=lambda c: c.fitted_recall
    )

    validated_frontier_ids = _compute_pareto_frontier_ids(
        all_candidates_list,
        x_func=lambda c: c.oob.reduction_mean if c.oob else c.reduction_rate,
        y_func=lambda c: c.oob.recall_mean if c.oob else 0.0
    )

    val_candidates = [candidates_dict[cid] for cid in validated_frontier_ids]
    recommended_id = _select_knee_candidate_id(val_candidates)

    static_oob_rec = cand_static.oob.recall_mean if cand_static.oob else cand_static.fitted_recall
    static_oob_red = cand_static.oob.reduction_mean if cand_static.oob else cand_static.reduction_rate

    static_doms = []
    for c in all_candidates_list:
        if c.candidate_id == "static":
            continue
        c_oob_rec = c.oob.recall_mean if c.oob else c.fitted_recall
        c_oob_red = c.oob.reduction_mean if c.oob else c.reduction_rate
        if (c_oob_red >= static_oob_red and c_oob_rec >= static_oob_rec) and \
           (c_oob_red > static_oob_red or c_oob_rec > static_oob_rec):
            static_doms.append(c.candidate_id)

    dominated_ids = [c.candidate_id for c in all_candidates_list if c.candidate_id not in validated_frontier_ids]

    return AdaptiveResult(
        static_candidate=cand_static,
        candidates=tuple(all_candidates_list),
        fitted_frontier=tuple(fitted_frontier_ids),
        validated_frontier=tuple(validated_frontier_ids),
        recommended_candidate=recommended_id,
        static_dominators=tuple(static_doms),
        dominated_candidates=tuple(dominated_ids),
        bootstrap_config={"b_bootstrap": b_bootstrap, "seed": seed},
        adaptive_config={"name": name, "caution": caution, "ensure_coverage": ensure_coverage}
    )
