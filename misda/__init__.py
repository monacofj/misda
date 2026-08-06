# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""
isda.py - Maximal Independent Structural Dimensionality Analysis

MISDA is a graph-theoretic framework designed for dimensionality reduction in Multi-Objective Problems (MOPs). It identifies the Maximal Independent Set (MIS) of objectives within a data-driven dependency network. Unlike projection-based methods like PCA, which transform attributes into abstract components, MISDA analyzes the structural topology of the correlation graph to extract the largest possible subset of original features that are mutually independent. By mathematically maximizing this independent set, the algorithm recovers the problem's intrinsic dimensionality while ensuring that no redundant information is retained. This Python module implements the core functionality of MISDA. Refere to the documentation for further information.
"""

import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx
import matplotlib.pyplot as plt
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
)
from ._pareto import (
    get_nondominated_mask,
    evaluate_pareto_consistency,
    evaluate_pareto_raw,
)
from ._validation import _validate_input_matrix
from .result import MISCandidate, MISDAResult
from .api import (
    report_significant_correlations,
    misda_significance_from_corr,
    misda_significance,
    _analyze_static_fast,
    _analyze_static,
)

__version__ = "0.4.2"

# Constants
AGGRESSIVE = 0
MODERATE = 0.5
CONSERVATIVE = 1

# Utilities


def _enforce_min_distance(pos, min_dist=0.28, iters=900, jitter=1e-3, seed=7):
    """Adjusts 2D layout positions to enforce a minimum distance between nodes."""
    rng = np.random.default_rng(seed)
    nodes = list(pos.keys())
    if not nodes:
        return pos

    P = np.array([pos[n] for n in nodes], dtype=float)
    P += 1e-12 * rng.normal(size=P.shape)

    for _ in range(iters):
        moved = False
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                d = P[j] - P[i]
                dist = float(np.hypot(d[0], d[1]))
                if dist < 1e-12:
                    P[j] += rng.normal(scale=jitter, size=2)
                    moved = True
                elif dist < min_dist:
                    push = d / dist
                    delta = 0.5 * (min_dist - dist) * push
                    P[i] -= delta
                    P[j] += delta
                    moved = True
        if not moved:
            break
    return {n: P[k] for k, n in enumerate(nodes)}


def _parse_node_to_1based(x, M):
    """Accepts 0-based int, 1-based int, 'fK', and 'K'."""
    if isinstance(x, (int, np.integer)):
        xi = int(x)
        if 0 <= xi < M:
            return xi + 1
        if 1 <= xi <= M:
            return xi
        return None

    s = str(x).strip()
    if len(s) >= 2 and s[0] in ("f", "F"):
        s = s[1:]

    try:
        xi = int(s)
    except Exception:
        return None

    if 0 <= xi < M:
        return xi + 1
    if 1 <= xi <= M:
        return xi
    return None



def calculate_spectral_entropy(Y):
    """
    Calculates the normalized spectral entropy of the correlation matrix of Y.
    High entropy (~1.0) indicates complex, spherical, or random structure.
    Low entropy (~0.0) indicates high redundancy/dimensionality reduction potential.
    """
    if hasattr(Y, "values"):
        data = np.asarray(Y.values, dtype=float)
    else:
        data = np.asarray(Y, dtype=float)
    
    n, m = data.shape
    if m < 2:
        return 0.0
        
    # Correlation matrix
    corr = np.corrcoef(data, rowvar=False)
    # Eigenvalues (Hermitian/Symmetric)
    eigvals = np.linalg.eigvalsh(corr)
    
    # Normalize eigenvalues to probability distribution
    # Filter small negative/zeros due to precision
    eigvals = eigvals[eigvals > 1e-9]
    if len(eigvals) == 0:
        return 0.0
        
    p = eigvals / np.sum(eigvals)
    
    # Entropy
    se = -np.sum(p * np.log(p))
    
    # Normalize by log(M)
    # Note: Max entropy for M variables is log(M) when all eigenvalues = 1
    # However, number of non-zero eigenvalues could be < M if N < M.
    # Usually we norm by log(min(N, M)) or log(len(eigvals)).
    # Using log(len(eigvals)) is safer.
    denom = np.log(len(eigvals))
    if denom == 0:
        return 0.0
        
    return se / denom


def _extract_mis_nodes_1based(mis_entry, M):
    """
    Strict extractor (no random hunting):
      - mis_indices: list of ints (0-based or 1-based)
      - mis: list (ints/labels)
      - mis_nodes: list (ints/labels)
    Returns 1..M nodes (deduplicated, preserving order).
    """
    if not isinstance(mis_entry, dict):
        raise ValueError(f"mis_ranked item is not a dict: {type(mis_entry)}")

    raw = None
    for k in ("mis_indices", "mis", "mis_nodes"):
        if k in mis_entry and mis_entry[k] not in (None, [], ()):
            raw = mis_entry[k]
            break

    if raw is None:
        keys = sorted(mis_entry.keys())
        raise ValueError(
            "mis_ranked item does not contain MIS in any canonical key "
            "('mis_indices', 'mis', 'mis_nodes'). "
            f"Item keys: {keys}"
        )

    xs = list(raw) if isinstance(raw, (list, tuple, set)) else [raw]

    out, seen = [], set()
    for x in xs:
        u = _parse_node_to_1based(x, M)
        if u is None:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)

    return out


def plot_custom_misda_graph(
    results: dict,
    figsize=(10, 8),
    min_dist=0.5,
    title="ISDA Graph",
    show_removed=True
):
    """
    Plots the dependency graph derived from MISDA analysis.
    Nodes are objectives, edges are significant correlations.
    """
    M = results["M"]
    A = np.asarray(results.get("adjacency", None))
    if A is None:
        raise ValueError("results['adjacency'] missing.")

    # Use actual labels if provided in the results dict
    labels = results.get("labels")
    if labels is not None:
        nodes = list(labels)
        # Map node name to 1-based index (internal ISDA logic uses 1-based)
        node_to_idx = {name: i for i, name in enumerate(nodes)}
    else:
        nodes = list(range(1, M + 1))
        node_to_idx = {i: i-1 for i in nodes}

    # --- MIS: UNIQUE and explicit source (mis_ranked) ---
    mis_ranked = results.get("mis_ranked", None)
    if not isinstance(mis_ranked, list) or len(mis_ranked) == 0:
        raise ValueError(
            "results['mis_ranked'] missing/empty. Required to color MIS."
        )

    best_rank = min(m.get("rank", 10**9) for m in mis_ranked)
    best_mis_entry = next(
        m for m in mis_ranked if m.get("rank", 10**9) == best_rank
    )
    mis1_ids = _extract_mis_nodes_1based(best_mis_entry, M) # These are 1-based indices
    mis1 = [nodes[i-1] for i in mis1_ids] # Map to actual node names (could be strings)

    if len(mis1) == 0:
        keys = sorted(best_mis_entry.keys()) if isinstance(best_mis_entry, dict) else []
        raise ValueError(
            "Rank1 MIS came empty after canonical extraction. "
            "This means the pipeline is generating empty MIS (or with values outside 0..M-1 / 1..M). "
            f"rank1={best_rank}; rank1 item keys: {keys}"
        )

    mis1_set = set(mis1)

    # --- graph (nodes 1..M) ---
    G = nx.Graph()
    G.add_nodes_from(nodes)

    preserved_edges = []
    removed_edges = []
    for i in range(M):
        for j in range(i + 1, M):
            u_name = nodes[i]
            v_name = nodes[j]
            if A[i, j] != 0:
                preserved_edges.append((u_name, v_name))
                G.add_edge(u_name, v_name)
            else:
                removed_edges.append((u_name, v_name))

    density = nx.density(G)

    # layout + anti-overlap
    pos = nx.spring_layout(
        G,
        seed=7,
        k=3.0 / np.sqrt(max(M, 1)),  # Slightly larger k for more separation
        iterations=1000,             # More iterations for better convergence
        scale=1.0                    # Explicit scale to fill the plot area
    )
    pos = _enforce_min_distance(pos, min_dist=min_dist, iters=1200, seed=7)

    fig, ax = plt.subplots(figsize=figsize)

    # removed (subsample)
    if show_removed and removed_edges:
        draw_removed = removed_edges
        max_removed_edges = 350 # Hardcoded for now, was a parameter
        if max_removed_edges is not None and len(draw_removed) > max_removed_edges:
            step = max(1, len(draw_removed) // max_removed_edges)
            draw_removed = draw_removed[::step][:max_removed_edges]

        nx.draw_networkx_edges(
            G, pos,
            edgelist=draw_removed,
            style="dashed",
            edge_color="0.65",
            width=0.9, # removed_width was a parameter
            alpha=0.45,
            ax=ax,
        )

    # neighbors of Rank1 MIS
    neigh_set = set()
    for u_mis_idx in mis1_ids:
        # A is 0-based
        neighbor_indices = np.where(A[u_mis_idx - 1] != 0)[0]
        for k in neighbor_indices:
            neigh_set.add(nodes[k])
    
    mis1_set = set(mis1)
    neigh_set -= mis1_set

    # Separate edges for coloring
    green_edges = []
    other_preserved_edges = []

    for u, v in preserved_edges:
        if u in mis1_set or v in mis1_set:
            green_edges.append((u, v))
        else:
            other_preserved_edges.append((u, v))

    # Draw other preserved edges
    if other_preserved_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=other_preserved_edges,
            edge_color="0.10",
            width=1.15, # edge_width was a parameter
            alpha=0.85,
            ax=ax,
        )

    # Draw green edges
    if green_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=green_edges,
            edge_color="C2",  # Green
            width=1.15, # edge_width was a parameter
            alpha=0.95,
            ax=ax,
        )

    # nodes
    node_colors = []
    node_border_colors = []
    label_colors = []

    for u in nodes:
        if u in mis1_set:
            node_colors.append("C2")  # Green for Rank 1 MIS
            node_border_colors.append("k")
            label_colors.append("white")
        elif u in neigh_set:
            node_colors.append("k")  # Black for neighbors of Rank1 MIS
            node_border_colors.append("k")
            label_colors.append("white")
        else:
            node_colors.append("white") # Fallback for disconnected nodes if any
            node_border_colors.append("k")
            label_colors.append("black")


    nx.draw_networkx_nodes(
        G, pos,
        nodelist=nodes,
        node_size=420, # node_size was a parameter
        node_color=node_colors,
        edgecolors=node_border_colors,
        linewidths=1.2,
        ax=ax,
    )

    # Labels
    for k, u in enumerate(nodes):
        x, y = pos[u]
        current_label_color = "white" if (u in mis1_set or u in neigh_set) else "black"
        ax.text(
            x, y, str(u), ha="center", va="center", fontsize=9, color=current_label_color, zorder=10 # font_size was a parameter
        )

    if title is None:
        title = f"Graph — density={density:.2f} | Rank1 green | Neighbors black"
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()

    return {
        "mis_rank1_first": list(mis1),
        "neighbors_of_mis": sorted(neigh_set),
        "density": density,
        "n_preserved": len(preserved_edges),
        "n_removed": len(removed_edges),
        "rank1": best_rank,
        "fig": fig,
        "ax": ax,
    }


# Stats / Alpha / Regime (reexported from misda._statistics)
# Graph / MIS (reexported from misda._graph)
# Ranking (reexported from misda._ranking)
# Core ISDA







# -------------------------------------------------------------------------
# MOP (Multi-Objective Pruning) - aka "Reduction" Helpers (for validation)
# -------------------------------------------------------------------------

def explain_ses(out, top_k=8, name=None, show_all=False):
    """
    Explains the result of calculate_ses(out). Returns string report.
    SES = Structural Evidence Score.
    """
    if out is None or not isinstance(out, dict):
        return "explain_ses: 'out' is invalid (expected dict)."

    lines = []
    def _p(x): lines.append(str(x))

    title = f"Structural Evidence Score for {name}" if name else "Structural Evidence Score"
    _p("\n" + " " * 72)
    _p(title)
    _p("-" * 72)

    status = out.get("status", None)
    mis = out.get("mis_size", None)
    if mis is not None:
        _p(f"Surrogate size (mis): {mis}")

    if status == "NO_REDUCTION":
        _p("Status: NO_REDUCTION (All objectives kept; reconstruction N/A).")
        _p("SES = N/A  |  F_real = N/A  |  F_null = N/A")
        return "\n".join(lines)

    ses = out.get("ses", None)
    F_real = out.get("F_real", None)
    F_null = out.get("F_null", None)
    r2_by_target = out.get("r2_real", None)

    if ses is None or F_real is None or F_null is None:
        _p("Status: UNDEFINED or missing metrics ('ses', 'F_real', 'F_null').")
        _p("SES = N/A  |  F_real = N/A  |  F_null = N/A")
        return "\n".join(lines)

    gap = F_real - F_null
    denom = max(1e-15, (1.0 - F_null))
    ses_recalc = np.clip(gap / denom, 0.0, 1.0)

    _p(f"SES = {ses:.4f}  (recalc = {ses_recalc:.4f})")
    _p(f"F_real = {F_real:.4f}  |  F_null = {F_null:.4f}  |  gap = {gap:.4f}")
    _p("Operational interpretation (Structural Evidence Score):")
    _p("  - SES≈1: surrogate reconstructs others very well, far above null.")
    _p("  - SES≈0: surrogate does not reconstruct better than null; suspicious reduction.")
    _p("  - intermediate values: some reconstruction, but there is relevant loss.")

    if ses >= 0.9:
        _p("Short read: strong SES (reduction tends to be safe for reconstruction).")
    elif ses >= 0.7:
        _p("Short read: moderate SES (reduction may work, but deserves checking).")
    else:
        _p("Short read: low SES (high risk of surrogate being too small).")

    if isinstance(r2_by_target, dict) and len(r2_by_target) > 0:
        items = list(r2_by_target.items())
        items_sorted = sorted(items, key=lambda kv: (-(np.inf) if kv[1] is None else kv[1]))
        items_sorted = [(k, (-np.inf if v is None else float(v))) for k, v in items_sorted]
        items_sorted = sorted(items_sorted, key=lambda kv: kv[1])

        worst = items_sorted[:min(top_k, len(items_sorted))]
        best = items_sorted[-min(top_k, len(items_sorted)):] if len(items_sorted) > 1 else []

        def _fmt_r2(v):
            if v is None or np.isneginf(v):
                return "N/A"
            return f"{v:.4f}"

        _p("\nWorst targets (lowest R² in test):")
        for k, v in worst:
            _p(f"  {k}: R² = {_fmt_r2(v)}")

        if best:
            _p("\nBest targets (highest R² in test):")
            for k, v in reversed(best):
                _p(f"  {k}: R² = {_fmt_r2(v)}")

        if show_all:
            _p("\nR² by target (all):")
            for k, v in items_sorted:
                _p(f"  {k}: R² = {_fmt_r2(v)}")
    else:
        _p("\nR² by target is not available.")

    return "\n".join(lines)


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
    caution=1.0,
    name=None,
    ensure_coverage=True,
    alpha=None,
    target_fidelity=None,
    max_iter=None,
    b_bootstrap=50,
    seed=123
):
    """
    Executes the MISDA pipeline on dataset Y.
    
    Strategies:
    - 'static' (Default): Uses `caution` to pick a single `alpha`. Fast, standard.
    - 'adaptive': Searches discrete critical alpha levels for optimal Pareto reduction-recall trade-off with OOB bootstrap.
    """
    _validate_input_matrix(Y)

    if method == 'static':
        return _analyze_static(Y, caution=caution, name=name, ensure_coverage=ensure_coverage, alpha=alpha)
    elif method == 'adaptive':
        return _analyze_adaptive(Y, caution=caution, b_bootstrap=b_bootstrap, seed=seed, name=name, ensure_coverage=ensure_coverage)
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
