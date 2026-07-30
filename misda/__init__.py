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
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Any

__version__ = "0.4.1"

# Constants
AGGRESSIVE = 0
MODERATE = 0.5
CONSERVATIVE = 1

# Internal Correlation Mode Configuration
# "absolute": |r| — Structural dependence / latent dimension (reconstruction)
# "positive": max(r, 0) — Directional redundancy / Pareto conflict preservation (r < 0 preserved)
#_CORRELATION_MODE = "absolute"
_CORRELATION_MODE = "positive"


def _correlation_strength(r):
    """Calculates correlation strength based on internal _CORRELATION_MODE."""
    if _CORRELATION_MODE == "absolute":
        return np.abs(r)
    elif _CORRELATION_MODE == "positive":
        return np.maximum(r, 0.0)
    else:
        raise ValueError(f"Unknown _CORRELATION_MODE: {_CORRELATION_MODE}. Expected 'absolute' or 'positive'.")

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


# Stats / Alpha / Regime

def alpha_from_r(r, n):
    """
    Converts a correlation coefficient |r| to a two-tailed p-value (alpha).

    Args:
        r (float): The absolute value of the correlation coefficient.
        n (int): The number of samples.

    Returns:
        float: The two-tailed p-value (alpha).
    """
    r = float(abs(r))
    if r <= 0.0:
        return 1.0
    
    # Use survival function (sf = 1 - cdf) for better precision at tails
    # Handle r -> 1.0 case implicitly via large z, clamping p at the end
    if r >= 1.0 - 1e-15:
        # Avoid arctanh(1) singularity
        z_stat = np.inf
    else:
        z = np.arctanh(r)
        se = 1.0 / np.sqrt(n - 3)
        z_stat = z / se
        
    # Use sf instead of (1-cdf) to avoid precision loss near 0
    p = 2.0 * stats.norm.sf(abs(z_stat))
    
    # Clamp to machine epsilon to represent "extremely significant" rather than 0
    # This allows z_crit lookup to return a finite large number instead of inf
    min_float = np.finfo(float).tiny
    if p < min_float:
        p = min_float
        
    return float(p)

def max_abs_corr(Y):
    """
    Calculates the largest absolute correlation coefficient among columns of Y
    and returns the full correlation matrix.

    Args:
        Y (np.ndarray or pd.DataFrame): Input data.

    Returns:
        tuple: A tuple containing:
            - float: The maximum absolute correlation coefficient.
            - np.ndarray: The correlation matrix.
    """
    if hasattr(Y, "values"):
        data = np.asarray(Y.values, dtype=float)
    else:
        data = np.asarray(Y, dtype=float)
    n, m = data.shape
    corr = np.corrcoef(data, rowvar=False)
    iu = np.triu_indices(m, k=1)
    vals = _correlation_strength(corr[iu])
    r_max = float(vals.max()) if vals.size > 0 else 0.0
    return r_max, corr

def estimate_null_max_r(Y, B=500, random_state=None):
    """
    Estimates, via permutation, the largest absolute correlation coefficient
    expected under the null hypothesis (no correlation).

    Args:
        Y (np.ndarray or pd.DataFrame): Input data.
        B (int): Number of permutations to perform.
        random_state (int or np.random.Generator, optional): Seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - float: The maximum absolute correlation coefficient under the null hypothesis.
            - np.ndarray: Array of maximum absolute correlations from each permutation.
    """
    if hasattr(Y, "values"):
        data = np.asarray(Y.values, dtype=float)
    else:
        data = np.asarray(Y, dtype=float)
    n, m = data.shape
    rng = np.random.default_rng(random_state)
    max_nulls = []
    for _ in range(B):
        perm = np.empty_like(data)
        for j in range(m):
            perm[:, j] = rng.permutation(data[:, j])
        corr_perm = np.corrcoef(perm, rowvar=False)
        iu = np.triu_indices(m, k=1)
        max_nulls.append(_correlation_strength(corr_perm[iu]).max())
    max_nulls = np.asarray(max_nulls, dtype=float)
    r_max_null = float(max_nulls.max()) if max_nulls.size > 0 else 0.0
    return r_max_null, max_nulls

def estimate_alpha_interval(Y, B=500, random_state=0):
    """
    Estimates the (alpha_min, alpha_max) interval from the input data Y.
    alpha_min corresponds to the most significant observed correlation.
    alpha_max corresponds to the most significant correlation expected under the null.

    Args:
        Y (np.ndarray or pd.DataFrame): Input data.
        B (int): Number of permutations for null estimation.
        random_state (int, optional): Seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - float: alpha_min (p-value of the strongest real correlation).
            - float: alpha_max (p-value of the strongest null correlation).
            - float: r_max_real (strongest real correlation).
            - float: r_max_null (strongest null correlation).
    """
    if hasattr(Y, "values"):
        data = np.asarray(Y.values, dtype=float)
    else:
        data = np.asarray(Y, dtype=float)
    n, m = data.shape
    r_max_real, corr_real = max_abs_corr(data)
    r_max_null, null_samples = estimate_null_max_r(data, B=B, random_state=random_state)
    alpha_min = alpha_from_r(r_max_real, n)
    alpha_max = alpha_from_r(r_max_null, n)
    return alpha_min, alpha_max, r_max_real, r_max_null

def select_alpha(alpha_min: float, alpha_max: float, caution: float) -> float:
    """
    A caution of 1.0 (conservative) targets alpha_max (noise floor) to ensure structural
    integrity by identifying more potential dependencies. A caution of 0.0 (aggressive)
    targets alpha_min (signal floor), prioritizing statistical pureness over structure.

    Args:
        alpha_min (float): The minimum alpha value (most significant real correlation).
        alpha_max (float): The maximum alpha value (most significant null correlation).
        caution (float): A value between 0 and 1, indicating the level of caution.

    Returns:
        float: The selected alpha value.

    Raises:
        ValueError: If caution is not between 0 and 1.
    """
    if not (0 <= caution <= 1):
        raise ValueError("Caution must be between 0 and 1.")
    # Consistent mapping:
    # caution=1.0 -> DEFAULT/STABLE -> alpha_max (Noise floor)
    # caution=0.0 -> SIGNAL_ONLY   -> alpha_min (Signal floor)
    return alpha_min * (1 - caution) + alpha_max * caution


class AlphaRegime(IntEnum):
    SIGNAL_BELOW_NOISE   = 1  # α_min > α_max
    END_OF_SCALE       = 2  # α_min = 0, α_max = 0
    IMMEDIATE_SEPARATION = 3  # α_min = 0, α_max > 0
    LIMINAL_SEPARATION        = 4  # 0 < α_min ≤ α_max


def diagnose_alpha_regime(alpha_min: float, alpha_max: float):
    """
    Diagnoses the statistical regime based on alpha_min and alpha_max,
    and calculates related metrics like S and S_norm.

    Args:
        alpha_min (float): The minimum alpha value.
        alpha_max (float): The maximum alpha value.

    Returns:
        dict: A dictionary containing the regime, alpha values, S, and S_norm.
    """
    if alpha_min > alpha_max:
        regime = AlphaRegime.SIGNAL_BELOW_NOISE
        try:
            S = math.log(alpha_max / alpha_min)
        except ValueError:
            S = math.nan
        S_norm = math.nan

    elif alpha_min == 0 and alpha_max == 0:
        regime = AlphaRegime.END_OF_SCALE
        S = math.nan
        S_norm = math.nan

    elif alpha_min == 0 and alpha_max > 0:
        regime = AlphaRegime.IMMEDIATE_SEPARATION
        S = math.inf
        S_norm = math.nan

    else:
        # REGULAR: 0 < alpha_min <= alpha_max
        regime = AlphaRegime.LIMINAL_SEPARATION
        S = math.log(alpha_max / alpha_min)
        S_norm = S / math.log(1.0 / alpha_min)

    return {
        "regime": int(regime),
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "S": S,
        "S_norm": S_norm,
    }


def describe_alpha_regime(metrics: dict) -> str:
    """
    Generates a human-readable text report describing the diagnosed alpha regime.

    Args:
        metrics (dict): A dictionary containing regime diagnosis metrics
                        (output of `diagnose_alpha_regime`).

    Returns:
        str: A formatted string report of the statistical regime.
    """
    regime = AlphaRegime(int(metrics["regime"]))
    alpha_min = float(metrics["alpha_min"])
    alpha_max = float(metrics["alpha_max"])
    S = float(metrics["S"])
    S_norm = float(metrics["S_norm"])

    def _fmt(x):
        if math.isnan(x): return "N/A"
        if math.isinf(x): return "+inf" if x > 0 else "-inf"
        return f"{x:.6g}"

    def _fp_rate(a):
        if not (a > 0) or math.isnan(a) or math.isinf(a): return "N/A"
        return f"≈ 1 in {1.0/a:.6g}"

    def _log10(a):
        if not (a > 0) or math.isnan(a) or math.isinf(a): return math.nan
        return math.log10(a)

    if regime == AlphaRegime.SIGNAL_BELOW_NOISE:
        condition = "α_min > α_max"
        name = "SIGNAL BELOW NOISE"
        interpretation = "There is no statistical evidence of dependence."
        mis_action = "Do not reduce dimensionality."
        S_meaning = "S is negative due to inversion."
        S_norm_meaning = "N/A"
    elif regime == AlphaRegime.END_OF_SCALE:
        condition = "α_min = 0 and α_max = 0"
        name = "END OF SCALE"
        interpretation = "Criterion collapsed."
        mis_action = "Do not reduce dimensionality."
        S_meaning = "S is undefined."
        S_norm_meaning = "N/A"
    elif regime == AlphaRegime.IMMEDIATE_SEPARATION:
        condition = "α_min = 0 and α_max > 0"
        name = "IMMEDIATE SEPARATION"
        interpretation = "Dependencies are robust."
        mis_action = "Reduction allowed."
        S_meaning = "S diverges."
        S_norm_meaning = "N/A"
    else:
        condition = "0 < α_min ≤ α_max"
        name = "LIMINAL SEPARATION"
        interpretation = "Valid interval found."
        mis_action = "Reduction allowed."
        S_meaning = "S measures separability on log scale."
        S_norm_meaning = "S_norm measures fraction of potential gap."

    log10_min = _log10(alpha_min)
    log10_max = _log10(alpha_max)

    report = (
        f"\nCondition: {condition}\n"
        f"Statistical regime: {name} (id={int(regime)})\n\n"
        f"Interpretation: {interpretation}\n"
        f"Action on MIS: {mis_action}\n"
        f"Parameters:\n"
        f"  α_min = {_fmt(alpha_min)}  ({_fp_rate(alpha_min)});  log10(α_min) = {_fmt(log10_min)}\n"
        f"  α_max = {_fmt(alpha_max)}  ({_fp_rate(alpha_max)});  log10(α_max) = {_fmt(log10_max)}\n"
        f"Metrics:\n"
        f"  S = {_fmt(S)}  -> {S_meaning}\n"
        f"  S_norm = {_fmt(S_norm)}  -> {S_norm_meaning}\n"
    )
    return report

# Core ISDA

def find_maximal_independent_sets(adjacency):
    """
    Finds all Maximal Independent Sets (MIS) of a given graph represented by its adjacency matrix.
    Uses the Bron-Kerbosch algorithm.

    Args:
        adjacency (np.ndarray): The adjacency matrix of the graph.

    Returns:
        list: A list of lists, where each inner list represents an MIS (node indices).
    """
    adjacency = np.asarray(adjacency)
    M = adjacency.shape[0]

    # The Bron-Kerbosch algorithm typically works on the complement graph for MIS.
    # An independent set in G is a clique in G_complement.
    comp_adj = np.ones_like(adjacency, dtype=int)
    np.fill_diagonal(comp_adj, 0)
    comp_adj[adjacency == 1] = 0

    mis_list = []

    def neighbors_in_comp(v):
        """Returns neighbors of node v in the complement graph."""
        return {u for u in range(M) if comp_adj[v, u] == 1}

    def bron_kerbosch(R, P, X):
        """Recursive Bron-Kerbosch algorithm to find maximal cliques."""
        if not P and not X:
            mis_list.append(sorted(list(R)))
            return
        
        # Pivot selection: choose u in P union X with most neighbors in P
        u = None
        max_neighbors = -1
        for v_cand in P.union(X):
            num_neighbors = len(P.intersection(neighbors_in_comp(v_cand)))
            if num_neighbors > max_neighbors:
                max_neighbors = num_neighbors
                u = v_cand

        # Iterate over P \ N(u)
        for v in list(P.difference(neighbors_in_comp(u))):
            N_v = neighbors_in_comp(v)
            bron_kerbosch(R.union({v}), P.intersection(N_v), X.intersection(N_v))
            P.remove(v)
            X.add(v)

    bron_kerbosch(set(), set(range(M)), set())
    return mis_list


def compute_mis_metrics(mis_list, adjacency, labels):
    """
    Computes various metrics for each Maximal Independent Set (MIS).

    Args:
        mis_list (list): A list of MIS, where each MIS is a list of node indices.
        adjacency (np.ndarray): The adjacency matrix of the graph.
        labels (list): A list of labels for the nodes.

    Returns:
        list: A list of dictionaries, each containing metrics for an MIS.
    """
    A = np.array(adjacency, dtype=int)
    n = A.shape[0]
    results = []

    for S in mis_list:
        S = sorted(S)
        S_set = set(S)
        notS = [i for i in range(n) if i not in S_set]

        internal_deg = [sum(A[u, v] for v in S) for u in S]
        avg_internal = float(np.mean(internal_deg)) if internal_deg else 0.0

        ext_deg = [sum(A[u, v] for v in notS) for u in S]
        avg_ext = float(np.mean(ext_deg)) if ext_deg else 0.0

        ext_nodes = set()
        for u in S:
            for v in notS:
                if A[u, v] == 1:
                    ext_nodes.add(v)
        neighborhood = len(ext_nodes)
        
        remainder = max(1, len(notS))
        neighborhood_ratio = neighborhood / remainder
        span = int(sum(ext_deg))

        results.append({
            "mis_indices": S,
            "mis_labels": [labels[i] for i in S],
            "size": len(S),
            "neighborhood": neighborhood,
            "neighborhood_ratio": neighborhood_ratio,
            "span": span,
            "avg_external_degree": avg_ext,
            "avg_internal_degree": avg_internal,
        })
    return results


def sort_mis_metrics(mis_metrics):
    """
    Sorts a list of MIS metrics dictionaries based on a predefined ranking criteria.
    The primary sorting keys are: size (desc), neighborhood (desc), avg_external_degree (desc),
    span (desc), and mis_labels (asc) for tie-breaking.

    Args:
        mis_metrics (list): A list of dictionaries, each containing metrics for an MIS.

    Returns:
        list: The sorted list of MIS metrics dictionaries.
    """
    return sorted(
        mis_metrics,
        key=lambda x: (
            -x["size"],
            -x["neighborhood"],
            -x["avg_external_degree"],
            -x["span"],
            tuple(x["mis_labels"]),
        )
    )

def report_significant_correlations(R, z_stat, z_crit, max_pairs=50, label_prefix="f"):
    """
    Generates a string report of significant correlations found in the data.

    Args:
        R (np.ndarray): The correlation matrix.
        z_stat (np.ndarray): The Fisher z-transformed correlation statistics.
        z_crit (float): The critical z-value for significance.
        max_pairs (int): Maximum number of significant pairs to report for each type (positive/negative).
        label_prefix (str): Prefix for feature labels (e.g., "f" for f1, f2).

    Returns:
        str: A formatted string report of significant correlations.
    """
    M = R.shape[0]
    pos_corr = []
    neg_corr = []

    for i in range(M):
        for j in range(i + 1, M):
            if abs(z_stat[i, j]) > z_crit:
                rij = R[i, j]
                if rij > 0:
                    pos_corr.append((i, j, rij))
                elif rij < 0:
                    neg_corr.append((i, j, rij))
    
    out = []
    out.append("\n--- SIGNIFICANT CORRELATIONS (Fisher z, two-tailed) ---")

    if pos_corr:
        out.append("\nSignificant POSITIVE correlation:")
        for i, j, r in pos_corr[:max_pairs]:
            out.append(f"  {label_prefix}{i+1} – {label_prefix}{j+1}:  ρ = {r:.4f}")
        if len(pos_corr) > max_pairs:
            out.append(f"  ... ({len(pos_corr) - max_pairs} pairs omitted)")
    else:
        out.append("\nSignificant POSITIVE correlation: none")

    if neg_corr:
        out.append("\nSignificant NEGATIVE correlation:")
        for i, j, r in neg_corr[:max_pairs]:
            out.append(f"  {label_prefix}{i+1} – {label_prefix}{j+1}:  ρ = {r:.4f}")
        if len(neg_corr) > max_pairs:
            out.append(f"  ... ({len(neg_corr) - max_pairs} pairs omitted)")
    else:
        out.append("\nSignificant NEGATIVE correlation: none")
        
    return "\n".join(out)


def calculate_component_compactness(corr_matrix, components):
    """
    Calculates component homogeneity metrics (Compactness and Ratio) for each connected component.
    Compactness is the minimum absolute correlation within a component.
    Ratio is min_corr / max_corr within a component.

    Args:
        corr_matrix (np.ndarray): The full correlation matrix.
        components (list): A list of lists, where each inner list represents the
                           indices of nodes in a connected component.

    Returns:
        tuple: A tuple containing:
            - float: The lowest internal correlation (min_compactness) across all components.
            - dict: A dictionary mapping component index to its compactness (min internal correlation).
            - dict: A dictionary with 'min_ratio', 'worst_comp_idx', 'ratios', and 'details'.
    """
    metrics = {}
    ratios = {}
    details = {}
    min_compactness = 1.0
    min_ratio = 1.0
    worst_comp_idx = -1
    
    for idx, comp in enumerate(components):
        if len(comp) < 2:
            metrics[idx] = 1.0 
            ratios[idx] = 1.0
            details[idx] = {
                "min_r": 1.0,
                "max_r": 1.0,
                "ratio": 1.0,
            }
            continue
            
        # Extract submatrix
        sub_corr = corr_matrix[np.ix_(comp, comp)]
        sub_corr_abs = _correlation_strength(sub_corr)
        
        mask = np.ones_like(sub_corr_abs, dtype=bool)
        np.fill_diagonal(mask, False)
        off_diag = sub_corr_abs[mask]
        
        if len(off_diag) > 0:
            c_min = float(np.min(off_diag))
            c_max = float(np.max(off_diag))
            ratio = c_min / c_max if c_max > 0 else 0.0
        else:
            c_min, c_max, ratio = 1.0, 1.0, 1.0
            
        metrics[idx] = c_min
        ratios[idx] = ratio
        details[idx] = {
            "min_r": c_min,
            "max_r": c_max,
            "ratio": ratio,
        }
        
        if c_min < min_compactness:
            min_compactness = c_min
            
        if ratio < min_ratio:
            min_ratio = ratio
            worst_comp_idx = idx
            
    homogeneity_stats = {
        "min_ratio": min_ratio,
        "worst_comp_idx": worst_comp_idx,
        "ratios": ratios,
        "details": details,
    }
            
    return min_compactness, metrics, homogeneity_stats


def repair_mis_coverage(corr_matrix, mis_indices, min_coverage=0.7):
    """
    Iteratively repairs the MIS to ensure all variables are covered by at least one
    member of the MIS with correlation > min_coverage.
    
    Args:
        corr_matrix (np.ndarray): MxM correlation matrix.
        mis_indices (list): List of indices currently in the MIS.
        min_coverage (float): Minimum absolute correlation required to consider a variable 'covered'.
        
    Returns:
        list: List of indices in the repaired (expanded) MIS.
    """
    M = corr_matrix.shape[0]
    current_mis = list(mis_indices)
    
    # Identify orphans (variables not sufficiently covered by any current MIS member)
    while True:
        orphans = []
        # Calculate max coverage for each variable
        # We look at |Corr(i, m)| for all m in current_mis
        if not current_mis:
            # Should not happen in ISDA context, but handle gracefully
            orphans = list(range(M))
        else:
            mis_cols = corr_matrix[:, current_mis]
            max_corrs = np.max(_correlation_strength(mis_cols), axis=1) # (M,)
            
            # Find those below threshold
            orphans = np.where(max_corrs < min_coverage)[0]
            
        if len(orphans) == 0:
            break
            
        # Select the best candidate to cover orphans
        # Heuristic: Pick the orphan that is "most central" among the remaining orphans?
        # Or simplify: pick the first orphan?
        # Better: Pick the orphan that covers the most other orphans.
        
        best_candidate = -1
        best_cover_count = -1
        
        # Optimization: only check nodes within the orphan set as candidates
        # (Though a non-orphan could arguably cover them too, but non-orphans are already 'represented')
        subset_corr = _correlation_strength(corr_matrix[np.ix_(orphans, orphans)])
        
        # Count how many orphans each orphan covers
        coverage_counts = np.sum(subset_corr > min_coverage, axis=1)
        
        best_idx_local = np.argmax(coverage_counts)
        best_candidate = orphans[best_idx_local]
        
        current_mis.append(best_candidate)
        
    return sorted(current_mis)


def misda_significance_from_corr(corr, N, M, alpha=0.05, labels=None, ensure_coverage=True, min_coverage=None):
    """
    Executes MISDA logic given a precomputed correlation matrix.
    """
    if labels is None:
        labels = [f"f{i+1}" for i in range(M)]

    corr = np.clip(corr, -0.999999, 0.999999)

    z = 0.5 * np.log((1 + corr) / (1 - corr))
    sigma_z = 1 / np.sqrt(N - 3)
    z_stat = z / sigma_z

    z_crit = stats.norm.isf(alpha / 2)
    z_threshold = z_crit * sigma_z
    r_crit = np.tanh(z_threshold)
    if r_crit >= 0.99999:
        r_crit = 0.99999
    
    corr_report = report_significant_correlations(corr, z_stat, z_crit, label_prefix="f")

    signif = (_correlation_strength(z_stat) > z_crit)
    adjacency = signif.astype(int)
    np.fill_diagonal(adjacency, 0)

    visited = [False] * M
    components = []

    def dfs(start):
        stack = [start]
        comp = []
        while stack:
            i = stack.pop()
            if not visited[i]:
                visited[i] = True
                comp.append(i)
                neighbors = np.where(adjacency[i] == 1)[0]
                for j in neighbors:
                    if not visited[j]:
                        stack.append(j)
        return sorted(comp)

    for i in range(M):
        if not visited[i]:
            components.append(dfs(i))

    components_labels = [[labels[i] for i in comp] for comp in components]

    mis_sets = find_maximal_independent_sets(adjacency)
    
    if ensure_coverage:
        repaired_sets = []
        coverage_threshold = min_coverage if min_coverage is not None else r_crit
        for ms in mis_sets:
            repaired = repair_mis_coverage(corr, ms, min_coverage=coverage_threshold)
            if repaired not in repaired_sets:
                repaired_sets.append(repaired)
        mis_sets = repaired_sets

    mis_sets_labels = [[labels[i] for i in mis] for mis in mis_sets]

    mis_metrics = compute_mis_metrics(mis_sets, adjacency, labels)
    mis_sorted = sort_mis_metrics(mis_metrics)

    def rank_key(m):
        return (
            m["neighborhood"],
            m["span"],
            round(m["avg_external_degree"], 4),
            m["size"],
        )

    mis_ranked = []
    rank_groups = {}
    current_rank = 0
    prev_key = None

    for m in mis_sorted:
        k = rank_key(m)
        if prev_key is None or k != prev_key:
            current_rank += 1
            prev_key = k
        m_with_rank = dict(m)
        m_with_rank["rank"] = current_rank
        mis_ranked.append(m_with_rank)
        rank_groups.setdefault(current_rank, []).append(m_with_rank)

    best_mis_rank1 = rank_groups.get(1, [None])[0]
    best_mis_rank2 = rank_groups.get(2, [None])[0] if 2 in rank_groups else None

    unique_metric_values = {
        "neighborhood": sorted({m["neighborhood"] for m in mis_metrics}),
        "neighborhood_ratio": sorted({m["neighborhood_ratio"] for m in mis_metrics}),
        "span": sorted({m["span"] for m in mis_metrics}),
        "avg_external_degree": sorted({m["avg_external_degree"] for m in mis_metrics}),
        "avg_internal_degree": sorted({m["avg_internal_degree"] for m in mis_metrics}),
    }

    min_compactness, component_metrics, homogeneity_stats = calculate_component_compactness(corr, components)

    return {
        "corr": corr,
        "adjacency": adjacency,
        "components": components,
        "components_labels": components_labels,
        "mis_sets": mis_sets,
        "mis_sets_labels": mis_sets_labels,
        "mis_metrics": mis_metrics,
        "mis_sorted": mis_sorted,
        "mis_ranked": mis_ranked,
        "rank_groups": rank_groups,
        "best_mis_rank1": best_mis_rank1,
        "best_mis_rank2": best_mis_rank2,
        "unique_metric_values": unique_metric_values,
        "min_component_compactness": min_compactness,
        "component_compactness": component_metrics,
        "homogeneity_stats": homogeneity_stats,
        "labels": labels,
        "alpha": alpha,
        "N": N,
        "M": M,
        "sigma_z": sigma_z,
        "z_crit": z_crit,
        "corr_report": corr_report
    }


def misda_significance(Y, alpha=0.05, ensure_coverage=True, min_coverage=None):
    """
    Executes the Maximal Independent Structural Dimensionality Analysis (MISDA) logic.
    """
    if isinstance(Y, pd.DataFrame):
        X = Y.values
        labels = list(Y.columns)
    else:
        X = np.asarray(Y)
        M = X.shape[1]
        labels = [f"f{i+1}" for i in range(M)]

    N, M = X.shape
    corr = np.corrcoef(X, rowvar=False)
    return misda_significance_from_corr(corr, N, M, alpha=alpha, labels=labels, ensure_coverage=ensure_coverage, min_coverage=min_coverage)


# -------------------------------------------------------------------------
# MOP (Multi-Objective Pruning) - aka "Reduction" Helpers (for validation)
# -------------------------------------------------------------------------

def _calculate_ses_core(
    Y,
    mis,
    model_type="linear",
    n_perm=20,
    test_size=0.3,
    seed=123,
    clip=True,
    n_estimators=100,
):
    """
    Unified core engine for Linear (OLS) and Non-Linear (Random Forest) SES.
    Predicts ONLY eliminated targets T from kept predictors S.
    """
    if hasattr(Y, "values") and hasattr(Y, "columns"):
        cols = list(Y.columns)
        Ymat = np.asarray(Y.values, dtype=float)
        names = cols
    else:
        Ymat = np.asarray(Y, dtype=float)
        if Ymat.ndim != 2:
            raise ValueError("Y must be 2D matrix (N x M).")
        cols = None
        names = [f"f{i+1}" for i in range(Ymat.shape[1])]

    N, M = Ymat.shape
    if N < 2:
        raise ValueError("Y must have at least 2 samples.")
    if M < 1:
        raise ValueError("Y must have at least 1 feature.")

    # Process mis (kept indices / labels)
    if isinstance(mis, dict) and "mis_indices" in mis:
        mis_list = mis["mis_indices"]
    else:
        mis_list = mis

    if len(mis_list) == 0:
        raise ValueError("mis cannot be empty.")

    if cols is not None and isinstance(mis_list[0], str):
        S_idx = [cols.index(c) for c in mis_list]
    else:
        S_idx = list(map(int, mis_list))

    S_idx = sorted(set(S_idx))
    if any(i < 0 or i >= M for i in S_idx):
        raise ValueError("mis contains index outside of range [0, M).")

    # Validate parameters
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be strictly between 0 and 1.")
    if n_perm < 1:
        raise ValueError("n_perm must be at least 1.")

    T_idx = [j for j in range(M) if j not in S_idx]

    # Edge Case: No Reduction (All objectives kept: T is empty)
    if len(T_idx) == 0:
        return {
            "ses": None,
            "F_real": None,
            "F_null": None,
            "mis_size": len(S_idx),
            "M": int(M),
            "N": int(N),
            "targets_reconstructed": [],
            "r2_real": {},
            "r2_null": {},
            "status": "NO_REDUCTION",
            "model_type": model_type,
            "settings": {
                "n_perm": int(n_perm),
                "test_size": float(test_size),
                "seed": int(seed),
                "clip": bool(clip),
            },
        }

    if model_type == "nonlinear":
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            raise ImportError(
                "scikit-learn is required to calculate non-linear SES (RandomForestRegressor)."
            )

    # Train / Test split
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_test = int(np.round(test_size * N))
    n_test = min(max(n_test, 1), N - 1)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    def compute_F_and_r2dict(X_tr, X_te, base_seed):
        Y_tr = Ymat[train_idx, :][:, T_idx]
        Y_te = Ymat[test_idx, :][:, T_idx]

        if model_type == "linear":
            Xtr_b = np.column_stack([np.ones((X_tr.shape[0], 1)), X_tr])
            Xte_b = np.column_stack([np.ones((X_te.shape[0], 1)), X_te])
            beta, *_ = np.linalg.lstsq(Xtr_b, Y_tr, rcond=None)
            Y_hat = Xte_b @ beta
        elif model_type == "nonlinear":
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(
                n_estimators=n_estimators, random_state=base_seed, n_jobs=-1
            )
            Y_tr_fit = Y_tr.ravel() if Y_tr.shape[1] == 1 else Y_tr
            rf.fit(X_tr, Y_tr_fit)
            Y_hat = rf.predict(X_te)
            if Y_hat.ndim == 1:
                Y_hat = Y_hat[:, np.newaxis]
        else:
            raise ValueError(f"Unknown model_type '{model_type}'")

        r2 = {}
        vals = []
        for idx_k, j in enumerate(T_idx):
            y_test_j = Y_te[:, idx_k]
            y_hat_j = Y_hat[:, idx_k]
            ss_res = float(np.sum((y_test_j - y_hat_j) ** 2))
            y_mean = float(np.mean(y_test_j))
            ss_tot = float(np.sum((y_test_j - y_mean) ** 2))
            if ss_tot <= 1e-15:
                r2[names[j]] = None
            else:
                r2_j = float(1.0 - (ss_res / ss_tot))
                r2[names[j]] = r2_j
                vals.append(max(0.0, r2_j))

        if len(vals) == 0:
            return None, r2
        return float(np.mean(vals)), r2

    X_real = Ymat[:, S_idx]
    X_tr_real = X_real[train_idx, :]
    X_te_real = X_real[test_idx, :]
    F_real, r2_real = compute_F_and_r2dict(X_tr_real, X_te_real, seed)

    if F_real is None:
        return {
            "ses": None,
            "F_real": None,
            "F_null": None,
            "mis_size": len(S_idx),
            "M": int(M),
            "N": int(N),
            "targets_reconstructed": [names[j] for j in T_idx],
            "r2_real": r2_real,
            "r2_null": {},
            "status": "UNDEFINED_TARGETS",
            "model_type": model_type,
            "settings": {
                "n_perm": int(n_perm),
                "test_size": float(test_size),
                "seed": int(seed),
                "clip": bool(clip),
            },
        }

    # Permutation null model: permute within train and test independently
    r2_null_acc = {names[j]: [] for j in T_idx}
    F_null_vals = []

    for b in range(int(n_perm)):
        perm_seed_tr = seed + 1000 + b * 2
        perm_seed_te = seed + 1000 + b * 2 + 1
        rng_tr = np.random.default_rng(perm_seed_tr)
        rng_te = np.random.default_rng(perm_seed_te)

        # Joint row permutation: permute rows of S in block to preserve internal multivariate structure
        p_tr = rng_tr.permutation(len(train_idx))
        X_tr_perm = X_tr_real[p_tr, :].copy()

        p_te = rng_te.permutation(len(test_idx))
        X_te_perm = X_te_real[p_te, :].copy()

        b_seed = seed + 5000 + b * 100
        Fb, r2b = compute_F_and_r2dict(X_tr_perm, X_te_perm, b_seed)
        if Fb is not None:
            F_null_vals.append(Fb)
            for k, v in r2b.items():
                if v is not None:
                    r2_null_acc[k].append(v)

    if len(F_null_vals) > 0:
        F_null = float(np.mean(F_null_vals))
    else:
        F_null = 0.0

    r2_null = {
        k: (float(np.mean(vs)) if len(vs) > 0 else None)
        for k, vs in r2_null_acc.items()
    }

    denom = 1.0 - F_null
    if denom <= 0:
        ses = 0.0 if (F_real <= F_null) else 1.0
    else:
        ses = (F_real - F_null) / denom

    if clip:
        ses = float(np.clip(ses, 0.0, 1.0))
    else:
        ses = float(ses)

    return {
        "ses": ses,
        "F_real": float(F_real),
        "F_null": float(F_null),
        "mis_size": len(S_idx),
        "M": int(M),
        "N": int(N),
        "targets_reconstructed": [names[j] for j in T_idx],
        "r2_real": r2_real,
        "r2_null": r2_null,
        "status": "SUCCESS",
        "model_type": model_type,
        "settings": {
            "n_perm": int(n_perm),
            "test_size": float(test_size),
            "seed": int(seed),
            "clip": bool(clip),
        },
    }


def calculate_ses_linear(
    Y,
    mis,
    *,
    n_perm=20,
    test_size=0.3,
    seed=123,
    clip=True,
    return_details=True,
):
    """
    Calculates Linear SES (Structural Evidence Score) using OLS Linear Regression.
    """
    out = _calculate_ses_core(
        Y,
        mis,
        model_type="linear",
        n_perm=n_perm,
        test_size=test_size,
        seed=seed,
        clip=clip,
    )
    return out if return_details else out["ses"]


def calculate_ses(
    Y,
    mis,
    *,
    n_perm=20,
    test_size=0.3,
    seed=123,
    clip=True,
    return_details=True,
):
    """
    calculate_ses(Y, mis) -> ses (0..1) + details

    SES = Structural Evidence Score (Linear OLS).
    Alias for calculate_ses_linear.
    """
    return calculate_ses_linear(
        Y,
        mis,
        n_perm=n_perm,
        test_size=test_size,
        seed=seed,
        clip=clip,
        return_details=return_details,
    )


def calculate_ses_nonlinear(
    Y,
    mis,
    *,
    n_perm=20,
    test_size=0.3,
    seed=123,
    clip=True,
    n_estimators=100,
    return_details=False,
):
    """
    Calculates Non-Linear SES using Random Forest Regression.
    Returns scalar SES by default (or detailed dict if return_details=True).

    Args:
        Y: (N, M) matrix or DataFrame
        mis: list of indices/labels (or dict from result)
        n_perm: number of permutations for null model
        test_size: fraction for test set (default 0.3)
        seed: random seed for reproducibility
        clip: clip SES score to [0, 1]
        n_estimators: trees in RF
        return_details: if True, returns full dict instead of float

    Returns:
        float or None (if return_details=False) or dict (if return_details=True)
    """
    out = _calculate_ses_core(
        Y,
        mis,
        model_type="nonlinear",
        n_perm=n_perm,
        test_size=test_size,
        seed=seed,
        clip=clip,
        n_estimators=n_estimators,
    )
    return out if return_details else out["ses"]



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


def get_nondominated_mask(Y):
    """
    Returns boolean mask of non-dominated solutions (Minimization) for a dataset Y.
    Complexity: O(N^2)
    Args:
        Y (np.ndarray): shape (N, M)
    Returns:
        np.array(bool): shape (N,), True if non-dominated.
    """
    # Ensure numpy
    Y = np.asarray(Y)
    N, M = Y.shape
    is_efficient = np.ones(N, dtype=bool)
    for i in range(N):
        # i is dominated by j if:
        # all(Y[j] <= Y[i]) AND any(Y[j] < Y[i])
        better_or_equal = (Y <= Y[i]).all(axis=1)
        better = (Y < Y[i]).any(axis=1)
        dominators = better_or_equal & better
        if dominators.any():
            is_efficient[i] = False
    return is_efficient

def evaluate_pareto_consistency(result_obj, df_original=None):
    """
    Compares the True Pareto Front (Full M) vs Surrogate Pareto Front (Reduced k).
    Calculates Precision (Safety) and Recall (Coverage).

    Args:
        result_obj (MISDAResult): The result object from misda.analyze()
        df_original (pd.DataFrame or np.ndarray): Original data. If None, tries to use result_obj.Y
    
    Returns:
        (precision, recall): 
            Precision = P(True Optimum | Surrogate Optimum) -> Safety
            Recall    = P(Surrogate Optimum | True Optimum) -> Coverage
    """
    Y_full = df_original if df_original is not None else result_obj.Y
    if hasattr(Y_full, "values"):
        Y_full = Y_full.values
    Y_full = np.asarray(Y_full)

    mis = result_obj.best_mis
    if not mis or not mis.indices:
        return 0.0, 0.0
    
    indices = mis.indices
    Y_sub = Y_full[:, indices]
    
    # 1. True Front
    mask_true = get_nondominated_mask(Y_full)
    
    # 2. Surrogate Front
    mask_surr = get_nondominated_mask(Y_sub)
    
    # Metrics
    intersection = (mask_true & mask_surr).sum()
    
    # Precision: Of the points the surrogate thinks are optimal, how many are truly optimal?
    denom_p = mask_surr.sum()
    precision = intersection / denom_p if denom_p > 0 else 0.0
    
    return precision, recall


def evaluate_pareto_raw(Y, selected_indices, directions=None):
    """
    Evaluates Pareto precision and recall directly on raw objective matrix Y.
    Assumes minimization by default for all objectives unless directions specifies otherwise.

    Args:
        Y (np.ndarray or pd.DataFrame): Shape (N, M) matrix of objective values.
        selected_indices (sequence of int): Indices of selected/kept objectives.
        directions (sequence of int, optional): Objective optimization directions (+1 for max, -1 for min).

    Returns:
        tuple[float, float]: (precision, recall)
    """
    if hasattr(Y, "values"):
        data = np.asarray(Y.values, dtype=float)
    else:
        data = np.asarray(Y, dtype=float)

    N, M = data.shape
    if N == 0 or M == 0:
        return 0.0, 0.0

    if directions is not None:
        dirs = np.asarray(directions, dtype=float)
        Y_eval = data * (-dirs)
    else:
        Y_eval = data

    mask_true = get_nondominated_mask(Y_eval)
    sel = list(selected_indices)

    if len(sel) == M:
        return 1.0, 1.0

    if len(sel) == 0:
        return 0.0, 0.0

    Y_sub = Y_eval[:, sel]
    mask_surr = get_nondominated_mask(Y_sub)

    intersection = (mask_true & mask_surr).sum()

    denom_p = mask_surr.sum()
    precision = float(intersection / denom_p) if denom_p > 0 else 0.0

    denom_r = mask_true.sum()
    recall = float(intersection / denom_r) if denom_r > 0 else 0.0

    return precision, recall







# --------------------------------------------------------------------------------------
# HIGH-LEVEL API
# --------------------------------------------------------------------------------------

class MISCandidate:
    """
    Represents a single Maximum Independent Set (MIS) solution found by the algorithm.
    Wrapper around the internal dictionary to provide object-oriented access.
    """
    def __init__(self, data: dict):
        self._data = data

    @property
    def indices(self):
        """List of column indices corresponding to the selected variables."""
        return self._data.get('mis_indices', [])

    @property
    def labels(self):
        """List of variable names (column headers) of the selected variables."""
        return self._data.get('mis_labels', [])

    @property
    def rank(self):
        """Rank of this solution (1 = Best)."""
        return self._data.get('rank', 999)

    @property
    def size(self):
        """Number of variables in this solution."""
        return len(self.indices)

    @property
    def total_correlation(self):
        """Sum of internal pair-wise correlations (lower is better)."""
        return self._data.get('total_correlation', float('inf'))

    @property
    def max_correlation(self):
        """Maximum single pair-wise correlation within this set (lower is better)."""
        return self._data.get('max_correlation', float('inf'))
    
    def __repr__(self):
        return f"<MISCandidate: {self.labels} (Size={self.size}, Rank={self.rank})>"

class MISDAResult:
    """
    Encapsulates the complete result of an MISDA analysis.
    Stores input parameters, diagnostic regimes, execution results (MIS),
    and validation metrics (SES).
    """
    def __init__(self, Y, caution, alpha_min, alpha_max, metrics, regime, alpha_exec, isda_res, name=None):
        self.Y = Y
        self.name = name
        self.caution = caution
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.metrics = metrics
        self.regime = regime
        self.alpha = alpha_exec # effectively used alpha
        self.isda_results = isda_res
        self.validation_metrics = {}

    def validate(self, check_linear=True, check_nonlinear=True, check_pareto=True):
        """
        Runs post-hoc validation metrics.
        Args:
            check_linear (bool): Run calculate_ses (Linear Regression).
            check_nonlinear (bool): Run calculate_ses_nonlinear (Random Forest).
            check_pareto (bool): Run evaluate_pareto_consistency.
        """
        # Linear SES
        if check_linear:
             if self.best_mis:
                best_ids = self.best_mis.indices
                Y_val = self.Y.values if hasattr(self.Y, "values") else self.Y
                self.validation_metrics['linear'] = calculate_ses(Y_val, best_ids, return_details=True)
        
        # Non-Linear SES
        if check_nonlinear:
             if self.best_mis:
                best_ids = self.best_mis.indices
                # Safeguard: prevent massive RF runs on huge data unless explicit
                if self.Y.shape[0] <= 10000:
                    try:
                        self.validation_metrics['nonlinear'] = calculate_ses_nonlinear(self.Y, best_ids, return_details=True)
                    except ImportError:
                        pass
        
        # Pareto
        if check_pareto:
             if self.best_mis:
                try:
                    p, r = evaluate_pareto_consistency(self)
                    self.validation_metrics['pareto'] = (p, r)
                except Exception:
                    pass

    @property
    def ses_results(self):
        """Backward compatibility for linear SES results."""
        return self.validation_metrics.get('linear')

    @property
    def correlations(self):
        """Returns the correlation report string from the ISDA execution."""
        return self.isda_results.get('corr_report')

    @property
    def min_compactness(self):
        """Returns the minimum component compactness found (worst internal correlation)."""
        return self.isda_results.get('min_component_compactness', 1.0)
    
    @property
    def homogeneity_ratio(self):
        """Returns the global homogeneity ratio (worst Min/Max within a component)."""
        stats = self.isda_results.get('homogeneity_stats', {})
        return stats.get('min_ratio', 1.0)

    @property
    def mis_sets(self):
        """
        Returns a list of all MISCandidate objects found, sorted by rank.
        """
        raw_sets = self.isda_results.get('mis_ranked', [])
        return [MISCandidate(d) for d in raw_sets]
    
    @property
    def best_mis(self):
        """Returns the top-ranked MISCandidate or None."""
        if self.isda_results.get('mis_ranked'):
            return MISCandidate(self.isda_results['mis_ranked'][0])
        return None

    @property
    def best_mis_indices(self):
        """Returns the list of indices of the best MIS."""
        mis = self.best_mis
        return mis.indices if mis else []

    @property
    def best_mis_labels(self):
        """Returns the list of labels of the best MIS."""
        mis = self.best_mis
        return mis.labels if mis else []
    
    @property
    def ranked_mis_sets(self):
        """
        Returns a dictionary mapping rank (int) -> list of MISCandidate objects.
        """
        raw_groups = self.isda_results.get('rank_groups', {})
        return {r: [MISCandidate(d) for d in l] for r, l in raw_groups.items()}

    def get_mis_by_rank(self, rank):
        """
        Returns the list of MISCandidate objects for the specified rank.
        Returns empty list if rank not found.
        """
        return self.ranked_mis_sets.get(rank, [])
        
    @property
    def reduction_applied(self):
        """Boolean: True if dim(MIS) < dim(Y)."""
        mis = self.best_mis
        if mis:
            return mis.size < self.Y.shape[1]
        return False

    # --- Flattened Metrics ---
    @property
    def separation_score(self):
        """The Separation Score (S). Higher is better."""
        return float(self.metrics.get("S", float('nan')))

    @property
    def normalized_separation_score(self):
        """Normalized S-score (S_norm). Closer to 1.0 is better."""
        return float(self.metrics.get("S_norm", float('nan')))

    # --- Flattened Validation ---
    @property
    def ses_nonlinear_results(self):
        """Detailed Non-Linear SES dict result. None if not run."""
        val = self.validation_metrics.get('nonlinear')
        return val if isinstance(val, dict) else None

    @property
    def ses_nonlinear(self):
        """Non-Linear SES scalar metric score (float or None)."""
        val = self.validation_metrics.get('nonlinear')
        if isinstance(val, dict):
            return val.get('ses')
        return val
    
    @property
    def pareto_precision(self):
        """Pareto Precision (Safety). None if not run."""
        p_r = self.validation_metrics.get('pareto')
        return p_r[0] if p_r else None
        
    @property
    def pareto_recall(self):
        """Pareto Recall (Coverage). None if not run."""
        p_r = self.validation_metrics.get('pareto')
        return p_r[1] if p_r else None
    
    def to_pandas(self):
        """
        Exports all found independent sets to a pandas DataFrame.
        Columns: ['rank', 'size', 'max_corr', 'total_corr', 'labels', 'indices']
        """
        import pandas as pd
        data = []
        for m in self.mis_sets:
            data.append({
                'rank': m.rank,
                'size': m.size,
                'max_corr': m.max_correlation,
                'total_corr': m.total_correlation,
                'labels': m.labels,
                'indices': m.indices
            })
        if not data:
            return pd.DataFrame(columns=['rank', 'size', 'max_corr', 'total_corr', 'labels', 'indices'])
        return pd.DataFrame(data)

    def __repr__(self):
        n_start = self.Y.shape[1]
        n_end = self.best_mis.size if self.best_mis else "?"
        name_str = f"'{self.name}'" if self.name else "Untitled"
        return f"<MISDAResult: {name_str} (Dim {n_start}->{n_end}, Rank={self.best_mis.rank if self.best_mis else '?'})>"

    @property
    def diagnosis(self):
        """Returns a short diagnostic string based on Fidelity and Homogeneity."""
        if not self.reduction_applied:
            return "Valid (No Reduction Required)"

        f = None
        status = None
        if self.ses_results and isinstance(self.ses_results, dict):
            f = self.ses_results.get('F_real', None)
            status = self.ses_results.get('status', None)

        if status == "NO_REDUCTION":
            return "Valid (No Reduction Required)"
        
        h = self.homogeneity_ratio
        
        if f is None or math.isnan(f):
            return "Unvalidated (Missing SES)"

        comps = self.isda_results.get('components_labels', [])
        num_comps = len(comps)

        # Strict clique completeness check (min_compactness >= alpha)
        is_true_clique = (self.min_compactness >= self.alpha)

        # Heuristic Decision Tree
        if f >= 0.9 and h >= 0.8:
            if num_comps > 1:
                return "Ideal (Disjoint Cliques)" if is_true_clique else "Ideal (Multiple Components)"
            return "Ideal (Clique)" if is_true_clique else "Good (Robust)"
        if f >= 0.9 and h < 0.2:
             return "Entangled (Mixed)"
        if f >= 0.9:
             return "Good (Robust)"
             
        if f < 0.8 and h >= 0.6:
             return "Drift (Chain)"
             
        if f < 0.6 and h < 0.6:
             return "Fragmented (Bridge)"
             
        return "Ambiguous/Warn"

    @property
    def validation_status(self):
        """Returns string describing what has been validated."""
        validated = []
        if 'linear' in self.validation_metrics: validated.append("Linear")
        if 'nonlinear' in self.validation_metrics: validated.append("Non-Linear")
        if 'pareto' in self.validation_metrics: validated.append("Pareto")
        return ", ".join(validated) if validated else "None"

    def summary(self):
        """Returns a textual summary of the analysis."""
        lines = []
        lines.append("\n" + "" * 70)
        title = f"MISDA Analysis Summary: {self.name}" if self.name else "MISDA Analysis Summary"
        lines.append(title)
        lines.append("-" * 70)
        
        # Ground Truth / Inputs
        lines.append(f"Input: [N={self.Y.shape[0]}, M={self.Y.shape[1]}]")
        lines.append(f"Caution: {self.caution}")
        
        # Diagnosis
        lines.append("\n--- 1. Diagnosis ---")
        lines.append(describe_alpha_regime(self.metrics))
        lines.append(f"Regime: {self.regime.name}")
        lines.append(f"Validation: {self.validation_status}")
        
        # Decision
        lines.append("\n--- 2. Decision ---")
        if self.reduction_applied:
            lines.append("Action: Reduction APPLIED")
        else:
            lines.append("Action: Full Dimension Kept (No Reduction)")
        lines.append(f"Alpha Used: {self.alpha:.6g} (Range: [{self.alpha_min:.6g}, {self.alpha_max:.6g}])")
                         
        # Results
        lines.append("\n--- 3. Results ---")
        mis = self.best_mis
        if mis:
             lines.append(f"Best MIS Size: {mis.size}")
             lines.append(f"Best MIS Labels: {mis.labels}")
        else:
             lines.append("No independent set found (or execution failed).")
             
        # Quality
        lines.append("\n--- 4. Quality ---")
        ratio = self.homogeneity_ratio
        diag = self.diagnosis
        
        def _fmt_ratio(r):
            if math.isnan(r): return "N/A"
            return f"{r:.4f}"
            
        lines.append(f"Homogeneity Ratio: {_fmt_ratio(ratio)}")
        lines.append(f"Auto-Diagnosis: {diag}")
        
        if not math.isnan(ratio) and ratio < 0.6:
            lines.append("WARNING: Low homogeneity ratio (< 0.6). Possible over-reduction due to transitive chains or bridges.")
        else:
            lines.append("Status: OK (Components are internally homogeneous)")

        # Global Complexity Warning (Sphere Paradox)
        if self.reduction_applied:
            se_norm = calculate_spectral_entropy(self.Y)
            if se_norm > 0.75:
                # User-requested warning
                lines.append("WARNING: High global complexity detected (SE={:.2f}) despite aggressive reduction. Suspected Latent Conflict (Sphere-like topology).".format(se_norm))

        # SES (Linear)
        if 'linear' in self.validation_metrics:
             lines.append("\n--- 5. Validation (SES - Linear) ---")
             lines.append(explain_ses(self.validation_metrics['linear'], name=self.name))
        
        # SES (Non-Linear)
        if 'nonlinear' in self.validation_metrics:
             nl_res = self.validation_metrics['nonlinear']
             if isinstance(nl_res, dict):
                 ses_nl = nl_res.get('ses')
                 f_real_nl = nl_res.get('F_real')
                 nl_str = f"{ses_nl:.4f}" if ses_nl is not None else "N/A"
                 f_str = f"{f_real_nl:.4f}" if f_real_nl is not None else "N/A"
                 lines.append(f"Non-Linear SES (RF): {nl_str} (F_real = {f_str})")
             else:
                 nl_str = f"{nl_res:.4f}" if nl_res is not None else "N/A"
                 lines.append(f"Non-Linear SES (RF): {nl_str}")

        # Pareto Consistency
        if 'pareto' in self.validation_metrics:
             lines.append("\n--- 6. Pareto Consistency ---")
             prec, rec = self.validation_metrics['pareto']
             lines.append(f"Precision (Safety):   {prec:.4f}  (Prob. that Surrogate Optimum is True Optimum)")
             lines.append(f"Recall    (Coverage): {rec:.4f}  (Prob. that True Optimum is retained)")
             if prec < 1.0:
                 lines.append("WARN: Surrogate introduces false optima (Precision < 1.0).")
             if rec < 0.8:
                 lines.append("WARN: Surrogate misses significant portion of Pareto front (Recall < 0.8).")
         
        return "\n".join(lines)


    def report(self, top_k=5):
        """
        Returns a comprehensive technical report of the analysis.
        Combines the standard summary with deep inspection of internal state.
        
        Args:
            top_k (int): Number of candidates to show per rank (default: 5).
        """
        # Start with standard summary
        base_report = self.summary()
        
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("                    DETAILED INSPECTION REPORT")
        lines.append("=" * 70)
        
        # 1. Statistical Foundation
        lines.append("\n--- A. Statistical Foundation ---")
        lines.append(f"MISDA Version: {__version__}")
        lines.append(f"Sample Size (N): {self.isda_results.get('N')} | Objectives (M): {self.isda_results.get('M')}")
        lines.append(f"Fisher Transform Error (sigma_z): {self.isda_results.get('sigma_z', 0):.6f}")
        lines.append(f"Alpha Range: [min={self.alpha_min:.3g} (Signal), max={self.alpha_max:.3g} (Noise)]")
        lines.append(f"Caution setting: {self.caution:.2f}")
        lines.append(f"Effective Alpha: {self.alpha:.6g} (based on regime={self.regime.name})")
        lines.append(f"Critical Z-score: {self.isda_results.get('z_crit', 0):.4f}")
        
        # 2. Graph & Component Details
        lines.append("\n--- B. Graph Topology Details ---")
        comps = self.isda_results.get('components_labels', [])
        homog_stats = self.isda_results.get('homogeneity_stats', {})
        
        lines.append(f"Connected Components: {len(comps)}")
        for i, c in enumerate(comps):
            # Try to get specific stats for this component if available
            c_stat = homog_stats.get('details', {}).get(i, {})
            min_r = c_stat.get('min_r', float('nan'))
            max_r = c_stat.get('max_r', float('nan'))
            ratio = c_stat.get('ratio', float('nan'))
            
            def _fmt_nan(v, default="N/A"):
                if math.isnan(v): return default
                return f"{v:.4f}"
            
            status = "Tight" if (not math.isnan(ratio) and ratio > 0.8) else "Loose"
            lines.append(f"  C{i+1}: {c}")
            lines.append(f"      Internal Correlation: [{_fmt_nan(min_r)} ... {_fmt_nan(max_r)}] | Homogeneity: {_fmt_nan(ratio)} ({status})")

        # 3. Solution Space (All Candidates)
        lines.append("\n--- C. Solution Space (All Candidates) ---")
        rank_groups = self.ranked_mis_sets
        
        if not rank_groups:
             lines.append("  No solutions found.")
        
        for r in sorted(rank_groups.keys()):
            cands = rank_groups[r]
            n_cands = len(cands)
            lines.append(f"  Rank {r} ({n_cands} candidates):")
            
            # Smart Truncation
            show_cands = cands[:top_k]
            for c in show_cands:
                lines.append(f"    - {c.labels} (Size={c.size})")
                lines.append(f"      Criteria: TotalCorr={c.total_correlation:.4f} | MaxCorr={c.max_correlation:.4f}")
            
            if n_cands > top_k:
                lines.append(f"      ... (+ {n_cands - top_k} more candidates. Use `res.to_pandas()` to view all.)")

        # 4. Extended Verification
        lines.append("\n--- D. Verification Details ---")
        if self.ses_results:
            ses = self.ses_results
            f_real = ses.get('F_real')
            f_null = ses.get('F_null')
            s_val = ses.get('ses')
            
            def _fmt_v(val):
                return f"{val:.4f}" if val is not None else "N/A"

            lines.append("  Linear SES Breakdown:")
            lines.append(f"    Fidelity (Real): {_fmt_v(f_real)}")
            lines.append(f"    Fidelity (Null): {_fmt_v(f_null)}")
            lines.append(f"    Raw SES Score:   {_fmt_v(s_val)}")
        else:
            lines.append("  Linear SES: Not run (or failed).")

        return base_report + "\n".join(lines)

    def plot(self, show=True):
        """
        Plots the ISDA graph.
        
        Args:
            show (bool): If True, calls plt.show() to display the plot immediately.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        ret = plot_custom_misda_graph(
            self.isda_results,
            title=f"{self.name or 'MISDA'} — alpha={self.alpha:.3g} — regime={self.regime.name}",
            show_removed=False
        )
        fig = ret['fig']
        
        if show:
            import matplotlib.pyplot as plt
            plt.show()
            
        return fig


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


def _validate_input_matrix(Y):
    """
    Validates input matrix Y for finite values and constant objectives (zero range).
    Raises ValueError if data contains NaNs, Infs, or exact constant columns.
    """
    if hasattr(Y, "values"):
        data = np.asarray(Y.values, dtype=float)
        labels = list(Y.columns)
    else:
        data = np.asarray(Y, dtype=float)
        labels = [f"f{i+1}" for i in range(data.shape[1])]

    if not np.all(np.isfinite(data)):
        raise ValueError("Input data Y contains non-finite values (NaNs or Infs).")

    M = data.shape[1]
    for j in range(M):
        col = data[:, j]
        if np.ptp(col) == 0:
            raise ValueError(f"Objective '{labels[j]}' (column index {j}) has zero range (constant objective). Remove uninformative objectives before running MISDA.")
    return data, labels


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


def _analyze_static_fast(Y, corr, alpha_min, alpha_max, alpha_exec, caution=1.0, name=None, ensure_coverage=True):
    """
    Executes static analysis using precomputed correlation matrix and alpha bounds.
    """
    metrics = diagnose_alpha_regime(alpha_min, alpha_max)
    regime = AlphaRegime(metrics["regime"])
    
    if hasattr(Y, "values"):
        data = np.asarray(Y.values, dtype=float)
        labels = list(Y.columns)
    else:
        data = np.asarray(Y, dtype=float)
        labels = [f"f{i+1}" for i in range(data.shape[1])]
        
    N, M = data.shape
    res = misda_significance_from_corr(corr, N, M, alpha=alpha_exec, labels=labels, ensure_coverage=ensure_coverage)
    
    return MISDAResult(
        Y=Y,
        caution=caution,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        metrics=metrics,
        regime=regime,
        alpha_exec=alpha_exec,
        isda_res=res,
        name=name
    )


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


def _analyze_static(Y, caution=1.0, name=None, ensure_coverage=True, alpha=None):
    """
    Executes the full MISDA pipeline on dataset Y.
    """
    _validate_input_matrix(Y)
    alpha_min, alpha_max, r_max_real, r_null = estimate_alpha_interval(Y)
    metrics = diagnose_alpha_regime(alpha_min, alpha_max)
    regime = AlphaRegime(metrics["regime"])
    
    if alpha is not None:
        alpha_exec = alpha
    else:
        alpha_exec = select_alpha(alpha_min, alpha_max, caution)
    
    res = misda_significance(Y, alpha=alpha_exec, ensure_coverage=ensure_coverage, min_coverage=None)
    
    return MISDAResult(
        Y=Y,
        caution=caution,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        metrics=metrics,
        regime=regime,
        alpha_exec=alpha_exec,
        isda_res=res,
        name=name
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
