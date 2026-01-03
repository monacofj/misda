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

    nodes = list(range(1, M + 1))

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
    mis1 = _extract_mis_nodes_1based(best_mis_entry, M)

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
            if A[i, j] != 0:
                preserved_edges.append((i + 1, j + 1))
                G.add_edge(i + 1, j + 1) # Add edges to G for layout calculations
            else:
                removed_edges.append((i + 1, j + 1))

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
    for u_mis in mis1:
        neigh_set.update([k + 1 for k in np.where(A[u_mis - 1] != 0)[0].tolist()])
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
    if r >= 0.999999:
        return 1e-12
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_stat = z / se
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))
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
    vals = np.abs(corr[iu])
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
        max_nulls.append(np.abs(corr_perm[iu]).max())
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
    Selects an alpha value between alpha_min and alpha_max based on a caution level.
    A caution of 0 selects alpha_min (aggressive), 1 selects alpha_max (conservative).

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
        if math.isnan(x): return "NaN"
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
            - dict: A dictionary with 'min_ratio', 'worst_comp_idx', and 'ratios' (dict of comp_idx to ratio).
    """
    metrics = {}
    ratios = {}
    min_compactness = 1.0
    min_ratio = 1.0
    worst_comp_idx = -1
    
    for idx, comp in enumerate(components):
        if len(comp) < 2:
            metrics[idx] = 1.0 
            ratios[idx] = 1.0
            continue
            
        # Extract submatrix
        sub_corr = corr_matrix[np.ix_(comp, comp)]
        sub_corr_abs = np.abs(sub_corr)
        
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
        
        if c_min < min_compactness:
            min_compactness = c_min
            
        if ratio < min_ratio:
            min_ratio = ratio
            worst_comp_idx = idx
            
    homogeneity_stats = {
        "min_ratio": min_ratio,
        "worst_comp_idx": worst_comp_idx,
        "ratios": ratios
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
            max_corrs = np.max(np.abs(mis_cols), axis=1) # (M,)
            
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
        subset_corr = np.abs(corr_matrix[np.ix_(orphans, orphans)])
        
        # Count how many orphans each orphan covers
        coverage_counts = np.sum(subset_corr > min_coverage, axis=1)
        
        best_idx_local = np.argmax(coverage_counts)
        best_candidate = orphans[best_idx_local]
        
        current_mis.append(best_candidate)
        
    return sorted(current_mis)


def misda_significance(Y, alpha=0.05, ensure_coverage=True, min_coverage=None):
    """
    Executes the Maximal Independent Structural Dimensionality Analysis (MISDA) logic.
    It constructs a dependency graph based on significant correlations, finds Maximal
    Independent Sets (MIS), and computes various metrics for them.

    Args:
        Y (pd.DataFrame or np.ndarray): Input data, where rows are samples and columns are features.
        alpha (float): Significance level for determining significant correlations (e.g., 0.05).
        ensure_coverage (bool): If True, the MIS will be expanded to ensure all original
                                variables are 'covered' by at least one MIS member.
        min_coverage (float, optional): Minimum absolute correlation required for a variable
                                        to be considered 'covered' by an MIS member.
                                        If None, it's derived from the `alpha` parameter.

    Returns:
        dict: A dictionary containing the results of the MISDA analysis, including:
            - 'corr': The full correlation matrix.
            - 'adjacency': The adjacency matrix of the dependency graph.
            - 'components': List of connected components (node indices).
            - 'components_labels': List of connected components (node labels).
            - 'mis_sets': All found Maximal Independent Sets (node indices).
            - 'mis_sets_labels': All found Maximal Independent Sets (node labels).
            - 'mis_metrics': Detailed metrics for each MIS.
            - 'mis_sorted': MIS metrics sorted by ranking criteria.
            - 'mis_ranked': MIS metrics with assigned ranks.
            - 'rank_groups': MIS grouped by rank.
            - 'best_mis_rank1': The top-ranked MIS.
            - 'best_mis_rank2': The second-ranked MIS (if available).
            - 'unique_metric_values': Unique values for various MIS metrics.
            - 'min_component_compactness': The minimum compactness across all components.
            - 'component_compactness': Compactness for each component.
            - 'homogeneity_stats': Statistics related to component homogeneity.
            - 'labels': List of feature labels.
            - 'alpha': The significance level used.
            - 'N': Number of samples.
            - 'M': Number of features.
            - 'sigma_z': Standard error of Fisher's z-transform.
            - 'z_crit': Critical z-value for the given alpha.
            - 'corr_report': A string report of significant correlations.
    """
    # Accepts DataFrame or array
    if isinstance(Y, pd.DataFrame):
        X = Y.values
        labels = list(Y.columns)
    else:
        X = np.asarray(Y)
        M = X.shape[1]
        labels = [f"f{i+1}" for i in range(M)]

    N, M = X.shape

    corr = np.corrcoef(X, rowvar=False)
    corr = np.clip(corr, -0.999999, 0.999999)

    z = 0.5 * np.log((1 + corr) / (1 - corr))
    sigma_z = 1 / np.sqrt(N - 3)
    z_stat = z / sigma_z
    z_crit = stats.norm.ppf(1 - alpha / 2)
    
    # Calculate correlation threshold corresponding to z_crit
    # z = z_crit * sigma_z
    # r = tanh(z)
    z_threshold = z_crit * sigma_z
    r_crit = np.tanh(z_threshold)
    
    corr_report = report_significant_correlations(corr, z_stat, z_crit, label_prefix="f")

    signif = (z_stat > z_crit)
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
        # Use r_crit as the coverage threshold to ensure consistency with graph construction
        coverage_threshold = min_coverage if min_coverage is not None else r_crit
        
        for ms in mis_sets:
            # repair returns sorted list
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

    results = {
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

    return results


# -------------------------------------------------------------------------
# MOP (Multi-Objective Pruning) - aka "Reduction" Helpers (for validation)
# -------------------------------------------------------------------------

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

    SES = Structural Evidence Score.
    """

    if hasattr(Y, "values") and hasattr(Y, "columns"):
        cols = list(Y.columns)
        Ymat = np.asarray(Y.values, dtype=float)
        if len(mis) == 0:
            raise ValueError("mis cannot be empty.")
        if isinstance(mis[0], str):
            S_idx = [cols.index(c) for c in mis]
        else:
            S_idx = list(map(int, mis))
        M = Ymat.shape[1]
        names = cols
    else:
        Ymat = np.asarray(Y, dtype=float)
        if Ymat.ndim != 2:
            raise ValueError("Y must be 2D (NxM).")
        if len(mis) == 0:
            raise ValueError("mis cannot be empty.")
        S_idx = list(map(int, mis))
        M = Ymat.shape[1]
        names = [f"f{i+1}" for i in range(M)]

    N = Ymat.shape[0]
    S_idx = sorted(set(S_idx))
    if any(i < 0 or i >= M for i in S_idx):
        raise ValueError("mis contains index outside of [0, M).")

    T_idx = [j for j in range(M) if j not in S_idx]
    if len(T_idx) == 0:
        out = {"ses": 1.0, "F_real": 1.0, "F_null": 0.0, "r2_real": {}, "r2_null": {}}
        return out if return_details else 1.0

    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_test = int(np.round(test_size * N))
    n_test = min(max(n_test, 1), N - 1)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    def fit_predict_r2(X_train, y_train, X_test, y_test):
        Xtr = np.column_stack([np.ones((X_train.shape[0], 1)), X_train])
        Xte = np.column_stack([np.ones((X_test.shape[0], 1)), X_test])
        beta, *_ = np.linalg.lstsq(Xtr, y_train, rcond=None)
        y_hat = Xte @ beta
        ss_res = np.sum((y_test - y_hat) ** 2)
        y_mean = np.mean(y_test)
        ss_tot = np.sum((y_test - y_mean) ** 2)
        if ss_tot <= 0:
            return 0.0
        return 1.0 - (ss_res / ss_tot)

    def compute_F_and_r2dict(X):
        X_train = X[train_idx, :]
        X_test = X[test_idx, :]
        r2 = {}
        vals = []
        for j in T_idx:
            y_train = Ymat[train_idx, j]
            y_test = Ymat[test_idx, j]
            r2_j = fit_predict_r2(X_train, y_train, X_test, y_test)
            r2[names[j]] = float(r2_j)
            vals.append(max(0.0, r2_j))
        return float(np.mean(vals)), r2

    X_real = Ymat[:, S_idx]
    F_real, r2_real = compute_F_and_r2dict(X_real)

    r2_null_acc = {names[j]: [] for j in T_idx}
    F_null_vals = []
    for b in range(int(n_perm)):
        Xn = X_real.copy()
        for c in range(Xn.shape[1]):
            perm = rng.permutation(N)
            Xn[:, c] = Xn[perm, c]
        Fb, r2b = compute_F_and_r2dict(Xn)
        F_null_vals.append(Fb)
        for k, v in r2b.items():
            r2_null_acc[k].append(v)

    F_null = float(np.mean(F_null_vals))
    r2_null = {k: float(np.mean(vs)) for k, vs in r2_null_acc.items()}

    denom = (1.0 - F_null)
    if denom <= 0:
        ses = 0.0 if (F_real <= F_null) else 1.0
    else:
        ses = (F_real - F_null) / denom
    if clip:
        ses = float(np.clip(ses, 0.0, 1.0))
    else:
        ses = float(ses)

    out = {
        "ses": ses,
        "F_real": float(F_real),
        "F_null": float(F_null),
        "mis_size": len(S_idx),
        "M": int(M),
        "N": int(N),
        "targets_reconstructed": [names[j] for j in T_idx],
        "r2_real": r2_real,
        "r2_null": r2_null,
        "settings": {
            "n_perm": int(n_perm),
            "test_size": float(test_size),
            "seed": int(seed),
            "clip": bool(clip),
        },
    }
    return out if return_details else out["ses"]


def calculate_ses_nonlinear(
    Y,
    mis,
    *,
    test_size=0.3,
    seed=123,
    n_estimators=100
):
    """
    Calculates Non-Linear SES using Random Forest Regression.
    Returns the R2 score of reconstructing Y from Y[:, mis].
    
    Args:
        Y: (N, M) matrix
        mis: list of indices (or dict from result)
        test_size: fraction for validation (default 0.3)
        n_estimators: trees for RF
        
    Returns:
        float: R2 score (0..1)
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("Warning: sklearn not found. calculate_ses_nonlinear returning 0.0.")
        return 0.0

    # Handle input types
    Y = np.asarray(Y)
    if isinstance(mis, dict) and 'mis_indices' in mis:
        indices = mis['mis_indices']
    else:
        indices = mis
        
    if not len(indices):
        return 0.0
        
    M = Y.shape[1]
    # Identify target columns (all, or just dependent ones? SES reconstructs ALL usually or remaining?)
    # Linear SES validates F_real: R2 of Y ~ Y_mis. 
    # Usually we want to know if Y_mis predicts Y (especially Y_dependent).
    # Since Y_mis predicts itself perfectly (R2=1), the score is dominated by dependents.
    # We will predict ALL M columns to match F_real logic (which is avg R2).
    
    Y_mis = Y[:, indices]
    
    X_train, X_test, Y_train, Y_test = train_test_split(Y_mis, Y, test_size=test_size, random_state=seed)
    
    reg = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    reg.fit(X_train, Y_train)
    
    score = reg.score(X_test, Y_test)
    return max(0.0, score)



def explain_ses(out, top_k=8, name=None, show_all=False):
    """
    Explains the result of calculate_ses(out). Returns string report.
    SES = Structural Evidence Score.
    """
    if out is None or not isinstance(out, dict):
        return "explain_ses: 'out' is invalid (expected dict)."

    lines = []
    def _p(x): lines.append(str(x))

    ses = out.get("ses", None)
    F_real = out.get("F_real", None)
    F_null = out.get("F_null", None)
    r2_by_target = out.get("r2_real", None) # Corrected key
    mis = out.get("mis_size", None)

    title = f"Structural Evidence Score for {name}" if name else "Structural Evidence Score"
    _p("\n" + " " * 72)
    _p(title)
    _p("-" * 72)

    if mis is not None:
        _p(f"Surrogate size (mis): {mis}")

    if ses is None or F_real is None or F_null is None:
        _p("Output does not contain expected keys ('ses', 'F_real', 'F_null').")
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

        _p("\nWorst targets (lowest R² in test):")
        for k, v in worst:
            _p(f"  {k}: R² = {v:.4f}")

        if best:
            _p("\nBest targets (highest R² in test):")
            for k, v in reversed(best):
                _p(f"  {k}: R² = {v:.4f}")

        if show_all:
            _p("\nR² by target (all):")
            for k, v in items_sorted:
                _p(f"  {k}: R² = {v:.4f}")
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
    if not mis or not mis['mis_indices']:
        return 0.0, 0.0
    
    indices = mis['mis_indices']
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
    
    # Recall: Of the true optima, how many did the surrogate keep?
    denom_r = mask_true.sum()
    recall = intersection / denom_r if denom_r > 0 else 0.0
    
    return precision, recall







# --------------------------------------------------------------------------------------
# HIGH-LEVEL API
# --------------------------------------------------------------------------------------

class MISDAResult:
    """
    Encapsulates the complete result of an MISDA analysis.
    Stores input parameters, diagnostic regimes, execution results (MIS),
    and validation metrics (SES).
    """
    def __init__(self, Y, caution, alpha_min, alpha_max, metrics, regime, alpha_exec, isda_res, ses_results=None, name=None):
        self.Y = Y
        self.name = name
        self.caution = caution
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.metrics = metrics
        self.regime = regime
        self.alpha = alpha_exec # effectively used alpha
        self.isda_results = isda_res
        self.ses_results = ses_results

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
        """Returns the list of found MIS sets."""
        return self.isda_results.get('mis_sets')
    
    @property
    def best_mis(self):
        """Returns the top-ranked MIS (Minimal Independent Set) or None."""
        if self.isda_results.get('mis_ranked'):
            return self.isda_results['mis_ranked'][0]
        return None
        
    @property
    def reduction_applied(self):
        """Boolean: True if dim(MIS) < dim(Y)."""
        mis = self.best_mis
        if mis:
            return len(mis['mis_indices']) < self.Y.shape[1]
        return False

    @property
    def diagnosis(self):
        """Returns a short diagnostic string based on Fidelity and Homogeneity."""
        f = 0.0
        if self.ses_results:
            f = self.ses_results.get('F_real', 0.0)
        
        h = self.homogeneity_ratio
        
        # Heuristic Decision Tree
        if f >= 0.9 and h >= 0.8:
            return "Ideal (Clique)"
        if f >= 0.9 and h < 0.2:
             return "Entangled (Mixed)"
        if f >= 0.9:
             return "Good (Robust)"
             
        if f < 0.8 and h >= 0.6:
             return "Drift (Chain)"
             
        if f < 0.6 and h < 0.6:
             return "Fragmented (Bridge)"
             
        return "Ambiguous/Warn"

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
             lines.append(f"Best MIS Size: {len(mis['mis_indices'])}")
             lines.append(f"Best MIS Labels: {mis['mis_labels']}")
        else:
             lines.append("No independent set found (or execution failed).")
             
        # Quality
        lines.append("\n--- 4. Quality ---")
        ratio = self.homogeneity_ratio
        diag = self.diagnosis
        
        lines.append(f"Homogeneity Ratio: {ratio:.4f}")
        lines.append(f"Auto-Diagnosis: {diag}")
        
        if ratio < 0.6:
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
        if self.ses_results:
             lines.append("\n--- 5. Validation (SES - Linear) ---")
             lines.append(explain_ses(self.ses_results, name=self.name))
        
        # SES (Non-Linear)
        if self.Y.shape[0] <= 2000: # Limit for RF performance
             try:
                 ses_nl = calculate_ses_nonlinear(self.Y, self.best_mis)
                 lines.append(f"Non-Linear SES (RF): {ses_nl:.4f} (R2 Score)")
             except Exception:
                 pass

        # Pareto Consistency
        # Only run if N is reasonable to avoid O(N^2) lag on massive datasets
        if self.Y.shape[0] <= 5000:
             lines.append("\n--- 6. Pareto Consistency ---")
             try:
                 prec, rec = evaluate_pareto_consistency(self)
                 lines.append(f"Precision (Safety):   {prec:.4f}  (Prob. that Surrogate Optimum is True Optimum)")
                 lines.append(f"Recall    (Coverage): {rec:.4f}  (Prob. that True Optimum is retained)")
                 if prec < 1.0:
                     lines.append("WARN: Surrogate introduces false optima (Precision < 1.0).")
                 if rec < 0.8:
                     lines.append("WARN: Surrogate misses significant portion of Pareto front (Recall < 0.8).")
             except Exception:
                 lines.append("Error calculating Pareto metrics.")
         
        return "\n".join(lines)

    def plot(self):
        """Returns the matplotlib figure of the ISDA graph."""
        return plot_custom_misda_graph(
            self.isda_results,
            title=f"{self.name or 'MISDA'} — alpha={self.alpha:.3g} — regime={self.regime.name}",
            show_removed=False
        )

def analyze(Y, caution=0.5, run_ses=True, name=None, ensure_coverage=True):
    """
    Executes the full MISDA pipeline on dataset Y.
    
    Steps:
    1. Estimate alpha interval (p=0.01 vs p=0.05).
    2. Diagnose alpha regime (Separation, Mixed, etc.).
    3. Select execution alpha based on 'caution'.
    4. Run ISDA clustering algorithm.
    5. (Optional) Run SES validation on the best MIS.
    
    Args:
        Y (np.ndarray or pd.DataFrame): Input data (N samples x M features).
        caution (float): Conservatism level [0, 1]. 0 = Aggressive reduction, 1 = Very conservative.
        run_ses (bool): If True, calculates Structural Evidence Score for the best MIS.
        name (str): Optional name for the case, used in reports.
        ensure_coverage (bool): If True, ensures result covers all input variables.
        
    Returns:
        MISDAResult: Object containing all analysis artifacts.
    """
    # 1. Regime Diagnosis
    alpha_min, alpha_max, r_max_real, r_null = estimate_alpha_interval(Y)
    metrics = diagnose_alpha_regime(alpha_min, alpha_max)
    regime = AlphaRegime(metrics["regime"])
    
    # 2. Decision Logic
    alpha_exec = select_alpha(alpha_min, alpha_max, caution)
    
    # 3. Execution
    # 3. Execution
    # 3. Execution
    res = misda_significance(Y, alpha=alpha_exec, ensure_coverage=ensure_coverage, min_coverage=None)
    
    # 4. Validation (SES)
    ses_out = None
    if run_ses and res.get("mis_ranked"):
        best_ids = res["mis_ranked"][0]["mis_indices"]
        # If Y is DataFrame, pass values; calculate_ses handles DataFrame but let's be safe
        Y_val = Y.values if hasattr(Y, "values") else Y
        # Pass full Y (original) to calculate_ses
        ses_out = calculate_ses(Y_val, best_ids)
        
    return MISDAResult(
        Y=Y,
        caution=caution,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        metrics=metrics,
        regime=regime,
        alpha_exec=alpha_exec,
        isda_res=res,
        ses_results=ses_out,
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
            truth_dim = item.get('truth', {}).get('intrinsic_dim_expected', None)
        
        if res is None:
            continue
            
        # Basic Params
        # Handle res.Y being dataframe or numpy
        N, M = res.Y.shape
        algo_alpha = res.alpha
        
        # MIS Info
        mis_indices = res.best_mis['mis_indices'] if res.best_mis else []
        dim_red = len(mis_indices)
        
        # Fidelity (Linear)
        fidel_lin = 0.0
        if res.ses_results:
             fidel_lin = res.ses_results.get("F_real", 0.0)
             
        # Fidelity (Non-Linear)
        fidel_nl = 0.0
        try:
            if N <= 5000: # Per safeguard
                fidel_nl = calculate_ses_nonlinear(res.Y, mis_indices, n_estimators=50)
        except:
             pass
             
        # Pareto Consistency
        prec, rec = 0.0, 0.0
        try:
             prec, rec = evaluate_pareto_consistency(res, res.Y)
        except:
             pass
             
        # Homogeneity
        homog = res.homogeneity_ratio
        
        # Status
        status = "OK"
        if fidel_lin < 0.9 and fidel_nl < 0.9:
            status = "LOW_FIDEL"
        if prec < 1.0:
            status = "UNSAFE(Prec)"
        if truth_dim and dim_red != truth_dim:
             status += f"|DimMismatch({dim_red}!={truth_dim})"
        
        # Alpha Bounds
        a_min = res.alpha_min if hasattr(res, 'alpha_min') else 0.0
        a_max = res.alpha_max if hasattr(res, 'alpha_max') else 1.0

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
            "Fidel(Lin)": f"{fidel_lin:.2f}",
            "Fidel(NL)": f"{fidel_nl:.2f}",
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
