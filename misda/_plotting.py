# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""MISDA graph visualization."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_result_graph(result, *, show=True):
    """Render the stored structural graph with the established MISDA design."""

    analysis = result.analysis
    n_objectives = analysis.original_dimension
    labels = [
        analysis.structural_graph.nodes[index]["label"]
        for index in range(n_objectives)
    ]
    payload = {
        "M": n_objectives,
        "adjacency": nx.to_numpy_array(
            analysis.structural_graph,
            nodelist=range(n_objectives),
            dtype=int,
            weight=None,
        ),
        "labels": labels,
        "mis_ranked": [
            {
                "rank": candidate.rank,
                "mis_indices": list(candidate.indices),
            }
            for candidate in result.mis
        ],
    }
    rendered = plot_custom_misda_graph(
        payload,
        title=(
            f"{result.name or 'MISDA'} — alpha={analysis.alpha:.3g} — "
            f"{analysis.separation_status.value}"
        ),
        show_removed=False,
    )
    if show:
        plt.show()
    return rendered["fig"]

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
