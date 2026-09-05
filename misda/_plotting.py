# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plotting views for the static MISSet API."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


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


def plot_mis_set_graph(mis_set, *, ranking, show=True):
    """Render the stored positive structural graph.

    The candidate selected by ``ranking`` is highlighted in green. Its direct
    structural neighbors are black. No scientific calculation is performed by
    this view.
    """

    if ranking.mis_set is not mis_set:
        raise ValueError("ranking belongs to a different MISSet.")

    graph = mis_set.analysis.structural_graph
    selected = ranking.selected
    selected_nodes = set(selected.indices if selected is not None else ())
    neighbor_nodes = set()
    for node in selected_nodes:
        neighbor_nodes.update(graph.neighbors(node))
    neighbor_nodes -= selected_nodes

    positions = nx.spring_layout(
        graph,
        seed=7,
        k=3.0 / max(graph.number_of_nodes(), 1) ** 0.5,
        iterations=1000,
    )
    positions = _enforce_min_distance(
        positions,
        min_dist=0.5,
        iters=1200,
        seed=7,
    )
    fig, ax = plt.subplots(figsize=(9, 7))

    selected_edges = [
        edge
        for edge in graph.edges
        if edge[0] in selected_nodes or edge[1] in selected_nodes
    ]
    other_edges = [edge for edge in graph.edges if edge not in selected_edges]

    if other_edges:
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=other_edges,
            edge_color="0.15",
            width=1.15,
            alpha=0.85,
            ax=ax,
        )
    if selected_edges:
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=selected_edges,
            edge_color="C2",
            width=1.15,
            alpha=0.95,
            ax=ax,
        )

    nodes = tuple(graph.nodes)
    node_colors = []
    label_colors = {}
    for node in nodes:
        if node in selected_nodes:
            node_colors.append("C2")
            label_colors[node] = "white"
        elif node in neighbor_nodes:
            node_colors.append("black")
            label_colors[node] = "white"
        else:
            node_colors.append("white")
            label_colors[node] = "black"

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=nodes,
        node_size=420,
        node_color=node_colors,
        edgecolors="black",
        linewidths=1.2,
        ax=ax,
    )

    labels = {
        node: str(graph.nodes[node].get("label", node))
        for node in nodes
    }
    for node in nodes:
        x, y = positions[node]
        ax.text(
            x,
            y,
            labels[node],
            ha="center",
            va="center",
            fontsize=9,
            color=label_colors[node],
            zorder=10,
        )

    ax.set_title(
        "MISDA structural graph — "
        f"{ranking.policy}; selected dimension={ranking.selected_dimension}"
    )
    ax.axis("off")
    fig.tight_layout()
    if show:
        plt.show()
    return fig
