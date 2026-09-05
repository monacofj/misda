"""Regression tests for MISDA graph rendering geometry."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from misda import _plotting


def _four_cliques_of_five():
    graph = nx.disjoint_union_all([nx.complete_graph(5) for _ in range(4)])
    nx.set_node_attributes(graph, {node: f"f{node + 1}" for node in graph}, "label")
    return graph


def _minimum_pairwise_distance(positions):
    points = np.asarray([positions[node] for node in positions], dtype=float)
    deltas = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    distances[np.eye(len(points), dtype=bool)] = np.inf
    return float(np.min(distances))


def test_enforce_min_distance_prevents_case3_style_component_collapse():
    graph = _four_cliques_of_five()
    positions = nx.spring_layout(
        graph,
        seed=7,
        k=3.0 / graph.number_of_nodes() ** 0.5,
        iterations=1000,
    )

    separated = _plotting._enforce_min_distance(
        positions,
        min_dist=0.5,
        iters=1200,
        seed=7,
    )

    assert _minimum_pairwise_distance(separated) >= 0.5 - 1e-12


def test_plot_mis_set_graph_applies_anti_overlap_postprocessing(monkeypatch):
    graph = _four_cliques_of_five()
    mis_set = SimpleNamespace(
        analysis=SimpleNamespace(structural_graph=graph),
    )
    selected = SimpleNamespace(indices=(0, 5, 10, 15))
    ranking = SimpleNamespace(
        mis_set=mis_set,
        selected=selected,
        policy="structural_coverage",
        selected_dimension=4,
    )
    calls = []
    original = _plotting._enforce_min_distance

    def recording_enforcer(positions, **kwargs):
        calls.append(kwargs)
        return original(positions, **kwargs)

    monkeypatch.setattr(_plotting, "_enforce_min_distance", recording_enforcer)
    figure = _plotting.plot_mis_set_graph(mis_set, ranking=ranking, show=False)

    try:
        assert calls == [{"min_dist": 0.5, "iters": 1200, "seed": 7}]
    finally:
        plt.close(figure)
