# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import pytest

import misda
from misda import _front_plotting
from misda.api import MISCandidate, MISSet, ParetoMetrics, Ranking, StructuralMetrics


def _candidate(objectives, indices, reduced):
    return MISCandidate(
        objectives=objectives,
        indices=indices,
        structural=StructuralMetrics(
            neighborhood=1,
            neighborhood_ratio=1.0,
            span=2,
            avg_external_degree=1.0,
            avg_internal_degree=0.0,
        ),
        pareto=ParetoMetrics(
            retention=2.0 / 3.0,
            validity=1.0,
            jaccard=2.0 / 3.0,
            full_front_size=3,
            reduced_front_size=2,
            intersection_size=2,
            union_size=3,
            exact_preservation=False,
            reduced_front_indices=reduced,
        ),
    )


def _mis_set():
    data = np.asarray(
        [
            [0.0, 3.0, 2.0],
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 0.0],
            [3.0, 3.0, 3.0],
        ]
    )
    graph = nx.Graph()
    graph.add_nodes_from(
        [
            (0, {"label": "f1"}),
            (1, {"label": "f2"}),
            (2, {"label": "f3"}),
        ]
    )
    graph.add_edges_from([(0, 2), (1, 2)])
    result = MISSet(
        analysis=SimpleNamespace(structural_graph=graph),
        candidates=(
            _candidate(("f1", "f2"), (0, 1), (0, 2)),
            _candidate(("f1", "f3"), (0, 2), (0, 1)),
        ),
        rank_groups=((0, 1),),
        data=data,
        labels=("f1", "f2", "f3"),
        seed=123,
    )
    result.pareto_stability = SimpleNamespace(observed_front_indices=(0, 1, 2))
    return result


def test_front_plot_uses_level_and_position_inside_tie_group():
    result = _mis_set()

    fig = result.front_plot(show=False, level=0, position=1)

    assert isinstance(fig, go.Figure)
    assert fig.layout.meta["ranking_policy"] == "size_span"
    assert fig.layout.meta["level"] == 0
    assert fig.layout.meta["position"] == 1
    assert fig.layout.meta["candidate_index"] == 1
    assert fig.layout.meta["selected_objectives"] == ("f1", "f3") or list(
        fig.layout.meta["selected_objectives"]
    ) == ["f1", "f3"]
    assert tuple(trace.name for trace in fig.data) == (
        "dominated",
        "lost",
        "preserved",
    )
    assert fig.data[0].marker.symbol == "circle-open"
    assert fig.data[1].marker.symbol == "circle"
    assert fig.data[2].marker.symbol == "circle"
    assert tuple(len(trace.x) for trace in fig.data) == (1, 1, 2)
    assert len(fig.layout.updatemenus) == 3
    assert fig.layout.scene.xaxis.title.text == "f1"
    assert fig.layout.scene.yaxis.title.text == "f3"
    assert fig.layout.scene.zaxis.title.text == "f2"


def test_front_plot_accepts_default_size_span_and_ranking_view():
    result = _mis_set()

    default = result.front_plot(show=False, ranking="default")
    explicit = result.front_plot(show=False, ranking="size_span")
    custom = Ranking(
        result,
        (1, 0),
        policy="size_span",
        groups=((1, 0),),
    )
    custom_fig = result.front_plot(show=False, ranking=custom)

    assert default.layout.meta["candidate_index"] == 0
    assert explicit.layout.meta["candidate_index"] == 0
    assert custom_fig.layout.meta["candidate_index"] == 1


def test_graph_plot_uses_same_level_position_selection_rule():
    result = _mis_set()

    fig = result.graph_plot(show=False, level=0, position=1)
    try:
        assert "candidate[1]" in fig.axes[0].get_title()
        assert "level=0" in fig.axes[0].get_title()
        assert "position=1" in fig.axes[0].get_title()
    finally:
        plt.close(fig)


def test_front_plot_requires_stored_pareto_evidence():
    result = _mis_set()
    object.__setattr__(result[0], "pareto", None)

    with pytest.raises(ValueError, match="stored Pareto evidence"):
        result.front_plot(show=False)


def test_front_plot_rejects_invalid_level_and_position():
    result = _mis_set()

    with pytest.raises(IndexError, match="ranking level"):
        result.front_plot(show=False, level=1)
    with pytest.raises(IndexError, match="position"):
        result.front_plot(show=False, position=2)


def test_front_plot_layout_keeps_controls_metadata_and_plot_separate():
    result = _mis_set()
    result.name = "Case 1 - Independent objectives"
    result.pareto_stability = SimpleNamespace(observed_front_indices=(0, 1, 2, 3))
    result._candidates = (
        _candidate(("f1", "f2"), (0, 1), (0, 1, 2, 3)),
        result._candidates[1],
    )

    fig = result.front_plot(show=False)

    assert tuple(trace.name for trace in fig.data) == ("preserved",)
    assert fig.layout.height == 720
    assert fig.layout.legend.orientation == "h"
    annotations = tuple(item.text for item in fig.layout.annotations)
    assert any(
        "Pareto-front preservation — Case 1 - Independent objectives" in text
        for text in annotations
    )
    assert any("MIS dimension 2" in text for text in annotations)
    assert all("MIS=[" not in text for text in annotations)
    assert all("candidate[" not in text for text in annotations)
    assert tuple(menu.y for menu in fig.layout.updatemenus) == (1.055, 1.055, 1.055)


def test_terminal_fallback_writes_standalone_html(monkeypatch):
    fig = go.Figure()
    monkeypatch.setattr(_front_plotting, "_in_notebook", lambda: False)
    monkeypatch.setattr(
        _front_plotting.webbrowser,
        "open_new_tab",
        lambda _uri: False,
    )

    with pytest.warns(RuntimeWarning, match="Open this interactive plot manually"):
        path = _front_plotting._show_plotly_figure(fig)

    assert isinstance(path, Path)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "plotly" in text.lower()
    path.unlink()
