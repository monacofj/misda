# SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ranking-aware stored-state visualization for MISDA results."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import warnings
import webbrowser

import numpy as np
import plotly.graph_objects as go

from .api import MISSet, Ranking, SIZE_SPAN, rank
from ._plotting import plot_mis_set_graph


def _resolve_ranking(mis_set, ranking):
    """Resolve a public ranking selector without changing stored state."""

    if ranking is None or ranking == "default":
        return mis_set.structural_ranking
    if isinstance(ranking, str):
        if ranking != SIZE_SPAN:
            raise ValueError(
                f"Unsupported ranking selector {ranking!r}; use 'default', "
                f"{SIZE_SPAN!r}, or a Ranking instance."
            )
        return rank(mis_set, policy=SIZE_SPAN)
    if isinstance(ranking, Ranking):
        if ranking.mis_set is not mis_set:
            raise ValueError("ranking belongs to a different MISSet.")
        return ranking
    raise TypeError(
        "ranking must be None, 'default', 'size_span', or a Ranking instance."
    )


def _nonnegative_index(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be a non-negative integer.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def resolve_ranking_selection(mis_set, ranking=None, *, level=0, position=0):
    """Resolve ``ranking -> scientific tie level -> MIS within that level``.

    The returned candidate index is the stable index in the owning ``MISSet``;
    ``position`` is local to the selected scientific tie group rather than a
    global position in the Ranking view.
    """

    if not isinstance(mis_set, MISSet):
        raise TypeError("mis_set must be an MISSet.")
    resolved = _resolve_ranking(mis_set, ranking)
    level = _nonnegative_index(level, "level")
    position = _nonnegative_index(position, "position")

    if level >= len(resolved.groups):
        raise IndexError(
            f"ranking level {level} is out of range for {len(resolved.groups)} levels."
        )
    group = resolved.groups[level]
    if position >= len(group):
        raise IndexError(
            f"position {position} is out of range for ranking level {level} "
            f"with {len(group)} candidates."
        )
    candidate_index = int(group[position])
    return resolved, candidate_index, mis_set[candidate_index]


def _objective_menu_label(label, selected):
    marker = "✓ " if selected else "  "
    return f"{marker}{label}"


def _default_axes(candidate, n_objectives, n_axes):
    preferred = list(candidate.indices[:n_axes])
    for index in range(n_objectives):
        if len(preferred) >= n_axes:
            break
        if index not in preferred:
            preferred.append(index)
    return tuple(preferred)


def _format_metric(value):
    return "N/A" if value is None else f"{float(value):.4f}"


def _in_notebook():
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    return getattr(shell, "kernel", None) is not None


def _show_plotly_figure(fig):
    """Show inline when possible, otherwise open a standalone browser view.

    The terminal fallback deliberately writes a self-contained HTML file. If a
    browser cannot be opened (for example on a headless machine), the file is
    retained and its path is reported so the caller still has a usable result.
    """

    if _in_notebook():
        fig.show()
        return None

    path = None
    try:
        handle, raw_path = tempfile.mkstemp(prefix="misda-front-", suffix=".html")
        os.close(handle)
        path = Path(raw_path).resolve()
        fig.write_html(
            str(path),
            include_plotlyjs=True,
            full_html=True,
            auto_open=False,
        )
        opened = bool(webbrowser.open_new_tab(path.as_uri()))
    except Exception as exc:  # pragma: no cover - environment-specific fallback
        warnings.warn(
            "MISDA could not open the interactive front plot from this terminal; "
            f"the Plotly Figure is still returned ({exc}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return path

    if opened:
        print(f"MISDA front_plot: opened interactive plot in browser ({path}).")
    else:
        warnings.warn(
            "MISDA could not open a browser automatically. "
            f"Open this interactive plot manually: {path}",
            RuntimeWarning,
            stacklevel=2,
        )
    return path


def _class_indices(mis_set, candidate):
    if candidate.pareto is None:
        raise ValueError(
            "front_plot() requires stored Pareto evidence for the selected MIS; "
            "run misda.evaluate(..., metrics=('pareto',), candidates=...) first."
        )
    diagnostics = getattr(mis_set, "pareto_stability", None)
    if diagnostics is None:
        raise ValueError(
            "front_plot() requires stored Pareto stability state; run "
            "misda.evaluate(..., metrics=('pareto',), candidates=...) first."
        )

    full = set(int(index) for index in diagnostics.observed_front_indices)
    reduced = set(int(index) for index in candidate.pareto.reduced_front_indices)
    if not reduced.issubset(full):
        raise ValueError(
            "stored Pareto state is inconsistent: reduced front is not a subset "
            "of the full front."
        )
    all_rows = set(range(mis_set._data.shape[0]))
    return (
        tuple(sorted(all_rows - full)),
        tuple(sorted(full - reduced)),
        tuple(sorted(reduced)),
    )


def _trace_specs():
    return (
        (
            "dominated",
            {
                "size": 5,
                "color": "#7f7f7f",
                "symbol": "circle-open",
                "opacity": 0.45,
                "line": {"width": 1.0, "color": "#7f7f7f"},
            },
        ),
        (
            "lost",
            {
                "size": 6,
                "color": "#d62728",
                "symbol": "circle",
                "opacity": 0.88,
            },
        ),
        (
            "preserved",
            {
                "size": 6,
                "color": "#2ca02c",
                "symbol": "circle",
                "opacity": 0.92,
            },
        ),
    )


def _hover_data(indices, status):
    return [[int(index), status] for index in indices]


def _axis_arrays(data, classes, objective_index):
    return [data[np.asarray(indices, dtype=int), objective_index].tolist() for indices in classes]


def _axis_menu(
    *,
    axis,
    labels,
    candidate,
    classes,
    data,
    default_index,
    is_3d,
    x,
):
    selected = set(candidate.indices)
    buttons = []
    for objective_index, label in enumerate(labels):
        display = _objective_menu_label(str(label), objective_index in selected)
        data_update = {axis: _axis_arrays(data, classes, objective_index)}
        layout_key = (
            f"scene.{axis}axis.title.text" if is_3d else f"{axis}axis.title.text"
        )
        buttons.append(
            {
                "label": display,
                "method": "update",
                "args": [data_update, {layout_key: str(label)}],
            }
        )
    return {
        "type": "dropdown",
        "direction": "down",
        "showactive": True,
        "active": int(default_index),
        "buttons": buttons,
        "x": x,
        "xanchor": "left",
        "y": 1.12,
        "yanchor": "bottom",
        "pad": {"r": 6, "t": 0},
    }


def plot_mis_set_front(
    mis_set,
    *,
    ranking=None,
    level=0,
    position=0,
    show=True,
):
    """Render stored full/reduced Pareto membership as an interactive view."""

    resolved, candidate_index, candidate = resolve_ranking_selection(
        mis_set,
        ranking,
        level=level,
        position=position,
    )
    data = np.asarray(mis_set._data, dtype=float)
    labels = tuple(mis_set._labels)
    n_objectives = data.shape[1]
    if n_objectives < 2:
        raise ValueError("front_plot() requires at least two objectives.")

    classes = _class_indices(mis_set, candidate)
    is_3d = n_objectives >= 3
    n_axes = 3 if is_3d else 2
    defaults = _default_axes(candidate, n_objectives, n_axes)
    specs = _trace_specs()

    fig = go.Figure()
    for (status, marker), indices in zip(specs, classes):
        rows = np.asarray(indices, dtype=int)
        customdata = _hover_data(indices, status)
        if is_3d:
            fig.add_trace(
                go.Scatter3d(
                    name=status,
                    mode="markers",
                    x=data[rows, defaults[0]],
                    y=data[rows, defaults[1]],
                    z=data[rows, defaults[2]],
                    marker=marker,
                    customdata=customdata,
                    hovertemplate=(
                        "observation=%{customdata[0]}<br>"
                        "status=%{customdata[1]}<br>"
                        "x=%{x:.4g}<br>y=%{y:.4g}<br>z=%{z:.4g}"
                        "<extra></extra>"
                    ),
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    name=status,
                    mode="markers",
                    x=data[rows, defaults[0]],
                    y=data[rows, defaults[1]],
                    marker=marker,
                    customdata=customdata,
                    hovertemplate=(
                        "observation=%{customdata[0]}<br>"
                        "status=%{customdata[1]}<br>"
                        "x=%{x:.4g}<br>y=%{y:.4g}<extra></extra>"
                    ),
                )
            )

    pareto = candidate.pareto
    title = (
        "MISDA Pareto-front preservation"
        "<br><sup>"
        f"ranking={resolved.policy}; level={int(level)}; position={int(position)}; "
        f"candidate[{candidate_index}]; MIS={list(map(str, candidate.objectives))} · "
        f"full={pareto.full_front_size}; reduced={pareto.reduced_front_size}; "
        f"retention={_format_metric(pareto.retention)}; "
        f"jaccard={_format_metric(pareto.jaccard)}"
        "</sup>"
    )

    menu_x = _axis_menu(
        axis="x",
        labels=labels,
        candidate=candidate,
        classes=classes,
        data=data,
        default_index=defaults[0],
        is_3d=is_3d,
        x=0.00,
    )
    menu_y = _axis_menu(
        axis="y",
        labels=labels,
        candidate=candidate,
        classes=classes,
        data=data,
        default_index=defaults[1],
        is_3d=is_3d,
        x=0.30,
    )
    menus = [menu_x, menu_y]
    if is_3d:
        menus.append(
            _axis_menu(
                axis="z",
                labels=labels,
                candidate=candidate,
                classes=classes,
                data=data,
                default_index=defaults[2],
                is_3d=True,
                x=0.60,
            )
        )

    meta = {
        "ranking_policy": resolved.policy,
        "level": int(level),
        "position": int(position),
        "candidate_index": candidate_index,
        "selected_objectives": [str(label) for label in candidate.objectives],
        "default_axes": [str(labels[index]) for index in defaults],
    }
    if is_3d:
        fig.update_layout(
            title={"text": title},
            meta=meta,
            updatemenus=menus,
            scene={
                "xaxis": {"title": {"text": str(labels[defaults[0]])}},
                "yaxis": {"title": {"text": str(labels[defaults[1]])}},
                "zaxis": {"title": {"text": str(labels[defaults[2]])}},
            },
            legend={"title": {"text": "membership"}},
            margin={"l": 0, "r": 0, "b": 0, "t": 120},
        )
    else:
        fig.update_layout(
            title={"text": title},
            meta=meta,
            updatemenus=menus,
            xaxis={"title": {"text": str(labels[defaults[0]])}},
            yaxis={"title": {"text": str(labels[defaults[1]])}},
            legend={"title": {"text": "membership"}},
            margin={"t": 120},
        )

    if show:
        _show_plotly_figure(fig)
    return fig


def _graph_plot(self, show=True, ranking=None, level=0, position=0):
    """Plot stored G+ using the public ranking/level/position selection rule."""

    resolved, candidate_index, candidate = resolve_ranking_selection(
        self,
        ranking,
        level=level,
        position=position,
    )
    return plot_mis_set_graph(
        self,
        ranking=resolved,
        candidate=candidate,
        candidate_index=candidate_index,
        level=int(level),
        position=int(position),
        show=show,
    )


def _front_plot(self, show=True, ranking=None, level=0, position=0):
    """Plot stored Pareto preservation for a ranking-selected MIS."""

    return plot_mis_set_front(
        self,
        ranking=ranking,
        level=level,
        position=position,
        show=show,
    )


def _install():
    MISSet.graph_plot = _graph_plot
    MISSet.front_plot = _front_plot


_install()
