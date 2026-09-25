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

from .api import (
    DOMINANCE_PRESERVATION,
    MISSet,
    Ranking,
    PARETO_RETENTION,
    SIZE_SPAN,
    rank,
)
from ._plotting import plot_mis_set_graph


def _resolve_ranking(mis_set, ranking):
    """Resolve a public ranking selector without changing stored state."""

    if ranking is None or ranking == "default":
        return mis_set.structural_ranking
    if isinstance(ranking, str):
        if ranking not in {SIZE_SPAN, PARETO_RETENTION, DOMINANCE_PRESERVATION}:
            raise ValueError(
                f"Unsupported ranking selector {ranking!r}; use 'default', "
                f"{SIZE_SPAN!r}, {PARETO_RETENTION!r}, "
                f"{DOMINANCE_PRESERVATION!r}, or a Ranking instance."
            )
        return rank(mis_set, policy=ranking)
    if isinstance(ranking, Ranking):
        if ranking.mis_set is not mis_set:
            raise ValueError("ranking belongs to a different MISSet.")
        return ranking
    raise TypeError(
        "ranking must be None, 'default', 'size_span', 'pareto_retention', "
        "'dominance_preservation', or a Ranking instance."
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


def _use_3d_projection(projection, n_objectives):
    if not isinstance(projection, str):
        raise TypeError("projection must be 'auto', '2d', or '3d'.")
    projection = projection.lower()
    if projection not in {"auto", "2d", "3d"}:
        raise ValueError("projection must be 'auto', '2d', or '3d'.")
    if projection == "3d" and n_objectives < 3:
        raise ValueError("projection='3d' requires at least three objectives.")
    if projection == "2d":
        return False
    return n_objectives >= 3


def _in_notebook():
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    return getattr(shell, "kernel", None) is not None


def _show_plotly_figure(fig, *, renderer=None):
    """Show inline when possible, otherwise open a standalone browser view.

    An explicit Plotly renderer is forwarded unchanged. This is useful when a
    notebook frontend renders Plotly controls but fails to draw a WebGL 3D
    scene. The terminal fallback deliberately writes a self-contained HTML
    file. If a browser cannot be opened (for example on a headless machine),
    the file is retained and its path is reported so the caller still has a
    usable result.
    """

    if renderer is not None:
        fig.show(renderer=renderer)
        return None

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
            "run mis_set.evaluate(..., metrics=('pareto',), candidates=...) first."
        )
    diagnostics = getattr(mis_set, "pareto_stability", None)
    if diagnostics is None:
        raise ValueError(
            "front_plot() requires stored Pareto stability state; run "
            "mis_set.evaluate(..., metrics=('pareto',), candidates=...) first."
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
    return [
        data[np.asarray(indices, dtype=int), objective_index].tolist()
        for indices in classes
    ]


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
        "y": 1.055,
        "yanchor": "top",
        "pad": {"r": 8, "t": 0},
    }


def _layout_annotations(
    *,
    mis_set,
    resolved,
    candidate,
    level,
    position,
    pareto,
    is_3d,
):
    title = "Pareto-front preservation"
    if mis_set.name:
        title = f"{title} — {mis_set.name}"

    axis_labels = ("X", "Y", "Z") if is_3d else ("X", "Y")
    axis_x = (0.00, 0.23, 0.46) if is_3d else (0.00, 0.23)
    annotations = [
        {
            "text": f"<b>{title}</b>",
            "xref": "paper",
            "yref": "paper",
            "x": 0.0,
            "y": 1.145,
            "xanchor": "left",
            "yanchor": "bottom",
            "showarrow": False,
            "font": {"size": 18},
        }
    ]
    annotations.extend(
        {
            "text": f"<b>{axis}</b>",
            "xref": "paper",
            "yref": "paper",
            "x": x,
            "y": 1.075,
            "xanchor": "left",
            "yanchor": "bottom",
            "showarrow": False,
            "font": {"size": 12},
        }
        for axis, x in zip(axis_labels, axis_x)
    )
    annotations.extend(
        [
            {
                "text": (
                    f"{resolved.policy} · level {int(level)} · position {int(position)} "
                    f"· MIS dimension {candidate.size}"
                    if resolved is not None
                    else f"MIS dimension {candidate.size}"
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": -0.135,
                "xanchor": "left",
                "yanchor": "top",
                "showarrow": False,
                "font": {"size": 12},
            },
            {
                "text": (
                    f"full front {pareto.full_front_size} · "
                    f"reduced front {pareto.reduced_front_size} · "
                    f"retention {_format_metric(pareto.retention)} · "
                    f"Jaccard {_format_metric(pareto.jaccard)}"
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": -0.185,
                "xanchor": "left",
                "yanchor": "top",
                "showarrow": False,
                "font": {"size": 12},
            },
        ]
    )
    return annotations


def plot_mis_set_front(
    mis_set,
    *,
    ranking=None,
    level=0,
    position=0,
    show=True,
    projection="auto",
    renderer=None,
    candidate=None,
):
    """Render stored full/reduced Pareto membership as an interactive view.

    projection='auto' uses a rotatable Plotly 3D scene when at least three
    objectives exist. projection='2d' forces an SVG-backed two-axis view,
    which also provides a non-WebGL fallback for restrictive notebook
    frontends. renderer is forwarded to Plotly when automatic display is
    requested.
    """

    if candidate is None:
        resolved, candidate_index, candidate = resolve_ranking_selection(
            mis_set,
            ranking,
            level=level,
            position=position,
        )
    else:
        if getattr(candidate, "_mis_set", None) is not mis_set:
            raise ValueError("candidate belongs to a different MISSet.")
        resolved = None
        candidate_index = next(
            (
                index
                for index, observed in enumerate(mis_set)
                if observed is candidate
            ),
            None,
        )
    data = np.asarray(mis_set._data, dtype=float)
    labels = tuple(mis_set._labels)
    n_objectives = data.shape[1]
    if n_objectives < 2:
        raise ValueError("front_plot() requires at least two objectives.")

    all_classes = _class_indices(mis_set, candidate)
    present = tuple(
        (spec, indices)
        for spec, indices in zip(_trace_specs(), all_classes)
        if indices
    )
    classes = tuple(indices for _spec, indices in present)
    is_3d = _use_3d_projection(projection, n_objectives)
    n_axes = 3 if is_3d else 2
    defaults = _default_axes(candidate, n_objectives, n_axes)

    fig = go.Figure()
    for (status, marker), indices in present:
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

    menu_x = _axis_menu(
        axis="x",
        labels=labels,
        candidate=candidate,
        classes=classes,
        data=data,
        default_index=defaults[0],
        is_3d=is_3d,
        x=0.025,
    )
    menu_y = _axis_menu(
        axis="y",
        labels=labels,
        candidate=candidate,
        classes=classes,
        data=data,
        default_index=defaults[1],
        is_3d=is_3d,
        x=0.255,
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
                x=0.485,
            )
        )

    pareto = candidate.pareto
    meta = {
        "selected_objectives": [str(label) for label in candidate.objectives],
        "default_axes": [str(labels[index]) for index in defaults],
        "projection": "3d" if is_3d else "2d",
    }
    if resolved is not None:
        meta.update(
            {
                "ranking_policy": resolved.policy,
                "level": int(level),
                "position": int(position),
                "candidate_index": candidate_index,
            }
        )
    common_layout = {
        "meta": meta,
        "updatemenus": menus,
        "annotations": _layout_annotations(
            mis_set=mis_set,
            resolved=resolved,
            candidate=candidate,
            level=level,
            position=position,
            pareto=pareto,
            is_3d=is_3d,
        ),
        "height": 720,
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.055,
            "xanchor": "left",
            "x": 0.0,
            "title": {"text": ""},
        },
        "margin": {"l": 20, "r": 20, "b": 125, "t": 135},
    }
    if is_3d:
        fig.update_layout(
            **common_layout,
            scene={
                "domain": {"x": [0.03, 0.97], "y": [0.03, 0.97]},
                "xaxis": {"title": {"text": str(labels[defaults[0]])}},
                "yaxis": {"title": {"text": str(labels[defaults[1]])}},
                "zaxis": {"title": {"text": str(labels[defaults[2]])}},
                "aspectmode": "cube",
            },
        )
    else:
        fig.update_layout(
            **common_layout,
            xaxis={"title": {"text": str(labels[defaults[0]])}},
            yaxis={"title": {"text": str(labels[defaults[1]])}},
        )

    if show:
        _show_plotly_figure(fig, renderer=renderer)
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


def _front_plot(
    self,
    show=True,
    ranking=None,
    level=0,
    position=0,
    projection="auto",
    renderer=None,
):
    """Plot stored Pareto preservation for a ranking-selected MIS."""

    return plot_mis_set_front(
        self,
        ranking=ranking,
        level=level,
        position=position,
        show=show,
        projection=projection,
        renderer=renderer,
    )


def plot_mis_candidate_front(
    candidate,
    *,
    show=True,
    projection="auto",
    renderer=None,
):
    """Render stored Pareto preservation for one already-selected MIS."""

    mis_set = candidate._owner()
    return plot_mis_set_front(
        mis_set,
        candidate=candidate,
        show=show,
        projection=projection,
        renderer=renderer,
    )


def _install():
    MISSet.graph_plot = _graph_plot
    MISSet.front_plot = _front_plot


_install()
