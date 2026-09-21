# MISDA visualization views

MISDA visualizations are views over already stored analysis/evaluation state.
They do not run discovery, MIS evaluation, or ranking implicitly.

## Preferred workflow: select an MIS, then visualize it

Ranking owns scientific tie levels and positions. Select one MIS from the
ranking, then ask that MIS to visualize itself:

```python
ranking = misda.rank(mis_set)

mis = ranking.mis()          # level=0, position=0
mis.graph_plot()
mis.front_plot()
```

To inspect another alternative:

```python
ranking.mis(0, 3).graph_plot()
ranking.mis(0, 3).front_plot()
```

`level` is a scientific tie level and `position` is local to that level;
both are zero-based. Users do not need canonical MISSet indices for this
workflow.

The selected MIS is the same object already owned by `mis_set`; ranking does
not create a copy. Because an MIS can be selected by more than one ranking,
MIS-level plots show intrinsic MIS information and do not claim an intrinsic
ranking policy, level, or position.

## Structural graph view

```python
ranking.mis().graph_plot()
```

renders the stored positive structural graph `G+` with that MIS highlighted.
No graph or ranking calculation is rerun.

## Pareto front view

Pareto evidence must already have been evaluated for the selected MIS:

```python
ranking = misda.rank(mis_set)
mis = ranking.mis()

mis_set.evaluate(
    metrics=("pareto",),
    candidates=mis,
)

mis.front_plot()
```

`front_plot()` renders the empirical Pareto preservation state with Plotly.
With at least three original objectives it is a rotatable 3D scatter. For two
objectives it falls back to a 2D scatter. The default `projection="auto"`
behavior can be overridden with `projection="2d"` or `projection="3d"`;
forcing 2D is also the non-WebGL fallback for notebook/browser environments
that cannot draw Plotly's 3D scene.

Three membership classes are displayed:

- `preserved`: observations on the reduced front; these are also on the full front;
- `lost`: observations on the full front that cease to be nondominated after objective reduction;
- `dominated`: observations outside the full empirical front, retained as hollow circles for geometric context.

For minimization on the same observations and a strict subset of objectives,
the reduced nondominated set is nested in the full nondominated set.
Consequently there is no `introduced` class.

The X/Y/Z selectors change only the displayed projection. They never recompute
front membership. All original objectives are available in the selectors;
objectives retained by the selected MIS are marked with a check mark. The
initial axes use retained objectives first and fill any remaining axes with
non-retained objectives.

No Pareto surface or mesh is drawn because a three-dimensional projection need
not preserve the geometry or dominance relations of the full objective space.

## Compatibility: MISSet-selected views

The previous public forms remain supported:

```python
mis_set.graph_plot(ranking="default", level=0, position=0)
mis_set.front_plot(ranking="default", level=0, position=0)
```

`ranking=None` is equivalent to `ranking="default"`.
`ranking="size_span"` explicitly pins the current named policy, and an
existing `Ranking` object may also be supplied.

These compatibility views retain ranking metadata in the plot title/layout.
The preferred MIS-level views omit that contextual metadata because the MIS
itself does not own an intrinsic rank.

## Notebook and terminal behavior

Inside a notebook, `front_plot(show=True)` displays the interactive Plotly
figure inline. An explicit Plotly renderer may be selected without changing
scientific state:

```python
mis.front_plot(renderer="notebook_connected")
mis.front_plot(renderer="colab")
```

A renderer can draw Plotly controls and annotations even when its WebGL layer
fails to draw a 3D scene. If that happens, use an environment-appropriate
renderer or force the non-WebGL 2D projection:

```python
mis.front_plot(projection="2d")
```

From a local terminal, `renderer="browser"` can also be used explicitly.
Without an explicit renderer, MISDA writes a self-contained temporary HTML
file and attempts to open it in the system browser. If no browser can be opened
automatically, MISDA retains the HTML file and reports its path. In every case
the Plotly `Figure` is returned, so callers may save or render it explicitly.
Use `show=False` to suppress automatic display:

```python
fig = mis.front_plot(show=False)
fig.write_html("front.html")
```
