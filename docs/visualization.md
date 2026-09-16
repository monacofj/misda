# MISDA visualization views

MISDA visualizations are views over already stored analysis/evaluation state. They do not run discovery, candidate evaluation, or ranking implicitly.

## Ranking-selected views

Operations that need one structural MIS use the same three-stage selector:

```python
ranking = "default"   # or "size_span" or a Ranking object
level = 0             # scientific tie group in that ranking
position = 0          # MIS within that tie group
```

`ranking=None` is equivalent to `ranking="default"`. The current default policy is `size_span`. Naming `"size_span"` explicitly pins that policy even if a future release changes the default.

`level` is a scientific rank level. Candidates in the same level are indistinguishable under the scientific criteria of that ranking policy; they are not necessarily equivalent under later reconstruction or Pareto evidence. `position` is local to the selected level, not a global candidate index.

For example:

```python
mis_set.graph_plot(ranking=r2, level=1, position=2)
mis_set.front_plot(ranking=r2, level=1, position=2)
```

Both calls select the same MIS from the same ranking context.

## Structural graph view

```python
mis_set.graph_plot()
```

renders the stored positive structural graph `G+`. The selected MIS is highlighted, together with its graph neighborhood. Supplying `ranking`, `level`, and `position` changes only which already-discovered candidate is highlighted.

## Pareto front view

Pareto evidence must already have been evaluated for the selected candidate:

```python
ranking = misda.rank(mis_set)
misda.evaluate(
    mis_set,
    metrics=("pareto",),
    candidates=ranking[:1],
)
mis_set.front_plot(ranking=ranking)
```

`front_plot()` renders the empirical Pareto preservation state with Plotly. With at least three original objectives it is a rotatable 3D scatter. For two objectives it falls back to a 2D scatter.

Three membership classes are displayed:

- `preserved`: observations on the reduced front; these are also on the full front;
- `lost`: observations on the full front that cease to be nondominated after objective reduction;
- `dominated`: observations outside the full empirical front, retained as hollow circles for geometric context.

For minimization on the same observations and a strict subset of objectives, the reduced nondominated set is nested in the full nondominated set. Consequently there is no `introduced` class.

The X/Y/Z selectors change only the displayed projection. They never recompute front membership. All original objectives are available in the selectors; objectives retained by the selected MIS are marked with a check mark. The initial axes use retained objectives first and fill any remaining axes with non-retained objectives.

The figure title records the ranking policy, tie level, position within that level, canonical candidate index, selected objectives, front sizes, retention, and Jaccard overlap. Hover text reports the observation index, membership class, and displayed coordinates.

No Pareto surface or mesh is drawn because a three-dimensional projection need not preserve the geometry or dominance relations of the full objective space.

## Notebook and terminal behavior

Inside a notebook, `front_plot(show=True)` displays the interactive Plotly figure inline.

From a terminal, MISDA writes a self-contained temporary HTML file and attempts to open it in the system browser. If no browser can be opened automatically, MISDA retains the HTML file and reports its path. In every case the Plotly `Figure` is returned, so callers may save or render it explicitly. Use `show=False` to suppress automatic display:

```python
fig = mis_set.front_plot(show=False)
fig.write_html("front.html")
```
