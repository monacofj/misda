# ADR 0018 — Ranking-selected visualization views

- Status: Accepted
- Date: 2026-09-16
- Related issue: #55

## Context

`MISSet` owns the complete discovered candidate universe and stored evaluation state. `Ranking` is a view over that universe rather than an independent scientific result object. Visualizations need to highlight one candidate, but duplicating every visualization method on `Ranking` would make it behave like a façade for `MISSet` and obscure where the underlying graph/data/evaluation state actually lives.

At the same time, a ranking can contain scientific tie groups. Selecting only the first candidate globally hides two separate choices: which scientific rank level is used and which deterministic representative inside that tie group is shown.

Pareto visualization additionally needs to preserve MISDA's no-hidden-evaluation contract. A plot must not silently recompute a scientific metric merely because a user requests a view.

## Decision

Visualization methods remain methods of `MISSet`.

Whenever a visualization depends on one ranking-selected MIS, the public selection semantics are:

```text
ranking -> level -> position
```

where:

- `ranking` accepts `None`, `"default"`, a named policy such as `"size_span"`, or a `Ranking` instance belonging to the same `MISSet`;
- `None` and `"default"` mean the current canonical default ranking;
- an explicit policy name pins that policy even if the future default changes;
- `level` is a zero-based scientific tie-group index in the resolved ranking;
- `position` is a zero-based MIS position inside that tie group;
- the canonical candidate index remains the operational identity of the selected MIS but is not used as a substitute for ranking-level selection semantics.

The default selection is:

```python
ranking="default", level=0, position=0
```

`graph_plot()` and `front_plot()` use this same rule.

## Pareto-front visualization

`MISSet.front_plot()` is an interactive Plotly view over already stored Pareto evidence.

It uses:

- `candidate.pareto.reduced_front_indices` for the reduced-front membership;
- `mis_set.pareto_stability.observed_front_indices` for the full empirical front;
- the stored objective matrix only for coordinates in the displayed projection.

It must not call Pareto evaluation or recompute nondominated membership.

For ordinary minimization on the same observations, reducing to a subset of objectives can only remove full-front members from the nondominated set; it cannot introduce a reduced-front point that was dominated in the full space. Therefore the view has three classes:

1. `preserved`: reduced-front observations;
2. `lost`: full-front observations not retained by the reduced front;
3. `dominated`: observations outside the full front, shown as hollow context points.

There is no `introduced` class under this contract.

For three or more original objectives the default view is a rotatable 3D scatter. X/Y/Z selectors may display any original objectives; retained MIS objectives are identified in the controls. Changing displayed axes changes only the projection and never front membership. Two-objective inputs use a 2D scatter.

No surface/mesh is inferred from projected points because a low-dimensional projection need not preserve the geometry or dominance relations of the full objective space.

## Display environments

Notebook-capable environments display the Plotly figure inline.

When `front_plot(show=True)` is called from a terminal, MISDA writes a self-contained HTML representation and attempts to open it in the system browser. If automatic browser launch is unavailable, the HTML file is retained and its path is reported. The Plotly `Figure` is returned in all cases.

## Rationale

This keeps ownership explicit:

- `MISSet` owns the data and stored scientific state;
- `Ranking` defines an ordering/tie-group view;
- `ranking`, `level`, and `position` specify which already-discovered candidate a view highlights.

The same selection vocabulary across visualizations avoids hidden conventions such as treating a global ranking position, a canonical candidate index, and a scientific rank level as interchangeable concepts.

## Invariants

1. Visualization does not alter discovery, evaluation, ranking, graph structure, or candidate order.
2. `front_plot()` does not compute missing Pareto evidence.
3. A `Ranking` supplied to a view must belong to the same `MISSet`.
4. `position` is local to `level`, not global in the ranking and not a canonical candidate index.
5. `None`/`"default"` track the canonical policy; explicit `"size_span"` pins the named policy.
6. Axis selectors alter only the visual projection.
7. Terminal fallback must not make the scientific result depend on display capabilities.

## Verification

Regression tests cover:

- default, named-policy, and explicit `Ranking` selectors;
- selection by tie `level` and within-level `position`;
- consistent selection semantics in `graph_plot()` and `front_plot()`;
- stored-state requirement for Pareto visualization;
- membership classes and default axes;
- standalone HTML fallback when a browser cannot be opened automatically.
