# ADR 0011 — Public API and object model

- Status: Accepted
- Recorded: 2026-09-11 (retrospective)

## Context

The public API must reflect the methodological separation between structural discovery, optional evidence, and contextual ranking. Object ownership must also prevent ambiguous concepts such as a candidate carrying an intrinsic rank or a result carrying a preference-selected dimension.

## Decision

The normative alpha-stage public flow is:

```python
mis_set = misda.discover(Y, ...)
misda.evaluate(mis_set, metrics=(...), candidates=...)
ranking = misda.rank(mis_set, policy="structural_coverage")
```

### MISSet

`MISSet` owns the discovered candidate universe and global analysis state: thresholds, graphs, dimensions, topology diagnostics, convergence status, support diagnostics, canonical candidate sequence, and data needed for later evaluation.

### MISCandidate

`MISCandidate` owns intrinsic candidate properties and candidate-level evidence. Its operational identity is its fixed canonical position in the owning `MISSet`.

A candidate does not own a public intrinsic rank, rank values, or ranking-dependent ID.

### Ranking

`Ranking` is an ordered snapshot view over an existing `MISSet`. It owns the policy name, canonical candidate indices, scientific tie groups, and selected candidate.

```text
mis_set.analysis.structural_dimension  = graph-derived dimension
ranking.selected_dimension             = size of preference-selected candidate
```

These quantities are deliberately distinct.

## Invariants

- `discover()` does not accept ranking policy;
- `evaluate()` does not change candidate membership, graphs, dimensions, or canonical order;
- `rank()` does not reorder or mutate the owning `MISSet`;
- integer ranking access returns underlying candidates; slicing returns a ranking view;
- candidate canonical positions remain stable for the life of the `MISSet`;
- selected dimension belongs to `Ranking`, not `MISSet`.

## Current implementation

The current API exposes typed analysis and metric domains rather than flattened public dictionaries. Examples include `candidate.linear.mean_r2`, `candidate.pareto.jaccard`, `mis_set.analysis.structural_dimension`, and `ranking.groups`.

The project is alpha-stage and intentionally removes superseded compatibility concepts rather than preserving wrappers for them.

## Permitted implementation variations

Internal dataclass layout, module boundaries, storage mechanisms, and caching strategy may change if the public semantic ownership above is preserved.

## Forbidden shortcuts / regression risks

Do not reintroduce discovery-time ranking policy, intrinsic candidate rank, a `result.selected_dimension` that conflates graph and ranking quantities, or hidden evaluation triggered merely by displaying/reporting an object.

## Verification

API contract tests should cover ownership, indexing/slicing semantics, object identity, immutability of canonical positions, and absence of discovery mutation after evaluation/ranking.