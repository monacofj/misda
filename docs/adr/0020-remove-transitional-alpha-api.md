# ADR 0020 — Remove transitional API compatibility during alpha

- Status: Accepted
- Recorded: 2026-09-21
- Tracking issue: #61
- Pull request: #62
- Supersedes transitional compatibility clauses in ADRs 0009, 0011, 0017, and 0019

## Context

Revision 07 established a coherent object model:

```text
discover -> MISSet -> evaluate
                 \-> rank -> Ranking -> mis -> inspect
```

During that transition, MISDA temporarily retained multiple equivalent public
paths: module-level evaluation, MISSet-level reporting and plotting, ranking
selection aliases, integer/index evaluation selectors, and an obsolete ranking
constant alias.

MISDA is still alpha software. Carrying those paths forward would make examples,
tests, and documentation teach more than one way to express the same operation
and would prematurely turn transitional names into compatibility obligations.

## Decision

The normative alpha public workflow is:

```python
mis_set = misda.discover(Y)
mis_set.evaluate(...)

ranking = misda.rank(mis_set)
ranking.report()

mis = ranking.mis()
mis.report()
mis.graph_plot()
mis.front_plot()
```

The following transitional public paths are removed without a deprecation
cycle:

- module-level `misda.evaluate(mis_set, ...)`;
- `MISSet.report()`;
- `MISSet.graph_plot(...)`;
- `MISSet.front_plot(...)`;
- `MISSet.structural_ranking`;
- `Ranking.selected`;
- integer-prefix and explicit-index evaluation selectors;
- `STRUCTURAL_COVERAGE` and the private old-name ranking helper alias.

Evaluation scope accepts `"all"`, one MIS object, a sequence of MIS objects,
or a `Ranking` view. Complete reporting belongs to `Ranking`. Candidate
reporting and plotting belong to an explicitly selected MIS.

## Rationale

Alpha status is the appropriate time to remove redundant API paths. One
canonical workflow reduces conceptual surface area and makes notebooks
executable documentation of the intended design.

The removal is API-only. It does not change threshold estimation, graph
construction, MIS enumeration, dimensional estimators, ranking semantics,
support diagnostics, reconstruction metrics, Pareto calculations, or benchmark
truth boundaries.

## Invariants

- `discover()` returns the complete discovered MIS universe;
- `MISSet.evaluate()` enriches stored evidence without changing discovery;
- `rank()` creates a view over existing MIS objects;
- `Ranking.mis()` selects by scientific tie level and local position;
- reports and plots never trigger hidden scientific evaluation;
- benchmark reports embed the complete native `Ranking.report()` verbatim;
- no compatibility alias may silently reintroduce a second public path.

## Verification

The acceptance gate must verify both presence of the canonical surface and
absence of the removed surface. Notebook contract tests must reject the removed
forms, and the complete scientific/reporting test suites must remain green.
