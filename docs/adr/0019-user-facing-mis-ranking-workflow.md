# ADR 0019 — User-facing MIS, ranking, and evaluation workflow

- Status: Accepted
- Recorded: 2026-09-20

## Context

ADR 0011 separates the discovered `MISSet`, contextual `Ranking`, and
candidate-level evidence. The implementation preserved that ownership, but the
public workflow still exposed internal selection mechanics: users could need to
move between ranking groups, canonical indices, and `MISSet` visualization
arguments in order to inspect one MIS.

That is unnecessarily programmer-oriented. A user of the method should be able
to discover the MIS universe, evaluate it, rank it, select one MIS in ranking
coordinates, and then inspect that MIS directly.

## Decision

The preferred public workflow is:

```python
mis_set = misda.discover(Y)
mis_set.evaluate()

ranking = misda.rank(mis_set)

ranking.report()
ranking.mis().report()
ranking.mis().graph_plot()
ranking.mis().front_plot()
```

### Discovery remains separate

`misda.discover()` remains the discovery entry point. It returns one
`MISSet` containing the discovered MIS universe, global analysis state,
structural metrics, and dimensional-support diagnostics.

The retired `analyze()` name is not reintroduced.

### Evaluation belongs to MISSet

`MISSet.evaluate()` is the preferred object-oriented facade for enriching the
already discovered MISs with evaluation evidence:

```python
mis_set.evaluate(metrics=("linear", "pareto"))
```

Without arguments it preserves the existing evaluator defaults: linear and
Pareto evidence are evaluated with the existing default candidate scope.

The module function remains public and equivalent for compatibility:

```python
misda.evaluate(mis_set, metrics=("linear", "pareto"))
```

Evaluation never belongs to `Ranking`; ranking does not create scientific
evidence.

Nonlinear reconstruction remains an explicitly requested evaluation family:

```python
mis_set.evaluate(
    metrics=("nonlinear",),
    candidates=ranking.mis(),
)
```

The retired `heavy()` name is not reintroduced.

### Ranking owns level/position selection

`Ranking` owns policy, scientific tie levels, and selection coordinates.
The user-facing selector is:

```python
ranking.mis(level=0, position=0)
```

Both arguments are zero-based. With no arguments:

```python
ranking.mis()
```

returns the same selected MIS as the first entry in the first scientific tie
level.

The returned object is the existing `MISCandidate` owned by the original
`MISSet`; `Ranking` does not copy or wrap it.

### One selected MIS can inspect itself

An MIS selected from a ranking exposes stored-state inspection:

```python
mis = ranking.mis(0, 3)

mis.report()
mis.graph_plot()
mis.front_plot()
```

These operations do not perform hidden evaluation.

The MIS itself does not own ranking coordinates or policy. The same MIS may be
selected by multiple ranking views. Consequently, MIS-level reports and plots
describe intrinsic candidate evidence and omit ranking-dependent identity.

### Reports

`Ranking.report()` is a complete, self-contained report. It preserves every
scientifically valid evidence family protected by the public reporting
non-regression contract.

`MISCandidate.report()` is an intrinsic report for one MIS: objectives,
structural metrics, and any stored linear, Pareto, nonlinear, and null-reference
evidence.

Existing `MISSet.report()` behavior remains available during this transition.
This ADR does not authorize shrinking or silently deleting report content.

### Compatibility

The following existing entry points remain supported:

- `misda.evaluate(mis_set, ...)`;
- `MISSet.report()`;
- `MISSet.graph_plot(...)`;
- `MISSet.front_plot(...)`;
- integer/index-oriented candidate selectors already accepted by the evaluator.

They are compatibility paths rather than the preferred user-facing workflow.

## Invariants

- `discover()` discovers; it does not accept a ranking policy.
- `evaluate()` enriches MIS evidence and never changes discovery or ordering.
- `rank()` creates a view and never copies candidate scientific state.
- `Ranking.mis()` returns an existing MIS object by identity.
- ranking coordinates never become intrinsic fields of an MIS.
- MIS-level reports/plots never trigger hidden evaluation.
- `Ranking.report()` must not lose evidence families present in the complete
  public report.
- no `heavy()` compatibility API is reintroduced.

## Rationale

The object model now mirrors the way a method user thinks about the task:

```text
discover data -> evaluate discovered MISs -> rank alternatives -> inspect one MIS
```

Canonical indices remain useful implementation details and compatibility
selectors, but users do not need them for the ordinary workflow.

## Verification

Regression tests must show that:

1. `MISSet.evaluate()` preserves `misda.evaluate()` defaults and state;
2. `Ranking.mis()` resolves tie level and local position without copying;
3. the evaluator accepts an MIS object directly as candidate scope;
4. `Ranking.report()` preserves the complete default report contract;
5. MIS reports and plots consume stored state only;
6. existing `MISSet` report and visualization entry points keep working;
7. the full acceptance gate remains green.
