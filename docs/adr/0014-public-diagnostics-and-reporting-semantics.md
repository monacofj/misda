# ADR 0014 — Public diagnostics and reporting semantics

- Status: Accepted
- Recorded: 2026-09-11 (retrospective)

## Context

A scientifically correct implementation can still mislead users if reports conflate dimensions with topology, hide incomplete evaluation, rerun calculations implicitly, or present warning states as successful conclusions. Public diagnostics are therefore part of the architecture rather than mere formatting.

## Decision

Reports and plots are views over already stored state. They must not rerun scientific evaluation, consult benchmark truth, or mutate the result.

Public output must distinguish at least:

```text
discovery / thresholds / null-estimator completion
graph-derived dimensions
graph topology diagnostics
dimensional support
structural candidate ordering and tie groups
candidate evaluation families and scopes
observed-data Pareto stability
ranking-selected quantities
benchmark comparisons, when in benchmark context
```

### Dimension versus topology

Reports must keep

```text
structural_dimension = alpha(G+)
latent_dimension     = alpha(G±)
```

separate from structural/latent connected-component counts.

### Discovery support

Dimensional support is a diagnostic about whether observed data contradict the sufficiency of the graph-derived description. It is not an alternative dimension estimator. Current statuses are:

```text
SUPPORTED
PARTIALLY_SUPPORTED
UNSUPPORTED
```

Support is evaluated across the complete first scientific tie group so that deterministic within-tie ordering cannot decide scientific status.

Current diagnostic mechanisms are:

- `TRANSITIVE_CHAINING`: detects strong indirect positive paths whose direct relation to the retained candidate is substantially weaker beyond a permutation-null reference;
- `HIDDEN_SPECTRAL_STRUCTURE`: detects organized rank-correlation structure beyond the estimated latent signal dimension relative to a column-permutation null reference.

### Pareto stability versus dimensional support

Dimensional support and Pareto stability answer different questions and must never be collapsed into one status. A graph-derived dimension may be strongly supported while projection onto a representative MIS changes exact Pareto membership substantially.

When Pareto evaluation is requested, reports may expose observed-data diagnostics computed only from `Y`, including:

- observed-front fraction `|P_Y| / N`;
- range-normalized dominance-margin summaries for `P_Y`;
- range-normalized additive epsilon approximation from the selected reduced front `P_R` to `P_Y` in the complete objective space.

These quantities describe saturation, perturbation sensitivity, and geometric approximation respectively. They do not establish that measurement noise is present or caused an observed instability.

Until a separate data-driven decision rule is justified, reports must present these quantities continuously and explain their direction of interpretation. They must not introduce arbitrary `good`/`bad`, `safe`/`unsafe`, or equivalent fixed-threshold labels.

### Partial evaluation

If fewer than all candidates are evaluated for a requested metric family, reports must state the effective scope and selection basis. Partial evaluation is a scope fact, not by itself an error.

### Alpha-null completion and cancellation

ADR 0015 superseded the sequential `alpha_null` convergence rule. The public estimator now completes a fixed `B=N` empirical null-envelope experiment. Normal completion may retain the legacy `converged=True` compatibility field, but reports must not describe that state as convergence of a Monte Carlo mean or interpret retained degenerate intervals as uncertainty intervals.

If estimation is explicitly cancelled before the fixed budget is completed, the stored state must remain visibly incomplete (`converged=False`, `reason="CANCELLED"`). Reporting must not silently normalize cancellation into successful completion.

### Ranking-selected quantities

A ranking-selected dimension must be labeled as preference-derived and must not be presented as the structural graph dimension.

## Invariants

- reports/plots have no hidden scientific side effects;
- dimensions and component counts are separately named and interpreted;
- support diagnostics remain diagnostic rather than corrective;
- dimensional support and Pareto stability remain separate;
- Pareto stability uses only observed `Y` outside benchmark-only comparisons;
- no arbitrary Pareto-stability threshold is presented as a scientific conclusion;
- partial scope and explicit cancellation remain visible;
- legacy `alpha_null` fields are not given obsolete Monte Carlo-mean semantics;
- scientific ties are not erased by deterministic display ordering;
- benchmark truth appears only in benchmark reports;
- benchmark reports reuse the native `MISSet.report()` output verbatim for their MISDA-measures block.

## Current implementation

`MISSet.report()` and graph plotting consume stored analysis, candidates, support, evaluation-scope metadata, rankings, and stored Pareto-stability diagnostics. `graph_plot()` visualizes the already discovered graph state. Benchmark reports embed `MISSet.report()` unchanged as their data-derived MISDA block and append declared-reference comparisons downstream in a separate benchmark-validation block.

The current support implementation reuses common permutation work where practical, especially because `HIDDEN_SPECTRAL_STRUCTURE` is global for a fixed latent dimension while `TRANSITIVE_CHAINING` can vary by retained candidate.

## Permitted implementation variations

Formatting, plotting library, table layout, color scheme, and rendering medium may change freely if semantic labels and stored-state provenance remain correct.

## Forbidden shortcuts / regression risks

Do not relabel connected components as dimensions, select one tied candidate arbitrarily for scientific support, trigger hidden nonlinear evaluation from `report()`, omit partial-scope notes, reinterpret fixed-budget `alpha_null` completion as sequential Monte Carlo convergence, suppress explicit `alpha_null` cancellation, describe Pareto sensitivity as proof of measurement noise, independently re-render MISDA measures inside benchmark reporting, or turn continuous Pareto-stability diagnostics into fixed-threshold confidence labels without a separately justified data-driven rule.

## Verification

Regression tests should inspect semantic report fields/text for dimension-versus-topology separation, partial scope, support state, Pareto-stability provenance, ranking-selected dimension, exact embedding of `MISSet.report()` inside benchmark output, and correct fixed-budget/cancellation semantics for `alpha_null`. Plot tests should verify that plotting does not change stored results.
