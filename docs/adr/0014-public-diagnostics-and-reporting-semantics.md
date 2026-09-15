# ADR 0014 — Public diagnostics and reporting semantics

- Status: Accepted
- Recorded: 2026-09-11 (retrospective)

## Context

A scientifically correct implementation can still mislead users if reports conflate dimensions with topology, hide incomplete evaluation, rerun calculations implicitly, or present warning states as successful conclusions. Public diagnostics are therefore part of the architecture rather than mere formatting.

## Decision

Reports and plots are views over already stored state. They must not rerun scientific evaluation, consult benchmark truth, or mutate the result.

Public output must distinguish at least:

```text
discovery / thresholds / convergence
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

### Nonconvergence

If `alpha_null` reaches `B_max` without decision stability, the result must retain `converged=False` and the machine-readable reason and the public call must warn. Reporting must not silently normalize this state into successful convergence.

### Ranking-selected quantities

A ranking-selected dimension must be labeled as preference-derived and must not be presented as the structural graph dimension.

## Invariants

- reports/plots have no hidden scientific side effects;
- dimensions and component counts are separately named and interpreted;
- support diagnostics remain diagnostic rather than corrective;
- dimensional support and Pareto stability remain separate;
- Pareto stability uses only observed `Y` outside benchmark-only comparisons;
- no arbitrary Pareto-stability threshold is presented as a scientific conclusion;
- partial scope and nonconvergence remain visible;
- scientific ties are not erased by deterministic display ordering;
- benchmark truth appears only in benchmark reports.

## Current implementation

`MISSet.report()` and graph plotting consume stored analysis, candidates, support, evaluation-scope metadata, rankings, and stored Pareto-stability diagnostics. `graph_plot()` visualizes the already discovered graph state. Benchmark reports add declared-reference comparisons downstream while retaining the observed-data stability block as a truth-independent diagnostic.

The current support implementation reuses common permutation work where practical, especially because `HIDDEN_SPECTRAL_STRUCTURE` is global for a fixed latent dimension while `TRANSITIVE_CHAINING` can vary by retained candidate.

## Permitted implementation variations

Formatting, plotting library, table layout, color scheme, and rendering medium may change freely if semantic labels and stored-state provenance remain correct.

## Forbidden shortcuts / regression risks

Do not relabel connected components as dimensions, select one tied candidate arbitrarily for scientific support, trigger hidden nonlinear evaluation from `report()`, omit partial-scope notes, suppress `alpha_null` nonconvergence, describe Pareto sensitivity as proof of measurement noise, or turn continuous Pareto-stability diagnostics into fixed-threshold confidence labels without a separately justified data-driven rule.

## Verification

Regression tests should inspect semantic report fields/text for dimension-versus-topology separation, partial scope, support state, Pareto-stability provenance, ranking-selected dimension, and nonconvergence. Plot tests should verify that plotting does not change stored results.
