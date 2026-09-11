# ADR 0002 — Data-driven inference boundary

- Status: Accepted
- Recorded: 2026-09-11 (retrospective)

## Context

MISDA is intended to infer dimensional structure from observed objective data without access to generating truth. Synthetic benchmark cases may know expected dimensions, blocks, graphs, or Pareto sets, but those declarations are validation information rather than method inputs.

## Decision

The information flow is strictly one-way:

```text
Y -> discover() -> MISSet -> evaluate()/rank() -> benchmark()
```

Benchmark declarations, expected dimensions, generating partitions, expected graphs, known limitations, or any other external truth must never influence threshold estimation, graph construction, MIS discovery, dimensional-support diagnostics, candidate evaluation, or ranking.

## Rationale

Allowing truth to enter discovery would make benchmark success circular and would prevent MISDA from being applied to real problems where truth is unavailable. The same code path must be used whether the data originate from a benchmark with known structure or from an unknown empirical problem.

## Invariants

- `discover()` depends only on observed data and explicit method controls.
- `evaluate()` depends only on the discovered result, observed data retained by that result, and explicit evaluation controls.
- `rank()` depends only on the `MISSet`, its computed evidence, and the selected policy.
- benchmark declarations are read only by benchmark infrastructure.
- dimensional-support diagnostics are internal evidence, not benchmark-informed corrections.

## Current implementation

Benchmark truth is represented separately from the `MISSet`. Benchmark comparison occurs after discovery and, where required, after candidate evidence has already been computed. The benchmark evaluator compares stored outputs with declarations but does not call back into discovery with those declarations.

## Forbidden shortcuts / regression risks

Do not use expected blocks to repair a graph, expected dimension to choose a threshold, known benchmark failures to select candidates, or expected Pareto indices to tune reduction.

## Verification

Tests should ensure that changing benchmark declarations while holding `Y`, seeds, and method controls fixed cannot change `discover()`, `evaluate()`, or `rank()` outputs.

## Consequences

Known methodological limitations remain visible as mismatches rather than being hidden by benchmark-specific logic. This is deliberate and is the basis for honest scientific validation.