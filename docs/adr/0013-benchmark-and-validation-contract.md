# ADR 0013 — Benchmark and validation contract

- Status: Accepted
- Recorded: 2026-09-11 (retrospective)

## Context

MISDA needs scientific validation against synthetic or otherwise declared truth without contaminating the inference path. Benchmark declarations may describe expected dimensions, structural/generating units, connected components, graphs, Pareto sets, and known diagnostic-backed limitations.

## Decision

Benchmarking is strictly downstream of discovery/evaluation. It compares declared truth with already computed outputs and never feeds truth back into the method.

The benchmark may compare, when declared:

- latent dimension;
- structural dimension;
- structural/generating blocks;
- connected components;
- graph structure;
- Pareto indices/evidence;
- other explicitly declared reference quantities.

Dimension declarations and component declarations are distinct. `blocks_expected` and `components_expected` are also distinct and must not be inferred from one another.

Dimension accuracy includes absolute error, relative error where defined, and exact-match status.

For finite sets `A` and `B`, block/set agreement may use Jaccard similarity

```text
J(A,B) = |A intersect B| / |A union B|.
```

If a reference quantity was not declared, the report must say that it is unavailable/not declared rather than fabricate an expectation.

## Expected mismatches

A benchmark may declare a field-specific expected mismatch only when it represents a known methodological limitation. Such a mismatch may be classified as expected only if the declared reason is actually present among MISDA's internal dimensional-support diagnostics for that result. Otherwise it remains an unexpected mismatch.

## Scientific acceptance

The canonical scientific battery uses the notebook/reference sample size `N=300` and evaluates cheap candidate evidence over its normal complete scope. Reduced `--quick` runs are smoke tests only and must not redefine the scientific baseline.

Strict acceptance fails on unexpected declaration mismatches and must identify case, field, observed value, expected value, and reason. Expected diagnostic-backed mismatches remain visible but do not fail the scientific gate.

## Invariants

- benchmark truth is read only by benchmark code;
- expected dimension is never treated as expected component count;
- undeclared truth yields an explicit N/A state;
- known-failure exemptions are field-specific and diagnostic-backed;
- quick execution is not the normative scientific result.

## Current implementation

The benchmark infrastructure consumes `MISSet`/`Ranking` outputs and stored evidence. The acceptance workflow runs the canonical benchmark in strict mode. Current project acceptance also exercises the static test suite, relevant slow tests, notebooks, the scientific battery, and comparative experiments.

## Forbidden shortcuts / regression risks

Do not tune discovery from expected values, silently infer missing declarations, globally waive a failing case because one field is known-problematic, or let quick-mode outcomes replace canonical scientific validation.

## Verification

Benchmark tests should deliberately vary declarations while keeping method outputs fixed, exercise N/A behavior, expected-vs-unexpected mismatch classification, and confirm strict-mode failure semantics.