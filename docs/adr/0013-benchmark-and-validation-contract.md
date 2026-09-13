# ADR 0013 — Benchmark and validation contract

- Status: Accepted
- Recorded: 2026-09-11 (retrospective)

## Context

MISDA needs scientific validation against synthetic or otherwise declared truth without contaminating the inference path. Benchmark declarations may describe expected dimensions, generating families, structural units, connected components, graphs, Pareto sets, and known diagnostic-backed limitations.

## Decision

Benchmarking is strictly downstream of discovery/evaluation. It compares declared truth with already computed outputs and never feeds truth back into the method.

The benchmark may compare or report, when declared:

- latent dimension;
- structural dimension;
- generating families (`families_expected`);
- structural units (`blocks_expected`);
- connected components (`components_expected`);
- graph structure;
- Pareto indices/evidence;
- other explicitly declared reference quantities.

Generating families, structural units, and connected components are distinct concepts. None may be inferred from another merely because their partitions happen to coincide in a particular diagnostic. In particular, `families_expected` describes the theoretical generating organization, `blocks_expected` declares an unambiguous structural-unit partition when one exists, and `components_expected` declares expected graph connectivity when independently justified.

Dimension declarations are likewise distinct from all three partitions. A generating family count, structural-unit count, or component count must not be substituted for latent or structural dimension.

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
- generating families, structural units, connected components, and dimensions remain separately declared concepts;
- expected dimension is never treated as expected component count;
- undeclared truth yields an explicit N/A state;
- known-failure exemptions are field-specific and diagnostic-backed;
- quick execution is not the normative scientific result.

## Current implementation

The benchmark infrastructure consumes `MISSet`/`Ranking` outputs and stored evidence. Controlled diagnostic truth reports generating families for every scenario, structural units only where the diagnostic specification provides an unambiguous partition, and graph components only when explicitly declared. The acceptance workflow runs the canonical benchmark in strict mode. Current project acceptance also exercises the static test suite, relevant slow tests, notebooks, the scientific battery, and comparative experiments.

## Forbidden shortcuts / regression risks

Do not tune discovery from expected values, silently infer missing declarations, treat generating families as structural units, infer components from either family or structural-unit partitions, globally waive a failing case because one field is known-problematic, or let quick-mode outcomes replace canonical scientific validation.

## Verification

Benchmark tests should deliberately vary declarations while keeping method outputs fixed, exercise N/A behavior, distinguish generating-family/structural-unit/component semantics (especially Case 5 and MOP-B), exercise expected-vs-unexpected mismatch classification, and confirm strict-mode failure semantics.