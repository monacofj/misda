# ADR 0007 — Discovery, evaluation and ranking separation

- Status: Accepted
- Recorded: 2026-09-11 (retrospective)

## Context

Structural inference, empirical candidate evaluation, and preference among already discovered candidates answer different questions and have different computational costs. Conflating them risks making discovery depend on ranking policy or on which expensive metrics happen to have been evaluated.

## Decision

MISDA separates three operations:

```text
discover(Y)        -> establishes the structural universe
 evaluate(MISSet)  -> adds evidence to existing candidates
 rank(MISSet)      -> creates an ordered view over existing candidates
```

`discover()` owns validation, correlation statistics, thresholds, graphs, graph-derived dimensions, complete MIS enumeration, structural metrics, canonical structural ordering, and dimensional-support diagnostics.

`evaluate()` may attach requested candidate evidence but must not discover candidates, change graph structure, change dimensions, or reorder the canonical `MISSet`.

`rank()` may order candidates according to a declared policy but must not mutate the `MISSet` or influence discovery retrospectively.

## Rationale

The separation makes scientific provenance explicit. A user can distinguish what the method inferred structurally from what was computed later for comparison or preference. It also permits expensive evaluation to be scoped without changing the method's candidate universe.

## Invariants

- ranking policy is not an input to structural discovery;
- evaluation scope is not an input to structural discovery;
- candidate identity and canonical position survive subsequent evaluations and rankings;
- repeated evaluation of an already available metric family is idempotent with respect to discovery state;
- ranking is a view, not ownership of candidate data.

## Current implementation

The public facade exposes `misda.discover`, `misda.evaluate`, and `misda.rank`. `MISSet` owns the discovered universe, `MISCandidate` owns candidate properties/evidence, and `Ranking` stores policy-dependent ordering and tie groups while referencing the original `MISSet`.

## Permitted implementation variations

Internal mutation used solely to cache newly computed evidence is permitted if it cannot alter discovery outputs or candidate identity. Purely functional implementations returning new evidence-bearing objects are also compatible.

## Forbidden shortcuts / regression risks

Do not allow ranking to select which MISs are enumerated, let evaluation results change graph dimensions, or trigger hidden expensive evaluation from reporting/ranking unless a future explicit API contract authorizes it.

## Verification

Tests should snapshot discovery outputs before and after `evaluate()`/`rank()` and verify graph edge sets, dimensions, candidate membership, and canonical order remain unchanged.