# ADR 0009 — Explicit evaluation scope

- Status: Accepted
- Recorded: 2026-09-11 (retrospective)

## Context

Candidate evidence has heterogeneous cost. Linear and Pareto evaluation are comparatively moderate; nonlinear reconstruction is substantially more expensive. Cost control must not be achieved by shrinking structural discovery.

## Decision

Evaluation scope is explicit and belongs to one `evaluate()` call as a whole. Supported selectors include all candidates, a canonical prefix, explicit canonical indices, and a `Ranking` slice.

Default scope is:

- all candidates for calls containing only cheap/moderate families such as linear and Pareto;
- one candidate for any call containing nonlinear evaluation.

For mixed calls, every requested family uses the same effective scope.

Any scope smaller than the complete candidate universe must remain explicitly reportable as partial evaluation.

## Invariants

- evaluation scope never changes discovery outputs;
- one call has one effective scope shared by all requested families;
- partial evidence is never represented as if all candidates were evaluated;
- users may request broader expensive evaluation explicitly.

## Current implementation

`evaluate(..., candidates=...)` accepts `"all"`, an integer prefix length, explicit indices, or a `Ranking`/slice. Reports retain scope metadata and describe partial evaluation as a scope note rather than an error warning.

## Forbidden shortcuts / regression risks

Do not use evaluation scope as a discovery cap, silently skip expensive candidates while claiming full coverage, or apply different hidden scopes to different metric families within one call.

## Verification

Tests should verify identical discovery state under different evaluation scopes and exact accounting of which candidate-family pairs were evaluated.