# MISDA Architecture Decision Records

This directory is the authoritative architectural and methodological specification of MISDA.

The ADRs serve two purposes simultaneously:

1. they must contain enough information for an independent implementation of MISDA without reverse-engineering the source code;
2. they define invariants that future refactors and performance optimizations must preserve unless a later ADR explicitly supersedes them.

Each ADR distinguishes, when applicable, the normative contract from the current implementation. Implementation details may change when the stated invariants are preserved.

## ADR structure

Each record may contain: Context, Definitions, Decision, Formal specification, Rationale, Invariants, Current implementation, Permitted implementation variations, Forbidden shortcuts / regression risks, Verification, Computational consequences, and References.

## Index

- [0001 — MISDA methodological model](0001-misda-methodological-model.md)
- [0002 — Data-driven inference boundary](0002-data-driven-inference-boundary.md)
- [0003 — Signed dependence graph model](0003-signed-dependence-graph-model.md)
- [0004 — Dimensionality from graph independence](0004-dimensionality-from-graph-independence.md)
- [0005 — Complete structural MIS discovery](0005-complete-structural-mis-discovery.md)
- [0006 — Candidate metrics and evidence model](0006-candidate-metrics-and-evidence-model.md)
- [0007 — Discovery, evaluation and ranking separation](0007-discovery-evaluation-ranking-separation.md)
- [0008 — Canonical structural coverage ordering](0008-canonical-structural-coverage-ordering.md)
- [0009 — Explicit evaluation scope](0009-explicit-evaluation-scope.md)
- [0010 — Decision-stable sequential alpha-null estimation](0010-decision-stable-alpha-null.md)
- [0011 — Public API and object model](0011-public-api-and-object-model.md)
- [0012 — Reproducibility and stochastic computation](0012-reproducibility-and-stochastic-computation.md)
- [0013 — Benchmark and validation contract](0013-benchmark-and-validation-contract.md)
- [0014 — Public diagnostics and reporting semantics](0014-public-diagnostics-and-reporting-semantics.md)

All records below are retrospective unless explicitly stated otherwise. They document the current MISDA design as of 2026-09-11.