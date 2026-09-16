# ADR 0008 — Canonical structural coverage ordering

- Status: Superseded by ADR 0017
- Recorded: 2026-09-11 (retrospective)

## Context

Complete MIS discovery can produce multiple structurally valid candidates. MISDA needs a deterministic natural order for presentation, reproducible positional identity, support diagnostics, and default selection without making ranking an input to discovery.

## Decision

The canonical structural policy was named `structural_coverage` and compared candidates lexicographically by:

```text
size                  descending
neighborhood          descending
avg_external_degree   descending
span                  descending
```

A deterministic label-based tie-break ordered otherwise equal candidates reproducibly, but did not create a scientific distinction. Candidates equal on all four scientific criteria belonged to the same rank group.

The default public ranking was equivalent to `policy="structural_coverage"`.

## Rationale at the time

Cardinality first preserved candidates with the largest number of mutually non-redundant objectives. The remaining criteria were intended to prefer candidates that structurally covered more excluded objectives and stronger external adjacency. Explicit tie groups prevented arbitrary deterministic ordering from being mistaken for scientific evidence.

## Supersession

ADR 0017 established that, for maximal independent sets, `neighborhood = n - size` and `avg_external_degree = span / size`. The four-criterion key therefore contained no additional scientific discrimination beyond `size` and `span`.

ADR 0017 replaces this policy with the canonical `size_span` ordering while preserving the effective candidate ordering and scientific tie groups.

## Historical invariants

- the four scientific criteria and their directions defined the former canonical policy;
- deterministic tie-breaking was operational only;
- scientific rank groups were determined before deterministic within-group order;
- discovery did not accept an alternative user ranking policy;
- no alternative ranking policy was part of the normative method.

## Historical implementation

Structural metrics were computed for every MIS, and the candidate list was canonically ordered according to the criteria above. `Ranking.groups` exposed scientific tie groups. `misda.rank(mis_set)` materialized a ranking over the existing canonical candidate universe.

## Verification

Historical regression tests may retain the former key only to demonstrate equivalence with the simplified `size_span` policy defined by ADR 0017.
