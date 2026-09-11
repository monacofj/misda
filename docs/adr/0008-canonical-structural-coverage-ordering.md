# ADR 0008 — Canonical structural coverage ordering

- Status: Accepted
- Recorded: 2026-09-11 (retrospective)

## Context

Complete MIS discovery can produce multiple structurally valid candidates. MISDA needs a deterministic natural order for presentation, reproducible positional identity, support diagnostics, and default selection without making ranking an input to discovery.

## Decision

The canonical structural policy is named `structural_coverage` and compares candidates lexicographically by:

```text
size                  descending
neighborhood          descending
avg_external_degree   descending
span                  descending
```

A deterministic label-based tie-break may order otherwise equal candidates reproducibly, but it must not create a scientific distinction. Candidates equal on all four scientific criteria belong to the same rank group.

The default public ranking is equivalent to `policy="structural_coverage"`.

## Rationale

Cardinality first preserves candidates with the largest number of mutually non-redundant objectives. The remaining criteria prefer candidates that structurally cover more excluded objectives and stronger external adjacency. Explicit tie groups prevent arbitrary deterministic ordering from being mistaken for scientific evidence.

## Invariants

- the four scientific criteria and their directions define the current canonical policy;
- deterministic tie-breaking is operational only;
- scientific rank groups are determined before deterministic within-group order;
- discovery does not accept an alternative user ranking policy;
- no alternative ranking policy is currently part of the normative method.

## Current implementation

Structural metrics are computed for every MIS, and the candidate list is canonically ordered according to the criteria above. `Ranking.groups` exposes scientific tie groups. `misda.rank(mis_set)` materializes a ranking over the existing canonical candidate universe.

## Permitted implementation variations

Sorting algorithms and deterministic label encodings may change if they preserve criterion values, scientific tie groups, and reproducible candidate mapping.

## Forbidden shortcuts / regression risks

Do not use floating implementation noise to split ties, let deterministic labels become scientific ranking criteria, or let ranking prune the discovery universe.

## Verification

Tests should include exact ties and verify that within-tie deterministic order may be stable while all tied members remain in the same scientific group.