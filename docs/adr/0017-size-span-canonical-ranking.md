# ADR 0017 — Size-span canonical structural ranking

- Status: Accepted
- Recorded: 2026-09-16
- Supersedes: ADR 0008
- Tracking issue: #53

## Context

ADR 0008 defined the canonical structural ranking policy `structural_coverage`
as a lexicographic ordering by `size`, `neighborhood`,
`avg_external_degree`, and `span`, all descending.

A later review showed that this key contains mathematical redundancies for the
objects actually being ranked: maximal independent sets of the positive
structural graph `G+`.

Let `S` be a maximal independent set of a graph with `n` vertices. Every vertex
outside `S` must be adjacent to at least one vertex in `S`; otherwise that
vertex could be added to `S`, contradicting maximality. Therefore

```text
neighborhood(S) = n - size(S).
```

Consequently, after `size` has tied, `neighborhood` must also tie.

The structural metric `span` is the sum of external degrees of the vertices in
`S`, while `avg_external_degree` is that same sum divided by `size(S)`:

```text
avg_external_degree(S) = span(S) / size(S).
```

After `size` has tied, ordering by `avg_external_degree` is therefore exactly
the same as ordering by `span`.

The former four-criterion policy was thus scientifically equivalent to a
simpler two-criterion policy.

## Decision

The canonical structural ranking policy is named `size_span` and compares
candidates lexicographically by:

```text
size   descending
span   descending
```

A deterministic label-based tie-break may order otherwise equal candidates for
reproducibility, but it does not create a scientific distinction. Candidates
with equal `size` and equal `span` belong to the same scientific rank group.

`misda.rank(mis_set)` is equivalent to
`misda.rank(mis_set, policy="size_span")`.

The previous exported constant name `STRUCTURAL_COVERAGE` may remain as a
compatibility alias, but its value resolves to the canonical `size_span` policy.
The literal policy name `"structural_coverage"` is not normative.

## Rationale

The policy name should describe the quantities that actually determine the
scientific ranking. Retaining redundant criteria in the ranking key gives the
false impression that multiple independent forms of structural coverage are
being combined when, for maximal independent sets, they cannot change the
ordering.

Cardinality remains first because it selects the largest mutually independent
objective subsets. `span` then distinguishes candidates of equal cardinality by
the total number of positive structural adjacencies connecting retained and
excluded objectives.

## Descriptive structural metrics

This decision changes the ranking key, not the candidate metric model.
`neighborhood`, `neighborhood_ratio`, `avg_external_degree`, and
`avg_internal_degree` may remain available as descriptive or audit quantities.
Their presence in reports or the public object model must not imply that they
are independent ranking criteria.

For every maximal independent set with at least one excluded vertex,
`neighborhood_ratio` is one. For an independent set, `avg_internal_degree` is
zero. These identities should be considered when deciding which structural
metrics are useful to display.

## Invariants

- the canonical policy name is `size_span`;
- scientific ordering is determined only by `size` descending and then `span`
  descending;
- scientific tie groups are determined by equality of `size` and `span`;
- deterministic label ordering is operational only and must not split a
  scientific tie group;
- complete MIS discovery is unchanged;
- ranking does not prune or alter the discovered candidate universe;
- `structural_dimension` remains the independence number of `G+` and is not
  redefined by ranking.

## Behavioral preservation

For maximal independent sets, replacing the ADR 0008 key

```text
size, neighborhood, avg_external_degree, span
```

with

```text
size, span
```

must preserve candidate ordering and scientific tie groups, apart from the
reported policy name. This is a semantic simplification, not a new preference
rule.

## Verification

Tests should verify:

1. the default and explicit public policy name is `size_span`;
2. the sort key is equivalent to descending `size`, then descending `span`;
3. candidates with equal `size` and `span` remain in the same scientific tie
   group even when deterministic labels differ;
4. representative graphs produce the same candidate order under the old
   redundant key and the simplified key;
5. benchmark serialization and public reports expose `size_span` consistently.
