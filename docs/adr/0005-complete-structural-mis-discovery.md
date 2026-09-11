# ADR 0005 — Complete structural MIS discovery

- Status: Accepted
- Recorded: 2026-09-11 (retrospective)

## Context

A maximal independent set (MIS) of `G+` is a structurally irreducible candidate: no additional objective can be added without creating a supported positive-redundancy pair. Different MISs can represent genuinely different structural alternatives.

Performance pressure, especially on graphs with many MISs, creates a temptation to truncate enumeration. That would alter the candidate universe and therefore the method itself.

## Definitions

An independent set `S` is maximal if no strict superset of `S` is independent. Maximality is local with respect to set inclusion and is distinct from maximum cardinality.

The MISDA structural candidate universe is

```text
C(G+) = {S subseteq V : S is a maximal independent set of G+}.
```

## Decision

`discover()` must enumerate the complete set `C(G+)`.

The candidate universe must not depend on ranking policy, evaluation budget, candidate prefix, runtime convenience, or expensive downstream metrics.

## Rationale

Truncating MIS discovery would make later rankings and comparisons conditional on an implementation accident: a candidate omitted during discovery could never be recovered downstream. Complete enumeration preserves the distinction between structural discovery and preference among candidates.

## Current implementation

MISDA transforms the MIS problem into maximal-clique enumeration on the complement graph. `misda._graph.find_maximal_independent_sets()` currently uses a pivoted Bron-Kerbosch recursion on the complement adjacency matrix, then deduplicates and deterministically sorts the returned vertex sets.

The complement identity is:

```text
S is a maximal independent set of G
<=>
S is a maximal clique of complement(G).
```

## Invariants

- every maximal independent set of `G+` appears exactly once;
- no non-maximal independent set appears as a structural candidate;
- discovery completeness is independent of candidate evaluation scope;
- ranking may reorder the complete universe but must not determine which MISs exist.

## Permitted implementation variations

Exact pivoting strategies, degeneracy ordering, graph decomposition, bitset kernels, memoization, streaming internally, parallel enumeration, or any other exact algorithm are permitted if the final candidate set is identical.

## Forbidden shortcuts / regression risks

The following are not permitted as silent optimizations:

- `max_mis` / `max_evaluated_mis`-style limits on discovery;
- first-`k` enumeration;
- timeout-based partial candidate sets reported as complete;
- rank-guided pruning that can remove valid MISs;
- random sampling of MISs;
- replacing maximal-set enumeration by one maximum independent set.

If bounded or approximate enumeration is ever introduced, it requires a separate explicit methodological decision and must not masquerade as the current MISDA discovery semantics.

## Verification

For reference graphs, tests should compare the returned candidate set with an independent exact enumerator or known closed-form MIS set, not merely candidate count or selected candidate. Empty, complete, path, cycle, block, and mixed graphs should be covered.

## Computational consequences

The number of maximal independent sets may be exponential in the number of vertices; equivalently, maximal-clique enumeration has exponential worst-case output size. This is an intrinsic output-sensitive cost, not justification for silent semantic truncation.

Moon and Moser showed that an `n`-vertex graph can contain up to `3^(n/3)` maximal cliques, and therefore a graph can contain the same order of maximal independent sets by complementation.

## References

- Bron, C.; Kerbosch, J. (1973). Algorithm 457: finding all cliques of an undirected graph. *Communications of the ACM*, 16(9), 575–577.
- Tomita, E.; Tanaka, A.; Takahashi, H. (2006). The worst-case time complexity for generating all maximal cliques and computational experiments. *Theoretical Computer Science*, 363, 28–42.
- Moon, J. W.; Moser, L. (1965). On cliques in graphs. *Israel Journal of Mathematics*, 3, 23–28.