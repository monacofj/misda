# ADR 0004 — Dimensionality from graph independence

- Status: Accepted
- Recorded: 2026-09-11 (retrospective)

## Context

Connected components describe graph topology, not the maximum number of mutually independent objectives. In connected non-clique graphs such as chains, multiple vertices may remain mutually nonadjacent even though all vertices belong to one component.

## Definitions

For graph `G=(V,E)`, an independent set `S subseteq V` satisfies

```text
for all u != v in S: (u,v) not in E.
```

The independence number is

```text
alpha(G) = max {|S| : S is an independent set of G}.
```

## Decision

MISDA defines:

```text
original_dimension   = M
structural_dimension = alpha(G+)
latent_dimension     = alpha(G±)
```

Connected-component counts are distinct topology diagnostics:

```text
structural_components = connected_components(G+)
latent_components     = connected_components(G±).
```

They must never be substituted for the dimensional quantities above.

## Rationale

`alpha(G+)` is the largest number of objectives that can coexist without a supported positive-redundancy edge. `alpha(G±)` is the largest number that can coexist without any supported signed dependence edge. These quantities correspond directly to the graph semantics established by ADR 0003.

A chain illustrates why components are insufficient: a connected path can contain several pairwise nonadjacent vertices. Connectivity therefore cannot imply dimension one.

## Invariants

- dimensions are exact graph independence numbers;
- component counts remain topology-only diagnostics;
- connectedness, clique count, or number of components cannot replace `alpha(G)`;
- any optimization of the independence-number computation must remain exact unless a future ADR explicitly changes the scientific contract.

## Current implementation

`misda._graph.independence_number()` computes the exact independence number by constructing the complement graph and taking the largest maximal clique returned by `networkx.find_cliques()`.

This uses the identity

```text
S independent in G  <=>  S clique in complement(G).
```

## Permitted implementation variations

Any exact maximum-independent-set or maximum-clique algorithm may replace the current implementation, including branch-and-bound, decomposition, memoization, or parallel exact methods.

## Forbidden shortcuts / regression risks

Do not use connected-component count, greedy independent sets, approximate maximum cliques, or the size of an arbitrarily selected maximal independent set as the dimension.

## Verification

Tests must include graphs where component count and independence number differ, especially paths/chains, as well as complete, empty, block, and mixed graphs. MOP-B-type cases must distinguish two independent retained vertices inside one connected component from dimension one.

## References

- West, D. B. (2001). *Introduction to Graph Theory*, 2nd ed. Prentice Hall.
- Bron, C.; Kerbosch, J. (1973). Algorithm 457: finding all cliques of an undirected graph. *Communications of the ACM*, 16(9), 575–577.