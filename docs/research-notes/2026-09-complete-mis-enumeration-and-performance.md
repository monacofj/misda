# Complete MIS enumeration and performance pressure

- Period: August–September 2026
- Status: **Adopted later**
- Normative outcome: ADR 0005 — Complete structural MIS discovery

## Question

Some controlled cases, especially dense/block-structured graphs, made maximal-independent-set enumeration expensive enough that runtime became a practical concern. This naturally suggested placing a limit such as `max_mis` or `max_evaluated_mis` on discovery.

The crucial question was whether such a limit would be a harmless engineering optimization or a change in the scientific method.

## Investigation

MISDA treats every maximal independent set of `G+` as a structurally irreducible candidate. Therefore the candidate universe is itself part of the method:

```text
C(G+) = all maximal independent sets of G+
```

A prefix limit, timeout, or arbitrary cap can omit valid candidates. Any ranking or downstream evaluation would then operate on an implementation-dependent subset rather than on the structural universe defined by the graph.

This matters even if only one candidate is ultimately evaluated expensively: evaluation scope and discovery completeness are separate concepts. Limiting the expensive evaluation of candidates is not equivalent to limiting which MISs exist.

## Alternatives considered

The following shortcuts were discussed or considered under performance pressure:

- cap the number of enumerated MISs;
- stop after the first `k` candidates;
- limit `max_evaluated_mis` during discovery;
- use a runtime ceiling and return a partial set;
- select one maximum independent set rather than enumerate all maximal sets.

These were rejected for the current method because they change the candidate universe.

Exact implementation improvements remain legitimate: pivoting, graph decomposition, degeneracy ordering, bitsets, memoization, parallelism, or any other exact/output-sensitive method that returns the same complete set.

## Outcome

The methodological response to Case 3-style performance pressure was **not** to truncate the universe. Complete enumeration was retained and later made normative in ADR 0005.

The accepted distinction is:

- discovery must remain complete;
- expensive downstream metrics may have an explicit evaluation scope;
- if approximate/bounded discovery is ever introduced, it must be a separately named method with explicit incomplete semantics.

## Why keep this note

The same performance problem will recur as dimensionality grows because the number of maximal independent sets can be exponential. A future contributor is likely to rediscover candidate caps as the simplest fix. This note records that such caps were considered and rejected because they solve a runtime problem by silently changing the estimand.
