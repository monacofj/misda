# ADR 0020 — Experimental empirical-Pareto ranking

- Status: Experimental
- Recorded: 2026-09-25
- Does not supersede: ADR 0017

## Context

The canonical `size_span` policy is a structural heuristic. The optimization
benchmark asks a stronger operational question: among structurally admissible
MISs, which retained objective subset best preserves observed trade-offs that
matter for optimization?

A controlled probe compares `size_span`, positive pairwise-correlation
coverage, linear reconstruction, and empirical Pareto-front retention on the
same independent Sobol screening sample.

For DTLZ5 with M=10, the analytically optimization-safe pair `f9,f10` is last
under the tested correlation and linear-reconstruction orderings, while it has
the highest observed Pareto retention. The explicit unsafe witness `f1,f10`
has lower retention.

For DPF1 with D=2 and M=10, the analytical base pair `f1,f2` is eleventh in
the canonical `size_span` order and poor under pairwise correlation coverage,
but it uniquely preserves the complete observed Pareto front in the probe.
For DTLZ2, where every non-empty proper objective subset is analytically
unsafe, none of the discovered candidates achieves exact observed Pareto
preservation.

These observations do not establish that empirical Pareto retention implies
optimization safety. They do establish that correlation/reconstruction quality
and optimization-preserving objective selection are distinct properties.

## Decision

Add an experimental ranking policy named `size_pareto` with lexicographic key:

```text
size                descending
pareto_retention    descending
```

A deterministic objective-label tie-break orders otherwise equal candidates
without splitting a scientific tie group.

The canonical default remains `size_span`. Discovery order, graph dimensions,
MIS enumeration, and `MISSet.structural_ranking` are unchanged.

## Evaluation contract

`size_pareto` depends on stored Pareto evidence. The explicit workflow is:

```python
mis_set.evaluate(metrics=("pareto",), candidates="all")
ranking = misda.rank(mis_set, policy="size_pareto")
```

Alternatively, `accept_cost=True` authorizes evaluation of missing Pareto
evidence for the requested ranking scope. Without stored evidence or explicit
cost authorization, ranking fails rather than performing hidden computation.

## Interpretation

`pareto_retention` is recall of the empirical nondominated row set after
objective projection. It uses only observed `Y`; benchmark truth, analytical
fronts, and optimizer results are never inputs to the policy. High retention
is therefore evidence of observed dominance preservation, not proof of global
optimization equivalence.

## Dimensional support

Dimensional support is currently attached during discovery to the canonical
first `size_span` tie group. An alternative ranking may select a candidate
outside that group. `size_pareto` does not silently recompute or reinterpret
support, and reports must state the scope distinction.

## Invariants

- `size_span` remains the canonical default and immutable MISSet order.
- `size_pareto` creates a Ranking view and never reorders the MISSet.
- structural and latent dimensions are unchanged by ranking policy.
- benchmark truth is never consulted by `size_pareto`.
- missing Pareto evidence never triggers hidden work unless `accept_cost=True`.
- empirical Pareto retention must not be described as a guarantee of future
  optimization safety.
