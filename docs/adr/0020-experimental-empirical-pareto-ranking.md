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

Add an experimental ranking policy named `pareto_retention`.

Its scientific rank value is only:

```text
pareto_retention    descending
```

Candidate cardinality is not evidence of Pareto preservation and therefore does
not enter the scientific rank. When two candidates have exactly the same
retention, the smaller MIS is ordered first as an operational
reduction-efficiency tie-break, followed by deterministic objective labels.
Neither operational tie-break splits the scientific tie group.

The canonical default remains `size_span`. Discovery order, graph dimensions,
MIS enumeration, and `MISSet.structural_ranking` are unchanged.

## Evaluation contract

`pareto_retention` depends on stored Pareto evidence. The explicit workflow is:

```python
mis_set.evaluate(metrics=("pareto",), candidates="all")
ranking = misda.rank(mis_set, policy="pareto_retention")
```

Alternatively, `accept_cost=True` authorizes evaluation of missing Pareto
evidence for the requested ranking scope. Without stored evidence or explicit
cost authorization, ranking fails rather than performing hidden computation.

## Projection identity

Under the current same-sample minimization contract, objective removal can only
add dominance relations. Any point dominated in the full objective space
remains dominated in every objective subset. Therefore

```text
ND(Y_S) ⊆ ND(Y)
```

for every retained objective subset `S`.

It follows exactly that, whenever the fronts are non-empty:

```text
pareto_validity = 1
pareto_jaccard  = pareto_retention
```

Thus retention is the only independent set-membership preservation signal among
those three metrics for this ranking problem. Validity and Jaccard remain
reported for compatibility and explicit diagnostics but must not be interpreted
as independent ranking evidence.

## Interpretation

`pareto_retention` is recall of the empirical nondominated row set after
objective projection. It uses only observed `Y`; benchmark truth, analytical
fronts, and optimizer results are never inputs to the policy. High retention
is therefore evidence of observed dominance preservation, not proof of global
optimization equivalence.

## Dimensional support

Dimensional support is currently attached during discovery to the canonical
first `size_span` tie group. An alternative ranking may select a candidate
outside that group. `pareto_retention` does not silently recompute or reinterpret
support, and reports must state the scope distinction.

## Limitation exposed by variable-cardinality control

A later analytical control contains two globally optimization-safe MISs of
different sizes: a singleton and a two-objective MIS. On finite Sobol clouds,
the larger safe MIS has perfect empirical Pareto retention while the smaller
safe MIS has low retention, even though both preserve exactly the same true
continuous Pareto set.

Therefore `pareto_retention` is explicitly an **observed-cloud preservation
ranking**, not an estimator of the smallest globally optimization-safe
objective subset. A stronger global pairwise dominance-distortion diagnostic
improves DTLZ5/DPF1 control selection but exhibits the same conservative
cardinality behavior on this control.

This limitation is informational rather than an implementation defect:
generic finite samples need not contain the true Pareto manifold. Optimization
safety/minimal safe cardinality requires additional evidence from the MOP,
targeted Pareto sampling, analytical structure, or optimizer-based validation.

## Invariants

- `size_span` remains the canonical default and immutable MISSet order.
- `pareto_retention` creates a Ranking view and never reorders the MISSet.
- structural and latent dimensions are unchanged by ranking policy.
- benchmark truth is never consulted by `pareto_retention`.
- missing Pareto evidence never triggers hidden work unless `accept_cost=True`.
- empirical Pareto retention must not be described as a guarantee of future
  optimization safety.
