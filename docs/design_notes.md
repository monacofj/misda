<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# MISDA static design notes

These notes explain the methodological boundaries behind the current static
implementation. Normative decisions live in `docs/decisions.md`.

## 1. Positive structure and signed dependence

Positive and negative dependence have different meanings for objective
reduction. A supported positive association can indicate redundancy; a
supported negative association expresses dependence/conflict but must not be
turned into a positive-redundancy edge.

MISDA therefore builds two projections at the same threshold:

```text
G+   supported positive dependence
G±   supported positive and negative dependence
```

Dimensions are exact graph independence numbers:

```text
structural_dimension = alpha(G+)
latent_dimension     = alpha(G±)
```

Connected-component counts describe topology only. This distinction matters for
chains and other connected non-clique structures, where one connected component
can still contain several mutually independent vertices.

Constant objectives remain explicit isolated vertices because a constant column
cannot support a valid pairwise Fisher-z test.

## 2. Statistical thresholds and empirical null envelope

The static layer stores probability evidence in log space to avoid underflow.
Two data-derived endpoints delimit the selected threshold:

- `alpha_onset`: first observed positive structural event;
- `alpha_null`: permutation-null endpoint derived from the empirical upper
  envelope of maximum positive correlations under independently permuted
  objective columns.

`aggressiveness` interpolates between these endpoints in log space.

For a dataset with `N` rows, null estimation uses exactly `B=N` independent-
column permutations. If `m_b` is the maximum positive pairwise correlation in
permutation `b`, then

```text
r_null = max(m_1, ..., m_N)
log_alpha_null = positive_correlation_log_p(r_null, N)
```

The same threshold is shared by `G+` and `G±`. The former sequential
`mean ± MC-SE` decision-stability rule and its `10N` cap are obsolete under this
estimator. Legacy convergence/interval fields remain only for API compatibility:
normal completion means the fixed `N`-permutation envelope was completed, and
its retained interval fields are degenerate rather than uncertainty intervals.
ADR 0015 is normative for this behavior.

## 3. Discovery, evaluation, and ranking are separate operations

The public static flow is:

```text
Y -> discover() -> MISSet -> MISSet.evaluate()
                         \-> rank() -> Ranking -> mis()
```

`discover()` owns threshold inference, graph construction, graph dimensions,
complete structural MIS enumeration, structural metrics, canonical ordering,
and dimensional support.

`MISSet.evaluate()` is the preferred user facade for adding candidate evidence
without changing graphs, dimensions, candidate membership, or canonical
positions. The equivalent module function `misda.evaluate(mis_set, ...)`
remains supported.

`rank()` materializes a view over the existing candidate universe. A ranking
policy is therefore a preference rule over discovered candidates, not an input
to threshold inference. `Ranking.mis(level, position)` resolves one existing
MIS object from that view without copying it.

## 4. Canonical structural order

The current natural policy is `size_span`:

```text
size   descending
span   descending
```

For a maximal independent set `S`, maximality implies that every vertex outside
`S` is adjacent to at least one selected vertex. Hence
`neighborhood = n - size`. In addition,
`avg_external_degree = span / size`. After `size` ties, neither quantity can
change the ordering independently of `span`.

`neighborhood`, `neighborhood_ratio`, `avg_external_degree`, and
`avg_internal_degree` remain descriptive structural metrics, not independent
ranking criteria. Labels provide a deterministic final tie-break only. Equal
`size` and `span` values remain one scientific rank group. ADR 0017 is normative
for this policy.

The canonical order makes integer positions stable and useful, but contextual
rank does not belong to `MISCandidate`. A future policy can rank the same
candidate differently without changing its identity.

## 5. Candidate evidence

Evidence is grouped by typed domain rather than stored in one unstructured
public dictionary:

```text
candidate.structural
candidate.linear
candidate.nonlinear
candidate.pareto
candidate.dominance
```

Intrinsic candidate properties remain direct:

```text
candidate.indices
candidate.objectives
candidate.size
```

Linear reconstruction predicts eliminated objectives from selected originals
using external PRESS/LOO semantics and keeps untruncated R² values. Nonlinear
reconstruction uses nested external leave-one-out Random Forest evaluation,
internal model selection, deterministic seeds, and uncertainty-driven tree
stopping. Its optional permutation-null reference remains decomposed evidence;
no single SES score is recreated.

Pareto preservation currently assumes same-sample minimization and records retention,
validity, Jaccard agreement, front sizes, and exact preservation. Under pure
objective projection, the reduced nondominated set is a subset of the full one;
therefore validity is identically one and Jaccard equals retention.

Dominance preservation separately counts row pairs with no dominance relation
in full `Y` that acquire a strict dominance relation after projection. The
experimental `dominance_preservation` ranking minimizes that observed
new-dominance rate and uses larger cardinality only as an operational tie-break.

## 6. Evaluation scope and computational cost

Candidate scope belongs to an `evaluate()` call as a whole. Linear/Pareto-only
calls default to all candidates; a call containing nonlinear evaluation defaults
to one candidate. The preferred user-facing selectors are MIS objects returned
by `Ranking.mis()`, sequences of such MISs, or Ranking views. Existing
canonical-prefix and explicit-index selectors remain compatibility paths.

A partial evaluation is scientifically valid but incomplete. Reports therefore
state partial scope explicitly rather than warning as though an error occurred.

Alternative ranking policies declare required metrics and computational cost.
Expensive automatic work over a large candidate universe requires explicit cost
opt-in. `pareto_retention` requires Pareto evidence and
`dominance_preservation` requires dominance evidence unless `accept_cost=True`;
`size_span` remains the canonical structural default.

## 7. Dimensional support

Dimensional support asks whether the observed data contain internal evidence
against the sufficiency of the graph-derived description. It does not use
benchmark truth and does not estimate a replacement dimension.

`TRANSITIVE_CHAINING` compares direct positive association to the retained
candidate with widest indirect max-min paths. It detects the failure mode in
which strong local links along a chain are mistaken for global substitutability.

`HIDDEN_SPECTRAL_STRUCTURE` examines the first rank-correlation eigenvalue
beyond the estimated latent signal dimension and subtracts a column-permutation
null reference. Positive excess indicates organized multivariate structure
remaining beyond the estimate.

All discovered candidates receive support evidence using shared null
permutations. This lets an alternative ranking inspect the support of its own
selected MIS. The compatibility aggregate `MISSet.support` remains scoped to
the first `size_span` group and has state `SUPPORTED`, `PARTIALLY_SUPPORTED`,
or `UNSUPPORTED`; `MISSet.support_for(candidate)` exposes individual evidence.

## 8. Result and reporting boundaries

`MISSet` owns the discovered universe and global analysis. `MISCandidate` owns
candidate evidence. `Ranking` owns contextual order and selection.

Consequently:

```text
mis_set.analysis.structural_dimension   graph-derived quantity
ranking.selected_dimension              preference-derived quantity
```

Reports and plots are views over stored state; they do not trigger hidden
evaluation. `Ranking.report()` is the complete self-contained user report,
while `Ranking.mis().report()`, `.graph_plot()`, and `.front_plot()`
inspect one already-selected MIS without assigning intrinsic ranking metadata to
that MIS. Existing MISSet report/plot entry points remain compatibility paths.

## 9. External benchmark boundary

Benchmark declarations are external expectations. They can compare discovered
quantities and stored evaluation evidence, but they never feed back into
`discover()`, `evaluate()`, or `rank()`.

The comparative suite keeps native estimands distinct. Direct MISDA/PCA
comparison uses a separately defined common external reconstruction metric at a
matched reduced dimension.

## 10. Current scope

Static MISDA is the active scientific path. Adaptive analysis, bounded MIS
enumeration, alternative ranking policies, additional dimensional-support
mechanisms, and maximization/mixed objective directions are deliberately left
for later work.
