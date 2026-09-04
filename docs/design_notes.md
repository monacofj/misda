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

## 2. Statistical thresholds and sequential null estimation

The static layer stores probability evidence in log space to avoid underflow.
Two data-derived endpoints delimit the selected threshold:

- `alpha_onset`: first observed positive structural event;
- `alpha_null`: permutation-null endpoint based on the maximum positive
  correlation under independently permuted objective columns.

`aggressiveness` interpolates between these endpoints in log space.

Null estimation begins with at least `N` permutations and is bounded by
`B_max=10N`. Its stopping criterion is deliberately structural rather than
numerical. At the lower and upper endpoints of the current Monte Carlo
uncertainty interval, MISDA compares:

```text
Sigma(alpha) = (
    structural_dimension,
    latent_dimension,
    complete structural_coverage tie-group ordering,
)
```

The estimator stops when the two signatures match. Dimensions alone would be
too weak because candidate sets/order could still change. Literal graph equality
would be too strong because an edge can change without changing any public
discovery conclusion. Raw metric equality would likewise pursue numerical
precision with no decision consequence.

## 3. Discovery, evaluation, and ranking are separate operations

The public static flow is:

```text
Y -> discover() -> MISSet -> evaluate()
                         \-> rank()
```

`discover()` owns threshold inference, graph construction, graph dimensions,
complete structural MIS enumeration, structural metrics, canonical ordering,
and dimensional support.

`evaluate()` adds candidate evidence without changing graphs, dimensions,
candidate membership, or canonical positions.

`rank()` materializes a view over the existing candidate universe. A ranking
policy is therefore a preference rule over discovered candidates, not an input
to threshold inference.

## 4. Canonical structural order

The current natural policy is `structural_coverage`:

```text
size                  descending
neighborhood          descending
avg_external_degree   descending
span                  descending
```

Labels provide a deterministic final tie-break only. Equal values for the four
scientific criteria remain one rank group.

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

Pareto preservation currently assumes minimization and records retention,
validity, Jaccard agreement, front sizes, and exact preservation.

## 6. Evaluation scope and computational cost

Candidate scope belongs to an `evaluate()` call as a whole. Linear/Pareto-only
calls default to all candidates; a call containing nonlinear evaluation defaults
to one candidate. Users can select all candidates, a canonical prefix, explicit
indices, or a `Ranking` slice.

A partial evaluation is scientifically valid but incomplete. Reports therefore
state partial scope explicitly rather than warning as though an error occurred.

Future ranking policies may declare required metrics and computational cost.
Expensive automatic work over a large candidate universe must then require
explicit cost opt-in. No alternative ranking policy is defined yet.

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

If several candidates tie at first `structural_coverage` rank, all are evaluated
using shared null permutations. This prevents an arbitrary deterministic
label-based tie-break from deciding a scientific support status. The aggregate
state is `SUPPORTED`, `PARTIALLY_SUPPORTED`, or `UNSUPPORTED`, while individual
candidate evidence remains inspectable.

## 8. Result and reporting boundaries

`MISSet` owns the discovered universe and global analysis. `MISCandidate` owns
candidate evidence. `Ranking` owns contextual order and selection.

Consequently:

```text
mis_set.analysis.structural_dimension   graph-derived quantity
ranking.selected_dimension              preference-derived quantity
```

Reports and graph plots are views over stored state; they do not trigger hidden
evaluation.

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
