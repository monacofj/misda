<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# MISDA static design notes

This document records the methodological boundaries of the refactored static
pipeline. The implementation favors explicit estimands, reproducible stopping
rules, and result objects that distinguish stored evidence from presentation.

## 1. Positive structure and signed dependence

Positive and negative correlation have different meanings in objective
reduction:

- a statistically supported positive association is a candidate redundancy
  edge;
- a supported negative association is a conflict and must not become a
  redundancy edge.

MISDA therefore constructs two graph projections. The positive structural
graph `G+` contains only supported positive relations and drives the reduction
MISs. The signed dependence graph `G±` contains supported relations of either
sign. Dimensional estimates are graph independence numbers: `structural_dimension`
is the maximum cardinality of an independent set in `G+`, while
`latent_dimension` is the maximum cardinality of an independent set in `G±`.
The sign of an edge therefore affects structural redundancy, while both signs
express latent dependence. Connected-component counts remain topology
diagnostics and are stored separately from both dimensions. MIS ranking chooses
among candidate reductions and does not define either dimensional estimate.

Constant objectives cannot support a valid pairwise Fisher-z test. They remain
isolated nodes with explicit metadata so the pipeline never loses an input
column silently.

## 2. Log-domain statistical layer

The static pipeline tests positive correlation with a one-tailed Fisher-z
probability. Probabilities are stored and compared in log space, preserving
extreme evidence that would underflow in ordinary floating-point probability
space.

Two thresholds delimit the analysis:

- `alpha_onset`: first observed positive structural event;
- `alpha_null`: null-calibrated endpoint obtained by permuting each objective
  independently and tracking the maximum positive correlation.

Null estimation begins with at least `N` permutations. It stops only when the
structural signature is identical across the Monte Carlo uncertainty interval,
or when an external cancellation request is received. There is no hidden
iteration cap. The result stores the number of permutations, uncertainty
intervals, convergence, seed, and RNG state.

`aggressiveness` interpolates between the endpoints in a numerically stable
log-domain calculation. A value of `0` selects the onset; `1` selects the
null-calibrated endpoint.

## 3. Internal dimensional support

A dimensional estimate is accompanied by an internal support diagnostic that
uses no benchmark declaration. `SUPPORTED` does not mean that the estimated
dimension has been proved correct; it means that the observed data did not
produce either of the two implemented contradictions. `UNSUPPORTED` means that
at least one contradiction was detected.

The diagnostic works in rank space so monotonic nonlinear transformations do
not create artificial disagreement. It uses two complementary statistics:

- transitivity: for each eliminated objective, compare its strongest direct
  positive rank correlation with the retained MIS to the strength of its
  strongest indirect max-min path to that MIS. The maximum indirect-minus-direct
  value detects chaining;
- spectrum: inspect the first rank-correlation eigenvalue beyond the estimated
  latent dimension. A large remaining direction indicates hidden systematic
  structure beyond the graph estimate.

Each statistic is calibrated by independently permuting every objective column.
The null reference is the mean statistic over exactly `N` permutations, so the
Monte Carlo budget is derived from the sample size rather than supplied as a
hyperparameter. The stored excesses are

`transitivity_excess = observed_transitivity - null_transitivity`

and

`spectral_excess = observed_next_eigenvalue - null_next_eigenvalue`.

The categorical rule has the intrinsic zero boundary:

- if either excess is strictly greater than zero, status is `UNSUPPORTED`;
- otherwise status is `SUPPORTED`.

Positive transitivity excess carries reason `TRANSITIVE_CHAINING`. Positive
spectral excess carries reason `HIDDEN_SPECTRAL_STRUCTURE`. These diagnostics
do not change the graph, dimension, MIS enumeration, or ranking; they only state
whether the data contain internal evidence contradicting the estimate.

As a general methodological rule, categorical MISDA decisions may use a
mathematically privileged boundary such as zero or a reference estimated from
the data. Fixed performance thresholds chosen externally (for example R² > 0.9)
are not introduced as hidden hyperparameters.

## 4. Candidate enumeration and ranking

Maximal independent sets are enumerated from the positive structural graph.
Every result is unique, maximal, and deterministically ordered. The default
ranking records its criterion values, while objective labels are used only as a
stable final tie-breaker. Equal criterion values share a rank.

All candidates are retained in `result.mis`; evaluation limits affect evidence
collection, not visibility or ranking.

## 5. Light and heavy evidence

Light evaluation runs for a ranked prefix during `analyze()`:

- linear reconstruction predicts eliminated objectives only and uses external
  predictions;
- delete-one jackknife estimates sampling uncertainty;
- Pareto retention, validity, and Jaccard compare nondominated row masks.

Undefined quantities carry `None` plus a machine-readable reason. In
particular, full retention has no eliminated-objective reconstruction score.

Heavy evaluation is separate and on demand. It uses nested leave-one-out
Random Forest reconstruction, model selection inside each outer fold, and a
tree count determined by uncertainty stability. An optional sequential
permutation null reports reconstruction beyond chance and incidental
reconstruction frequency. Both expensive layers preserve partial results and
explicitly record cancellation or non-convergence.

## 6. Result and reporting boundaries

The result tree has three responsibilities:

- `AnalysisResult`: global scientific properties and graph state;
- `MISCandidate`: stable candidate identity, rank values, and attached evidence;
- `ExecutionResult`: effective controls, reproducibility, timing, and
  convergence.

`summary()`, `report()`, and `graph_plot()` are views over stored state. They do
not run validation, tune parameters, or mutate scientific results. Metric names
and explanations are centralized in reporting metadata.

## 7. External benchmarks

Benchmark declarations are external expectations, not inputs to `analyze()`.
Each run records its input digest, seed, software versions, estimates, and
assessment. Comparing with a frozen pre-refactor artifact is a regression gate;
it does not force the new schema to reproduce legacy field names or known
legacy defects.

The comparative suite keeps two estimands separate:

- PCA: global standardized reconstruction R² over all objectives;
- MISDA: external reconstruction of eliminated objectives from selected
  original objectives.

They may be displayed together as complementary evidence but must not be
collapsed into one unqualified fidelity axis.

## 8. Compatibility boundary

The static v2 pipeline is the supported scientific contract. Unambiguous legacy
spellings are temporary deprecated forwards. Ambiguous legacy validation fields
are not recreated on the new result.

The previous adaptive implementation is suspended. It remains isolated for
future methodological work and is excluded from current acceptance tests,
benchmarks, examples, and claims.