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
graph drives maximal independent set enumeration and structural components.
The signed dependence graph includes both signs and describes latent
connectivity. This is why latent dimension, structural dimension, and MIS size
are stored as separate fields.

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

## 3. Candidate enumeration and ranking

Maximal independent sets are enumerated from the positive structural graph.
Every result is unique, maximal, and deterministically ordered. The default
ranking records its criterion values, while objective labels are used only as a
stable final tie-breaker. Equal criterion values share a rank.

All candidates are retained in `result.mis`; evaluation limits affect evidence
collection, not visibility or ranking.

## 4. Light and heavy evidence

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

## 5. Result and reporting boundaries

The result tree has three responsibilities:

- `AnalysisResult`: global scientific properties and graph state;
- `MISCandidate`: stable candidate identity, rank values, and attached evidence;
- `ExecutionResult`: effective controls, reproducibility, timing, and
  convergence.

`summary()`, `report()`, and `graph_plot()` are views over stored state. They do
not run validation, tune parameters, or mutate scientific results. Metric names
and explanations are centralized in reporting metadata.

## 6. External benchmarks

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

## 7. Compatibility boundary

The static v2 pipeline is the supported scientific contract. Unambiguous legacy
spellings are temporary deprecated forwards. Ambiguous legacy validation fields
are not recreated on the new result.

The previous adaptive implementation is suspended. It remains isolated for
future methodological work and is excluded from current acceptance tests,
benchmarks, examples, and claims.
