# ADR 0006 — Candidate metrics and evidence model

- Status: Accepted
- Recorded: 2026-09-11 (retrospective)

## Context

A maximal independent set is already a valid structural candidate by construction. Metrics characterize candidates; they do not create or invalidate MISs and must not feed back into graph construction or graph-derived dimensions.

MISDA separates intrinsic structural measurements from optional empirical evidence about reconstruction and Pareto preservation.

## Decision

Candidate evidence is organized into four domains:

```text
structural
linear
nonlinear
pareto
```

`candidate.indices`, `candidate.objectives`, and `candidate.size` are intrinsic properties rather than metric-family values.

### Structural metrics

For candidate `S` in `G+`, let `O = V \ S`. Define the external neighborhood

```text
N_ext(S) = {v in O : exists u in S with (u,v) in E}.
```

Current structural quantities are:

```text
neighborhood       = |N_ext(S)|
neighborhood_ratio = |N_ext(S)| / max(1, |O|)
external_degree(u) = number of neighbors of u in O
span               = sum_{u in S} external_degree(u)
avg_external_degree= span / |S|  (0 for empty S)
```

Because `S` is independent in `G+`, its true internal degree is zero; the implementation nevertheless stores `avg_internal_degree` as a generic structural field.

These quantities describe how strongly the retained set covers vertices excluded by the reduction.

### Linear reconstruction evidence

For each eliminated objective `j`, MISDA predicts `y_j` from the retained original objectives using external leave-one-out / PRESS semantics. The coefficient of determination is kept untruncated:

```text
R2_j = 1 - SSE_j / SST_j.
```

Negative values are meaningful evidence that the reconstruction is worse than predicting the target mean. Aggregate evidence includes mean and worst target R2 plus jackknife uncertainty where defined.

### Nonlinear reconstruction evidence

Nonlinear reconstruction asks the same predictive question without restricting the mapping to linear form. The current implementation uses nested external leave-one-out Random Forest regression, deterministic seed derivation, internal model selection, and uncertainty-driven tree stopping. Its outputs remain decomposed reconstruction evidence rather than being collapsed into a single synthetic score.

### Pareto-preservation evidence

Let `P` be nondominated row indices in the original objective space and `P_S` the nondominated row indices after projection to candidate `S`, currently under minimization. MISDA records empirical set agreement including:

```text
retention = |P intersect P_S| / |P|
validity  = |P intersect P_S| / |P_S|
jaccard   = |P intersect P_S| / |P union P_S|.
```

Exact preservation is the set equality `P = P_S`.

## Rationale

These families answer different questions. Structural metrics describe graph coverage; reconstruction measures information recoverability; Pareto metrics measure preservation of the multiobjective dominance structure. No one family is a substitute for the others.

Keeping evidence decomposed avoids hiding trade-offs behind an arbitrary composite score.

## Invariants

- metrics never change `G+`, `G±`, dimensions, candidate membership, or canonical candidate identity;
- missing/undefined evidence is represented explicitly, not fabricated as a perfect or zero score;
- linear R2 is not clipped at zero;
- Pareto evidence compares observed nondominated row sets and currently assumes minimization;
- nonlinear evidence remains optional because of its computational cost.

## Current implementation

Structural metrics are computed during discovery in `misda._ranking.compute_mis_metrics()`. Linear, nonlinear, and Pareto families are attached through `evaluate()` only when requested. Public typed domains expose values such as `candidate.structural.neighborhood`, `candidate.linear.mean_r2`, `candidate.nonlinear.mean_r2`, and `candidate.pareto.jaccard`.

The current performance-oriented implementation includes thin-QR/PRESS computation for linear reconstruction and lexicographical pruning in Pareto calculations where semantics are preserved.

## Permitted implementation variations

Numerically equivalent linear algebra, exact nondominance algorithms, caching, vectorization, and alternative Random Forest kernels are permitted if they preserve the observable metric definitions, validation protocol, seeds, and tolerance policy.

## Forbidden shortcuts / regression risks

Do not use candidate metrics to prune discovery, clip negative R2, replace external validation by in-sample fit, silently evaluate only a subset while presenting the family as complete, or use benchmark truth as candidate evidence.

## Verification

Tests should compare metric values against independently computed small examples, exercise undefined cases, confirm untruncated negative R2, and check Pareto sets by direct dominance enumeration.

## References

- Allen, D. M. (1974). The relationship between variable selection and data augmentation and a method for prediction. *Technometrics*, 16(1), 125–127. (PRESS.)
- Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions. *Journal of the Royal Statistical Society B*, 36(2), 111–147.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5–32.
- Deb, K. (2001). *Multi-Objective Optimization Using Evolutionary Algorithms*. Wiley.