# MISDA decision ledger

This document records the binding methodological, architectural, API, benchmark, and reproducibility decisions consolidated during the static MISDA refactor. It is a repository-local preservation of the internal decision notebook that guided the `refactor` work.

It intentionally distinguishes three categories:

- **Implemented decision** — a closed decision that was implemented during the static refactor and is part of the normative baseline to be preserved unless explicitly superseded.
- **Binding pending item** — a closed decision whose implementation or validation remained incomplete after the post-implementation audit.
- **Future investigation** — an open methodological question that is not part of the current implementation contract.

This file is not yet split into ADRs. A later documentation pass may convert individual records into ADRs while preserving their semantics and status.

## 1. Scope and method/benchmark boundary

**Status: implemented decision.**

The current work concerns only `method="static"`. The adaptive method is suspended and is outside the static acceptance gate.

`analyze()` must be strictly data-driven and self-contained with respect to benchmark truth. It may use only the supplied objective matrix, method configuration, and quantities derived from those data. Declared latent/structural dimensions, generating partitions, expected graphs, or any other external truth must never influence analysis, selection, ranking, estimation, or internal evaluation.

The information flow is one-way:

```text
Y -> analyze() -> MISDAResult -> benchmark evaluation
```

Metrics computable only from `Y` and a selected subset, such as reconstruction and observed Pareto preservation, may belong to MISDA analysis. Metrics requiring declared truth belong exclusively to benchmark infrastructure.

## 2. Dimensional quantities

**Status: implemented decision.**

The result distinguishes three quantities:

- `original_dimension`: number of objective columns in the supplied matrix;
- `latent_dimension`: estimated number of connected components in the signed dependence graph `G±`;
- `structural_dimension`: estimated number of connected components in the positive structural graph `G+`.

The signed graph contains statistically supported positive and negative dependence. The positive graph contains only statistically supported positive relations used as candidate redundancy edges.

Formally:

```text
latent_dimension     = connected_components(G±)
structural_dimension = connected_components(G+)
```

These dimensions are not defined by MIS cardinality. MISs are enumerated only on `G+` and represent candidate selections. In disjoint unions of cliques, preferred MIS size may coincide with structural dimension; in a connected non-clique graph, such as a chain, they may differ legitimately.

Negative edges are dimensional/diagnostic only. They do not enter positive redundancy reduction, `alpha_onset`, `alpha_null`, aggressiveness interpolation, MIS enumeration, or MIS ranking.

## 3. Operational interpretation of correlations

**Status: implemented decision, with known methodological limitation.**

The static method currently interprets:

- strong positive correlation as local evidence of redundancy/substitutability;
- strong negative correlation as latent dependence plus structural conflict, not redundancy;
- weak correlation as no evidence of dependence/redundancy, so both objectives are preserved.

This interpretation is explicitly local. It is not a proof that chains of positive associations imply global substitutability.

### Case 5: cumulative chain

**Status: implemented diagnostic; future methodological investigation remains open.**

The cumulative-chain benchmark is intentionally adversarial. Each objective contains an independent innovation even though adjacent objectives can be strongly positively correlated. The current graph semantics can therefore produce one connected structural component and a large independent set while the declared intrinsic/structural dimension remains full.

This mismatch must not be “repaired” merely to force agreement with the benchmark. It exposes the limitation:

```text
local statistical dependence != global structural redundancy
```

Future work may investigate structural substitutability, incremental information, conditional redundancy, or signed-neighborhood/profile equivalence. None of those is part of the current contract.

## 4. MIS enumeration, ranking, and evaluation scope

**Status: implemented decision.**

MISDA retains all maximal independent sets of `G+`. A maximal independent set need not have maximum cardinality.

The result stores the complete ordered MIS list. The preferred MIS is the first candidate under the selected ranking policy. Preferred MIS size is a selection quantity and does not define structural dimension.

`rank_policy="default"` names and preserves the ranking behavior of the pre-refactor implementation. The refactor exposes the policy explicitly without redefining it methodologically. Every candidate stores `rank_values`; candidates with identical ranking values share the same ordinal rank. Deterministic tie-breaking may order candidates within a rank but must not create a new rank.

The result records:

```text
n_mis
n_evaluated_mis
n_heavy_mis
```

`max_evaluated_mis` limits only light evaluation of the already enumerated and ranked list. It must not alter MIS discovery, ranking, graph structure, dimensional estimates, or `n_mis`. `None` means evaluate all candidates.

There is no public partial-enumeration budget in the current version. A future bounded enumerator would require explicit partial-result semantics such as `enumeration_complete=False`.

## 5. Statistical endpoints: `alpha_onset` and `alpha_null`

**Status: implemented decision.**

Legacy `alpha_min`/`alpha_max` semantics are replaced by:

- `alpha_onset`: the positive-correlation threshold at which the first positive structural edge appears;
- `alpha_null`: the threshold corresponding to the expected maximum positive correlation under independent within-column permutation.

The null reference is data-driven:

```text
R0+ = E_pi[max(0, max_{i<j} rho_ij^(pi)) | Y]
```

The null estimator starts with at least `B=N` permutations and uses the accumulated Monte Carlo mean and standard error. It evaluates the structural signature at the two endpoints of the operational uncertainty interval and stops when both endpoints produce the same structural signature.

The structural signature includes the structural estimate and the ordered/ranked MIS output needed to determine whether further Monte Carlo precision can still alter the method’s structural decision.

No arbitrary fixed 500-permutation rule and no arbitrary percentile such as 95% is part of the method.

### Autonomous null-estimation cap

The post-implementation audit closed the decision that the structural `alpha_null` estimator must terminate autonomously at:

```text
B_max = 10N
```

If the structural signatures have not stabilized at the cap, the public result returns the current estimate with:

```text
converged = False
n_permutations = 10N
reason = MAX_PERMUTATIONS_REACHED
```

and emits exactly one `RuntimeWarning` per public call. The reason propagates to execution diagnostics and reporting. An internal cancellation callback remains an additional, distinct mechanism.

**Status: implemented after the post-audit review; final full-gate validation remains pending.**

## 6. Separation status and aggressiveness

**Status: implemented decision.**

The old multi-regime labels are replaced by:

```text
NULL_SEPARATION
NO_NULL_SEPARATION
```

`NULL_SEPARATION` means `alpha_onset < alpha_null`. Equality belongs to `NO_NULL_SEPARATION`. Absence of any valid positive correlation also yields `NO_NULL_SEPARATION` with undefined `alpha_onset`.

`separation_status` is diagnostic and does not by itself block result production.

`aggressiveness` remains a continuous parameter in `[0,1]`, interpolating arithmetically between `alpha_onset` and `alpha_null`. Under null separation:

- `aggressiveness=0` selects the onset threshold, i.e. the least non-empty positive reduction;
- `aggressiveness=1` selects the null-calibrated endpoint.

Intermediate values are positions in the calibrated alpha interval, not fractions of edges, dimensions, or structural regimes.

Future work may investigate interpolation by structural regimes, but this is not part of the current contract.

## 7. One-tailed positive test and log-domain numerics

**Status: implemented decision.**

Positive redundancy uses the one-sided alternative `rho > 0`. Pairwise probabilities and thresholds are represented canonically in log space to avoid underflow for very strong correlations.

Operationally:

```text
log_p = norm.logsf(arctanh(rho) * sqrt(N - 3))
edge in G+ iff rho > 0 and log_p <= log_alpha
```

`alpha_onset`, `alpha_null`, and aggressiveness interpolation are handled consistently in this domain. Arithmetic interpolation of alpha is implemented stably through log-sum-exp; this does not redefine aggressiveness as logarithmic interpolation.

Artificial clipping of extreme p-values to `float.tiny` is not part of the refactored contract.

## 8. Result object and API boundary

**Status: implemented decision.**

The static result is organized as:

```text
MISDAResult
  analysis
  mis[]
  execution
```

`analysis` stores global scientific state, graphs, dimensions, thresholds, separation status, ranking policy, counts, and diagnostics.

Each MIS candidate stores stable identity, selected objectives/indices, size, rank, ranking values, and attached evaluations.

`execution` stores effective configuration, seed, timings, and convergence/reproducibility diagnostics.

Benchmark truth must never be stored in ordinary `MISDAResult` state.

A metric that was not requested is absent. A requested metric that is mathematically undefined is stored as `None` with a machine-readable reason. Data/implementation errors raise exceptions rather than being silently converted into missing metrics.

MIS identifiers are deterministic (`mis_000`, `mis_001`, ... after ordering). Original objective labels remain graph-node attributes so later operations such as `heavy()` recover labels from the result itself.

## 9. Light reconstruction evidence

**Status: implemented decision.**

Light evaluation belongs to `analyze()` for the configured ranked prefix.

Linear reconstruction predicts only eliminated objectives from selected original objectives. Predictions must be external to the fit used for that observation. The refactor uses PRESS/leave-one-out semantics with a stable explicit-LOO fallback when the PRESS denominator is numerically unsafe.

For each eliminated objective, store untruncated out-of-sample `R²`; negative values remain meaningful. Summaries include at least:

```text
r2_by_objective
mean_r2
worst_r2
```

No-reduction cases and mathematically undefined targets are represented explicitly as undefined, not as artificial perfect/zero scores.

Legacy `SES`, `F_real`, and `F_null` are not part of the new semantic contract.

## 10. Pareto preservation

**Status: implemented decision.**

The current static release supports minimization only. Maximization and mixed directions remain future extensions.

Pareto evaluation compares empirical nondominated row sets under all objectives and under the selected objectives. The full-objective nondominated set is computed once and reused across candidate evaluations.

Public terminology uses observed-set semantics rather than “true Pareto front” or “surrogate Pareto front”. The refactor replaces the legacy consistency routine with Pareto-preservation semantics and stores the corresponding retention/validity/Jaccard-style diagnostics as defined by the implementation contract.

Truth-dependent Pareto expectations, when declared by a benchmark, belong to benchmark evaluation rather than ordinary analysis.

## 11. Heavy nonlinear evaluation

**Status: implemented decision.**

`misda.heavy(result, selection)` complements the existing result in place without reordering candidates or replacing light evidence. It is idempotent for already stored heavy metrics.

Nonlinear reconstruction uses Random Forest models only for eliminated objectives and an external LOO protocol with internal model selection.

The audited reproducibility contract fixes, among other details:

- `max_features` searched over `1..p`;
- `min_samples_leaf` searched over its full admissible discrete range for the inner training set;
- MSE criterion for internal selection;
- numeric tie handling followed by the simpler configuration (larger leaf size, then smaller `max_features`);
- deterministic seeds derived from the global seed using stable operation coordinates;
- `squared_error`, `n_jobs=1`, and warm-started tree growth;
- model configuration selected at `T=N` trees and held fixed while the forest grows;
- stopping based on computational tree uncertainty no longer exceeding sample uncertainty for all defined eliminated-objective scores;
- explicit non-convergence reasons for undefined sample uncertainty or external cancellation.

There is no methodological hidden tree cap in heavy nonlinear reconstruction.

### Heavy optional null reference

The optional heavy null reference permutes preserved/eliminated association and repeats the same external reconstruction protocol. It starts at `B=N` and stops according to its own computational-vs-sampling-uncertainty rule. It produces separate evidence such as `above_null_r2` and `incidental_reconstruction_rate`; it must not recreate a single SES score.

The structural `alpha_null` cap of `10N` does **not** apply to this heavy null reference.

## 12. Reporting and plotting

**Status: implemented decision.**

`summary()`, `report()`, and `graph_plot()` are views over already stored state. They must not rerun validation, tune parameters, consult benchmark truth, or mutate scientific results.

Reporting metadata should centralize public metric names, formatting, technical meaning, and concise interpretation. Reports show global analysis, ranking counts, evaluated-vs-total MIS counts, and representative top-ranked candidates without dumping large matrices or the full candidate space.

Legacy `plot()` may remain temporarily as a deprecated alias of `graph_plot()`.

Claims such as `Ideal (Disjoint Cliques)` and uses of homogeneity/compactness as proof of external structural truth are not part of the refactored report contract.

## 13. Benchmark architecture

**Status: implemented decision.**

Benchmark declarations are external expectations. They may include declared dimensions, expected graph/topology, structural units/blocks, Pareto declarations, notes, and adversarial-case status.

Baseline artifacts record the input hash, seed, and software versions so data changes cannot be confused with algorithm changes.

Two benchmark operations are semantically distinct and use different vocabularies.

### 13.1 Conformity with the theoretical declaration

Closed vocabulary:

```text
DECLARATION_MATCH
DECLARATION_MISMATCH
EXPECTED_DECLARATION_MISMATCH
NO_DECLARATION
```

Known adversarial Case 5 uses `EXPECTED_DECLARATION_MISMATCH` where appropriate.

### 13.2 Historical non-regression against a frozen baseline

Vocabulary remains:

```text
PASS
IMPROVED
EXPECTED_CHANGE
REGRESSION
```

Only this historical operation may use the word `REGRESSION`.

The post-audit terminology migration has been implemented in benchmark evaluation and its tests. Historical comparison retains the original regression vocabulary.

**Status: implemented after the post-audit review; final full-gate validation remains pending.**

## 14. Numerical portability of gates

Discrete structural quantities — input hashes, indices, ranks, counts, graph components, and edges — require exact equality.

Floating-point scientific metrics must not depend on last-bit equality across supported numerical environments.

The versioned gate policy is:

```text
GATE_TOLERANCE_VERSION = 1
GATE_RTOL = 0
GATE_ATOL = 1e-12
```

Only floating-point scientific metrics use this tolerance. Discrete structure remains exact. The tolerance is intended to absorb platform/library last-bit variation and must not mask scientifically material degradation.

**Status: implemented after the post-audit review; final full-gate validation remains pending.**

## 15. Benchmark modes, canonical sizes, and comparative protocol

**Status: implemented operational decision, with final full-gate validation pending.**

Three uses are distinguished:

- `--quick`: `N=64`, smoke testing only; declaration mismatches in this mode are not scientific acceptance failures;
- canonical scientific battery (7 canonical cases + 6 MOPs): `N=1000`, seed `123`, `max_evaluated_mis=1`;
- canonical comparative battery (3 comparative experiments): `N=500`, seed `123`, `max_evaluated_mis=1`.

`benchmark.ipynb` is an interactive scientific interface rather than the acceptance manifest. Its historical default is `N=300`, it runs only the static method, displays each declaration, calls the public analysis/benchmark API, draws `result.graph_plot()`, and preserves results for inspection.

`comparative.ipynb` follows the same interactive API pattern. It generates the three comparative cases explicitly, calls `misda.analyze(..., method="static")` and `misda.benchmark()`, displays the report and graph for each case, and stores the resulting objects for inspection. `run_comparative.py` remains the non-interactive JSON/gate frontend rather than the user-facing notebook API.

### 15.1 Native reconstruction metrics remain distinct

MISDA's native light-reconstruction summaries (`mean_r2`, `worst_r2`, and per-objective R²) concern only original objectives eliminated by the selected MIS and use external PRESS/LOO predictions.

PCA's conventional `global_standardized_r2` curve is retained as a separate in-sample compression diagnostic over all original objectives after standardization.

These native quantities have different estimands and must not be numerically compared to each other.

### 15.2 Common direct MISDA-versus-PCA metric

For direct comparison, both methods use the additional benchmark-only metric:

```text
global_standardized_external_r2
```

For each non-constant original objective `j`, compute an external reconstruction R² `R²_j`, then give every objective equal weight:

```text
global_standardized_external_r2 = mean_j(R²_j)
```

This is equivalent to measuring squared reconstruction error after variance-standardizing each original objective, so physical scale does not determine its weight. Constant objectives are excluded because R² is undefined for them. Negative R² values are retained.

For MISDA:

- the comparison dimension is the preferred MIS size;
- preserved objectives are part of the reduced representation and therefore reconstruct exactly (`R²=1`);
- eliminated objectives use the external PRESS/LOO R² already produced by light reconstruction.

For PCA:

- centering, scaling, and principal directions are fit without the held-out row;
- the held-out row is projected onto `k` learned components and reconstructed;
- the same equal-objective-weight external R² is then computed over the original objective space.

The principal direct comparison is made at the same reduced dimension, `k = preferred MIS size`. The PCA external curve may additionally be shown across dimensions.

This comparison measures **information preservation of reduced representations**, not objective-acquisition cost. PCA uses transformed components whose scores are formed from the original objective vector, whereas MISDA preserves a subset of original objectives. This distinction must remain explicit in interpretation.

**Status: implemented decision for the comparative benchmark/notebook; final full-gate validation remains pending.**

## 16. Reproducibility contracts

**Status: implemented decision.**

Internal random streams are deterministically derived from the global seed through `numpy.random.SeedSequence` and stable operation coordinates rather than shared mutable RNG state.

The result stores enough diagnostics to reproduce or inspect stochastic work, including seed/state information for null estimation where applicable.

Cancellation interfaces have intentionally distinct contracts: the internal structural null estimator callback receives the current permutation count; heavy-evaluation cancellation is a no-argument predicate. These signatures must remain documented while both exist.

A heavy null-reference calculation returning undefined null mean reconstruction when the observed reconstruction is defined is a contract error and must raise rather than loop or masquerade as convergence.

## 17. Compatibility and deprecated concepts

**Status: implemented decision / transitional contract.**

Public terminology was migrated toward:

```text
caution -> aggressiveness
pareto_recall -> pareto_retention
evaluate_pareto_consistency() -> evaluate_pareto_preservation()
```

Temporary, unambiguous compatibility forwards may emit `DeprecationWarning`. Ambiguous or scientifically abandoned legacy quantities do not receive aliases merely for compatibility.

`SES`, `F_real`, `F_null`, and truth-dependent structural-recovery metrics are not part of the static-v2 result contract.

NaN and infinite input values are rejected explicitly. Constant objectives are handled explicitly rather than silently discarded or used to invalidate the entire analysis.

## 18. Post-implementation audit closure status

The static refactor was functional and extensively tested, but it was explicitly not considered definitively closed until five post-audit obligations were completed and validated.

Current implementation status:

1. **Autonomous `alpha_null` cap — implemented:** `B_max=10N`, `converged=False`, diagnostic reason, and one `RuntimeWarning` when the structural null estimator does not stabilize by the cap.
2. **Benchmark vocabulary separation — implemented:** declaration-specific statuses are distinct from historical baseline regression statuses.
3. **Portable numerical comparisons — implemented:** floating-point gate metrics use versioned `rtol=0`, `atol=1e-12`; discrete structure remains exact.
4. **Interactive review/rebuild of `comparative.ipynb` — implemented:** canonical `N=500`, direct public API usage, native metrics kept distinct, and a common external reconstruction score added for numerical PCA/MISDA comparison.
5. **Final static gate — pending:** rerun the full static suite, relevant `slow` tests, both notebooks end-to-end, the canonical 13-case scientific battery, and the 3 comparative experiments; confirm adaptive remains excluded; then update remaining active documentation and declare the static refactor closed.

Items 1–4 are implemented but remain subject to the final gate. Item 5 is the remaining binding validation obligation.

## 19. Future investigations that are not binding implementation obligations

The following remain open research/design questions and must not be mistaken for unfinished implementation of the current contract:

- a stronger structural-substitutability criterion for cumulative chains and other non-clique positive graphs;
- equivalence of signed structural profiles as possible redundancy evidence;
- alternative ranking policies over the same MIS set;
- structurally meaningful reparameterization of `aggressiveness` by graph regimes;
- bounded/partial MIS enumeration with explicit incomplete-result semantics;
- support for maximization and mixed objective directions;
- broader uses of the signed dependence layer beyond current latent-dimension diagnostics;
- future reactivation/redesign of the adaptive method.

## 20. Normative interpretation for future reconciliation

This ledger preserves the decisions that governed the static `refactor` plus the binding post-audit obligations. During reconciliation with later branches or later reviews:

1. an implemented decision remains normative unless a later decision explicitly supersedes it;
2. code behavior that conflicts with a closed decision is a divergence to be reconciled, not an automatic replacement of the decision;
3. binding pending items remain obligations until explicitly implemented and validated or superseded by a later recorded decision;
4. future investigations do not constrain the current implementation until a new decision is recorded.
