# MISDA decision ledger

This file is the normative repository-local record of the current static MISDA design. It supersedes older refactor notes whenever they conflict with the decisions below.

## 1. Scope

The active scientific path is the static method only. Adaptive analysis is suspended and is not part of the current API or acceptance gate.

The analysis path is strictly data-driven. Benchmark declarations, expected dimensions, generating partitions, expected graphs, or any other external truth must never influence discovery, evaluation, ranking, threshold estimation, or dimensional-support diagnostics.

Information flow is one-way:

```text
Y -> discover() -> MISSet -> evaluate()/rank() -> benchmark()
```

## 2. Dimensional quantities

The current dimensional definitions are graph independence numbers, not connected-component counts.

```text
original_dimension   = number of objective columns
structural_dimension = alpha(G+)
latent_dimension     = alpha(G±)
```

Here `alpha(G)` denotes the exact independence number of graph `G`.

`G+` contains statistically supported positive dependence used as structural redundancy. `G±` contains statistically supported positive and negative dependence. Negative dependence therefore affects latent dimension but does not create positive-redundancy edges in `G+`.

Connected-component counts remain topology diagnostics only:

```text
structural_components = connected_components(G+)
latent_components     = connected_components(G±)
```

The distinction is essential. A connected non-clique graph, such as a chain, can contain several mutually independent vertices; connectivity alone must not collapse its dimensional estimate to one.

## 3. Static public API

The alpha-stage public API is deliberately split into three operations:

```python
mis_set = misda.discover(Y)
misda.evaluate(mis_set, metrics=(...))
ranking = misda.rank(mis_set, policy=...)
```

### 3.1 `discover()`

`discover()` answers: **what structural candidates exist in these data?**

It performs:

1. input normalization and validation;
2. correlation statistics;
3. `alpha_onset` and sequential `alpha_null` estimation;
4. aggressiveness interpolation;
5. construction of `G+` and `G±`;
6. structural and latent dimensions by independence number;
7. complete enumeration of structural MISs of `G+`;
8. structural candidate metrics;
9. canonical structural ordering;
10. dimensional-support diagnostics.

`discover()` does **not** accept `policy` or `rank_policy`. User-selectable ranking policy is a later concern and must not influence threshold estimation or structural discovery.

### 3.2 `evaluate()`

`evaluate()` adds candidate-level evidence to an already discovered `MISSet`. It never discovers new MISs, changes either graph, changes either dimensional estimate, or reorders the canonical `MISSet`.

Current metric families are:

```text
structural
linear
nonlinear
pareto
```

Structural metrics already exist after `discover()`; requesting them again is idempotent.

Candidate metrics use typed domains rather than public dictionaries or flattened prefixes, for example:

```python
candidate.structural.neighborhood
candidate.linear.mean_r2
candidate.linear.r2("f7")
candidate.nonlinear.mean_r2
candidate.pareto.retention
candidate.pareto.validity
candidate.pareto.jaccard
```

`candidate.size`, `candidate.indices`, and `candidate.objectives` are intrinsic candidate properties.

### 3.3 Evaluation scope

`candidates` selects the universe on which one `evaluate()` call operates. Supported forms include:

```python
candidates="all"
candidates=10
candidates=[0, 4, 17]
candidates=ranking[:10]
```

If a call contains only cheap/moderate families such as linear and Pareto, the default scope is all candidates. If the call contains nonlinear evaluation, the default scope is one candidate. A mixed call such as `metrics=("linear", "nonlinear")` therefore evaluates both families on the same one-candidate default scope.

Whenever the effective scope is not all candidates, the stored/reportable output must explicitly state how many candidates were evaluated and how they were selected. This is a scope note, not a warning.

### 3.4 `rank()` and `Ranking`

`rank()` creates an ordered snapshot view over an existing `MISSet`. It never reorders the `MISSet` itself.

A `Ranking` stores a reference to the original `MISSet`, a policy name, canonical candidate indices, and scientific tie groups. Integer indexing returns the underlying `MISCandidate`; slicing returns another `Ranking` view.

```python
ranking[0]
ranking[:10]
ranking.selected
ranking.selected_dimension
```

`selected_dimension` is a property of `Ranking`, not of `MISSet`, because different ranking policies may legitimately select MISs of different sizes.

The scientific graph dimension remains:

```python
mis_set.analysis.structural_dimension
```

The dimension selected by a ranking is:

```python
ranking.selected_dimension
```

## 4. Candidate identity and ranking context

`MISCandidate` does not carry an artificial public ID, intrinsic rank, or `rank_values`. Its operational identity is its fixed canonical position within the immutable-order `MISSet`.

The same candidate may occupy different positions under different future ranking policies. Contextual rank therefore belongs to `Ranking`, never to the candidate.

The canonical position of a candidate never changes during the life of a `MISSet`.

## 5. Canonical structural policy: `structural_coverage`

Discovery establishes one canonical natural order named:

```text
structural_coverage
```

Its current criteria are, in order:

```text
size                  descending
neighborhood          descending
avg_external_degree   descending
span                  descending
```

A deterministic label-based tie-break may order candidates for reproducibility, but it must not create a new scientific rank. Candidates with identical values for the four policy criteria belong to the same rank group.

The name is intentionally specific. Future policies may also be structural while using different structural criteria or priorities.

The default:

```python
misda.rank(mis_set)
```

means `policy="structural_coverage"`.

No alternative policy is defined yet; the infrastructure may support one later without changing discovery.

## 6. Sequential `alpha_null` convergence

`alpha_null` is a statistical threshold quantity. It does not conceptually depend on a user-selected ranking policy.

The estimator starts with at least `B=N` permutations and tracks Monte Carlo uncertainty. At each stopping check, the lower and upper endpoints of the current uncertainty interval are converted to thresholds and their discovery signatures are compared.

The canonical signature is:

```text
Sigma(alpha) = (structural_dimension, latent_dimension, structural_rank_groups)
```

`structural_rank_groups` is the complete `structural_coverage` ordering represented as a sequence of scientific tie groups whose members are the discovered MISs. Numerical metric values themselves are not part of the signature.

Convergence is declared when:

```text
Sigma(alpha_low) == Sigma(alpha_high)
```

### Rationale

The sequential estimator is not intended to pursue arbitrary numerical precision in `alpha_null`. It should refine the null estimate only while the remaining Monte Carlo uncertainty can change a discrete public conclusion of `discover()`.

This signature is the minimal sufficient representation of those structural conclusions:

- comparing only dimensions would be too weak because the candidate MIS universe or its structural ordering could still change;
- requiring literal graph equality would be too strong because an edge can change without changing any structural output of the method;
- requiring equality of raw metric values would also be too strong because small numerical changes that leave all rank groups unchanged have no decision consequence;
- including both structural and latent dimensions ensures that discovery does not declare convergence while `G±` can still alter the latent estimate;
- including the complete tie-group ordering ensures that downstream prefix selections such as `candidates=10` are stable, while ignoring arbitrary deterministic ordering within a true tie.

Thus the null estimator stops exactly when further Monte Carlo precision can no longer alter a discrete structural output exposed by `discover()`.

The autonomous cap remains:

```text
B_max = 10N
```

If the signature has not stabilized by the cap, the current estimate is returned with `converged=False`, reason `MAX_PERMUTATIONS_REACHED`, and one runtime warning per public call.

## 7. Dimensional support

Dimensional support is a global discovery diagnostic, not a candidate-ranking metric family and not an alternative dimensional estimator.

It asks whether the observed data contain evidence contradicting the sufficiency of the graph-derived dimensional description.

Two diagnostics are currently used:

### 7.1 `TRANSITIVE_CHAINING`

This detects cases where an eliminated objective is strongly connected to a retained objective only through an indirect chain of strong positive associations, while its direct association to the retained set is much weaker. The statistic compares widest max-min indirect path strength with direct positive association and subtracts a permutation-null reference.

Its purpose is to detect the failure mode:

```text
strong local links along a chain != global substitutability
```

### 7.2 `HIDDEN_SPECTRAL_STRUCTURE`

This examines the first rank-correlation eigenvalue beyond the estimated latent signal dimension and compares it with a column-wise permutation-null mean. Positive excess indicates organized multivariate structure remaining beyond the graph-derived latent dimension.

This diagnostic does not say what the correct replacement dimension is. It says only that the current estimate is not fully supported by these internal checks.

### 7.3 First-rank tie group

Support must not depend on an arbitrary deterministic tie-break. Therefore it is evaluated for every candidate in the first `structural_coverage` rank group.

Aggregate status is:

```text
SUPPORTED             all first-rank candidates supported
PARTIALLY_SUPPORTED   a mixture of supported and unsupported candidates
UNSUPPORTED           no first-rank candidate supported
```

The result must expose which candidates were supported or unsupported and the reasons/evidence for each candidate, for example through:

```python
mis_set.support.status
mis_set.support.supported
mis_set.support.unsupported
mis_set.support.for_candidate(...)
```

`HIDDEN_SPECTRAL_STRUCTURE` is global for a fixed latent dimension; `TRANSITIVE_CHAINING` depends on the retained candidate. Implementations should reuse common permutation work where practical rather than rerunning identical global calculations.

## 8. Reconstruction and Pareto evidence

Linear reconstruction predicts only eliminated objectives from selected original objectives using external PRESS/LOO semantics. Scores remain untruncated; negative R2 is meaningful.

Nonlinear reconstruction reuses the established nested external-LOO Random Forest protocol, deterministic seed derivation, internal model selection, and uncertainty-driven tree stopping. The old `heavy()` API is removed; nonlinear evidence is requested through `evaluate()`.

The optional nonlinear null reference remains evidence attached to nonlinear reconstruction; it must not recreate a single SES score.

Pareto preservation currently assumes minimization and records empirical observed-set retention, validity, and Jaccard agreement. Maximization and mixed objective directions remain future work.

## 9. Benchmark boundary

Benchmark truth belongs exclusively to benchmark infrastructure.

The benchmark may compare:

- declared latent dimension with `mis_set.analysis.latent_dimension`;
- declared structural dimension with `mis_set.analysis.structural_dimension`;
- declared structural blocks with observed structural topology;
- declared Pareto indices with already-computed candidate Pareto evidence.

The benchmark must not feed truth back into `discover()`, `evaluate()`, or `rank()`.

Dimension declarations are not graph component-count declarations. A benchmark must never infer expected connected-component counts merely from expected structural or latent dimension.

## 10. Reporting and plotting

Reports and plots are views over already stored state. They must not rerun scientific evaluation or consult benchmark truth.

Reports must distinguish:

```text
discovery / dimensions / threshold convergence
dimensional support
structural ranking
candidate evaluation families and their scopes
```

Any partial evaluation must state its scope explicitly.

## 11. Alpha-stage cleanup

Because the project is still alpha, the new API is a direct cleanup rather than a deprecation migration. The final static surface must contain no compatibility wrappers or deprecated aliases for the superseded model.

The following old concepts are to be removed from the final static path:

```text
analyze()
heavy()
MISDAResult / LegacyMISDAResult
candidate.id
candidate.rank
candidate.rank_values
candidate.evaluation
result.selected_dimension
best_mis* aliases
rank_policy inside discovery
max_evaluated_mis
```

Tests, reports, benchmarks, serializers, notebooks, and documentation must be migrated in the same branch before the final gate.

## 12. Numerical gates and reproducibility

Discrete structural quantities require exact equality. Floating scientific metrics use the versioned tolerance policy already centralized in `_tolerances.py`:

```text
GATE_RTOL = 0
GATE_ATOL = 1e-12
```

Seeds remain explicit and deterministic. Structural `alpha_null` estimation uses the autonomous `10N` cap.

## 13. Deliberately deferred work

The following are deliberately outside this implementation pass:

- definition of the first alternative ranking policy;
- additional structural ranking policies beyond `structural_coverage`;
- new dimensional-support diagnostics or a support-driven correction of dimension;
- bounded/partial MIS enumeration semantics;
- adaptive analysis;
- maximization or mixed objective directions.

The ranking infrastructure may be designed so that future policies declare required metric families, directions, and cost. Expensive automatic evaluation should then require an explicit cost opt-in when the requested universe is large. No such alternative policy is defined in the present pass.

## 14. Acceptance sequence

The implementation sequence is:

```text
baseline verification
new object model
discover() and alpha_null signature
dimensional support over first-rank ties
evaluate()
Ranking/rank()
benchmark/report/plot/notebook migration
legacy removal
final documentation and scientific gate
```

The final gate must exercise the static test suite, relevant slow tests, both notebooks end-to-end, the canonical 13-case scientific battery, and the three comparative experiments. Adaptive remains excluded.
