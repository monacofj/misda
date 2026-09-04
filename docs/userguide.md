<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# MISDA static user guide

The static API separates structural discovery, candidate evaluation, and
preference among candidates. This separation prevents a result from implying
that an evaluation was computed when it was not and prevents ranking choices
from feeding back into structural inference.

## 1. Input contract

`misda.discover()` accepts a two-dimensional NumPy array or pandas DataFrame.
Values must be finite, real, numeric, and contain at least four observations.
DataFrame labels are preserved; arrays receive `f1`, `f2`, and so on. Constant
objectives remain explicit graph vertices.

```python
import pandas as pd
import misda

frame = pd.read_csv("objectives.csv")
mis_set = misda.discover(frame, seed=123)
```

## 2. Discovery controls

```python
mis_set = misda.discover(
    frame,
    aggressiveness=0.5,
    seed=123,
    name="experiment-a",
)
```

`aggressiveness` is a float in `[0,1]`. `0` selects the positive-signal onset
and `1` the null-calibrated endpoint. `seed` controls the reproducible
permutation procedures. `name` is an optional display label.

`discover()` does not accept a ranking policy and does not perform linear,
Pareto, or nonlinear candidate evaluation.

## 3. Dimensions and graphs

MISDA builds two graphs at the same data-driven threshold:

1. `G+`, the positive structural graph. Its independence number is
   `structural_dimension`.
2. `G±`, the signed dependence graph. Its independence number is
   `latent_dimension`.

Connected-component counts are topology diagnostics rather than dimensional
estimates. A connected graph can therefore have dimension greater than one.

```python
analysis = mis_set.analysis
print(analysis.original_dimension)
print(analysis.latent_dimension)
print(analysis.structural_dimension)
print(analysis.structural_components)
print(analysis.latent_components)
```

The separation status is also stored on `analysis`. `NULL_SEPARATION` means the
positive-signal onset precedes the null-calibrated endpoint;
`NO_NULL_SEPARATION` means no strict separation was established.

## 4. The canonical MIS universe

Every maximal independent set of `G+` is retained. `MISSet` has a fixed
canonical order established by the structural policy `structural_coverage`:

```text
size                  descending
neighborhood          descending
avg_external_degree   descending
span                  descending
```

A deterministic label-based tie-break makes the sequence reproducible but does
not create a new scientific rank.

```python
candidate = mis_set[0]

candidate.indices
candidate.objectives
candidate.size
candidate.structural.neighborhood
candidate.structural.neighborhood_ratio
candidate.structural.avg_external_degree
candidate.structural.span
```

A candidate does not carry a public ID or intrinsic `rank`. Its fixed position
in the owning `MISSet` is its operational identity.

## 5. Ranking

Use `rank()` to materialize an ordered view:

```python
ranking = misda.rank(mis_set)
```

The default is `policy="structural_coverage"`. The current release defines no
alternative policy yet.

A `Ranking` references the same candidate objects and does not mutate the
`MISSet`:

```python
ranking[0]             # underlying MISCandidate
ranking[:10]           # another Ranking view
ranking.selected       # ranking[0]
ranking.selected_dimension
ranking.groups         # scientific tie groups
```

The graph-derived structural dimension and the selected dimension are distinct
concepts:

```python
mis_set.analysis.structural_dimension
ranking.selected_dimension
```

Under the current complete enumeration and size-first canonical policy, the
canonical selected candidate necessarily has size equal to the structural
independence number. The definitions nevertheless remain separate so future
ranking policies can select differently without redefining graph dimension.

## 6. Dimensional support

`discover()` evaluates internal evidence for whether the graph-derived
dimensional description is sufficient. This is a global discovery diagnostic,
not a ranking metric and not a replacement dimension estimator.

The current mechanisms are:

- `TRANSITIVE_CHAINING`: indirect max-min positive paths to the retained
  candidate are stronger than direct positive association beyond a permutation
  null reference;
- `HIDDEN_SPECTRAL_STRUCTURE`: the first rank-correlation eigenvalue beyond the
  estimated latent signal dimension exceeds its column-permutation null mean.

If several candidates tie at the first `structural_coverage` rank, support is
evaluated for all of them using the same null permutations. Aggregate states
are:

```text
SUPPORTED             all tied first-rank candidates supported
PARTIALLY_SUPPORTED   some supported and some unsupported
UNSUPPORTED           none supported
```

Inspect the aggregate and individual evidence with:

```python
mis_set.support.status
mis_set.support.supported
mis_set.support.unsupported
support = mis_set.support.for_candidate(0)

support.status
support.reasons
support.transitivity_excess
support.spectral_excess
```

`SUPPORTED` means that these diagnostics found no contradiction. It does not
prove that the unknown true dimension equals the estimate.

## 7. Candidate evaluation

Use one API for all candidate-level evidence:

```python
misda.evaluate(
    mis_set,
    metrics=("linear", "pareto"),
)
```

Current families are:

```text
structural
linear
nonlinear
pareto
```

Structural metrics are already present after discovery. The other families are
attached only when requested. Evaluation never changes graph structure,
dimensions, the candidate universe, or the canonical order.

### 7.1 Candidate selection

The evaluation scope accepts:

```python
candidates="all"
candidates=10
candidates=[0, 4, 17]
candidates=ranking[:10]
```

If no selector is given:

- linear/Pareto-only calls evaluate all candidates;
- any call containing nonlinear evaluation operates on one candidate.

The scope applies to the whole call. Therefore:

```python
misda.evaluate(mis_set, metrics=("linear", "nonlinear"))
```

evaluates both families on one candidate. Use separate calls if different
scopes are desired.

Whenever fewer than all candidates are evaluated, `mis_set.report()` states the
scope and selection basis explicitly.

### 7.2 Linear reconstruction

Linear reconstruction predicts only eliminated objectives from retained
objectives using external PRESS/LOO semantics. It records untruncated R² and
jackknife uncertainty.

```python
candidate = ranking.selected

candidate.linear.mean_r2
candidate.linear.worst_r2
candidate.linear.r2("f7")
candidate.linear.jackknife.mean_r2_se
candidate.linear.jackknife.r2_se("f7")
```

When no objective is eliminated or a target is mathematically undefined, the
corresponding quantity is `None` with a machine-readable reason; artificial
perfect scores are not inserted.

### 7.3 Pareto preservation

Pareto evaluation currently assumes minimization and compares empirical
nondominated row sets:

```python
candidate.pareto.retention
candidate.pareto.validity
candidate.pareto.jaccard
candidate.pareto.exact_preservation
candidate.pareto.reduced_front_indices
```

Mixed directions are outside the current contract.

### 7.4 Nonlinear reconstruction

Nonlinear evidence is explicitly requested:

```python
misda.evaluate(
    mis_set,
    metrics=("nonlinear",),
    candidates=1,
)

candidate = ranking.selected
candidate.nonlinear.mean_r2
candidate.nonlinear.worst_r2
candidate.nonlinear.r2("f7")
```

The engine uses nested external leave-one-out Random Forest reconstruction,
internal discrete model selection, deterministic seed derivation, and tree
stopping based on computational versus sample uncertainty.

An optional sequential null reference is attached to the same nonlinear domain:

```python
misda.evaluate(
    mis_set,
    metrics=("nonlinear",),
    candidates=1,
    null_reference=True,
)

null = ranking.selected.nonlinear.null_reference
null.mean_null_r2
null.above_null_r2
null.incidental_reconstruction_rate
null.mc_se_mean_null_r2
```

## 8. Reports and graphs

```python
print(mis_set.report())
figure = mis_set.graph_plot(show=False)
```

The report renders only stored evidence; it does not trigger hidden evaluation.
`graph_plot()` draws `G+` and highlights the candidate selected by the supplied
ranking. To visualize another ranking snapshot:

```python
mis_set.graph_plot(ranking=ranking)
```

## 9. Benchmark evaluation

External truth is evaluated only after analysis:

```python
truth = {
    "name": "Synthetic case",
    "latent_expected": 2,
    "structural_expected": 2,
    "blocks_expected": [["f1", "f2"], ["f3", "f4"]],
    "pareto_expected": [0, 2, 5],
}

bench = misda.benchmark(mis_set, truth)
print(bench.report())
```

Truth never enters `discover()`, `evaluate()`, or `rank()`. Declared latent and
structural dimensions are compared with their corresponding graph independence
numbers. The canonical ranking's selected dimension is reported separately.

If Pareto truth is declared, the selected candidate must already have Pareto
evidence; `benchmark()` does not perform hidden candidate evaluation.

Repository-level reproducible batteries are available as:

```bash
python -m examples.benchmarks.run_benchmark --output results/benchmark.json
python -m examples.benchmarks.run_comparative --output results/comparative.json
```

The comparative battery uses a common external reconstruction metric for direct
MISDA/PCA comparison while preserving each method's native diagnostics as
separate estimands.

## 10. `alpha_null` convergence

The structural null estimator begins with at least `N` permutations and tracks
Monte Carlo uncertainty. It compares the discovery signature at the lower and
upper endpoints of that uncertainty interval:

```text
Sigma(alpha) = (
    structural_dimension,
    latent_dimension,
    complete structural_coverage tie-group ordering,
)
```

The estimate converges when both endpoints yield the same signature. Raw metric
values and literal graph identity are not required to match if they cannot
change a discrete discovery output. The autonomous upper limit remains
`B_max=10N`; reaching it returns the current estimate with explicit
non-convergence diagnostics.

## 11. Scope and limitations

- Static MISDA is the current active scientific path.
- Adaptive analysis is suspended and outside the current API and acceptance
  gate.
- Pairwise graph structure is correlation-based and does not establish
  causality.
- Structural dimension, latent dimension, connected-component counts, and a
  ranking-selected dimension are distinct quantities.
- Complete MIS enumeration is currently assumed; bounded partial enumeration
  remains future work.
- Alternative ranking policies remain future work.
- Maximization and mixed objective directions remain future work.
