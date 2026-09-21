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
canonical order established by the structural policy `size_span`:

```text
size   descending
span   descending
```

For a maximal independent set `S`, every vertex outside `S` must be adjacent to
at least one vertex in `S`; otherwise `S` would not be maximal. Therefore
`neighborhood = n - size`. Also, `avg_external_degree = span / size`. Once
`size` is fixed, neither quantity adds an independent ranking criterion. These
metrics remain available for structural description, but the scientific
ranking key is explicitly `size` followed by `span`.

A deterministic label-based tie-break makes the sequence reproducible but does
not create a new scientific rank. Equal `size` and `span` therefore define a
scientific tie group.

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

The default is `policy="size_span"`. The current release defines no alternative
policy yet.

A `Ranking` references the same MIS objects and does not mutate the
`MISSet`. The user-facing selector is `mis(level, position)`:

```python
mis = ranking.mis()                  # level=0, position=0
alternative = ranking.mis(0, 3)
alternative = ranking.mis(level=0, position=3)

ranking[:10]                         # another Ranking view
ranking.selected_dimension
ranking.groups                       # scientific tie groups
```

`level` selects a scientific tie group and `position` selects one MIS
inside that group; both are zero-based. The returned object is the existing
`MISCandidate` owned by `mis_set`, not a copy or ranking-specific wrapper.

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

If several candidates tie at the first `size_span` rank, support is evaluated
for all of them using the same null permutations. Aggregate states are:

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
support = mis_set.support.for_candidate(ranking.mis())

support.status
support.reasons
support.transitivity_excess
support.spectral_excess
```

`SUPPORTED` means that these diagnostics found no contradiction. It does not
prove that the unknown true dimension equals the estimate.

## 7. MIS evaluation

Use the `MISSet` facade for candidate-level evidence:

```python
mis_set.evaluate()
```

With no arguments this preserves the established default:
`metrics=("linear", "pareto")`. During alpha, `MISSet.evaluate(...)` is the
only public evaluation entry point.

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
candidates=ranking.mis()
candidates=[ranking.mis(0, 0), ranking.mis(0, 1)]
candidates=ranking[:10]
```

If no selector is given:

- linear/Pareto-only calls evaluate all candidates;
- any call containing nonlinear evaluation operates on one candidate.

The scope applies to the whole call. Therefore:

```python
mis_set.evaluate(metrics=("linear", "nonlinear"))
```

evaluates both families on one candidate. Use separate calls if different
scopes are desired.

Whenever fewer than all candidates are evaluated, `ranking.report()` states the
scope and selection basis explicitly.

### 7.2 Linear reconstruction

Linear reconstruction predicts only eliminated objectives from retained
objectives using external PRESS/LOO semantics. It records untruncated R² and
jackknife uncertainty.

```python
mis = ranking.mis()

mis.linear.mean_r2
mis.linear.worst_r2
mis.linear.r2("f7")
mis.linear.jackknife.mean_r2_se
mis.linear.jackknife.r2_se("f7")
```

When no objective is eliminated or a target is mathematically undefined, the
corresponding quantity is `None` with a machine-readable reason; artificial
perfect scores are not inserted.

### 7.3 Pareto preservation

Pareto evaluation currently assumes minimization and compares empirical
nondominated row sets:

```python
mis.pareto.retention
mis.pareto.validity
mis.pareto.jaccard
mis.pareto.exact_preservation
mis.pareto.reduced_front_indices
```

Exact membership agreement is deliberately separate from observed-data Pareto
stability. When Pareto evaluation is requested, `mis_set.pareto_stability`
reports the observed-front fraction, range-normalized dominance margins, and
range-normalized additive epsilon from a reduced front to the observed full
front. These quantities help distinguish a saturated/perturbation-sensitive
front from a geometrically poor reduction; no fixed pass/fail cutoff is imposed.

Mixed directions are outside the current contract.

### 7.4 Nonlinear reconstruction

Nonlinear evidence is explicitly requested:

```python
mis_set.evaluate(
    metrics=("nonlinear",),
    candidates=ranking.mis(),
)

mis = ranking.mis()
mis.nonlinear.mean_r2
mis.nonlinear.worst_r2
mis.nonlinear.r2("f7")
```

The engine uses nested external leave-one-out Random Forest reconstruction,
internal discrete model selection, deterministic seed derivation, and tree
stopping based on computational versus sample uncertainty.

An optional sequential null reference is attached to the same nonlinear domain:

```python
mis_set.evaluate(
    metrics=("nonlinear",),
    candidates=ranking.mis(),
    null_reference=True,
)

null = ranking.mis().nonlinear.null_reference
null.mean_null_r2
null.above_null_r2
null.incidental_reconstruction_rate
null.mc_se_mean_null_r2
```

## 8. Reports and visualizations

The ranking report is the complete, self-contained user report:

```python
ranking = misda.rank(mis_set)

print(ranking.report())
```

It preserves the complete discovery, support, evaluation, ranking, and
selected-MIS evidence already available in the public report contract. Report fields that are easy to misread carry a concise interpretation after
an em dash, for example graph dimensions versus component counts, threshold
calibration endpoints, null-envelope completion, ranking ties, support
diagnostics, and Pareto summaries. Short annotations remain inline. Longer
technical explanations wrap below the value and align from the value column;
when a separate intuitive gloss exists, it follows on an aligned parenthesized
line. This keeps terminal/notebook output readable without discarding either
technical or intuitive meaning. These explanations are descriptive only:
reporting never recomputes scientific state.

Existing `mis_set.report()` behavior remains supported for compatibility; the
reporting contract does not authorize shrinking or omitting evidence.

Inspect one selected MIS directly:

```python
mis = ranking.mis()

print(mis.report())
graph = mis.graph_plot(show=False)
front = mis.front_plot(show=False)
```

Explore another MIS in ranking coordinates without manipulating canonical
indices:

```python
ranking.mis(0, 3).report()
ranking.mis(0, 3).front_plot()
```

All report and visualization views consume only stored state; they do not
trigger hidden candidate evaluation. `graph_plot()` draws stored `G+` and
highlights the selected MIS. `front_plot()` requires Pareto evidence already
stored for that MIS and renders the full/reduced empirical-front membership
with Plotly. With at least three objectives it uses a rotatable 3D scatter;
with two objectives it uses a 2D scatter. Objective selectors alter only the
displayed projection, not Pareto membership.

There is no MISSet-level plotting selector in the alpha API. Selection belongs
to `Ranking.mis()`; plotting belongs to the returned MIS.

Inside notebooks, `front_plot(show=True)` displays inline. From a terminal it
writes a self-contained temporary HTML file and attempts to open it in the
system browser; if automatic browser launch is unavailable, the retained file
path is reported. The Plotly `Figure` is returned in all cases.

See `docs/visualization.md` and ADR 0018 for the full visualization contract.

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

Truth never enters `discover()`, `MISSet.evaluate()`, or `rank()`. Declared latent and
structural dimensions are compared with their corresponding graph independence
numbers. The canonical ranking's selected dimension is reported separately.

If Pareto truth is declared, the selected candidate must already have Pareto
evidence; `benchmark()` does not perform hidden candidate evaluation.

Repository-level reproducible batteries are available as:

```bash
python -m benchmarks.run_controlled --output results/controlled.json
python -m benchmarks.run_controlled_noisy --output results/controlled-noisy.json
python -m benchmarks.run_sampling_robustness --output results/sampling-robustness.json
python -m benchmarks.run_noisy_robustness --output results/noisy-robustness.json
python -m benchmarks.run_comparison --output results/comparison.json
python -m benchmarks.run_classical --output results/classical.json
```

Notebook-level validation is organized under `benchmarks/`:

- `controlled.ipynb`: exact-observation 13-case reference;
- `controlled_noisy.ipynb`: fixed `sigma=0.10` reference condition;
- `sampling_robustness.ipynb`: independent clean samples with `sigma=0`;
- `noisy_robustness.ipynb`: observation-noise sweep across `sigma` and replicate streams;
- `comparison.ipynb`: MISDA/PCA comparison on controlled truth;
- `classical.ipynb`: classical DTLZ reference problems.

The comparison battery uses a common external reconstruction metric for direct
MISDA/PCA comparison while preserving each method's native diagnostics as
separate estimands.

The ordinary acceptance workflow keeps the clean scientific battery and method
comparison as routine regression gates. The heavier fixed-noise and robustness
studies run in the path-scoped/manual `extended benchmark validation` workflow,
which preserves their JSON artifacts. Current empirical evidence is recorded in
`docs/validation_results.md`; that file is a reproducibility record, not a
normative ADR.

## 10. `alpha_null` empirical null envelope

The structural null endpoint uses a fixed data-derived permutation budget
`B=N`. Each independent-column permutation contributes its maximum positive
pairwise correlation, and the estimator uses the empirical upper envelope:

```text
r_null = max(m_1, ..., m_N)
log_alpha_null = positive_correlation_log_p(r_null, N)
```

The sequential `mean ± MC-SE` stopping rule and its former `10N` cap are no
longer part of the public estimator. Repeated calls with the same seed are
reproducible. Legacy uncertainty fields remain available for compatibility, but
quantities whose former Monte Carlo mean interpretation no longer applies are
reported without misleading uncertainty semantics. See ADR 0015 for the
normative decision.

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
