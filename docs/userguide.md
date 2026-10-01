<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# MISDA static user guide

The static API separates structural discovery, candidate evaluation, and
preference among candidates. This prevents evaluation or ranking choices from
feeding back into structural inference.

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
and `1` the null-calibrated endpoint, with interpolation in log-alpha space.
`seed` controls reproducible permutation procedures. `name` is an optional
display label.

`discover()` does not accept a ranking policy and does not perform linear,
Pareto, dominance, or nonlinear candidate evaluation.

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

## 4. The canonical MIS universe

Every maximal independent set of `G+` is retained. `MISSet` has a fixed
canonical order established by `size_span`:

```text
size   descending
span   descending
```

For a maximal independent set `S`, maximality implies that every vertex outside
`S` is adjacent to at least one selected vertex. Therefore
`neighborhood = n - size`; also, `avg_external_degree = span / size`.
`neighborhood`, `neighborhood_ratio`, and `avg_external_degree` remain useful
descriptive metrics but do not add independent ranking criteria once `size`
and `span` are known.

A deterministic label tie-break makes the sequence reproducible without
creating a new scientific rank.

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

## 5. Ranking

Use `rank()` to materialize an ordered view:

```python
ranking = misda.rank(mis_set)
mis = ranking.mis()
```

The public default remains `policy="size_span"`.

### 5.1 Pareto retention

The experimental `pareto_retention` policy ranks only by empirical Pareto
retention on observed `Y`. It requires stored Pareto evidence:

```python
mis_set.evaluate(metrics=("pareto",), candidates="all")
ranking = misda.rank(mis_set, policy="pareto_retention")
```

Or the user may explicitly authorize the evaluation cost:

```python
ranking = misda.rank(mis_set, policy="pareto_retention", accept_cost=True)
```

Exact retention ties are ordered with the smaller MIS first only as an
operational reduction-efficiency tie-break. Tied candidates remain in the same
scientific rank group.

### 5.2 Dominance preservation

The experimental `dominance_preservation` policy minimizes the fraction of row
pairs with no dominance in full `Y` that acquire a strict dominance relation
after projection. Lower is better; exact scientific ties are ordered with the
larger MIS first as a conservative operational tie-break.

```python
mis_set.evaluate(metrics=("dominance",), candidates="all")
ranking = misda.rank(mis_set, policy="dominance_preservation")
mis = ranking.mis()
print(ranking.assessment.status)
```

`ranking.assessment` never suppresses the selected MIS. It annotates it as
`NO_REDUNDANCY`, `SUPPORTED_REDUCTION`, or `UNSUPPORTED_REDUCTION`. Unsupported
candidates remain returned so their empirical or benchmark error can be
inspected.

Neither experimental policy is a proof of global optimization equivalence. The
October 2026 audit found that `dominance_preservation` is selection-equivalent
to `size_span` on the tested clean controlled batteries, but noisy observed `Y`
can make it prefer another representative MIS without a consistent improvement
in external Pareto truth. The default therefore remains `size_span`.

The scientific history and the aggressiveness follow-up are recorded in
`docs/research-notes/2026-10-ranking-policy-default-audit.md`.

### 5.3 Ranking views

A `Ranking` references the same MIS objects and does not mutate `MISSet`.
Selection uses scientific tie level and local position:

```python
mis = ranking.mis()                  # level=0, position=0
alternative = ranking.mis(0, 3)
alternative = ranking.mis(level=0, position=3)

ranking[:10]                         # another Ranking view
ranking.selected                     # compatibility alias for the first MIS
ranking.selected_dimension
ranking.groups
```

The graph-derived structural dimension and ranking-selected dimension are
distinct concepts:

```python
mis_set.analysis.structural_dimension
ranking.selected_dimension
```

## 6. Dimensional support

`discover()` evaluates internal evidence against the sufficiency of the
observed graph-derived dimensional description. The current mechanisms are:

- `TRANSITIVE_CHAINING`;
- `HIDDEN_SPECTRAL_STRUCTURE`.

Candidate-specific support is stored for every discovered MIS using shared null
permutations. The compatibility aggregate `mis_set.support` remains defined
over the first `size_span` tie group. Use `mis_set.support_for(candidate)` for a
candidate selected by any ranking.

```python
support = mis_set.support_for(ranking.mis())

support.status
support.reasons
support.transitivity_excess
support.spectral_excess
```

Aggregate first-group states are `SUPPORTED`, `PARTIALLY_SUPPORTED`, and
`UNSUPPORTED`. A supported result means that the current diagnostics found no
internal contradiction; it does not certify unknown global truth.

## 7. Candidate evaluation

Use the `MISSet` facade:

```python
mis_set.evaluate()
```

With no arguments the established default is
`metrics=("linear", "pareto")`. The module form remains public and equivalent:

```python
misda.evaluate(mis_set, metrics=("linear", "pareto"))
```

Current families are:

```text
structural
linear
nonlinear
pareto
dominance
```

Evaluation never changes graph structure, dimensions, the candidate universe,
or canonical order.

### 7.1 Evaluation scope

```python
candidates="all"
candidates=ranking.mis()
candidates=[ranking.mis(0, 0), ranking.mis(0, 1)]
candidates=ranking[:10]
```

Existing integer/index selectors remain supported for compatibility. Calls
containing nonlinear evaluation default to one candidate because nonlinear
reconstruction is expensive; linear/Pareto-only calls default to all
candidates.

### 7.2 Linear reconstruction

Linear reconstruction predicts only eliminated objectives from retained
objectives using external PRESS/LOO semantics and records untruncated R².

```python
mis = ranking.mis()
mis.linear.mean_r2
mis.linear.worst_r2
mis.linear.r2("f7")
mis.linear.jackknife.mean_r2_se
```

Undefined quantities are represented explicitly rather than replaced by
artificial perfect scores.

### 7.3 Pareto preservation

Pareto evaluation assumes minimization and compares empirical nondominated row
sets:

```python
mis.pareto.retention
mis.pareto.validity
mis.pareto.jaccard
mis.pareto.exact_preservation
mis.pareto.reduced_front_indices
```

Objective projection can create new strict dominance relations among rows that
were incomparable in full `Y`. It can also erase an existing strict dominance
relation when all strict coordinates are removed and projected rows become
exactly tied. Therefore neither empirical nondominated set is guaranteed to be
a subset of the other. Retention, validity, and Jaccard remain distinct.

The counterexample that corrected the earlier subset assumption is preserved in
`docs/research-notes/2026-09-pareto-projection-semantics-correction.md`.

`mis_set.pareto_stability` separately reports observed-front fraction,
range-normalized dominance margins, and range-normalized additive epsilon. No
fixed pass/fail cutoff is imposed.

### 7.4 Nonlinear reconstruction

```python
mis_set.evaluate(
    metrics=("nonlinear",),
    candidates=ranking.mis(),
)

mis = ranking.mis()
mis.nonlinear.mean_r2
mis.nonlinear.worst_r2
```

The engine uses nested external leave-one-out Random Forest reconstruction,
internal discrete model selection, deterministic seed derivation, and
uncertainty-driven tree stopping. An optional null reference is requested with
`null_reference=True`.

## 8. Reports and visualizations

The ranking report is the complete self-contained user report:

```python
ranking = misda.rank(mis_set)
print(ranking.report())
```

Reports and visualizations consume stored state and do not trigger hidden
candidate evaluation.

```python
mis = ranking.mis()
print(mis.report())
mis.graph_plot(show=False)
mis.front_plot(show=False)
```

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

Truth never enters `discover()`, `MISSet.evaluate()`, or `rank()`.

Repository-level reproducible batteries include:

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
- `noisy_robustness.ipynb`: observation-noise sweep;
- `comparison.ipynb`: MISDA/PCA comparison on controlled truth;
- `classical.ipynb`: classical DTLZ reference problems;
- `optimization.ipynb`: paired Full/Reduced optimization proof-of-concept.

The ordinary acceptance workflow keeps lightweight regression gates automatic.
Heavier scientific validation batteries are run explicitly. The ranking-policy
impact workflow introduced by PR #79 is manual-only and requires an explicit
selection of the desired audit battery.

The empirical history of benchmark investigations is not duplicated in a
separate validation ledger. It lives chronologically in
`docs/research-notes/README.md`.

## 10. `alpha_null` empirical null envelope

The structural null endpoint uses a fixed data-derived permutation budget
`B=N`. Each independent-column permutation contributes its maximum positive
pairwise correlation, and the estimator uses the empirical upper envelope:

```text
r_null = max(m_1, ..., m_N)
log_alpha_null = positive_correlation_log_p(r_null, N)
```

The former sequential `mean ± MC-SE` rule and its `10N` cap are not part of the
current estimator. See ADR 0015 for the normative decision.

## 11. Documentation and scientific provenance

The documentation layers are intentionally distinct:

- `docs/adr/` is the authoritative methodological and architectural
  specification;
- `docs/research-notes/` is the chronological laboratory notebook containing
  hypotheses, experiments, negative results, corrections, and open questions;
- this user guide describes current operation of the public API;
- specialized current-capability documents cover visualization and explicitly
  gated experimental backends.

If a research note and an ADR ever conflict, the ADR is normative. A new
research result becomes an ADR only when it changes the method or software
contract.

## 12. Scope and limitations

- Static MISDA is the current active scientific path.
- Adaptive analysis is suspended and outside the current API and acceptance gate.
- Pairwise graph structure is correlation-based and does not establish causality.
- Structural dimension, latent dimension, component counts, and selected dimension are distinct quantities.
- Complete MIS enumeration is currently assumed; bounded partial enumeration remains future work.
- `pareto_retention` and `dominance_preservation` are experimental ranking policies.
- Maximization and mixed objective directions remain future work.
