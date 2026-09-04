# Maximal Independent Structural Dimensionality Analysis

<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/monacofj/misda/blob/main/examples/benchmark.ipynb)
[![REUSE status](https://api.reuse.software/badge/github.com/monacofj/misda)](https://api.reuse.software/info/github.com/monacofj/misda)

MISDA is a graph-theoretic method for studying and reducing the objective space
of multi-objective problems while retaining original, interpretable variables.
The current scientific path is the static method.

MISDA distinguishes:

- **latent dimension**: the independence number of the signed dependence graph
  `G±`, where significant positive and negative dependencies form edges;
- **structural dimension**: the independence number of the positive-redundancy
  graph `G+`;
- **selected dimension**: the size of the candidate selected by a particular
  ranking.

Connected-component counts are topology diagnostics, not dimensional
estimates. Negative associations affect latent dependence but do not create
positive-redundancy edges in `G+`.

## Installation

```bash
git clone https://github.com/monacofj/misda.git
cd misda
python -m pip install .
```

MISDA requires Python 3.8 or newer and depends on NumPy, pandas, SciPy,
NetworkX, Matplotlib, and scikit-learn.

## Quick start

The static API deliberately separates structural discovery, candidate
evaluation, and ranking.

```python
import pandas as pd
import misda

frame = pd.read_csv("my_mop_data.csv")

mis_set = misda.discover(
    frame,
    aggressiveness=0.5,
    seed=123,
    name="Demo",
)

misda.evaluate(
    mis_set,
    metrics=("linear", "pareto"),
)

structural = misda.rank(mis_set)

print(mis_set.report())
print(structural.selected.objectives)
print(structural.selected_dimension)
print(structural.selected.linear.mean_r2)

figure = mis_set.graph_plot(show=False, ranking=structural)
```

`discover()` determines thresholds, builds `G+` and `G±`, estimates dimensions,
enumerates all structural MISs, computes structural metrics, establishes the
canonical structural order, and evaluates dimensional support. It does not
accept a user-selected ranking policy.

`evaluate()` enriches already-discovered candidates without changing their
canonical positions. Current metric families are:

```text
structural
linear
nonlinear
pareto
```

Candidate evidence is exposed through typed domains:

```python
candidate = structural.selected

candidate.size
candidate.structural.neighborhood
candidate.linear.mean_r2
candidate.linear.r2("f7")
candidate.pareto.retention
candidate.pareto.validity
candidate.pareto.jaccard
```

Linear and Pareto evaluation default to all candidates. A call containing
`nonlinear` defaults to the first candidate because nonlinear reconstruction is
expensive. The scope can always be made explicit:

```python
misda.evaluate(mis_set, metrics=("linear",), candidates="all")
misda.evaluate(mis_set, metrics=("nonlinear",), candidates=1)
misda.evaluate(mis_set, metrics=("nonlinear",), candidates=structural[:5])
```

Whenever an evaluation covers fewer than all candidates, reports state that
scope explicitly.

## Ranking

The current canonical policy is `structural_coverage`:

```text
size                  descending
neighborhood          descending
avg_external_degree   descending
span                  descending
```

Thus:

```python
structural = misda.rank(mis_set)
```

is equivalent to:

```python
structural = misda.rank(mis_set, policy="structural_coverage")
```

A `Ranking` is a view over the same candidates; it does not reorder `mis_set`.
Slicing returns another ranking view:

```python
top10 = structural[:10]
```

Candidate identity remains its fixed position in `mis_set`. Contextual rank is
not stored on the candidate.

The graph-derived structural dimension and a ranking-selected dimension are
deliberately distinct concepts:

```python
mis_set.analysis.structural_dimension
structural.selected_dimension
```

Under the current complete enumeration and size-first `structural_coverage`
policy they coincide for the canonical selection, but they are defined
independently.

## Dimensional support

`discover()` also evaluates whether the data contain internal evidence against
the sufficiency of the graph-derived dimensional description. The current
diagnostics are:

- `TRANSITIVE_CHAINING`: strong indirect positive chains are substantially
  stronger than direct association to the retained candidate;
- `HIDDEN_SPECTRAL_STRUCTURE`: organized rank-correlation structure remains
  beyond the estimated latent dimension.

If several candidates are scientifically tied at the first structural rank,
support is evaluated for all of them rather than depending on an arbitrary
deterministic tie-break.

```python
mis_set.support.status
mis_set.support.supported
mis_set.support.unsupported
mis_set.support.for_candidate(0)
```

Aggregate status is `SUPPORTED`, `PARTIALLY_SUPPORTED`, or `UNSUPPORTED`.

## Nonlinear evaluation

Nonlinear reconstruction is requested through the same evaluation API:

```python
misda.evaluate(
    mis_set,
    metrics=("nonlinear",),
    candidates=1,
)

print(structural.selected.nonlinear.mean_r2)
```

The nonlinear engine uses nested external leave-one-out Random Forest
reconstruction, internal model selection, deterministic seeds, and
data-driven tree stopping. Its optional sequential null reference is requested
with `null_reference=True`.

## Benchmarks

External truth belongs exclusively to benchmark infrastructure. It is never
passed into `discover()`, `evaluate()`, or `rank()`.

```python
truth = {
    "name": "Demo benchmark",
    "latent_expected": 2,
    "structural_expected": 3,
    "blocks_expected": [["f1", "f2"], ["f3"], ["f4", "f5"]],
    "pareto_expected": [0, 4, 9],
}

bench = misda.benchmark(mis_set, truth)
print(bench.report())
```

Executable benchmark front ends:

```bash
python -m examples.benchmarks.run_benchmark --output results/benchmark.json
python -m examples.benchmarks.run_comparative --output results/comparative.json
```

- [Canonical benchmark notebook](examples/benchmark.ipynb)
- [Static MISDA and PCA notebook](examples/comparative.ipynb)

The comparative artifact keeps MISDA's native eliminated-objective
reconstruction diagnostics separate from PCA's native reconstruction curve.
Direct MISDA/PCA comparison uses the common external
`global_standardized_external_r2` metric.

## Development status

MISDA is currently alpha software. The previous `analyze()`/`heavy()` result
model is not retained as a deprecated compatibility layer in the new static
API. Adaptive analysis is suspended and outside the current scientific
acceptance gate.

See [docs/userguide.md](docs/userguide.md) for the API and metric semantics, and
[docs/decisions.md](docs/decisions.md) for the normative methodological and
architectural decisions.

## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

## Reference

Souza, C. H., Monaco, F. J., Delbem, A. C. B., and Kuruvilla, J. A.
*Maximal Independent Structural Dimensionality Analysis* (in print), 2026.
