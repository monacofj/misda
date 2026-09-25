# Maximal Independent Structural Dimensionality Analysis

<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/monacofj/misda/blob/main/benchmarks/controlled.ipynb)
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
NetworkX, Matplotlib, Plotly, and scikit-learn.

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

mis_set.evaluate()

ranking = misda.rank(mis_set)
mis = ranking.mis()

print(ranking.report())
print(mis.objectives)
print(ranking.selected_dimension)
print(mis.linear.mean_r2)

graph_figure = mis.graph_plot(show=False)
front_figure = mis.front_plot(show=False)
```

`ranking.report()` is self-contained. Compact single-line annotations use
`— technical definition (intuitive gloss)` to clarify dimensions, graph
topology, threshold/null calibration, ranking ties, support diagnostics, and
evaluation summaries without triggering hidden computation or removing any
evidence.

`discover()` determines thresholds, builds `G+` and `G±`, estimates dimensions,
enumerates all structural MISs, computes structural metrics, establishes the
canonical structural order, and evaluates dimensional support. It does not
accept a user-selected ranking policy.

`MISSet.evaluate()` enriches already-discovered MISs without changing their
canonical positions. With no arguments it preserves the established default:
linear and Pareto evidence for the default candidate scope. The module form
`misda.evaluate(mis_set, ...)` remains equivalent for compatibility.

Current metric families are:

```text
structural
linear
nonlinear
pareto
```

Candidate evidence is exposed through typed domains:

```python
mis = ranking.mis()

mis.size
mis.structural.neighborhood
mis.linear.mean_r2
mis.linear.r2("f7")
mis.pareto.retention
mis.pareto.validity
mis.pareto.jaccard
```

Linear and Pareto evaluation default to all candidates. A call containing
`nonlinear` defaults to the first candidate because nonlinear reconstruction is
expensive. The scope can always be made explicit:

```python
mis_set.evaluate(metrics=("linear",), candidates="all")
mis_set.evaluate(metrics=("nonlinear",), candidates=ranking.mis())
mis_set.evaluate(metrics=("nonlinear",), candidates=ranking[:5])
```

Whenever fewer than all candidates are evaluated, reports state that scope
explicitly.

## Ranking

The current canonical policy is `size_span`:

```text
size   descending
span   descending
```

For a maximal independent set `S` of `G+`, every vertex outside `S` must be
adjacent to at least one vertex in `S`; otherwise `S` would not be maximal.
Therefore `neighborhood = n - size`. Also,
`avg_external_degree = span / size`. Once `size` is fixed, neither quantity
adds an independent ranking criterion. They remain available as descriptive
structural diagnostics, but the scientific ranking key is explicitly
`size` followed by `span`.

Thus:

```python
ranking = misda.rank(mis_set)
```

is equivalent to:

```python
ranking = misda.rank(mis_set, policy="size_span")
```

An experimental Pareto-aware view is also available after candidate Pareto
evaluation:

```python
mis_set.evaluate(metrics=("pareto",), candidates="all")
ranking = misda.rank(mis_set, policy="size_pareto")
```

`size_pareto` keeps size first and then prefers greater empirical Pareto
retention. It is not the default and is not a guarantee of global optimization
safety.

A deterministic label-based tie-break provides reproducible order inside a
scientific tie but does not create a new rank group.

A `Ranking` is a view over the same MIS objects; it does not reorder
`mis_set`. Select one MIS by scientific tie level and local position:

```python
mis = ranking.mis()                  # level=0, position=0
alternative = ranking.mis(0, 3)
alternative = ranking.mis(level=0, position=3)
```

Slicing still returns another ranking view:

```python
top10 = ranking[:10]
```

The selected object is the existing MIS owned by `mis_set`, not a copy.
Contextual rank, level, and position are not stored on the MIS itself.

The graph-derived structural dimension and a ranking-selected dimension are
deliberately distinct concepts:

```python
mis_set.analysis.structural_dimension
ranking.selected_dimension
```

Under the current complete enumeration and size-first `size_span` policy they
coincide for the canonical selection, but they are defined independently.

## Visualizations

The preferred visualization workflow selects an MIS from a ranking and lets
that MIS visualize its stored state:

```python
mis = ranking.mis()
mis.graph_plot()
mis.front_plot()

ranking.mis(0, 3).front_plot()
```

`level` and `position` therefore belong to `Ranking.mis()`, not to the MIS
itself. Existing `MISSet.graph_plot(...)` and `MISSet.front_plot(...)`
selection forms remain supported for compatibility.

`graph_plot()` renders the stored structural graph. `front_plot()` renders
already evaluated Pareto-preservation state as an interactive Plotly scatter,
with rotatable 3D projection when at least three objectives are available and
objective selectors for the displayed axes. It does not trigger hidden Pareto
evaluation. See [docs/visualization.md](docs/visualization.md) for the complete
visualization contract and terminal/browser fallback behavior.

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
mis_set.support.for_candidate(ranking.mis())
```

Aggregate status is `SUPPORTED`, `PARTIALLY_SUPPORTED`, or `UNSUPPORTED`.

## Nonlinear evaluation

Nonlinear reconstruction is requested through the same evaluation API:

```python
mis_set.evaluate(
    metrics=("nonlinear",),
    candidates=ranking.mis(),
)

print(ranking.mis().nonlinear.mean_r2)
```

The nonlinear engine uses nested external leave-one-out Random Forest
reconstruction, internal model selection, deterministic seeds, and
data-driven tree stopping. Its optional sequential null reference is requested
with `null_reference=True`.

## Benchmarks

External truth belongs exclusively to benchmark infrastructure. It is never
passed into `discover()`, `MISSet.evaluate()`, or `rank()`.

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
python -m benchmarks.run_controlled --output results/controlled.json
python -m benchmarks.run_controlled_noisy --output results/controlled-noisy.json
python -m benchmarks.run_sampling_robustness --output results/sampling-robustness.json
python -m benchmarks.run_noisy_robustness --output results/noisy-robustness.json
python -m benchmarks.run_comparison --output results/comparison.json
python -m benchmarks.run_classical --output results/classical.json
```

The repository-level benchmark notebooks are organized by what varies:

- [Controlled benchmark](benchmarks/controlled.ipynb): canonical 13-case clean reference (`Y=Z`).
- [Controlled benchmark with fixed noise](benchmarks/controlled_noisy.ipynb): the same 13 cases at the reproducible `sigma=0.10` reference condition.
- [Sampling robustness](benchmarks/sampling_robustness.ipynb): repeated clean samples with `sigma=0`, isolating finite-sample variability.
- [Noise robustness](benchmarks/noisy_robustness.ipynb): repeated samples and observation streams across a grid of `sigma` values.
- [MISDA/PCA comparison](benchmarks/comparison.ipynb): clean controlled diagnostics with explicit truth and a common external reconstruction score.
- [Classical DTLZ reference problems](benchmarks/classical.ipynb): reproducible DTLZ2/DTLZ5 Pareto-front samples with analytical front geometry retained only as reference context.

The fixed-noise controlled notebook uses a scale-relative `sigma=0.10`
observation regime with a distinct observation seed as a reproducible reference
condition. It is not a robustness threshold. `sampling_robustness.ipynb`
varies only the clean sample, whereas `noisy_robustness.ipynb` varies
observation noise intensity and replicate streams. The heavy robustness runs
are kept outside the ordinary acceptance gate; the path-scoped
`extended benchmark validation` workflow executes them and preserves their JSON
artifacts when the validation infrastructure changes or when triggered manually.

The method comparison uses clean controlled problems with explicit truth. MISDA's
latent and structural dimensions remain native MISDA estimands; PCA remains a
linear reconstruction curve unless a component-selection protocol is explicitly
defined. The comparison therefore does not impose an arbitrary explained-
variance cutoff. Direct MISDA/PCA comparison uses the common external
`global_standardized_external_r2` metric at explicitly named dimensions.

`classical.ipynb` applies MISDA to reproducible on-front DTLZ2 and DTLZ5
samples. Their analytical Pareto-manifold geometry is retained as reference
context, but is not re-labelled as MISDA latent or structural ground truth.

The latest extended validation evidence, including the separation between
dimensional recovery and Pareto-membership stability, is recorded in
[docs/validation_results.md](docs/validation_results.md).

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
