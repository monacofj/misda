# Maximal Independent Structural Dimensionality Analysis

<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/monacofj/misda/blob/refactor/examples/benchmark.ipynb)
[![REUSE status](https://api.reuse.software/badge/github.com/monacofj/misda)](https://api.reuse.software/info/github.com/monacofj/misda)

MISDA is a graph-theoretic method for reducing the objectives of a
multi-objective problem while retaining original, interpretable variables. The
current supported workflow is the refactored **static** pipeline.

MISDA reports three deliberately distinct quantities:

- **latent dimension**: connected components in the signed dependence graph;
- **structural dimension**: connected components in the positive-dependence
  graph;
- **preferred MIS size**: number of original objectives selected in the
  highest-ranked maximal independent set.

Negative associations connect latent structure but never create redundancy
edges in the structural graph. Constant objectives remain explicit isolated
vertices.

## Installation

```bash
git clone https://github.com/monacofj/misda.git
cd misda
python -m pip install .
```

MISDA requires Python 3.8 or newer and depends on NumPy, pandas, SciPy,
NetworkX, Matplotlib, and scikit-learn.

## Quick start

```python
import pandas as pd
import misda

frame = pd.read_csv("my_mop_data.csv")

result = misda.analyze(
    frame,
    aggressiveness=0.5,
    max_evaluated_mis=3,
    seed=123,
    name="Demo",
)

print(result.summary())
print(result.best_mis.objectives)
print(result.best_mis.evaluation["linear_reconstruction"])

figure = result.graph_plot(show=False)
```

`analyze()` always stores every ranked MIS. Light evaluation — external linear
reconstruction, delete-one uncertainty, and Pareto preservation — is attached
to the requested ranked prefix. Use `max_evaluated_mis` to limit that work.

The result tree separates global analysis, candidates, and reproducibility:

```python
result.analysis.structural_dimension
result.analysis.latent_dimension
result.selected_dimension
result.analysis.separation_status
result.mis
result.execution.configuration
result.execution.timings
result.execution.convergence
```

For synthetic or empirical cases with an external declaration, compare the
completed analysis without passing that declaration into `analyze()`:

```python
truth = {
    "name": "Demo benchmark",
    "latent_expected": 2,
    "structural_expected": 3,
    "blocks_expected": [["f1", "f2"], ["f3"], ["f4", "f5"]],
    "pareto_expected": [0, 4, 9],
    "feature": "Known synthetic structure.",
    "intuition": "One representative per structural block.",
    "graph_expected": "Three disjoint positive components.",
}

bench = misda.benchmark(result, truth)
print(bench.report())
```

Analysis values remain in `bench.result`; only comparisons requiring the
external declaration are stored directly in `bench`.

`structural_dimension` and `latent_dimension` on `result.analysis` are graph
component counts. The selected dimension is `result.selected_dimension`, the
size of the preferred MIS. `benchmark()` compares this selected dimension with
`structural_expected`; it does not present dependence connectivity as an
estimate of latent generative dimension.

## On-demand heavy evaluation

Nonlinear reconstruction is intentionally opt-in. Select candidates by their
position in `result.mis`:

```python
misda.heavy(result, [0, 2], null_reference=True)
print(result.report())
```

The heavy path uses nested leave-one-out Random Forest evaluation, internal
model selection, data-driven tree stopping, and an optional sequential null
reference. It mutates the selected candidates in place and records convergence,
uncertainty, seed, and timing. A callable `cancel_requested` may be supplied by
interactive applications.

## Reproducible benchmarks

The executable benchmark modules are the source of truth; the notebooks are
thin front ends over the same functions.

```bash
python -m examples.benchmarks.run_benchmark --output results/benchmark.json
python -m examples.benchmarks.run_comparative --output results/comparative.json
```

- [Canonical benchmark notebook](examples/benchmark.ipynb)
- [Static MISDA and PCA notebook](examples/comparative.ipynb)

The comparative artifact keeps PCA global standardized reconstruction R²
separate from MISDA reconstruction of eliminated objectives. They describe
different estimands and are not merged into a single score.

## Compatibility status

The supported and acceptance-tested path is `misda.analyze(...,
method="static")`, which is also the default. Transitional aliases such as
`caution`, `result.plot()`, `result.alpha_min`, and `result.mis_sets` emit
`DeprecationWarning`; use `aggressiveness`, `graph_plot()`, the `analysis` tree,
and `result.mis` instead.

The earlier adaptive implementation remains suspended for compatibility work.
It is not part of the current scientific baseline, documentation contract, or
test acceptance gate.

See the [user guide](docs/userguide.md) for the complete result schema,
evaluation semantics, migration notes, and limitations. See
[design notes](docs/design_notes.md) for the statistical and architectural
rationale.

## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

## Reference

Souza, C. H., Monaco, F. J., Delbem, A. C. B., and Kuruvilla, J. A.
*Maximal Independent Structural Dimensionality Analysis* (in print), 2026.
