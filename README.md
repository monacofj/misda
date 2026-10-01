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

`discover()` determines thresholds, builds `G+` and `G±`, estimates dimensions,
enumerates all structural MISs, computes structural metrics, establishes the
canonical structural order, and evaluates dimensional support. It does not
accept a user-selected ranking policy.

`MISSet.evaluate()` enriches already-discovered candidates without changing
their graph structure, dimensions, membership, or canonical positions. The
module form remains public and equivalent:

```python
misda.evaluate(mis_set, metrics=("linear", "pareto"))
```

Current metric families are:

```text
structural
linear
nonlinear
pareto
dominance
```

## Ranking

The canonical ranking is `size_span`:

```text
size   descending
span   descending
```

Thus:

```python
ranking = misda.rank(mis_set)
```

is equivalent to:

```python
ranking = misda.rank(mis_set, policy="size_span")
```

A `Ranking` is a view over the same MIS objects; it never reorders the owning
`MISSet`. Selection is explicit:

```python
mis = ranking.mis()                  # level=0, position=0
alternative = ranking.mis(0, 3)
alternative = ranking.mis(level=0, position=3)
```

Two experimental optimization-oriented ranking views are available after the
corresponding evidence has been evaluated:

```python
mis_set.evaluate(metrics=("pareto",), candidates="all")
pareto_ranking = misda.rank(mis_set, policy="pareto_retention")

mis_set.evaluate(metrics=("dominance",), candidates="all")
dominance_ranking = misda.rank(mis_set, policy="dominance_preservation")
```

`pareto_retention` ranks by empirical Pareto retention. It is not a guarantee
of global optimization equivalence.

`dominance_preservation` minimizes the fraction of row pairs with no dominance
in full observed `Y` that acquire a dominance relation after projection. Exact
scientific ties prefer the larger MIS operationally. Its selected reduction is
annotated as `NO_REDUNDANCY`, `SUPPORTED_REDUCTION`, or
`UNSUPPORTED_REDUCTION`; unsupported candidates remain returned and inspectable.

Neither experimental policy changes discovery or the canonical `size_span`
order. The October 2026 default-policy audit did **not** justify promoting
`dominance_preservation` to the public default under observation noise; the
full experimental history is recorded in the research notebook.

## Dimensional support

`discover()` evaluates internal evidence against the sufficiency of the
observed graph-derived description. The current diagnostics are:

- `TRANSITIVE_CHAINING`;
- `HIDDEN_SPECTRAL_STRUCTURE`.

Support is candidate-specific. The compatibility aggregate `mis_set.support`
summarizes the first structural-rank group, while
`mis_set.support_for(candidate)` retrieves evidence for a candidate selected by
any ranking.

`SUPPORTED` means that the current diagnostics found no internal contradiction;
it is not a proof of unknown global truth.

## Candidate evidence

Candidate evidence is typed and explicit:

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

Linear and Pareto evaluation normally operate on all candidates. Nonlinear
reconstruction is expensive and is explicitly scoped:

```python
mis_set.evaluate(metrics=("nonlinear",), candidates=ranking.mis())
```

Pareto evaluation assumes minimization. Objective projection can create new
strict dominance among rows that were incomparable in full `Y`, and can also
erase existing strict dominance when projection creates an exact tie.
Consequently, retention, validity, and Jaccard are distinct empirical
set-membership diagnostics.

## Reports and visualizations

`ranking.report()` is the complete self-contained report for a ranking-selected
answer. Reports consume stored state and do not trigger hidden evaluation.

```python
print(ranking.report())
print(ranking.mis().report())

ranking.mis().graph_plot()
ranking.mis().front_plot()
```

See [docs/visualization.md](docs/visualization.md) for the complete visualization
contract and notebook/terminal behavior.

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

Executable benchmark front ends include:

```bash
python -m benchmarks.run_controlled --output results/controlled.json
python -m benchmarks.run_controlled_noisy --output results/controlled-noisy.json
python -m benchmarks.run_sampling_robustness --output results/sampling-robustness.json
python -m benchmarks.run_noisy_robustness --output results/noisy-robustness.json
python -m benchmarks.run_comparison --output results/comparison.json
python -m benchmarks.run_classical --output results/classical.json
```

Repository benchmark notebooks are organized by what varies:

- [Controlled benchmark](benchmarks/controlled.ipynb): canonical clean 13-case reference (`Y=Z`).
- [Controlled benchmark with fixed noise](benchmarks/controlled_noisy.ipynb): the same cases at a reproducible `sigma=0.10` condition.
- [Sampling robustness](benchmarks/sampling_robustness.ipynb): independent clean samples with `sigma=0`.
- [Noise robustness](benchmarks/noisy_robustness.ipynb): repeated observation streams across a grid of `sigma` values.
- [MISDA/PCA comparison](benchmarks/comparison.ipynb): clean controlled diagnostics with explicit truth and a common external reconstruction score.
- [Classical DTLZ reference problems](benchmarks/classical.ipynb): reproducible DTLZ reference samples.
- [Optimization benchmark](benchmarks/optimization.ipynb): paired Full/Reduced optimization experiments evaluated in the original objective space.

The fixed-noise and robustness studies are validation experiments, not user-time
inference rules. Scientific policy-impact probes are retained as reproducible
manual workflows and do not run implicitly on ordinary pull requests.

`classical.ipynb` keeps analytical Pareto-manifold geometry as reference
context; it is **not re-labelled as MISDA latent or structural ground truth**.

## Documentation map

The repository has deliberately separate documentation layers:

- [docs/userguide.md](docs/userguide.md) — current public API and usage;
- [docs/adr/README.md](docs/adr/README.md) — authoritative normative
  methodological and architectural decisions;
- [docs/research-notes/README.md](docs/research-notes/README.md) — chronological
  research notebook containing hypotheses, experiments, negative results,
  corrections, validation evidence, and open questions;
- [docs/visualization.md](docs/visualization.md) and
  [docs/experimental-correlation.md](docs/experimental-correlation.md) —
  specialized current-capability documentation.

ADRs define what the current method is. Research notes explain how the project
arrived there and preserve investigations that were rejected, suspended, or
left open.

## Development status

MISDA is currently alpha software. The previous `analyze()`/`heavy()` result
model is not retained as a deprecated compatibility layer in the new static
API. Adaptive analysis is suspended and outside the current scientific
acceptance gate.

## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

## Reference

Souza, C. H., Monaco, F. J., Delbem, A. C. B., and Kuruvilla, J. A.
*Maximal Independent Structural Dimensionality Analysis* (in print), 2026.
