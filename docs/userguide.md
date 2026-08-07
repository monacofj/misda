<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# MISDA static user guide

This guide documents the refactored static API. It distinguishes structural
estimation, representative selection, and optional evaluation so that a result
never implies evidence that was not computed.

## 1. Input contract

`misda.analyze()` accepts a two-dimensional NumPy array or pandas DataFrame.
Values must be finite, real, numeric, and contain at least four observations.
DataFrame labels are preserved; arrays receive `f1`, `f2`, and so on. Constant
objectives are retained as explicit isolated vertices rather than silently
discarded.

```python
import pandas as pd
import misda

frame = pd.read_csv("objectives.csv")
result = misda.analyze(frame, seed=123)
```

## 2. Static analysis controls

```python
result = misda.analyze(
    frame,
    method="static",
    aggressiveness=0.5,
    rank_policy="default",
    max_evaluated_mis=3,
    seed=123,
    name="experiment-a",
)
```

| Argument | Meaning |
|---|---|
| `aggressiveness` | Float in `[0, 1]`. `0` selects the positive-signal onset and `1` the null-calibrated endpoint; larger values generally admit more positive redundancy edges. |
| `rank_policy` | Candidate ordering policy. The current supported value is `"default"`. |
| `max_evaluated_mis` | Positive integer or `None`. Limits light evaluation to a ranked prefix but never hides ranked MISs. |
| `seed` | Seed for reproducible sequential null estimation and derived evaluation seeds. |
| `name` | Optional display label stored in the result. |

Manual `alpha` is intentionally unsupported by the static v2 entry point. The
threshold is derived from the observed positive correlations and a sequential
permutation null estimate.

## 3. What the dimensions mean

MISDA builds two graphs at the same data-driven threshold:

1. The **structural graph** contains statistically supported positive edges.
   Its connected-component count is `structural_dimension`; its maximal
   independent sets are representative subsets of original objectives.
2. The **dependence graph** contains supported positive and negative edges.
   Its connected-component count is `latent_dimension`.

These counts and the preferred MIS size answer different questions and may
differ legitimately:

```python
analysis = result.analysis
print(analysis.original_dimension)
print(analysis.latent_dimension)
print(analysis.structural_dimension)
print(result.best_mis.size)
```

The separation status is explicit. `NULL_SEPARATION` means the positive-signal
onset precedes the null-calibrated endpoint. `NO_NULL_SEPARATION` means no such
strict separation was established; it is not rewritten as a successful
separation.

## 4. Result tree

`MISDAResult` contains three top-level branches:

- `result.analysis`: global dimensions, graphs, thresholds, rank counts, and
  evaluation counts;
- `result.mis`: every ranked `MISCandidate`, in deterministic order;
- `result.execution`: effective configuration, seed, timings, convergence, and
  RNG diagnostics.

Each candidate exposes stable identity and stored evidence:

```python
candidate = result.mis[0]

candidate.id
candidate.objectives
candidate.indices
candidate.size
candidate.rank
candidate.rank_values
candidate.evaluation
```

`result.best_mis` is the first ranked candidate. `best_mis_indices` and
`best_mis_labels` are transitional convenience properties.

## 5. Light evaluation

Light evaluation is performed during `analyze()` for the requested ranked
prefix. It stores two blocks per evaluated candidate.

### External linear reconstruction

`linear_reconstruction` predicts only eliminated objectives from the retained
MIS. It records per-objective out-of-sample R², mean and worst R², delete-one
jackknife standard errors, and explicit reasons for undefined values. When no
objective is eliminated, the metric is `None` with reason
`NO_ELIMINATED_OBJECTIVES`; it is never replaced by an artificial perfect
score.

### Pareto preservation

`pareto_preservation` compares nondominated row masks under minimization and
stores:

- `pareto_retention`: fraction of the full front retained;
- `pareto_validity`: fraction of the reduced front that is valid in the full
  front;
- `pareto_jaccard`: intersection over union;
- front sizes, `exact_preservation`, and `reduced_front_indices`.

Duplicate rows are mapped back to their original observations. Mixed objective
directions are rejected explicitly rather than silently normalized.

## 6. On-demand heavy evaluation

Use `misda.heavy()` only for candidates that need nonlinear evidence:

```python
misda.heavy(
    result,
    selection=[0, 2],
    null_reference=True,
)

nonlinear = result.mis[0].evaluation["nonlinear_reconstruction"]
```

`selection` accepts one index, a range, or an explicit index sequence into
`result.mis`. Existing metrics are preserved and already-computed heavy
evaluations are not repeated.

The nonlinear protocol uses nested leave-one-out Random Forest evaluation,
internal discrete model selection, and a tree count determined by uncertainty
stopping. Setting `null_reference=True` adds a sequential permutation reference
and above-null metrics. Neither loop has a hidden fixed cap. Both record
convergence or non-convergence explicitly and accept an application-provided
`cancel_requested` callback.

## 7. Reports and graphs

```python
print(result.summary())
print(result.report())
figure = result.graph_plot(show=False)
```

`summary()` is compact. `report()` renders only metrics already stored in the
result; it never triggers hidden evaluation. `graph_plot()` visualizes the
positive structural graph. The former `plot()` spelling is deprecated.

## 8. Reproducible benchmark artifacts

An individual result can be evaluated against external ground truth without
contaminating `analyze()`:

```python
truth = {
    "name": "Synthetic case",
    "latent_expected": 2,
    "structural_expected": 2,
    "blocks_expected": [["f1", "f2"], ["f3", "f4"]],
    "pareto_expected": [0, 2, 5],
    "feature": "Two known structural families.",
    "intuition": "MISDA should retain one objective per family.",
    "graph_expected": "Two disjoint positive components.",
    "notes": "Optional free-form note.",
}

bench = misda.benchmark(result, truth)
print(bench.result.analysis.structural_dimension)
print(bench.structural_jaccard)
print(bench.pareto_recall)
print(bench.report())
```

`truth` is a plain mapping. Dimension fields, `blocks_expected`, and
`pareto_expected` are optional; an absent reference makes only its metric
family `None`. `blocks_expected` uses objective labels. `pareto_expected` uses
zero-based row indices from the same observations analyzed in `result`.
`feature`, `intuition`, `graph_expected`, and `notes` are preserved for the
report and are never interpreted by the method.

Dimensional errors compare the estimated and declared counts. Structural
reconstruction uses optimal one-to-one block matching by Jaccard plus pairwise
precision, recall, and F1. Pareto metrics compare the stored reduced-front row
indices with `pareto_expected`; `benchmark()` does not need the original
objective matrix.

The repository-level suites remain available for reproducible artifacts.

Run the canonical and comparative suites from the repository root:

```bash
python -m examples.benchmarks.run_benchmark --output results/benchmark.json
python -m examples.benchmarks.run_comparative --output results/comparative.json
```

Each JSON artifact includes its schema version, method, parameters, software
versions, input digest, declarations, estimates, graph summaries, evaluation
metrics, and assessment. `compare_results()` compares a current artifact with a
frozen external baseline without changing current estimates to fit legacy
expectations.

The benchmark and comparative notebooks call the same Python functions. The
PCA curve reports global standardized reconstruction R². MISDA reports
reconstruction of eliminated objectives from selected original objectives;
these estimands remain separate.

## 9. Migration from the legacy result API

| Legacy spelling | Static v2 spelling |
|---|---|
| `caution=x` | `aggressiveness=x` |
| `result.alpha_min` | `result.analysis.alpha_onset` |
| `result.alpha_max` | `result.analysis.alpha_null` |
| `result.mis_sets` | `result.mis` |
| `result.ranked_mis_sets` | group `result.mis` by `candidate.rank` |
| `result.plot()` | `result.graph_plot()` |
| post-hoc validation method | light metrics in `candidate.evaluation` or explicit `misda.heavy()` |

The unambiguous transitional aliases emit `DeprecationWarning`. Ambiguous legacy
validation fields are not synthesized on the new result tree.

## 10. Scope and limitations

- The static pipeline is the current supported and acceptance-tested method.
- The earlier adaptive implementation is suspended and excluded from the
  scientific baseline.
- Pairwise structure is correlation-based and does not establish causality.
- Structural dimension, latent dimension, and MIS size are not interchangeable.
- Heavy evaluation can be expensive because its stopping rules are
  data-driven; use candidate selection and cancellation deliberately.
