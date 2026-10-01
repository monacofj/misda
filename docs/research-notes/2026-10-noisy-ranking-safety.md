# Noisy ranking-policy safety audit

## Question

Issue #83 established that, under clean discovery, `dominance_preservation` can produce a better end-to-end reduction than `size_span` when the two policies select different MISs. In DTLZ5 the difference was very large; in DPF1 it was smaller but still favorable on average.

The unresolved question is reliability when the discovery observations are noisy:

> If ranking is performed on a distorted `Y`, which policy is less likely to produce a harmful reduction, and when should MISDA abstain from reducing at all?

This is issue #84 on branch `issue-84-noisy-ranking-safety`.

## Operational distinction

The audit separates two decisions that should not be conflated:

1. **Gate** — is there enough support to reduce at all?
2. **Ranking** — conditional on reducing, which MIS should be selected?

The primary operational safety criterion is Reduced-versus-Full degradation in the original objective space. A common GT remains useful as an absolute diagnostic reference, but a reduction is unsafe if it is materially worse than the corresponding Full treatment even when Full itself is imperfect.

## Noise model

The experiments reuse the repository's established scale-relative additive Gaussian observation model:

\[
Y_j = Z_j + \sigma\,\operatorname{sd}(Z_j)\,\varepsilon_j,
\qquad \varepsilon_j \sim N(0,1).
\]

Only the discovery matrix is perturbed. The underlying MOP remains clean, and downstream optimizer outputs are re-evaluated on the clean original objectives.

## Stage 1a — DTLZ5 and DPF1 screening

The first screening reused the #83 clean optimization controls with

- `sigma = 0.00, 0.05, 0.10, 0.20, 0.40`;
- five observation seeds (`101, 202, 303, 404, 505`);
- the same 512-point scrambled Sobol discovery sample;
- MISDA seed `123`.

The result was negative but useful: **neither ranking policy changed its selected MIS in any of the 50 noisy conditions** (two problems × five sigmas × five noise realizations). DTLZ5 always retained the clean `size_span = {f1,f10}` and `dominance_preservation = {f9,f10}` selections; DPF1 likewise retained its clean selections.

Therefore an end-to-end MOEA rerun on this grid would not test noisy ranking safety: once the same MIS is selected, the clean downstream optimization problem is exactly the #83 treatment again. DTLZ5 and DPF1 remain useful clean controls, but this noise range does not perturb their ranking decision.

A second limitation is that both DTLZ5 and DPF1 reductions are already annotated `UNSUPPORTED_REDUCTION` because of `TRANSITIVE_CHAINING`. They cannot validate a gate rule such as `low support -> abstain`, because that rule stops before ranking even in the clean condition.

## Stage 1b — established noisy controlled battery

The audit therefore returned to the exact `noisy_robustness` battery used in Revisão 09 instead of inventing a new noisy generator. Across the established 200 runs:

- policy divergences: `77`;
- same selected dimension: `200/200`;
- trust-status changes: `0/200`;
- clean-sample Pareto-truth improvements for `dominance_preservation`: `32`;
- regressions: `40`;
- neutral comparisons: `128`.

The important point is unchanged from Revisão 09: every divergent run improved the observed dominance score by construction, but this did not reliably improve agreement with the clean-sample Pareto declaration.

The divergence counts by controlled problem were:

| problem | divergences | truth improved | truth regressed | neutral |
|---|---:|---:|---:|---:|
| `blocks_4x5` | 20 | 12 | 7 | 1 |
| `antagonistic_linear_groups` | 20 | 8 | 9 | 3 |
| `nonlinear_blocks_4x5` | 20 | 6 | 14 | 0 |
| `antagonistic_nonlinear_groups` | 17 | 6 | 10 | 1 |

The other noisy-robustness controls did not produce ranking divergences on this grid.

`nonlinear_blocks_4x5` is the strongest static negative case, but it has a nearly trivial global optimization structure: all four latent coordinates can be minimized simultaneously. For the first end-to-end noisy test we therefore preferred `antagonistic_nonlinear_groups`, a bounded one-dimensional problem with a genuine trade-off between monotone transforms of `x` and `1-x`.

## Stage 2 — end-to-end noisy discovery on an antagonistic nonlinear trade-off

`benchmarks/run_noisy_ranking_optimization.py` adapts the executable controlled problem to MoeaBench without changing its formulas. It reproduces the exact `noisy_robustness` sample/noise seed construction, performs MISDA discovery on noisy `Y`, and then runs Full and both Reduced formulations on the **clean** underlying MOP.

Protocol:

- problem: `antagonistic_nonlinear_groups` (`M=20`, one bounded decision variable `x in [0,1]`);
- discovery: `N=300`, MISDA seed `123`;
- observation noise: `sigma=0.10`;
- two fixed noise replicates selected before the MOEA comparison;
- population/reference directions: `240`;
- budget: `400` generations;
- five MOEA seeds: `321, 654, 987, 135, 246`;
- dense clean 2000-point Pareto reference;
- all optimizer populations re-evaluated in the original 20-objective clean space.

The two noisy discovery conditions deliberately point in opposite directions according to the Revisão-09 static clean-sample Pareto diagnostic:

| replicate | static outcome for dominance | `size_span` | `dominance_preservation` | clean-sample Pareto Jaccard size -> dominance |
|---:|---|---|---|---:|
| 202 | **regressed** | `f1,f11` | `f10,f18` | `0.1733 -> 0.1400` |
| 101 | **improved** | `f1,f11` | `f1,f12` | `0.1600 -> 0.2000` |

Both policy selections are `SUPPORTED_REDUCTION` in both conditions.

### Replicate 202 — static diagnostic says dominance regressed

Mean final metrics across five MOEA seeds:

| treatment | GD+ | IGD+ | relative HV |
|---|---:|---:|---:|
| Full | 0.000696 | 0.015474 | 0.881405 |
| Reduced / `size_span` | 0.000456 | 0.005385 | 0.953592 |
| Reduced / `dominance_preservation` | 0.000435 | 0.005575 | 0.945215 |

`size_span` is slightly better than `dominance_preservation` on IGD+ and relative HV, consistent in direction with the static regression label. However, the operational safety result is more important: **both reductions are substantially better than Full** at the tested budget.

Using the oriented loss convention where positive always means "Reduced is worse than Full":

- `size_span`: mean IGD+ loss `-0.010089`, HV loss `-0.072187`;
- `dominance_preservation`: mean IGD+ loss `-0.009899`, HV loss `-0.063809`.

Thus the noisy dominance choice is inferior to the structural choice in this condition, but it is **not a harmful false-positive reduction relative to Full**.

### Replicate 101 — static diagnostic says dominance improved

Mean final metrics:

| treatment | GD+ | IGD+ | relative HV |
|---|---:|---:|---:|
| Full | 0.000696 | 0.015474 | 0.881405 |
| Reduced / `size_span` | 0.000456 | 0.005385 | 0.953592 |
| Reduced / `dominance_preservation` | 0.000456 | 0.005384 | 0.953616 |

The two Reduced formulations are practically indistinguishable. `dominance_preservation` has only a tiny mean advantage (`Delta IGD+ ≈ -1.1e-6`, `Delta HV ≈ +2.4e-5`). Again, both are clearly better than Full.

## Interpretation after the first end-to-end noisy test

This experiment changes how the Revisão-09 noisy regressions should be interpreted.

The clean-sample Pareto-Jaccard comparison on a finite noisy discovery sample can distinguish representatives and its **direction** happened to agree with the small end-to-end ordering in these two selected replicates. But a static label such as `regressed` does **not** imply that the resulting reduction is operationally unsafe. In the explicit regression condition above, `dominance_preservation` was slightly worse than `size_span` yet still substantially better than Full.

That matters because the project goal is not to select the mathematically best representative MIS at any cost. The primary goal is to avoid reductions that materially degrade the optimization result relative to Full.

There is also an important limitation: in `antagonistic_nonlinear_groups`, every selected pair above contains one monotone transform of `x` and one monotone transform of `1-x`. These representatives encode the same underlying true Pareto trade-off. Noise can make their finite-sample dominance/Pareto diagnostics look different, but the global clean reduction remains structurally safe. This control therefore demonstrates that **a noisy static ranking regression can overstate operational harm**, but it does not yet show what happens when noise makes MISDA choose a globally unsafe representative.

Consequently, the current evidence does **not** establish the simple rule `noisy Y -> size_span`. It supports a weaker statement:

- `dominance_preservation` is more sensitive to observed noisy geometry and can choose a slightly worse representative;
- `size_span` can be more conservative in such a condition;
- but the observed representative error need not create a harmful reduction relative to Full;
- the decisive missing case is one where noisy discovery changes the selected MIS in a way that changes the **global clean optimization problem**, not merely the finite-sample ordering.

## Next experimental target

The next useful control should satisfy all of the following:

1. the clean problem has at least two admissible MIS representatives with genuinely different global optimization consequences;
2. noisy discovery can switch the selected representative;
3. the gate still reports sufficient support, so ranking rather than abstention is actually being tested;
4. Full and both Reduced formulations can be evaluated in a common clean objective space.

DTLZ5 already demonstrates condition (1) strongly under clean discovery, but the tested `sigma <= 0.40` grid did not change either ranking selection. A natural next probe is therefore to search for a noisy/sampling regime that changes the DTLZ5 representative **without first invalidating support**, or to construct/use another controlled MOP with the same property. Only such a case can establish whether `size_span` really prevents more operational false positives than `dominance_preservation` under noisy discovery.

## Status

Issue #84 and branch `issue-84-noisy-ranking-safety` remain active. Screening instrumentation, reproducibility tests, the full established noisy-policy map, and the first five-seed end-to-end noisy-discovery comparison all pass. The first end-to-end result rejects an overly strong reading of Revisão-09 static regressions, but the central noisy-policy safety question remains open pending a control where noise can alter the globally relevant representative choice.
