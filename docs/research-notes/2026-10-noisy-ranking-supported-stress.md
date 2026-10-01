# Noisy ranking safety — supported stress follow-up

- Period: 2026-10-01
- Tracking: issue #84
- Branch: `issue-84-noisy-ranking-safety`
- Parent note: `2026-10-noisy-ranking-safety.md`
- Status: **stress evidence recorded; no default-policy change**

## Why this follow-up exists

The first end-to-end noisy-discovery experiment showed that a static Pareto-truth regression in a finite noisy sample does not necessarily imply an operationally harmful reduction. In the antagonistic nonlinear control, both `size_span` and `dominance_preservation` remained better than Full even when the static audit preferred one representative over the other.

That experiment still did not answer the sharper ranking-safety question:

> If observation noise changes a globally consequential MIS choice while the current support machinery still accepts the reduction, which ranking policy is safer relative to Full?

DTLZ5 is particularly useful for this stress question because issue #83 already established a large clean end-to-end difference between its alternative representatives.

## High-noise screening

The ordinary DTLZ5/DPF1 noise grid (`sigma <= 0.40`) never changed either clean ranking selection. We therefore performed a deliberately extreme **stress screening** at

```text
sigma = 0.60, 0.80, 1.00, 1.50, 2.00
observation seeds = 101, 202, 303, 404, 505
```

These conditions are boundary probes, not proposed realistic observation-noise levels.

The screening showed that the clean representatives eventually change. It also exposed two distinct questions:

1. **support boundary:** can observed-Y support remain positive for a reduction that is operationally unsafe?
2. **ranking conditional on support:** when both policies remain `SUPPORTED_REDUCTION`, can one ranking be materially safer than the other?

## Stress A — support is not a safety guarantee

Condition:

```text
DTLZ5
sigma = 2.0
observation_seed = 101
```

Discovery returned structural dimension 7.

- `size_span`: `f1,f2,f4,f5,f6,f8,f10` — **SUPPORTED_REDUCTION**
- `dominance_preservation`: `f1,f2,f5,f7,f8,f9,f10` — **UNSUPPORTED_REDUCTION** (`TRANSITIVE_CHAINING`)

Both forced Reduced treatments were then evaluated on the clean DTLZ5 MOP with population 240, 800 generations and five MOEA seeds.

Mean final metrics:

| treatment | GD+ | IGD+ | relative HV |
|---|---:|---:|---:|
| Full | 5.3747 | 0.3340 | 0.1146 |
| Reduced / `size_span` | 3.7702 | 2.5297 | 0.0000 |
| Reduced / `dominance_preservation` | 4.6903 | 0.4340 | 0.0058 |

Using oriented loss where positive means Reduced is worse than Full:

- `size_span`: IGD+ loss **+2.1958**, HV loss **+0.1146**;
- `dominance_preservation`: IGD+ loss **+0.1000**, HV loss **+0.1088**.

### Interpretation

This is a stress-limit counterexample to treating current support as a safety certificate. The policy selected by `size_span` was explicitly marked supported, yet its Reduced optimization was drastically worse than Full by IGD+ and lost essentially all relative HV.

The unsupported dominance-selected candidate happened to degrade much less, but that does **not** mean the support warning should be ignored. The important conclusion is narrower:

> Under severe distortion of `Y`, support inferred from that same `Y` can itself be wrong about downstream reduction safety.

Therefore `SUPPORTED_REDUCTION` is evidence conditional on the observed data, not a guarantee against arbitrary observation noise.

## Stress B — both rankings supported, DTLZ5

Condition:

```text
DTLZ5
sigma = 1.0
observation_seed = 404
```

Discovery returned structural dimension 3 and both policies were `SUPPORTED_REDUCTION`:

- `size_span`: `f1,f9,f10`
- `dominance_preservation`: `f2,f8,f10`

Protocol: population 240, 800 generations, five matched MOEA seeds; all outputs evaluated in the original ten-objective clean DTLZ5 space against the same calibration.

Mean final metrics:

| treatment | GD+ | IGD+ | relative HV |
|---|---:|---:|---:|
| Full | 5.3747 | 0.3340 | 0.1146 |
| Reduced / `size_span` | **1.4551** | **0.03755** | **0.8386** |
| Reduced / `dominance_preservation` | 3.2290 | 0.85475 | 0.5537 |

Oriented mean loss relative to Full:

- `size_span`: IGD+ loss **-0.2964**, HV loss **-0.7240** — clearly better than Full;
- `dominance_preservation`: IGD+ loss **+0.5208**, HV loss **-0.4391** — mixed aggregate behavior.

The mean alone understates the instability of the dominance-selected reduction. Its final IGD+ across the five MOEA seeds was approximately

```text
0.304, 2.431, 0.353, 0.352, 0.834
```

whereas `size_span` remained tightly around

```text
0.032, 0.040, 0.041, 0.037, 0.039
```

Thus this stress case provides direct end-to-end evidence for the intuition that a structural ranking can be safer when observed dominance geometry is strongly corrupted, even though both candidates pass the current support assessment.

## Stress C — both rankings supported, DPF1

Condition:

```text
DPF1
sigma = 2.0
observation_seed = 505
```

Discovery returned structural dimension 3 and both policies were supported:

- `size_span`: `f4,f6,f10`
- `dominance_preservation`: `f1,f2,f7`

Mean final metrics:

| treatment | GD+ | IGD+ | relative HV |
|---|---:|---:|---:|
| Full | 0.003519 | 0.032929 | 0.90075 |
| Reduced / `size_span` | **0.000473** | 0.008508 | 0.93815 |
| Reduced / `dominance_preservation` | 0.000589 | **0.008356** | **0.94215** |

Both reductions clearly outperform Full and are very close to each other. `size_span` is slightly better on GD+; `dominance_preservation` is slightly better on IGD+ and HV. This case does not support a universal `noisy -> size_span` rule.

## What the stress tests establish

The combined evidence now supports a more careful picture.

### 1. Clean / representative `Y`

Issue #83 showed that `dominance_preservation` can choose a much better optimization-relevant MIS than `size_span` when the observed geometry is trustworthy. The clean DTLZ5 result is the strongest example.

### 2. Noisy `Y`

`dominance_preservation` is more exposed to distortions in observed dominance geometry. The DTLZ5 supported-stress condition shows that this can produce a substantially worse and less stable downstream reduction than `size_span` even while both reductions are marked supported.

However, DPF1 shows that high noise does not automatically make `size_span` superior. Both policies can remain operationally safe and nearly equivalent.

### 3. Support / abstention

The DTLZ5 `sigma=2.0`, seed 101 condition shows that the present support assessment cannot serve as an unconditional safety gate when the input matrix itself is severely corrupted. A supported candidate can still be operationally unsafe.

This does not invalidate the support mechanism in its intended data regime. It identifies its epistemic boundary: support is inferred from `Y`, so sufficiently bad `Y` can mislead both structural inference and ranking.

## Current working hypothesis

The evidence is consistent with, but does not yet prove, a noise-aware decision rule of the form

```text
poor structural/support evidence -> abstain
credible low-noise Y             -> dominance_preservation is attractive
known noisy / uncertain Y        -> prefer a conservative structural policy or abstain
```

Two qualifications are now essential:

- `size_span` is **not** proven universally safer under noise;
- `SUPPORTED_REDUCTION` is **not** sufficient by itself when data quality is poor.

Therefore the unresolved variable is increasingly not another ranking score derived from the same `Y`, but **how much confidence should be placed in `Y` itself**. External information about observation noise, replicates, or eventually `(X,Y)` may be needed to move beyond this limit.

## Reproducibility

The branch now contains:

- `benchmarks/run_noisy_ranking_safety.py` — noisy screening map;
- `benchmarks/run_noisy_ranking_optimization.py` — established controlled noisy end-to-end audit;
- `benchmarks/run_noisy_dtlz5_stress_optimization.py` — DTLZ5 support-boundary probe;
- `benchmarks/run_noisy_moeabench_stress_optimization.py` — supported DTLZ5/DPF1 ranking stress audit;
- `tests/test_noisy_ranking_safety.py` — deterministic reproduction tests for the selected noisy conditions;
- `.github/workflows/noisy-ranking-safety.yml` — reproducible screening and opt-in scientific jobs.

No public ranking default is changed by this work.

## Bottom line

The simple narrative

```text
clean -> dominance
noisy -> size_span
```

is directionally useful but too strong as a scientific conclusion.

A better statement from the current evidence is:

> `dominance_preservation` appears more optimization-aware when `Y` is representative, but it is more vulnerable to corrupted dominance geometry. `size_span` can be substantially safer in a noisy ranking failure, yet neither structural support nor structural ranking can guarantee safety when the observed data are severely distorted.

This places the next methodological question at the **data-confidence layer**, not merely at the ranking layer.
