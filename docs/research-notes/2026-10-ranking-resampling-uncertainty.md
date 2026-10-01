# Ranking resampling uncertainty

- Period: 2026-10-01
- Status: **Rejected as a sufficient decision criterion; retained as diagnostic evidence**
- Tracking: issue #80, PR #81
- Parent investigation: `2026-10-ranking-policy-default-audit.md`
- Normative reference: ADR 0021

## Question

PR #79 showed that `dominance_preservation` can prefer a different representative MIS under observation noise while structural dimension and support remain unchanged. The aggressiveness follow-up showed that this behavior is not mainly an `aggressiveness=1.0` endpoint effect.

The next hypothesis was that a genuinely useful dominance advantage might be distinguishable from a noise-driven representative preference by its stability under row resampling of the observed matrix `Y`.

For the original `Y`, define

```text
S_dom    = candidate selected by dominance_preservation
S_struct = candidate selected by size_span
DeltaD   = D(S_dom) - D(S_struct)
```

where `D(S)` is `new_dominance_rate`. Negative `DeltaD` favors the dominance-selected candidate.

The working hypothesis was:

- a strong, meaningful dominance advantage should keep `DeltaD < 0` across nearly all resamples;
- a noise-driven representative preference should show weak support, ties, or frequent sign reversals around zero.

## Experimental design

The first experiment deliberately isolated **ranking-evidence uncertainty** from discovery uncertainty.

For each original `Y`:

1. run discovery once;
2. select `S_dom` and `S_struct` once;
3. keep both objective subsets fixed;
4. bootstrap rows of `Y` with replacement;
5. recompute `D(S_dom)`, `D(S_struct)`, and `DeltaD` on each resample;
6. attach benchmark truth only afterward for interpretation.

Thus the bootstrap does not rebuild thresholds, graphs, dimensions, or the MIS universe.

The run used:

```text
bootstrap replicates       128
bootstrap master seed      20261001
optimization sample seeds  101, 202, 303, 404
optimization screen size   256 rows
controlled sample size     300 rows
noisy controlled cases     3 policy divergences per problem
noise sigma                0.05, 0.10, 0.20, 0.40 scanned until 3 divergences found
```

The analytical optimization controls were DTLZ5 and DPF1. The noisy controlled families were `blocks_4x5`, `antagonistic_linear_groups`, `nonlinear_blocks_4x5`, and `antagonistic_nonlinear_groups`.

## Results

### Analytical controls

The strong-control side behaved exactly as hoped.

For DTLZ5, `dominance_preservation` selected the known safe pair `f9,f10` in all four independent samples, while `size_span` selected `f1,f10`. In all four runs:

```text
P_hat(DeltaD < 0) = 1.0
sign-reversal rate = 0
```

The original `DeltaD` values ranged approximately from `-0.060` to `-0.079`.

For DPF1, `dominance_preservation` selected the known safe pair `f1,f2` in all four independent samples, while `size_span` selected `f2,f8`. Again:

```text
P_hat(DeltaD < 0) = 1.0
sign-reversal rate = 0
```

The original `DeltaD` values ranged approximately from `-0.310` to `-0.344`. The dominance-selected pair had zero new-dominance rate in every original and bootstrap sample.

Across all eight optimization-control runs, the median `P_hat(DeltaD < 0)` was `1.0` and the median sign-reversal rate was `0`.

### Noisy controlled divergences

The noisy cases were more informative because they falsified the simple decision hypothesis.

Across the 12 selected divergences:

```text
median P_hat(DeltaD < 0)     0.875
minimum                        0.602
maximum                        0.992
median sign-reversal rate      0.121
```

Some families did show the expected instability. `antagonistic_nonlinear_groups`, for example, had only about `0.60--0.67` negative-bootstrap fractions and `0.33--0.40` sign-reversal rates.

However, other externally bad choices were highly stable under bootstrap:

- `blocks_4x5`, `sigma=0.05`: `P_hat(DeltaD < 0)=0.984`, sign-reversal rate `0.016`, yet clean-truth Pareto Jaccard regressed by about `0.061`;
- `nonlinear_blocks_4x5`, `sigma=0.20`: `P_hat(DeltaD < 0)=0.969`, sign-reversal rate `0.031`, yet clean-truth Pareto Jaccard regressed by about `0.074`;
- all three tested `nonlinear_blocks_4x5` divergences regressed in external Pareto truth, despite negative-bootstrap fractions between about `0.867` and `0.969`.

Conversely, `blocks_4x5` also contained two externally improved choices with similarly high bootstrap support. Thus high resampling stability did not separate truth-improving and truth-regressing representative choices within that family.

## Interpretation

The key limitation is conceptual rather than numerical.

A row bootstrap conditions on the empirical distribution represented by the observed `Y`. If observation noise has shifted that empirical distribution so that one MIS genuinely preserves the **observed noisy order** better, bootstrap resamples reproduce that same contaminated distribution. The preference can therefore be extremely stable even when it is worse relative to the unobserved clean objectives.

In other words:

```text
bootstrap stability of observed-Y dominance advantage
    does not imply
latent clean-order or Pareto-truth superiority
```

The experiment therefore distinguishes some fragile sample-specific preferences, but it cannot generally diagnose observation-noise bias from a single contaminated `Y`.

The magnitude of `DeltaD` was much larger for the tested DTLZ5/DPF1 controls than for the noisy controlled divergences. That empirical separation is interesting but does not justify a new cutoff: the scale of `new_dominance_rate` depends on the problem, dimensional structure, candidate geometry, and observed distribution. Introducing a threshold chosen from these benchmarks would violate the intended data-driven/no-tuned-threshold philosophy and is not supported by this experiment.

## Broader implication: inference from noisy `Y`

This limitation is not specific to MISDA. Any method that infers latent structure only from the observed matrix `Y` is, in general, exposed to the distinction between sampling variability and measurement/observation noise.

Conceptually, if

```text
Y = Z + epsilon
```

with unobserved clean objectives `Z` and unknown observation error `epsilon`, MISDA can only operate on the dependence and order relations present in `Y`. Noise can therefore alter correlations, graph edges, MIS structure, and observed dominance relations.

PCA has an analogous limitation. It decomposes the variance/covariance structure of the observed data; in broad terms, the covariance of `Y` contains both signal and noise contributions. Without additional assumptions about the noise process, repeated measurements, or other external information, PCA also cannot in general identify which part of the observed covariance is latent structure and which part is measurement noise.

The mechanisms are different:

- MISDA is affected through correlations, graph structure, and observed order/dominance relations;
- PCA is affected through observed covariance, eigenvalues, and principal directions.

This investigation did **not** empirically compare PCA and MISDA noise robustness, so no claim follows here that one method is generically more robust than the other. The relevant conclusion is only the shared identifiability boundary: a single noisy `Y` does not automatically reveal the corresponding clean `Z`.

## Policy trade-off: structural conservatism versus optimization-oriented evidence

The accumulated evidence also suggests a useful but carefully limited interpretation of `size_span` versus `dominance_preservation`.

It would be too strong to say:

```text
size_span suffers less from noise but optimizes worse;
dominance_preservation suffers more from noise but optimizes better.
```

What the current evidence supports is narrower:

- `size_span` is a structural policy. It depends on the discovered graph and structural candidate descriptors and does not use the finer observed dominance geometry to distinguish representatives. In the current noisy audits it behaved more conservatively in the sense that it did not chase every observed-Y dominance advantage.
- `dominance_preservation` is explicitly optimization-oriented **with respect to observed order in `Y`**. It uses more of the empirical geometry that is relevant to Pareto dominance and, in DTLZ5 and DPF1, selected the analytically known safe candidates that `size_span` did not select.
- the same additional sensitivity that gives `dominance_preservation` useful discrimination also exposes it to observation-noise distortions in `Y`; the noisy audits contain both improvements and regressions relative to clean Pareto truth.

Thus the present working interpretation is:

```text
size_span
    = structurally conservative default

dominance_preservation
    = richer optimization-oriented evidence,
      but more exposed to the detailed observed geometry of noisy Y
```

This does not yet establish that `size_span` produces worse optimizer outcomes or that `dominance_preservation` produces better ones in general. That stronger claim requires the Full-versus-Reduced MOEA validation in the original objective space. The distinction here concerns what information each ranking policy uses and what failure mode the current experiments expose.

This interpretation supports the current software boundary: keep `size_span` as the public default and retain `dominance_preservation` as an explicit experimental ranking rather than promoting it to a universal replacement.

## Conclusion

The simple uncertainty-aware rule

```text
use dominance_preservation when DeltaD is stably negative under row bootstrap;
otherwise fall back to size_span
```

is **rejected as a sufficient general decision rule**.

The experiment does support weaker conclusions:

1. DTLZ5 and DPF1 exhibit a very strong and reproducible observed-order advantage for the analytically known safe candidates;
2. some noisy representative choices are visibly fragile under resampling;
3. other noisy representative choices are stably favored in observed `Y` even when external clean-truth Pareto agreement gets worse;
4. ordinary row resampling cannot, by itself, separate sampling uncertainty from systematic observation-noise distortion when only one observed `Y` is available.

No public API, ranking default, trust rule, or threshold should change on the basis of this probe. `size_span` remains the public default and `dominance_preservation` remains experimental.

## Methodological consequence

The unresolved problem is now sharper: if only a single observed `Y` is available and no measurement-noise model, replicate measurements, generator, or external truth is known, the latent clean ordering may not be identifiable from resampling alone.

Future work should therefore distinguish between:

- **sampling uncertainty conditional on observed `Y`**, which bootstrap/subsampling can probe;
- **observation-noise uncertainty**, which requires additional assumptions or information, such as repeated observations, an explicit noise model, or another independently justified diagnostic.

Before adding further machinery, the project should decide whether MISDA's runtime contract is expected to solve the latter problem from a single `Y` at all, or whether conservative structural ranking plus explicit experimental optimization-oriented rankings is the scientifically appropriate boundary.

## Reproducibility

The probe remains available as `benchmarks/run_ranking_resampling_probe.py` with lightweight contract tests in `tests/test_ranking_resampling.py`.

It is intentionally **not** part of the GitHub Actions scientific workflow. This investigation is an ad hoc reproducible experiment; local execution plus this research record is sufficient. If a future result turns it into a canonical validation battery, that infrastructure decision can be made separately.
